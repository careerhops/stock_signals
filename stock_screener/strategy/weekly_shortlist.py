from __future__ import annotations

from datetime import timedelta
from typing import Any

import pandas as pd

from stock_screener.data.storage import Storage
from stock_screener.resample import resample_daily_to_weekly
from stock_screener.strategy.weekly_buy_sell import run_weekly_buy_sell


DEFAULT_BENCHMARK_SYMBOL = "NIFTY 50"
SECTOR_BENCHMARK_SYMBOLS = {
    "NIFTY 50",
    "NIFTY AUTO",
    "NIFTY BANK",
    "NIFTY CONSR DURBL",
    "NIFTY ENERGY",
    "NIFTY FIN SERVICE",
    "NIFTY FMCG",
    "NIFTY HEALTHCARE",
    "NIFTY INFRA",
    "NIFTY IT",
    "NIFTY METAL",
    "NIFTY OIL AND GAS",
    "NIFTY PHARMA",
    "NIFTY PSE",
    "NIFTY PSU BANK",
    "NIFTY REALTY",
}
SHORTLIST_BENCHMARK_SYMBOLS = tuple(sorted(SECTOR_BENCHMARK_SYMBOLS))

SHORTLIST_COLUMNS = (
    "shortlist_score",
    "htf_alignment_confirmation",
    "htf_open_buy_regime",
    "htf_latest_signal",
    "htf_latest_signal_date",
    "volume_breakout_confirmation",
    "relative_strength_confirmation",
    "relative_strength_benchmark",
    "stock_return_12w_pct",
    "benchmark_return_12w_pct",
    "relative_strength_12w_pct",
    "location_confirmation",
    "distance_from_demand_pct",
    "risk_pct_to_demand_zone",
    "expected_target_return_pct",
    "risk_reward_ratio",
    "risk_reward_confirmation",
)


def shortlist_benchmark_symbols() -> tuple[str, ...]:
    return SHORTLIST_BENCHMARK_SYMBOLS


def benchmark_symbol_for_industry(industry: str | None) -> str:
    industry_text = str(industry or "").upper()
    keyword_map = (
        (("BANK", "PRIVATE BANK", "PSU BANK"), "NIFTY BANK"),
        (("FINANCIAL", "NBFC", "FIN SERVICE", "INSURANCE", "MUTUAL"), "NIFTY FIN SERVICE"),
        (("INFORMATION TECHNOLOGY", "SOFTWARE", "IT"), "NIFTY IT"),
        (("PHARMA", "HEALTHCARE", "HOSPITAL", "LIFE SCIENCE"), "NIFTY PHARMA"),
        (("AUTO", "AUTOMOBILE", "TYRE"), "NIFTY AUTO"),
        (("METAL", "STEEL", "MINING"), "NIFTY METAL"),
        (("REALTY", "REAL ESTATE"), "NIFTY REALTY"),
        (("FMCG", "FOOD", "BEVERAGE", "PERSONAL CARE"), "NIFTY FMCG"),
        (("OIL", "GAS", "PETRO", "REFIN"), "NIFTY OIL AND GAS"),
        (("POWER", "ENERGY", "UTILITY"), "NIFTY ENERGY"),
        (("CONSUMER DURABLE", "APPLIANCE", "JEWELL", "RETAIL"), "NIFTY CONSR DURBL"),
        (("INFRA", "CONSTRUCTION", "CAPITAL GOODS", "ENGINEERING", "CEMENT"), "NIFTY INFRA"),
        (("HEALTHCARE", "DIAGNOSTIC"), "NIFTY HEALTHCARE"),
        (("PSU", "PUBLIC SECTOR"), "NIFTY PSE"),
    )
    for keywords, benchmark in keyword_map:
        if any(keyword in industry_text for keyword in keywords):
            return benchmark
    return DEFAULT_BENCHMARK_SYMBOL


def enrich_weekly_signal_shortlist_frame(
    frame: pd.DataFrame,
    storage: Storage,
    config: dict[str, Any],
) -> pd.DataFrame:
    if frame.empty:
        return frame

    enriched = frame.copy()
    for column in SHORTLIST_COLUMNS:
        if column not in enriched.columns:
            enriched[column] = pd.NA

    strategy_cfg = config.get("strategy", {})
    weekly_anchor = strategy_cfg.get("weekly_anchor", "W-FRI")
    use_completed_weeks_only = bool(strategy_cfg.get("use_completed_weeks_only", True))
    shortlist_cfg = config.get("shortlist", {}) or {}
    lookback_weeks = int(shortlist_cfg.get("relative_strength_lookback_weeks", 12))
    min_volume_ratio = float(shortlist_cfg.get("default_volume_ratio", 1.5))
    max_distance_from_demand = float(shortlist_cfg.get("default_max_distance_from_demand_pct", 8.0))
    min_risk_reward = float(shortlist_cfg.get("default_min_risk_reward_ratio", 2.0))

    benchmark_cache: dict[str, pd.DataFrame] = {}

    for index, row in enriched.iterrows():
        exchange = str(row.get("exchange", "")).strip().upper()
        symbol = str(row.get("symbol") or row.get("tradingsymbol") or "").strip().upper()
        if not exchange or not symbol:
            continue

        daily = storage.load_candles(exchange, symbol, "1D")
        if daily.empty:
            continue

        as_of_date = pd.to_datetime(row.get("date"), errors="coerce")
        if pd.isna(as_of_date):
            as_of_date = pd.to_datetime(daily["date"], errors="coerce").max()
        daily_as_of = daily[pd.to_datetime(daily["date"], errors="coerce") <= as_of_date].copy()
        if daily_as_of.empty:
            continue

        weekly = resample_daily_to_weekly(daily_as_of, weekly_anchor, use_completed_weeks_only)
        if weekly.empty:
            continue

        industry = row.get("industry", "")
        benchmark_symbol = benchmark_symbol_for_industry(industry)
        if benchmark_symbol not in benchmark_cache:
            benchmark_cache[benchmark_symbol] = storage.load_candles("NSE_INDEX", benchmark_symbol, "1D")
        benchmark_daily = benchmark_cache.get(benchmark_symbol, pd.DataFrame())
        benchmark_as_of = benchmark_daily[pd.to_datetime(benchmark_daily.get("date"), errors="coerce") <= as_of_date].copy() if not benchmark_daily.empty else pd.DataFrame()
        if benchmark_as_of.empty and benchmark_symbol != DEFAULT_BENCHMARK_SYMBOL:
            if DEFAULT_BENCHMARK_SYMBOL not in benchmark_cache:
                benchmark_cache[DEFAULT_BENCHMARK_SYMBOL] = storage.load_candles("NSE_INDEX", DEFAULT_BENCHMARK_SYMBOL, "1D")
            benchmark_symbol = DEFAULT_BENCHMARK_SYMBOL
            benchmark_daily = benchmark_cache.get(benchmark_symbol, pd.DataFrame())
            benchmark_as_of = benchmark_daily[pd.to_datetime(benchmark_daily.get("date"), errors="coerce") <= as_of_date].copy() if not benchmark_daily.empty else pd.DataFrame()

        monthly = _resample_daily_to_monthly(daily_as_of)
        monthly_strategy = run_weekly_buy_sell(monthly, config) if not monthly.empty else pd.DataFrame()
        monthly_signals = monthly_strategy[monthly_strategy.get("signal", pd.Series(dtype="object")).isin(["BUY", "SELL"])].copy() if not monthly_strategy.empty else pd.DataFrame()
        htf_latest_signal = "NONE"
        htf_latest_signal_date = pd.NA
        htf_open_buy_regime = False
        if not monthly_signals.empty:
            latest_monthly_signal = monthly_signals.sort_values("date").iloc[-1]
            htf_latest_signal = str(latest_monthly_signal.get("signal", "NONE"))
            htf_latest_signal_date = latest_monthly_signal.get("date", pd.NA)
            htf_open_buy_regime = htf_latest_signal == "BUY"

        volume_ratio = _float_or_na(row.get("volume_confirmation_ratio"))
        volume_breakout_confirmation = pd.notna(volume_ratio) and float(volume_ratio) >= min_volume_ratio

        stock_return_pct, benchmark_return_pct, rs_spread_pct = _relative_strength_snapshot(
            daily_as_of,
            benchmark_as_of,
            lookback_weeks,
        )
        rs_confirmation = pd.notna(rs_spread_pct) and float(rs_spread_pct) > 0

        close = _float_or_na(row.get("close"))
        demand_zone = _float_or_na(row.get("demand_zone"))
        distance_from_demand = pd.NA
        location_confirmation = pd.NA
        risk_pct = pd.NA
        if pd.notna(close) and pd.notna(demand_zone) and float(demand_zone) > 0 and float(close) >= float(demand_zone):
            distance_from_demand = ((float(close) - float(demand_zone)) / float(demand_zone)) * 100.0
            location_confirmation = distance_from_demand <= max_distance_from_demand
            if float(close) > 0:
                risk_pct = ((float(close) - float(demand_zone)) / float(close)) * 100.0

        expected_target_return = _expected_target_return_pct(row)
        risk_reward_ratio = pd.NA
        risk_reward_confirmation = pd.NA
        if pd.notna(expected_target_return) and pd.notna(risk_pct) and float(risk_pct) > 0:
            risk_reward_ratio = float(expected_target_return) / float(risk_pct)
            risk_reward_confirmation = float(risk_reward_ratio) >= min_risk_reward

        confirmations = [
            bool(htf_open_buy_regime),
            bool(volume_breakout_confirmation),
            bool(rs_confirmation),
            bool(location_confirmation) if pd.notna(location_confirmation) else False,
            bool(risk_reward_confirmation) if pd.notna(risk_reward_confirmation) else False,
        ]

        enriched.at[index, "shortlist_score"] = int(sum(confirmations))
        enriched.at[index, "htf_alignment_confirmation"] = bool(htf_open_buy_regime)
        enriched.at[index, "htf_open_buy_regime"] = bool(htf_open_buy_regime)
        enriched.at[index, "htf_latest_signal"] = htf_latest_signal
        enriched.at[index, "htf_latest_signal_date"] = htf_latest_signal_date
        enriched.at[index, "volume_breakout_confirmation"] = bool(volume_breakout_confirmation)
        enriched.at[index, "relative_strength_confirmation"] = bool(rs_confirmation)
        enriched.at[index, "relative_strength_benchmark"] = benchmark_symbol
        enriched.at[index, "stock_return_12w_pct"] = stock_return_pct
        enriched.at[index, "benchmark_return_12w_pct"] = benchmark_return_pct
        enriched.at[index, "relative_strength_12w_pct"] = rs_spread_pct
        enriched.at[index, "location_confirmation"] = location_confirmation
        enriched.at[index, "distance_from_demand_pct"] = distance_from_demand
        enriched.at[index, "risk_pct_to_demand_zone"] = risk_pct
        enriched.at[index, "expected_target_return_pct"] = expected_target_return
        enriched.at[index, "risk_reward_ratio"] = risk_reward_ratio
        enriched.at[index, "risk_reward_confirmation"] = risk_reward_confirmation

    return enriched


def _resample_daily_to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date")
    monthly = (
        frame.set_index("date")
        .resample("ME")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )
    return monthly


def _relative_strength_snapshot(
    daily: pd.DataFrame,
    benchmark_daily: pd.DataFrame,
    lookback_weeks: int,
) -> tuple[float | pd.NA, float | pd.NA, float | pd.NA]:
    if daily.empty or benchmark_daily.empty:
        return pd.NA, pd.NA, pd.NA

    end_date = min(
        pd.to_datetime(daily["date"], errors="coerce").max(),
        pd.to_datetime(benchmark_daily["date"], errors="coerce").max(),
    )
    start_date = end_date - timedelta(weeks=max(lookback_weeks, 1))

    stock_start = _latest_close_on_or_before(daily, start_date)
    stock_end = _latest_close_on_or_before(daily, end_date)
    benchmark_start = _latest_close_on_or_before(benchmark_daily, start_date)
    benchmark_end = _latest_close_on_or_before(benchmark_daily, end_date)

    if any(pd.isna(value) or float(value) <= 0 for value in (stock_start, stock_end, benchmark_start, benchmark_end)):
        return pd.NA, pd.NA, pd.NA

    stock_return = ((float(stock_end) / float(stock_start)) - 1.0) * 100.0
    benchmark_return = ((float(benchmark_end) / float(benchmark_start)) - 1.0) * 100.0
    return stock_return, benchmark_return, stock_return - benchmark_return


def _latest_close_on_or_before(frame: pd.DataFrame, as_of_date: pd.Timestamp) -> float | pd.NA:
    dated = frame.copy()
    dated["date"] = pd.to_datetime(dated["date"], errors="coerce")
    dated = dated[dated["date"] <= as_of_date]
    if dated.empty:
        return pd.NA
    return _float_or_na(dated.sort_values("date").iloc[-1].get("close"))


def _expected_target_return_pct(row: pd.Series) -> float | pd.NA:
    candidates = (
        row.get("median_pair_return_last_3_pct"),
        row.get("prior_pair_return_last_1_pct"),
        row.get("sell_pair_return_pct"),
    )
    for candidate in candidates:
        numeric = _float_or_na(candidate)
        if pd.notna(numeric) and float(numeric) > 0:
            return float(numeric)
    return pd.NA


def _float_or_na(value: Any) -> float | pd.NA:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return pd.NA
    if pd.isna(numeric):
        return pd.NA
    return numeric
