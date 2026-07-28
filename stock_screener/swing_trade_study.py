from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from stock_screener.data.storage import Storage
from stock_screener.resample import resample_daily_to_weekly
from stock_screener.strategy.daily_confirmation import compute_daily_confirmations
from stock_screener.strategy.technical_ratings import latest_technical_rating
from stock_screener.strategy.weekly_shortlist import benchmark_symbol_for_industry


@dataclass(frozen=True)
class SwingTradeStudyResult:
    summary: dict[str, Any]
    candidates: pd.DataFrame
    all_setups: pd.DataFrame


DEFAULT_CONFIG = {
    "min_history_days": 220,
    "rs_lookback_days": 60,
    "pivot_lookback_days": 55,
    "base_lookback_days": 20,
    "max_distance_to_pivot_pct": 3.0,
    "max_breakout_extension_pct": 1.5,
    "max_base_contraction_pct": 18.0,
    "min_volume_ratio": 1.2,
    "min_breakout_volume_ratio": 1.5,
    "min_rs_spread_pct": 0.0,
    "min_risk_reward_ratio": 2.0,
    "candidate_min_score": 4,
}


def run_swing_trade_study(
    config: dict[str, Any],
    storage: Storage,
    exchange: str = "NSE",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> SwingTradeStudyResult:
    study_cfg = {**DEFAULT_CONFIG, **(config.get("swing_trade_study", {}) or {})}
    universe = _load_universe_from_latest_scan(storage, exchange)
    if universe.empty:
        raise RuntimeError("No scanned stocks are available. Run the Home screener first, then retry the Swing Trade Study.")

    benchmark_cache: dict[str, pd.DataFrame] = {}
    benchmark_cache["NIFTY 50"] = _prepare_daily(storage.load_candles("NSE_INDEX", "NIFTY 50", "1D"))

    rows: list[dict[str, Any]] = []
    strategy_cfg = config.get("strategy", {})
    weekly_anchor = strategy_cfg.get("weekly_anchor", "W-FRI")
    use_completed_weeks_only = bool(strategy_cfg.get("use_completed_weeks_only", True))

    _emit_progress(progress_callback, phase="Scoring swing trade setups", completed=0, total=len(universe), current_symbol="", current_exchange=exchange)
    for index, row in enumerate(universe.to_dict(orient="records"), start=1):
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        row_exchange = str(row.get("exchange") or exchange).strip().upper() or exchange
        daily = _prepare_daily(storage.load_candles(row_exchange, symbol, "1D"))
        if daily.empty:
            _emit_progress(progress_callback, phase="Scoring swing trade setups", completed=index, total=len(universe), current_symbol=symbol, current_exchange=row_exchange)
            continue

        setup = evaluate_swing_trade_setup(
            daily=daily,
            benchmark_cache=benchmark_cache,
            storage=storage,
            config=config,
            study_cfg=study_cfg,
            weekly_anchor=weekly_anchor,
            use_completed_weeks_only=use_completed_weeks_only,
            industry=row.get("industry"),
        )
        if not setup:
            _emit_progress(progress_callback, phase="Scoring swing trade setups", completed=index, total=len(universe), current_symbol=symbol, current_exchange=row_exchange)
            continue

        rows.append(
            {
                "exchange": row_exchange,
                "symbol": symbol,
                "name": row.get("name", symbol),
                "industry": row.get("industry", ""),
                "market_cap_cr": row.get("market_cap_cr", pd.NA),
                "market_cap_bucket": row.get("market_cap_bucket", ""),
                "latest_signal": row.get("latest_signal", ""),
                "latest_signal_date": row.get("latest_signal_date", ""),
                **setup,
            }
        )
        _emit_progress(progress_callback, phase="Scoring swing trade setups", completed=index, total=len(universe), current_symbol=symbol, current_exchange=row_exchange)

    all_setups = pd.DataFrame(rows, columns=_setup_columns())
    candidates = _build_candidates(all_setups, study_cfg)
    summary = _build_summary(exchange, universe, all_setups, candidates, study_cfg)
    return SwingTradeStudyResult(summary, _sort_candidates(candidates), _sort_all_setups(all_setups))


def save_swing_trade_outputs(result: SwingTradeStudyResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result.summary]).to_csv(output_dir / "latest_summary.csv", index=False)
    result.candidates.to_csv(output_dir / "latest_candidates.csv", index=False)
    result.all_setups.to_csv(output_dir / "latest_all_setups.csv", index=False)


def load_swing_trade_outputs(output_dir: Path) -> SwingTradeStudyResult:
    summary: dict[str, Any] = {}
    summary_path = output_dir / "latest_summary.csv"
    if summary_path.exists():
        summary_frame = _read_csv(summary_path)
        if not summary_frame.empty:
            summary = summary_frame.iloc[0].to_dict()
    return SwingTradeStudyResult(
        summary=summary,
        candidates=_read_csv(output_dir / "latest_candidates.csv"),
        all_setups=_read_csv(output_dir / "latest_all_setups.csv"),
    )


def evaluate_swing_trade_setup(
    daily: pd.DataFrame,
    benchmark_cache: dict[str, pd.DataFrame],
    storage: Storage,
    config: dict[str, Any],
    study_cfg: dict[str, Any],
    weekly_anchor: str,
    use_completed_weeks_only: bool,
    industry: str | None = None,
) -> dict[str, Any]:
    if daily.empty or len(daily) < int(study_cfg["min_history_days"]):
        return {}

    frame = daily.copy()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["high"] = pd.to_numeric(frame["high"], errors="coerce")
    frame["low"] = pd.to_numeric(frame["low"], errors="coerce")
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    frame = frame.dropna(subset=["date", "close", "high", "low", "volume"]).sort_values("date").reset_index(drop=True)
    if frame.empty or len(frame) < int(study_cfg["min_history_days"]):
        return {}

    daily_conf = compute_daily_confirmations(frame)
    latest_conf = daily_conf.iloc[-1]
    latest = frame.iloc[-1]
    close = float(latest["close"])
    volume = float(latest["volume"])

    close_series = frame["close"]
    sma_50 = close_series.rolling(50, min_periods=50).mean().iloc[-1]
    sma_150 = close_series.rolling(150, min_periods=150).mean().iloc[-1]
    sma_200 = close_series.rolling(200, min_periods=200).mean().iloc[-1]
    sma_200_20d_ago = close_series.rolling(200, min_periods=200).mean().shift(20).iloc[-1]
    high_52w = frame["high"].rolling(252, min_periods=180).max().iloc[-1]
    low_52w = frame["low"].rolling(252, min_periods=180).min().iloc[-1]

    trend_template_pass = bool(
        pd.notna(sma_50)
        and pd.notna(sma_150)
        and pd.notna(sma_200)
        and pd.notna(sma_200_20d_ago)
        and pd.notna(high_52w)
        and pd.notna(low_52w)
        and close > float(sma_50)
        and close > float(sma_150)
        and close > float(sma_200)
        and float(sma_50) > float(sma_150)
        and float(sma_150) > float(sma_200)
        and float(sma_200) > float(sma_200_20d_ago)
        and close >= float(low_52w) * 1.30
        and close >= float(high_52w) * 0.75
    )

    weekly = resample_daily_to_weekly(frame, weekly_anchor, use_completed_weeks_only)
    weekly_rating = latest_technical_rating(weekly) if not weekly.empty else {}
    weekly_rating_status = str(weekly_rating.get("rating_status") or "")
    weekly_alignment_pass = weekly_rating_status in {"Buy", "Strong Buy"}

    benchmark_symbol = benchmark_symbol_for_industry(industry)
    benchmark_daily = benchmark_cache.get(benchmark_symbol)
    if benchmark_daily is None or benchmark_daily.empty:
        fallback = benchmark_cache.get("NIFTY 50", pd.DataFrame())
        benchmark_symbol = "NIFTY 50"
        benchmark_daily = fallback
    rs_stock_return, rs_benchmark_return, rs_spread = _relative_strength_snapshot(
        frame,
        benchmark_daily,
        int(study_cfg["rs_lookback_days"]),
    )
    relative_strength_pass = pd.notna(rs_spread) and float(rs_spread) >= float(study_cfg["min_rs_spread_pct"])

    avg_volume_20 = frame["volume"].rolling(20, min_periods=20).mean().iloc[-1]
    volume_ratio = (volume / float(avg_volume_20)) if pd.notna(avg_volume_20) and float(avg_volume_20) > 0 else pd.NA
    obv_slope_20d = latest_conf.get("daily_obv_slope_20d", pd.NA)
    volume_setup_pass = bool(
        (pd.notna(volume_ratio) and float(volume_ratio) >= float(study_cfg["min_volume_ratio"]))
        or (pd.notna(obv_slope_20d) and float(obv_slope_20d) > 0)
    )

    pivot_lookback = int(study_cfg["pivot_lookback_days"])
    base_lookback = int(study_cfg["base_lookback_days"])
    pivot_high = frame["high"].rolling(pivot_lookback, min_periods=min(pivot_lookback, 20)).max().shift(1).iloc[-1]
    base_slice = frame.tail(base_lookback)
    base_high = pd.to_numeric(base_slice["high"], errors="coerce").max() if not base_slice.empty else pd.NA
    base_low = pd.to_numeric(base_slice["low"], errors="coerce").min() if not base_slice.empty else pd.NA
    base_contraction_pct = (
        ((float(base_high) - float(base_low)) / float(base_low)) * 100.0
        if pd.notna(base_high) and pd.notna(base_low) and float(base_low) > 0
        else pd.NA
    )
    distance_to_pivot_pct = (
        ((float(pivot_high) - close) / float(pivot_high)) * 100.0
        if pd.notna(pivot_high) and float(pivot_high) > 0
        else pd.NA
    )
    breakout_proximity_pass = bool(
        pd.notna(distance_to_pivot_pct)
        and float(distance_to_pivot_pct) <= float(study_cfg["max_distance_to_pivot_pct"])
        and float(distance_to_pivot_pct) >= -float(study_cfg["max_breakout_extension_pct"])
        and pd.notna(base_contraction_pct)
        and float(base_contraction_pct) <= float(study_cfg["max_base_contraction_pct"])
        and close >= float(latest_conf.get("daily_ema_20", pd.NA)) if pd.notna(latest_conf.get("daily_ema_20", pd.NA)) else True
    )

    stop_price = min(
        value
        for value in [
            _safe_float(base_low),
            _safe_float(latest_conf.get("daily_ema_20", pd.NA)),
            _safe_float(frame["low"].rolling(10, min_periods=5).min().iloc[-1]),
        ]
        if value is not None and value > 0
    ) if any(
        value is not None and value > 0
        for value in [
            _safe_float(base_low),
            _safe_float(latest_conf.get("daily_ema_20", pd.NA)),
            _safe_float(frame["low"].rolling(10, min_periods=5).min().iloc[-1]),
        ]
    ) else pd.NA

    target_price = (
        float(pivot_high) + max(float(pivot_high) - float(base_low), 0.0)
        if pd.notna(pivot_high) and pd.notna(base_low)
        else pd.NA
    )
    risk_pct = (
        ((close - float(stop_price)) / close) * 100.0
        if pd.notna(stop_price) and float(stop_price) < close
        else pd.NA
    )
    reward_pct = (
        ((float(target_price) - close) / close) * 100.0
        if pd.notna(target_price) and float(target_price) > close
        else pd.NA
    )
    risk_reward_ratio = (
        float(reward_pct) / float(risk_pct)
        if pd.notna(risk_pct) and pd.notna(reward_pct) and float(risk_pct) > 0
        else pd.NA
    )
    risk_reward_pass = pd.notna(risk_reward_ratio) and float(risk_reward_ratio) >= float(study_cfg["min_risk_reward_ratio"])

    score_flags = {
        "trend_template": trend_template_pass,
        "weekly_alignment": weekly_alignment_pass,
        "relative_strength": bool(relative_strength_pass),
        "volume_setup": volume_setup_pass,
        "breakout_proximity": breakout_proximity_pass,
        "risk_reward": bool(risk_reward_pass),
    }
    swing_score = int(sum(1 for flag in score_flags.values() if flag))

    breakout_now = bool(
        pd.notna(pivot_high)
        and close > float(pivot_high)
        and pd.notna(volume_ratio)
        and float(volume_ratio) >= float(study_cfg["min_breakout_volume_ratio"])
    )
    if swing_score >= 5 and trend_template_pass and relative_strength_pass and breakout_proximity_pass:
        setup_status = "Breakout Triggered" if breakout_now else "Ready Now"
    elif swing_score >= int(study_cfg["candidate_min_score"]) and trend_template_pass and relative_strength_pass and breakout_proximity_pass:
        setup_status = "Near Breakout"
    elif swing_score >= int(study_cfg["candidate_min_score"]) and trend_template_pass:
        setup_status = "Trend Watchlist"
    else:
        setup_status = "Not Ready"

    return {
        "latest_close": close,
        "latest_close_date": latest["date"],
        "swing_score": swing_score,
        "setup_status": setup_status,
        "trend_template_pass": trend_template_pass,
        "weekly_alignment_pass": weekly_alignment_pass,
        "weekly_technical_rating": weekly_rating_status,
        "relative_strength_pass": bool(relative_strength_pass),
        "relative_strength_benchmark": benchmark_symbol,
        "stock_return_12w_pct": rs_stock_return,
        "benchmark_return_12w_pct": rs_benchmark_return,
        "relative_strength_12w_pct": rs_spread,
        "volume_setup_pass": volume_setup_pass,
        "volume_ratio_20d": volume_ratio,
        "obv_slope_20d": obv_slope_20d,
        "breakout_proximity_pass": breakout_proximity_pass,
        "pivot_high": pivot_high,
        "distance_to_pivot_pct": distance_to_pivot_pct,
        "base_contraction_pct": base_contraction_pct,
        "risk_reward_pass": bool(risk_reward_pass) if pd.notna(risk_reward_pass) else False,
        "stop_price": stop_price,
        "target_price": target_price,
        "risk_pct": risk_pct,
        "reward_pct": reward_pct,
        "risk_reward_ratio": risk_reward_ratio,
        "breakout_now": breakout_now,
    }


def _load_universe_from_latest_scan(storage: Storage, exchange: str) -> pd.DataFrame:
    latest = storage.load_signals("latest_scan_details.csv")
    if latest.empty:
        return pd.DataFrame()
    latest = latest.copy()
    latest["exchange"] = latest.get("exchange", pd.Series(dtype="object")).astype(str).str.upper()
    latest["symbol"] = latest.get("symbol", pd.Series(dtype="object")).astype(str).str.upper()
    latest = latest[latest["exchange"] == str(exchange).upper()].copy()
    latest = latest.drop_duplicates(subset=["exchange", "symbol"], keep="last")
    return latest.reset_index(drop=True)


def _relative_strength_snapshot(
    daily: pd.DataFrame,
    benchmark_daily: pd.DataFrame,
    lookback_days: int,
) -> tuple[float | pd.NA, float | pd.NA, float | pd.NA]:
    if daily.empty or benchmark_daily.empty:
        return pd.NA, pd.NA, pd.NA
    stock_start = _latest_close_on_or_before(daily, daily["date"].max() - pd.Timedelta(days=lookback_days))
    stock_end = _latest_close_on_or_before(daily, daily["date"].max())
    benchmark_start = _latest_close_on_or_before(benchmark_daily, daily["date"].max() - pd.Timedelta(days=lookback_days))
    benchmark_end = _latest_close_on_or_before(benchmark_daily, daily["date"].max())
    if any(pd.isna(value) or float(value) <= 0 for value in (stock_start, stock_end, benchmark_start, benchmark_end)):
        return pd.NA, pd.NA, pd.NA
    stock_return = ((float(stock_end) / float(stock_start)) - 1.0) * 100.0
    benchmark_return = ((float(benchmark_end) / float(benchmark_start)) - 1.0) * 100.0
    return stock_return, benchmark_return, stock_return - benchmark_return


def _latest_close_on_or_before(frame: pd.DataFrame, as_of_date: pd.Timestamp) -> float | pd.NA:
    dated = frame.copy()
    dated["date"] = pd.to_datetime(dated["date"], errors="coerce")
    dated = dated.dropna(subset=["date"]).sort_values("date")
    dated = dated[dated["date"] <= as_of_date]
    if dated.empty:
        return pd.NA
    value = pd.to_numeric(pd.Series([dated.iloc[-1].get("close")]), errors="coerce").iloc[0]
    return value if pd.notna(value) else pd.NA


def _build_candidates(all_setups: pd.DataFrame, study_cfg: dict[str, Any]) -> pd.DataFrame:
    if all_setups.empty:
        return all_setups.copy()
    frame = all_setups.copy()
    min_score = int(study_cfg["candidate_min_score"])
    frame = frame[
        frame["setup_status"].astype(str).isin(["Breakout Triggered", "Ready Now", "Near Breakout", "Trend Watchlist"])
        & (pd.to_numeric(frame["swing_score"], errors="coerce") >= min_score)
    ].copy()
    return frame.reset_index(drop=True)


def _build_summary(
    exchange: str,
    universe: pd.DataFrame,
    all_setups: pd.DataFrame,
    candidates: pd.DataFrame,
    study_cfg: dict[str, Any],
) -> dict[str, Any]:
    setup_counts = candidates["setup_status"].astype(str).value_counts() if not candidates.empty else pd.Series(dtype="int64")
    avg_score = pd.to_numeric(all_setups.get("swing_score", pd.Series(dtype="float64")), errors="coerce").mean()
    avg_rs = pd.to_numeric(candidates.get("relative_strength_12w_pct", pd.Series(dtype="float64")), errors="coerce").mean()
    return {
        "exchange": exchange,
        "symbols_in_scan": len(universe),
        "symbols_scored": len(all_setups),
        "candidates_found": len(candidates),
        "breakout_triggered_count": int(setup_counts.get("Breakout Triggered", 0)),
        "ready_now_count": int(setup_counts.get("Ready Now", 0)),
        "near_breakout_count": int(setup_counts.get("Near Breakout", 0)),
        "trend_watchlist_count": int(setup_counts.get("Trend Watchlist", 0)),
        "avg_swing_score": float(avg_score) if pd.notna(avg_score) else 0.0,
        "avg_relative_strength_12w_pct": float(avg_rs) if pd.notna(avg_rs) else 0.0,
        "candidate_min_score": int(study_cfg["candidate_min_score"]),
        "max_distance_to_pivot_pct": float(study_cfg["max_distance_to_pivot_pct"]),
        "min_volume_ratio": float(study_cfg["min_volume_ratio"]),
        "min_risk_reward_ratio": float(study_cfg["min_risk_reward_ratio"]),
    }


def _prepare_daily(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _sort_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    ordered = frame.copy()
    status_order = {
        "Breakout Triggered": 0,
        "Ready Now": 1,
        "Near Breakout": 2,
        "Trend Watchlist": 3,
        "Not Ready": 4,
    }
    ordered["status_order"] = ordered["setup_status"].astype(str).map(status_order).fillna(9)
    for column in ("swing_score", "relative_strength_12w_pct", "risk_reward_ratio", "volume_ratio_20d"):
        if column in ordered.columns:
            ordered[column] = pd.to_numeric(ordered[column], errors="coerce")
    ordered = ordered.sort_values(
        ["status_order", "swing_score", "relative_strength_12w_pct", "risk_reward_ratio", "volume_ratio_20d", "symbol"],
        ascending=[True, False, False, False, False, True],
    )
    return ordered.drop(columns=["status_order"], errors="ignore").reset_index(drop=True)


def _sort_all_setups(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    ordered = frame.copy()
    ordered["swing_score"] = pd.to_numeric(ordered["swing_score"], errors="coerce")
    ordered["relative_strength_12w_pct"] = pd.to_numeric(ordered["relative_strength_12w_pct"], errors="coerce")
    return ordered.sort_values(["swing_score", "relative_strength_12w_pct", "symbol"], ascending=[False, False, True]).reset_index(drop=True)


def _setup_columns() -> list[str]:
    return [
        "exchange",
        "symbol",
        "name",
        "industry",
        "market_cap_cr",
        "market_cap_bucket",
        "latest_signal",
        "latest_signal_date",
        "latest_close",
        "latest_close_date",
        "swing_score",
        "setup_status",
        "trend_template_pass",
        "weekly_alignment_pass",
        "weekly_technical_rating",
        "relative_strength_pass",
        "relative_strength_benchmark",
        "stock_return_12w_pct",
        "benchmark_return_12w_pct",
        "relative_strength_12w_pct",
        "volume_setup_pass",
        "volume_ratio_20d",
        "obv_slope_20d",
        "breakout_proximity_pass",
        "pivot_high",
        "distance_to_pivot_pct",
        "base_contraction_pct",
        "risk_reward_pass",
        "stop_price",
        "target_price",
        "risk_pct",
        "reward_pct",
        "risk_reward_ratio",
        "breakout_now",
    ]


def _safe_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _emit_progress(progress_callback: Callable[[dict[str, Any]], None] | None, **payload: Any) -> None:
    if progress_callback:
        progress_callback(payload)
