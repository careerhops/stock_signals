from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from stock_screener.data.storage import Storage
from stock_screener.resample import resample_daily_to_weekly
from stock_screener.strategy.weekly_buy_sell import run_weekly_buy_sell


@dataclass(frozen=True)
class SignalOutcomeStudyResult:
    summary: dict[str, Any]
    signal_universe: pd.DataFrame
    stock_stats: pd.DataFrame
    pair_details: pd.DataFrame


DEFAULT_CONFIG = {
    "lookback_years": 5,
    "target_gain_pct": 10.0,
    "signal_scope": "buy",
}


def run_signal_outcome_study(
    config: dict[str, Any],
    storage: Storage,
    exchange: str = "NSE",
    signal_scope: str = "buy",
    target_gain_pct: float = 10.0,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> SignalOutcomeStudyResult:
    study_cfg = {**DEFAULT_CONFIG, **(config.get("signal_outcome_study", {}) or {})}
    scope = (signal_scope or str(study_cfg.get("signal_scope", "buy"))).strip().lower()
    if scope not in {"buy", "sell", "both"}:
        scope = "buy"
    target_gain_pct = float(target_gain_pct or study_cfg.get("target_gain_pct", 10.0))

    signal_universe = _current_signal_universe_from_saved_scan(storage, exchange, scope)
    if signal_universe.empty:
        raise RuntimeError("No fresh weekly BUY/SELL signals are available. Run the screener first, then retry the Signal Outcome Study.")

    strategy_cfg = config.get("strategy", {})
    weekly_anchor = strategy_cfg.get("weekly_anchor", "W-FRI")
    use_completed_weeks_only = bool(strategy_cfg.get("use_completed_weeks_only", True))
    lookback_start = pd.Timestamp.today().normalize() - pd.Timedelta(days=365 * int(study_cfg.get("lookback_years", 5)))

    pair_rows: list[dict[str, Any]] = []

    _emit_progress(progress_callback, phase="Analyzing historical BUY outcomes", completed=0, total=len(signal_universe), current_symbol="", current_exchange=exchange)
    for index, universe_row in enumerate(signal_universe.to_dict(orient="records"), start=1):
        row_exchange = str(universe_row.get("exchange") or exchange).upper()
        symbol = str(universe_row.get("symbol") or "").upper()
        name = str(universe_row.get("name") or symbol)
        if not symbol:
            continue

        daily = storage.load_candles(row_exchange, symbol, "1D")
        if daily.empty:
            _emit_progress(progress_callback, phase="Analyzing historical BUY outcomes", completed=index, total=len(signal_universe), current_symbol=symbol, current_exchange=row_exchange)
            continue

        daily = _prepare_daily(daily)
        weekly = resample_daily_to_weekly(daily, weekly_anchor, use_completed_weeks_only)
        if weekly.empty:
            _emit_progress(progress_callback, phase="Analyzing historical BUY outcomes", completed=index, total=len(signal_universe), current_symbol=symbol, current_exchange=row_exchange)
            continue

        strategy_output = run_weekly_buy_sell(weekly, config)

        pair_rows.extend(
            build_signal_outcome_pairs(
                daily=daily,
                strategy_output=strategy_output,
                exchange=row_exchange,
                symbol=symbol,
                name=name,
                target_gain_pct=target_gain_pct,
                lookback_start=lookback_start,
            ).to_dict(orient="records")
        )
        _emit_progress(progress_callback, phase="Analyzing historical BUY outcomes", completed=index, total=len(signal_universe), current_symbol=symbol, current_exchange=row_exchange)

    pair_details = pd.DataFrame(pair_rows, columns=_pair_detail_columns())
    stock_stats = build_stock_outcome_stats(pair_details, signal_universe, target_gain_pct)
    summary = build_signal_outcome_summary(signal_universe, stock_stats, pair_details, exchange, scope, target_gain_pct)
    return SignalOutcomeStudyResult(summary, _sort_signal_universe(signal_universe), _sort_stock_stats(stock_stats), _sort_pair_details(pair_details))


def save_signal_outcome_outputs(result: SignalOutcomeStudyResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result.summary]).to_csv(output_dir / "latest_summary.csv", index=False)
    result.signal_universe.to_csv(output_dir / "latest_signal_universe.csv", index=False)
    result.stock_stats.to_csv(output_dir / "latest_stock_stats.csv", index=False)
    result.pair_details.to_csv(output_dir / "latest_pair_details.csv", index=False)


def load_signal_outcome_outputs(output_dir: Path) -> SignalOutcomeStudyResult:
    summary = {}
    summary_path = output_dir / "latest_summary.csv"
    if summary_path.exists():
        summary_frame = _read_csv(summary_path)
        if not summary_frame.empty:
            summary = summary_frame.iloc[0].to_dict()
    return SignalOutcomeStudyResult(
        summary=summary,
        signal_universe=_read_csv(output_dir / "latest_signal_universe.csv"),
        stock_stats=_read_csv(output_dir / "latest_stock_stats.csv"),
        pair_details=_read_csv(output_dir / "latest_pair_details.csv"),
    )


def build_signal_outcome_pairs(
    daily: pd.DataFrame,
    strategy_output: pd.DataFrame,
    exchange: str,
    symbol: str,
    name: str,
    target_gain_pct: float,
    lookback_start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if daily.empty or strategy_output.empty:
        return _empty_pair_details_frame()

    daily_frame = _prepare_daily(daily)
    strategy = strategy_output.copy()
    strategy["date"] = pd.to_datetime(strategy["date"], errors="coerce")
    strategy = strategy.sort_values("date").reset_index(drop=True)

    active_buy: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []

    for _, row in strategy.iterrows():
        row_date = pd.to_datetime(row["date"], errors="coerce")
        if pd.isna(row_date):
            continue
        if lookback_start is not None and row_date < lookback_start:
            if bool(row.get("final_buy", False)):
                active_buy = None
            continue

        if bool(row.get("final_buy", False)):
            active_buy = {
                "exchange": exchange,
                "symbol": symbol,
                "name": name,
                "buy_date": row_date,
                "buy_close": float(row["close"]),
            }
        elif bool(row.get("final_sell", False)) and active_buy is not None:
            sell_date = row_date
            sell_close = float(row["close"])
            pair_stats = _pair_outcome_between_dates(daily_frame, active_buy["buy_date"], sell_date, float(active_buy["buy_close"]), target_gain_pct)
            sell_return_pct = ((sell_close - float(active_buy["buy_close"])) / float(active_buy["buy_close"])) * 100
            rows.append(
                {
                    **active_buy,
                    "sell_date": sell_date,
                    "sell_close": sell_close,
                    "sell_return_pct": sell_return_pct,
                    "outcome": "WIN" if sell_return_pct > 0 else "LOSS" if sell_return_pct < 0 else "FLAT",
                    **pair_stats,
                    "target_miss": not bool(pair_stats["hit_target_pct"]),
                    "failed_buy_flag": (not bool(pair_stats["hit_target_pct"])) and sell_return_pct <= 0,
                }
            )
            active_buy = None

    return pd.DataFrame(rows, columns=_pair_detail_columns())


def build_stock_outcome_stats(
    pair_details: pd.DataFrame,
    signal_universe: pd.DataFrame,
    target_gain_pct: float,
) -> pd.DataFrame:
    if signal_universe.empty:
        return _empty_stock_stats_frame()

    base = signal_universe.copy()
    if pair_details.empty:
        for column in _stock_stat_columns():
            if column not in base.columns:
                base[column] = pd.NA
        base["historical_buy_count"] = 0
        base["target_hit_count"] = 0
        base["target_hit_rate_pct"] = 0.0
        base["win_count"] = 0
        base["loss_count"] = 0
        base["target_miss_count"] = 0
        base["failed_buy_count"] = 0
        base["failed_buy_rate_pct"] = 0.0
        return base[_stock_stat_columns()]

    grouped = pair_details.groupby(["exchange", "symbol", "name"], dropna=False)
    stats = grouped["buy_date"].count().rename("historical_buy_count").reset_index()
    stats["target_hit_count"] = grouped["hit_target_pct"].sum().values
    stats["target_hit_rate_pct"] = (stats["target_hit_count"] / stats["historical_buy_count"]) * 100
    stats["median_days_to_target"] = grouped["days_to_target"].median().values
    stats["avg_days_to_target"] = grouped["days_to_target"].mean().values
    stats["median_peak_gain_pct"] = grouped["max_gain_pct_before_sell"].median().values
    stats["avg_peak_gain_pct"] = grouped["max_gain_pct_before_sell"].mean().values
    stats["best_peak_gain_pct"] = grouped["max_gain_pct_before_sell"].max().values
    stats["median_sell_return_pct"] = grouped["sell_return_pct"].median().values
    stats["avg_sell_return_pct"] = grouped["sell_return_pct"].mean().values
    stats["win_count"] = grouped.apply(lambda frame: int((pd.to_numeric(frame["sell_return_pct"], errors="coerce") > 0).sum()), include_groups=False).values
    stats["loss_count"] = grouped.apply(lambda frame: int((pd.to_numeric(frame["sell_return_pct"], errors="coerce") < 0).sum()), include_groups=False).values
    stats["target_miss_count"] = grouped["target_miss"].sum().values
    stats["failed_buy_count"] = grouped["failed_buy_flag"].sum().values
    stats["failed_buy_rate_pct"] = (stats["failed_buy_count"] / stats["historical_buy_count"]) * 100
    stats["target_gain_pct"] = float(target_gain_pct)
    stats["interpretation"] = stats.apply(_interpretation_label, axis=1)

    merged = base.merge(stats, on=["exchange", "symbol", "name"], how="left")
    numeric_fill_zero = [
        "historical_buy_count",
        "target_hit_count",
        "target_hit_rate_pct",
        "win_count",
        "loss_count",
        "target_miss_count",
        "failed_buy_count",
        "failed_buy_rate_pct",
    ]
    for column in numeric_fill_zero:
        if column in merged.columns:
            merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0)
    return merged[_stock_stat_columns()]


def build_signal_outcome_summary(
    signal_universe: pd.DataFrame,
    stock_stats: pd.DataFrame,
    pair_details: pd.DataFrame,
    exchange: str,
    signal_scope: str,
    target_gain_pct: float,
) -> dict[str, Any]:
    buy_count = int((signal_universe.get("current_signal", pd.Series(dtype=str)).astype(str).str.upper() == "BUY").sum()) if not signal_universe.empty else 0
    sell_count = int((signal_universe.get("current_signal", pd.Series(dtype=str)).astype(str).str.upper() == "SELL").sum()) if not signal_universe.empty else 0
    avg_hit_rate = pd.to_numeric(stock_stats.get("target_hit_rate_pct", pd.Series(dtype="float64")), errors="coerce").mean()
    median_days = pd.to_numeric(stock_stats.get("median_days_to_target", pd.Series(dtype="float64")), errors="coerce").median()
    median_peak = pd.to_numeric(stock_stats.get("median_peak_gain_pct", pd.Series(dtype="float64")), errors="coerce").median()
    failed_rate = pd.to_numeric(stock_stats.get("failed_buy_rate_pct", pd.Series(dtype="float64")), errors="coerce").mean()
    return {
        "exchange": exchange,
        "signal_scope": signal_scope,
        "target_gain_pct": float(target_gain_pct),
        "current_signal_universe_count": len(signal_universe),
        "current_buy_count": buy_count,
        "current_sell_count": sell_count,
        "historical_pairs_analyzed": len(pair_details),
        "avg_target_hit_rate_pct": float(avg_hit_rate) if pd.notna(avg_hit_rate) else 0.0,
        "median_days_to_target": float(median_days) if pd.notna(median_days) else 0.0,
        "median_peak_gain_pct": float(median_peak) if pd.notna(median_peak) else 0.0,
        "avg_failed_buy_rate_pct": float(failed_rate) if pd.notna(failed_rate) else 0.0,
    }


def _latest_week_signal_context(strategy_output: pd.DataFrame, exchange: str, symbol: str, name: str) -> dict[str, Any]:
    if strategy_output.empty:
        return {
            "exchange": exchange,
            "symbol": symbol,
            "name": name,
            "latest_week_date": pd.NA,
            "current_signal": "NONE",
            "current_signal_date": pd.NA,
        }

    frame = strategy_output.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.sort_values("date").reset_index(drop=True)
    latest = frame.iloc[-1]
    latest_week_date = pd.to_datetime(latest.get("date"), errors="coerce")
    current_signal = str(latest.get("signal", "NONE")).upper()
    if current_signal not in {"BUY", "SELL"}:
        current_signal = "NONE"
    return {
        "exchange": exchange,
        "symbol": symbol,
        "name": name,
        "latest_week_date": latest_week_date,
        "current_signal": current_signal,
        "current_signal_date": latest_week_date if current_signal in {"BUY", "SELL"} else pd.NA,
    }


def _pair_outcome_between_dates(
    daily: pd.DataFrame,
    buy_date: pd.Timestamp,
    sell_date: pd.Timestamp,
    buy_close: float,
    target_gain_pct: float,
) -> dict[str, Any]:
    frame = _prepare_daily(daily)
    window = frame[(frame["date"] > buy_date) & (frame["date"] < sell_date)].copy()
    if window.empty:
        return {
            "highest_price_before_sell": pd.NA,
            "highest_price_date": pd.NA,
            "days_to_peak": pd.NA,
            "max_gain_pct_before_sell": pd.NA,
            "hit_target_pct": False,
            "days_to_target": pd.NA,
        }

    highs = pd.to_numeric(window["high"], errors="coerce")
    valid_highs = highs.dropna()
    if valid_highs.empty:
        return {
            "highest_price_before_sell": pd.NA,
            "highest_price_date": pd.NA,
            "days_to_peak": pd.NA,
            "max_gain_pct_before_sell": pd.NA,
            "hit_target_pct": False,
            "days_to_target": pd.NA,
        }

    max_index = valid_highs.idxmax()
    highest_price = float(valid_highs.loc[max_index])
    highest_price_date = pd.Timestamp(window.loc[max_index, "date"])
    max_gain_pct = ((highest_price - buy_close) / buy_close) * 100

    target_price = buy_close * (1 + (float(target_gain_pct) / 100.0))
    target_hits = window[pd.to_numeric(window["high"], errors="coerce") >= target_price]
    if target_hits.empty:
        days_to_target = pd.NA
        hit_target = False
    else:
        first_hit_date = pd.Timestamp(target_hits.iloc[0]["date"])
        days_to_target = int((first_hit_date - pd.Timestamp(buy_date)).days)
        hit_target = True

    return {
        "highest_price_before_sell": highest_price,
        "highest_price_date": highest_price_date,
        "days_to_peak": int((highest_price_date - pd.Timestamp(buy_date)).days),
        "max_gain_pct_before_sell": max_gain_pct,
        "hit_target_pct": hit_target,
        "days_to_target": days_to_target,
    }


def _interpretation_label(row: pd.Series) -> str:
    hit_rate = float(pd.to_numeric(pd.Series([row.get("target_hit_rate_pct")]), errors="coerce").iloc[0] or 0.0)
    median_days = pd.to_numeric(pd.Series([row.get("median_days_to_target")]), errors="coerce").iloc[0]
    failed_rate = float(pd.to_numeric(pd.Series([row.get("failed_buy_rate_pct")]), errors="coerce").iloc[0] or 0.0)
    median_peak = float(pd.to_numeric(pd.Series([row.get("median_peak_gain_pct")]), errors="coerce").iloc[0] or 0.0)
    historical_buy_count = int(pd.to_numeric(pd.Series([row.get("historical_buy_count")]), errors="coerce").fillna(0).iloc[0])
    if historical_buy_count < 2:
        return "Low sample"
    if hit_rate >= 60 and pd.notna(median_days) and float(median_days) <= 30:
        return "Fast mover"
    if hit_rate >= 50 and median_peak >= 12:
        return "Strong follow-through"
    if failed_rate >= 60:
        return "Weak follow-through"
    return "Mixed"


def _matches_scope(signal: str, signal_scope: str) -> bool:
    normalized = str(signal or "NONE").upper()
    if signal_scope == "both":
        return normalized in {"BUY", "SELL"}
    return normalized == signal_scope.upper()


def _sort_signal_universe(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.sort_values(["current_signal", "symbol"], ascending=[True, True]).reset_index(drop=True)


def _sort_stock_stats(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    sortable = frame.copy()
    for column in ("target_hit_rate_pct", "median_peak_gain_pct", "median_days_to_target", "failed_buy_rate_pct"):
        if column in sortable.columns:
            sortable[column] = pd.to_numeric(sortable[column], errors="coerce")
    return sortable.sort_values(
        ["target_hit_rate_pct", "median_peak_gain_pct", "median_days_to_target", "failed_buy_rate_pct", "symbol"],
        ascending=[False, False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)


def _sort_pair_details(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.sort_values(["buy_date", "symbol"], ascending=[False, True]).reset_index(drop=True)


def _stock_stat_columns() -> list[str]:
    return [
        "exchange",
        "symbol",
        "name",
        "latest_week_date",
        "current_signal",
        "current_signal_date",
        "historical_buy_count",
        "target_gain_pct",
        "target_hit_count",
        "target_hit_rate_pct",
        "median_days_to_target",
        "avg_days_to_target",
        "median_peak_gain_pct",
        "avg_peak_gain_pct",
        "best_peak_gain_pct",
        "median_sell_return_pct",
        "avg_sell_return_pct",
        "win_count",
        "loss_count",
        "target_miss_count",
        "failed_buy_count",
        "failed_buy_rate_pct",
        "interpretation",
    ]


def _current_signal_universe_from_saved_scan(storage: Storage, exchange: str, signal_scope: str) -> pd.DataFrame:
    raw = storage.load_signals("latest_raw_signals.csv")
    if raw.empty or not {"exchange", "symbol", "date", "signal"}.issubset(raw.columns):
        return _empty_signal_universe_frame()

    frame = raw.copy()
    frame["exchange"] = frame["exchange"].astype(str).str.upper()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["signal"] = frame["signal"].astype(str).str.upper()
    frame = frame[
        (frame["exchange"] == exchange.upper())
        & (frame["signal"].isin(["BUY", "SELL"]))
        & frame["date"].notna()
    ].copy()
    if frame.empty:
        return _empty_signal_universe_frame()

    latest_global_signal_date = frame["date"].max()
    latest_per_symbol = frame.sort_values("date").groupby(["exchange", "symbol"], dropna=False).tail(1).copy()
    latest_per_symbol = latest_per_symbol[latest_per_symbol["date"] == latest_global_signal_date].copy()
    if latest_per_symbol.empty:
        return _empty_signal_universe_frame()

    latest_per_symbol["current_signal"] = latest_per_symbol["signal"]
    latest_per_symbol["latest_week_date"] = latest_per_symbol["date"]
    latest_per_symbol["current_signal_date"] = latest_per_symbol["date"]

    scan_details = storage.load_signals("latest_scan_details.csv")
    if not scan_details.empty and {"exchange", "symbol"}.issubset(scan_details.columns):
        details = scan_details.copy()
        details["exchange"] = details["exchange"].astype(str).str.upper()
        details["symbol"] = details["symbol"].astype(str).str.upper()
        keep_columns = [column for column in ["exchange", "symbol", "name"] if column in details.columns]
        if len(keep_columns) >= 2:
            latest_per_symbol = latest_per_symbol.merge(
                details[keep_columns].drop_duplicates(subset=["exchange", "symbol"], keep="last"),
                on=["exchange", "symbol"],
                how="left",
                suffixes=("", "_detail"),
            )

    if "name" not in latest_per_symbol.columns:
        latest_per_symbol["name"] = latest_per_symbol["symbol"]
    latest_per_symbol["name"] = (
        latest_per_symbol["name"]
        .fillna("")
        .astype(str)
        .str.strip()
        .mask(lambda s: s == "", latest_per_symbol["symbol"])
    )

    latest_per_symbol = latest_per_symbol[latest_per_symbol["current_signal"].apply(lambda value: _matches_scope(value, signal_scope))]
    if latest_per_symbol.empty:
        return _empty_signal_universe_frame()

    return latest_per_symbol[_signal_universe_columns()].reset_index(drop=True)


def _signal_universe_columns() -> list[str]:
    return [
        "exchange",
        "symbol",
        "name",
        "latest_week_date",
        "current_signal",
        "current_signal_date",
    ]


def _pair_detail_columns() -> list[str]:
    return [
        "exchange",
        "symbol",
        "name",
        "buy_date",
        "buy_close",
        "sell_date",
        "sell_close",
        "sell_return_pct",
        "outcome",
        "highest_price_before_sell",
        "highest_price_date",
        "days_to_peak",
        "max_gain_pct_before_sell",
        "hit_target_pct",
        "days_to_target",
        "target_miss",
        "failed_buy_flag",
    ]


def _empty_signal_universe_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_signal_universe_columns())


def _empty_pair_details_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_pair_detail_columns())


def _empty_stock_stats_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_stock_stat_columns())


def _prepare_daily(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame.sort_values("date").reset_index(drop=True)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _empty_summary(exchange: str, signal_scope: str, target_gain_pct: float) -> dict[str, Any]:
    return {
        "exchange": exchange,
        "signal_scope": signal_scope,
        "target_gain_pct": float(target_gain_pct),
        "current_signal_universe_count": 0,
        "current_buy_count": 0,
        "current_sell_count": 0,
        "historical_pairs_analyzed": 0,
        "avg_target_hit_rate_pct": 0.0,
        "median_days_to_target": 0.0,
        "median_peak_gain_pct": 0.0,
        "avg_failed_buy_rate_pct": 0.0,
    }


def _emit_progress(progress_callback: Callable[[dict[str, Any]], None] | None, **payload: Any) -> None:
    if progress_callback:
        progress_callback(payload)
