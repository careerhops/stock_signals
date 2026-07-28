from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from stock_screener.data.storage import Storage
from stock_screener.resample import resample_daily_to_weekly
from stock_screener.strategy.weekly_buy_sell import run_weekly_buy_sell


DEFAULT_START_DATE = "2026-01-01"


@dataclass(frozen=True)
class SensitivityOverlapResult:
    summary: dict[str, Any]
    weekly_breakdown: pd.DataFrame
    latest_cohort: pd.DataFrame
    conversion_details: pd.DataFrame


def build_next_week_conversion_markers(
    config: dict[str, Any],
    storage: Storage,
    exchange: str = "NSE",
    symbols: set[str] | None = None,
    start_date: str = DEFAULT_START_DATE,
) -> pd.DataFrame:
    symbol_set = {str(symbol).upper().strip() for symbol in (symbols or set()) if str(symbol).strip()}
    if not symbol_set:
        return pd.DataFrame(columns=[
            "exchange",
            "symbol",
            "s2_to_s3_next_week_seen",
            "s2_to_s3_next_week_count",
            "s2_to_s3_first_s2_date",
            "s2_to_s3_latest_s2_date",
        ])

    result = run_sensitivity_overlap_study(
        config,
        storage,
        exchange=exchange,
        start_date=start_date,
        symbols=symbol_set,
        progress_callback=None,
    )
    details = result.conversion_details.copy()
    if details.empty:
        return pd.DataFrame(columns=[
            "exchange",
            "symbol",
            "s2_to_s3_next_week_seen",
            "s2_to_s3_next_week_count",
            "s2_to_s3_first_s2_date",
            "s2_to_s3_latest_s2_date",
        ])
    details["symbol"] = details["symbol"].astype(str).str.upper().str.strip()
    details = details[details["symbol"].isin(symbol_set)].copy()
    details = details[details["next_week_match"] == True].copy()
    if details.empty:
        base = pd.DataFrame({"exchange": exchange, "symbol": sorted(symbol_set)})
        base["s2_to_s3_next_week_seen"] = False
        base["s2_to_s3_next_week_count"] = 0
        base["s2_to_s3_first_s2_date"] = pd.NA
        base["s2_to_s3_latest_s2_date"] = pd.NA
        return base
    grouped = (
        details.groupby(["exchange", "symbol"], dropna=False)
        .agg(
            s2_to_s3_next_week_count=("next_week_match", "sum"),
            s2_to_s3_first_s2_date=("s2_date", "min"),
            s2_to_s3_latest_s2_date=("s2_date", "max"),
        )
        .reset_index()
    )
    grouped["s2_to_s3_next_week_seen"] = True
    base = pd.DataFrame({"exchange": exchange, "symbol": sorted(symbol_set)})
    merged = base.merge(grouped, on=["exchange", "symbol"], how="left")
    merged["s2_to_s3_next_week_seen"] = merged["s2_to_s3_next_week_seen"].fillna(False).astype(bool)
    merged["s2_to_s3_next_week_count"] = pd.to_numeric(merged["s2_to_s3_next_week_count"], errors="coerce").fillna(0).astype(int)
    return merged


def run_sensitivity_overlap_study(
    config: dict[str, Any],
    storage: Storage,
    exchange: str = "NSE",
    start_date: str = DEFAULT_START_DATE,
    symbols: set[str] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> SensitivityOverlapResult:
    data_root = storage.data_root
    all_symbols = sorted(p.stem for p in (data_root / "candles" / exchange / "1D").glob("*.csv"))
    if symbols is not None:
        allowed_symbols = {str(symbol).upper().strip() for symbol in symbols if str(symbol).strip()}
        all_symbols = [symbol for symbol in all_symbols if symbol.upper() in allowed_symbols]
    total = len(all_symbols)
    weekly_anchor = config.get("strategy", {}).get("weekly_anchor", "W-FRI")
    use_completed_weeks_only = bool(config.get("strategy", {}).get("use_completed_weeks_only", True))
    start_ts = pd.Timestamp(start_date)
    name_map = _load_name_map(storage, exchange)

    s2_frames: list[pd.DataFrame] = []
    s3_frames: list[pd.DataFrame] = []
    processed = 0

    for symbol in all_symbols:
        daily = storage.load_candles(exchange, symbol, "1D")
        processed += 1
        _emit_progress(
            progress_callback,
            phase="Comparing weekly BUY signals",
            completed=processed,
            total=total,
            current_symbol=symbol,
            current_exchange=exchange,
        )
        if daily.empty or len(daily) < 40:
            continue
        weekly = resample_daily_to_weekly(
            daily,
            weekly_anchor=weekly_anchor,
            use_completed_weeks_only=use_completed_weeks_only,
        )
        if weekly.empty:
            continue
        for sensitivity, bucket in ((2, s2_frames), (3, s3_frames)):
            strategy_config = _config_with_sensitivity(config, sensitivity)
            strategy_output = run_weekly_buy_sell(weekly, strategy_config)
            if strategy_output.empty:
                continue
            buys = strategy_output[
                (strategy_output["signal"].astype(str).str.upper() == "BUY")
                & (pd.to_datetime(strategy_output["date"], errors="coerce") >= start_ts)
            ].copy()
            if buys.empty:
                continue
            buys["date"] = pd.to_datetime(buys["date"], errors="coerce")
            buys = buys[buys["date"].notna()].copy()
            if buys.empty:
                continue
            buys["exchange"] = exchange
            buys["symbol"] = symbol
            buys["name"] = name_map.get(symbol, symbol)
            buys["sensitivity"] = sensitivity
            bucket.append(buys[["exchange", "symbol", "name", "date", "close", "sensitivity"]])

    s2_events = pd.concat(s2_frames, ignore_index=True) if s2_frames else _empty_event_frame()
    s3_events = pd.concat(s3_frames, ignore_index=True) if s3_frames else _empty_event_frame()
    return _build_overlap_outputs(exchange, start_ts, all_symbols, s2_events, s3_events)


def save_sensitivity_overlap_outputs(result: SensitivityOverlapResult, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "latest_summary.csv"
    weekly_breakdown_path = output_dir / "latest_weekly_breakdown.csv"
    latest_cohort_path = output_dir / "latest_latest_cohort.csv"
    conversion_details_path = output_dir / "latest_conversion_details.csv"
    pd.DataFrame([result.summary]).to_csv(summary_path, index=False)
    result.weekly_breakdown.to_csv(weekly_breakdown_path, index=False)
    result.latest_cohort.to_csv(latest_cohort_path, index=False)
    result.conversion_details.to_csv(conversion_details_path, index=False)
    return {
        "summary": summary_path,
        "weekly_breakdown": weekly_breakdown_path,
        "latest_cohort": latest_cohort_path,
        "conversion_details": conversion_details_path,
    }


def load_sensitivity_overlap_outputs(output_dir: Path) -> SensitivityOverlapResult:
    def _read(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    summary: dict[str, Any] = {}
    summary_path = output_dir / "latest_summary.csv"
    if summary_path.exists():
        try:
            summary_frame = pd.read_csv(summary_path)
            if not summary_frame.empty:
                summary = summary_frame.iloc[0].to_dict()
        except pd.errors.EmptyDataError:
            summary = {}

    return SensitivityOverlapResult(
        summary=summary,
        weekly_breakdown=_read(output_dir / "latest_weekly_breakdown.csv"),
        latest_cohort=_read(output_dir / "latest_latest_cohort.csv"),
        conversion_details=_read(output_dir / "latest_conversion_details.csv"),
    )


def _build_overlap_outputs(
    exchange: str,
    start_date: pd.Timestamp,
    symbols: list[str],
    s2_events: pd.DataFrame,
    s3_events: pd.DataFrame,
) -> SensitivityOverlapResult:
    s2 = _normalize_events(s2_events)
    s3 = _normalize_events(s3_events)
    s3_rows_by_symbol = {
        symbol: group.sort_values("date")[["date", "close"]].to_dict(orient="records")
        for symbol, group in s3.groupby("symbol", dropna=False)
    }

    conversion_rows: list[dict[str, Any]] = []
    for row in s2.itertuples(index=False):
        records = s3_rows_by_symbol.get(row.symbol, [])
        same_week_match = False
        first_s3_date = pd.NaT
        lead_weeks: int | None = None
        next_week_match = False
        next_week_s3_date = pd.NaT
        next_week_return_pct = pd.NA
        within_1w = False
        within_2w = False
        within_4w = False
        for record in records:
            candidate = pd.Timestamp(record["date"])
            delta_weeks = int((candidate - row.date).days // 7)
            if delta_weeks < 0:
                continue
            if first_s3_date is pd.NaT:
                first_s3_date = candidate
                lead_weeks = delta_weeks
            if delta_weeks == 1 and not next_week_match:
                next_week_match = True
                next_week_s3_date = candidate
                s2_close = _to_float(row.close)
                s3_close = _to_float(record["close"])
                if s2_close is not None and s2_close != 0 and s3_close is not None:
                    next_week_return_pct = ((s3_close - s2_close) / s2_close) * 100.0
            if delta_weeks == 0:
                same_week_match = True
            if delta_weeks <= 1:
                within_1w = True
            if delta_weeks <= 2:
                within_2w = True
            if delta_weeks <= 4:
                within_4w = True
                if same_week_match:
                    break
            if delta_weeks > 4:
                break
        conversion_rows.append(
            {
                "exchange": row.exchange,
                "symbol": row.symbol,
                "name": row.name,
                "s2_date": row.date,
                "same_week_match": same_week_match,
                "first_s3_date": first_s3_date,
                "lead_weeks": lead_weeks,
                "s2_close": row.close,
                "next_week_match": next_week_match,
                "next_week_s3_date": next_week_s3_date,
                "next_week_return_pct": next_week_return_pct,
                "within_1w": within_1w,
                "within_2w": within_2w,
                "within_4w": within_4w,
                "later_convert_within_4w": (not same_week_match) and within_4w,
            }
        )

    conversions = pd.DataFrame(conversion_rows)
    if conversions.empty:
        summary = {
            "exchange": exchange,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "symbols_processed": len(symbols),
            "s2_buy_events": 0,
            "s3_buy_events": 0,
            "s2_unique_symbols": 0,
            "s3_unique_symbols": 0,
            "same_week_overlap_events": 0,
            "same_week_overlap_pct_of_s2": 0.0,
            "within_1w_overlap_events": 0,
            "within_1w_overlap_pct_of_s2": 0.0,
            "within_2w_overlap_events": 0,
            "within_2w_overlap_pct_of_s2": 0.0,
            "within_4w_overlap_events": 0,
            "within_4w_overlap_pct_of_s2": 0.0,
            "extra_s2_events": 0,
            "extra_s2_later_convert_4w_events": 0,
            "extra_s2_later_convert_4w_pct": 0.0,
            "latest_week_date": "",
            "latest_s2_count": 0,
            "latest_s3_count": 0,
            "latest_overlap_count": 0,
            "latest_s2_only_count": 0,
            "latest_s3_only_count": 0,
        }
        return SensitivityOverlapResult(summary, pd.DataFrame(), pd.DataFrame(), conversions)

    same_week_count = int(conversions["same_week_match"].sum())
    next_week_count = int(conversions["next_week_match"].sum())
    within_1w_count = int(conversions["within_1w"].sum())
    within_2w_count = int(conversions["within_2w"].sum())
    within_4w_count = int(conversions["within_4w"].sum())
    extra_s2_events = int((~conversions["same_week_match"]).sum())
    extra_later_convert_4w = int(conversions["later_convert_within_4w"].sum())
    next_week_returns = pd.to_numeric(
        conversions.loc[conversions["next_week_match"], "next_week_return_pct"],
        errors="coerce",
    ).dropna()

    weekly_breakdown = (
        conversions.groupby("s2_date", dropna=False)
        .agg(
            s2_buy_count=("symbol", "count"),
            same_week_overlap_count=("same_week_match", "sum"),
            next_week_overlap_count=("next_week_match", "sum"),
            within_1w_overlap_count=("within_1w", "sum"),
            within_2w_overlap_count=("within_2w", "sum"),
            within_4w_overlap_count=("within_4w", "sum"),
            extra_s2_count=("same_week_match", lambda s: int((~s.astype(bool)).sum())),
            extra_later_convert_4w_count=("later_convert_within_4w", "sum"),
            next_week_avg_return_pct=("next_week_return_pct", _safe_mean),
            next_week_median_return_pct=("next_week_return_pct", _safe_median),
            next_week_positive_return_pct=("next_week_return_pct", _positive_return_pct),
        )
        .reset_index()
        .rename(columns={"s2_date": "week_date"})
    )
    next_week_symbols = (
        conversions[conversions["next_week_match"] == True]
        .groupby("s2_date", dropna=False)["symbol"]
        .agg(_comma_join_symbols)
        .reset_index()
        .rename(columns={"s2_date": "week_date", "symbol": "next_week_symbols"})
    )
    weekly_breakdown = weekly_breakdown.merge(next_week_symbols, on="week_date", how="left")
    weekly_breakdown["next_week_symbols"] = weekly_breakdown["next_week_symbols"].fillna("")
    s3_week_counts = s3.groupby("date", dropna=False).size().rename("s3_buy_count").reset_index().rename(columns={"date": "week_date"})
    weekly_breakdown = weekly_breakdown.merge(s3_week_counts, on="week_date", how="left")
    weekly_breakdown["s3_buy_count"] = weekly_breakdown["s3_buy_count"].fillna(0).astype(int)
    for column in (
        "same_week_overlap_count",
        "next_week_overlap_count",
        "within_1w_overlap_count",
        "within_2w_overlap_count",
        "within_4w_overlap_count",
        "extra_s2_count",
        "extra_later_convert_4w_count",
    ):
        weekly_breakdown[column] = weekly_breakdown[column].fillna(0).astype(int)
    weekly_breakdown["same_week_overlap_pct_of_s2"] = 100 * weekly_breakdown["same_week_overlap_count"] / weekly_breakdown["s2_buy_count"].where(weekly_breakdown["s2_buy_count"] > 0, 1)
    weekly_breakdown["next_week_overlap_pct_of_s2"] = 100 * weekly_breakdown["next_week_overlap_count"] / weekly_breakdown["s2_buy_count"].where(weekly_breakdown["s2_buy_count"] > 0, 1)
    weekly_breakdown["within_4w_overlap_pct_of_s2"] = 100 * weekly_breakdown["within_4w_overlap_count"] / weekly_breakdown["s2_buy_count"].where(weekly_breakdown["s2_buy_count"] > 0, 1)
    weekly_breakdown["extra_later_convert_4w_pct"] = 100 * weekly_breakdown["extra_later_convert_4w_count"] / weekly_breakdown["extra_s2_count"].where(weekly_breakdown["extra_s2_count"] > 0, 1)
    weekly_breakdown = weekly_breakdown.sort_values("week_date", ascending=False).reset_index(drop=True)

    latest_week_date = max(
        s2["date"].max() if not s2.empty else pd.Timestamp.min,
        s3["date"].max() if not s3.empty else pd.Timestamp.min,
    )
    latest_s2 = s2[s2["date"] == latest_week_date][["exchange", "symbol", "name"]].drop_duplicates(subset=["exchange", "symbol"])
    latest_s3 = s3[s3["date"] == latest_week_date][["exchange", "symbol", "name"]].drop_duplicates(subset=["exchange", "symbol"])
    latest_cohort = pd.merge(
        latest_s2.assign(in_s2=True),
        latest_s3.assign(in_s3=True),
        on=["exchange", "symbol"],
        how="outer",
        suffixes=("_s2", "_s3"),
    )
    latest_cohort["name"] = latest_cohort.get("name_s2", "").fillna("").astype(str).str.strip()
    latest_cohort["name"] = latest_cohort["name"].mask(
        latest_cohort["name"] == "",
        latest_cohort.get("name_s3", "").fillna("").astype(str).str.strip(),
    )
    latest_cohort["in_s2"] = latest_cohort["in_s2"] == True
    latest_cohort["in_s3"] = latest_cohort["in_s3"] == True
    latest_cohort["bucket"] = "Sensitivity 2 only"
    latest_cohort.loc[latest_cohort["in_s3"] & ~latest_cohort["in_s2"], "bucket"] = "Sensitivity 3 only"
    latest_cohort.loc[latest_cohort["in_s2"] & latest_cohort["in_s3"], "bucket"] = "Both"
    latest_cohort["week_date"] = latest_week_date
    latest_cohort = latest_cohort[["week_date", "exchange", "symbol", "name", "bucket", "in_s2", "in_s3"]].sort_values(["bucket", "symbol"]).reset_index(drop=True)

    summary = {
        "exchange": exchange,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "symbols_processed": len(symbols),
        "s2_buy_events": int(len(s2)),
        "s3_buy_events": int(len(s3)),
        "s2_unique_symbols": int(s2["symbol"].nunique()) if not s2.empty else 0,
        "s3_unique_symbols": int(s3["symbol"].nunique()) if not s3.empty else 0,
        "same_week_overlap_events": same_week_count,
        "same_week_overlap_pct_of_s2": _pct(same_week_count, len(s2)),
        "next_week_overlap_events": next_week_count,
        "next_week_overlap_pct_of_s2": _pct(next_week_count, len(s2)),
        "within_1w_overlap_events": within_1w_count,
        "within_1w_overlap_pct_of_s2": _pct(within_1w_count, len(s2)),
        "within_2w_overlap_events": within_2w_count,
        "within_2w_overlap_pct_of_s2": _pct(within_2w_count, len(s2)),
        "within_4w_overlap_events": within_4w_count,
        "within_4w_overlap_pct_of_s2": _pct(within_4w_count, len(s2)),
        "extra_s2_events": extra_s2_events,
        "extra_s2_later_convert_4w_events": extra_later_convert_4w,
        "extra_s2_later_convert_4w_pct": _pct(extra_later_convert_4w, extra_s2_events),
        "next_week_avg_return_pct": float(next_week_returns.mean()) if not next_week_returns.empty else 0.0,
        "next_week_median_return_pct": float(next_week_returns.median()) if not next_week_returns.empty else 0.0,
        "next_week_positive_return_pct": _pct(int((next_week_returns > 0).sum()), len(next_week_returns)),
        "latest_week_date": latest_week_date.strftime("%Y-%m-%d") if pd.notna(latest_week_date) else "",
        "latest_s2_count": int(len(latest_s2)),
        "latest_s3_count": int(len(latest_s3)),
        "latest_overlap_count": int(((latest_cohort["in_s2"]) & (latest_cohort["in_s3"])).sum()) if not latest_cohort.empty else 0,
        "latest_s2_only_count": int(((latest_cohort["in_s2"]) & (~latest_cohort["in_s3"])).sum()) if not latest_cohort.empty else 0,
        "latest_s3_only_count": int(((~latest_cohort["in_s2"]) & (latest_cohort["in_s3"])).sum()) if not latest_cohort.empty else 0,
    }

    conversions["s2_date"] = pd.to_datetime(conversions["s2_date"], errors="coerce")
    conversions["first_s3_date"] = pd.to_datetime(conversions["first_s3_date"], errors="coerce")
    conversions["next_week_s3_date"] = pd.to_datetime(conversions["next_week_s3_date"], errors="coerce")
    conversions = conversions.sort_values(["s2_date", "symbol"], ascending=[False, True]).reset_index(drop=True)

    return SensitivityOverlapResult(summary, weekly_breakdown, latest_cohort, conversions)


def _normalize_events(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_event_frame()
    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized = normalized[normalized["date"].notna()].copy()
    normalized["symbol"] = normalized["symbol"].astype(str).str.upper().str.strip()
    normalized["name"] = normalized["name"].fillna("").astype(str).str.strip().mask(lambda s: s == "", normalized["symbol"])
    normalized["close"] = pd.to_numeric(normalized["close"], errors="coerce")
    return normalized[["exchange", "symbol", "name", "date", "close", "sensitivity"]].drop_duplicates(subset=["exchange", "symbol", "date", "sensitivity"], keep="last").sort_values(["symbol", "date"]).reset_index(drop=True)


def _empty_event_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["exchange", "symbol", "name", "date", "close", "sensitivity"])


def _load_name_map(storage: Storage, exchange: str) -> dict[str, str]:
    instruments = storage.load_instruments()
    if instruments.empty:
        return {}
    frame = instruments.copy()
    frame["exchange"] = frame.get("exchange", exchange).astype(str).str.upper()
    frame = frame[frame["exchange"] == exchange.upper()].copy()
    if "tradingsymbol" not in frame.columns:
        return {}
    frame["symbol"] = frame["tradingsymbol"].astype(str).str.upper().str.strip()
    frame["name"] = frame.get("name", frame["symbol"]).fillna("").astype(str).str.strip().mask(lambda s: s == "", frame["symbol"])
    return frame.drop_duplicates(subset=["symbol"], keep="last").set_index("symbol")["name"].to_dict()


def _config_with_sensitivity(config: dict[str, Any], sensitivity: int) -> dict[str, Any]:
    updated = {
        **config,
        "strategy": {
            **config.get("strategy", {}),
            "sensitivity": int(sensitivity),
        },
    }
    return updated


def _emit_progress(progress_callback: Callable[[dict[str, Any]], None] | None, **payload: Any) -> None:
    if progress_callback is not None:
        progress_callback(payload)


def _pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return 100.0 * float(numerator) / float(denominator)


def _to_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric


def _safe_mean(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(numeric.mean())


def _safe_median(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(numeric.median())


def _positive_return_pct(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return _pct(int((numeric > 0).sum()), len(numeric))


def _comma_join_symbols(series: pd.Series) -> str:
    values = sorted({str(value).strip().upper() for value in series if str(value).strip()})
    return ", ".join(values)
