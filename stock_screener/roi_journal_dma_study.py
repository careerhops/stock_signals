from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Callable

import numpy as np
import pandas as pd

from stock_screener.data.storage import Storage
from stock_screener.stock_signature_study import run_stock_signature_study


DEFAULT_SHEET_NAME = "ROI Journal"
DEFAULT_LOOKBACK_MONTHS = 6
DEFAULT_PIVOT_ORDER = 3
DEFAULT_PRE_SIGNAL_SESSIONS = 10
DEFAULT_RECOVERY_THRESHOLD_PCT = 10.0
DEFAULT_DMA_WINDOWS = (75, 100)
DEFAULT_DMA_TREND_SESSIONS = 20
DEFAULT_75DMA_RECLAIM_SESSIONS = 7
DEFAULT_75DMA_RECLAIM_PRE_SIGNAL_SESSIONS = 1
DEFAULT_75DMA_RECLAIM_POST_SIGNAL_SESSIONS = 7
DMA_TREND_FLAT_THRESHOLD_PCT = 0.25
REQUIRED_COLUMNS = (
    "Date_Identifier",
    "Strategy",
    "Top_5",
    "Stocks",
    "Price_Asof",
    "Price_Latest",
    "Signal_Days",
    "Profit_Loss",
    "Actual_Return",
)
COLUMN_ALIASES = {
    "dateidentified": "Date_Identifier",
}


@dataclass(frozen=True)
class RoiJournalDmaResult:
    summary: dict[str, Any]
    signal_rows: pd.DataFrame
    distance_history: pd.DataFrame


def load_roi_journal_input(path: Path, *, sheet_name: str = DEFAULT_SHEET_NAME) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix in {".xlsx", ".xlsm"}:
        frame = pd.read_excel(path, sheet_name=sheet_name)
    elif suffix == ".xls":
        try:
            frame = pd.read_excel(path, sheet_name=sheet_name)
        except ImportError as exc:
            raise RuntimeError("Legacy .xls uploads require xlrd. Export the sheet as .xlsx or .csv.") from exc
    else:
        raise ValueError("Upload a .csv, .xlsx, .xlsm, or .xls file.")
    return _prepare_journal_frame(frame)


def run_roi_journal_dma_analysis(
    storage: Storage,
    journal: pd.DataFrame,
    *,
    latest_stock_date: Any | None = None,
    lookback_months: int = DEFAULT_LOOKBACK_MONTHS,
    pivot_order: int = DEFAULT_PIVOT_ORDER,
    pre_signal_sessions: int = DEFAULT_PRE_SIGNAL_SESSIONS,
    recovery_threshold_pct: float = DEFAULT_RECOVERY_THRESHOLD_PCT,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> RoiJournalDmaResult:
    prepared = _prepare_journal_frame(journal)
    latest_cutoff = _coerce_date(latest_stock_date)
    rows: list[dict[str, Any]] = []
    history_frames: list[pd.DataFrame] = []
    total = len(prepared)

    _emit_progress(
        progress_callback,
        phase="Analyzing ROI Journal",
        completed=0,
        total=total,
        current_symbol="",
        current_exchange="",
    )
    for offset, (_, source_row) in enumerate(prepared.iterrows(), start=1):
        signal_id = offset
        row, distance = _analyze_signal_row(
            storage,
            source_row,
            signal_id=signal_id,
            latest_stock_date=latest_cutoff,
            lookback_months=max(int(lookback_months), 1),
            pivot_order=max(int(pivot_order), 1),
            pre_signal_sessions=max(int(pre_signal_sessions), 1),
            recovery_threshold_pct=max(float(recovery_threshold_pct), 0.0),
        )
        rows.append(row)
        if not distance.empty:
            history_frames.append(distance)
        _emit_progress(
            progress_callback,
            phase="Analyzing ROI Journal",
            completed=offset,
            total=total,
            current_symbol=str(row.get("symbol", "")),
            current_exchange=str(row.get("exchange", "")),
        )

    signal_rows = pd.DataFrame(rows)
    if not signal_rows.empty:
        signal_rows = _add_research_ranking(signal_rows, recovery_threshold_pct)
        status_rank = signal_rows["data_status"].map({"READY": 0}).fillna(1).astype(int)
        score_sort = pd.to_numeric(signal_rows.get("research_score"), errors="coerce").fillna(-1.0)
        raw_score_sort = pd.to_numeric(signal_rows.get("raw_research_score"), errors="coerce").fillna(-1.0)
        signal_rows = (
            signal_rows.assign(_status_rank=status_rank, _score_sort=score_sort, _raw_score_sort=raw_score_sort)
            .sort_values(
                ["_status_rank", "_score_sort", "_raw_score_sort", "date_identifier", "symbol", "signal_id"],
                ascending=[True, False, False, False, True, True],
                na_position="last",
            )
            .drop(columns=["_status_rank", "_score_sort", "_raw_score_sort"])
            .reset_index(drop=True)
        )
        ready_rank = signal_rows["data_status"].astype(str).str.upper().eq("READY") & pd.to_numeric(
            signal_rows.get("research_score"),
            errors="coerce",
        ).notna()
        signal_rows["candidate_rank"] = pd.NA
        if ready_rank.any():
            signal_rows.loc[ready_rank, "candidate_rank"] = range(1, int(ready_rank.sum()) + 1)
    distance_history = pd.concat(history_frames, ignore_index=True) if history_frames else _empty_distance_history()
    ready = signal_rows["data_status"].eq("READY") if not signal_rows.empty and "data_status" in signal_rows.columns else pd.Series(dtype=bool)
    recovery_pass = _bool_series(signal_rows, "ll_recovery_10pct_pass")
    dma_recovering = _bool_series(signal_rows, "recovering_toward_any_dma")
    all_dma_reducing = _bool_series(signal_rows, "all_dma_distance_reducing")
    long_trend_aligned = _bool_series(signal_rows, "long_trend_aligned")
    dma_trend = signal_rows.get("dma_trend_20d_label", pd.Series("", index=signal_rows.index)).fillna("").astype(str).str.upper()
    long_bias = signal_rows.get("next_day_long_bias", pd.Series("", index=signal_rows.index)).fillna("").astype(str).str.upper()
    reclaim_pass = _bool_series(signal_rows, "dma75_reclaim_signal_window_pass")
    reclaim_return = pd.to_numeric(signal_rows.get("dma75_return_pct", pd.Series(dtype="float64")), errors="coerce")
    latest_dates = pd.to_datetime(signal_rows.get("latest_return_date", pd.Series(dtype="object")), errors="coerce")
    summary = {
        "rows_uploaded": int(len(prepared)),
        "rows_analyzed": int(len(signal_rows)),
        "ready_rows": int(ready.sum()) if len(ready) else 0,
        "unique_stocks": int(signal_rows["symbol"].nunique()) if not signal_rows.empty and "symbol" in signal_rows.columns else 0,
        "recovering_toward_dma_rows": int(dma_recovering.sum()) if len(dma_recovering) else 0,
        "all_dma_reducing_rows": int((ready & all_dma_reducing).sum()) if len(ready) else 0,
        "long_trend_aligned_rows": int((ready & long_trend_aligned).sum()) if len(ready) else 0,
        "next_day_long_candidate_rows": int((ready & long_bias.str.startswith("LONG CANDIDATE")).sum()) if len(ready) else 0,
        "next_day_long_caution_rows": int((ready & long_bias.str.startswith("CAUTION")).sum()) if len(ready) else 0,
        "positive_dma_trend_rows": int((ready & dma_trend.isin({"BULLISH", "IMPROVING"})).sum()) if len(ready) else 0,
        "bearish_dma_trend_rows": int((ready & dma_trend.isin({"BEARISH", "WEAKENING"})).sum()) if len(ready) else 0,
        "ll_recovery_10pct_rows": int((ready & recovery_pass).sum()) if len(ready) else 0,
        "dma75_reclaim_signal_window_rows": int((ready & reclaim_pass).sum()) if len(ready) else 0,
        "dma75_reclaim_entry_return_rows": int((ready & reclaim_return.notna()).sum()) if len(ready) else 0,
        "dma75_reclaim_positive_return_rows": int((ready & reclaim_return.gt(0.0)).sum()) if len(ready) else 0,
        "dma75_reclaim_avg_return_pct": float(reclaim_return[ready & reclaim_return.notna()].mean()) if len(ready) and (ready & reclaim_return.notna()).any() else np.nan,
        "dma75_reclaim_median_return_pct": float(reclaim_return[ready & reclaim_return.notna()].median()) if len(ready) and (ready & reclaim_return.notna()).any() else np.nan,
        "ranked_candidate_rows": int(
            (ready & (pd.to_numeric(signal_rows.get("research_score", pd.Series(dtype="float64")), errors="coerce") > 0)).sum()
        ) if len(ready) else 0,
        "latest_stock_date": latest_dates.max().strftime("%Y-%m-%d") if latest_dates.notna().any() else "",
        "requested_latest_stock_date": "" if latest_cutoff is None else latest_cutoff.strftime("%Y-%m-%d"),
        "lookback_months": max(int(lookback_months), 1),
        "pivot_order": max(int(pivot_order), 1),
        "pre_signal_sessions": max(int(pre_signal_sessions), 1),
        "recovery_threshold_pct": max(float(recovery_threshold_pct), 0.0),
        "dma_windows": ",".join(str(value) for value in DEFAULT_DMA_WINDOWS),
        "dma_trend_sessions": DEFAULT_DMA_TREND_SESSIONS,
        "dma75_reclaim_sessions": DEFAULT_75DMA_RECLAIM_SESSIONS,
        "dma75_reclaim_pre_signal_sessions": DEFAULT_75DMA_RECLAIM_PRE_SIGNAL_SESSIONS,
        "dma75_reclaim_post_signal_sessions": DEFAULT_75DMA_RECLAIM_POST_SIGNAL_SESSIONS,
    }
    return RoiJournalDmaResult(
        summary=summary,
        signal_rows=signal_rows,
        distance_history=distance_history,
    )


def save_roi_journal_dma_outputs(result: RoiJournalDmaResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(_json_safe(result.summary), indent=2), encoding="utf-8")
    result.signal_rows.to_csv(output_dir / "signal_rows.csv", index=False)
    result.distance_history.to_csv(output_dir / "distance_history.csv", index=False)


def load_roi_journal_dma_outputs(output_dir: Path) -> RoiJournalDmaResult:
    summary_path = output_dir / "summary.json"
    summary: dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}
    return RoiJournalDmaResult(
        summary=summary,
        signal_rows=_read_csv(output_dir / "signal_rows.csv"),
        distance_history=_read_csv(output_dir / "distance_history.csv"),
    )


def _analyze_signal_row(
    storage: Storage,
    source_row: pd.Series,
    *,
    signal_id: int,
    latest_stock_date: pd.Timestamp | None,
    lookback_months: int,
    pivot_order: int,
    pre_signal_sessions: int,
    recovery_threshold_pct: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    exchange, symbol = _parse_stock_reference(source_row.get("Stocks"))
    base = _base_result_row(source_row, signal_id=signal_id, exchange=exchange, symbol=symbol)
    date_identifier = _coerce_date(source_row.get("Date_Identifier"))
    if date_identifier is None:
        base["data_status"] = "INVALID_DATE"
        base["reason"] = "Date_Identifier could not be parsed."
        return base, _empty_distance_history()
    base["date_identifier"] = date_identifier.strftime("%Y-%m-%d")
    if not symbol:
        base["data_status"] = "INVALID_SYMBOL"
        base["reason"] = "Stocks value is blank or invalid."
        return base, _empty_distance_history()

    daily = _prepare_daily(storage.load_candles(exchange, symbol, "1D"))
    if daily.empty and exchange == "NSE":
        bse_daily = _prepare_daily(storage.load_candles("BSE", symbol, "1D"))
        if not bse_daily.empty:
            exchange = "BSE"
            base["exchange"] = exchange
            daily = bse_daily
    if daily.empty:
        base["data_status"] = "NO_DATA"
        base["reason"] = f"No local daily candles found for {exchange}:{symbol}."
        return base, _empty_distance_history()

    enriched = _add_dma_distances(daily)
    setup_candidates = enriched[enriched["date"].dt.normalize() <= date_identifier].copy()
    if setup_candidates.empty:
        base["data_status"] = "NO_SETUP_DATE"
        base["reason"] = "No candle exists on or before Date_Identifier."
        return base, _empty_distance_history()

    setup_position = int(setup_candidates.index[-1])
    setup = enriched.loc[setup_position]
    setup_date = pd.Timestamp(setup["date"]).normalize()
    lookback_start = setup_date - pd.DateOffset(months=lookback_months)
    setup_window = enriched[
        (enriched["date"].dt.normalize() >= lookback_start)
        & (enriched["date"].dt.normalize() <= setup_date)
    ].copy()
    if len(setup_window) < 30:
        base["data_status"] = "INSUFFICIENT_HISTORY"
        base["reason"] = "Fewer than 30 candles exist in the six-month setup window."
        return base, _empty_distance_history()

    signature = run_stock_signature_study(
        enriched,
        exchange=exchange,
        symbol=symbol,
        pivot_order=pivot_order,
        dma_windows=DEFAULT_DMA_WINDOWS,
        lookback_months=lookback_months,
        as_of_date=setup_date,
    )
    pivots = signature.pivots.copy()
    last_pivot = pivots.iloc[-1] if not pivots.empty else pd.Series(dtype="object")
    high_pivots = pivots[pivots["type"].eq("HIGH")].copy() if not pivots.empty else pd.DataFrame()
    low_pivots = pivots[pivots["type"].eq("LOW")].copy() if not pivots.empty else pd.DataFrame()
    last_high = high_pivots.iloc[-1] if not high_pivots.empty else pd.Series(dtype="object")
    last_hh = high_pivots[high_pivots["structure"].eq("HH")].iloc[-1] if not high_pivots.empty and high_pivots["structure"].eq("HH").any() else pd.Series(dtype="object")
    last_low = low_pivots.iloc[-1] if not low_pivots.empty else pd.Series(dtype="object")
    last_hl = low_pivots[low_pivots["structure"].eq("HL")].iloc[-1] if not low_pivots.empty and low_pivots["structure"].eq("HL").any() else pd.Series(dtype="object")
    last_ll = low_pivots[low_pivots["structure"].eq("LL")].iloc[-1] if not low_pivots.empty and low_pivots["structure"].eq("LL").any() else pd.Series(dtype="object")

    observation_position = max(0, setup_position - pre_signal_sessions)
    observation = enriched.loc[observation_position]
    distance_75 = _finite_float(setup.get("distance_75dma_pct"))
    distance_100 = _finite_float(setup.get("distance_100dma_pct"))
    obs_distance_75 = _finite_float(observation.get("distance_75dma_pct"))
    obs_distance_100 = _finite_float(observation.get("distance_100dma_pct"))
    reduction_75 = _distance_reduction(obs_distance_75, distance_75)
    reduction_100 = _distance_reduction(obs_distance_100, distance_100)
    recovering_75 = _recovering_toward_dma(obs_distance_75, distance_75)
    recovering_100 = _recovering_toward_dma(obs_distance_100, distance_100)
    trend_reference_position = max(0, setup_position - DEFAULT_DMA_TREND_SESSIONS)
    trend_reference = enriched.loc[trend_reference_position]
    dma_75_trend = _dma_trend_pct(trend_reference.get("dma_75"), setup.get("dma_75"))
    dma_100_trend = _dma_trend_pct(trend_reference.get("dma_100"), setup.get("dma_100"))
    dma_stack = _dma_stack_label(setup.get("dma_75"), setup.get("dma_100"))
    dma_trend_label = _dma_trend_label(dma_75_trend, dma_100_trend, dma_stack)
    all_dma_trend_positive = _all_dma_trend_positive(dma_75_trend, dma_100_trend)
    all_dma_trend_non_negative = _all_dma_trend_non_negative(dma_75_trend, dma_100_trend)
    all_dma_distance_reducing = _all_dma_distance_reducing(reduction_75, reduction_100)
    setup_close = float(setup["close"])
    price_above_75dma = _price_above_dma(setup_close, setup.get("dma_75"))
    price_above_100dma = _price_above_dma(setup_close, setup.get("dma_100"))
    price_above_all_dma = bool(price_above_75dma and price_above_100dma)
    long_trend_aligned = bool(all_dma_trend_positive and dma_stack in {"75DMA above 100DMA", "75DMA equal 100DMA"})
    trend_around_date_identifier = _trend_around_date_identifier(
        dma_trend_label=dma_trend_label,
        dma_stack=dma_stack,
        price_above_all_dma=price_above_all_dma,
        all_dma_trend_positive=all_dma_trend_positive,
        all_dma_trend_non_negative=all_dma_trend_non_negative,
    )
    distance_to_last_hh = _pivot_distance_pct(setup_close, last_hh)
    distance_to_last_hl = _pivot_distance_pct(setup_close, last_hl)
    closest_hh_hl, closest_hh_hl_gap = _closest_hh_hl_reference(distance_to_last_hh, distance_to_last_hl)
    pullback = _pullback_summary(pivots, setup_close, last_high)
    next_day_long_bias = _next_day_long_bias(
        long_trend_aligned=long_trend_aligned,
        all_dma_trend_non_negative=all_dma_trend_non_negative,
        dma_trend_label=dma_trend_label,
        price_above_all_dma=price_above_all_dma,
        all_dma_distance_reducing=all_dma_distance_reducing,
        last_low_structure="" if last_low.empty else str(last_low.get("structure", "")),
        closest_hh_hl=closest_hh_hl,
        distance_to_last_hl=distance_to_last_hl,
        current_pullback_vs_avg=pullback["current_pullback_vs_avg"],
    )

    ll_recovery_pct = np.nan
    ll_age_sessions = np.nan
    if not last_ll.empty:
        ll_idx = int(last_ll.get("row_idx"))
        ll_price = _finite_float(last_ll.get("price"))
        if ll_price is not None and ll_price > 0:
            ll_recovery_pct = (float(setup["close"]) / ll_price - 1.0) * 100.0
            ll_age_sessions = int(len(signature.daily) - 1 - ll_idx)

    entry, exit_row = _entry_exit_rows(enriched, setup_date=setup_date, latest_stock_date=latest_stock_date)
    forward_return = np.nan
    if entry is not None and exit_row is not None:
        entry_close = _finite_float(entry.get("close"))
        exit_close = _finite_float(exit_row.get("close"))
        if entry_close is not None and exit_close is not None and entry_close > 0:
            forward_return = (exit_close / entry_close - 1.0) * 100.0
    dma75_reclaim = _dma75_reclaim_entry_summary(
        enriched,
        setup_date=setup_date,
        latest_stock_date=latest_stock_date,
    )

    distance_history = _distance_history_frame(
        enriched,
        signal_id=signal_id,
        exchange=exchange,
        symbol=symbol,
        start_date=lookback_start,
        setup_date=setup_date,
        latest_stock_date=latest_stock_date,
    )
    base.update(
        {
            "data_status": "READY",
            "reason": "",
            "setup_date": setup_date.strftime("%Y-%m-%d"),
            "setup_close": setup_close,
            "history_start": pd.Timestamp(setup_window["date"].min()).strftime("%Y-%m-%d"),
            "history_end": pd.Timestamp(setup_window["date"].max()).strftime("%Y-%m-%d"),
            "dma_75": _finite_float(setup.get("dma_75")),
            "dma_100": _finite_float(setup.get("dma_100")),
            "distance_75dma_pct": distance_75,
            "distance_100dma_pct": distance_100,
            "dma_75_trend_20d_pct": dma_75_trend,
            "dma_100_trend_20d_pct": dma_100_trend,
            "dma_stack": dma_stack,
            "dma_trend_20d_label": dma_trend_label,
            "all_dma_trend_positive": all_dma_trend_positive,
            "all_dma_trend_non_negative": all_dma_trend_non_negative,
            "all_dma_distance_reducing": all_dma_distance_reducing,
            "price_above_75dma": price_above_75dma,
            "price_above_100dma": price_above_100dma,
            "price_above_all_dma": price_above_all_dma,
            "long_trend_aligned": long_trend_aligned,
            "trend_around_date_identifier": trend_around_date_identifier,
            "next_day_long_bias": next_day_long_bias,
            "observation_start_date": pd.Timestamp(observation["date"]).strftime("%Y-%m-%d"),
            "observation_distance_75dma_pct": obs_distance_75,
            "observation_distance_100dma_pct": obs_distance_100,
            "distance_reduction_75dma_pct_points": reduction_75,
            "distance_reduction_100dma_pct_points": reduction_100,
            "recovering_toward_75dma": recovering_75,
            "recovering_toward_100dma": recovering_100,
            "recovering_toward_any_dma": bool(recovering_75 or recovering_100),
            "last_pivot_type": "" if last_pivot.empty else str(last_pivot.get("type", "")),
            "last_pivot_structure": "" if last_pivot.empty else str(last_pivot.get("structure", "")),
            "last_pivot_date": "" if last_pivot.empty else _date_text(last_pivot.get("date")),
            "last_high_pivot_structure": "" if last_high.empty else str(last_high.get("structure", "")),
            "last_high_pivot_date": "" if last_high.empty else _date_text(last_high.get("date")),
            "last_high_pivot_price": np.nan if last_high.empty else _finite_float(last_high.get("price")),
            "last_hh_pivot_date": "" if last_hh.empty else _date_text(last_hh.get("date")),
            "last_hh_pivot_price": np.nan if last_hh.empty else _finite_float(last_hh.get("price")),
            "distance_to_last_hh_pct": distance_to_last_hh,
            "last_hl_pivot_date": "" if last_hl.empty else _date_text(last_hl.get("date")),
            "last_hl_pivot_price": np.nan if last_hl.empty else _finite_float(last_hl.get("price")),
            "distance_to_last_hl_pct": distance_to_last_hl,
            "closest_hh_hl_reference": closest_hh_hl,
            "closest_hh_hl_gap_pct": closest_hh_hl_gap,
            "last_low_pivot_structure": "" if last_low.empty else str(last_low.get("structure", "")),
            "last_low_pivot_date": "" if last_low.empty else _date_text(last_low.get("date")),
            "last_ll_pivot_date": "" if last_ll.empty else _date_text(last_ll.get("date")),
            "last_ll_pivot_price": np.nan if last_ll.empty else _finite_float(last_ll.get("price")),
            "ll_recovery_pct": ll_recovery_pct,
            "ll_pivot_age_sessions": ll_age_sessions,
            "ll_within_pre_signal_window": bool(_is_finite(ll_age_sessions) and float(ll_age_sessions) <= pre_signal_sessions),
            "ll_recovery_10pct_pass": bool(_is_finite(ll_recovery_pct) and float(ll_recovery_pct) >= recovery_threshold_pct),
            "swing_count_6m": int(len(pivots)),
            "low_high_swings_6m": _low_high_swing_count(pivots),
            "pullback_events_6m": pullback["pullback_events_6m"],
            "average_pullback_pct": pullback["average_pullback_pct"],
            "median_pullback_pct": pullback["median_pullback_pct"],
            "current_pullback_from_last_high_pct": pullback["current_pullback_from_last_high_pct"],
            "current_pullback_abs_pct": pullback["current_pullback_abs_pct"],
            "current_pullback_vs_avg": pullback["current_pullback_vs_avg"],
            "pullback_context": pullback["pullback_context"],
            "next_trade_date": "" if entry is None else _date_text(entry.get("date")),
            "next_trade_close": np.nan if entry is None else _finite_float(entry.get("close")),
            "latest_return_date": "" if exit_row is None else _date_text(exit_row.get("date")),
            "latest_return_close": np.nan if exit_row is None else _finite_float(exit_row.get("close")),
            "forward_return_pct": forward_return,
            **dma75_reclaim,
        }
    )
    return base, distance_history


def _add_research_ranking(signal_rows: pd.DataFrame, recovery_threshold_pct: float) -> pd.DataFrame:
    frame = signal_rows.copy()
    ready = frame.get("data_status", pd.Series("", index=frame.index)).astype(str).str.upper().eq("READY")
    recovering = _bool_series(frame, "recovering_toward_any_dma")
    all_dma_reducing = _bool_series(frame, "all_dma_distance_reducing")
    all_dma_positive = _bool_series(frame, "all_dma_trend_positive")
    all_dma_non_negative = _bool_series(frame, "all_dma_trend_non_negative")
    long_trend_aligned = _bool_series(frame, "long_trend_aligned")
    price_above_all_dma = _bool_series(frame, "price_above_all_dma")
    ll_recovery_pass = _bool_series(frame, "ll_recovery_10pct_pass")
    last_low_structure = frame.get("last_low_pivot_structure", pd.Series("", index=frame.index)).fillna("").astype(str).str.upper()
    last_high_structure = frame.get("last_high_pivot_structure", pd.Series("", index=frame.index)).fillna("").astype(str).str.upper()
    closest_hh_hl = frame.get("closest_hh_hl_reference", pd.Series("", index=frame.index)).fillna("").astype(str).str.upper()
    reclaim_pass = _bool_series(frame, "dma75_reclaim_signal_window_pass")
    reclaim_offset = _numeric_series(frame, "dma75_reclaim_offset_sessions")

    distance_75 = _numeric_series(frame, "distance_75dma_pct")
    distance_100 = _numeric_series(frame, "distance_100dma_pct")
    reduction_75 = _numeric_series(frame, "distance_reduction_75dma_pct_points")
    reduction_100 = _numeric_series(frame, "distance_reduction_100dma_pct_points")
    dma_trend_label = frame.get("dma_trend_20d_label", pd.Series("", index=frame.index)).fillna("").astype(str).str.upper()
    ll_recovery = _numeric_series(frame, "ll_recovery_pct")
    swings = _numeric_series(frame, "swing_count_6m")
    distance_to_hh = _numeric_series(frame, "distance_to_last_hh_pct")
    distance_to_hl = _numeric_series(frame, "distance_to_last_hl_pct")
    pullback_ratio = _numeric_series(frame, "current_pullback_vs_avg")
    below_last_hl = distance_to_hl.notna() & distance_to_hl.lt(0.0)

    best_reduction = pd.concat([reduction_75, reduction_100], axis=1).max(axis=1)
    nearest_dma_gap = pd.concat([distance_75.abs(), distance_100.abs()], axis=1).min(axis=1)

    trend_score = pd.Series(
        np.select(
            [
                long_trend_aligned,
                all_dma_positive & dma_trend_label.eq("IMPROVING"),
                dma_trend_label.eq("BULLISH"),
                all_dma_non_negative & dma_trend_label.eq("MIXED"),
                dma_trend_label.eq("MIXED"),
                dma_trend_label.eq("WEAKENING"),
                dma_trend_label.eq("BEARISH"),
            ],
            [30.0, 24.0, 26.0, 16.0, 12.0, 4.0, 0.0],
            default=0.0,
        ),
        index=frame.index,
        dtype="float64",
    )
    dma_reduction_score = best_reduction.clip(lower=0.0, upper=15.0).fillna(0.0) / 15.0 * 8.0
    all_dma_reducing_score = all_dma_reducing.astype(float) * 7.0
    dma_proximity_score = (8.0 - nearest_dma_gap.clip(lower=0.0, upper=8.0)).fillna(0.0) / 8.0 * 5.0
    dma_reclaim_score = pd.Series(
        np.select(
            [
                reclaim_pass & reclaim_offset.eq(0.0),
                reclaim_pass & reclaim_offset.eq(-1.0),
                reclaim_pass & reclaim_offset.between(1.0, 2.0, inclusive="both"),
                reclaim_pass & reclaim_offset.between(3.0, float(DEFAULT_75DMA_RECLAIM_POST_SIGNAL_SESSIONS), inclusive="both"),
            ],
            [6.0, 5.0, 5.0, 3.0],
            default=0.0,
        ),
        index=frame.index,
        dtype="float64",
    )
    dma_entry_score = dma_reduction_score + all_dma_reducing_score + dma_proximity_score + dma_reclaim_score

    threshold = max(float(recovery_threshold_pct), 1.0)
    capped_ll_recovery = ll_recovery.clip(lower=0.0, upper=max(threshold * 2.0, 20.0)).fillna(0.0)
    ll_recovery_score = capped_ll_recovery / max(threshold * 2.0, 20.0) * 10.0
    ll_recovery_score = ll_recovery_score.where(ll_recovery_pass, ll_recovery_score.clip(upper=5.0))

    low_structure_score = pd.Series(
        np.select(
            [last_low_structure.eq("LL"), last_low_structure.eq("HL")],
            [3.0, 8.0],
            default=0.0,
        ),
        index=frame.index,
        dtype="float64",
    )
    high_structure_score = pd.Series(np.where(last_high_structure.eq("HH"), 4.0, 0.0), index=frame.index, dtype="float64")
    hl_proximity_score = pd.Series(
        np.select(
            [
                distance_to_hl.ge(0.0) & distance_to_hl.le(5.0),
                distance_to_hl.ge(0.0) & distance_to_hl.le(10.0),
                distance_to_hl.ge(0.0),
                distance_to_hl.lt(0.0),
            ],
            [4.0, 3.0, 1.0, 0.0],
            default=0.0,
        ),
        index=frame.index,
        dtype="float64",
    )
    hh_proximity_score = pd.Series(
        np.select(
            [
                distance_to_hh.ge(-5.0) & distance_to_hh.le(2.0),
                distance_to_hh.gt(2.0),
                distance_to_hh.ge(-12.0) & distance_to_hh.lt(-5.0),
            ],
            [4.0, 3.0, 2.0],
            default=0.0,
        ),
        index=frame.index,
        dtype="float64",
    )
    pivot_context_score = low_structure_score + high_structure_score + hl_proximity_score + hh_proximity_score
    pullback_score = pd.Series(
        np.select(
            [
                pullback_ratio.ge(0.50) & pullback_ratio.le(1.25),
                pullback_ratio.ge(0.25) & pullback_ratio.le(1.75),
                pullback_ratio.gt(1.75),
                pullback_ratio.gt(0.0) & pullback_ratio.lt(0.25),
            ],
            [15.0, 10.0, 4.0, 6.0],
            default=0.0,
        ),
        index=frame.index,
        dtype="float64",
    )
    pullback_score = pullback_score.mask(closest_hh_hl.eq("HL") & pullback_score.gt(0.0), (pullback_score + 2.0).clip(upper=15.0))
    pullback_score = pullback_score.mask(price_above_all_dma & pullback_score.gt(0.0), (pullback_score + 1.0).clip(upper=15.0))
    swing_score = pd.Series(0.0, index=frame.index, dtype="float64")
    swing_score = swing_score.mask(swings < 4.0, swings.fillna(0.0).clip(lower=0.0) / 4.0 * 5.0)
    swing_score = swing_score.mask(swings.between(4.0, 20.0, inclusive="both"), 5.0)
    swing_score = swing_score.mask(swings > 20.0, (5.0 - (swings - 20.0) * 0.4).clip(lower=0.0))
    support_break_penalty = pd.Series(np.where(below_last_hl, 8.0, 0.0), index=frame.index, dtype="float64")

    raw_score = (
        trend_score
        + dma_entry_score
        + pivot_context_score
        + pullback_score
        + ll_recovery_score
        + swing_score
        - support_break_penalty
    )
    trend_score_cap = pd.Series(
        np.select(
            [
                dma_trend_label.eq("BULLISH"),
                dma_trend_label.eq("IMPROVING"),
                dma_trend_label.eq("MIXED"),
                dma_trend_label.eq("WEAKENING"),
                dma_trend_label.eq("BEARISH"),
            ],
            [100.0, 90.0, 78.0, 62.0, 55.0],
            default=60.0,
        ),
        index=frame.index,
        dtype="float64",
    )
    trend_score_cap = trend_score_cap.mask(long_trend_aligned, 100.0)
    trend_score_cap = trend_score_cap.mask(below_last_hl & trend_score_cap.gt(82.0), 82.0)
    score = raw_score.mask(raw_score > trend_score_cap, trend_score_cap).where(ready, np.nan)

    bearish_dma = dma_trend_label.isin({"BEARISH", "WEAKENING"})
    setup_case = pd.Series("Not ready", index=frame.index, dtype="object")
    setup_case = setup_case.mask(ready, "Watch only")
    setup_case = setup_case.mask(ready & long_trend_aligned & all_dma_reducing, "Long trend pullback toward DMA")
    setup_case = setup_case.mask(ready & long_trend_aligned & below_last_hl, "Long trend but below HL")
    setup_case = setup_case.mask(ready & long_trend_aligned & last_high_structure.eq("HH") & distance_to_hh.between(-5.0, 2.0, inclusive="both"), "Long trend continuation near HH")
    setup_case = setup_case.mask(ready & long_trend_aligned & last_low_structure.eq("HL") & closest_hh_hl.eq("HL") & ~below_last_hl, "Long trend pullback near HL")
    setup_case = setup_case.mask(ready & all_dma_positive & recovering, "Early long recovery toward DMA")
    setup_case = setup_case.mask(ready & dma_trend_label.eq("MIXED"), "Mixed trend watch")
    setup_case = setup_case.mask(ready & bearish_dma & recovering, "Countertrend recovery toward falling DMA")
    setup_case = setup_case.mask(ready & bearish_dma & ~recovering, "Countertrend watch only")

    frame["best_dma_reduction_pct_points"] = best_reduction
    frame["nearest_dma_distance_abs_pct"] = nearest_dma_gap
    frame["research_setup_case"] = setup_case
    frame["raw_research_score"] = raw_score.where(ready, np.nan).round(2)
    frame["dma_trend_score_cap"] = trend_score_cap.where(ready, np.nan)
    frame["research_score"] = score.round(2)
    frame["candidate_rank"] = pd.NA
    ranked = score.dropna().sort_values(ascending=False)
    if not ranked.empty:
        frame.loc[ranked.index, "candidate_rank"] = range(1, len(ranked) + 1)
    frame["ranking_reason"] = [
        _ranking_reason(row)
        for _, row in frame.iterrows()
    ]
    return frame


def _ranking_reason(row: pd.Series) -> str:
    if str(row.get("data_status", "")).upper() != "READY":
        return str(row.get("reason", "") or "")
    parts = [
        f"next-day long bias {row.get('next_day_long_bias', '') or 'NA'}",
        f"trend at signal {row.get('trend_around_date_identifier', '') or 'NA'}",
        f"best DMA improvement {_format_points(row.get('best_dma_reduction_pct_points'))}",
        f"nearest DMA gap {_format_points(row.get('nearest_dma_distance_abs_pct'))}",
        _format_dma_trend_reason(row),
    ]
    reclaim_pass = _truthy_value(row.get("dma75_reclaim_signal_window_pass"))
    reclaim_date = str(row.get("dma75_reclaim_date", "") or "").strip()
    reclaim_offset = _finite_float(row.get("dma75_reclaim_offset_sessions"))
    if reclaim_pass:
        offset_text = "NA" if reclaim_offset is None else f"{int(reclaim_offset):+d} sessions"
        parts.append(f"75DMA reclaim {reclaim_date or 'NA'} ({offset_text})")
    else:
        parts.append("75DMA reclaim in signal window no")
    all_dma_reducing = _truthy_value(row.get("all_dma_distance_reducing"))
    all_dma_positive = _truthy_value(row.get("all_dma_trend_positive"))
    parts.append(f"both DMAs reducing {'yes' if all_dma_reducing else 'no'}")
    parts.append(f"both DMA trends positive {'yes' if all_dma_positive else 'no'}")
    hh_distance = _finite_float(row.get("distance_to_last_hh_pct"))
    hl_distance = _finite_float(row.get("distance_to_last_hl_pct"))
    closest = str(row.get("closest_hh_hl_reference", "") or "").strip()
    if closest:
        hh_text = "NA" if hh_distance is None else f"{hh_distance:.1f}%"
        hl_text = "NA" if hl_distance is None else f"{hl_distance:.1f}%"
        parts.append(f"closer to {closest} (HH {hh_text}, HL {hl_text})")
    if hl_distance is not None and hl_distance < 0.0:
        parts.append("below last HL penalty")
    average_pullback = _finite_float(row.get("average_pullback_pct"))
    current_pullback = _finite_float(row.get("current_pullback_abs_pct"))
    pullback_ratio = _finite_float(row.get("current_pullback_vs_avg"))
    if average_pullback is not None and current_pullback is not None:
        ratio_text = "NA" if pullback_ratio is None else f"{pullback_ratio:.2f}x"
        parts.append(f"pullback {current_pullback:.1f}% vs avg {average_pullback:.1f}% ({ratio_text})")
    structure = str(row.get("last_low_pivot_structure", "") or "").strip().upper()
    if structure:
        parts.append(f"last low {structure}")
    recovery = _finite_float(row.get("ll_recovery_pct"))
    if recovery is not None:
        parts.append(f"LL recovery {recovery:.1f}%")
    raw_score = _finite_float(row.get("raw_research_score"))
    trend_cap = _finite_float(row.get("dma_trend_score_cap"))
    if raw_score is not None and trend_cap is not None and raw_score > trend_cap:
        parts.append(f"trend cap {trend_cap:.0f}")
    swings = _finite_float(row.get("swing_count_6m"))
    if swings is not None:
        parts.append(f"{int(round(swings))} swings")
    return "; ".join(parts)


def _format_points(value: Any) -> str:
    numeric = _finite_float(value)
    return "NA" if numeric is None else f"{numeric:.1f} pts"


def _format_dma_trend_reason(row: pd.Series) -> str:
    label = str(row.get("dma_trend_20d_label", "") or "Unknown").strip() or "Unknown"
    trend_75 = _finite_float(row.get("dma_75_trend_20d_pct"))
    trend_100 = _finite_float(row.get("dma_100_trend_20d_pct"))
    trend_75_text = "NA" if trend_75 is None else f"75DMA {trend_75:.1f}%"
    trend_100_text = "NA" if trend_100 is None else f"100DMA {trend_100:.1f}%"
    return f"DMA trend {label} ({trend_75_text}, {trend_100_text})"


def _prepare_journal_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("The uploaded ROI Journal sheet is empty.")
    working = frame.copy()
    rename: dict[str, str] = {}
    expected = {_column_key(column): column for column in REQUIRED_COLUMNS}
    expected.update(COLUMN_ALIASES)
    for column in working.columns:
        key = _column_key(column)
        if key in expected:
            rename[column] = expected[key]
    working = working.rename(columns=rename)
    missing = [column for column in ("Date_Identifier", "Stocks") if column not in working.columns]
    if missing:
        raise ValueError(f"ROI Journal must include these columns: {', '.join(missing)}.")
    for column in REQUIRED_COLUMNS:
        if column not in working.columns:
            working[column] = ""
    working = working[list(REQUIRED_COLUMNS)].copy()
    working["Stocks"] = working["Stocks"].fillna("").astype(str).str.strip()
    working = working[working["Stocks"].ne("")].reset_index(drop=True)
    if working.empty:
        raise ValueError("No stock symbols were found in the ROI Journal sheet.")
    return working


def _column_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _parse_stock_reference(value: Any) -> tuple[str, str]:
    text = str(value or "").strip().upper()
    exchange = "NSE"
    if ":" in text:
        prefix, text = text.split(":", 1)
        if prefix.strip().upper() in {"NSE", "BSE"}:
            exchange = prefix.strip().upper()
    if text.endswith(".NS"):
        text = text[:-3]
        exchange = "NSE"
    elif text.endswith(".BO"):
        text = text[:-3]
        exchange = "BSE"
    symbol = re.split(r"[\s,;/|]+", text.strip())[0].strip()
    symbol = symbol.strip("'\"")
    return exchange, symbol


def _prepare_daily(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    frame = daily.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    required = ["date", "open", "high", "low", "close"]
    if any(column not in frame.columns for column in required):
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", format="mixed")
    for column in required[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "volume" not in frame.columns:
        frame["volume"] = np.nan
    else:
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    return (
        frame.dropna(subset=required)
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )


def _add_dma_distances(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy().reset_index(drop=True)
    for window in DEFAULT_DMA_WINDOWS:
        frame[f"dma_{window}"] = frame["close"].rolling(window, min_periods=window).mean()
        frame[f"distance_{window}dma_pct"] = (frame["close"] / frame[f"dma_{window}"] - 1.0) * 100.0
    return frame


def _entry_exit_rows(
    frame: pd.DataFrame,
    *,
    setup_date: pd.Timestamp,
    latest_stock_date: pd.Timestamp | None,
) -> tuple[pd.Series | None, pd.Series | None]:
    eligible = frame.copy()
    if latest_stock_date is not None:
        eligible = eligible[eligible["date"].dt.normalize() <= latest_stock_date].copy()
    after_setup = eligible[eligible["date"].dt.normalize() > setup_date].copy()
    if after_setup.empty:
        return None, None
    return after_setup.iloc[0], after_setup.iloc[-1]


def _dma75_reclaim_entry_summary(
    frame: pd.DataFrame,
    *,
    setup_date: pd.Timestamp,
    latest_stock_date: pd.Timestamp | None,
) -> dict[str, Any]:
    result = {
        "dma75_reclaim_signal_window_pass": False,
        "dma75_reclaim_timing": "",
        "dma75_break_below_date": "",
        "dma75_reclaim_date": "",
        "dma75_sessions_to_reclaim_after_break": np.nan,
        "dma75_reclaim_offset_sessions": np.nan,
        "dma75_entry_date": "",
        "dma75_entry_close": np.nan,
        "dma75_exit_date": "",
        "dma75_exit_close": np.nan,
        "dma75_hold_calendar_days": np.nan,
        "dma75_hold_trading_sessions": np.nan,
        "dma75_used_requested_as_of_close": False,
        "dma75_return_pct": np.nan,
        "dma75_break_distance_pct": np.nan,
        "dma75_reclaim_distance_pct": np.nan,
        "dma75_entry_status": "No 75DMA reclaim in signal window",
    }
    eligible = frame.copy()
    if latest_stock_date is not None:
        eligible = eligible[eligible["date"].dt.normalize() <= latest_stock_date].copy()
    if eligible.empty:
        result["dma75_entry_status"] = "No candles available before exit date"
        return result
    eligible = eligible.sort_values("date").reset_index(drop=True)
    setup_candidates = eligible[eligible["date"].dt.normalize() <= setup_date].copy()
    if setup_candidates.empty:
        result["dma75_entry_status"] = "No candle on or before Date_Identifier"
        return result

    setup_position = int(setup_candidates.index[-1])
    start_position = max(0, setup_position - DEFAULT_75DMA_RECLAIM_PRE_SIGNAL_SESSIONS)
    end_position = min(len(eligible) - 1, setup_position + DEFAULT_75DMA_RECLAIM_POST_SIGNAL_SESSIONS)
    events = _dma75_reclaim_events(eligible, max_reclaim_sessions=DEFAULT_75DMA_RECLAIM_SESSIONS)
    signal_window_events = [
        event
        for event in events
        if start_position <= int(event["reclaim_position"]) <= end_position
    ]
    if not signal_window_events:
        result["dma75_entry_status"] = (
            f"No 75DMA reclaim from {DEFAULT_75DMA_RECLAIM_PRE_SIGNAL_SESSIONS} session before "
            f"to {DEFAULT_75DMA_RECLAIM_POST_SIGNAL_SESSIONS} sessions after Date_Identifier"
        )
        return result

    future_events = [event for event in signal_window_events if int(event["reclaim_position"]) >= setup_position]
    selected = min(future_events, key=lambda event: int(event["reclaim_position"])) if future_events else max(
        signal_window_events,
        key=lambda event: int(event["reclaim_position"]),
    )
    break_position = int(selected["break_position"])
    reclaim_position = int(selected["reclaim_position"])
    offset_sessions = reclaim_position - setup_position
    break_row = eligible.iloc[break_position]
    reclaim_row = eligible.iloc[reclaim_position]
    entry_position = reclaim_position + 1

    result.update(
        {
            "dma75_reclaim_signal_window_pass": True,
            "dma75_reclaim_timing": _reclaim_timing_label(offset_sessions),
            "dma75_break_below_date": _date_text(break_row.get("date")),
            "dma75_reclaim_date": _date_text(reclaim_row.get("date")),
            "dma75_sessions_to_reclaim_after_break": int(selected["sessions_to_reclaim"]),
            "dma75_reclaim_offset_sessions": int(offset_sessions),
            "dma75_break_distance_pct": _finite_float(break_row.get("distance_75dma_pct")),
            "dma75_reclaim_distance_pct": _finite_float(reclaim_row.get("distance_75dma_pct")),
        }
    )
    if entry_position >= len(eligible):
        result["dma75_entry_status"] = "75DMA reclaim found; next trading entry pending"
        return result

    entry_row = eligible.iloc[entry_position]
    exit_row = eligible.iloc[-1]
    entry_close = _finite_float(entry_row.get("close"))
    exit_close = _finite_float(exit_row.get("close"))
    entry_date = pd.Timestamp(entry_row.get("date")).normalize()
    exit_date = pd.Timestamp(exit_row.get("date")).normalize()
    if entry_close is not None and exit_close is not None and entry_close > 0.0:
        result["dma75_return_pct"] = (exit_close / entry_close - 1.0) * 100.0
        result["dma75_entry_status"] = "Entry available after 75DMA reclaim"
    else:
        result["dma75_entry_status"] = "75DMA reclaim found; entry return unavailable"
    result.update(
        {
            "dma75_entry_date": entry_date.strftime("%Y-%m-%d"),
            "dma75_entry_close": entry_close if entry_close is not None else np.nan,
            "dma75_exit_date": exit_date.strftime("%Y-%m-%d"),
            "dma75_exit_close": exit_close if exit_close is not None else np.nan,
            "dma75_hold_calendar_days": int((exit_date - entry_date).days),
            "dma75_hold_trading_sessions": int(max(len(eligible.iloc[entry_position:]) - 1, 0)),
            "dma75_used_requested_as_of_close": bool(latest_stock_date is not None and exit_date == latest_stock_date),
        }
    )
    return result


def _dma75_reclaim_events(frame: pd.DataFrame, *, max_reclaim_sessions: int) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if len(frame) < 2 or "dma_75" not in frame.columns:
        return events
    close = pd.to_numeric(frame["close"], errors="coerce")
    dma = pd.to_numeric(frame["dma_75"], errors="coerce")
    for position in range(1, len(frame)):
        previous_close = _finite_float(close.iloc[position - 1])
        previous_dma = _finite_float(dma.iloc[position - 1])
        current_close = _finite_float(close.iloc[position])
        current_dma = _finite_float(dma.iloc[position])
        if previous_close is None or previous_dma is None or current_close is None or current_dma is None:
            continue
        crossed_below = previous_close >= previous_dma and current_close < current_dma
        if not crossed_below:
            continue
        last_reclaim_position = min(len(frame) - 1, position + max(int(max_reclaim_sessions), 1))
        for reclaim_position in range(position + 1, last_reclaim_position + 1):
            reclaim_close = _finite_float(close.iloc[reclaim_position])
            reclaim_dma = _finite_float(dma.iloc[reclaim_position])
            if reclaim_close is None or reclaim_dma is None:
                continue
            if reclaim_close > reclaim_dma:
                events.append(
                    {
                        "break_position": int(position),
                        "reclaim_position": int(reclaim_position),
                        "sessions_to_reclaim": int(reclaim_position - position),
                    }
                )
                break
    return events


def _reclaim_timing_label(offset_sessions: int) -> str:
    if offset_sessions == 0:
        return "Reclaim on Date_Identifier"
    if offset_sessions < 0:
        return f"Reclaim {abs(int(offset_sessions))} session before Date_Identifier"
    if offset_sessions == 1:
        return "Reclaim 1 session after Date_Identifier"
    return f"Reclaim {int(offset_sessions)} sessions after Date_Identifier"


def _distance_history_frame(
    frame: pd.DataFrame,
    *,
    signal_id: int,
    exchange: str,
    symbol: str,
    start_date: pd.Timestamp,
    setup_date: pd.Timestamp,
    latest_stock_date: pd.Timestamp | None,
) -> pd.DataFrame:
    eligible = frame.copy()
    if latest_stock_date is not None:
        eligible = eligible[eligible["date"].dt.normalize() <= latest_stock_date].copy()
    if eligible.empty:
        return _empty_distance_history()
    setup_candidates = eligible[eligible["date"].dt.normalize() <= setup_date].copy()
    if setup_candidates.empty:
        return _empty_distance_history()
    setup_position = int(setup_candidates.index[-1])
    decision_end_position = min(
        int(eligible.index[-1]),
        setup_position + DEFAULT_75DMA_RECLAIM_POST_SIGNAL_SESSIONS,
    )
    end_date = pd.Timestamp(eligible.loc[decision_end_position, "date"]).normalize()
    history = frame[
        (frame["date"].dt.normalize() >= start_date)
        & (frame["date"].dt.normalize() <= end_date)
    ].copy()
    if history.empty:
        return _empty_distance_history()
    history["signal_id"] = int(signal_id)
    history["exchange"] = exchange
    history["symbol"] = symbol
    history["date_identifier"] = setup_date.strftime("%Y-%m-%d")
    return history[
        [
            "signal_id",
            "exchange",
            "symbol",
            "date_identifier",
            "date",
            "close",
            "dma_75",
            "dma_100",
            "distance_75dma_pct",
            "distance_100dma_pct",
        ]
    ]


def _base_result_row(source_row: pd.Series, *, signal_id: int, exchange: str, symbol: str) -> dict[str, Any]:
    return {
        "signal_id": int(signal_id),
        "exchange": exchange,
        "symbol": symbol,
        "date_identifier": "",
        "strategy": source_row.get("Strategy", ""),
        "top_5": source_row.get("Top_5", ""),
        "price_asof_input": source_row.get("Price_Asof", ""),
        "price_latest_input": source_row.get("Price_Latest", ""),
        "signal_days_input": source_row.get("Signal_Days", ""),
        "profit_loss_input": source_row.get("Profit_Loss", ""),
        "actual_return_input": source_row.get("Actual_Return", ""),
        "data_status": "NOT_RUN",
        "reason": "",
    }


def _distance_reduction(start_distance: float | None, end_distance: float | None) -> float:
    if start_distance is None or end_distance is None:
        return np.nan
    return abs(start_distance) - abs(end_distance)


def _recovering_toward_dma(start_distance: float | None, end_distance: float | None) -> bool:
    if start_distance is None or end_distance is None:
        return False
    return bool(start_distance < 0.0 and end_distance < 0.0 and end_distance > start_distance and abs(end_distance) < abs(start_distance))


def _price_above_dma(close: float, dma: Any) -> bool:
    dma_value = _finite_float(dma)
    return bool(dma_value is not None and close >= dma_value)


def _all_dma_trend_positive(trend_75: Any, trend_100: Any) -> bool:
    values = [_finite_float(trend_75), _finite_float(trend_100)]
    return bool(all(value is not None and value > DMA_TREND_FLAT_THRESHOLD_PCT for value in values))


def _all_dma_trend_non_negative(trend_75: Any, trend_100: Any) -> bool:
    values = [_finite_float(trend_75), _finite_float(trend_100)]
    return bool(all(value is not None and value >= -DMA_TREND_FLAT_THRESHOLD_PCT for value in values))


def _all_dma_distance_reducing(reduction_75: Any, reduction_100: Any) -> bool:
    values = [_finite_float(reduction_75), _finite_float(reduction_100)]
    return bool(all(value is not None and value > 0.0 for value in values))


def _trend_around_date_identifier(
    *,
    dma_trend_label: str,
    dma_stack: str,
    price_above_all_dma: bool,
    all_dma_trend_positive: bool,
    all_dma_trend_non_negative: bool,
) -> str:
    label = str(dma_trend_label or "Unknown").strip()
    if all_dma_trend_positive and dma_stack == "75DMA above 100DMA":
        return "Bullish long trend" if price_above_all_dma else "Bullish pullback below DMA"
    if all_dma_trend_positive:
        return "Early improving long trend"
    if all_dma_trend_non_negative:
        return f"{label} long watch"
    if label in {"Bearish", "Weakening"}:
        return "Countertrend for long"
    return "Trend unclear"


def _pivot_distance_pct(close: float, pivot: pd.Series) -> float:
    if pivot.empty:
        return np.nan
    pivot_price = _finite_float(pivot.get("price"))
    if pivot_price is None or pivot_price <= 0.0:
        return np.nan
    return (close / pivot_price - 1.0) * 100.0


def _closest_hh_hl_reference(distance_to_hh: Any, distance_to_hl: Any) -> tuple[str, float]:
    distances = {
        "HH": _finite_float(distance_to_hh),
        "HL": _finite_float(distance_to_hl),
    }
    distances = {label: value for label, value in distances.items() if value is not None}
    if not distances:
        return "", np.nan
    label = min(distances, key=lambda key: abs(float(distances[key])))
    return label, abs(float(distances[label]))


def _pullback_summary(pivots: pd.DataFrame, setup_close: float, last_high: pd.Series) -> dict[str, Any]:
    pullbacks: list[float] = []
    if not pivots.empty:
        for index in range(len(pivots) - 1):
            high = pivots.iloc[index]
            low = pivots.iloc[index + 1]
            if not (high.get("type") == "HIGH" and low.get("type") == "LOW"):
                continue
            high_price = _finite_float(high.get("price"))
            low_price = _finite_float(low.get("price"))
            if high_price is None or low_price is None or high_price <= 0.0:
                continue
            pullbacks.append(abs((low_price / high_price - 1.0) * 100.0))

    average_pullback = float(np.mean(pullbacks)) if pullbacks else np.nan
    median_pullback = float(np.median(pullbacks)) if pullbacks else np.nan
    current_pullback = np.nan
    current_pullback_abs = np.nan
    current_pullback_vs_avg = np.nan
    if not last_high.empty:
        last_high_price = _finite_float(last_high.get("price"))
        if last_high_price is not None and last_high_price > 0.0:
            current_pullback = (setup_close / last_high_price - 1.0) * 100.0
            current_pullback_abs = max(0.0, -current_pullback)
            if _is_finite(average_pullback) and float(average_pullback) > 0.0:
                current_pullback_vs_avg = current_pullback_abs / float(average_pullback)

    return {
        "pullback_events_6m": int(len(pullbacks)),
        "average_pullback_pct": average_pullback,
        "median_pullback_pct": median_pullback,
        "current_pullback_from_last_high_pct": current_pullback,
        "current_pullback_abs_pct": current_pullback_abs,
        "current_pullback_vs_avg": current_pullback_vs_avg,
        "pullback_context": _pullback_context(current_pullback, current_pullback_vs_avg),
    }


def _pullback_context(current_pullback: Any, current_pullback_vs_avg: Any) -> str:
    pullback = _finite_float(current_pullback)
    ratio = _finite_float(current_pullback_vs_avg)
    if pullback is None:
        return "Unknown"
    if pullback >= 0.0:
        return "At or above last high"
    if ratio is None:
        return "Pullback present; no average"
    if ratio < 0.25:
        return "Shallow vs average pullback"
    if ratio <= 1.25:
        return "Normal pullback"
    if ratio <= 1.75:
        return "Deep pullback"
    return "Extreme pullback"


def _next_day_long_bias(
    *,
    long_trend_aligned: bool,
    all_dma_trend_non_negative: bool,
    dma_trend_label: str,
    price_above_all_dma: bool,
    all_dma_distance_reducing: bool,
    last_low_structure: str,
    closest_hh_hl: str,
    distance_to_last_hl: Any,
    current_pullback_vs_avg: Any,
) -> str:
    trend_label = str(dma_trend_label or "").strip()
    if trend_label in {"Bearish", "Weakening"}:
        return "Avoid long: falling DMA trend"
    if long_trend_aligned:
        hl_distance = _finite_float(distance_to_last_hl)
        pullback_ratio = _finite_float(current_pullback_vs_avg)
        near_hl = closest_hh_hl == "HL" and hl_distance is not None and hl_distance >= 0.0 and hl_distance <= 10.0
        normal_pullback = pullback_ratio is not None and 0.25 <= pullback_ratio <= 1.75
        if hl_distance is not None and hl_distance < 0.0:
            return "Caution: bullish trend but below HL"
        if str(last_low_structure or "").upper() == "HL" and near_hl and normal_pullback:
            return "Long candidate: trend pullback near HL"
        if all_dma_distance_reducing and normal_pullback:
            return "Long candidate: trend pullback toward DMA"
        if price_above_all_dma:
            return "Long watch: bullish trend"
        return "Long watch: bullish pullback below DMA"
    if all_dma_trend_non_negative:
        return "Watch: early or mixed long trend"
    return "Avoid long: trend not aligned"


def _dma_trend_pct(start_dma: Any, end_dma: Any) -> float:
    start = _finite_float(start_dma)
    end = _finite_float(end_dma)
    if start is None or end is None or start <= 0.0:
        return np.nan
    return (end / start - 1.0) * 100.0


def _dma_stack_label(dma_75: Any, dma_100: Any) -> str:
    dma_75_value = _finite_float(dma_75)
    dma_100_value = _finite_float(dma_100)
    if dma_75_value is None or dma_100_value is None:
        return "Unknown"
    if dma_75_value > dma_100_value:
        return "75DMA above 100DMA"
    if dma_75_value < dma_100_value:
        return "75DMA below 100DMA"
    return "75DMA equal 100DMA"


def _dma_trend_label(trend_75: Any, trend_100: Any, dma_stack: str) -> str:
    trend_values = [_finite_float(trend_75), _finite_float(trend_100)]
    if any(value is None for value in trend_values):
        return "Unknown"
    rising = [float(value) > DMA_TREND_FLAT_THRESHOLD_PCT for value in trend_values if value is not None]
    falling = [float(value) < -DMA_TREND_FLAT_THRESHOLD_PCT for value in trend_values if value is not None]
    rising_count = sum(rising)
    falling_count = sum(falling)
    if falling_count == len(trend_values):
        return "Bearish"
    if rising_count == len(trend_values):
        return "Bullish" if dma_stack != "75DMA below 100DMA" else "Improving"
    if rising_count > 0 and falling_count == 0:
        return "Improving"
    if falling_count > 0 and rising_count == 0:
        return "Weakening"
    return "Mixed"


def _low_high_swing_count(pivots: pd.DataFrame) -> int:
    if pivots.empty:
        return 0
    count = 0
    for idx in range(len(pivots) - 1):
        if pivots.iloc[idx]["type"] == "LOW" and pivots.iloc[idx + 1]["type"] == "HIGH":
            count += 1
    return int(count)


def _coerce_date(value: Any | None) -> pd.Timestamp | None:
    if value is None or str(value).strip() == "":
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).normalize()


def _date_text(value: Any) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(parsed) else pd.Timestamp(parsed).strftime("%Y-%m-%d")


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _is_finite(value: Any) -> bool:
    return _finite_float(value) is not None


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if frame.empty or column not in frame.columns:
        return pd.Series(dtype=bool)
    return frame[column].map(_truthy_value)


def _truthy_value(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"", "0", "false", "f", "no", "n", "nan", "none"}:
        return False
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    return bool(text)


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if frame.empty or column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _emit_progress(
    progress_callback: Callable[[dict[str, Any]], None] | None,
    **payload: Any,
) -> None:
    if progress_callback is not None:
        progress_callback(payload)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _empty_distance_history() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "signal_id",
            "exchange",
            "symbol",
            "date_identifier",
            "date",
            "close",
            "dma_75",
            "dma_100",
            "distance_75dma_pct",
            "distance_100dma_pct",
        ]
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
