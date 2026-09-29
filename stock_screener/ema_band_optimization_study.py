from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from stock_screener.data.storage import Storage
from stock_screener.ema20_band_strategy_study import (
    DEFAULT_CURRENT_SIGNAL_LOOKBACK_SESSIONS,
    DEFAULT_HOLDING_SESSIONS,
    DEFAULT_PROFIT_TARGET_PCT,
    DEFAULT_ROUND_TRIP_COST_PCT,
    DEFAULT_STOP_BUFFER_PCT,
    DEFAULT_STOP_LOSS_PCT,
    ENTRY_PRICE_MODES,
    _add_features,
    _backtest_symbol,
    _coerce_date,
    _finite_float,
    _json_safe,
    _latest_candidate,
    _prepare_constituents,
    _prepare_daily,
)


DEFAULT_EMA_LENGTHS: tuple[int, ...] = (10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 75, 100)
DEFAULT_VALIDATION_MONTHS = 24
DEFAULT_MIN_TOTAL_TRADES = 5
DEFAULT_MIN_VALIDATION_TRADES = 2
DEFAULT_ROBUST_MIN_TOTAL_TRADES = 10
DEFAULT_ROBUST_MIN_VALIDATION_TRADES = 4
DEFAULT_ROBUST_MIN_PROFIT_FACTOR = 1.25
DEFAULT_WORKBOOK_MAX_TRADE_ROWS = 100_000

EMA_STRATEGY_VARIANTS: tuple[tuple[str, str], ...] = (
    ("Any EMA Band Bullish Pattern", "any_signal"),
    ("Bullish Inside Bar Below EMA Band", "bullish_inside_bar"),
    ("Bullish Engulfing Bar Below EMA Band", "bullish_engulfing"),
)

_EMA20_TO_GENERIC_STRATEGY = {
    "Any EMA20 Band Bullish Pattern": "Any EMA Band Bullish Pattern",
    "Bullish Inside Bar Below EMA20 Band": "Bullish Inside Bar Below EMA Band",
    "Bullish Engulfing Bar Below EMA20 Band": "Bullish Engulfing Bar Below EMA Band",
}

_PASS_COLUMN_BY_STRATEGY = dict(EMA_STRATEGY_VARIANTS)


@dataclass(frozen=True)
class EmaBandOptimizationResult:
    summary: dict[str, Any]
    best_by_stock: pd.DataFrame
    robust_best_by_stock: pd.DataFrame
    ema_stats: pd.DataFrame
    current_candidates: pd.DataFrame
    trades: pd.DataFrame


def run_ema_band_optimization_study(
    storage: Storage,
    constituents: pd.DataFrame,
    *,
    universe_name: str,
    start_date: Any,
    as_of_date: Any,
    ema_lengths: Iterable[int] = DEFAULT_EMA_LENGTHS,
    validation_months: int = DEFAULT_VALIDATION_MONTHS,
    min_total_trades: int = DEFAULT_MIN_TOTAL_TRADES,
    min_validation_trades: int = DEFAULT_MIN_VALIDATION_TRADES,
    current_signal_lookback_sessions: int = DEFAULT_CURRENT_SIGNAL_LOOKBACK_SESSIONS,
    holding_sessions: int = DEFAULT_HOLDING_SESSIONS,
    profit_target_pct: float = DEFAULT_PROFIT_TARGET_PCT,
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
    stop_buffer_pct: float = DEFAULT_STOP_BUFFER_PCT,
    round_trip_cost_pct: float = DEFAULT_ROUND_TRIP_COST_PCT,
    entry_price_mode: str = "next_open",
    exit_on_ema_close_reclaim: bool = True,
    required_latest_date: Any | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> EmaBandOptimizationResult:
    start_ts = _coerce_date(start_date)
    as_of_ts = _coerce_date(as_of_date)
    required_latest_ts = _coerce_date(required_latest_date)
    if start_ts is None:
        raise ValueError("Backtest start date is required.")
    if as_of_ts is None:
        raise ValueError("As-of date is required.")
    if start_ts > as_of_ts:
        raise ValueError("Backtest start date cannot be after the as-of date.")

    lengths = tuple(sorted({max(int(length), 1) for length in ema_lengths}))
    if not lengths:
        raise ValueError("At least one EMA length is required.")

    entry_mode = str(entry_price_mode or "next_open").strip().lower()
    if entry_mode not in ENTRY_PRICE_MODES:
        entry_mode = "next_open"

    validation_months = max(int(validation_months), 1)
    validation_start_ts = max(start_ts, as_of_ts - pd.DateOffset(months=validation_months))
    universe = _prepare_constituents(constituents)

    total_work = len(universe) * len(lengths)
    completed_work = 0
    trade_frames: list[pd.DataFrame] = []
    latest_frame_by_key: dict[tuple[str, str, int], pd.DataFrame] = {}
    symbol_meta: dict[tuple[str, str], dict[str, Any]] = {}
    skipped_no_data = 0
    skipped_short = 0
    skipped_stale = 0

    _emit_progress(
        progress_callback,
        phase="Optimizing EMA band lengths",
        completed=0,
        total=total_work,
        current_symbol="",
        current_exchange="",
    )

    for _, row in universe.iterrows():
        exchange = str(row.get("exchange") or "NSE").strip().upper()
        symbol = str(row.get("symbol") or "").strip().upper()
        name = str(row.get("name") or symbol).strip()
        source_universe = str(row.get("source_universe") or "").strip()
        if not symbol:
            completed_work += len(lengths)
            continue

        symbol_meta[(exchange, symbol)] = {
            "exchange": exchange,
            "symbol": symbol,
            "name": name,
            "source_universe": source_universe,
        }
        daily = _prepare_daily(storage.load_candles(exchange, symbol, "1D"))
        if daily.empty:
            skipped_no_data += 1
            completed_work += len(lengths)
            continue
        daily = daily[daily["date"].dt.normalize() <= as_of_ts].copy()
        if daily.empty:
            skipped_no_data += 1
            completed_work += len(lengths)
            continue
        latest_date = pd.Timestamp(daily.iloc[-1]["date"]).normalize()
        if required_latest_ts is not None and latest_date < required_latest_ts:
            skipped_stale += 1
            completed_work += len(lengths)
            continue

        for ema_length in lengths:
            completed_work += 1
            _emit_progress(
                progress_callback,
                phase="Optimizing EMA band lengths",
                completed=completed_work,
                total=total_work,
                current_symbol=symbol,
                current_exchange=exchange,
            )
            if len(daily) < ema_length + 2:
                skipped_short += 1
                continue
            frame = _add_features(daily, ema_length=ema_length)
            latest_frame_by_key[(exchange, symbol, ema_length)] = frame
            trades = _backtest_symbol(
                frame,
                _events_for_column(frame, "any_signal"),
                exchange=exchange,
                symbol=symbol,
                name=name,
                source_universe=source_universe,
                start_ts=start_ts,
                end_ts=as_of_ts,
                holding_sessions=holding_sessions,
                profit_target_pct=profit_target_pct,
                stop_loss_pct=stop_loss_pct,
                stop_buffer_pct=stop_buffer_pct,
                round_trip_cost_pct=round_trip_cost_pct,
                entry_price_mode=entry_mode,
                exit_on_ema_close_reclaim=exit_on_ema_close_reclaim,
            )
            if trades.empty:
                continue
            trades = trades.copy()
            trades["ema_length"] = int(ema_length)
            trades["strategy"] = trades["strategy"].replace(_EMA20_TO_GENERIC_STRATEGY)
            trade_frames.append(trades)

    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else _empty_optimization_trades()
    if not trades.empty:
        trades = trades.sort_values(["entry_date", "symbol", "ema_length", "strategy"], ascending=[False, True, True, True]).reset_index(drop=True)

    ema_stats = _build_ema_stats(
        trades,
        validation_start_ts=validation_start_ts,
        min_total_trades=min_total_trades,
        min_validation_trades=min_validation_trades,
    )
    best_by_stock = _best_rows_by_stock(ema_stats)
    robust_best_by_stock = _robust_best_rows_by_stock(ema_stats)
    current_candidates = _current_candidates_for_best_rows(
        best_by_stock,
        latest_frame_by_key,
        symbol_meta,
        as_of_ts=as_of_ts,
        lookback_sessions=current_signal_lookback_sessions,
        profit_target_pct=profit_target_pct,
        stop_loss_pct=stop_loss_pct,
        stop_buffer_pct=stop_buffer_pct,
        entry_price_mode=entry_mode,
    )
    summary = _build_summary(
        universe_name=universe_name,
        start_ts=start_ts,
        as_of_ts=as_of_ts,
        validation_start_ts=validation_start_ts,
        ema_lengths=lengths,
        total_symbols=len(universe),
        skipped_no_data=skipped_no_data,
        skipped_short=skipped_short,
        skipped_stale=skipped_stale,
        trades=trades,
        ema_stats=ema_stats,
        best_by_stock=best_by_stock,
        robust_best_by_stock=robust_best_by_stock,
        current_candidates=current_candidates,
        validation_months=validation_months,
        min_total_trades=min_total_trades,
        min_validation_trades=min_validation_trades,
        current_signal_lookback_sessions=current_signal_lookback_sessions,
        holding_sessions=holding_sessions,
        profit_target_pct=profit_target_pct,
        stop_loss_pct=stop_loss_pct,
        stop_buffer_pct=stop_buffer_pct,
        round_trip_cost_pct=round_trip_cost_pct,
        entry_price_mode=entry_mode,
        exit_on_ema_close_reclaim=exit_on_ema_close_reclaim,
    )
    return EmaBandOptimizationResult(
        summary=summary,
        best_by_stock=best_by_stock,
        robust_best_by_stock=robust_best_by_stock,
        ema_stats=ema_stats,
        current_candidates=current_candidates,
        trades=trades,
    )


def save_ema_band_optimization_outputs(result: EmaBandOptimizationResult, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "summary.json",
        "best_by_stock": output_dir / "best_by_stock.csv",
        "robust_best_by_stock": output_dir / "robust_best_by_stock.csv",
        "ema_stats": output_dir / "ema_stats.csv",
        "current_candidates": output_dir / "current_candidates.csv",
        "trades": output_dir / "trades.csv",
    }
    paths["summary"].write_text(json.dumps(_json_safe(result.summary), indent=2), encoding="utf-8")
    result.best_by_stock.to_csv(paths["best_by_stock"], index=False)
    result.robust_best_by_stock.to_csv(paths["robust_best_by_stock"], index=False)
    result.ema_stats.to_csv(paths["ema_stats"], index=False)
    result.current_candidates.to_csv(paths["current_candidates"], index=False)
    result.trades.to_csv(paths["trades"], index=False)
    return paths


def write_ema_band_optimization_workbook(
    result: EmaBandOptimizationResult,
    path: Path,
    *,
    max_trade_rows: int = DEFAULT_WORKBOOK_MAX_TRADE_ROWS,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trade_rows = len(result.trades)
    workbook_trades = result.trades.head(max(int(max_trade_rows), 0)).copy()
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        result.best_by_stock.to_excel(writer, sheet_name="Best EMA By Stock", index=False)
        result.robust_best_by_stock.to_excel(writer, sheet_name="Robust Best By Stock", index=False)
        result.current_candidates.to_excel(writer, sheet_name="Current Candidates", index=False)
        result.ema_stats.to_excel(writer, sheet_name="All EMA Stats", index=False)
        workbook_trades.to_excel(writer, sheet_name="Trades", index=False)
        pd.DataFrame([result.summary]).to_excel(writer, sheet_name="Summary", index=False)
        _methodology_frame().to_excel(writer, sheet_name="Methodology", index=False)
        if trade_rows > len(workbook_trades):
            pd.DataFrame(
                [
                    {
                        "note": "The workbook Trades sheet is capped to keep Excel responsive. Use trades.csv for the complete trade ledger.",
                        "workbook_trade_rows": int(len(workbook_trades)),
                        "full_trade_rows": int(trade_rows),
                    }
                ]
            ).to_excel(writer, sheet_name="Trade Export Note", index=False)


def _build_ema_stats(
    trades: pd.DataFrame,
    *,
    validation_start_ts: pd.Timestamp,
    min_total_trades: int,
    min_validation_trades: int,
) -> pd.DataFrame:
    columns = [
        "exchange",
        "symbol",
        "name",
        "source_universe",
        "ema_length",
        "strategy",
        "total_trades",
        "win_rate_pct",
        "avg_return_pct",
        "median_return_pct",
        "profit_factor",
        "avg_hold_sessions",
        "avg_mfe_pct",
        "avg_mae_pct",
        "avg_max_return_10_sessions_pct",
        "median_max_return_10_sessions_pct",
        "avg_close_return_10_sessions_pct",
        "validation_trades",
        "validation_win_rate_pct",
        "validation_avg_return_pct",
        "validation_profit_factor",
        "stopped_pct",
        "target_pct",
        "edge_score",
        "qualified",
        "score_notes",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns)

    frame = trades.copy()
    frame["entry_date_ts"] = pd.to_datetime(frame["entry_date"], errors="coerce").dt.normalize()
    group_columns = ["exchange", "symbol", "name", "source_universe", "ema_length", "strategy"]
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_columns, dropna=False):
        record = dict(zip(group_columns, tuple(key), strict=False))
        all_metrics = _return_metrics(group)
        validation_group = group[group["entry_date_ts"] >= validation_start_ts].copy()
        validation_metrics = _return_metrics(validation_group)
        exit_reasons = group["exit_reason"].fillna("").astype(str).str.upper()
        record.update(
            {
                **all_metrics,
                "validation_trades": validation_metrics["total_trades"],
                "validation_win_rate_pct": validation_metrics["win_rate_pct"],
                "validation_avg_return_pct": validation_metrics["avg_return_pct"],
                "validation_profit_factor": validation_metrics["profit_factor"],
                "stopped_pct": float(exit_reasons.str.contains("STOP", regex=False).mean() * 100.0) if len(exit_reasons) else np.nan,
                "target_pct": float(exit_reasons.str.contains("TARGET", regex=False).mean() * 100.0) if len(exit_reasons) else np.nan,
            }
        )
        score, qualified, notes = _edge_score(
            record,
            min_total_trades=min_total_trades,
            min_validation_trades=min_validation_trades,
        )
        record["edge_score"] = score
        record["qualified"] = qualified
        record["score_notes"] = notes
        rows.append(record)

    result = pd.DataFrame(rows)
    result = result.sort_values(
        ["qualified", "edge_score", "validation_avg_return_pct", "profit_factor", "total_trades"],
        ascending=[False, False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    return result.reindex(columns=columns)


def _best_rows_by_stock(ema_stats: pd.DataFrame) -> pd.DataFrame:
    if ema_stats.empty:
        return pd.DataFrame()
    rows: list[pd.Series] = []
    for _, group in ema_stats.groupby(["exchange", "symbol"], dropna=False):
        rows.append(group.iloc[0])
    result = pd.DataFrame(rows)
    result = result.sort_values(
        ["qualified", "edge_score", "validation_avg_return_pct", "profit_factor", "total_trades", "symbol"],
        ascending=[False, False, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    result.insert(0, "rank", range(1, len(result) + 1))
    return result


def _robust_best_rows_by_stock(ema_stats: pd.DataFrame) -> pd.DataFrame:
    if ema_stats.empty:
        return pd.DataFrame()
    profit_factor = pd.to_numeric(ema_stats["profit_factor"], errors="coerce")
    robust = ema_stats[
        (pd.to_numeric(ema_stats["total_trades"], errors="coerce") >= DEFAULT_ROBUST_MIN_TOTAL_TRADES)
        & (pd.to_numeric(ema_stats["validation_trades"], errors="coerce") >= DEFAULT_ROBUST_MIN_VALIDATION_TRADES)
        & (pd.to_numeric(ema_stats["avg_return_pct"], errors="coerce") > 0.0)
        & (pd.to_numeric(ema_stats["validation_avg_return_pct"], errors="coerce") > 0.0)
        & (profit_factor > DEFAULT_ROBUST_MIN_PROFIT_FACTOR)
    ].copy()
    if robust.empty:
        return robust
    rows: list[pd.Series] = []
    for _, group in robust.groupby(["exchange", "symbol"], dropna=False):
        rows.append(group.iloc[0])
    result = pd.DataFrame(rows)
    result = result.sort_values(
        ["edge_score", "validation_avg_return_pct", "profit_factor", "total_trades", "symbol"],
        ascending=[False, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    result.insert(0, "robust_rank", range(1, len(result) + 1))
    return result


def _current_candidates_for_best_rows(
    best_by_stock: pd.DataFrame,
    latest_frame_by_key: dict[tuple[str, str, int], pd.DataFrame],
    symbol_meta: dict[tuple[str, str], dict[str, Any]],
    *,
    as_of_ts: pd.Timestamp,
    lookback_sessions: int,
    profit_target_pct: float,
    stop_loss_pct: float,
    stop_buffer_pct: float,
    entry_price_mode: str,
) -> pd.DataFrame:
    if best_by_stock.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, best in best_by_stock.iterrows():
        exchange = str(best.get("exchange") or "NSE").upper()
        symbol = str(best.get("symbol") or "").upper()
        ema_length = int(best.get("ema_length") or 0)
        strategy = str(best.get("strategy") or "Any EMA Band Bullish Pattern")
        pass_column = _PASS_COLUMN_BY_STRATEGY.get(strategy, "any_signal")
        frame = latest_frame_by_key.get((exchange, symbol, ema_length))
        if frame is None or frame.empty:
            continue
        meta = symbol_meta.get((exchange, symbol), {})
        candidate = _latest_candidate(
            frame,
            _events_for_column(frame, pass_column),
            exchange=exchange,
            symbol=symbol,
            name=str(meta.get("name") or best.get("name") or symbol),
            source_universe=str(meta.get("source_universe") or best.get("source_universe") or ""),
            as_of_ts=as_of_ts,
            lookback_sessions=lookback_sessions,
            profit_target_pct=profit_target_pct,
            stop_loss_pct=stop_loss_pct,
            stop_buffer_pct=stop_buffer_pct,
            entry_price_mode=entry_price_mode,
        )
        if candidate is None:
            continue
        candidate.update(
            {
                "optimized_ema_length": ema_length,
                "optimized_strategy": strategy,
                "optimized_edge_score": best.get("edge_score"),
                "optimized_rank": best.get("rank"),
                "optimized_total_trades": best.get("total_trades"),
                "optimized_win_rate_pct": best.get("win_rate_pct"),
                "optimized_avg_return_pct": best.get("avg_return_pct"),
                "optimized_profit_factor": best.get("profit_factor"),
                "optimized_validation_trades": best.get("validation_trades"),
                "optimized_validation_avg_return_pct": best.get("validation_avg_return_pct"),
                "optimized_avg_max_return_10_sessions_pct": best.get("avg_max_return_10_sessions_pct"),
                "optimized_median_max_return_10_sessions_pct": best.get("median_max_return_10_sessions_pct"),
                "optimized_avg_close_return_10_sessions_pct": best.get("avg_close_return_10_sessions_pct"),
                "optimized_qualified": best.get("qualified"),
            }
        )
        rows.append(candidate)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["optimized_qualified", "optimized_edge_score", "candidate_score", "sessions_since_signal", "symbol"],
        ascending=[False, False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)


def build_emax_current_candidates(
    storage: Storage,
    best_by_stock: pd.DataFrame,
    *,
    as_of_date: Any,
    robust_best_by_stock: pd.DataFrame | None = None,
    lookback_sessions: int = DEFAULT_CURRENT_SIGNAL_LOOKBACK_SESSIONS,
    profit_target_pct: float = DEFAULT_PROFIT_TARGET_PCT,
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
    stop_buffer_pct: float = DEFAULT_STOP_BUFFER_PCT,
    entry_price_mode: str = "next_open",
    required_latest_date: Any | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> pd.DataFrame:
    """Build current EMAX candidates using each stock's optimized EMA length."""
    as_of_ts = _coerce_date(as_of_date)
    required_latest_ts = _coerce_date(required_latest_date)
    if as_of_ts is None:
        raise ValueError("As-of date is required.")
    if best_by_stock.empty:
        return pd.DataFrame()

    entry_mode = str(entry_price_mode or "next_open").strip().lower()
    if entry_mode not in ENTRY_PRICE_MODES:
        entry_mode = "next_open"

    rows: list[dict[str, Any]] = []
    total = len(best_by_stock)
    _emit_progress(
        progress_callback,
        phase="Running EMAX screener",
        completed=0,
        total=total,
        current_symbol="",
        current_exchange="",
    )
    for completed, (_, best) in enumerate(best_by_stock.iterrows(), start=1):
        exchange = str(best.get("exchange") or "NSE").strip().upper()
        symbol = str(best.get("symbol") or "").strip().upper()
        _emit_progress(
            progress_callback,
            phase="Running EMAX screener",
            completed=completed,
            total=total,
            current_symbol=symbol,
            current_exchange=exchange,
        )
        if not symbol:
            continue
        ema_length = int(best.get("ema_length") or 0)
        if ema_length <= 0:
            continue
        daily = _prepare_daily(storage.load_candles(exchange, symbol, "1D"))
        if daily.empty:
            continue
        daily = daily[daily["date"].dt.normalize() <= as_of_ts].copy()
        if daily.empty:
            continue
        latest_date = pd.Timestamp(daily.iloc[-1]["date"]).normalize()
        if required_latest_ts is not None and latest_date < required_latest_ts:
            continue
        if len(daily) < ema_length + 2:
            continue

        frame = _add_features(daily, ema_length=ema_length)
        strategy = str(best.get("strategy") or "Any EMA Band Bullish Pattern")
        pass_column = _PASS_COLUMN_BY_STRATEGY.get(strategy, "any_signal")
        candidate = _latest_candidate(
            frame,
            _events_for_column(frame, pass_column),
            exchange=exchange,
            symbol=symbol,
            name=str(best.get("name") or symbol),
            source_universe=str(best.get("source_universe") or ""),
            as_of_ts=as_of_ts,
            lookback_sessions=lookback_sessions,
            profit_target_pct=profit_target_pct,
            stop_loss_pct=stop_loss_pct,
            stop_buffer_pct=stop_buffer_pct,
            entry_price_mode=entry_mode,
        )
        if candidate is None:
            continue
        candidate.update(
            {
                "optimized_ema_length": ema_length,
                "optimized_strategy": strategy,
                "optimized_edge_score": best.get("edge_score"),
                "optimized_rank": best.get("rank"),
                "optimized_total_trades": best.get("total_trades"),
                "optimized_win_rate_pct": best.get("win_rate_pct"),
                "optimized_avg_return_pct": best.get("avg_return_pct"),
                "optimized_profit_factor": best.get("profit_factor"),
                "optimized_validation_trades": best.get("validation_trades"),
                "optimized_validation_avg_return_pct": best.get("validation_avg_return_pct"),
                "optimized_avg_max_return_10_sessions_pct": best.get("avg_max_return_10_sessions_pct"),
                "optimized_median_max_return_10_sessions_pct": best.get("median_max_return_10_sessions_pct"),
                "optimized_avg_close_return_10_sessions_pct": best.get("avg_close_return_10_sessions_pct"),
                "optimized_qualified": best.get("qualified"),
            }
        )
        rows.append(candidate)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = _add_robust_context(result, robust_best_by_stock)
    result["daily_prediction_score"] = (
        pd.to_numeric(result.get("optimized_edge_score"), errors="coerce").fillna(0.0) * 0.65
        + pd.to_numeric(result.get("candidate_score"), errors="coerce").fillna(0.0) * 0.25
        + result.get("robust_best_ema", pd.Series(False, index=result.index)).fillna(False).astype(bool).astype(int) * 10.0
    ).round(2)
    result = result.sort_values(
        ["robust_best_ema", "daily_prediction_score", "sessions_since_signal", "symbol"],
        ascending=[False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    result.insert(0, "daily_prediction_rank", range(1, len(result) + 1))
    result["candidate_rank"] = result["daily_prediction_rank"]
    return result


def _add_robust_context(candidates: pd.DataFrame, robust_best_by_stock: pd.DataFrame | None) -> pd.DataFrame:
    result = candidates.copy()
    if robust_best_by_stock is None or robust_best_by_stock.empty:
        result["robust_best_ema"] = False
        return result
    robust = robust_best_by_stock.copy()
    rename = {
        "ema_length": "optimized_ema_length",
        "strategy": "optimized_strategy",
        "edge_score": "robust_edge_score",
        "total_trades": "robust_total_trades",
        "validation_trades": "robust_validation_trades",
        "validation_avg_return_pct": "robust_validation_avg_return_pct",
        "profit_factor": "robust_profit_factor",
        "avg_max_return_10_sessions_pct": "robust_avg_max_return_10_sessions_pct",
        "median_max_return_10_sessions_pct": "robust_median_max_return_10_sessions_pct",
        "avg_close_return_10_sessions_pct": "robust_avg_close_return_10_sessions_pct",
    }
    robust = robust.rename(columns=rename)
    keep = [
        column
        for column in (
            "exchange",
            "symbol",
            "optimized_ema_length",
            "optimized_strategy",
            "robust_rank",
            "robust_edge_score",
            "robust_total_trades",
            "robust_validation_trades",
            "robust_validation_avg_return_pct",
            "robust_profit_factor",
            "robust_avg_max_return_10_sessions_pct",
            "robust_median_max_return_10_sessions_pct",
            "robust_avg_close_return_10_sessions_pct",
        )
        if column in robust.columns
    ]
    if not {"exchange", "symbol", "optimized_ema_length", "optimized_strategy"}.issubset(keep):
        result["robust_best_ema"] = False
        return result
    result = result.merge(
        robust[keep],
        on=["exchange", "symbol", "optimized_ema_length", "optimized_strategy"],
        how="left",
    )
    result["robust_best_ema"] = result.get("robust_rank", pd.Series(np.nan, index=result.index)).notna()
    return result


def _return_metrics(group: pd.DataFrame) -> dict[str, Any]:
    returns = pd.to_numeric(group.get("net_return_pct", pd.Series(dtype="float64")), errors="coerce").dropna()
    wins = returns[returns > 0.0]
    losses = returns[returns <= 0.0]
    return {
        "total_trades": int(len(returns)),
        "win_rate_pct": float(len(wins) / len(returns) * 100.0) if len(returns) else 0.0,
        "avg_return_pct": float(returns.mean()) if len(returns) else 0.0,
        "median_return_pct": float(returns.median()) if len(returns) else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(wins) and len(losses) and losses.sum() != 0 else np.nan,
        "avg_hold_sessions": float(pd.to_numeric(group.get("hold_trading_sessions", pd.Series(dtype="float64")), errors="coerce").mean()),
        "avg_mfe_pct": float(pd.to_numeric(group.get("mfe_pct", pd.Series(dtype="float64")), errors="coerce").mean()),
        "avg_mae_pct": float(pd.to_numeric(group.get("mae_pct", pd.Series(dtype="float64")), errors="coerce").mean()),
        "avg_max_return_10_sessions_pct": float(pd.to_numeric(group.get("max_return_10_sessions_pct", pd.Series(dtype="float64")), errors="coerce").mean()),
        "median_max_return_10_sessions_pct": float(pd.to_numeric(group.get("max_return_10_sessions_pct", pd.Series(dtype="float64")), errors="coerce").median()),
        "avg_close_return_10_sessions_pct": float(pd.to_numeric(group.get("close_return_10_sessions_pct", pd.Series(dtype="float64")), errors="coerce").mean()),
    }


def _edge_score(record: dict[str, Any], *, min_total_trades: int, min_validation_trades: int) -> tuple[float, bool, str]:
    total_trades = int(record.get("total_trades") or 0)
    validation_trades = int(record.get("validation_trades") or 0)
    win_rate = _num(record.get("win_rate_pct"), 0.0)
    avg_return = _num(record.get("avg_return_pct"), 0.0)
    validation_avg_return = _num(record.get("validation_avg_return_pct"), 0.0)
    profit_factor = _num(record.get("profit_factor"), 0.0)
    stopped_pct = _num(record.get("stopped_pct"), 0.0)

    score = 0.0
    score += _clamp((avg_return + 1.0) / 5.0, 0.0, 1.0) * 22.0
    score += _clamp((validation_avg_return + 1.0) / 5.0, 0.0, 1.0) * 26.0
    score += _clamp((win_rate - 40.0) / 25.0, 0.0, 1.0) * 14.0
    score += _clamp((profit_factor - 1.0) / 1.5, 0.0, 1.0) * 14.0
    score += _clamp(total_trades / max(float(min_total_trades), 1.0), 0.0, 1.0) * 12.0
    score += _clamp(validation_trades / max(float(min_validation_trades), 1.0), 0.0, 1.0) * 12.0
    if avg_return < 0.0:
        score -= 18.0
    if validation_avg_return < 0.0:
        score -= 22.0
    if profit_factor and profit_factor < 1.0:
        score -= 8.0
    if stopped_pct >= 55.0:
        score -= 8.0

    notes: list[str] = []
    if total_trades < min_total_trades:
        notes.append("low total sample")
    if validation_trades < min_validation_trades:
        notes.append("low validation sample")
    if avg_return <= 0.0:
        notes.append("non-positive average return")
    if validation_avg_return <= 0.0:
        notes.append("weak validation return")
    if stopped_pct >= 55.0:
        notes.append("high stop rate")
    qualified = not notes
    return round(float(_clamp(score, 0.0, 100.0)), 2), qualified, "; ".join(notes) if notes else "qualified"


def _events_for_column(frame: pd.DataFrame, column: str) -> list[int]:
    if frame.empty or column not in frame.columns:
        return []
    mask = frame[column].fillna(False).astype(bool)
    return [int(index) for index in frame.index[mask]]


def _build_summary(
    *,
    universe_name: str,
    start_ts: pd.Timestamp,
    as_of_ts: pd.Timestamp,
    validation_start_ts: pd.Timestamp,
    ema_lengths: tuple[int, ...],
    total_symbols: int,
    skipped_no_data: int,
    skipped_short: int,
    skipped_stale: int,
    trades: pd.DataFrame,
    ema_stats: pd.DataFrame,
    best_by_stock: pd.DataFrame,
    robust_best_by_stock: pd.DataFrame,
    current_candidates: pd.DataFrame,
    validation_months: int,
    min_total_trades: int,
    min_validation_trades: int,
    current_signal_lookback_sessions: int,
    holding_sessions: int,
    profit_target_pct: float,
    stop_loss_pct: float,
    stop_buffer_pct: float,
    round_trip_cost_pct: float,
    entry_price_mode: str,
    exit_on_ema_close_reclaim: bool,
) -> dict[str, Any]:
    qualified = int(best_by_stock["qualified"].sum()) if not best_by_stock.empty and "qualified" in best_by_stock.columns else 0
    best_ema_counts = (
        best_by_stock["ema_length"].value_counts().sort_index().to_dict()
        if not best_by_stock.empty and "ema_length" in best_by_stock.columns
        else {}
    )
    return {
        "study_name": "EMA Band IB/EB Length Optimization",
        "universe": universe_name,
        "start_date": start_ts.strftime("%Y-%m-%d"),
        "as_of_date": as_of_ts.strftime("%Y-%m-%d"),
        "validation_start_date": validation_start_ts.strftime("%Y-%m-%d"),
        "validation_months": int(validation_months),
        "ema_lengths": ",".join(str(length) for length in ema_lengths),
        "min_total_trades": int(min_total_trades),
        "min_validation_trades": int(min_validation_trades),
        "current_signal_lookback_sessions": int(current_signal_lookback_sessions),
        "holding_sessions": int(holding_sessions),
        "profit_target_pct": float(profit_target_pct),
        "stop_loss_pct": float(stop_loss_pct),
        "stop_buffer_pct": float(stop_buffer_pct),
        "round_trip_cost_pct": float(round_trip_cost_pct),
        "entry_price_mode": entry_price_mode,
        "exit_on_ema_close_reclaim": bool(exit_on_ema_close_reclaim),
        "symbols_requested": int(total_symbols),
        "symbols_with_any_result": int(best_by_stock[["exchange", "symbol"]].drop_duplicates().shape[0]) if not best_by_stock.empty else 0,
        "symbols_with_qualified_best_ema": qualified,
        "symbols_with_robust_best_ema": int(robust_best_by_stock[["exchange", "symbol"]].drop_duplicates().shape[0]) if not robust_best_by_stock.empty else 0,
        "robust_min_total_trades": int(DEFAULT_ROBUST_MIN_TOTAL_TRADES),
        "robust_min_validation_trades": int(DEFAULT_ROBUST_MIN_VALIDATION_TRADES),
        "robust_min_profit_factor": float(DEFAULT_ROBUST_MIN_PROFIT_FACTOR),
        "symbols_skipped_no_data": int(skipped_no_data),
        "symbols_skipped_short_history": int(skipped_short),
        "symbols_skipped_stale_history": int(skipped_stale),
        "total_ema_strategy_rows": int(len(ema_stats)),
        "total_backtest_trades": int(len(trades)),
        "current_candidates": int(len(current_candidates)),
        "best_ema_distribution": best_ema_counts,
    }


def _methodology_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "step": "Signal definition",
                "method": "For each EMA length, calculate EMA high/low/close bands. A signal requires prior candle red, current candle green, both candle highs below the EMA low band, and the current candle to be an inside bar or range-engulfing bar.",
            },
            {
                "step": "Backtest entry",
                "method": "Enter on the next session using the configured next-open or next-close mode.",
            },
            {
                "step": "Backtest exit",
                "method": "Exit on target, stop, optional EMA-close reclaim, or max holding sessions, net of configured round-trip cost.",
            },
            {
                "step": "10-session maximum return",
                "method": "For every historical IB/EB event, measure the maximum high reached after buying the next session and holding for up to 10 trading sessions. This is independent of target, stop, or EMA-reclaim exits.",
            },
            {
                "step": "Per-stock optimization",
                "method": "Rank every stock/EMA/pattern variant by edge score. The score rewards positive average return, validation-period return, win rate, profit factor, and enough trades. It penalizes thin samples, weak validation, negative average return, and high stop rate.",
            },
            {
                "step": "Interpretation",
                "method": "A best EMA is a research hypothesis, not a guaranteed parameter. Prefer qualified rows with enough total and validation trades; treat low-sample winners as overfit until retested.",
            },
            {
                "step": "Robust shortlist",
                "method": "The robust sheet keeps only stock/EMA/pattern rows with at least 10 total trades, at least 4 validation trades, positive all-period and validation average returns, and profit factor above 1.25.",
            },
        ]
    )


def _empty_optimization_trades() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "exchange",
            "symbol",
            "name",
            "source_universe",
            "strategy",
            "ema_length",
            "signal_date",
            "entry_date",
            "entry_price",
            "exit_date",
            "exit_price",
            "exit_reason",
            "net_return_pct",
            "win_flag",
            "mfe_pct",
            "mae_pct",
        ]
    )


def _num(value: Any, default: float = 0.0) -> float:
    number = _finite_float(value)
    return default if number is None else float(number)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _emit_progress(callback: Callable[[dict[str, Any]], None] | None, **payload: Any) -> None:
    if callback is not None:
        callback(payload)
