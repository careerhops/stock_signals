from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from stock_screener.data.storage import Storage


DEFAULT_WARMUP_MONTHS = 8
DEFAULT_MAX_RECLAIM_SESSIONS = 7
DEFAULT_CURRENT_SIGNAL_LOOKBACK_SESSIONS = 7
DEFAULT_DMA_TREND_SESSIONS = 20
DEFAULT_PIVOT_ORDER = 3
DEFAULT_HOLDING_SESSIONS = 20
DEFAULT_PROFIT_TARGET_PCT = 12.0
DEFAULT_STOP_LOSS_PCT = 6.0
DEFAULT_STOP_BUFFER_PCT = 0.5
DEFAULT_ROUND_TRIP_COST_PCT = 0.20
DMA_TREND_FLAT_THRESHOLD_PCT = 0.25
ENTRY_PRICE_MODES = {"next_open", "next_close"}

STRATEGY_VARIANTS: tuple[tuple[str, str], ...] = (
    ("Raw 75DMA Reclaim", "raw_pass"),
    ("Trend Filtered 75DMA Reclaim", "trend_pass"),
    ("Strict Long 75DMA Reclaim", "strict_pass"),
)


@dataclass(frozen=True)
class DmaReclaimStrategyResult:
    summary: dict[str, Any]
    candidates: pd.DataFrame
    strategy_stats: pd.DataFrame
    yearly_stats: pd.DataFrame
    stock_stats: pd.DataFrame
    trades: pd.DataFrame


def run_dma_reclaim_strategy_study(
    storage: Storage,
    constituents: pd.DataFrame,
    *,
    universe_name: str,
    start_date: Any,
    as_of_date: Any,
    warmup_months: int = DEFAULT_WARMUP_MONTHS,
    max_reclaim_sessions: int = DEFAULT_MAX_RECLAIM_SESSIONS,
    current_signal_lookback_sessions: int = DEFAULT_CURRENT_SIGNAL_LOOKBACK_SESSIONS,
    holding_sessions: int = DEFAULT_HOLDING_SESSIONS,
    profit_target_pct: float = DEFAULT_PROFIT_TARGET_PCT,
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
    stop_buffer_pct: float = DEFAULT_STOP_BUFFER_PCT,
    round_trip_cost_pct: float = DEFAULT_ROUND_TRIP_COST_PCT,
    entry_price_mode: str = "next_open",
    exit_on_close_below_75dma: bool = True,
    pivot_order: int = DEFAULT_PIVOT_ORDER,
    required_latest_date: Any | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> DmaReclaimStrategyResult:
    start_ts = _coerce_date(start_date)
    as_of_ts = _coerce_date(as_of_date)
    required_latest_ts = _coerce_date(required_latest_date)
    if start_ts is None:
        raise ValueError("Backtest start date is required.")
    if as_of_ts is None:
        raise ValueError("As-of date is required.")
    if start_ts > as_of_ts:
        raise ValueError("Backtest start date cannot be after the as-of date.")
    entry_mode = str(entry_price_mode or "next_open").strip().lower()
    if entry_mode not in ENTRY_PRICE_MODES:
        entry_mode = "next_open"

    universe = _prepare_constituents(constituents)
    total = len(universe)
    candidate_rows: list[dict[str, Any]] = []
    trade_frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []
    skipped_stale = 0
    skipped_short = 0
    skipped_no_data = 0

    _emit_progress(
        progress_callback,
        phase="Running 75DMA reclaim strategy",
        completed=0,
        total=total,
        current_symbol="",
        current_exchange="",
    )

    for completed, (_, row) in enumerate(universe.iterrows(), start=1):
        exchange = str(row.get("exchange") or "NSE").strip().upper()
        symbol = str(row.get("symbol") or "").strip().upper()
        name = str(row.get("name") or symbol).strip()
        source_universe = str(row.get("source_universe") or "").strip()
        _emit_progress(
            progress_callback,
            phase="Running 75DMA reclaim strategy",
            completed=completed,
            total=total,
            current_symbol=symbol,
            current_exchange=exchange,
        )
        if not symbol:
            continue

        daily = _prepare_daily(storage.load_candles(exchange, symbol, "1D"))
        if daily.empty:
            skipped_no_data += 1
            continue
        daily = daily[daily["date"].dt.normalize() <= as_of_ts].copy()
        if daily.empty:
            skipped_no_data += 1
            continue
        latest_date = pd.Timestamp(daily.iloc[-1]["date"]).normalize()
        if required_latest_ts is not None and latest_date < required_latest_ts:
            skipped_stale += 1
            continue
        if len(daily) < 110:
            skipped_short += 1
            continue

        frame = _add_features(daily)
        pivots = _confirmed_pivots(frame, order=max(int(pivot_order), 1))
        coverage_rows.append(
            {
                "exchange": exchange,
                "symbol": symbol,
                "name": name,
                "history_start": frame.iloc[0]["date"],
                "history_end": latest_date,
                "history_rows": len(frame),
                "source_universe": source_universe,
            }
        )

        events = _reclaim_events(frame, max_reclaim_sessions=max_reclaim_sessions)
        candidate = _latest_candidate(
            frame,
            pivots,
            events,
            exchange=exchange,
            symbol=symbol,
            name=name,
            source_universe=source_universe,
            as_of_ts=as_of_ts,
            lookback_sessions=current_signal_lookback_sessions,
            profit_target_pct=profit_target_pct,
            stop_loss_pct=stop_loss_pct,
            stop_buffer_pct=stop_buffer_pct,
            entry_price_mode=entry_mode,
        )
        if candidate is not None:
            candidate_rows.append(candidate)

        trades = _backtest_symbol(
            frame,
            pivots,
            events,
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
            exit_on_close_below_75dma=exit_on_close_below_75dma,
        )
        if not trades.empty:
            trade_frames.append(trades)

    candidates = pd.DataFrame(candidate_rows)
    if not candidates.empty:
        candidates = candidates.sort_values(
            ["candidate_score", "strategy_tier_rank", "sessions_since_reclaim", "symbol"],
            ascending=[False, True, True, True],
            na_position="last",
        ).reset_index(drop=True)
        candidates["candidate_rank"] = range(1, len(candidates) + 1)

    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else _empty_trades()
    if not trades.empty:
        trades = trades.sort_values(
            ["entry_date", "strategy", "symbol"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
    strategy_stats = _aggregate_stats(trades, ["strategy"])
    yearly_stats = _aggregate_yearly(trades)
    stock_stats = _aggregate_stats(trades, ["strategy", "exchange", "symbol", "name", "source_universe"])
    coverage = pd.DataFrame(coverage_rows)
    summary = _build_summary(
        candidates,
        trades,
        strategy_stats,
        coverage,
        universe_name=universe_name,
        start_ts=start_ts,
        as_of_ts=as_of_ts,
        total_symbols=total,
        skipped_no_data=skipped_no_data,
        skipped_short=skipped_short,
        skipped_stale=skipped_stale,
        warmup_months=warmup_months,
        max_reclaim_sessions=max_reclaim_sessions,
        current_signal_lookback_sessions=current_signal_lookback_sessions,
        holding_sessions=holding_sessions,
        profit_target_pct=profit_target_pct,
        stop_loss_pct=stop_loss_pct,
        stop_buffer_pct=stop_buffer_pct,
        round_trip_cost_pct=round_trip_cost_pct,
        entry_price_mode=entry_mode,
        exit_on_close_below_75dma=exit_on_close_below_75dma,
        pivot_order=pivot_order,
    )
    return DmaReclaimStrategyResult(
        summary=summary,
        candidates=candidates,
        strategy_stats=strategy_stats,
        yearly_stats=yearly_stats,
        stock_stats=stock_stats,
        trades=trades,
    )


def save_dma_reclaim_strategy_outputs(result: DmaReclaimStrategyResult, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "summary.json",
        "candidates": output_dir / "latest_candidates.csv",
        "strategy_stats": output_dir / "strategy_stats.csv",
        "yearly_stats": output_dir / "yearly_stats.csv",
        "stock_stats": output_dir / "stock_stats.csv",
        "trades": output_dir / "trades.csv",
    }
    paths["summary"].write_text(json.dumps(_json_safe(result.summary), indent=2), encoding="utf-8")
    result.candidates.to_csv(paths["candidates"], index=False)
    result.strategy_stats.to_csv(paths["strategy_stats"], index=False)
    result.yearly_stats.to_csv(paths["yearly_stats"], index=False)
    result.stock_stats.to_csv(paths["stock_stats"], index=False)
    result.trades.to_csv(paths["trades"], index=False)
    return paths


def load_dma_reclaim_strategy_outputs(output_dir: Path) -> DmaReclaimStrategyResult:
    summary_path = output_dir / "summary.json"
    summary: dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}
    return DmaReclaimStrategyResult(
        summary=summary,
        candidates=_read_csv(output_dir / "latest_candidates.csv"),
        strategy_stats=_read_csv(output_dir / "strategy_stats.csv"),
        yearly_stats=_read_csv(output_dir / "yearly_stats.csv"),
        stock_stats=_read_csv(output_dir / "stock_stats.csv"),
        trades=_read_csv(output_dir / "trades.csv"),
    )


def write_dma_reclaim_strategy_workbook(result: DmaReclaimStrategyResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        result.candidates.to_excel(writer, sheet_name="Current Candidates", index=False)
        result.strategy_stats.to_excel(writer, sheet_name="Strategy Stats", index=False)
        result.stock_stats.to_excel(writer, sheet_name="Stock Stats", index=False)
        result.yearly_stats.to_excel(writer, sheet_name="Yearly Stats", index=False)
        result.trades.to_excel(writer, sheet_name="Trades", index=False)
        pd.DataFrame([result.summary]).to_excel(writer, sheet_name="Summary", index=False)


def _prepare_constituents(constituents: pd.DataFrame) -> pd.DataFrame:
    if constituents.empty:
        return pd.DataFrame(columns=["exchange", "symbol", "name", "source_universe"])
    frame = constituents.copy()
    symbol_column = next(
        (column for column in ("Symbol", "symbol", "tradingsymbol", "Tradingsymbol") if column in frame.columns),
        "",
    )
    if not symbol_column:
        return pd.DataFrame(columns=["exchange", "symbol", "name", "source_universe"])
    frame["symbol"] = frame[symbol_column].fillna("").astype(str).str.upper().str.strip()
    if "exchange" not in frame.columns:
        frame["exchange"] = "NSE"
    frame["exchange"] = frame["exchange"].fillna("NSE").astype(str).str.upper().str.strip()
    frame["exchange"] = frame["exchange"].where(frame["exchange"].isin({"NSE", "BSE"}), "NSE")
    if "Company Name" in frame.columns:
        frame["name"] = frame["Company Name"]
    elif "name" not in frame.columns:
        frame["name"] = frame["symbol"]
    frame["name"] = frame["name"].fillna("").astype(str).str.strip()
    if "source_universe" not in frame.columns:
        frame["source_universe"] = ""
    frame["source_universe"] = frame["source_universe"].fillna("").astype(str).str.strip()
    frame = frame[frame["symbol"].ne("")].drop_duplicates(["exchange", "symbol"], keep="last")
    return frame[["exchange", "symbol", "name", "source_universe"]].sort_values(["exchange", "symbol"]).reset_index(drop=True)


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


def _add_features(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy().reset_index(drop=True)
    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    volume = frame["volume"]
    frame["dma_75"] = close.rolling(75, min_periods=75).mean()
    frame["dma_100"] = close.rolling(100, min_periods=100).mean()
    frame["distance_75dma_pct"] = (close / frame["dma_75"] - 1.0) * 100.0
    frame["distance_100dma_pct"] = (close / frame["dma_100"] - 1.0) * 100.0
    frame["dma_75_trend_20d_pct"] = (frame["dma_75"] / frame["dma_75"].shift(DEFAULT_DMA_TREND_SESSIONS) - 1.0) * 100.0
    frame["dma_100_trend_20d_pct"] = (frame["dma_100"] / frame["dma_100"].shift(DEFAULT_DMA_TREND_SESSIONS) - 1.0) * 100.0
    frame["volume_median_50"] = volume.rolling(50, min_periods=20).median()
    frame["volume_ratio_50"] = volume / frame["volume_median_50"].replace(0, np.nan)
    true_range = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["atr_14"] = true_range.rolling(14, min_periods=14).mean()
    return frame


def _reclaim_events(frame: pd.DataFrame, *, max_reclaim_sessions: int) -> list[dict[str, int]]:
    events: list[dict[str, int]] = []
    close = pd.to_numeric(frame.get("close"), errors="coerce")
    dma = pd.to_numeric(frame.get("dma_75"), errors="coerce")
    if len(frame) < 2:
        return events
    for idx in range(1, len(frame)):
        previous_close = _finite_float(close.iloc[idx - 1])
        previous_dma = _finite_float(dma.iloc[idx - 1])
        current_close = _finite_float(close.iloc[idx])
        current_dma = _finite_float(dma.iloc[idx])
        if previous_close is None or previous_dma is None or current_close is None or current_dma is None:
            continue
        if not (previous_close >= previous_dma and current_close < current_dma):
            continue
        last_idx = min(len(frame) - 1, idx + max(int(max_reclaim_sessions), 1))
        for reclaim_idx in range(idx + 1, last_idx + 1):
            reclaim_close = _finite_float(close.iloc[reclaim_idx])
            reclaim_dma = _finite_float(dma.iloc[reclaim_idx])
            if reclaim_close is None or reclaim_dma is None:
                continue
            if reclaim_close > reclaim_dma:
                events.append(
                    {
                        "break_index": int(idx),
                        "reclaim_index": int(reclaim_idx),
                        "sessions_to_reclaim": int(reclaim_idx - idx),
                    }
                )
                break
    return events


def _latest_candidate(
    frame: pd.DataFrame,
    pivots: pd.DataFrame,
    events: list[dict[str, int]],
    *,
    exchange: str,
    symbol: str,
    name: str,
    source_universe: str,
    as_of_ts: pd.Timestamp,
    lookback_sessions: int,
    profit_target_pct: float,
    stop_loss_pct: float,
    stop_buffer_pct: float,
    entry_price_mode: str,
) -> dict[str, Any] | None:
    if frame.empty or not events:
        return None
    as_of_candidates = frame[frame["date"].dt.normalize() <= as_of_ts]
    if as_of_candidates.empty:
        return None
    as_of_index = int(as_of_candidates.index[-1])
    min_reclaim_index = max(0, as_of_index - max(int(lookback_sessions), 1) + 1)
    recent_events = [
        event
        for event in events
        if min_reclaim_index <= int(event["reclaim_index"]) <= as_of_index
    ]
    if not recent_events:
        return None
    event = max(recent_events, key=lambda item: int(item["reclaim_index"]))
    context = _signal_context(frame, pivots, event)
    reclaim_index = int(event["reclaim_index"])
    entry_index = reclaim_index + 1
    entry_row = frame.iloc[entry_index] if entry_index < len(frame) else None
    entry_price = _entry_price(entry_row, entry_price_mode) if entry_row is not None else None
    reclaim_close = _finite_float(frame.iloc[reclaim_index].get("close"))
    planning_price = entry_price if entry_price is not None else reclaim_close
    stop_price = np.nan
    target_price = np.nan
    if planning_price is not None:
        stop_price = _planned_stop_price(
            float(planning_price),
            frame.iloc[reclaim_index],
            stop_loss_pct=stop_loss_pct,
            stop_buffer_pct=stop_buffer_pct,
        )
        target_price = float(planning_price) * (1.0 + float(profit_target_pct) / 100.0)

    if entry_row is None:
        entry_status = "Entry next session pending"
        entry_date = ""
    elif pd.Timestamp(entry_row["date"]).normalize() > as_of_ts:
        entry_status = "Entry next session pending"
        entry_date = pd.Timestamp(entry_row["date"]).strftime("%Y-%m-%d")
    else:
        entry_status = "Entry already triggered"
        entry_date = pd.Timestamp(entry_row["date"]).strftime("%Y-%m-%d")

    row = {
        "exchange": exchange,
        "symbol": symbol,
        "name": name,
        "source_universe": source_universe,
        "candidate_rank": np.nan,
        "candidate_score": context["candidate_score"],
        "strategy_tier": _strategy_tier(context),
        "strategy_tier_rank": _strategy_tier_rank(context),
        "signal_date": _date_text(frame.iloc[reclaim_index].get("date")),
        "break_below_75dma_date": _date_text(frame.iloc[int(event["break_index"])].get("date")),
        "sessions_to_reclaim": int(event["sessions_to_reclaim"]),
        "sessions_since_reclaim": int(as_of_index - reclaim_index),
        "entry_status": entry_status,
        "entry_date": entry_date,
        "planned_entry_price": np.nan if entry_price is None else float(entry_price),
        "planned_stop_price": stop_price,
        "planned_target_price": target_price,
        "as_of_date": _date_text(frame.iloc[as_of_index].get("date")),
        "as_of_close": _finite_float(frame.iloc[as_of_index].get("close")),
        **context,
    }
    return row


def _backtest_symbol(
    frame: pd.DataFrame,
    pivots: pd.DataFrame,
    events: list[dict[str, int]],
    *,
    exchange: str,
    symbol: str,
    name: str,
    source_universe: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    holding_sessions: int,
    profit_target_pct: float,
    stop_loss_pct: float,
    stop_buffer_pct: float,
    round_trip_cost_pct: float,
    entry_price_mode: str,
    exit_on_close_below_75dma: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    next_allowed_entry: dict[str, int] = {strategy: 0 for strategy, _ in STRATEGY_VARIANTS}
    for event in events:
        reclaim_index = int(event["reclaim_index"])
        signal_date = pd.Timestamp(frame.iloc[reclaim_index]["date"]).normalize()
        if signal_date < start_ts or signal_date > end_ts:
            continue
        context = _signal_context(frame, pivots, event)
        entry_index = reclaim_index + 1
        if entry_index >= len(frame):
            continue
        if pd.Timestamp(frame.iloc[entry_index]["date"]).normalize() > end_ts:
            continue
        for strategy_name, pass_column in STRATEGY_VARIANTS:
            if not bool(context.get(pass_column, False)):
                continue
            if entry_index < next_allowed_entry[strategy_name]:
                continue
            trade = _simulate_trade(
                frame,
                event=event,
                entry_index=entry_index,
                holding_sessions=holding_sessions,
                profit_target_pct=profit_target_pct,
                stop_loss_pct=stop_loss_pct,
                stop_buffer_pct=stop_buffer_pct,
                round_trip_cost_pct=round_trip_cost_pct,
                entry_price_mode=entry_price_mode,
                exit_on_close_below_75dma=exit_on_close_below_75dma,
                end_ts=end_ts,
            )
            if trade is None:
                continue
            rows.append(
                {
                    "exchange": exchange,
                    "symbol": symbol,
                    "name": name,
                    "source_universe": source_universe,
                    "strategy": strategy_name,
                    **trade,
                    "candidate_score": context.get("candidate_score"),
                    "dma_trend_20d_label": context.get("dma_trend_20d_label"),
                    "dma_stack": context.get("dma_stack"),
                    "last_low_pivot_structure": context.get("last_low_pivot_structure"),
                    "pullback_context": context.get("pullback_context"),
                    "distance_75dma_pct": context.get("distance_75dma_pct"),
                    "distance_100dma_pct": context.get("distance_100dma_pct"),
                    "volume_ratio_50": context.get("volume_ratio_50"),
                }
            )
            next_allowed_entry[strategy_name] = int(trade["exit_index"]) + 1
    return pd.DataFrame(rows)


def _simulate_trade(
    frame: pd.DataFrame,
    *,
    event: dict[str, int],
    entry_index: int,
    holding_sessions: int,
    profit_target_pct: float,
    stop_loss_pct: float,
    stop_buffer_pct: float,
    round_trip_cost_pct: float,
    entry_price_mode: str,
    exit_on_close_below_75dma: bool,
    end_ts: pd.Timestamp,
) -> dict[str, Any] | None:
    entry_row = frame.iloc[entry_index]
    entry_price = _entry_price(entry_row, entry_price_mode)
    if entry_price is None or entry_price <= 0.0:
        return None
    planned_exit_index = min(len(frame) - 1, entry_index + max(int(holding_sessions), 1))
    while planned_exit_index > entry_index and pd.Timestamp(frame.iloc[planned_exit_index]["date"]).normalize() > end_ts:
        planned_exit_index -= 1
    if planned_exit_index <= entry_index:
        return None

    reclaim_row = frame.iloc[int(event["reclaim_index"])]
    stop_price = _planned_stop_price(
        float(entry_price),
        reclaim_row,
        stop_loss_pct=stop_loss_pct,
        stop_buffer_pct=stop_buffer_pct,
    )
    target_price = float(entry_price) * (1.0 + float(profit_target_pct) / 100.0)
    exit_index = planned_exit_index
    exit_price = _finite_float(frame.iloc[exit_index].get("close"))
    exit_reason = "MAX_HOLD"

    for idx in range(entry_index, planned_exit_index + 1):
        bar = frame.iloc[idx]
        bar_open = _finite_float(bar.get("open"))
        bar_high = _finite_float(bar.get("high"))
        bar_low = _finite_float(bar.get("low"))
        bar_close = _finite_float(bar.get("close"))
        if bar_open is None or bar_high is None or bar_low is None or bar_close is None:
            continue
        if bar_open <= stop_price:
            exit_index, exit_price, exit_reason = idx, bar_open, "STOP_GAP"
            break
        if bar_open >= target_price:
            exit_index, exit_price, exit_reason = idx, bar_open, "TARGET_GAP"
            break
        stop_hit = bar_low <= stop_price
        target_hit = bar_high >= target_price
        if stop_hit:
            exit_index, exit_price, exit_reason = idx, stop_price, "STOP"
            break
        if target_hit:
            exit_index, exit_price, exit_reason = idx, target_price, "TARGET"
            break
        if exit_on_close_below_75dma and idx > entry_index:
            dma_75 = _finite_float(bar.get("dma_75"))
            if dma_75 is not None and bar_close < dma_75:
                exit_index, exit_price, exit_reason = idx, bar_close, "CLOSE_BELOW_75DMA"
                break

    if exit_price is None:
        return None
    gross_return = (float(exit_price) / float(entry_price) - 1.0) * 100.0
    net_return = gross_return - float(round_trip_cost_pct)
    trade_window = frame.iloc[entry_index : exit_index + 1]
    mfe = (float(trade_window["high"].max()) / float(entry_price) - 1.0) * 100.0 if not trade_window.empty else np.nan
    mae = (float(trade_window["low"].min()) / float(entry_price) - 1.0) * 100.0 if not trade_window.empty else np.nan
    return {
        "signal_date": _date_text(frame.iloc[int(event["reclaim_index"])].get("date")),
        "break_below_75dma_date": _date_text(frame.iloc[int(event["break_index"])].get("date")),
        "sessions_to_reclaim": int(event["sessions_to_reclaim"]),
        "entry_date": _date_text(entry_row.get("date")),
        "entry_price": float(entry_price),
        "exit_date": _date_text(frame.iloc[exit_index].get("date")),
        "exit_price": float(exit_price),
        "exit_reason": exit_reason,
        "stop_price": float(stop_price),
        "target_price": float(target_price),
        "hold_trading_sessions": int(exit_index - entry_index),
        "hold_calendar_days": int((pd.Timestamp(frame.iloc[exit_index]["date"]).normalize() - pd.Timestamp(entry_row["date"]).normalize()).days),
        "gross_return_pct": gross_return,
        "net_return_pct": net_return,
        "win_flag": bool(net_return > 0.0),
        "mfe_pct": mfe,
        "mae_pct": mae,
        "entry_index": int(entry_index),
        "exit_index": int(exit_index),
    }


def _signal_context(frame: pd.DataFrame, pivots: pd.DataFrame, event: dict[str, int]) -> dict[str, Any]:
    break_index = int(event["break_index"])
    reclaim_index = int(event["reclaim_index"])
    break_row = frame.iloc[break_index]
    signal_row = frame.iloc[reclaim_index]
    close = float(signal_row["close"])
    dma_75 = _finite_float(signal_row.get("dma_75"))
    dma_100 = _finite_float(signal_row.get("dma_100"))
    distance_75 = _finite_float(signal_row.get("distance_75dma_pct"))
    distance_100 = _finite_float(signal_row.get("distance_100dma_pct"))
    break_distance_75 = _finite_float(break_row.get("distance_75dma_pct"))
    break_distance_100 = _finite_float(break_row.get("distance_100dma_pct"))
    reduction_75 = _distance_reduction(break_distance_75, distance_75)
    reduction_100 = _distance_reduction(break_distance_100, distance_100)
    dma_75_trend = _finite_float(signal_row.get("dma_75_trend_20d_pct"))
    dma_100_trend = _finite_float(signal_row.get("dma_100_trend_20d_pct"))
    dma_stack = _dma_stack_label(dma_75, dma_100)
    dma_trend_label = _dma_trend_label(dma_75_trend, dma_100_trend, dma_stack)
    all_dma_trend_positive = _all_dma_trend_positive(dma_75_trend, dma_100_trend)
    all_dma_trend_non_negative = _all_dma_trend_non_negative(dma_75_trend, dma_100_trend)
    all_dma_distance_reducing = _all_dma_distance_reducing(reduction_75, reduction_100)
    price_above_75dma = bool(dma_75 is not None and close > dma_75)
    price_above_100dma = bool(dma_100 is not None and close > dma_100)
    price_above_all_dma = bool(price_above_75dma and price_above_100dma)
    long_trend_aligned = bool(all_dma_trend_positive and dma_stack in {"75DMA above 100DMA", "75DMA equal 100DMA"})

    available_pivots = (
        pivots[pivots["available_index"] <= reclaim_index].copy()
        if not pivots.empty
        else pd.DataFrame()
    )
    low_pivots = available_pivots[available_pivots["type"].eq("LOW")].copy() if not available_pivots.empty else pd.DataFrame()
    high_pivots = available_pivots[available_pivots["type"].eq("HIGH")].copy() if not available_pivots.empty else pd.DataFrame()
    last_low = low_pivots.iloc[-1] if not low_pivots.empty else pd.Series(dtype="object")
    last_high = high_pivots.iloc[-1] if not high_pivots.empty else pd.Series(dtype="object")
    last_hl = low_pivots[low_pivots["structure"].eq("HL")].iloc[-1] if not low_pivots.empty and low_pivots["structure"].eq("HL").any() else pd.Series(dtype="object")
    last_hh = high_pivots[high_pivots["structure"].eq("HH")].iloc[-1] if not high_pivots.empty and high_pivots["structure"].eq("HH").any() else pd.Series(dtype="object")

    distance_to_hl = _pivot_distance_pct(close, last_hl)
    distance_to_hh = _pivot_distance_pct(close, last_hh)
    pullback = _pullback_summary(available_pivots, close, last_high)
    below_last_hl = bool(_finite_float(distance_to_hl) is not None and float(distance_to_hl) < 0.0)
    normal_pullback = _normal_pullback(pullback.get("current_pullback_vs_avg"))
    trend_pass = bool(dma_trend_label in {"Bullish", "Improving"} and all_dma_trend_non_negative)
    strict_pass = bool(
        long_trend_aligned
        and not below_last_hl
        and (normal_pullback or str(pullback.get("pullback_context", "")) == "At or above last high")
        and (all_dma_distance_reducing or (_finite_float(reduction_75) is not None and float(reduction_75) > 0.0))
    )
    candidate_score = _candidate_score(
        dma_trend_label=dma_trend_label,
        long_trend_aligned=long_trend_aligned,
        all_dma_trend_positive=all_dma_trend_positive,
        all_dma_distance_reducing=all_dma_distance_reducing,
        price_above_all_dma=price_above_all_dma,
        reduction_75=reduction_75,
        reduction_100=reduction_100,
        distance_75=distance_75,
        distance_100=distance_100,
        last_low_structure=str(last_low.get("structure", "")) if not last_low.empty else "",
        below_last_hl=below_last_hl,
        pullback_ratio=pullback.get("current_pullback_vs_avg"),
        volume_ratio=signal_row.get("volume_ratio_50"),
    )

    return {
        "raw_pass": True,
        "trend_pass": trend_pass,
        "strict_pass": strict_pass,
        "candidate_score": candidate_score,
        "entry_bias": _entry_bias(strict_pass, trend_pass, dma_trend_label, below_last_hl),
        "latest_close": close,
        "dma_75": dma_75,
        "dma_100": dma_100,
        "distance_75dma_pct": distance_75,
        "distance_100dma_pct": distance_100,
        "break_distance_75dma_pct": break_distance_75,
        "break_distance_100dma_pct": break_distance_100,
        "distance_reduction_75dma_pct_points": reduction_75,
        "distance_reduction_100dma_pct_points": reduction_100,
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
        "volume_ratio_50": _finite_float(signal_row.get("volume_ratio_50")),
        "last_low_pivot_structure": "" if last_low.empty else str(last_low.get("structure", "")),
        "last_low_pivot_date": "" if last_low.empty else _date_text(last_low.get("date")),
        "last_low_pivot_price": np.nan if last_low.empty else _finite_float(last_low.get("price")),
        "last_hl_pivot_date": "" if last_hl.empty else _date_text(last_hl.get("date")),
        "last_hl_pivot_price": np.nan if last_hl.empty else _finite_float(last_hl.get("price")),
        "distance_to_last_hl_pct": distance_to_hl,
        "last_high_pivot_structure": "" if last_high.empty else str(last_high.get("structure", "")),
        "last_high_pivot_date": "" if last_high.empty else _date_text(last_high.get("date")),
        "last_hh_pivot_date": "" if last_hh.empty else _date_text(last_hh.get("date")),
        "distance_to_last_hh_pct": distance_to_hh,
        **pullback,
    }


def _confirmed_pivots(frame: pd.DataFrame, *, order: int) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    order = max(int(order), 1)
    rows: list[dict[str, Any]] = []
    highs = pd.to_numeric(frame["high"], errors="coerce")
    lows = pd.to_numeric(frame["low"], errors="coerce")
    for center in range(order, len(frame) - order):
        high = highs.iloc[center]
        low = lows.iloc[center]
        if pd.notna(high):
            left = highs.iloc[center - order : center]
            right = highs.iloc[center + 1 : center + order + 1]
            if left.notna().all() and right.notna().all() and high >= left.max() and high > right.max():
                rows.append(
                    {
                        "type": "HIGH",
                        "pivot_index": int(center),
                        "available_index": int(center + order),
                        "date": frame.iloc[center]["date"],
                        "price": float(high),
                    }
                )
        if pd.notna(low):
            left = lows.iloc[center - order : center]
            right = lows.iloc[center + 1 : center + order + 1]
            if left.notna().all() and right.notna().all() and low <= left.min() and low < right.min():
                rows.append(
                    {
                        "type": "LOW",
                        "pivot_index": int(center),
                        "available_index": int(center + order),
                        "date": frame.iloc[center]["date"],
                        "price": float(low),
                    }
                )
    pivots = pd.DataFrame(rows)
    if pivots.empty:
        return pivots
    pivots = pivots.sort_values(["available_index", "pivot_index", "type"]).reset_index(drop=True)
    structures: list[str] = []
    last_high: float | None = None
    last_low: float | None = None
    for _, row in pivots.iterrows():
        price = float(row["price"])
        if row["type"] == "HIGH":
            structure = "H" if last_high is None else ("HH" if price > last_high else "LH")
            last_high = price
        else:
            structure = "L" if last_low is None else ("HL" if price > last_low else "LL")
            last_low = price
        structures.append(structure)
    pivots["structure"] = structures
    return pivots


def _candidate_score(
    *,
    dma_trend_label: str,
    long_trend_aligned: bool,
    all_dma_trend_positive: bool,
    all_dma_distance_reducing: bool,
    price_above_all_dma: bool,
    reduction_75: Any,
    reduction_100: Any,
    distance_75: Any,
    distance_100: Any,
    last_low_structure: str,
    below_last_hl: bool,
    pullback_ratio: Any,
    volume_ratio: Any,
) -> float:
    label = str(dma_trend_label or "").strip()
    score = 0.0
    if long_trend_aligned:
        score += 30.0
    elif label == "Bullish":
        score += 26.0
    elif label == "Improving":
        score += 22.0
    elif label == "Mixed":
        score += 10.0
    elif label == "Weakening":
        score += 3.0

    best_reduction = max(_finite_float(reduction_75) or 0.0, _finite_float(reduction_100) or 0.0)
    score += min(max(best_reduction, 0.0), 12.0) / 12.0 * 14.0
    if all_dma_distance_reducing:
        score += 8.0
    nearest_gap_values = [abs(value) for value in (_finite_float(distance_75), _finite_float(distance_100)) if value is not None]
    if nearest_gap_values:
        score += (8.0 - min(min(nearest_gap_values), 8.0)) / 8.0 * 8.0
    if price_above_all_dma:
        score += 8.0
    if all_dma_trend_positive:
        score += 8.0
    structure = str(last_low_structure or "").upper()
    if structure == "HL":
        score += 10.0
    elif structure == "LL":
        score += 3.0
    if below_last_hl:
        score -= 10.0
    ratio = _finite_float(pullback_ratio)
    if ratio is not None:
        if 0.50 <= ratio <= 1.25:
            score += 10.0
        elif 0.25 <= ratio <= 1.75:
            score += 7.0
        elif ratio > 1.75:
            score += 2.0
    vol_ratio = _finite_float(volume_ratio)
    if vol_ratio is not None:
        if vol_ratio >= 2.0:
            score += 4.0
        elif vol_ratio >= 1.2:
            score += 2.0
    cap = 100.0
    if label == "Bearish":
        cap = 55.0
    elif label == "Weakening":
        cap = 62.0
    elif label == "Mixed":
        cap = 75.0
    elif label == "Improving":
        cap = 90.0
    return round(float(max(0.0, min(score, cap))), 2)


def _strategy_tier(context: dict[str, Any]) -> str:
    if context.get("strict_pass"):
        return "Strict Long"
    if context.get("trend_pass"):
        return "Trend Filtered"
    return "Raw Reclaim"


def _strategy_tier_rank(context: dict[str, Any]) -> int:
    if context.get("strict_pass"):
        return 1
    if context.get("trend_pass"):
        return 2
    return 3


def _entry_bias(strict_pass: bool, trend_pass: bool, dma_trend_label: str, below_last_hl: bool) -> str:
    if strict_pass:
        return "Long candidate"
    if trend_pass:
        return "Long watch"
    if below_last_hl:
        return "Avoid: below last HL"
    if str(dma_trend_label or "") in {"Bearish", "Weakening"}:
        return "Avoid: DMA trend weak"
    return "Watch only"


def _planned_stop_price(entry_price: float, reclaim_row: pd.Series, *, stop_loss_pct: float, stop_buffer_pct: float) -> float:
    fixed_stop = float(entry_price) * (1.0 - float(stop_loss_pct) / 100.0)
    reclaim_low = _finite_float(reclaim_row.get("low"))
    if reclaim_low is None:
        return fixed_stop
    reclaim_stop = float(reclaim_low) * (1.0 - float(stop_buffer_pct) / 100.0)
    return max(fixed_stop, reclaim_stop)


def _entry_price(row: pd.Series | None, entry_price_mode: str) -> float | None:
    if row is None:
        return None
    column = "close" if str(entry_price_mode or "").lower() == "next_close" else "open"
    return _finite_float(row.get(column))


def _aggregate_stats(trades: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=[*group_columns, "trades", "wins", "losses", "win_rate_pct", "avg_return_pct"])
    rows: list[dict[str, Any]] = []
    group_key: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for key, group in trades.groupby(group_key, dropna=False):
        keys = (key,) if len(group_columns) == 1 else tuple(key)
        returns = pd.to_numeric(group["net_return_pct"], errors="coerce").dropna()
        wins = returns[returns > 0.0]
        losses = returns[returns <= 0.0]
        record = dict(zip(group_columns, keys, strict=False))
        record.update(
            {
                "trades": int(len(returns)),
                "wins": int(len(wins)),
                "losses": int(len(losses)),
                "win_rate_pct": float(len(wins) / len(returns) * 100.0) if len(returns) else 0.0,
                "avg_return_pct": float(returns.mean()) if len(returns) else 0.0,
                "median_return_pct": float(returns.median()) if len(returns) else 0.0,
                "avg_win_pct": float(wins.mean()) if len(wins) else 0.0,
                "avg_loss_pct": float(losses.mean()) if len(losses) else 0.0,
                "payoff_ratio": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) and losses.mean() != 0 else np.nan,
                "profit_factor": float(wins.sum() / abs(losses.sum())) if len(wins) and len(losses) and losses.sum() != 0 else np.nan,
                "avg_hold_sessions": float(pd.to_numeric(group["hold_trading_sessions"], errors="coerce").mean()),
                "avg_mfe_pct": float(pd.to_numeric(group["mfe_pct"], errors="coerce").mean()),
                "avg_mae_pct": float(pd.to_numeric(group["mae_pct"], errors="coerce").mean()),
            }
        )
        rows.append(record)
    result = pd.DataFrame(rows)
    sort_columns = [column for column in ("profit_factor", "win_rate_pct", "avg_return_pct", "trades") if column in result.columns]
    return result.sort_values(sort_columns, ascending=[False] * len(sort_columns), na_position="last").reset_index(drop=True)


def _aggregate_yearly(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    frame = trades.copy()
    frame["year"] = pd.to_datetime(frame["entry_date"], errors="coerce").dt.year
    return _aggregate_stats(frame.dropna(subset=["year"]), ["strategy", "year"])


def _build_summary(
    candidates: pd.DataFrame,
    trades: pd.DataFrame,
    strategy_stats: pd.DataFrame,
    coverage: pd.DataFrame,
    *,
    universe_name: str,
    start_ts: pd.Timestamp,
    as_of_ts: pd.Timestamp,
    total_symbols: int,
    skipped_no_data: int,
    skipped_short: int,
    skipped_stale: int,
    warmup_months: int,
    max_reclaim_sessions: int,
    current_signal_lookback_sessions: int,
    holding_sessions: int,
    profit_target_pct: float,
    stop_loss_pct: float,
    stop_buffer_pct: float,
    round_trip_cost_pct: float,
    entry_price_mode: str,
    exit_on_close_below_75dma: bool,
    pivot_order: int,
) -> dict[str, Any]:
    best = strategy_stats.iloc[0].to_dict() if not strategy_stats.empty else {}
    strict_candidates = (
        candidates[candidates["strategy_tier"].eq("Strict Long")]
        if not candidates.empty and "strategy_tier" in candidates.columns
        else pd.DataFrame()
    )
    trend_candidates = (
        candidates[candidates["strategy_tier"].isin(["Strict Long", "Trend Filtered"])]
        if not candidates.empty and "strategy_tier" in candidates.columns
        else pd.DataFrame()
    )
    return {
        "universe": universe_name,
        "requested_start_date": start_ts.strftime("%Y-%m-%d"),
        "requested_as_of_date": as_of_ts.strftime("%Y-%m-%d"),
        "symbols_requested": int(total_symbols),
        "symbols_with_history": int(len(coverage)),
        "symbols_no_data": int(skipped_no_data),
        "symbols_short_history": int(skipped_short),
        "symbols_stale": int(skipped_stale),
        "current_candidates": int(len(candidates)),
        "current_strict_long_candidates": int(len(strict_candidates)),
        "current_trend_candidates": int(len(trend_candidates)),
        "total_trades": int(len(trades)),
        "best_strategy": best.get("strategy", ""),
        "best_strategy_trades": int(best.get("trades", 0) or 0),
        "best_strategy_win_rate_pct": best.get("win_rate_pct", np.nan),
        "best_strategy_avg_return_pct": best.get("avg_return_pct", np.nan),
        "best_strategy_profit_factor": best.get("profit_factor", np.nan),
        "earliest_history_date": coverage["history_start"].min().strftime("%Y-%m-%d") if not coverage.empty else "",
        "latest_history_date": coverage["history_end"].max().strftime("%Y-%m-%d") if not coverage.empty else "",
        "warmup_months": int(warmup_months),
        "max_reclaim_sessions": int(max_reclaim_sessions),
        "current_signal_lookback_sessions": int(current_signal_lookback_sessions),
        "holding_sessions": int(holding_sessions),
        "profit_target_pct": float(profit_target_pct),
        "stop_loss_pct": float(stop_loss_pct),
        "stop_buffer_pct": float(stop_buffer_pct),
        "round_trip_cost_pct": float(round_trip_cost_pct),
        "entry_price_mode": entry_price_mode,
        "exit_on_close_below_75dma": bool(exit_on_close_below_75dma),
        "pivot_order": int(pivot_order),
        "generated_at_ist": pd.Timestamp.now(tz="Asia/Kolkata").strftime("%Y-%m-%d %H:%M:%S IST"),
    }


def _dma_stack_label(dma_75: Any, dma_100: Any) -> str:
    dma75 = _finite_float(dma_75)
    dma100 = _finite_float(dma_100)
    if dma75 is None or dma100 is None:
        return "Unknown"
    if dma75 > dma100:
        return "75DMA above 100DMA"
    if dma75 < dma100:
        return "75DMA below 100DMA"
    return "75DMA equal 100DMA"


def _dma_trend_label(trend_75: Any, trend_100: Any, dma_stack: str) -> str:
    values = [_finite_float(trend_75), _finite_float(trend_100)]
    if any(value is None for value in values):
        return "Unknown"
    rising = [float(value) > DMA_TREND_FLAT_THRESHOLD_PCT for value in values if value is not None]
    falling = [float(value) < -DMA_TREND_FLAT_THRESHOLD_PCT for value in values if value is not None]
    if sum(falling) == len(values):
        return "Bearish"
    if sum(rising) == len(values):
        return "Bullish" if dma_stack != "75DMA below 100DMA" else "Improving"
    if any(rising) and not any(falling):
        return "Improving"
    if any(falling) and not any(rising):
        return "Weakening"
    return "Mixed"


def _all_dma_trend_positive(trend_75: Any, trend_100: Any) -> bool:
    values = [_finite_float(trend_75), _finite_float(trend_100)]
    return bool(all(value is not None and value > DMA_TREND_FLAT_THRESHOLD_PCT for value in values))


def _all_dma_trend_non_negative(trend_75: Any, trend_100: Any) -> bool:
    values = [_finite_float(trend_75), _finite_float(trend_100)]
    return bool(all(value is not None and value >= -DMA_TREND_FLAT_THRESHOLD_PCT for value in values))


def _all_dma_distance_reducing(reduction_75: Any, reduction_100: Any) -> bool:
    values = [_finite_float(reduction_75), _finite_float(reduction_100)]
    return bool(all(value is not None and value > 0.0 for value in values))


def _distance_reduction(start_distance: Any, end_distance: Any) -> float:
    start = _finite_float(start_distance)
    end = _finite_float(end_distance)
    if start is None or end is None:
        return np.nan
    return abs(start) - abs(end)


def _pivot_distance_pct(close: float, pivot: pd.Series) -> float:
    if pivot.empty:
        return np.nan
    price = _finite_float(pivot.get("price"))
    if price is None or price <= 0.0:
        return np.nan
    return (float(close) / price - 1.0) * 100.0


def _pullback_summary(pivots: pd.DataFrame, close: float, last_high: pd.Series) -> dict[str, Any]:
    pullbacks: list[float] = []
    if not pivots.empty:
        ordered = pivots.sort_values(["pivot_index", "available_index"]).reset_index(drop=True)
        for index in range(len(ordered) - 1):
            high = ordered.iloc[index]
            low = ordered.iloc[index + 1]
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
        high_price = _finite_float(last_high.get("price"))
        if high_price is not None and high_price > 0.0:
            current_pullback = (float(close) / high_price - 1.0) * 100.0
            current_pullback_abs = max(0.0, -current_pullback)
            if np.isfinite(average_pullback) and average_pullback > 0.0:
                current_pullback_vs_avg = current_pullback_abs / average_pullback
    return {
        "pullback_events": int(len(pullbacks)),
        "average_pullback_pct": average_pullback,
        "median_pullback_pct": median_pullback,
        "current_pullback_from_last_high_pct": current_pullback,
        "current_pullback_abs_pct": current_pullback_abs,
        "current_pullback_vs_avg": current_pullback_vs_avg,
        "pullback_context": _pullback_context(current_pullback, current_pullback_vs_avg),
    }


def _normal_pullback(value: Any) -> bool:
    ratio = _finite_float(value)
    return bool(ratio is not None and 0.25 <= ratio <= 1.75)


def _pullback_context(current_pullback: Any, pullback_ratio: Any) -> str:
    pullback = _finite_float(current_pullback)
    ratio = _finite_float(pullback_ratio)
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


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "exchange",
            "symbol",
            "name",
            "source_universe",
            "strategy",
            "signal_date",
            "entry_date",
            "exit_date",
            "entry_price",
            "exit_price",
            "exit_reason",
            "net_return_pct",
        ]
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


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
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _emit_progress(progress_callback: Callable[[dict[str, Any]], None] | None, **payload: Any) -> None:
    if progress_callback is not None:
        progress_callback(payload)
