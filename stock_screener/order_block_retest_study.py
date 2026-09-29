# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Price-logic adaptation of "Order Blocks Volume Delta 3D | Flux Charts"
# by fluxchart. The chart drawing code is not included.

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from stock_screener.data.storage import Storage
from stock_screener.weekly_buy_tracker_study import _emit_progress, _load_name_map


DEFAULT_SWING_LENGTH = 5
DEFAULT_POC_BINS = 40
DEFAULT_RECENT_SIGNAL_BARS = 5
DEFAULT_HOLDING_SESSIONS = 20
DEFAULT_PROFIT_TARGET_PCT = 10.0
DEFAULT_MAX_STOP_LOSS_PCT = 5.0
DEFAULT_STOP_BUFFER_PCT = 0.25
DEFAULT_ROUND_TRIP_COST_PCT = 0.20
DEFAULT_MIN_HISTORICAL_SAMPLES = 0
DEFAULT_INVALIDATION_METHOD = "wick"
DEFAULT_SIGNAL_DIRECTION = "bullish"
ORDER_BLOCK_LOGIC_VERSION = "flux_poc_order_block_retest_daily_v2"
ORDER_BLOCK_COMPATIBLE_LOGIC_VERSIONS = {
    "flux_poc_order_block_retest_daily_v1",
    ORDER_BLOCK_LOGIC_VERSION,
}

INVALIDATION_METHODS = {"wick", "close"}
SIGNAL_DIRECTIONS = {"bullish", "bearish", "either"}
MAX_STORED_ORDER_BLOCKS = 50
RETEST_SPACING_BARS = 4


@dataclass
class _OrderBlock:
    block_id: int
    left_index: int
    created_index: int
    top: float
    bottom: float
    is_bull: bool
    bull_volume: float
    bear_volume: float
    total_volume: float
    bull_pct: float
    bear_pct: float
    active: bool = True
    invalid_index: int | None = None


@dataclass(frozen=True)
class OrderBlockRetestStudyResult:
    summary: dict[str, Any]
    candidates: pd.DataFrame
    backtest_stats: pd.DataFrame
    trades: pd.DataFrame
    events: pd.DataFrame


def calculate_order_block_events(
    daily: pd.DataFrame,
    *,
    swing_length: int = DEFAULT_SWING_LENGTH,
    poc_bins: int = DEFAULT_POC_BINS,
    invalidation_method: str = DEFAULT_INVALIDATION_METHOD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Translate the Pine order-block and retest events to completed daily bars."""
    frame = _prepare_daily(daily)
    if frame.empty:
        return frame, _empty_events()

    swing_length = max(int(swing_length), 1)
    poc_bins = max(int(poc_bins), 2)
    invalidation_method = str(invalidation_method or DEFAULT_INVALIDATION_METHOD).lower()
    if invalidation_method not in INVALIDATION_METHODS:
        invalidation_method = DEFAULT_INVALIDATION_METHOD

    open_values = frame["open"].to_numpy(dtype=float)
    high_values = frame["high"].to_numpy(dtype=float)
    low_values = frame["low"].to_numpy(dtype=float)
    close_values = frame["close"].to_numpy(dtype=float)
    volume_values = frame["volume"].fillna(0.0).to_numpy(dtype=float)
    dates = frame["date"].tolist()

    blocks: list[_OrderBlock] = []
    all_blocks: dict[int, _OrderBlock] = {}
    event_rows: list[dict[str, Any]] = []
    next_block_id = 1

    swing_high_price: float | None = None
    swing_high_index: int | None = None
    swing_low_price: float | None = None
    swing_low_index: int | None = None
    previous_bull_bos_now = False
    previous_bear_bos_now = False
    last_bull_retest_bar: int | None = None
    last_bear_retest_bar: int | None = None

    for index in range(len(frame)):
        previous_swing_high_index = swing_high_index
        previous_swing_low_index = swing_low_index

        pivot_center = index - swing_length
        pivot_start = index - (2 * swing_length)
        if pivot_start >= 0:
            high_window = high_values[pivot_start : index + 1]
            low_window = low_values[pivot_start : index + 1]
            center_high = high_values[pivot_center]
            center_low = low_values[pivot_center]
            if np.isfinite(center_high) and center_high == np.nanmax(high_window):
                swing_high_price = float(center_high)
                swing_high_index = int(pivot_center)
            if np.isfinite(center_low) and center_low == np.nanmin(low_window):
                swing_low_price = float(center_low)
                swing_low_index = int(pivot_center)

        previous_close = close_values[index - 1] if index > 0 else np.nan
        bear_bos_now = bool(
            swing_low_price is not None
            and swing_low_index is not None
            and index > swing_low_index
            and close_values[index] < swing_low_price
            and np.isfinite(previous_close)
            and previous_close >= swing_low_price
        )
        bull_bos_now = bool(
            swing_high_price is not None
            and swing_high_index is not None
            and index > swing_high_index
            and close_values[index] > swing_high_price
            and np.isfinite(previous_close)
            and previous_close <= swing_high_price
        )

        bear_bos = previous_bear_bos_now
        bull_bos = previous_bull_bos_now
        bos_index = index - 1

        if bear_bos and previous_swing_low_index is not None and bos_index >= previous_swing_low_index:
            poc_stats = _poc_volume_stats(
                open_values,
                high_values,
                low_values,
                close_values,
                volume_values,
                previous_swing_low_index,
                bos_index,
                poc_bins,
            )
            anchor_index = _bearish_anchor_index(
                high_values,
                low_values,
                previous_swing_low_index,
                bos_index,
                poc_stats[0],
            )
            if (
                anchor_index is not None
                and poc_stats[1] > 0
                and not _has_gap(high_values, low_values, anchor_index, bos_index, is_bull=False)
            ):
                top = float(high_values[anchor_index])
                bottom = float(low_values[anchor_index])
                if not _overlaps_active(blocks, top, bottom):
                    block = _new_block(
                        next_block_id,
                        anchor_index,
                        bos_index,
                        top,
                        bottom,
                        False,
                        poc_stats,
                    )
                    next_block_id += 1
                    blocks.insert(0, block)
                    all_blocks[block.block_id] = block
                    event_rows.append(
                        _event_row(
                            block,
                            "NEW_BEARISH_OB",
                            index,
                            bos_index,
                            dates,
                            close_values,
                        )
                    )
                    _prune_blocks(blocks, index)
            swing_low_price = None
            swing_low_index = None

        if bull_bos and previous_swing_high_index is not None and bos_index >= previous_swing_high_index:
            poc_stats = _poc_volume_stats(
                open_values,
                high_values,
                low_values,
                close_values,
                volume_values,
                previous_swing_high_index,
                bos_index,
                poc_bins,
            )
            anchor_index = _bullish_anchor_index(
                high_values,
                low_values,
                previous_swing_high_index,
                bos_index,
                poc_stats[0],
            )
            if (
                anchor_index is not None
                and poc_stats[1] > 0
                and not _has_gap(high_values, low_values, anchor_index, bos_index, is_bull=True)
            ):
                top = float(high_values[anchor_index])
                bottom = float(low_values[anchor_index])
                if not _overlaps_active(blocks, top, bottom):
                    block = _new_block(
                        next_block_id,
                        anchor_index,
                        bos_index,
                        top,
                        bottom,
                        True,
                        poc_stats,
                    )
                    next_block_id += 1
                    blocks.insert(0, block)
                    all_blocks[block.block_id] = block
                    event_rows.append(
                        _event_row(
                            block,
                            "NEW_BULLISH_OB",
                            index,
                            bos_index,
                            dates,
                            close_values,
                        )
                    )
                    _prune_blocks(blocks, index)
            swing_high_price = None
            swing_high_index = None

        for block in blocks:
            if not block.active:
                continue

            if invalidation_method == "wick":
                invalid = bool(
                    low_values[index] < block.bottom
                    if block.is_bull
                    else high_values[index] > block.top
                )
                invalid_index = index
            else:
                invalid = bool(
                    index > 0
                    and (
                        close_values[index - 1] < block.bottom
                        if block.is_bull
                        else close_values[index - 1] > block.top
                    )
                )
                invalid_index = index - 1
            if invalid:
                block.active = False
                block.invalid_index = invalid_index

            if index <= 0:
                continue
            if block.is_bull:
                retest = bool(
                    open_values[index - 1] > block.top
                    and close_values[index - 1] > block.top
                    and low_values[index - 1] <= block.top
                    and low_values[index - 1] >= block.bottom
                )
                last_side_bar = last_bull_retest_bar
            else:
                retest = bool(
                    open_values[index - 1] < block.bottom
                    and close_values[index - 1] < block.bottom
                    and high_values[index - 1] >= block.bottom
                    and high_values[index - 1] <= block.top
                )
                last_side_bar = last_bear_retest_bar

            retest_index = index - 1
            can_log = last_side_bar is None or retest_index - last_side_bar >= RETEST_SPACING_BARS
            if retest and retest_index > block.created_index and can_log:
                event_rows.append(
                    _event_row(
                        block,
                        "BULLISH_RETEST" if block.is_bull else "BEARISH_RETEST",
                        index,
                        retest_index,
                        dates,
                        close_values,
                    )
                )
                if block.is_bull:
                    last_bull_retest_bar = retest_index
                else:
                    last_bear_retest_bar = retest_index

        previous_bear_bos_now = bear_bos_now
        previous_bull_bos_now = bull_bos_now

    events = pd.DataFrame(event_rows) if event_rows else _empty_events()
    if not events.empty:
        events["active"] = events["block_id"].map(
            {block_id: block.active for block_id, block in all_blocks.items()}
        ).fillna(False)
        events["invalidation_date"] = events["block_id"].map(
            {
                block_id: (
                    dates[block.invalid_index]
                    if block.invalid_index is not None and 0 <= block.invalid_index < len(dates)
                    else pd.NaT
                )
                for block_id, block in all_blocks.items()
            }
        )
        events = events.sort_values(["event_index", "block_id"]).reset_index(drop=True)
    return frame, events


def run_order_block_retest_study(
    storage: Storage,
    *,
    exchange: str = "NSE",
    symbols: list[str] | None = None,
    start_date: date | str | pd.Timestamp | None = None,
    as_of_date: date | str | pd.Timestamp | None = None,
    swing_length: int = DEFAULT_SWING_LENGTH,
    poc_bins: int = DEFAULT_POC_BINS,
    invalidation_method: str = DEFAULT_INVALIDATION_METHOD,
    signal_direction: str = DEFAULT_SIGNAL_DIRECTION,
    recent_signal_bars: int = DEFAULT_RECENT_SIGNAL_BARS,
    require_active_zone: bool = True,
    holding_sessions: int = DEFAULT_HOLDING_SESSIONS,
    profit_target_pct: float = DEFAULT_PROFIT_TARGET_PCT,
    max_stop_loss_pct: float = DEFAULT_MAX_STOP_LOSS_PCT,
    stop_buffer_pct: float = DEFAULT_STOP_BUFFER_PCT,
    round_trip_cost_pct: float = DEFAULT_ROUND_TRIP_COST_PCT,
    min_historical_samples: int = DEFAULT_MIN_HISTORICAL_SAMPLES,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> OrderBlockRetestStudyResult:
    exchange = str(exchange or "NSE").upper()
    signal_direction = str(signal_direction or DEFAULT_SIGNAL_DIRECTION).lower()
    if signal_direction not in SIGNAL_DIRECTIONS:
        signal_direction = DEFAULT_SIGNAL_DIRECTION
    invalidation_method = str(invalidation_method or DEFAULT_INVALIDATION_METHOD).lower()
    if invalidation_method not in INVALIDATION_METHODS:
        invalidation_method = DEFAULT_INVALIDATION_METHOD

    start_ts = _coerce_date(start_date)
    as_of_ts = _coerce_date(as_of_date)
    if start_ts is None:
        start_ts = pd.Timestamp.today().normalize() - pd.DateOffset(years=5)
    if as_of_ts is None:
        as_of_ts = pd.Timestamp.today().normalize()
    if start_ts > as_of_ts:
        raise ValueError("Backtest start date cannot be after the as-of date.")

    swing_length = max(int(swing_length), 1)
    poc_bins = max(int(poc_bins), 2)
    recent_signal_bars = max(int(recent_signal_bars), 1)
    holding_sessions = max(int(holding_sessions), 1)
    profit_target_pct = max(float(profit_target_pct), 0.01)
    max_stop_loss_pct = max(float(max_stop_loss_pct), 0.01)
    stop_buffer_pct = max(float(stop_buffer_pct), 0.0)
    round_trip_cost_pct = max(float(round_trip_cost_pct), 0.0)
    min_historical_samples = max(int(min_historical_samples), 0)

    if symbols is None:
        candidates = sorted(
            path.stem
            for path in (storage.data_root / "candles" / exchange / "1D").glob("*.csv")
            if not _is_excluded_symbol(path.stem)
        )
    else:
        candidates = sorted(
            {
                str(symbol or "").strip().upper()
                for symbol in symbols
                if str(symbol or "").strip() and not _is_excluded_symbol(str(symbol))
            }
        )

    name_map = _load_name_map(storage, exchange)
    candidate_rows: list[dict[str, Any]] = []
    trade_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    symbols_with_history = 0
    min_history_rows = max((2 * swing_length) + 5, 20)

    _emit_progress(
        progress_callback,
        phase="Scanning order-block retests and backtesting",
        completed=0,
        total=len(candidates),
        current_symbol="",
        current_exchange=exchange,
    )
    for completed, symbol in enumerate(candidates, start=1):
        _emit_progress(
            progress_callback,
            phase="Scanning order-block retests and backtesting",
            completed=completed,
            total=len(candidates),
            current_symbol=symbol,
            current_exchange=exchange,
        )
        daily = storage.load_candles(exchange, symbol, "1D")
        if daily.empty:
            continue
        prepared = _prepare_daily(daily)
        prepared = prepared[prepared["date"].dt.normalize() <= as_of_ts].reset_index(drop=True)
        if len(prepared) < min_history_rows:
            continue
        symbols_with_history += 1
        frame, events = calculate_order_block_events(
            prepared,
            swing_length=swing_length,
            poc_bins=poc_bins,
            invalidation_method=invalidation_method,
        )
        if events.empty:
            continue

        events = events.copy()
        events.insert(0, "exchange", exchange)
        events.insert(1, "symbol", symbol)
        events.insert(2, "name", name_map.get(symbol, symbol))
        event_frames.append(events)

        trades = _backtest_symbol(
            frame,
            events,
            start_ts=start_ts,
            end_ts=as_of_ts,
            signal_direction=signal_direction,
            holding_sessions=holding_sessions,
            profit_target_pct=profit_target_pct,
            max_stop_loss_pct=max_stop_loss_pct,
            stop_buffer_pct=stop_buffer_pct,
            round_trip_cost_pct=round_trip_cost_pct,
        )
        if not trades.empty:
            trade_frames.append(trades)

        latest_candidate = _latest_candidate(
            frame,
            events,
            trades,
            exchange=exchange,
            symbol=symbol,
            name=name_map.get(symbol, symbol),
            signal_direction=signal_direction,
            recent_signal_bars=recent_signal_bars,
            require_active_zone=require_active_zone,
            profit_target_pct=profit_target_pct,
            max_stop_loss_pct=max_stop_loss_pct,
            stop_buffer_pct=stop_buffer_pct,
            min_historical_samples=min_historical_samples,
        )
        if latest_candidate is not None:
            candidate_rows.append(latest_candidate)

    all_events = pd.concat(event_frames, ignore_index=True) if event_frames else _empty_events_with_symbol()
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else _empty_trades()
    if not trades.empty:
        trades = trades.sort_values(["signal_date", "symbol"], ascending=[False, True]).reset_index(drop=True)
    candidates_frame = pd.DataFrame(candidate_rows) if candidate_rows else _empty_candidates()
    if not candidates_frame.empty:
        candidates_frame = candidates_frame.sort_values(
            ["historical_target_hit_rate_pct", "historical_avg_net_return_pct", "historical_samples", "event_date", "symbol"],
            ascending=[False, False, False, False, True],
            na_position="last",
        ).reset_index(drop=True)
        candidates_frame.insert(0, "rank", range(1, len(candidates_frame) + 1))

    backtest_stats = _aggregate_backtest_stats(trades)
    summary = _build_summary(
        candidates_frame,
        trades,
        all_events,
        exchange=exchange,
        symbols_processed=len(candidates),
        symbols_with_history=symbols_with_history,
        start_date=start_ts.date().isoformat(),
        as_of_date=as_of_ts.date().isoformat(),
        swing_length=swing_length,
        poc_bins=poc_bins,
        invalidation_method=invalidation_method,
        signal_direction=signal_direction,
        recent_signal_bars=recent_signal_bars,
        require_active_zone=bool(require_active_zone),
        holding_sessions=holding_sessions,
        profit_target_pct=profit_target_pct,
        max_stop_loss_pct=max_stop_loss_pct,
        stop_buffer_pct=stop_buffer_pct,
        round_trip_cost_pct=round_trip_cost_pct,
        min_historical_samples=min_historical_samples,
    )
    return OrderBlockRetestStudyResult(
        summary=summary,
        candidates=candidates_frame,
        backtest_stats=backtest_stats,
        trades=trades,
        events=all_events,
    )


def save_order_block_retest_outputs(
    result: OrderBlockRetestStudyResult,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "summary.json",
        "candidates": output_dir / "latest_candidates.csv",
        "backtest_stats": output_dir / "backtest_stats.csv",
        "trades": output_dir / "trades.csv",
        "events": output_dir / "events.csv",
    }
    paths["summary"].write_text(json.dumps(_json_safe(result.summary), indent=2), encoding="utf-8")
    result.candidates.to_csv(paths["candidates"], index=False)
    result.backtest_stats.to_csv(paths["backtest_stats"], index=False)
    result.trades.to_csv(paths["trades"], index=False)
    result.events.to_csv(paths["events"], index=False)
    return paths


def load_order_block_retest_outputs(output_dir: Path) -> OrderBlockRetestStudyResult:
    summary: dict[str, Any] = {}
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}
    return OrderBlockRetestStudyResult(
        summary=summary,
        candidates=_read_csv(output_dir / "latest_candidates.csv", _empty_candidates),
        backtest_stats=_read_csv(output_dir / "backtest_stats.csv", _empty_backtest_stats),
        trades=_read_csv(output_dir / "trades.csv", _empty_trades),
        events=_read_csv(output_dir / "events.csv", _empty_events_with_symbol),
    )


def _prepare_daily(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame.get("date"), errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    return (
        frame.dropna(subset=["date", "open", "high", "low", "close"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )


def _poc_volume_stats(
    open_values: np.ndarray,
    high_values: np.ndarray,
    low_values: np.ndarray,
    close_values: np.ndarray,
    volume_values: np.ndarray,
    from_index: int,
    to_index: int,
    bins: int,
) -> tuple[float | None, int, float, float, float]:
    if from_index < 0 or to_index < from_index:
        return None, 0, 0.0, 0.0, 0.0
    lows = low_values[from_index : to_index + 1]
    highs = high_values[from_index : to_index + 1]
    min_price = float(np.nanmin(lows))
    max_price = float(np.nanmax(highs))
    if not np.isfinite(min_price) or not np.isfinite(max_price) or min_price >= max_price:
        return None, 0, 0.0, 0.0, 0.0

    step = (max_price - min_price) / bins
    counts = np.zeros(bins, dtype=float)
    for low_value, high_value in zip(lows, highs):
        start_bin = int(np.floor((low_value - min_price) / step))
        end_bin = int(np.floor((high_value - min_price) / step))
        start_bin = max(0, min(bins - 1, start_bin))
        end_bin = max(0, min(bins - 1, end_bin))
        counts[start_bin : end_bin + 1] += 1.0
    best_bin = int(np.argmax(counts))
    poc = min_price + (best_bin + 0.5) * step

    touches = 0
    total_volume = 0.0
    bull_volume = 0.0
    bear_volume = 0.0
    for index in range(from_index, to_index + 1):
        if low_values[index] <= poc <= high_values[index]:
            touches += 1
            volume = float(volume_values[index]) if np.isfinite(volume_values[index]) else 0.0
            total_volume += volume
            if close_values[index] > open_values[index]:
                bull_volume += volume
            elif close_values[index] < open_values[index]:
                bear_volume += volume
    return poc, touches, total_volume, bull_volume, bear_volume


def _bullish_anchor_index(
    high_values: np.ndarray,
    low_values: np.ndarray,
    from_index: int,
    to_index: int,
    poc: float | None,
) -> int | None:
    if poc is None:
        return None
    best_index: int | None = None
    running_min = np.nan
    for index in range(to_index, from_index - 1, -1):
        running_min = low_values[index] if np.isnan(running_min) else min(running_min, low_values[index])
        if low_values[index] <= poc <= high_values[index] and low_values[index] == running_min:
            best_index = index
    return best_index


def _bearish_anchor_index(
    high_values: np.ndarray,
    low_values: np.ndarray,
    from_index: int,
    to_index: int,
    poc: float | None,
) -> int | None:
    if poc is None:
        return None
    best_index: int | None = None
    running_max = np.nan
    for index in range(to_index, from_index - 1, -1):
        running_max = high_values[index] if np.isnan(running_max) else max(running_max, high_values[index])
        if low_values[index] <= poc <= high_values[index] and high_values[index] == running_max:
            best_index = index
    return best_index


def _has_gap(
    high_values: np.ndarray,
    low_values: np.ndarray,
    anchor_index: int,
    bos_index: int,
    *,
    is_bull: bool,
) -> bool:
    for index in range(min(anchor_index, bos_index) + 1, max(anchor_index, bos_index) + 1):
        if is_bull and low_values[index] > high_values[index - 1]:
            return True
        if not is_bull and high_values[index] < low_values[index - 1]:
            return True
    return False


def _overlaps_active(blocks: list[_OrderBlock], top: float, bottom: float) -> bool:
    zone_top = max(top, bottom)
    zone_bottom = min(top, bottom)
    return any(
        block.active and zone_top >= min(block.top, block.bottom) and zone_bottom <= max(block.top, block.bottom)
        for block in blocks
    )


def _new_block(
    block_id: int,
    left_index: int,
    created_index: int,
    top: float,
    bottom: float,
    is_bull: bool,
    poc_stats: tuple[float | None, int, float, float, float],
) -> _OrderBlock:
    _, _, total_volume, bull_volume, bear_volume = poc_stats
    has_delta = total_volume > 0.0
    bull_pct = round((bull_volume / total_volume) * 100.0) if has_delta else 50.0
    bear_pct = 100.0 - bull_pct if has_delta else 50.0
    return _OrderBlock(
        block_id=block_id,
        left_index=left_index,
        created_index=created_index,
        top=max(float(top), float(bottom)),
        bottom=min(float(top), float(bottom)),
        is_bull=is_bull,
        bull_volume=bull_volume,
        bear_volume=bear_volume,
        total_volume=total_volume,
        bull_pct=bull_pct,
        bear_pct=bear_pct,
    )


def _event_row(
    block: _OrderBlock,
    event_type: str,
    event_index: int,
    observation_index: int,
    dates: list[pd.Timestamp],
    close_values: np.ndarray,
) -> dict[str, Any]:
    return {
        "event_type": event_type,
        "side": "BULLISH" if block.is_bull else "BEARISH",
        "event_index": int(event_index),
        "event_date": dates[event_index],
        "observation_index": int(observation_index),
        "observation_date": dates[observation_index],
        "block_id": block.block_id,
        "anchor_index": block.left_index,
        "anchor_date": dates[block.left_index],
        "bos_index": block.created_index,
        "bos_date": dates[block.created_index],
        "zone_top": block.top,
        "zone_bottom": block.bottom,
        "zone_width_pct": ((block.top / block.bottom) - 1.0) * 100.0 if block.bottom > 0 else np.nan,
        "observation_close": float(close_values[observation_index]),
        "bull_volume_pct": block.bull_pct,
        "bear_volume_pct": block.bear_pct,
        "volume_delta_pct": block.bull_pct - block.bear_pct,
        "total_volume_at_poc": block.total_volume,
        "active_at_alert": block.active,
    }


def _prune_blocks(blocks: list[_OrderBlock], current_index: int) -> None:
    minimum_left = max(0, current_index - 4999)
    blocks[:] = [block for block in blocks if block.left_index >= minimum_left]
    while len(blocks) > MAX_STORED_ORDER_BLOCKS:
        inactive_index = next(
            (index for index in range(len(blocks) - 1, -1, -1) if not blocks[index].active),
            None,
        )
        blocks.pop(inactive_index if inactive_index is not None else -1)


def _backtest_symbol(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    *,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    signal_direction: str,
    holding_sessions: int,
    profit_target_pct: float,
    max_stop_loss_pct: float,
    stop_buffer_pct: float,
    round_trip_cost_pct: float,
) -> pd.DataFrame:
    selected = _selected_retests(events, signal_direction).sort_values("event_index")
    rows: list[dict[str, Any]] = []
    next_allowed_entry = 0
    for _, event in selected.iterrows():
        signal_index = int(event["event_index"])
        signal_date = pd.Timestamp(event["event_date"]).normalize()
        entry_index = signal_index + 1
        if signal_date < start_ts or signal_date > end_ts or entry_index >= len(frame):
            continue
        if entry_index < next_allowed_entry:
            continue
        if pd.Timestamp(frame.iloc[entry_index]["date"]).normalize() > end_ts:
            continue
        trade = _simulate_trade(
            frame,
            event,
            entry_index=entry_index,
            end_ts=end_ts,
            holding_sessions=holding_sessions,
            profit_target_pct=profit_target_pct,
            max_stop_loss_pct=max_stop_loss_pct,
            stop_buffer_pct=stop_buffer_pct,
            round_trip_cost_pct=round_trip_cost_pct,
        )
        if trade is None:
            continue
        rows.append(
            {
                "exchange": event.get("exchange", "NSE"),
                "symbol": event.get("symbol", ""),
                "name": event.get("name", event.get("symbol", "")),
                **trade,
            }
        )
        next_allowed_entry = int(trade["exit_index"]) + 1
    return pd.DataFrame(rows) if rows else _empty_trades()


def _simulate_trade(
    frame: pd.DataFrame,
    event: pd.Series,
    *,
    entry_index: int,
    end_ts: pd.Timestamp,
    holding_sessions: int,
    profit_target_pct: float,
    max_stop_loss_pct: float,
    stop_buffer_pct: float,
    round_trip_cost_pct: float,
) -> dict[str, Any] | None:
    entry_price = _finite_float(frame.iloc[entry_index].get("open"))
    if entry_price is None or entry_price <= 0.0:
        return None
    is_bull = str(event.get("side", "")).upper() == "BULLISH"
    stop_price = _planned_stop_price(
        entry_price,
        float(event["zone_top"]),
        float(event["zone_bottom"]),
        is_bull=is_bull,
        max_stop_loss_pct=max_stop_loss_pct,
        stop_buffer_pct=stop_buffer_pct,
    )
    target_price = entry_price * (
        1.0 + profit_target_pct / 100.0 if is_bull else 1.0 - profit_target_pct / 100.0
    )
    planned_exit_index = min(len(frame) - 1, entry_index + holding_sessions - 1)
    while planned_exit_index > entry_index and pd.Timestamp(frame.iloc[planned_exit_index]["date"]).normalize() > end_ts:
        planned_exit_index -= 1

    exit_index = planned_exit_index
    exit_price = _finite_float(frame.iloc[exit_index].get("close"))
    exit_reason = "MAX_HOLD"
    for index in range(entry_index, planned_exit_index + 1):
        bar = frame.iloc[index]
        bar_open = _finite_float(bar.get("open"))
        bar_high = _finite_float(bar.get("high"))
        bar_low = _finite_float(bar.get("low"))
        if bar_open is None or bar_high is None or bar_low is None:
            continue
        if is_bull:
            if bar_open <= stop_price:
                exit_index, exit_price, exit_reason = index, bar_open, "STOP_GAP"
                break
            if bar_open >= target_price:
                exit_index, exit_price, exit_reason = index, bar_open, "TARGET_GAP"
                break
            if bar_low <= stop_price:
                exit_index, exit_price, exit_reason = index, stop_price, "STOP"
                break
            if bar_high >= target_price:
                exit_index, exit_price, exit_reason = index, target_price, "TARGET"
                break
        else:
            if bar_open >= stop_price:
                exit_index, exit_price, exit_reason = index, bar_open, "STOP_GAP"
                break
            if bar_open <= target_price:
                exit_index, exit_price, exit_reason = index, bar_open, "TARGET_GAP"
                break
            if bar_high >= stop_price:
                exit_index, exit_price, exit_reason = index, stop_price, "STOP"
                break
            if bar_low <= target_price:
                exit_index, exit_price, exit_reason = index, target_price, "TARGET"
                break
    if exit_price is None or exit_price <= 0.0:
        return None

    direction = 1.0 if is_bull else -1.0
    gross_return_pct = ((exit_price / entry_price) - 1.0) * 100.0 * direction
    net_return_pct = gross_return_pct - round_trip_cost_pct
    window = frame.iloc[entry_index : exit_index + 1]
    if is_bull:
        mfe_pct = (float(window["high"].max()) / entry_price - 1.0) * 100.0
        mae_pct = (float(window["low"].min()) / entry_price - 1.0) * 100.0
    else:
        mfe_pct = (1.0 - float(window["low"].min()) / entry_price) * 100.0
        mae_pct = (1.0 - float(window["high"].max()) / entry_price) * 100.0
    return {
        "side": event["side"],
        "signal_date": _date_text(event["event_date"]),
        "retest_date": _date_text(event["observation_date"]),
        "entry_date": _date_text(frame.iloc[entry_index]["date"]),
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "exit_date": _date_text(frame.iloc[exit_index]["date"]),
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "holding_sessions": int(exit_index - entry_index + 1),
        "gross_return_pct": gross_return_pct,
        "net_return_pct": net_return_pct,
        "mfe_pct": mfe_pct,
        "mae_pct": mae_pct,
        "target_hit": exit_reason.startswith("TARGET"),
        "won": net_return_pct > 0.0,
        "zone_top": event["zone_top"],
        "zone_bottom": event["zone_bottom"],
        "bull_volume_pct": event["bull_volume_pct"],
        "bear_volume_pct": event["bear_volume_pct"],
        "volume_delta_pct": event["volume_delta_pct"],
        "active_at_alert": event["active_at_alert"],
        "entry_index": entry_index,
        "exit_index": exit_index,
    }


def _latest_candidate(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    exchange: str,
    symbol: str,
    name: str,
    signal_direction: str,
    recent_signal_bars: int,
    require_active_zone: bool,
    profit_target_pct: float,
    max_stop_loss_pct: float,
    stop_buffer_pct: float,
    min_historical_samples: int,
) -> dict[str, Any] | None:
    visible_retests = _selected_retests(events, "either")
    if require_active_zone:
        visible_retests = visible_retests[
            visible_retests["active"].fillna(False).astype(bool)
        ]
    if visible_retests.empty:
        return None
    latest_index = len(frame) - 1
    visible_retests = visible_retests[
        (latest_index - visible_retests["event_index"].astype(int)) < recent_signal_bars
    ].copy()
    if visible_retests.empty:
        return None
    latest_event_index = int(visible_retests["event_index"].max())
    latest_retests = visible_retests[
        visible_retests["event_index"].astype(int).eq(latest_event_index)
    ].copy()
    latest_sides = set(latest_retests["side"].astype(str).str.upper())
    if len(latest_sides) != 1:
        return None
    latest_side = next(iter(latest_sides))
    if signal_direction != "either" and latest_side.lower() != signal_direction:
        return None
    event = latest_retests.sort_values("block_id").iloc[-1]
    event_index = int(event["event_index"])
    entry_index = event_index + 1
    entry_price = _finite_float(frame.iloc[entry_index].get("open")) if entry_index < len(frame) else None
    latest_close = _finite_float(frame.iloc[-1].get("close"))
    is_bull = str(event["side"]).upper() == "BULLISH"
    planning_price = entry_price if entry_price is not None else latest_close
    stop_price = np.nan
    target_price = np.nan
    if planning_price is not None and planning_price > 0.0:
        stop_price = _planned_stop_price(
            planning_price,
            float(event["zone_top"]),
            float(event["zone_bottom"]),
            is_bull=is_bull,
            max_stop_loss_pct=max_stop_loss_pct,
            stop_buffer_pct=stop_buffer_pct,
        )
        target_price = planning_price * (
            1.0 + profit_target_pct / 100.0 if is_bull else 1.0 - profit_target_pct / 100.0
        )

    history = trades.copy()
    if not history.empty:
        history_dates = pd.to_datetime(history["signal_date"], errors="coerce")
        history = history[
            history["side"].astype(str).str.upper().eq(str(event["side"]).upper())
            & (history_dates < pd.Timestamp(event["event_date"]))
        ]
    historical_samples = int(len(history))
    if historical_samples < min_historical_samples:
        return None

    if latest_close is None:
        distance_to_zone_pct = np.nan
    elif latest_close > float(event["zone_top"]):
        distance_to_zone_pct = (latest_close / float(event["zone_top"]) - 1.0) * 100.0
    elif latest_close < float(event["zone_bottom"]):
        distance_to_zone_pct = (latest_close / float(event["zone_bottom"]) - 1.0) * 100.0
    else:
        distance_to_zone_pct = 0.0

    return {
        "exchange": exchange,
        "symbol": symbol,
        "name": name,
        "side": event["side"],
        "signal_match": (
            "Green tick + green block" if is_bull else "Red tick + red block"
        ),
        "block_bos_date": _date_text(event["bos_date"]),
        "tick_date": _date_text(event["observation_date"]),
        "confirmation_date": _date_text(event["event_date"]),
        "event_date": _date_text(event["event_date"]),
        "retest_date": _date_text(event["observation_date"]),
        "signal_age_bars": latest_index - event_index,
        "entry_status": "Next session pending" if entry_index >= len(frame) else "Entry already triggered",
        "entry_date": _date_text(frame.iloc[entry_index]["date"]) if entry_index < len(frame) else "",
        "planned_entry_price": np.nan if entry_price is None else entry_price,
        "planned_stop_price": stop_price,
        "planned_target_price": target_price,
        "latest_date": _date_text(frame.iloc[-1]["date"]),
        "latest_close": latest_close,
        "zone_top": event["zone_top"],
        "zone_bottom": event["zone_bottom"],
        "distance_to_zone_pct": distance_to_zone_pct,
        "zone_active": bool(event["active"]),
        "bull_volume_pct": event["bull_volume_pct"],
        "bear_volume_pct": event["bear_volume_pct"],
        "volume_delta_pct": event["volume_delta_pct"],
        "historical_samples": historical_samples,
        "historical_win_rate_pct": _rate(history, "won"),
        "historical_target_hit_rate_pct": _rate(history, "target_hit"),
        "historical_avg_net_return_pct": _mean(history, "net_return_pct"),
        "historical_median_net_return_pct": _median(history, "net_return_pct"),
        "historical_avg_mfe_pct": _mean(history, "mfe_pct"),
        "historical_avg_mae_pct": _mean(history, "mae_pct"),
    }


def _planned_stop_price(
    entry_price: float,
    zone_top: float,
    zone_bottom: float,
    *,
    is_bull: bool,
    max_stop_loss_pct: float,
    stop_buffer_pct: float,
) -> float:
    buffer = stop_buffer_pct / 100.0
    if is_bull:
        fixed_stop = entry_price * (1.0 - max_stop_loss_pct / 100.0)
        zone_stop = zone_bottom * (1.0 - buffer)
        return max(fixed_stop, zone_stop) if zone_stop < entry_price else fixed_stop
    fixed_stop = entry_price * (1.0 + max_stop_loss_pct / 100.0)
    zone_stop = zone_top * (1.0 + buffer)
    return min(fixed_stop, zone_stop) if zone_stop > entry_price else fixed_stop


def _selected_retests(events: pd.DataFrame, signal_direction: str) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    selected = events[events["event_type"].isin({"BULLISH_RETEST", "BEARISH_RETEST"})].copy()
    if signal_direction != "either":
        selected = selected[selected["side"].str.lower().eq(signal_direction)]
    return selected


def _aggregate_backtest_stats(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return _empty_backtest_stats()
    rows = [_backtest_stats_row("ALL", trades)]
    for side, group in trades.groupby("side", sort=True):
        rows.append(_backtest_stats_row(str(side), group))
    return pd.DataFrame(rows)


def _backtest_stats_row(label: str, trades: pd.DataFrame) -> dict[str, Any]:
    returns = pd.to_numeric(trades["net_return_pct"], errors="coerce").dropna()
    positive = float(returns[returns > 0.0].sum())
    negative = abs(float(returns[returns < 0.0].sum()))
    return {
        "side": label,
        "trades": int(len(trades)),
        "win_rate_pct": _rate(trades, "won"),
        "target_hit_rate_pct": _rate(trades, "target_hit"),
        "avg_net_return_pct": float(returns.mean()) if not returns.empty else np.nan,
        "median_net_return_pct": float(returns.median()) if not returns.empty else np.nan,
        "avg_mfe_pct": _mean(trades, "mfe_pct"),
        "avg_mae_pct": _mean(trades, "mae_pct"),
        "profit_factor": positive / negative if negative > 0.0 else np.nan,
    }


def _build_summary(
    candidates: pd.DataFrame,
    trades: pd.DataFrame,
    events: pd.DataFrame,
    **settings: Any,
) -> dict[str, Any]:
    retests = (
        events[events["event_type"].isin({"BULLISH_RETEST", "BEARISH_RETEST"})]
        if not events.empty
        else events
    )
    return {
        "logic_version": ORDER_BLOCK_LOGIC_VERSION,
        "candidate_count": int(len(candidates)),
        "historical_retests": int(len(retests)),
        "backtest_trades": int(len(trades)),
        "win_rate_pct": _rate(trades, "won"),
        "target_hit_rate_pct": _rate(trades, "target_hit"),
        "avg_net_return_pct": _mean(trades, "net_return_pct"),
        **settings,
    }


def _coerce_date(value: Any) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed).normalize()


def _date_text(value: Any) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(parsed) else pd.Timestamp(parsed).strftime("%Y-%m-%d")


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _rate(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return np.nan
    values = frame[column]
    if values.dtype != bool:
        values = values.astype(str).str.lower().isin({"true", "1", "yes"})
    return float(values.mean() * 100.0)


def _mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else np.nan


def _median(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.median()) if not values.empty else np.nan


def _is_excluded_symbol(symbol: str) -> bool:
    value = str(symbol or "").strip().upper()
    if not value or "-" in value or "NIFTY" in value or "BEES" in value or value.endswith("ETF"):
        return True
    return any(character.isdigit() for character in value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, date)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return None
    return value


def _read_csv(path: Path, empty_factory: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    if not path.exists():
        return empty_factory()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return empty_factory()


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_type", "side", "event_index", "event_date", "observation_index",
            "observation_date", "block_id", "anchor_index", "anchor_date", "bos_index",
            "bos_date", "zone_top", "zone_bottom", "zone_width_pct", "observation_close",
            "bull_volume_pct", "bear_volume_pct", "volume_delta_pct", "total_volume_at_poc",
            "active_at_alert", "active", "invalidation_date",
        ]
    )


def _empty_events_with_symbol() -> pd.DataFrame:
    frame = _empty_events()
    for index, column in enumerate(("exchange", "symbol", "name")):
        frame.insert(index, column, pd.Series(dtype="object"))
    return frame


def _empty_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "rank", "exchange", "symbol", "name", "side", "signal_match",
            "block_bos_date", "tick_date", "confirmation_date", "event_date",
            "retest_date", "signal_age_bars", "entry_status", "entry_date",
            "planned_entry_price", "planned_stop_price", "planned_target_price",
            "latest_date", "latest_close", "zone_top", "zone_bottom",
            "distance_to_zone_pct", "zone_active", "bull_volume_pct",
            "bear_volume_pct", "volume_delta_pct", "historical_samples",
            "historical_win_rate_pct", "historical_target_hit_rate_pct",
            "historical_avg_net_return_pct", "historical_median_net_return_pct",
            "historical_avg_mfe_pct", "historical_avg_mae_pct",
        ]
    )


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "exchange", "symbol", "name", "side", "signal_date", "retest_date",
            "entry_date", "entry_price", "stop_price", "target_price", "exit_date",
            "exit_price", "exit_reason", "holding_sessions", "gross_return_pct",
            "net_return_pct", "mfe_pct", "mae_pct", "target_hit", "won", "zone_top",
            "zone_bottom", "bull_volume_pct", "bear_volume_pct", "volume_delta_pct",
            "active_at_alert", "entry_index", "exit_index",
        ]
    )


def _empty_backtest_stats() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "side", "trades", "win_rate_pct", "target_hit_rate_pct", "avg_net_return_pct",
            "median_net_return_pct", "avg_mfe_pct", "avg_mae_pct", "profit_factor",
        ]
    )
