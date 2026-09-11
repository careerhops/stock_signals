from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from stock_screener.data.storage import Storage


DEFAULT_EMA_LENGTH = 20
DEFAULT_WARMUP_MONTHS = 3
DEFAULT_CURRENT_SIGNAL_LOOKBACK_SESSIONS = 5
DEFAULT_HOLDING_SESSIONS = 10
DEFAULT_PROFIT_TARGET_PCT = 8.0
DEFAULT_STOP_LOSS_PCT = 4.0
DEFAULT_STOP_BUFFER_PCT = 0.5
DEFAULT_ROUND_TRIP_COST_PCT = 0.20
DEFAULT_EMA_TREND_SESSIONS = 20
ENTRY_PRICE_MODES = {"next_open", "next_close"}

STRATEGY_VARIANTS: tuple[tuple[str, str], ...] = (
    ("Any EMA20 Band Bullish Pattern", "any_signal"),
    ("Bullish Inside Bar Below EMA20 Band", "bullish_inside_bar"),
    ("Bullish Engulfing Below EMA20 Band", "bullish_engulfing"),
)


@dataclass(frozen=True)
class Ema20BandStrategyResult:
    summary: dict[str, Any]
    candidates: pd.DataFrame
    strategy_stats: pd.DataFrame
    yearly_stats: pd.DataFrame
    stock_stats: pd.DataFrame
    trades: pd.DataFrame


def run_ema20_band_strategy_study(
    storage: Storage,
    constituents: pd.DataFrame,
    *,
    universe_name: str,
    start_date: Any,
    as_of_date: Any,
    ema_length: int = DEFAULT_EMA_LENGTH,
    warmup_months: int = DEFAULT_WARMUP_MONTHS,
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
) -> Ema20BandStrategyResult:
    start_ts = _coerce_date(start_date)
    as_of_ts = _coerce_date(as_of_date)
    required_latest_ts = _coerce_date(required_latest_date)
    if start_ts is None:
        raise ValueError("Backtest start date is required.")
    if as_of_ts is None:
        raise ValueError("As-of date is required.")
    if start_ts > as_of_ts:
        raise ValueError("Backtest start date cannot be after the as-of date.")

    ema_length = max(int(ema_length), 1)
    entry_mode = str(entry_price_mode or "next_open").strip().lower()
    if entry_mode not in ENTRY_PRICE_MODES:
        entry_mode = "next_open"

    universe = _prepare_constituents(constituents)
    total = len(universe)
    candidate_rows: list[dict[str, Any]] = []
    trade_frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []
    skipped_no_data = 0
    skipped_short = 0
    skipped_stale = 0

    _emit_progress(
        progress_callback,
        phase="Running EMA20 band strategy",
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
            phase="Running EMA20 band strategy",
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
        if len(daily) < ema_length + 2:
            skipped_short += 1
            continue

        frame = _add_features(daily, ema_length=ema_length)
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

        events = _signal_events(frame)
        candidate = _latest_candidate(
            frame,
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
            exit_on_ema_close_reclaim=exit_on_ema_close_reclaim,
        )
        if not trades.empty:
            trade_frames.append(trades)

    candidates = pd.DataFrame(candidate_rows)
    if not candidates.empty:
        candidates = candidates.sort_values(
            ["candidate_score", "sessions_since_signal", "symbol"],
            ascending=[False, True, True],
            na_position="last",
        ).reset_index(drop=True)
        candidates["candidate_rank"] = range(1, len(candidates) + 1)

    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else _empty_trades()
    if not trades.empty:
        trades = trades.sort_values(["entry_date", "strategy", "symbol"], ascending=[False, True, True]).reset_index(drop=True)
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
        ema_length=ema_length,
        warmup_months=warmup_months,
        current_signal_lookback_sessions=current_signal_lookback_sessions,
        holding_sessions=holding_sessions,
        profit_target_pct=profit_target_pct,
        stop_loss_pct=stop_loss_pct,
        stop_buffer_pct=stop_buffer_pct,
        round_trip_cost_pct=round_trip_cost_pct,
        entry_price_mode=entry_mode,
        exit_on_ema_close_reclaim=exit_on_ema_close_reclaim,
    )
    return Ema20BandStrategyResult(
        summary=summary,
        candidates=candidates,
        strategy_stats=strategy_stats,
        yearly_stats=yearly_stats,
        stock_stats=stock_stats,
        trades=trades,
    )


def save_ema20_band_strategy_outputs(result: Ema20BandStrategyResult, output_dir: Path) -> dict[str, Path]:
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


def load_ema20_band_strategy_outputs(output_dir: Path) -> Ema20BandStrategyResult:
    summary_path = output_dir / "summary.json"
    summary: dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}
    return Ema20BandStrategyResult(
        summary=summary,
        candidates=_read_csv(output_dir / "latest_candidates.csv"),
        strategy_stats=_read_csv(output_dir / "strategy_stats.csv"),
        yearly_stats=_read_csv(output_dir / "yearly_stats.csv"),
        stock_stats=_read_csv(output_dir / "stock_stats.csv"),
        trades=_read_csv(output_dir / "trades.csv"),
    )


def write_ema20_band_strategy_workbook(result: Ema20BandStrategyResult, path: Path) -> None:
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


def _add_features(daily: pd.DataFrame, *, ema_length: int) -> pd.DataFrame:
    frame = daily.copy().reset_index(drop=True)
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    open_ = pd.to_numeric(frame["open"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce")

    frame["ema_high"] = high.ewm(span=ema_length, adjust=False, min_periods=1).mean()
    frame["ema_low"] = low.ewm(span=ema_length, adjust=False, min_periods=1).mean()
    frame["ema_close"] = close.ewm(span=ema_length, adjust=False, min_periods=1).mean()
    frame["ema_close_trend_20d_pct"] = (frame["ema_close"] / frame["ema_close"].shift(DEFAULT_EMA_TREND_SESSIONS) - 1.0) * 100.0
    frame["distance_to_ema_low_pct"] = (close / frame["ema_low"] - 1.0) * 100.0
    frame["distance_to_ema_close_pct"] = (close / frame["ema_close"] - 1.0) * 100.0
    frame["band_gap_pct"] = (high / frame["ema_low"] - 1.0) * 100.0
    frame["volume_median_50"] = volume.rolling(50, min_periods=20).median()
    frame["volume_ratio_50"] = volume / frame["volume_median_50"].replace(0, np.nan)

    previous_high = high.shift(1)
    previous_low = low.shift(1)
    previous_open = open_.shift(1)
    previous_close = close.shift(1)
    previous_ema_low = frame["ema_low"].shift(1)

    frame["fully_below_band"] = high < frame["ema_low"]
    frame["previous_fully_below_band"] = previous_high < previous_ema_low
    frame["both_bars_outside"] = frame["previous_fully_below_band"] & frame["fully_below_band"]
    frame["bullish_current"] = close > open_
    frame["inside_bar"] = (high < previous_high) & (low > previous_low)
    frame["previous_bearish"] = previous_close < previous_open
    frame["body_engulfing"] = (open_ <= previous_close) & (close >= previous_open)
    frame["bullish_inside_bar"] = frame["both_bars_outside"] & frame["inside_bar"] & frame["bullish_current"]
    frame["bullish_engulfing"] = (
        frame["both_bars_outside"]
        & frame["previous_bearish"]
        & frame["bullish_current"]
        & frame["body_engulfing"]
    )
    frame["any_signal"] = frame["bullish_inside_bar"] | frame["bullish_engulfing"]

    candle_range = (high - low).replace(0, np.nan)
    body = (close - open_).abs()
    frame["body_to_range_pct"] = body / candle_range * 100.0
    frame["close_location_pct"] = (close - low) / candle_range * 100.0
    return frame


def _signal_events(frame: pd.DataFrame) -> list[int]:
    if frame.empty or "any_signal" not in frame.columns:
        return []
    mask = frame["any_signal"].fillna(False).astype(bool)
    return [int(index) for index in frame.index[mask]]


def _latest_candidate(
    frame: pd.DataFrame,
    events: list[int],
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
    min_signal_index = max(0, as_of_index - max(int(lookback_sessions), 1) + 1)
    recent_events = [index for index in events if min_signal_index <= index <= as_of_index]
    if not recent_events:
        return None

    signal_index = int(max(recent_events))
    signal_row = frame.iloc[signal_index]
    context = _signal_context(frame, signal_index)
    entry_index = signal_index + 1
    entry_row = frame.iloc[entry_index] if entry_index < len(frame) else None
    entry_price = _entry_price(entry_row, entry_price_mode) if entry_row is not None else None
    signal_close = _finite_float(signal_row.get("close"))
    planning_price = entry_price if entry_price is not None else signal_close
    stop_price = np.nan
    target_price = np.nan
    if planning_price is not None:
        stop_price = _planned_stop_price(
            float(planning_price),
            frame,
            signal_index,
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

    return {
        "exchange": exchange,
        "symbol": symbol,
        "name": name,
        "source_universe": source_universe,
        "candidate_rank": np.nan,
        "candidate_score": context["candidate_score"],
        "pattern": context["pattern"],
        "signal_date": _date_text(signal_row.get("date")),
        "sessions_since_signal": int(as_of_index - signal_index),
        "entry_status": entry_status,
        "entry_date": entry_date,
        "planned_entry_price": np.nan if entry_price is None else float(entry_price),
        "planned_stop_price": stop_price,
        "planned_target_price": target_price,
        "as_of_date": _date_text(frame.iloc[as_of_index].get("date")),
        "as_of_close": _finite_float(frame.iloc[as_of_index].get("close")),
        **context,
    }


def _backtest_symbol(
    frame: pd.DataFrame,
    events: list[int],
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
    exit_on_ema_close_reclaim: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    next_allowed_entry: dict[str, int] = {strategy: 0 for strategy, _ in STRATEGY_VARIANTS}
    for signal_index in events:
        signal_date = pd.Timestamp(frame.iloc[signal_index]["date"]).normalize()
        if signal_date < start_ts or signal_date > end_ts:
            continue
        entry_index = int(signal_index) + 1
        if entry_index >= len(frame):
            continue
        if pd.Timestamp(frame.iloc[entry_index]["date"]).normalize() > end_ts:
            continue
        context = _signal_context(frame, int(signal_index))
        for strategy_name, pass_column in STRATEGY_VARIANTS:
            if not bool(context.get(pass_column, False)):
                continue
            if entry_index < next_allowed_entry[strategy_name]:
                continue
            trade = _simulate_trade(
                frame,
                signal_index=int(signal_index),
                entry_index=entry_index,
                holding_sessions=holding_sessions,
                profit_target_pct=profit_target_pct,
                stop_loss_pct=stop_loss_pct,
                stop_buffer_pct=stop_buffer_pct,
                round_trip_cost_pct=round_trip_cost_pct,
                entry_price_mode=entry_price_mode,
                exit_on_ema_close_reclaim=exit_on_ema_close_reclaim,
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
                    "pattern": context.get("pattern"),
                    "ema_trend_20d_label": context.get("ema_trend_20d_label"),
                    "distance_to_ema_low_pct": context.get("distance_to_ema_low_pct"),
                    "distance_to_ema_close_pct": context.get("distance_to_ema_close_pct"),
                    "band_gap_pct": context.get("band_gap_pct"),
                    "volume_ratio_50": context.get("volume_ratio_50"),
                    "body_to_range_pct": context.get("body_to_range_pct"),
                    "close_location_pct": context.get("close_location_pct"),
                }
            )
            next_allowed_entry[strategy_name] = int(trade["exit_index"]) + 1
    return pd.DataFrame(rows)


def _simulate_trade(
    frame: pd.DataFrame,
    *,
    signal_index: int,
    entry_index: int,
    holding_sessions: int,
    profit_target_pct: float,
    stop_loss_pct: float,
    stop_buffer_pct: float,
    round_trip_cost_pct: float,
    entry_price_mode: str,
    exit_on_ema_close_reclaim: bool,
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

    stop_price = _planned_stop_price(
        float(entry_price),
        frame,
        signal_index,
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
        ema_close = _finite_float(bar.get("ema_close"))
        if exit_on_ema_close_reclaim and idx > entry_index and ema_close is not None and bar_close >= ema_close:
            exit_index, exit_price, exit_reason = idx, bar_close, "EMA_CLOSE_RECLAIM"
            break

    if exit_price is None:
        return None
    gross_return = (float(exit_price) / float(entry_price) - 1.0) * 100.0
    net_return = gross_return - float(round_trip_cost_pct)
    trade_window = frame.iloc[entry_index : exit_index + 1]
    mfe = (float(trade_window["high"].max()) / float(entry_price) - 1.0) * 100.0 if not trade_window.empty else np.nan
    mae = (float(trade_window["low"].min()) / float(entry_price) - 1.0) * 100.0 if not trade_window.empty else np.nan
    return {
        "signal_date": _date_text(frame.iloc[signal_index].get("date")),
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


def _signal_context(frame: pd.DataFrame, signal_index: int) -> dict[str, Any]:
    row = frame.iloc[signal_index]
    previous_row = frame.iloc[signal_index - 1] if signal_index > 0 else pd.Series(dtype="object")
    inside = bool(row.get("bullish_inside_bar", False))
    engulfing = bool(row.get("bullish_engulfing", False))
    if inside and engulfing:
        pattern = "Inside + Engulfing"
    elif engulfing:
        pattern = "Bullish Engulfing"
    else:
        pattern = "Bullish Inside Bar"
    ema_trend = _finite_float(row.get("ema_close_trend_20d_pct"))
    trend_label = _ema_trend_label(ema_trend)
    context = {
        "any_signal": bool(row.get("any_signal", False)),
        "bullish_inside_bar": inside,
        "bullish_engulfing": engulfing,
        "pattern": pattern,
        "open": _finite_float(row.get("open")),
        "high": _finite_float(row.get("high")),
        "low": _finite_float(row.get("low")),
        "close": _finite_float(row.get("close")),
        "previous_open": _finite_float(previous_row.get("open")),
        "previous_high": _finite_float(previous_row.get("high")),
        "previous_low": _finite_float(previous_row.get("low")),
        "previous_close": _finite_float(previous_row.get("close")),
        "ema_high": _finite_float(row.get("ema_high")),
        "ema_low": _finite_float(row.get("ema_low")),
        "ema_close": _finite_float(row.get("ema_close")),
        "ema_close_trend_20d_pct": ema_trend,
        "ema_trend_20d_label": trend_label,
        "distance_to_ema_low_pct": _finite_float(row.get("distance_to_ema_low_pct")),
        "distance_to_ema_close_pct": _finite_float(row.get("distance_to_ema_close_pct")),
        "band_gap_pct": _finite_float(row.get("band_gap_pct")),
        "body_to_range_pct": _finite_float(row.get("body_to_range_pct")),
        "close_location_pct": _finite_float(row.get("close_location_pct")),
        "volume_ratio_50": _finite_float(row.get("volume_ratio_50")),
        "fully_below_band": bool(row.get("fully_below_band", False)),
        "previous_fully_below_band": bool(row.get("previous_fully_below_band", False)),
        "both_bars_outside": bool(row.get("both_bars_outside", False)),
    }
    context["candidate_score"] = _candidate_score(context)
    return context


def _candidate_score(context: dict[str, Any]) -> float:
    score = 0.0
    if context.get("bullish_engulfing"):
        score += 34.0
    if context.get("bullish_inside_bar"):
        score += 28.0
    if context.get("bullish_engulfing") and context.get("bullish_inside_bar"):
        score += 6.0

    close_location = _finite_float(context.get("close_location_pct"))
    if close_location is not None:
        score += min(max(close_location, 0.0), 100.0) / 100.0 * 12.0

    body_ratio = _finite_float(context.get("body_to_range_pct"))
    if body_ratio is not None:
        score += min(max(body_ratio, 0.0), 80.0) / 80.0 * 10.0

    band_gap = _finite_float(context.get("band_gap_pct"))
    if band_gap is not None:
        depth = abs(min(band_gap, 0.0))
        if 0.25 <= depth <= 8.0:
            score += 12.0
        elif depth <= 12.0:
            score += 8.0
        elif depth <= 18.0:
            score += 3.0

    volume_ratio = _finite_float(context.get("volume_ratio_50"))
    if volume_ratio is not None:
        if volume_ratio >= 2.0:
            score += 8.0
        elif volume_ratio >= 1.3:
            score += 5.0
        elif volume_ratio >= 1.0:
            score += 2.0

    trend_label = str(context.get("ema_trend_20d_label") or "")
    if trend_label == "Rising":
        score += 10.0
    elif trend_label == "Flat":
        score += 8.0
    elif trend_label == "Falling":
        score += 3.0

    distance_to_close = _finite_float(context.get("distance_to_ema_close_pct"))
    if distance_to_close is not None:
        gap = abs(min(distance_to_close, 0.0))
        if gap <= 8.0:
            score += 6.0
        elif gap <= 14.0:
            score += 3.0

    return round(float(max(0.0, min(score, 100.0))), 2)


def _planned_stop_price(
    entry_price: float,
    frame: pd.DataFrame,
    signal_index: int,
    *,
    stop_loss_pct: float,
    stop_buffer_pct: float,
) -> float:
    fixed_stop = float(entry_price) * (1.0 - float(stop_loss_pct) / 100.0)
    signal_low = _finite_float(frame.iloc[signal_index].get("low"))
    previous_low = _finite_float(frame.iloc[signal_index - 1].get("low")) if signal_index > 0 else None
    pattern_lows = [value for value in (signal_low, previous_low) if value is not None]
    if not pattern_lows:
        return fixed_stop
    pattern_stop = min(pattern_lows) * (1.0 - float(stop_buffer_pct) / 100.0)
    stop = max(fixed_stop, pattern_stop)
    if stop >= entry_price:
        return fixed_stop
    return float(stop)


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
    ema_length: int,
    warmup_months: int,
    current_signal_lookback_sessions: int,
    holding_sessions: int,
    profit_target_pct: float,
    stop_loss_pct: float,
    stop_buffer_pct: float,
    round_trip_cost_pct: float,
    entry_price_mode: str,
    exit_on_ema_close_reclaim: bool,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "strategy_name": "EMA20 Band Strict Outside Bullish Patterns",
        "universe": universe_name,
        "requested_as_of_date": as_of_ts.strftime("%Y-%m-%d"),
        "analysis_as_of_date": as_of_ts.strftime("%Y-%m-%d"),
        "start_date": start_ts.strftime("%Y-%m-%d"),
        "ema_length": int(ema_length),
        "warmup_months": int(warmup_months),
        "current_signal_lookback_sessions": int(current_signal_lookback_sessions),
        "holding_sessions": int(holding_sessions),
        "profit_target_pct": float(profit_target_pct),
        "stop_loss_pct": float(stop_loss_pct),
        "stop_buffer_pct": float(stop_buffer_pct),
        "round_trip_cost_pct": float(round_trip_cost_pct),
        "entry_price_mode": entry_price_mode,
        "exit_on_ema_close_reclaim": bool(exit_on_ema_close_reclaim),
        "symbols_requested": int(total_symbols),
        "symbols_with_history": int(len(coverage)),
        "symbols_skipped_no_data": int(skipped_no_data),
        "symbols_skipped_short_history": int(skipped_short),
        "symbols_skipped_stale_history": int(skipped_stale),
        "current_candidates": int(len(candidates)),
        "current_inside_bar_candidates": int(candidates["bullish_inside_bar"].sum()) if not candidates.empty and "bullish_inside_bar" in candidates.columns else 0,
        "current_engulfing_candidates": int(candidates["bullish_engulfing"].sum()) if not candidates.empty and "bullish_engulfing" in candidates.columns else 0,
        "total_trades": int(len(trades)),
    }
    if not strategy_stats.empty:
        best = strategy_stats.iloc[0]
        summary.update(
            {
                "best_strategy": str(best.get("strategy", "")),
                "best_strategy_trades": int(best.get("trades", 0) or 0),
                "best_strategy_win_rate_pct": float(best.get("win_rate_pct", 0.0) or 0.0),
                "best_strategy_avg_return_pct": float(best.get("avg_return_pct", 0.0) or 0.0),
                "best_strategy_profit_factor": _finite_float(best.get("profit_factor")),
            }
        )
    else:
        summary.update(
            {
                "best_strategy": "",
                "best_strategy_trades": 0,
                "best_strategy_win_rate_pct": 0.0,
                "best_strategy_avg_return_pct": 0.0,
                "best_strategy_profit_factor": np.nan,
            }
        )
    return summary


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
            "entry_price",
            "exit_date",
            "exit_price",
            "exit_reason",
            "stop_price",
            "target_price",
            "hold_trading_sessions",
            "hold_calendar_days",
            "gross_return_pct",
            "net_return_pct",
            "win_flag",
            "mfe_pct",
            "mae_pct",
            "entry_index",
            "exit_index",
            "candidate_score",
            "pattern",
            "ema_trend_20d_label",
            "distance_to_ema_low_pct",
            "distance_to_ema_close_pct",
            "band_gap_pct",
            "volume_ratio_50",
            "body_to_range_pct",
            "close_location_pct",
        ]
    )


def _ema_trend_label(value: Any) -> str:
    trend = _finite_float(value)
    if trend is None:
        return "Unknown"
    if trend > 0.25:
        return "Rising"
    if trend < -0.25:
        return "Falling"
    return "Flat"


def _coerce_date(value: Any) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).normalize()


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _date_text(value: Any) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return ""
    return pd.Timestamp(parsed).strftime("%Y-%m-%d")


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _emit_progress(progress_callback: Callable[[dict[str, Any]], None] | None, **payload: Any) -> None:
    if progress_callback is not None:
        progress_callback(payload)
