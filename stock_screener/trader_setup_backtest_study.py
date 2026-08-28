from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from stock_screener.data.storage import Storage
from stock_screener.weekly_buy_tracker_study import _emit_progress, _load_name_map, _to_float

# Signal rules are derived from the user-provided TraderSetup script, including
# LuxAlgo support/resistance logic licensed under CC BY-NC-SA 4.0.


DEFAULT_LEFT_BARS = 5
DEFAULT_RIGHT_BARS = 5
DEFAULT_VOLUME_OSC_THRESHOLD = 15.0
DEFAULT_RVOL_THRESHOLD = 1.40
DEFAULT_ATR_BUFFER = 0.10
DEFAULT_HOLDING_DAYS = 20
DEFAULT_PROFIT_TARGET_PCT = 10.0
DEFAULT_STOP_LOSS_PCT = 5.0
DEFAULT_ROUND_TRIP_COST_PCT = 0.20


@dataclass(frozen=True)
class TraderSetupBacktestResult:
    summary: dict[str, Any]
    strategy_stats: pd.DataFrame
    yearly_stats: pd.DataFrame
    stock_stats: pd.DataFrame
    trades: pd.DataFrame


def calculate_trader_setup_signals(
    daily: pd.DataFrame,
    *,
    left_bars: int = DEFAULT_LEFT_BARS,
    right_bars: int = DEFAULT_RIGHT_BARS,
    volume_osc_threshold: float = DEFAULT_VOLUME_OSC_THRESHOLD,
    rvol_threshold: float = DEFAULT_RVOL_THRESHOLD,
    atr_buffer: float = DEFAULT_ATR_BUFFER,
) -> pd.DataFrame:
    frame = _prepare_daily(daily)
    if frame.empty:
        return frame

    high = frame["high"]
    low = frame["low"]
    open_ = frame["open"]
    close = frame["close"]
    volume = frame["volume"]

    true_range = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = _pine_rma(true_range, 14)
    volume_short_ema = volume.ewm(span=5, adjust=False).mean()
    volume_long_ema = volume.ewm(span=10, adjust=False).mean()
    volume_oscillator = (
        100.0 * (volume_short_ema - volume_long_ema) / volume_long_ema.replace(0, np.nan)
    ).fillna(0.0)
    avg_volume20 = volume.rolling(20, min_periods=20).mean()
    rvol = volume / avg_volume20.replace(0, np.nan)

    body = (close - open_).abs()
    upper_wick = high - pd.concat([open_, close], axis=1).max(axis=1)
    lower_wick = pd.concat([open_, close], axis=1).min(axis=1) - low
    good_bull_candle = (close > open_) & body.gt(0) & (upper_wick <= body * 0.50)
    good_bear_candle = (close < open_) & body.gt(0) & (lower_wick <= body * 0.50)

    resistance = _confirmed_pivot_level(
        high,
        left_bars=int(left_bars),
        right_bars=int(right_bars),
        mode="high",
    )
    support = _confirmed_pivot_level(
        low,
        left_bars=int(left_bars),
        right_bars=int(right_bars),
        mode="low",
    )
    crossed_above_resistance = (
        close.shift(1).notna()
        & resistance.shift(1).notna()
        & resistance.notna()
        & (close.shift(1) <= resistance.shift(1))
        & (close > resistance)
    )
    crossed_below_support = (
        close.shift(1).notna()
        & support.shift(1).notna()
        & support.notna()
        & (close.shift(1) >= support.shift(1))
        & (close < support)
    )
    resistance_break = (
        crossed_above_resistance
        & good_bull_candle
        & (volume_oscillator > float(volume_osc_threshold))
        & (rvol >= float(rvol_threshold))
        & (close > resistance + atr * float(atr_buffer))
    )
    support_break = (
        crossed_below_support
        & good_bear_candle
        & (volume_oscillator > float(volume_osc_threshold))
        & (rvol >= float(rvol_threshold))
        & (close < support - atr * float(atr_buffer))
    )

    momentum_pivot = high.rolling(20, min_periods=20).max().shift(1)
    crossed_above_pivot = (
        close.shift(1).notna()
        & momentum_pivot.shift(1).notna()
        & momentum_pivot.notna()
        & (close.shift(1) <= momentum_pivot.shift(1))
        & (close > momentum_pivot)
    )
    pivot_breakout = (
        crossed_above_pivot
        & good_bull_candle
        & (rvol >= float(rvol_threshold))
        & (close > momentum_pivot + atr * float(atr_buffer))
    )

    frame["atr"] = atr
    frame["volume_oscillator"] = volume_oscillator
    frame["rvol"] = rvol
    frame["resistance_level"] = resistance
    frame["support_level"] = support
    frame["momentum_pivot"] = momentum_pivot
    frame["good_bull_candle"] = good_bull_candle.fillna(False)
    frame["good_bear_candle"] = good_bear_candle.fillna(False)
    frame["resistance_break"] = resistance_break.fillna(False)
    frame["support_break"] = support_break.fillna(False)
    frame["pivot_breakout"] = pivot_breakout.fillna(False)
    frame["combined_long_breakout"] = (
        frame["resistance_break"] | frame["pivot_breakout"]
    )
    return frame


def backtest_trader_setup_frame(
    signals: pd.DataFrame,
    *,
    symbol: str,
    exchange: str = "NSE",
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    holding_days: int = DEFAULT_HOLDING_DAYS,
    profit_target_pct: float = DEFAULT_PROFIT_TARGET_PCT,
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
    round_trip_cost_pct: float = DEFAULT_ROUND_TRIP_COST_PCT,
) -> pd.DataFrame:
    frame = signals.reset_index(drop=True).copy()
    if frame.empty:
        return pd.DataFrame()
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    strategies = (
        ("Momentum Pivot Breakout", "pivot_breakout", "LONG"),
        ("Resistance Break", "resistance_break", "LONG"),
        ("Combined Long Breakout", "combined_long_breakout", "LONG"),
        ("Support Break", "support_break", "SHORT"),
    )
    rows: list[dict[str, Any]] = []
    for strategy, signal_column, side in strategies:
        if signal_column not in frame.columns:
            continue
        signal_indexes = frame.index[
            frame[signal_column].fillna(False).astype(bool)
            & frame["date"].dt.normalize().between(start_ts, end_ts)
        ].tolist()
        next_allowed_entry = 0
        for signal_index in signal_indexes:
            entry_index = int(signal_index) + 1
            if entry_index < next_allowed_entry or entry_index >= len(frame):
                continue
            planned_exit_index = entry_index + max(int(holding_days), 1) - 1
            if frame.iloc[entry_index]["date"].normalize() > end_ts:
                continue
            # Only score closed, full-horizon trades. A signal near the study end
            # remains open and must not be converted into a shorter winning/losing trade.
            if planned_exit_index >= len(frame):
                continue
            if frame.iloc[planned_exit_index]["date"].normalize() > end_ts:
                continue
            trade = _simulate_trade(
                frame,
                signal_index=int(signal_index),
                entry_index=entry_index,
                planned_exit_index=planned_exit_index,
                side=side,
                profit_target_pct=float(profit_target_pct),
                stop_loss_pct=float(stop_loss_pct),
                round_trip_cost_pct=float(round_trip_cost_pct),
            )
            if trade is None:
                continue
            rows.append(
                {
                    "exchange": exchange,
                    "symbol": symbol,
                    "strategy": strategy,
                    "side": side,
                    **trade,
                }
            )
            next_allowed_entry = int(trade["exit_index"]) + 1
    return pd.DataFrame(rows)


def run_trader_setup_backtest_study(
    storage: Storage,
    exchange: str = "NSE",
    *,
    symbols: list[str] | None = None,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    left_bars: int = DEFAULT_LEFT_BARS,
    right_bars: int = DEFAULT_RIGHT_BARS,
    volume_osc_threshold: float = DEFAULT_VOLUME_OSC_THRESHOLD,
    rvol_threshold: float = DEFAULT_RVOL_THRESHOLD,
    atr_buffer: float = DEFAULT_ATR_BUFFER,
    holding_days: int = DEFAULT_HOLDING_DAYS,
    profit_target_pct: float = DEFAULT_PROFIT_TARGET_PCT,
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
    round_trip_cost_pct: float = DEFAULT_ROUND_TRIP_COST_PCT,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> TraderSetupBacktestResult:
    if symbols is None:
        all_symbols = sorted(
            path.stem for path in (storage.data_root / "candles" / exchange / "1D").glob("*.csv")
        )
    else:
        all_symbols = sorted({str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()})
    name_map = _load_name_map(storage, exchange)
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    trade_frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []

    _emit_progress(
        progress_callback,
        phase="Backtesting TraderSetup signals",
        completed=0,
        total=len(all_symbols),
        current_symbol="",
        current_exchange=exchange,
    )
    for completed, symbol in enumerate(all_symbols, start=1):
        daily = _prepare_daily(storage.load_candles(exchange, symbol, "1D"))
        _emit_progress(
            progress_callback,
            phase="Backtesting TraderSetup signals",
            completed=completed,
            total=len(all_symbols),
            current_symbol=symbol,
            current_exchange=exchange,
        )
        if daily.empty:
            continue
        available_start = pd.Timestamp(daily.iloc[0]["date"]).normalize()
        available_end = pd.Timestamp(daily.iloc[-1]["date"]).normalize()
        coverage_rows.append(
            {
                "symbol": symbol,
                "name": name_map.get(symbol, symbol),
                "history_start": available_start,
                "history_end": available_end,
                "history_years": max((available_end - available_start).days / 365.25, 0.0),
            }
        )
        signals = calculate_trader_setup_signals(
            daily,
            left_bars=left_bars,
            right_bars=right_bars,
            volume_osc_threshold=volume_osc_threshold,
            rvol_threshold=rvol_threshold,
            atr_buffer=atr_buffer,
        )
        trades = backtest_trader_setup_frame(
            signals,
            symbol=symbol,
            exchange=exchange,
            start_date=start_ts,
            end_date=end_ts,
            holding_days=holding_days,
            profit_target_pct=profit_target_pct,
            stop_loss_pct=stop_loss_pct,
            round_trip_cost_pct=round_trip_cost_pct,
        )
        if not trades.empty:
            trades["name"] = name_map.get(symbol, symbol)
            trade_frames.append(trades)

    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    coverage = pd.DataFrame(coverage_rows)
    strategy_stats = _aggregate_stats(trades, ["strategy"])
    yearly_stats = _aggregate_yearly(trades)
    stock_stats = _aggregate_stats(trades, ["strategy", "symbol", "name"])
    if not trades.empty:
        trades = trades.sort_values(["entry_date", "strategy", "symbol"], ascending=[False, True, True]).reset_index(drop=True)

    summary = {
        "exchange": exchange,
        "requested_start_date": start_ts.strftime("%Y-%m-%d"),
        "requested_end_date": end_ts.strftime("%Y-%m-%d"),
        "symbols_processed": len(all_symbols),
        "symbols_with_history": len(coverage),
        "symbols_with_9y_history": int((coverage.get("history_years", pd.Series(dtype=float)) >= 9.0).sum()),
        "earliest_history_date": coverage["history_start"].min().strftime("%Y-%m-%d") if not coverage.empty else "",
        "latest_history_date": coverage["history_end"].max().strftime("%Y-%m-%d") if not coverage.empty else "",
        "total_trades": len(trades),
        "holding_days": int(holding_days),
        "profit_target_pct": float(profit_target_pct),
        "stop_loss_pct": float(stop_loss_pct),
        "round_trip_cost_pct": float(round_trip_cost_pct),
        "left_bars": int(left_bars),
        "right_bars": int(right_bars),
        "volume_osc_threshold": float(volume_osc_threshold),
        "rvol_threshold": float(rvol_threshold),
        "atr_buffer": float(atr_buffer),
        "generated_at_ist": pd.Timestamp.now(tz="Asia/Kolkata").strftime("%Y-%m-%d %H:%M:%S IST"),
    }
    return TraderSetupBacktestResult(
        summary=summary,
        strategy_stats=strategy_stats,
        yearly_stats=yearly_stats,
        stock_stats=stock_stats,
        trades=trades,
    )


def save_trader_setup_backtest_outputs(
    result: TraderSetupBacktestResult,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "latest_summary.csv",
        "strategy_stats": output_dir / "latest_strategy_stats.csv",
        "yearly_stats": output_dir / "latest_yearly_stats.csv",
        "stock_stats": output_dir / "latest_stock_stats.csv",
        "trades": output_dir / "latest_trades.csv",
    }
    pd.DataFrame([result.summary]).to_csv(paths["summary"], index=False)
    result.strategy_stats.to_csv(paths["strategy_stats"], index=False)
    result.yearly_stats.to_csv(paths["yearly_stats"], index=False)
    result.stock_stats.to_csv(paths["stock_stats"], index=False)
    result.trades.to_csv(paths["trades"], index=False)
    return paths


def load_trader_setup_backtest_outputs(output_dir: Path) -> TraderSetupBacktestResult:
    def read(name: str) -> pd.DataFrame:
        path = output_dir / name
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    summary_frame = read("latest_summary.csv")
    return TraderSetupBacktestResult(
        summary=summary_frame.iloc[0].to_dict() if not summary_frame.empty else {},
        strategy_stats=read("latest_strategy_stats.csv"),
        yearly_stats=read("latest_yearly_stats.csv"),
        stock_stats=read("latest_stock_stats.csv"),
        trades=read("latest_trades.csv"),
    )


def _simulate_trade(
    frame: pd.DataFrame,
    *,
    signal_index: int,
    entry_index: int,
    planned_exit_index: int,
    side: str,
    profit_target_pct: float,
    stop_loss_pct: float,
    round_trip_cost_pct: float,
) -> dict[str, Any] | None:
    entry_price = _to_float(frame.iloc[entry_index].get("open"))
    if entry_price is None or entry_price <= 0:
        return None
    long_side = side == "LONG"
    target_price = entry_price * (1.0 + profit_target_pct / 100.0) if long_side else entry_price * (1.0 - profit_target_pct / 100.0)
    stop_price = entry_price * (1.0 - stop_loss_pct / 100.0) if long_side else entry_price * (1.0 + stop_loss_pct / 100.0)
    exit_index = planned_exit_index
    exit_price = _to_float(frame.iloc[exit_index].get("close"))
    exit_reason = "TIME"
    for idx in range(entry_index, planned_exit_index + 1):
        bar = frame.iloc[idx]
        bar_open = float(bar["open"])
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        if long_side:
            if bar_open <= stop_price:
                exit_index, exit_price, exit_reason = idx, bar_open, "STOP_GAP"
                break
            if bar_open >= target_price:
                exit_index, exit_price, exit_reason = idx, bar_open, "TARGET_GAP"
                break
            stop_hit = bar_low <= stop_price
            target_hit = bar_high >= target_price
        else:
            if bar_open >= stop_price:
                exit_index, exit_price, exit_reason = idx, bar_open, "STOP_GAP"
                break
            if bar_open <= target_price:
                exit_index, exit_price, exit_reason = idx, bar_open, "TARGET_GAP"
                break
            stop_hit = bar_high >= stop_price
            target_hit = bar_low <= target_price
        if stop_hit:
            exit_index, exit_price, exit_reason = idx, stop_price, "STOP"
            break
        if target_hit:
            exit_index, exit_price, exit_reason = idx, target_price, "TARGET"
            break
    if exit_price is None:
        return None
    gross_return = (
        (exit_price - entry_price) / entry_price * 100.0
        if long_side
        else (entry_price - exit_price) / entry_price * 100.0
    )
    net_return = gross_return - round_trip_cost_pct
    window = frame.iloc[entry_index : exit_index + 1]
    if long_side:
        mfe = (float(window["high"].max()) - entry_price) / entry_price * 100.0
        mae = (float(window["low"].min()) - entry_price) / entry_price * 100.0
    else:
        mfe = (entry_price - float(window["low"].min())) / entry_price * 100.0
        mae = (entry_price - float(window["high"].max())) / entry_price * 100.0
    signal_row = frame.iloc[signal_index]
    return {
        "signal_date": signal_row["date"].strftime("%Y-%m-%d"),
        "entry_date": frame.iloc[entry_index]["date"].strftime("%Y-%m-%d"),
        "exit_date": frame.iloc[exit_index]["date"].strftime("%Y-%m-%d"),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "bars_held": int(exit_index - entry_index + 1),
        "gross_return_pct": gross_return,
        "net_return_pct": net_return,
        "win_flag": net_return > 0,
        "mfe_pct": mfe,
        "mae_pct": mae,
        "signal_rvol": _to_float(signal_row.get("rvol")),
        "signal_volume_oscillator": _to_float(signal_row.get("volume_oscillator")),
        "signal_level": _to_float(
            signal_row.get("momentum_pivot")
            if bool(signal_row.get("pivot_breakout"))
            else signal_row.get("resistance_level")
            if bool(signal_row.get("resistance_break"))
            else signal_row.get("support_level")
        ),
        "exit_index": int(exit_index),
    }


def _aggregate_stats(trades: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_key: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for key, group in trades.groupby(group_key, dropna=False):
        keys = (key,) if len(group_columns) == 1 else tuple(key)
        returns = pd.to_numeric(group["net_return_pct"], errors="coerce").dropna()
        wins = returns[returns > 0]
        losses = returns[returns <= 0]
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "trades": len(returns),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate_pct": len(wins) / len(returns) * 100.0 if len(returns) else 0.0,
                "avg_return_pct": float(returns.mean()) if len(returns) else 0.0,
                "median_return_pct": float(returns.median()) if len(returns) else 0.0,
                "avg_win_pct": float(wins.mean()) if len(wins) else 0.0,
                "avg_loss_pct": float(losses.mean()) if len(losses) else 0.0,
                "payoff_ratio": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) and losses.mean() != 0 else np.nan,
                "profit_factor": float(wins.sum() / abs(losses.sum())) if len(wins) and len(losses) and losses.sum() != 0 else np.nan,
                "avg_mfe_pct": float(pd.to_numeric(group["mfe_pct"], errors="coerce").mean()),
                "avg_mae_pct": float(pd.to_numeric(group["mae_pct"], errors="coerce").mean()),
            }
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    return result.sort_values(["win_rate_pct", "avg_return_pct", "trades"], ascending=[False, False, False]).reset_index(drop=True)


def _aggregate_yearly(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    frame = trades.copy()
    frame["year"] = pd.to_datetime(frame["entry_date"], errors="coerce").dt.year
    return _aggregate_stats(frame.dropna(subset=["year"]), ["strategy", "year"])


def _prepare_daily(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["date", "open", "high", "low", "close", "volume"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _pine_rma(series: pd.Series, length: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    result = pd.Series(np.nan, index=values.index, dtype=float)
    if len(values) < length:
        return result
    seed = values.iloc[:length].mean()
    if pd.isna(seed):
        return result
    result.iloc[length - 1] = seed
    previous = float(seed)
    for index in range(length, len(values)):
        value = values.iloc[index]
        if pd.isna(value):
            continue
        previous = (previous * (length - 1) + float(value)) / float(length)
        result.iloc[index] = previous
    return result


def _confirmed_pivot_level(
    series: pd.Series,
    *,
    left_bars: int,
    right_bars: int,
    mode: str,
) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").reset_index(drop=True)
    confirmed = pd.Series(np.nan, index=values.index, dtype=float)
    for center in range(left_bars, len(values) - right_bars):
        value = values.iloc[center]
        left = values.iloc[center - left_bars : center]
        right = values.iloc[center + 1 : center + right_bars + 1]
        if pd.isna(value) or left.isna().any() or right.isna().any():
            continue
        is_pivot = (
            value >= left.max() and value > right.max()
            if mode == "high"
            else value <= left.min() and value < right.min()
        )
        if is_pivot:
            confirmed.iloc[center + right_bars] = float(value)
    return confirmed.ffill()
