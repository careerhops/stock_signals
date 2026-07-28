"""Edge-trading strategy module.

Generates two daily candidate lists from KITE EOD candles:

1. ``BREAKOUT`` setups  - momentum / volatility-adjusted Donchian break-outs.
2. ``MEAN_REVERSION`` setups - Connors-style RSI(2) pull-backs to a rising
   long-term moving average.

Each candidate carries:
    * regime classification (TREND / RANGE / RISK_OFF) based on 200-day MA
      slope plus realized vs. ATR volatility,
    * an entry price suggestion (next-day open or limit),
    * an ATR-based hard stop and an ATR-based target,
    * a position-size suggestion in rupees (fixed fractional risk),
    * a historical edge score derived from a walk-forward backtest of the same
      rules on the symbol's own KITE history.

The module is **read-only** with respect to the rest of the codebase: it does
not touch ``daily_scan`` / ``filters`` / ``telegram`` directly. Wiring those is
a separate, opt-in step (see ``docs/EDGE_TRADES_INTEGRATION.md``).

Design notes
------------
* No look-ahead. Every feature on bar ``t`` uses bar ``t`` close information
  only; entries simulate at next-day open in the backtest harness.
* Vectorized pandas/NumPy throughout. The inner loop in
  ``simulate_trades`` is a NumPy state machine, not ``iterrows``.
* Regime filter gates trades. Mean-reversion is only allowed when the regime
  is ``TREND`` and short-term oversold; breakouts only when the regime is
  ``TREND`` or ``RANGE`` (never ``RISK_OFF``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Public configuration
# -----------------------------------------------------------------------------

REGIME_TREND = "TREND"
REGIME_RANGE = "RANGE"
REGIME_RISK_OFF = "RISK_OFF"

SETUP_BREAKOUT = "BREAKOUT"
SETUP_MEAN_REVERSION = "MEAN_REVERSION"


@dataclass(frozen=True)
class EdgeTradeConfig:
    """Knobs for the edge-trading strategy.

    All defaults are chosen to be conservative on Indian large-mid caps with
    5y of daily history. Override per-symbol or per-universe via
    ``config["edge_trades"]`` in ``settings.yaml``.
    """

    # Regime detection
    trend_ma_length: int = 200
    trend_slope_lookback: int = 20
    atr_length: int = 14
    realized_vol_length: int = 20
    risk_off_drawdown_pct: float = 18.0  # close vs. 252d high

    # Breakout setup
    donchian_length: int = 55
    breakout_volume_lookback: int = 20
    breakout_volume_multiplier: float = 1.5
    breakout_atr_stop_mult: float = 2.5
    breakout_atr_target_mult: float = 5.0
    breakout_max_extension_atr: float = 1.0  # don't chase > 1 ATR past pivot

    # Mean reversion setup
    mr_rsi_length: int = 2
    mr_rsi_oversold: float = 10.0
    mr_above_ma_required: bool = True
    mr_ma_length: int = 200
    mr_atr_stop_mult: float = 1.5
    mr_atr_target_mult: float = 2.5
    mr_max_drawdown_pct_from_high: float = 12.0  # not too deep a pullback

    # Position sizing
    risk_per_trade_pct: float = 0.5       # 0.5% of equity per trade
    max_position_pct: float = 7.0          # never > 7% of equity in one name

    # Liquidity / sanity
    min_avg_traded_value_inr: float = 5e7  # ₹5 crore / day median
    min_history_days: int = 260

    # Backtest
    backtest_years: int = 5
    holding_max_days: int = 60


@dataclass
class EdgeSignal:
    """One candidate signal ready to be acted on at the next session."""

    symbol: str
    exchange: str
    setup: str
    signal_date: pd.Timestamp
    close: float
    suggested_entry: float
    stop_loss: float
    target: float
    atr: float
    regime: str
    rsi_2: float
    distance_to_52w_high_pct: float
    realized_vol_20d_pct: float
    avg_traded_value_20d: float
    risk_per_share: float
    risk_reward_ratio: float
    # Historical edge (from per-symbol walk-forward)
    historical_trades: int
    historical_win_rate_pct: float
    historical_avg_return_pct: float
    historical_expectancy_pct: float
    historical_sample_size_flag: str
    # Extras
    notes: list[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Public entry points
# -----------------------------------------------------------------------------


def compute_features(candles: pd.DataFrame, cfg: EdgeTradeConfig | None = None) -> pd.DataFrame:
    """Attach indicator columns the strategy needs. Pure function, vectorized.

    Required input columns: ``date, open, high, low, close, volume``.
    """
    cfg = cfg or EdgeTradeConfig()
    if candles.empty:
        return candles.copy()

    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(candles.columns)
    if missing:
        raise ValueError(f"compute_features missing columns: {sorted(missing)}")

    frame = candles.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    close = pd.to_numeric(frame["close"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)

    # Long-term moving average and slope (TREND filter)
    frame["sma_long"] = close.rolling(cfg.trend_ma_length, min_periods=cfg.trend_ma_length).mean()
    frame["sma_long_slope"] = frame["sma_long"].diff(cfg.trend_slope_lookback)

    # ATR (Wilder)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    frame["atr"] = true_range.ewm(alpha=1 / cfg.atr_length, adjust=False, min_periods=cfg.atr_length).mean()
    frame["atr_pct"] = frame["atr"] / close * 100.0

    # Realized volatility (annualized, in %)
    log_ret = np.log(close / close.shift(1))
    frame["realized_vol_pct"] = log_ret.rolling(cfg.realized_vol_length).std() * np.sqrt(252) * 100.0

    # 52-week / 252-day reference
    frame["high_252"] = high.rolling(252, min_periods=60).max()
    frame["low_252"] = low.rolling(252, min_periods=60).min()
    frame["drawdown_from_252h_pct"] = (close - frame["high_252"]) / frame["high_252"] * 100.0

    # Donchian channel (breakout)
    frame["donchian_high"] = high.shift(1).rolling(cfg.donchian_length, min_periods=cfg.donchian_length).max()
    frame["donchian_low"] = low.shift(1).rolling(cfg.donchian_length, min_periods=cfg.donchian_length).min()

    # Volume confirmation (prior-only baseline; no look-ahead)
    frame["volume_baseline_20"] = (
        volume.shift(1).rolling(cfg.breakout_volume_lookback, min_periods=cfg.breakout_volume_lookback).mean()
    )
    frame["volume_ratio"] = volume / frame["volume_baseline_20"]

    # Average traded value (₹)
    frame["avg_traded_value_20"] = (close * volume).rolling(20, min_periods=20).mean()

    # RSI(2) - vectorized Wilder smoothing via EWMA alpha=1/length
    frame["rsi_2"] = _rsi(close, cfg.mr_rsi_length)

    # Regime classification
    frame["regime"] = _classify_regime(frame, cfg)

    # Breakout flag (does today qualify as a breakout entry?)
    frame["breakout_signal"] = _breakout_mask(frame, cfg)

    # Mean-reversion flag
    frame["mean_reversion_signal"] = _mean_reversion_mask(frame, cfg)

    return frame


def simulate_trades(
    features: pd.DataFrame,
    setup: str,
    cfg: EdgeTradeConfig | None = None,
) -> pd.DataFrame:
    """Walk-forward simulation of a single setup on one symbol.

    Entry  : next-day open after signal bar.
    Stop   : ATR-based, set on signal-bar close. Intraday stop fills at the
             stop price (conservative: assume worst-case fill at the stop).
    Target : ATR-based; first-touch fills at the target price.
    Time   : exit at close after ``holding_max_days`` if neither hit.

    Returns one row per closed trade.
    """
    cfg = cfg or EdgeTradeConfig()
    if features.empty:
        return _empty_trades()

    if setup == SETUP_BREAKOUT:
        signal_col = "breakout_signal"
        atr_stop = cfg.breakout_atr_stop_mult
        atr_target = cfg.breakout_atr_target_mult
    elif setup == SETUP_MEAN_REVERSION:
        signal_col = "mean_reversion_signal"
        atr_stop = cfg.mr_atr_stop_mult
        atr_target = cfg.mr_atr_target_mult
    else:
        raise ValueError(f"Unknown setup {setup!r}")

    if signal_col not in features.columns:
        return _empty_trades()

    dates = features["date"].to_numpy()
    opens = features["open"].to_numpy(dtype=float)
    highs = features["high"].to_numpy(dtype=float)
    lows = features["low"].to_numpy(dtype=float)
    closes = features["close"].to_numpy(dtype=float)
    atrs = features["atr"].to_numpy(dtype=float)
    signals = features[signal_col].fillna(False).to_numpy(dtype=bool)

    rows: list[dict[str, Any]] = []
    i = 0
    n = len(features)
    while i < n - 1:
        if not signals[i] or not np.isfinite(atrs[i]) or atrs[i] <= 0:
            i += 1
            continue

        entry_idx = i + 1
        entry_price = opens[entry_idx]
        if not np.isfinite(entry_price) or entry_price <= 0:
            i += 1
            continue

        atr_at_signal = atrs[i]
        if setup == SETUP_BREAKOUT:
            stop_price = entry_price - atr_stop * atr_at_signal
            target_price = entry_price + atr_target * atr_at_signal
        else:  # mean reversion long
            stop_price = entry_price - atr_stop * atr_at_signal
            target_price = entry_price + atr_target * atr_at_signal

        last_idx = min(entry_idx + cfg.holding_max_days, n - 1)
        exit_idx = last_idx
        exit_price = closes[last_idx]
        exit_reason = "TIME"

        for j in range(entry_idx, last_idx + 1):
            # Gap-through-stop: open already below stop
            if opens[j] <= stop_price:
                exit_idx = j
                exit_price = opens[j]
                exit_reason = "GAP_STOP"
                break
            # Intraday stop hit (assume fill at stop)
            if lows[j] <= stop_price:
                exit_idx = j
                exit_price = stop_price
                exit_reason = "STOP"
                break
            # Intraday target hit (assume fill at target)
            if highs[j] >= target_price:
                exit_idx = j
                exit_price = target_price
                exit_reason = "TARGET"
                break

        return_pct = (exit_price - entry_price) / entry_price * 100.0
        rows.append(
            {
                "setup": setup,
                "signal_date": pd.Timestamp(dates[i]),
                "entry_date": pd.Timestamp(dates[entry_idx]),
                "entry_price": float(entry_price),
                "exit_date": pd.Timestamp(dates[exit_idx]),
                "exit_price": float(exit_price),
                "exit_reason": exit_reason,
                "stop_price": float(stop_price),
                "target_price": float(target_price),
                "atr_at_signal": float(atr_at_signal),
                "return_pct": float(return_pct),
                "holding_days": int((pd.Timestamp(dates[exit_idx]) - pd.Timestamp(dates[entry_idx])).days),
            }
        )

        # No new trade until current one closes (single-position-per-symbol).
        i = exit_idx + 1

    return pd.DataFrame(rows, columns=_trade_columns())


def evaluate_symbol(
    candles: pd.DataFrame,
    symbol: str,
    exchange: str,
    cfg: EdgeTradeConfig | None = None,
) -> tuple[list[EdgeSignal], pd.DataFrame]:
    """End-to-end: features → backtest → latest-bar signals.

    Returns ``(today_signals, backtest_trades)``. ``today_signals`` is empty
    if the latest bar doesn't trigger either setup or fails the liquidity/
    history gate. ``backtest_trades`` is the per-trade frame for both setups
    combined, useful for QA and the historical-edge columns.
    """
    cfg = cfg or EdgeTradeConfig()
    if candles.empty or len(candles) < cfg.min_history_days:
        return [], _empty_trades()

    features = compute_features(candles, cfg)
    if features.empty:
        return [], _empty_trades()

    # Backtest both setups on this symbol's own history
    bt_breakout = simulate_trades(features, SETUP_BREAKOUT, cfg)
    bt_mr = simulate_trades(features, SETUP_MEAN_REVERSION, cfg)
    trades = pd.concat([bt_breakout, bt_mr], ignore_index=True) if not (bt_breakout.empty and bt_mr.empty) else _empty_trades()
    if not trades.empty:
        trades.insert(0, "symbol", symbol)
        trades.insert(1, "exchange", exchange)

    edge_breakout = _edge_stats(bt_breakout)
    edge_mr = _edge_stats(bt_mr)

    latest = features.iloc[-1]
    signals: list[EdgeSignal] = []

    avg_traded_value = float(latest.get("avg_traded_value_20") or 0)
    if avg_traded_value < cfg.min_avg_traded_value_inr:
        return [], trades

    for setup, mask_col, atr_stop, atr_target, edge in (
        (SETUP_BREAKOUT, "breakout_signal", cfg.breakout_atr_stop_mult, cfg.breakout_atr_target_mult, edge_breakout),
        (SETUP_MEAN_REVERSION, "mean_reversion_signal", cfg.mr_atr_stop_mult, cfg.mr_atr_target_mult, edge_mr),
    ):
        if not bool(latest.get(mask_col, False)):
            continue
        atr_value = float(latest.get("atr") or 0)
        close = float(latest.get("close") or 0)
        if atr_value <= 0 or close <= 0:
            continue
        entry = close  # placeholder; the real entry is next-day open
        stop = entry - atr_stop * atr_value
        target = entry + atr_target * atr_value
        risk_per_share = entry - stop
        rr = (target - entry) / risk_per_share if risk_per_share > 0 else float("nan")
        signals.append(
            EdgeSignal(
                symbol=symbol,
                exchange=exchange,
                setup=setup,
                signal_date=pd.Timestamp(latest["date"]),
                close=close,
                suggested_entry=entry,
                stop_loss=stop,
                target=target,
                atr=atr_value,
                regime=str(latest.get("regime") or REGIME_RANGE),
                rsi_2=float(latest.get("rsi_2") or 0.0),
                distance_to_52w_high_pct=float(latest.get("drawdown_from_252h_pct") or 0.0),
                realized_vol_20d_pct=float(latest.get("realized_vol_pct") or 0.0),
                avg_traded_value_20d=avg_traded_value,
                risk_per_share=risk_per_share,
                risk_reward_ratio=rr,
                historical_trades=edge["trades"],
                historical_win_rate_pct=edge["win_rate"],
                historical_avg_return_pct=edge["avg_return"],
                historical_expectancy_pct=edge["expectancy"],
                historical_sample_size_flag=edge["sample_size_flag"],
            )
        )

    return signals, trades


def position_size(equity_inr: float, signal: EdgeSignal, cfg: EdgeTradeConfig | None = None) -> dict[str, float]:
    """Fixed-fractional position sizer.

    Risks ``cfg.risk_per_trade_pct`` of equity on the trade, capped at
    ``cfg.max_position_pct`` of equity in a single name.
    """
    cfg = cfg or EdgeTradeConfig()
    risk_inr = equity_inr * cfg.risk_per_trade_pct / 100.0
    if signal.risk_per_share <= 0:
        return {"quantity": 0.0, "notional_inr": 0.0, "risk_inr": 0.0}
    raw_qty = risk_inr / signal.risk_per_share
    notional = raw_qty * signal.suggested_entry
    cap = equity_inr * cfg.max_position_pct / 100.0
    if notional > cap:
        raw_qty = cap / signal.suggested_entry
        notional = raw_qty * signal.suggested_entry
    return {
        "quantity": float(np.floor(raw_qty)),
        "notional_inr": float(notional),
        "risk_inr": float(min(risk_inr, raw_qty * signal.risk_per_share)),
    }


def signals_to_dataframe(signals: list[EdgeSignal]) -> pd.DataFrame:
    """Convert a list of ``EdgeSignal`` into a flat DataFrame for storage /
    Telegram / dashboard."""
    if not signals:
        return pd.DataFrame(columns=_signal_columns())
    rows = [s.__dict__ for s in signals]
    for row in rows:
        row["notes"] = "; ".join(row.get("notes", []) or [])
    return pd.DataFrame(rows, columns=_signal_columns())


# -----------------------------------------------------------------------------
# Internals
# -----------------------------------------------------------------------------


def _rsi(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_down = down.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask(avg_down == 0, 100.0)
    rsi = rsi.mask((avg_up == 0) & (avg_down == 0), 50.0)
    return rsi


def _classify_regime(features: pd.DataFrame, cfg: EdgeTradeConfig) -> pd.Series:
    close = features["close"]
    sma_long = features["sma_long"]
    sma_long_slope = features["sma_long_slope"]
    drawdown = features["drawdown_from_252h_pct"]
    realized_vol = features["realized_vol_pct"]

    trend_up = (close > sma_long) & (sma_long_slope > 0)
    risk_off_drawdown = drawdown <= -cfg.risk_off_drawdown_pct
    # Compare each bar's realized vol to its own trailing 252-day 90th percentile
    # (apples-to-apples, same series). This flags genuine volatility blow-ups
    # without misfiring on chronically high-vol names.
    vol_threshold = realized_vol.rolling(252, min_periods=60).quantile(0.9)
    high_vol = realized_vol > vol_threshold

    regime = pd.Series(REGIME_RANGE, index=features.index, dtype="object")
    regime[trend_up] = REGIME_TREND
    regime[risk_off_drawdown | high_vol] = REGIME_RISK_OFF
    return regime


def _breakout_mask(features: pd.DataFrame, cfg: EdgeTradeConfig) -> pd.Series:
    close = features["close"]
    donchian_high = features["donchian_high"]
    atr = features["atr"]
    volume_ratio = features["volume_ratio"]
    regime = features["regime"]
    extension = (close - donchian_high) / atr

    mask = (
        (close > donchian_high)
        & (extension <= cfg.breakout_max_extension_atr)
        & (volume_ratio >= cfg.breakout_volume_multiplier)
        & (regime != REGIME_RISK_OFF)
    )
    return mask.fillna(False)


def _mean_reversion_mask(features: pd.DataFrame, cfg: EdgeTradeConfig) -> pd.Series:
    close = features["close"]
    sma_long = features["sma_long"]
    rsi_2 = features["rsi_2"]
    regime = features["regime"]
    drawdown = features["drawdown_from_252h_pct"]

    above_ma = (close > sma_long) if cfg.mr_above_ma_required else pd.Series(True, index=features.index)
    shallow_pullback = drawdown >= -cfg.mr_max_drawdown_pct_from_high

    mask = (
        (rsi_2 < cfg.mr_rsi_oversold)
        & above_ma
        & shallow_pullback
        & (regime == REGIME_TREND)
    )
    return mask.fillna(False)


def _edge_stats(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {
            "trades": 0,
            "win_rate": float("nan"),
            "avg_return": float("nan"),
            "expectancy": float("nan"),
            "sample_size_flag": "NO_TRADES",
        }
    returns = trades["return_pct"].astype(float)
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    win_rate = len(wins) / len(returns) * 100.0
    avg_return = float(returns.mean())
    expectancy = (
        (len(wins) / len(returns)) * (wins.mean() if not wins.empty else 0.0)
        + (len(losses) / len(returns)) * (losses.mean() if not losses.empty else 0.0)
    )
    flag = "OK" if len(returns) >= 20 else "LOW_SAMPLE"
    return {
        "trades": int(len(returns)),
        "win_rate": float(win_rate),
        "avg_return": avg_return,
        "expectancy": float(expectancy),
        "sample_size_flag": flag,
    }


def _trade_columns() -> list[str]:
    return [
        "setup",
        "signal_date",
        "entry_date",
        "entry_price",
        "exit_date",
        "exit_price",
        "exit_reason",
        "stop_price",
        "target_price",
        "atr_at_signal",
        "return_pct",
        "holding_days",
    ]


def _signal_columns() -> list[str]:
    return [
        "symbol",
        "exchange",
        "setup",
        "signal_date",
        "close",
        "suggested_entry",
        "stop_loss",
        "target",
        "atr",
        "regime",
        "rsi_2",
        "distance_to_52w_high_pct",
        "realized_vol_20d_pct",
        "avg_traded_value_20d",
        "risk_per_share",
        "risk_reward_ratio",
        "historical_trades",
        "historical_win_rate_pct",
        "historical_avg_return_pct",
        "historical_expectancy_pct",
        "historical_sample_size_flag",
        "notes",
    ]


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(columns=_trade_columns())
