from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _pine_crossover(left: pd.Series, right: pd.Series) -> pd.Series:
    """True when left crosses above right (was <= right on prior bar)."""
    cond = (left > right) & (left.shift(1) <= right.shift(1))
    return cond.fillna(False)


def _pine_crossunder(left: pd.Series, right: pd.Series) -> pd.Series:
    """True when left crosses below right (was >= right on prior bar)."""
    cond = (left < right) & (left.shift(1) >= right.shift(1))
    return cond.fillna(False)


def _pine_sum(values: pd.Series, length: int) -> pd.Series:
    return values.astype(float).rolling(length, min_periods=1).sum()


def run_weekly_buy_sell(candles: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    strategy_cfg = config.get("strategy", {})
    sensitivity = int(strategy_cfg.get("sensitivity", 3))
    fvg_lookback = int(strategy_cfg.get("fvg_lookback", 5))
    prevent_repeated = bool(strategy_cfg.get("prevent_repeated_direction", True))
    volume_confirmation_lookback = int(strategy_cfg.get("volume_confirmation_lookback", 20))
    volume_confirmation_multiplier = float(strategy_cfg.get("volume_confirmation_multiplier", 1.25))
    pair_return_lookback_weeks = int(strategy_cfg.get("pair_return_lookback_weeks", 104))

    if candles.empty:
        return _empty_strategy_frame(candles)

    frame = candles.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)

    # Structure Levels
    frame["upper_level"] = frame["high"].shift(1).rolling(sensitivity, min_periods=sensitivity).max()
    frame["lower_level"] = frame["low"].shift(1).rolling(sensitivity, min_periods=sensitivity).min()

    # bool bullBreak = ta.crossover(close, upperLevel)
    # bool bearBreak = ta.crossunder(close, lowerLevel)
    frame["bull_break"] = _pine_crossover(frame["close"], frame["upper_level"])
    frame["bear_break"] = _pine_crossunder(frame["close"], frame["lower_level"])

    # bool fvgBull = low > high[2]
    # bool fvgBear = high < low[2]
    frame["fvg_bull"] = (frame["low"] > frame["high"].shift(2)).fillna(False)
    frame["fvg_bear"] = (frame["high"] < frame["low"].shift(2)).fillna(False)

    # float fvgBullRecent = math.sum(fvgBull ? 1.0 : 0.0, fvgLookback)
    # float fvgBearRecent = math.sum(fvgBear ? 1.0 : 0.0, fvgLookback)
    frame["fvg_bull_recent"] = _pine_sum(frame["fvg_bull"], fvg_lookback)
    frame["fvg_bear_recent"] = _pine_sum(frame["fvg_bear"], fvg_lookback)

    # bool buySignal = bullBreak and fvgBullRecent > 0
    # bool sellSignal = bearBreak and fvgBearRecent > 0
    frame["buy_signal"] = frame["bull_break"] & (frame["fvg_bull_recent"] > 0)
    frame["sell_signal"] = frame["bear_break"] & (frame["fvg_bear_recent"] > 0)

    highs = frame["high"].to_numpy()
    lows = frame["low"].to_numpy()
    bull_breaks = frame["bull_break"].to_numpy()
    bear_breaks = frame["bear_break"].to_numpy()
    raw_buys = frame["buy_signal"].to_numpy()
    raw_sells = frame["sell_signal"].to_numpy()

    demand_zones = np.zeros(len(frame), dtype=float)
    supply_zones = np.zeros(len(frame), dtype=float)
    final_buy = np.zeros(len(frame), dtype=bool)
    final_sell = np.zeros(len(frame), dtype=bool)

    current_demand_zone = 0.0
    current_supply_zone = 0.0
    last_signal_direction = 0

    for i in range(len(frame)):
        if bull_breaks[i]:
            prev_low = lows[i - 1] if i > 0 else lows[i]
            current_demand_zone = min(float(lows[i]), float(prev_low))

        if bear_breaks[i]:
            prev_high = highs[i - 1] if i > 0 else highs[i]
            current_supply_zone = max(float(highs[i]), float(prev_high))

        raw_buy = bool(raw_buys[i])
        raw_sell = bool(raw_sells[i]) and not raw_buy

        if prevent_repeated:
            is_buy = raw_buy and last_signal_direction <= 0
            is_sell = raw_sell and last_signal_direction >= 0
        else:
            is_buy = raw_buy
            is_sell = raw_sell

        if is_buy and is_sell:
            is_sell = False

        final_buy[i] = is_buy
        final_sell[i] = is_sell

        if is_buy:
            last_signal_direction = 1
        elif is_sell:
            last_signal_direction = -1

        demand_zones[i] = current_demand_zone
        supply_zones[i] = current_supply_zone

    frame["final_buy"] = final_buy
    frame["final_sell"] = final_sell
    frame["demand_zone"] = demand_zones
    frame["supply_zone"] = supply_zones
    frame["signal"] = "NONE"
    frame.loc[frame["final_buy"], "signal"] = "BUY"
    frame.loc[frame["final_sell"], "signal"] = "SELL"

    frame["avg_volume_20"] = frame["volume"].rolling(20).mean()
    frame["avg_traded_value_20"] = (frame["close"] * frame["volume"]).rolling(20).mean()
    frame["ema_20"] = frame["close"].ewm(span=20, adjust=False).mean()
    frame["ema_50"] = frame["close"].ewm(span=50, adjust=False).mean()
    frame["ema_200"] = frame["close"].ewm(span=200, adjust=False).mean()

    frame["avg_volume_confirmation"] = (
        frame["volume"].shift(1).rolling(volume_confirmation_lookback, min_periods=1).mean()
    )
    frame["volume_confirmation"] = (
        frame["avg_volume_confirmation"].notna()
        & (frame["volume"] >= frame["avg_volume_confirmation"] * volume_confirmation_multiplier)
    )
    frame["volume_confirmation_ratio"] = frame["volume"] / frame["avg_volume_confirmation"]
    frame["trend_confirmation"] = (frame["close"] > frame["ema_20"]) & (frame["ema_20"] > frame["ema_50"])

    pair_return_lookback_start = None
    if pair_return_lookback_weeks > 0:
        pair_return_lookback_start = frame["date"].max() - pd.Timedelta(weeks=pair_return_lookback_weeks)

    _add_completed_trade_return_metrics(frame, pair_return_lookback_start)

    return frame


def _empty_strategy_frame(candles: pd.DataFrame) -> pd.DataFrame:
    frame = candles.copy()
    default_columns = {
        "upper_level": pd.Series(dtype="float64"),
        "lower_level": pd.Series(dtype="float64"),
        "bull_break": pd.Series(dtype="bool"),
        "bear_break": pd.Series(dtype="bool"),
        "fvg_bull": pd.Series(dtype="bool"),
        "fvg_bear": pd.Series(dtype="bool"),
        "fvg_bull_recent": pd.Series(dtype="float64"),
        "fvg_bear_recent": pd.Series(dtype="float64"),
        "buy_signal": pd.Series(dtype="bool"),
        "sell_signal": pd.Series(dtype="bool"),
        "demand_zone": pd.Series(dtype="float64"),
        "supply_zone": pd.Series(dtype="float64"),
        "final_buy": pd.Series(dtype="bool"),
        "final_sell": pd.Series(dtype="bool"),
        "signal": pd.Series(dtype="object"),
        "avg_volume_20": pd.Series(dtype="float64"),
        "avg_traded_value_20": pd.Series(dtype="float64"),
        "ema_20": pd.Series(dtype="float64"),
        "ema_50": pd.Series(dtype="float64"),
        "ema_200": pd.Series(dtype="float64"),
        "avg_volume_confirmation": pd.Series(dtype="float64"),
        "volume_confirmation": pd.Series(dtype="bool"),
        "volume_confirmation_ratio": pd.Series(dtype="float64"),
        "trend_confirmation": pd.Series(dtype="bool"),
        "prior_pair_return_last_1_pct": pd.Series(dtype="float64"),
        "median_pair_return_last_3_pct": pd.Series(dtype="float64"),
        "sell_pair_return_pct": pd.Series(dtype="float64"),
    }
    for column, empty_series in default_columns.items():
        if column not in frame.columns:
            frame[column] = empty_series
    return frame


def _add_completed_trade_return_metrics(
    frame: pd.DataFrame,
    lookback_start: pd.Timestamp | None = None,
) -> None:
    prior_return_last_1: list[float | pd.NA] = []
    median_return_last_3: list[float | pd.NA] = []
    sell_pair_return_pct: list[float | pd.NA] = []
    completed_returns: list[float] = []
    active_buy_close: float | None = None

    for _, row in frame.iterrows():
        row_date = pd.to_datetime(row["date"])
        prior_return_last_1.append(completed_returns[-1] if completed_returns else pd.NA)
        if len(completed_returns) >= 3:
            median_return_last_3.append(float(pd.Series(completed_returns[-3:]).median()))
        else:
            median_return_last_3.append(pd.NA)

        current_sell_return = pd.NA
        close = float(row["close"])
        inside_return_lookback = lookback_start is None or row_date >= lookback_start
        if not inside_return_lookback:
            active_buy_close = None
        elif bool(row["final_buy"]):
            active_buy_close = close
        elif bool(row["final_sell"]) and active_buy_close:
            current_sell_return = ((close - active_buy_close) / active_buy_close) * 100
            completed_returns.append(float(current_sell_return))
            active_buy_close = None

        sell_pair_return_pct.append(current_sell_return)

    frame["prior_pair_return_last_1_pct"] = prior_return_last_1
    frame["median_pair_return_last_3_pct"] = median_return_last_3
    frame["sell_pair_return_pct"] = sell_pair_return_pct
