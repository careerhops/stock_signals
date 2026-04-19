from __future__ import annotations

from typing import Any

import pandas as pd


def run_weekly_buy_sell(candles: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    strategy_cfg = config.get("strategy", {})
    sensitivity = int(strategy_cfg.get("sensitivity", 3))
    fvg_lookback = int(strategy_cfg.get("fvg_lookback", 5))
    prevent_repeated = bool(strategy_cfg.get("prevent_repeated_direction", True))
    volume_confirmation_lookback = int(strategy_cfg.get("volume_confirmation_lookback", 20))
    volume_confirmation_multiplier = float(strategy_cfg.get("volume_confirmation_multiplier", 1.25))
    pair_return_lookback_weeks = int(strategy_cfg.get("pair_return_lookback_weeks", 104))

    if candles.empty:
        return candles

    frame = candles.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)

    frame["upper_level"] = frame["high"].shift(1).rolling(sensitivity).max()
    frame["lower_level"] = frame["low"].shift(1).rolling(sensitivity).min()

    prev_close = frame["close"].shift(1)
    prev_upper = frame["upper_level"].shift(1)
    prev_lower = frame["lower_level"].shift(1)

    frame["bull_break"] = (frame["close"] > frame["upper_level"]) & (prev_close <= prev_upper)
    frame["bear_break"] = (frame["close"] < frame["lower_level"]) & (prev_close >= prev_lower)

    frame["fvg_bull"] = frame["low"] > frame["high"].shift(2)
    frame["fvg_bear"] = frame["high"] < frame["low"].shift(2)

    frame["fvg_bull_recent"] = frame["fvg_bull"].astype(int).rolling(fvg_lookback, min_periods=1).sum()
    frame["fvg_bear_recent"] = frame["fvg_bear"].astype(int).rolling(fvg_lookback, min_periods=1).sum()

    frame["buy_signal"] = frame["bull_break"] & (frame["fvg_bull_recent"] > 0)
    frame["sell_signal"] = frame["bear_break"] & (frame["fvg_bear_recent"] > 0)

    frame["demand_zone"] = pd.NA
    frame.loc[frame["bull_break"], "demand_zone"] = frame.loc[frame["bull_break"], ["low"]].join(
        frame["low"].shift(1).rename("prior_low")
    ).min(axis=1)

    frame["supply_zone"] = pd.NA
    frame.loc[frame["bear_break"], "supply_zone"] = frame.loc[frame["bear_break"], ["high"]].join(
        frame["high"].shift(1).rename("prior_high")
    ).max(axis=1)

    final_buy: list[bool] = []
    final_sell: list[bool] = []
    last_signal_direction = 0

    for _, row in frame.iterrows():
        raw_buy = bool(row["buy_signal"])
        raw_sell = bool(row["sell_signal"])

        if prevent_repeated:
            is_buy = raw_buy and last_signal_direction <= 0
            is_sell = raw_sell and last_signal_direction >= 0
        else:
            is_buy = raw_buy
            is_sell = raw_sell

        final_buy.append(is_buy)
        final_sell.append(is_sell)

        if is_buy:
            last_signal_direction = 1
        elif is_sell:
            last_signal_direction = -1

    frame["final_buy"] = final_buy
    frame["final_sell"] = final_sell
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
