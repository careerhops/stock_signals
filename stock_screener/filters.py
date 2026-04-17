from __future__ import annotations

from typing import Any

import pandas as pd


def apply_filters(signals: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    filters_cfg = config.get("filters", {})
    if not filters_cfg.get("enabled", True) or signals.empty:
        return signals

    frame = signals.copy()

    signal_cfg = filters_cfg.get("signal", {})
    frame = frame[frame["signal"].isin(["BUY", "SELL"])]

    if signal_cfg.get("latest_only", True) and not frame.empty:
        frame = frame.copy()
        frame["date_sort"] = pd.to_datetime(frame["date"], errors="coerce")
        group_columns = [column for column in ("exchange", "symbol") if column in frame.columns]
        if group_columns:
            latest_index = frame.sort_values("date_sort").groupby(group_columns, dropna=False).tail(1).index
            frame = frame.loc[latest_index].copy()
        else:
            max_date = frame["date_sort"].max()
            frame = frame[frame["date_sort"] == max_date].copy()
        frame = frame.drop(columns=["date_sort"], errors="ignore")

    direction = str(signal_cfg.get("direction", "BUY")).upper()
    if direction in {"BUY", "SELL"}:
        frame = frame[frame["signal"] == direction]

    price_cfg = filters_cfg.get("price", {})
    if price_cfg.get("enabled", False):
        frame = frame[(frame["close"] >= float(price_cfg.get("min", 0))) & (frame["close"] <= float(price_cfg.get("max", 10**9)))]

    liquidity_cfg = filters_cfg.get("liquidity", {})
    if liquidity_cfg.get("enabled", False):
        min_volume = float(liquidity_cfg.get("min_avg_volume_20", 0))
        min_value = float(liquidity_cfg.get("min_avg_traded_value_20", 0))
        frame = frame[(frame["avg_volume_20"] >= min_volume) & (frame["avg_traded_value_20"] >= min_value)]

    trend_cfg = filters_cfg.get("trend", {})
    if trend_cfg.get("enabled", False):
        if trend_cfg.get("require_close_above_ema_50", False):
            frame = frame[frame["close"] > frame["ema_50"]]
        if trend_cfg.get("require_close_above_ema_200", False):
            frame = frame[frame["close"] > frame["ema_200"]]

    return frame.reset_index(drop=True)
