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
    all_signal_dates = pd.to_datetime(frame["date"], errors="coerce") if not frame.empty else pd.Series(dtype="datetime64[ns]")

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

    max_signal_age_bars = signal_cfg.get("max_signal_age_bars")
    if max_signal_age_bars is not None and not frame.empty and not all_signal_dates.empty:
        max_signal_date = all_signal_dates.max()
        max_age = max(int(max_signal_age_bars), 1)
        scan_timeframe = str(config.get("data", {}).get("scan_timeframe", "1W")).upper()
        if scan_timeframe == "1W":
            min_allowed_date = max_signal_date - pd.Timedelta(weeks=max_age - 1)
        else:
            min_allowed_date = max_signal_date - pd.Timedelta(days=max_age - 1)
        frame = frame[pd.to_datetime(frame["date"], errors="coerce") >= min_allowed_date]

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
