from __future__ import annotations

from typing import Any

import pandas as pd


EMA_STACK_COLUMNS = (
    "daily_ema_20",
    "daily_ema_50",
    "daily_ema_100",
    "daily_ema_200",
    "daily_ema50_slope",
    "daily_ema100_slope",
    "daily_ema200_slope",
    "daily_ema_stack_confirmation",
)
OBV_COLUMNS = (
    "daily_obv",
    "daily_obv_slope_20d",
    "daily_obv_confirmation",
)
DAILY_CONFIRMATION_COLUMNS = EMA_STACK_COLUMNS + OBV_COLUMNS


def compute_daily_confirmations(candles: pd.DataFrame) -> pd.DataFrame:
    if candles.empty:
        return candles.copy()

    required = {"date", "close", "volume"}
    missing = required - set(candles.columns)
    if missing:
        raise ValueError(f"Missing required daily confirmation columns: {sorted(missing)}")

    frame = candles.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.sort_values("date").reset_index(drop=True)

    close = pd.to_numeric(frame["close"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce").fillna(0)

    frame["daily_ema_20"] = close.ewm(span=20, adjust=False).mean()
    frame["daily_ema_50"] = close.ewm(span=50, adjust=False).mean()
    frame["daily_ema_100"] = close.ewm(span=100, adjust=False).mean()
    frame["daily_ema_200"] = close.ewm(span=200, adjust=False).mean()
    frame["daily_ema50_slope"] = frame["daily_ema_50"].diff()
    frame["daily_ema100_slope"] = frame["daily_ema_100"].diff()
    frame["daily_ema200_slope"] = frame["daily_ema_200"].diff()

    frame["daily_ema_stack_confirmation"] = (
        (close > frame["daily_ema_20"])
        & (frame["daily_ema_20"] > frame["daily_ema_50"])
        & (frame["daily_ema_50"] > frame["daily_ema_100"])
        & (frame["daily_ema_100"] > frame["daily_ema_200"])
        & (frame["daily_ema50_slope"] > 0)
        & (frame["daily_ema100_slope"] > 0)
        & (frame["daily_ema200_slope"] > 0)
    )

    direction = close.diff()
    obv_delta = pd.Series(0.0, index=frame.index)
    obv_delta.loc[direction > 0] = volume.loc[direction > 0]
    obv_delta.loc[direction < 0] = -volume.loc[direction < 0]
    frame["daily_obv"] = obv_delta.cumsum()
    frame["daily_obv_slope_20d"] = _rolling_slope(frame["daily_obv"], 20)
    frame["daily_obv_confirmation"] = frame["daily_obv_slope_20d"] > 0

    return frame


def latest_daily_confirmation(candles: pd.DataFrame) -> dict[str, Any]:
    confirmations = compute_daily_confirmations(candles)
    if confirmations.empty:
        return {column: pd.NA for column in DAILY_CONFIRMATION_COLUMNS}
    latest = confirmations.iloc[-1]
    return {column: latest.get(column, pd.NA) for column in DAILY_CONFIRMATION_COLUMNS}


def add_latest_daily_confirmation_columns(frame: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    enriched = frame.copy()
    latest = latest_daily_confirmation(daily)
    for column, value in latest.items():
        enriched[column] = value
    if not daily.empty and {"date", "close"}.issubset(daily.columns):
        latest_daily = daily.copy()
        latest_daily["date"] = pd.to_datetime(latest_daily["date"], errors="coerce")
        latest_daily = latest_daily.dropna(subset=["date"]).sort_values("date")
        if not latest_daily.empty:
            latest_row = latest_daily.iloc[-1]
            enriched["latest_close"] = pd.to_numeric(pd.Series([latest_row.get("close")]), errors="coerce").iloc[0]
            enriched["latest_close_date"] = latest_row.get("date", pd.NA)
    enriched["trend_confirmation"] = enriched["daily_ema_stack_confirmation"]
    enriched["obv_confirmation"] = enriched["daily_obv_confirmation"]
    return enriched


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = pd.Series(range(window), dtype="float64").to_numpy()
    x_mean = float(x.mean())
    centered_x = x - x_mean
    denominator = float((centered_x**2).sum())

    def slope(values: pd.Series) -> float:
        y = pd.Series(values, dtype="float64").to_numpy()
        if pd.isna(y).any() or denominator == 0:
            return float("nan")
        return float((centered_x * (y - float(y.mean()))).sum() / denominator)

    return series.rolling(window, min_periods=window).apply(slope, raw=False)
