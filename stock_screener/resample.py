from __future__ import annotations

from datetime import date

import pandas as pd


def resample_daily_to_weekly(
    daily: pd.DataFrame,
    weekly_anchor: str = "W-FRI",
    use_completed_weeks_only: bool = True,
) -> pd.DataFrame:
    if daily.empty:
        return daily

    today = pd.Timestamp(date.today()).normalize()
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame[frame["date"].dt.normalize() <= today]
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame.empty:
        return frame

    # TradingView's weekly chart for NSE symbols is Monday-labeled and the
    # current in-progress week is visible on the chart. To match that behavior,
    # build calendar weeks keyed by their Monday start instead of relying on
    # pandas' W-MON period end semantics.
    if str(weekly_anchor).strip().upper() == "W-MON":
        frame["week_key"] = (
            frame["date"] - pd.to_timedelta(frame["date"].dt.weekday, unit="D")
        ).dt.normalize()
        weekly = (
            frame.groupby("week_key", as_index=False)
            .agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                volume=("volume", "sum"),
            )
            .rename(columns={"week_key": "date"})
        )
        return weekly.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

    weekly = (
        frame.set_index("date")
        .resample(weekly_anchor)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )

    if use_completed_weeks_only and not weekly.empty:
        last_week_end = pd.Timestamp(weekly.iloc[-1]["date"]).normalize()
        if today < last_week_end:
            weekly = weekly.iloc[:-1].copy()

    return weekly.reset_index(drop=True)
