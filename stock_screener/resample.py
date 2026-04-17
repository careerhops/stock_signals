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

    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").set_index("date")

    weekly = frame.resample(weekly_anchor).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    weekly = weekly.dropna(subset=["open", "high", "low", "close"]).reset_index()

    if use_completed_weeks_only and not weekly.empty:
        today = pd.Timestamp(date.today())
        weekly = weekly[weekly["date"] <= today]

    return weekly
