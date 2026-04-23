from __future__ import annotations

from datetime import date
from unittest.mock import patch
import unittest

import pandas as pd

from stock_screener.resample import resample_daily_to_weekly


class WeeklyResampleTests(unittest.TestCase):
    def test_excludes_current_incomplete_week_on_monday(self) -> None:
        daily = pd.DataFrame(
            [
                {"date": "2026-04-17", "open": 100, "high": 110, "low": 99, "close": 108, "volume": 1000},
                {"date": "2026-04-20", "open": 109, "high": 112, "low": 107, "close": 111, "volume": 1200},
            ]
        )

        with patch("stock_screener.resample.date") as mock_date:
            mock_date.today.return_value = date(2026, 4, 20)
            weekly = resample_daily_to_weekly(daily, "W-FRI", use_completed_weeks_only=True)

        self.assertEqual(weekly["date"].dt.strftime("%Y-%m-%d").tolist(), ["2026-04-17"])

    def test_includes_current_week_on_friday(self) -> None:
        daily = pd.DataFrame(
            [
                {"date": "2026-04-17", "open": 100, "high": 110, "low": 99, "close": 108, "volume": 1000},
                {"date": "2026-04-20", "open": 109, "high": 112, "low": 107, "close": 111, "volume": 1200},
                {"date": "2026-04-24", "open": 111, "high": 116, "low": 110, "close": 115, "volume": 1300},
            ]
        )

        with patch("stock_screener.resample.date") as mock_date:
            mock_date.today.return_value = date(2026, 4, 24)
            weekly = resample_daily_to_weekly(daily, "W-FRI", use_completed_weeks_only=True)

        self.assertEqual(weekly["date"].dt.strftime("%Y-%m-%d").tolist(), ["2026-04-17", "2026-04-24"])
        self.assertEqual(float(weekly.iloc[-1]["close"]), 115.0)


if __name__ == "__main__":
    unittest.main()
