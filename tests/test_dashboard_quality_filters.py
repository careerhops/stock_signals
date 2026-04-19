from __future__ import annotations

import unittest

import pandas as pd

from stock_screener.web.main import _apply_signal_quality_filters


class DashboardQualityFilterTests(unittest.TestCase):
    def test_quality_filters_require_volume_trend_and_return_threshold(self) -> None:
        signals = pd.DataFrame(
            [
                {
                    "symbol": "PASS",
                    "volume_confirmation": True,
                    "trend_confirmation": True,
                    "median_pair_return_last_3_pct": 12.5,
                },
                {
                    "symbol": "LOWRET",
                    "volume_confirmation": True,
                    "trend_confirmation": True,
                    "median_pair_return_last_3_pct": 2.0,
                },
                {
                    "symbol": "NOVOL",
                    "volume_confirmation": False,
                    "trend_confirmation": True,
                    "median_pair_return_last_3_pct": 20.0,
                },
                {
                    "symbol": "NOTREND",
                    "volume_confirmation": True,
                    "trend_confirmation": False,
                    "median_pair_return_last_3_pct": 20.0,
                },
            ]
        )

        filtered = _apply_signal_quality_filters(
            signals,
            require_volume_confirmation=True,
            require_trend_confirmation=True,
            return_metric="median_3",
            min_pair_return=5.0,
        )

        self.assertEqual(filtered["symbol"].tolist(), ["PASS"])

    def test_quality_filters_can_use_last_completed_pair_return(self) -> None:
        signals = pd.DataFrame(
            [
                {
                    "symbol": "PASS",
                    "volume_confirmation": True,
                    "trend_confirmation": True,
                    "prior_pair_return_last_1_pct": 8.0,
                    "median_pair_return_last_3_pct": -2.0,
                },
            ]
        )

        filtered = _apply_signal_quality_filters(
            signals,
            require_volume_confirmation=False,
            require_trend_confirmation=False,
            return_metric="last_1",
            min_pair_return=5.0,
        )

        self.assertEqual(filtered["symbol"].tolist(), ["PASS"])


if __name__ == "__main__":
    unittest.main()
