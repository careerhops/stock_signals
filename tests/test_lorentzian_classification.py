from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from stock_screener.lorentzian_classification import (
    LorentzianFeatureConfig,
    LorentzianSettings,
    lorentzian_trade_stats,
    run_lorentzian_classification,
)


class LorentzianClassificationTests(unittest.TestCase):
    def test_trade_stats_counts_wins_losses_and_early_flips(self) -> None:
        frame = pd.DataFrame(
            [
                {"date": "2026-01-01", "open": 100.0, "close": 100.0, "start_long_trade": False, "end_long_trade": False, "start_short_trade": False, "end_short_trade": False, "is_early_signal_flip": False},
                {"date": "2026-01-02", "open": 100.0, "close": 101.0, "start_long_trade": True, "end_long_trade": False, "start_short_trade": False, "end_short_trade": False, "is_early_signal_flip": True},
                {"date": "2026-01-03", "open": 108.0, "close": 107.0, "start_long_trade": False, "end_long_trade": True, "start_short_trade": False, "end_short_trade": False, "is_early_signal_flip": False},
                {"date": "2026-01-04", "open": 90.0, "close": 91.0, "start_long_trade": False, "end_long_trade": False, "start_short_trade": True, "end_short_trade": False, "is_early_signal_flip": False},
                {"date": "2026-01-05", "open": 95.0, "close": 96.0, "start_long_trade": False, "end_long_trade": False, "start_short_trade": False, "end_short_trade": True, "is_early_signal_flip": False},
            ]
        )

        stats = lorentzian_trade_stats(frame, LorentzianSettings())

        self.assertEqual(stats["total_trades"], 2)
        self.assertEqual(stats["total_wins"], 1)
        self.assertEqual(stats["total_losses"], 1)
        self.assertEqual(stats["total_early_signal_flips"], 1)
        self.assertAlmostEqual(stats["win_rate"], 50.0)
        self.assertAlmostEqual(stats["win_rate_raw"], 0.5)
        self.assertAlmostEqual(stats["win_loss_ratio"], 0.5)
        self.assertAlmostEqual(stats["wins_over_losses_ratio"], 1.0)

    def test_run_lorentzian_classification_produces_expected_columns(self) -> None:
        periods = 120
        dates = pd.date_range("2025-01-01", periods=periods, freq="D")
        base = np.linspace(100.0, 140.0, periods)
        wave = 3.0 * np.sin(np.linspace(0, 10, periods))
        close = base + wave
        open_ = close - 0.5
        high = close + 1.5
        low = close - 1.5
        volume = np.linspace(100_000, 200_000, periods)

        candles = pd.DataFrame(
            {
                "date": dates,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

        settings = LorentzianSettings(
            neighbors_count=5,
            max_bars_back=60,
            feature_count=3,
            features=(
                LorentzianFeatureConfig("RSI", 14, 2),
                LorentzianFeatureConfig("WT", 10, 11),
                LorentzianFeatureConfig("CCI", 20, 2),
            ),
        )

        result = run_lorentzian_classification(candles, settings)

        expected_columns = {
            "prediction",
            "signal",
            "start_long_trade",
            "start_short_trade",
            "end_long_trade",
            "end_short_trade",
            "is_early_signal_flip",
            "kernel_estimate",
        }
        self.assertTrue(expected_columns.issubset(result.columns))
        self.assertEqual(len(result), periods)
        self.assertTrue(result["prediction"].notna().any())


if __name__ == "__main__":
    unittest.main()
