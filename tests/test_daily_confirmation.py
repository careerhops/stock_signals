from __future__ import annotations

import unittest

import pandas as pd

from stock_screener.strategy.daily_confirmation import compute_daily_confirmations, latest_daily_confirmation


class DailyConfirmationTests(unittest.TestCase):
    def test_daily_ema_stack_and_obv_confirmations_are_true_for_accumulating_uptrend(self) -> None:
        candles = _trend_candles(260, direction=1)

        latest = latest_daily_confirmation(candles)

        self.assertTrue(bool(latest["daily_ema_stack_confirmation"]))
        self.assertGreater(float(latest["daily_ema50_slope"]), 0)
        self.assertGreater(float(latest["daily_ema100_slope"]), 0)
        self.assertGreater(float(latest["daily_ema200_slope"]), 0)
        self.assertTrue(bool(latest["daily_obv_confirmation"]))
        self.assertGreater(float(latest["daily_obv_slope_20d"]), 0)

    def test_daily_ema_stack_and_obv_confirmations_are_false_for_distribution_downtrend(self) -> None:
        candles = _trend_candles(260, direction=-1)

        latest = latest_daily_confirmation(candles)

        self.assertFalse(bool(latest["daily_ema_stack_confirmation"]))
        self.assertFalse(bool(latest["daily_obv_confirmation"]))
        self.assertLess(float(latest["daily_obv_slope_20d"]), 0)

    def test_compute_daily_confirmations_requires_daily_ohlcv_columns(self) -> None:
        with self.assertRaises(ValueError):
            compute_daily_confirmations(pd.DataFrame([{"date": "2025-01-01", "close": 100.0}]))


def _trend_candles(length: int, direction: int) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    start = 100.0 if direction > 0 else 400.0
    for index in range(length):
        close = start + (direction * index * 1.5)
        rows.append(
            {
                "date": (pd.Timestamp("2025-01-01") + pd.Timedelta(days=index)).strftime("%Y-%m-%d"),
                "open": close - (direction * 0.4),
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 100000 + (index * 1000),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
