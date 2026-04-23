from __future__ import annotations

import unittest

import pandas as pd

from stock_screener.strategy.technical_ratings import (
    MA_SIGNAL_COLUMNS,
    OSCILLATOR_SIGNAL_COLUMNS,
    compute_technical_ratings,
    latest_technical_rating,
    rating_status,
)


class TechnicalRatingsTests(unittest.TestCase):
    def test_rating_status_matches_tradingview_thresholds(self) -> None:
        self.assertEqual(rating_status(0.75), "Strong Buy")
        self.assertEqual(rating_status(0.2), "Buy")
        self.assertEqual(rating_status(0.0), "Neutral")
        self.assertEqual(rating_status(-0.2), "Sell")
        self.assertEqual(rating_status(-0.75), "Strong Sell")
        self.assertEqual(rating_status(pd.NA), "NA")

    def test_compute_technical_ratings_adds_26_signal_columns(self) -> None:
        candles = _sample_candles(260)

        result = compute_technical_ratings(candles)

        self.assertEqual(len(MA_SIGNAL_COLUMNS), 15)
        self.assertEqual(len(OSCILLATOR_SIGNAL_COLUMNS), 11)
        for column in (*MA_SIGNAL_COLUMNS, *OSCILLATOR_SIGNAL_COLUMNS):
            self.assertIn(column, result.columns)
        self.assertIn("rating", result.columns)
        self.assertIn("rating_status", result.columns)
        self.assertIn("ma_rating", result.columns)
        self.assertIn("oscillator_rating", result.columns)

    def test_uptrend_sample_produces_bullish_ma_rating(self) -> None:
        candles = _sample_candles(320)

        result = compute_technical_ratings(candles)
        latest = result.iloc[-1]

        self.assertGreaterEqual(float(latest["ma_rating"]), 0.9)
        self.assertIn(latest["ma_rating_status"], {"Buy", "Strong Buy"})
        self.assertEqual(int(latest["ma_indicator_count"]), 15)
        self.assertEqual(int(latest["oscillator_indicator_count"]), 11)

    def test_latest_technical_rating_returns_summary_payload(self) -> None:
        candles = _sample_candles(320)

        summary = latest_technical_rating(candles)

        self.assertIn("rating", summary)
        self.assertIn("rating_status", summary)
        self.assertIn(summary["rating_status"], {"Strong Buy", "Buy", "Neutral", "Sell", "Strong Sell"})

    def test_unavailable_indicators_are_excluded_not_counted_as_neutral(self) -> None:
        candles = _sample_candles(150)

        result = compute_technical_ratings(candles)
        latest = result.iloc[-1]

        self.assertTrue(pd.isna(latest["rating_sma_200"]))
        self.assertEqual(int(latest["ma_indicator_count"]), 14)
        self.assertGreater(float(latest["ma_rating"]), 0.9)


def _sample_candles(length: int) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    base_price = 100.0
    for index in range(length):
        close = base_price + (index * 1.2)
        open_ = close - 0.6
        high = close + 1.4 + ((index % 5) * 0.1)
        low = open_ - 1.0 - ((index % 3) * 0.1)
        rows.append(
            {
                "date": (pd.Timestamp("2024-01-01") + pd.Timedelta(days=index)).strftime("%Y-%m-%d"),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": 100000 + (index * 250),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
