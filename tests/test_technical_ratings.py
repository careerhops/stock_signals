from __future__ import annotations

import unittest

import pandas as pd

from stock_screener.strategy.technical_ratings import (
    TECHNICAL_RATING_COMPONENTS,
    MA_SIGNAL_COLUMNS,
    OSCILLATOR_SIGNAL_COLUMNS,
    compare_technical_rating_snapshot,
    compute_technical_ratings,
    latest_technical_rating,
    latest_technical_rating_audit,
    rating_action,
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

    def test_rating_action_maps_component_scores_to_panel_actions(self) -> None:
        self.assertEqual(rating_action(1), "Buy")
        self.assertEqual(rating_action(0), "Neutral")
        self.assertEqual(rating_action(-1), "Sell")
        self.assertEqual(rating_action(pd.NA), "NA")

    def test_component_catalog_matches_tradingview_panel(self) -> None:
        expected_names = [
            "Relative Strength Index (14)",
            "Stochastic %K (14, 3, 3)",
            "Commodity Channel Index (20)",
            "Average Directional Index (14)",
            "Awesome Oscillator",
            "Momentum (10)",
            "MACD Level (12, 26)",
            "Stochastic RSI Fast (3, 3, 14, 14)",
            "Williams Percent Range (14)",
            "Bull Bear Power",
            "Ultimate Oscillator (7, 14, 28)",
            "Exponential Moving Average (10)",
            "Simple Moving Average (10)",
            "Exponential Moving Average (20)",
            "Simple Moving Average (20)",
            "Exponential Moving Average (30)",
            "Simple Moving Average (30)",
            "Exponential Moving Average (50)",
            "Simple Moving Average (50)",
            "Exponential Moving Average (100)",
            "Simple Moving Average (100)",
            "Exponential Moving Average (200)",
            "Simple Moving Average (200)",
            "Ichimoku Base Line (9, 26, 52, 26)",
            "Volume Weighted Moving Average (20)",
            "Hull Moving Average (9)",
        ]
        self.assertEqual(len(TECHNICAL_RATING_COMPONENTS), 26)
        self.assertEqual([component.name for component in TECHNICAL_RATING_COMPONENTS], expected_names)
        self.assertEqual(TECHNICAL_RATING_COMPONENTS[9].value_column, "bull_bear_power_50")

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
        for component in TECHNICAL_RATING_COMPONENTS:
            self.assertIn(component.value_column, result.columns)
            self.assertIn(component.action_column, result.columns)

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

    def test_macd_level_uses_macd_minus_signal_for_panel_value(self) -> None:
        candles = _sample_candles(320)

        result = compute_technical_ratings(candles)
        latest = result.iloc[-1]

        self.assertAlmostEqual(
            float(latest["macd_level_12_26"]),
            float(latest["macd_12_26_9"] - latest["macd_signal_12_26_9"]),
        )

    def test_latest_technical_rating_audit_returns_all_components(self) -> None:
        candles = _sample_candles(320)

        audit = latest_technical_rating_audit(candles)

        self.assertEqual(len(audit["components"]), 26)
        self.assertEqual(sum(1 for component in audit["components"] if component["group"] == "Oscillators"), 11)
        self.assertEqual(sum(1 for component in audit["components"] if component["group"] == "Moving Averages"), 15)
        self.assertIn(audit["components"][0]["action"], {"Buy", "Sell", "Neutral", "NA"})

    def test_compare_snapshot_detects_matches_and_mismatches(self) -> None:
        candles = _sample_candles(320)
        audit = latest_technical_rating_audit(candles)
        opposite_action = {
            "Buy": "Sell",
            "Sell": "Buy",
            "Neutral": "Buy",
            "NA": "Buy",
        }
        expected_components = [
            {"name": audit["components"][0]["name"], "action": audit["components"][0]["action"], "value": audit["components"][0]["value"]},
            {
                "name": audit["components"][1]["name"],
                "action": opposite_action.get(str(audit["components"][1]["action"]), "Buy"),
                "value": audit["components"][1]["value"],
            },
        ]

        comparison = compare_technical_rating_snapshot(candles, expected_components)

        self.assertTrue(bool(comparison.iloc[0]["action_matches"]))
        self.assertFalse(bool(comparison.iloc[1]["action_matches"]))
        self.assertTrue(bool(comparison.iloc[0]["value_matches"]))


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
