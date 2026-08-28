from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from stock_screener.knox_minervini_backtest_study import (
    SignalVariant,
    _bearish_knoxville,
    _bullish_knoxville,
    _simulate_long_trade,
    calculate_knox_minervini_features,
)
from stock_screener.knox_envelope_study import _pine_rsi


class KnoxMinerviniBacktestTests(unittest.TestCase):
    def test_bullish_knoxville_matches_published_reference_loop(self) -> None:
        close = pd.Series(
            [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 80, 82, 90],
            dtype=float,
        )
        low = close - 1.0
        low.iloc[-1] = 70.0
        momentum = close - close.shift(2)
        result = _bullish_knoxville(low, momentum, _pine_rsi(close, 3), 10)

        self.assertTrue(bool(result.iloc[-1]))

    def test_same_bar_stop_and_target_uses_conservative_stop_first(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=3, freq="B"),
                "open": [100.0, 100.0, 100.0],
                "high": [101.0, 108.0, 101.0],
                "low": [99.0, 95.0, 99.0],
                "close": [100.0, 102.0, 100.0],
                "volume": 1000,
            }
        )
        trade = _simulate_long_trade(
            frame,
            signal_index=0,
            entry_index=1,
            planned_exit_index=2,
            target_pct=6.0,
            stop_pct=4.0,
            round_trip_cost_pct=0.35,
        )

        self.assertIsNotNone(trade)
        self.assertEqual(trade["exit_reason"], "STOP")
        self.assertAlmostEqual(float(trade["net_return_pct"]), -4.35)

    def test_bearish_knoxville_matches_mirrored_reference_loop(self) -> None:
        close = pd.Series(
            [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 120, 118, 110],
            dtype=float,
        )
        high = close + 1.0
        high.iloc[-1] = 130.0
        momentum = close - close.shift(2)
        result = _bearish_knoxville(high, momentum, _pine_rsi(close, 3), 10)

        self.assertTrue(bool(result.iloc[-1]))

    def test_pure_mode_does_not_require_minervini_pass(self) -> None:
        close = [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 80, 82, 90]
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=len(close), freq="B"),
                "open": close,
                "high": [value + 1 for value in close],
                "low": [value - 1 for value in close],
                "close": close,
                "volume": 1000,
            }
        )
        frame.loc[len(frame) - 1, "low"] = 70
        benchmark = pd.DataFrame(
            {
                "date": frame["date"],
                "weighted_performance": 0.0,
                "market_bullish": False,
            }
        )
        variant = SignalVariant("PURE", 10, 5, 0.0)
        pure_knox = pd.Series(False, index=frame.index)
        pure_knox.iloc[-1] = True
        with patch(
            "stock_screener.knox_minervini_backtest_study._bullish_knoxville",
            return_value=pure_knox,
        ):
            result = calculate_knox_minervini_features(
                frame,
                benchmark,
                signal_variants=[variant],
                band_proximity_pct=10.0,
                use_minervini_filter=False,
            )

        self.assertFalse(bool(result.iloc[-1]["strict_minervini_pass"]))
        self.assertTrue(bool(result.iloc[-1]["signal_PURE"]))


if __name__ == "__main__":
    unittest.main()
