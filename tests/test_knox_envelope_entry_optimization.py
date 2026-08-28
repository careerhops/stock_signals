from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from stock_screener.knox_envelope_entry_optimization import (
    EntryStrategyParameters,
    _cmf_mask,
    _entry_outcomes,
    _fast_pine_rsi,
    _top_parameters,
    _wilson_interval,
)
from stock_screener.knox_envelope_study import _pine_rsi


class KnoxEnvelopeEntryOptimizationTests(unittest.TestCase):
    def test_fast_rsi_matches_pine_rsi(self) -> None:
        close = pd.Series([100.0 + np.sin(index / 3.0) * 5.0 for index in range(100)])
        expected = _pine_rsi(close, 14)
        actual = _fast_pine_rsi(close, 14)
        np.testing.assert_allclose(actual, expected, equal_nan=True, rtol=1e-12)

    def test_entry_uses_next_session_high_and_measures_later_sessions(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=25, freq="B"),
                "open": 100.0,
                "high": [100.0, 110.0] + [118.0] * 23,
                "low": [99.0, 90.0] + [108.0] * 23,
                "close": [100.0, 105.0] + [115.0] * 23,
                "volume": 1000.0,
            }
        )
        rows = _entry_outcomes(
            frame,
            np.array([True] + [False] * 24),
            symbol="TEST",
            exchange="NSE",
            parameter_name="P",
            start_ts=pd.Timestamp("2024-01-01"),
            end_ts=pd.Timestamp("2024-12-31"),
            horizon=20,
            round_trip_cost_pct=0.0,
            lower_distance=np.zeros(25),
            cmf=pd.Series(0.1, index=frame.index),
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["entry_price"], 110.0)
        self.assertTrue(rows[0]["target_7_within_10"])
        self.assertFalse(rows[0]["stop_5_within_20"])

    def test_cmf_band_rejects_extreme_positive_value(self) -> None:
        item = EntryStrategyParameters(cmf_mode="band", cmf_min=0.0, cmf_max=0.4)
        mask = _cmf_mask(pd.Series([-0.1, 0.1, 0.4, 0.82]), item)
        self.assertEqual(mask.tolist(), [False, True, True, False])

    def test_parameter_selection_uses_validation_not_holdout(self) -> None:
        first = EntryStrategyParameters(knox_lookback=50)
        second = EntryStrategyParameters(knox_lookback=200)
        stats = pd.DataFrame(
            [
                {
                    "parameter_name": first.name,
                    "cohort": "VALIDATION",
                    "trades": 200,
                    "entry_score": 70.0,
                    "target_7_before_stop_5_pct": 60.0,
                    "target_7_within_20_pct": 70.0,
                    "median_return_20_pct": 5.0,
                },
                {
                    "parameter_name": second.name,
                    "cohort": "VALIDATION",
                    "trades": 200,
                    "entry_score": 60.0,
                    "target_7_before_stop_5_pct": 50.0,
                    "target_7_within_20_pct": 60.0,
                    "median_return_20_pct": 5.0,
                },
                {
                    "parameter_name": second.name,
                    "cohort": "HOLDOUT",
                    "trades": 200,
                    "entry_score": 99.0,
                    "target_7_before_stop_5_pct": 99.0,
                    "target_7_within_20_pct": 99.0,
                    "median_return_20_pct": 20.0,
                },
            ]
        )

        selected = _top_parameters(stats, (first, second), 100, 1)

        self.assertEqual(selected, [first])

    def test_wilson_interval_contains_observed_probability(self) -> None:
        low, high = _wilson_interval(88, 379)

        self.assertLess(low, 88 / 379)
        self.assertGreater(high, 88 / 379)
        self.assertAlmostEqual(low, 0.1925, places=4)
        self.assertAlmostEqual(high, 0.2773, places=4)


if __name__ == "__main__":
    unittest.main()
