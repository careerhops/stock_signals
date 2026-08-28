from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from stock_screener.data.storage import Storage
from stock_screener.knox_envelope_pair_backtest import (
    PairStrategyParameters,
    _candidate_symbols,
    _envelope_proximity,
    _entry_quality_mask,
    _select_recommended_parameter,
    simulate_paired_signals,
)


class KnoxEnvelopePairBacktestTests(unittest.TestCase):
    def test_candidate_symbols_exclude_etf_aliases_using_instrument_name(self) -> None:
        with TemporaryDirectory() as directory:
            storage = Storage(Path(directory))
            storage.save_instruments(
                pd.DataFrame(
                    [
                        {"exchange": "NSE", "tradingsymbol": "MIDCAPBETA", "name": "UTI NIFTY MIDCAP 150 ETF"},
                        {"exchange": "NSE", "tradingsymbol": "MIDBANKADD", "name": "DSPAMC - MIDBANKADD"},
                        {"exchange": "NSE", "tradingsymbol": "RELIANCE", "name": "RELIANCE INDUSTRIES"},
                    ]
                )
            )

            actual = _candidate_symbols(storage, "NSE", ["MIDCAPBETA", "MIDBANKADD", "RELIANCE"])

            self.assertEqual(actual, ["RELIANCE"])

    def test_uses_next_day_high_for_entry_and_next_day_low_for_exit(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=6, freq="B"),
                "open": [100, 102, 105, 110, 115, 120],
                "high": [105, 110, 112, 118, 122, 130],
                "low": [95, 100, 101, 104, 108, 121],
                "close": [101, 106, 108, 115, 120, 125],
                "volume": 1000,
            }
        )
        entry = np.array([False, True, False, False, False, False])
        exit_signal = np.array([False, False, False, False, True, False])
        trades, open_position = simulate_paired_signals(
            frame,
            entry_signals=entry,
            exit_signals=exit_signal,
            target_pct=10.0,
            round_trip_cost_pct=0.0,
        )

        self.assertIsNone(open_position)
        self.assertEqual(len(trades), 1)
        trade = trades.iloc[0]
        self.assertEqual(float(trade["entry_price"]), 112.0)
        self.assertEqual(float(trade["exit_price"]), 121.0)
        self.assertEqual(pd.Timestamp(trade["entry_date"]), frame.iloc[2]["date"])
        self.assertEqual(pd.Timestamp(trade["exit_date"]), frame.iloc[5]["date"])
        self.assertFalse(bool(trade["target_10_hit"]))
        self.assertTrue(pd.isna(trade["bars_to_target"]))

    def test_target_hit_uses_highs_before_exit_execution_day(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=5, freq="B"),
                "open": 100.0,
                "high": [100.0, 100.0, 111.0, 105.0, 130.0],
                "low": [99.0, 99.0, 99.0, 99.0, 95.0],
                "close": 100.0,
                "volume": 1000,
            }
        )
        entry = np.array([True, False, False, False, False])
        exit_signal = np.array([False, False, False, True, False])
        trades, _ = simulate_paired_signals(
            frame,
            entry_signals=entry,
            exit_signals=exit_signal,
            round_trip_cost_pct=0.0,
        )

        self.assertTrue(bool(trades.iloc[0]["target_10_hit"]))
        self.assertAlmostEqual(float(trades.iloc[0]["mfe_pct"]), 11.0)
        self.assertEqual(float(trades.iloc[0]["bars_to_target"]), 2.0)

    def test_trade_crossing_large_overnight_discontinuity_is_flagged(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=5, freq="B"),
                "open": [100.0, 100.0, 50.0, 51.0, 52.0],
                "high": [101.0, 101.0, 52.0, 53.0, 54.0],
                "low": [99.0, 99.0, 49.0, 50.0, 51.0],
                "close": [100.0, 100.0, 51.0, 52.0, 53.0],
                "volume": 1000,
            }
        )
        entry = np.array([True, False, False, False, False])
        exit_signal = np.array([False, False, False, True, False])
        trades, _ = simulate_paired_signals(
            frame,
            entry_signals=entry,
            exit_signals=exit_signal,
            round_trip_cost_pct=0.0,
        )

        self.assertFalse(bool(trades.iloc[0]["data_quality_pass"]))
        self.assertEqual(trades.iloc[0]["data_quality_reason"], "overnight_price_discontinuity")

    def test_band_proximity_uses_low_and_high_not_close(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=5, freq="B"),
                "open": 100.0,
                "high": 110.0,
                "low": 90.0,
                "close": 100.0,
                "volume": 1000,
            }
        )
        near_lower, near_upper, lower_distance, upper_distance = _envelope_proximity(
            frame,
            envelope_length=5,
            envelope_percent=10.0,
            envelope_ma_type="SMA",
            proximity_pct=0.1,
        )

        self.assertTrue(bool(near_lower[-1]))
        self.assertTrue(bool(near_upper[-1]))
        self.assertAlmostEqual(float(lower_distance[-1]), 0.0)
        self.assertAlmostEqual(float(upper_distance[-1]), 0.0)

    def test_entry_quality_mask_uses_cmf_and_prior_volume_average(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=21, freq="B"),
                "open": 100.0,
                "high": 110.0,
                "low": 90.0,
                "close": [100.0] * 20 + [109.0],
                "volume": [100.0] * 20 + [300.0],
            }
        )
        mask = _entry_quality_mask(
            frame,
            cmf_length=20,
            min_cmf=0.0,
            min_rvol20=3.0,
            obv_accumulation_days=None,
        )

        self.assertTrue(bool(mask[-1]))
        self.assertFalse(bool(mask[-2]))

    def test_entry_quality_mask_can_require_accumulating_obv(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=22, freq="B"),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": list(range(100, 122)),
                "volume": 100.0,
            }
        )
        mask = _entry_quality_mask(
            frame,
            cmf_length=None,
            min_cmf=0.0,
            min_rvol20=None,
            obv_accumulation_days=20,
        )

        self.assertFalse(bool(mask[19]))
        self.assertTrue(bool(mask[20]))

    def test_parameter_name_records_every_setting(self) -> None:
        parameters = PairStrategyParameters(100, 14, 20, 100, 14.0)
        self.assertEqual(parameters.name, "K100_R14_M20_E100_P14")

    def test_recommendation_uses_validation_columns(self) -> None:
        stats = pd.DataFrame(
            [
                {
                    "parameter_name": "VALIDATION_WINNER",
                    "selection_eligible": True,
                    "validation_trades": 40,
                    "validation_target_10_hit_rate_pct": 60.0,
                    "test_target_10_hit_rate_pct": 1.0,
                },
                {
                    "parameter_name": "TEST_WINNER",
                    "selection_eligible": True,
                    "validation_trades": 40,
                    "validation_target_10_hit_rate_pct": 50.0,
                    "test_target_10_hit_rate_pct": 99.0,
                },
            ]
        )
        self.assertEqual(_select_recommended_parameter(stats), "VALIDATION_WINNER")


if __name__ == "__main__":
    unittest.main()
