from __future__ import annotations

import unittest

import pandas as pd

from pathlib import Path
from tempfile import TemporaryDirectory

from stock_screener.knox_alpha_filter_research import _feature_frame, _load_positions


class KnoxAlphaFilterResearchTests(unittest.TestCase):
    def test_features_only_use_data_available_through_current_row(self) -> None:
        dates = pd.date_range("2024-01-01", periods=300, freq="B")
        original = pd.DataFrame(
            {
                "date": dates,
                "open": range(100, 400),
                "high": range(101, 401),
                "low": range(99, 399),
                "close": range(100, 400),
                "volume": 1000,
            }
        )
        changed_future = original.copy()
        changed_future.loc[changed_future.index[-20:], "close"] = 1_000_000

        before = _feature_frame(original).iloc[270]
        after = _feature_frame(changed_future).iloc[270]

        for column in (
            "momentum_12_1_pct",
            "high_52w_ratio",
            "volume_ratio20",
            "cmf20",
            "obv_accumulating",
            "obv_above_sma13",
            "obv_cross_sma13_recent",
            "realized_vol20_pct",
            "di_plus",
            "di_minus",
            "adx14",
        ):
            self.assertEqual(before[column], after[column], column)

    def test_position_loader_accepts_date_and_timestamp_forms(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pd.DataFrame(
                [
                    {
                        "symbol": "AAA",
                        "parameter_name": "TEST",
                        "entry_signal_date": "2026-01-01 00:00:00",
                        "entry_date": "2026-01-02",
                        "exit_date": "2026-02-01",
                        "net_return_pct": 5.0,
                        "bars_held": 20,
                        "target_10_hit": False,
                        "bars_to_target": None,
                        "data_quality_pass": True,
                    }
                ]
            ).to_csv(root / "baseline_trades.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "symbol": "BBB",
                        "parameter_name": "TEST",
                        "entry_signal_date": "2026-01-03",
                        "entry_date": "2026-01-05",
                        "unrealized_net_return_pct": -2.0,
                        "bars_open": 10,
                        "target_10_hit": False,
                        "bars_to_target": None,
                        "data_quality_pass": True,
                    }
                ]
            ).to_csv(root / "open_positions.csv", index=False)

            loaded = _load_positions(root, "TEST")

        self.assertEqual(len(loaded), 2)
        self.assertTrue(loaded["entry_signal_date"].notna().all())


if __name__ == "__main__":
    unittest.main()
