from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from stock_screener.data.storage import Storage
from stock_screener.rotation_study import (
    _build_groups_from_correlation,
    _build_group_outputs,
    _build_candidates_frame,
    run_rotation_study,
)


class RotationStudyTests(unittest.TestCase):
    def test_build_groups_from_correlation_finds_bounded_clusters(self) -> None:
        correlation = pd.DataFrame(
            [
                [1.0, 0.82, 0.10, 0.05],
                [0.82, 1.0, 0.12, 0.07],
                [0.10, 0.12, 1.0, 0.79],
                [0.05, 0.07, 0.79, 1.0],
            ],
            index=["AAA", "AAB", "BBB", "BBC"],
            columns=["AAA", "AAB", "BBB", "BBC"],
        )

        groups = _build_groups_from_correlation(
            correlation,
            min_correlation=0.7,
            min_group_size=2,
            target_group_size=2,
            max_group_size=3,
        )

        self.assertEqual(groups, [["AAA", "AAB"], ["BBB", "BBC"]])

    def test_build_groups_from_correlation_caps_large_groups(self) -> None:
        symbols = ["AAA", "AAB", "AAC", "AAD", "AAE"]
        correlation = pd.DataFrame(0.88, index=symbols, columns=symbols)
        for symbol in symbols:
            correlation.loc[symbol, symbol] = 1.0

        groups = _build_groups_from_correlation(
            correlation,
            min_correlation=0.7,
            min_group_size=2,
            target_group_size=3,
            max_group_size=3,
        )

        self.assertEqual([len(group) for group in groups], [3, 2])

    def test_group_outputs_marks_catch_up_candidates(self) -> None:
        dates = pd.date_range("2025-01-03", periods=20, freq="W-FRI")
        close_matrix = pd.DataFrame(
            {
                "LEAD": [100, 101, 102, 103, 105, 108, 112, 116, 120, 124, 129, 135, 141, 148, 156, 165, 175, 186, 198, 211],
                "LAG": [100, 101, 101, 102, 103, 104, 105, 106, 108, 110, 113, 117, 121, 126, 132, 139, 147, 156, 166, 170],
                "SYNC": [100, 101, 102, 103, 104, 107, 111, 115, 119, 124, 130, 136, 142, 149, 157, 166, 176, 187, 199, 212],
            },
            index=dates,
        )
        returns = close_matrix.pct_change()
        correlation = returns.corr(min_periods=5)
        groups = [["LEAD", "LAG", "SYNC"]]
        symbol_names = {"LEAD": "Leader Ltd", "LAG": "Lagging Ltd", "SYNC": "Sync Ltd"}
        cfg = {
            "lag_window_weeks": 8,
            "min_lag_correlation": 0.1,
            "catch_up_gap_pct": 6.0,
            "group_strength_min_8w": 5.0,
        }

        groups_frame, members_frame = _build_group_outputs(
            groups=groups,
            close_matrix=close_matrix,
            returns=returns,
            correlation=correlation,
            symbol_names=symbol_names,
            latest_signals={
                "LEAD": {
                    "latest_week_signal": "BUY",
                    "latest_week_signal_date": dates[-1],
                    "latest_week_signal_is_fresh": True,
                },
                "LAG": {
                    "latest_week_signal": "SELL",
                    "latest_week_signal_date": dates[-1],
                    "latest_week_signal_is_fresh": True,
                },
                "SYNC": {
                    "latest_week_signal": "BUY",
                    "latest_week_signal_date": dates[-2],
                    "latest_week_signal_is_fresh": False,
                },
            },
            exchange="NSE",
            study_cfg=cfg,
        )
        candidates = _build_candidates_frame(members_frame)

        self.assertEqual(len(groups_frame), 1)
        self.assertTrue((members_frame["movement_status"] == "Leader").any())
        lag_row = members_frame[members_frame["symbol"] == "LAG"].iloc[0]
        self.assertEqual(lag_row["movement_status"], "Catch-up Candidate")
        self.assertEqual(candidates.iloc[0]["symbol"], "LAG")
        self.assertEqual(int(groups_frame.iloc[0]["latest_weekly_buy_count"]), 1)
        self.assertEqual(int(groups_frame.iloc[0]["latest_weekly_sell_count"]), 1)

    def test_run_rotation_study_saves_grouped_output_for_cached_candles(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            storage = Storage(data_root)
            storage.save_instruments(
                pd.DataFrame(
                    [
                        {"exchange": "NSE", "tradingsymbol": "AAA", "name": "AAA Ltd", "instrument_type": "EQ", "segment": "NSE"},
                        {"exchange": "NSE", "tradingsymbol": "BBB", "name": "BBB Ltd", "instrument_type": "EQ", "segment": "NSE"},
                        {"exchange": "NSE", "tradingsymbol": "CCC", "name": "CCC Ltd", "instrument_type": "EQ", "segment": "NSE"},
                    ]
                )
            )
            dates = pd.date_range("2021-01-01", periods=130, freq="W-FRI")
            base = pd.Series(range(100, 230), index=dates)
            for symbol, series in {
                "AAA": base,
                "BBB": base * 1.01,
                "CCC": pd.Series(range(200, 330), index=dates),
            }.items():
                daily = pd.DataFrame(
                    {
                        "date": dates,
                        "open": series.values,
                        "high": (series * 1.01).values,
                        "low": (series * 0.99).values,
                        "close": series.values,
                        "volume": [1000] * len(series),
                    }
                )
                storage.save_candles("NSE", symbol, daily, "1D")

            config = {
                "universe": {
                    "mode": "nse_all",
                    "instrument_types": ["EQ"],
                    "restrict_to_metadata_symbols": False,
                    "approximate_nse_traded_universe": {"enabled": False},
                },
                "strategy": {"weekly_anchor": "W-FRI", "use_completed_weeks_only": True},
                "rotation_study": {
                    "lookback_weeks": 104,
                    "min_history_weeks": 52,
                    "min_overlap_weeks": 26,
                    "min_group_size": 2,
                    "target_group_size": 2,
                    "max_group_size": 3,
                    "min_correlation": 0.99,
                },
            }

            result = run_rotation_study(config, storage, exchange="NSE")

        self.assertGreaterEqual(result.summary["symbols_processed"], 3)
        self.assertGreaterEqual(result.summary["groups_found"], 1)
        self.assertIn("group_id", result.members.columns)


if __name__ == "__main__":
    unittest.main()
