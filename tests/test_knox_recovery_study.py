from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from stock_screener.knox_recovery_study import (
    KNOX_RECOVERY_LOGIC_VERSION,
    KnoxRecoveryStudyResult,
    backtest_knox_recovery_frame,
    load_knox_recovery_outputs,
    save_knox_recovery_outputs,
)
from stock_screener.web.main import app


class KnoxRecoveryStudyTests(unittest.TestCase):
    def test_line_length_drop_and_distinct_recovery_targets(self) -> None:
        events = backtest_knox_recovery_frame(
            self._calculated_frame(),
            symbol="TEST",
            start_date="2026-01-01",
            end_date="2026-02-01",
            envelope_proximity_pct=3.0,
        )

        event = events.iloc[0]
        self.assertEqual(event["line_trading_bars"], 3)
        self.assertEqual(event["line_calendar_days"], 5)
        self.assertAlmostEqual(event["drop_pct"], 20.0)
        self.assertAlmostEqual(event["equal_bounce_target"], 96.0)
        self.assertAlmostEqual(event["full_recovery_target"], 100.0)
        self.assertAlmostEqual(event["full_recovery_gain_required_pct"], 25.0)
        self.assertTrue(event["proximity_pass"])

    def test_recovery_starts_after_endpoint_and_tracks_high_and_close_separately(self) -> None:
        frame = self._calculated_frame()
        frame.loc[3, ["high", "close"]] = [120.0, 110.0]

        event = backtest_knox_recovery_frame(
            frame,
            symbol="TEST",
            start_date="2026-01-01",
            end_date="2026-02-01",
        ).iloc[0]

        self.assertEqual(event["equal_high_trading_days"], 2)
        self.assertEqual(event["equal_close_trading_days"], 2)
        self.assertEqual(event["full_high_trading_days"], 3)
        self.assertEqual(event["full_close_trading_days"], 3)

    def test_unrecovered_event_remains_visible(self) -> None:
        frame = self._calculated_frame()
        frame.loc[4:, ["high", "close"]] = [85.0, 84.0]

        event = backtest_knox_recovery_frame(
            frame,
            symbol="TEST",
            start_date="2026-01-01",
            end_date="2026-02-01",
        ).iloc[0]

        self.assertFalse(event["equal_high_recovered"])
        self.assertFalse(event["full_close_recovered"])
        self.assertTrue(pd.isna(event["full_close_trading_days"]))
        self.assertEqual(event["observation_sessions"], 4)

    def test_overnight_corporate_action_discontinuity_is_flagged(self) -> None:
        frame = self._calculated_frame()
        frame["open"] = frame["close"].shift(1).fillna(frame["close"])
        frame.loc[5, "open"] = 40.0

        event = backtest_knox_recovery_frame(
            frame,
            symbol="TEST",
            start_date="2026-01-01",
            end_date="2026-02-01",
        ).iloc[0]

        self.assertFalse(event["data_quality_pass"])
        self.assertEqual(event["data_quality_reason"], "overnight_price_discontinuity")

    def test_actual_trade_enters_next_day_high_and_scores_fixed_horizon_close(self) -> None:
        frame = self._calculated_frame()
        frame.loc[7, "high"] = 110.0
        frame = pd.concat(
            [
                frame,
                pd.DataFrame(
                    [
                        {
                            "date": pd.Timestamp("2026-01-13"),
                            "low": 98.0,
                            "high": 102.0,
                            "close": 100.0,
                            "envelope_lower": 90.0,
                            "knox_bullish": False,
                            "knox_reference_bars": 0,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        event = backtest_knox_recovery_frame(
            frame,
            symbol="TEST",
            start_date="2026-01-01",
            end_date="2026-02-01",
            round_trip_cost_pct=0.35,
        ).iloc[0]

        self.assertEqual(event["entry_date"], "2026-01-07")
        self.assertEqual(event["entry_price_next_day_high"], 90.0)
        self.assertTrue(event["entry_5d_eligible"])
        self.assertEqual(event["entry_5d_exit_date"], "2026-01-13")
        self.assertAlmostEqual(event["entry_5d_net_return_pct"], 10.7611111111)
        self.assertTrue(event["entry_5d_win"])
        self.assertEqual(event["drop_target_price_from_entry"], 108.0)
        self.assertTrue(event["drop_target_hit"])
        self.assertEqual(event["drop_target_exit_date"], "2026-01-12")
        self.assertEqual(event["drop_target_exit_trading_days"], 3)
        self.assertTrue(event["entry_5d_drop_target_hit"])
        self.assertAlmostEqual(event["drop_target_exit_net_return_pct"], 19.65)

    def test_persistence_round_trip(self) -> None:
        events = backtest_knox_recovery_frame(
            self._calculated_frame(),
            symbol="TEST",
            start_date="2026-01-01",
            end_date="2026-02-01",
        )
        result = KnoxRecoveryStudyResult(
            summary={"logic_version": KNOX_RECOVERY_LOGIC_VERSION, "qualifying_events": 1},
            current_candidates=events.copy(),
            stock_stats=pd.DataFrame([{"symbol": "TEST", "events": 1}]),
            events=events,
        )
        with TemporaryDirectory() as temp_dir:
            save_knox_recovery_outputs(result, Path(temp_dir))
            loaded = load_knox_recovery_outputs(Path(temp_dir))

        self.assertEqual(loaded.summary["logic_version"], KNOX_RECOVERY_LOGIC_VERSION)
        self.assertEqual(len(loaded.events), 1)
        self.assertEqual(loaded.current_candidates.iloc[0]["symbol"], "TEST")

    def test_web_page_displays_saved_study(self) -> None:
        with TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            result = KnoxRecoveryStudyResult(
                summary={
                    "logic_version": KNOX_RECOVERY_LOGIC_VERSION,
                    "qualifying_events": 12,
                    "stocks_with_qualifying_events": 8,
                    "full_close_recovery_rate_pct": 62.5,
                    "entry_20d_win_rate_pct": 62.5,
                    "entry_20d_drop_target_hit_rate_pct": 62.5,
                    "entry_20d_avg_net_return_pct": 3.2,
                    "entry_20d_trades": 10,
                    "median_full_close_trading_days": 18,
                    "latest_history_date": "2026-08-24",
                    "start_date": "2021-08-24",
                    "end_date": "2026-08-24",
                },
                current_candidates=pd.DataFrame(
                    [{"symbol": "TEST", "name": "Test Ltd", "second_endpoint_date": "2026-08-24"}]
                ),
                stock_stats=pd.DataFrame(),
                events=pd.DataFrame(),
            )
            save_knox_recovery_outputs(result, data_root / "knox_recovery")
            with (
                patch.dict(os.environ, {"DATA_ROOT": temp_dir}),
                patch("stock_screener.web.main.get_data_root", return_value=data_root),
            ):
                response = TestClient(app).get("/knox-recovery")

        self.assertEqual(response.status_code, 200)
        self.assertIn("KNOXVILLE RECOVERY STUDY", response.text)
        self.assertIn("62.5%", response.text)
        self.assertIn("TEST", response.text)

    @staticmethod
    def _calculated_frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": pd.bdate_range("2026-01-01", periods=8),
                "low": [100.0, 98.0, 90.0, 80.0, 82.0, 88.0, 95.0, 99.0],
                "high": [101.0, 99.0, 91.0, 82.0, 90.0, 97.0, 101.0, 103.0],
                "close": [100.0, 98.0, 90.0, 81.0, 89.0, 96.0, 100.0, 102.0],
                "envelope_lower": [90.0, 89.0, 85.0, 79.0, 80.0, 82.0, 85.0, 88.0],
                "knox_bullish": [False, False, False, True, False, False, False, False],
                "knox_reference_bars": [0, 0, 0, 3, 0, 0, 0, 0],
            }
        )


if __name__ == "__main__":
    unittest.main()
