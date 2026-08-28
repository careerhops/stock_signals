from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from stock_screener.data.storage import Storage
from stock_screener.resistance_breaks_study import (
    ResistanceBreaksStudyResult,
    load_resistance_breaks_outputs,
    run_resistance_breaks_study,
    save_resistance_breaks_outputs,
)
from stock_screener.web.main import app


class ResistanceBreaksStudyTests(unittest.TestCase):
    def test_resistance_breaks_study_flags_two_volume_confirmed_breakouts(self) -> None:
        with TemporaryDirectory() as temp_dir:
            storage = Storage(Path(temp_dir))
            storage.save_instruments(
                pd.DataFrame(
                    [
                        {"exchange": "NSE", "tradingsymbol": "PASS", "name": "Pass Ltd"},
                        {"exchange": "NSE", "tradingsymbol": "FAIL", "name": "Fail Ltd"},
                    ]
                )
            )
            storage.save_candles("NSE", "PASS", self._pass_frame(), "1D")
            storage.save_candles("NSE", "FAIL", self._fail_frame(), "1D")

            result = run_resistance_breaks_study(
                storage,
                exchange="NSE",
                left_bars=2,
                right_bars=2,
                volume_avg_window=3,
                volume_multiplier=2.0,
                min_break_count=2,
                recent_window_days=20,
                reference_date="2026-01-22",
            )
            save_resistance_breaks_outputs(result, Path(temp_dir) / "resistance_breaks")
            loaded = load_resistance_breaks_outputs(Path(temp_dir) / "resistance_breaks")

        self.assertEqual(int(loaded.summary["symbols_processed"]), 2)
        self.assertEqual(int(loaded.summary["resistance_break_matches"]), 1)
        pass_row = loaded.stock_stats[loaded.stock_stats["symbol"] == "PASS"].iloc[0]
        fail_row = loaded.stock_stats[loaded.stock_stats["symbol"] == "FAIL"].iloc[0]
        self.assertEqual(int(pass_row["volume_confirmed_resistance_break_count"]), 2)
        self.assertTrue(bool(pass_row["passes_volume_confirmed_resistance_breaks"]))
        self.assertTrue(bool(pass_row["close_above_recent_resistance"]))
        self.assertTrue(bool(pass_row["resistance_within_25pct_of_ath"]))
        self.assertEqual(int(fail_row["volume_confirmed_resistance_break_count"]), 1)
        self.assertFalse(bool(fail_row["passes_volume_confirmed_resistance_breaks"]))

    def test_resistance_breaks_page_renders_saved_outputs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            result = ResistanceBreaksStudyResult(
                summary={
                    "exchange": "NSE",
                    "symbols_processed": 2,
                    "stocks_with_history": 2,
                    "resistance_break_matches": 1,
                    "latest_close_date": "2026-07-21",
                    "avg_volume_confirmed_break_count": 1.5,
                    "left_bars": 15,
                    "right_bars": 15,
                    "volume_avg_window": 20,
                    "volume_multiplier": 2.0,
                    "min_break_count": 2,
                    "recent_breakout_window_days": 7,
                },
                stock_stats=pd.DataFrame(
                    [
                        {
                            "exchange": "NSE",
                            "symbol": "PASS",
                            "name": "Pass Ltd",
                            "latest_close": 150.0,
                            "latest_close_date": "2026-07-21",
                            "volume_confirmed_resistance_break_count": 2,
                            "latest_resistance_level": 145.0,
                            "latest_52w_high": 160.0,
                            "close_above_recent_resistance": True,
                            "resistance_within_25pct_of_ath": True,
                            "latest_break_date": "2026-07-20",
                            "latest_break_volume_ratio": 2.5,
                            "recent_break_dates_csv": "2026-07-10,2026-07-20",
                            "recent_breakout_window_days": 7,
                            "passes_volume_confirmed_resistance_breaks": True,
                        }
                    ]
                ),
                breakout_events=pd.DataFrame(
                    [
                        {
                            "date": "2026-07-20",
                            "exchange": "NSE",
                            "symbol": "PASS",
                            "name": "Pass Ltd",
                            "close": 150.0,
                            "resistance_level": 145.0,
                            "volume": 300000.0,
                            "avg_volume": 120000.0,
                            "volume_ratio": 2.5,
                        }
                    ]
                ),
            )
            save_resistance_breaks_outputs(result, data_root / "resistance_breaks")

            with (
                patch.dict(os.environ, {"DATA_ROOT": temp_dir}),
                patch("stock_screener.web.main.get_data_root", return_value=data_root),
            ):
                client = TestClient(app)
                response = client.get("/resistance-breaks")

        self.assertEqual(response.status_code, 404)
        self.assertIn("Resistance Breaks", response.text)
        self.assertIn("temporarily removed from the workspace", response.text)

    def test_highest_resistance_zone_is_selected_and_only_post_zone_breaks_count(self) -> None:
        with TemporaryDirectory() as temp_dir:
            storage = Storage(Path(temp_dir))
            storage.save_instruments(
                pd.DataFrame(
                    [
                        {"exchange": "NSE", "tradingsymbol": "MULTI", "name": "Multi Zone Ltd"},
                    ]
                )
            )
            storage.save_candles("NSE", "MULTI", self._multi_zone_frame(), "1D")

            result = run_resistance_breaks_study(
                storage,
                exchange="NSE",
                left_bars=2,
                right_bars=2,
                volume_avg_window=3,
                volume_multiplier=2.0,
                min_break_count=2,
                recent_window_days=7,
                reference_date="2026-01-22",
            )

        row = result.stock_stats[result.stock_stats["symbol"] == "MULTI"].iloc[0]
        self.assertEqual(row["selected_resistance_zone_date"], "2026-01-13")
        self.assertAlmostEqual(float(row["latest_resistance_level"]), 14.2, places=4)
        self.assertEqual(int(row["volume_confirmed_resistance_break_count"]), 2)
        self.assertEqual(row["recent_break_dates_csv"], "2026-01-19,2026-01-21")
        self.assertTrue(bool(row["passes_volume_confirmed_resistance_breaks"]))
        event_dates = [pd.Timestamp(value).strftime("%Y-%m-%d") for value in result.breakout_events["date"].tolist()]
        self.assertEqual(event_dates, ["2026-01-21", "2026-01-19"])

    @staticmethod
    def _pass_frame() -> pd.DataFrame:
        dates = pd.date_range("2026-01-01", periods=16, freq="B")
        close = [10.0, 10.5, 11.2, 11.0, 11.3, 11.7, 12.4, 13.1, 13.9, 13.1, 13.3, 13.0, 14.4, 13.7, 14.6, 14.5]
        high = [10.3, 10.8, 11.5, 11.2, 11.5, 12.0, 12.6, 13.4, 14.2, 13.4, 13.5, 13.2, 14.5, 13.9, 14.8, 15.0]
        volume = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 320.0, 100.0, 360.0, 100.0]
        return pd.DataFrame(
            {
                "date": dates,
                "open": close,
                "high": high,
                "low": [value - 0.8 for value in close],
                "close": close,
                "volume": volume,
            }
        )

    @staticmethod
    def _fail_frame() -> pd.DataFrame:
        dates = pd.date_range("2026-01-01", periods=16, freq="B")
        close = [10.0, 10.5, 11.2, 11.0, 11.3, 11.7, 12.4, 13.1, 13.9, 13.1, 13.3, 13.0, 14.4, 13.7, 14.6, 14.5]
        high = [10.3, 10.8, 11.5, 11.2, 11.5, 12.0, 12.6, 13.4, 14.2, 13.4, 13.5, 13.2, 14.5, 13.9, 14.8, 15.0]
        volume = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 180.0, 100.0, 340.0, 100.0]
        return pd.DataFrame(
            {
                "date": dates,
                "open": close,
                "high": high,
                "low": [value - 0.8 for value in close],
                "close": close,
                "volume": volume,
            }
        )

    @staticmethod
    def _multi_zone_frame() -> pd.DataFrame:
        dates = pd.date_range("2026-01-01", periods=16, freq="B")
        close = [10.0, 10.5, 11.2, 11.0, 11.3, 11.7, 12.4, 13.1, 13.9, 13.1, 13.3, 13.0, 14.4, 13.7, 14.6, 14.5]
        high = [10.3, 10.8, 11.5, 11.2, 11.5, 12.0, 12.6, 13.4, 14.2, 13.4, 13.5, 13.2, 14.5, 13.9, 14.8, 15.0]
        volume = [100.0, 100.0, 260.0, 100.0, 100.0, 100.0, 240.0, 100.0, 100.0, 100.0, 100.0, 100.0, 320.0, 100.0, 360.0, 100.0]
        return pd.DataFrame(
            {
                "date": dates,
                "open": close,
                "high": high,
                "low": [value - 0.8 for value in close],
                "close": close,
                "volume": volume,
            }
        )


if __name__ == "__main__":
    unittest.main()
