from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from stock_screener.google_sheets import (
    build_weekly_buy_tracker_sheet_values,
    google_oauth_status,
    load_google_sheets_settings,
    load_google_oauth_client,
    save_google_oauth_client,
    save_google_sheet_target,
    save_google_sheets_credentials,
)
from stock_screener.weekly_buy_tracker_study import (
    WeeklyBuyTrackerResult,
    _evaluate_latest_volume_burst,
    _evaluate_minervini_template,
    _evaluate_obv_macd,
    load_weekly_buy_tracker_outputs,
    save_weekly_buy_tracker_outputs,
)
from stock_screener.web.main import app


class WeeklyBuyTrackerStudyTests(unittest.TestCase):
    def test_sheet_values_include_googlefinance_and_gain_formulas(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "exchange": "NSE",
                    "symbol": "AAA",
                    "name": "A Ltd",
                    "latest_close": 120.0,
                    "latest_close_date": "2026-06-20",
                    "s2_buy_count": 2,
                    "s3_buy_count": 1,
                    "total_buy_count": 3,
                    "first_buy_date": "2026-04-04",
                    "first_buy_price": 100.0,
                    "latest_buy_date": "2026-06-13",
                    "latest_buy_price": 110.0,
                    "latest_s2_buy_date": "2026-06-13",
                    "latest_s2_buy_price": 110.0,
                    "latest_s3_buy_date": "2026-05-30",
                    "latest_s3_buy_price": 108.0,
                }
            ]
        )

        values = build_weekly_buy_tracker_sheet_values(frame)

        self.assertEqual(values[0][0], "Exchange")
        self.assertEqual(values[1][0], "NSE")
        self.assertEqual(values[1][1], "AAA")
        self.assertEqual(values[1][16], '=IFERROR(GOOGLEFINANCE(A2&":"&B2,"price"),"")')
        self.assertEqual(values[1][17], '=IF(Q2="",D2,Q2)')
        self.assertEqual(values[1][18], '=IFERROR((R2-J2)/J2,"")')

    def test_google_sheet_settings_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = save_google_sheets_credentials(
                Path(temp_dir),
                json.dumps(
                    {
                        "client_email": "svc@example.iam.gserviceaccount.com",
                        "private_key": "-----BEGIN PRIVATE KEY-----\\nabc\\n-----END PRIVATE KEY-----\\n",
                        "token_uri": "https://oauth2.googleapis.com/token",
                    }
                ),
                "sheet123",
                "Tracker",
            )
            loaded = load_google_sheets_settings(Path(temp_dir))

        self.assertTrue(settings.configured)
        self.assertEqual(loaded.spreadsheet_id, "sheet123")
        self.assertEqual(loaded.worksheet_title, "Tracker")
        self.assertEqual(loaded.client_email, "svc@example.iam.gserviceaccount.com")

    def test_google_oauth_client_and_sheet_target_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = save_google_sheet_target(Path(temp_dir), "sheet123", "Tracker")
            client = save_google_oauth_client(Path(temp_dir), "client-id", "client-secret")
            loaded_target = load_google_sheets_settings(Path(temp_dir))
            loaded_client = load_google_oauth_client(Path(temp_dir))
            status = google_oauth_status(Path(temp_dir))

        self.assertTrue(target.configured)
        self.assertEqual(loaded_target.spreadsheet_id, "sheet123")
        self.assertEqual(loaded_target.worksheet_title, "Tracker")
        self.assertTrue(client.configured)
        self.assertEqual(loaded_client.client_id, "client-id")
        self.assertEqual(loaded_client.client_secret, "client-secret")
        self.assertTrue(status["client_configured"])
        self.assertFalse(status["logged_in"])

    def test_weekly_buy_gains_page_renders_saved_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            result = WeeklyBuyTrackerResult(
                summary={
                    "exchange": "NSE",
                    "start_date": "2026-04-01",
                    "stocks_with_buy_history": 2,
                    "s2_buy_events": 2,
                    "s3_buy_events": 1,
                    "latest_close_date": "2026-06-20",
                },
                stock_stats=pd.DataFrame(
                    [
                        {
                            "exchange": "NSE",
                            "symbol": "AAA",
                            "name": "A Ltd",
                            "latest_close": 120.0,
                            "latest_close_date": "2026-06-20",
                            "s2_buy_count": 2,
                            "s3_buy_count": 1,
                            "total_buy_count": 3,
                            "first_buy_date": "2026-04-04",
                            "first_buy_price": 100.0,
                            "latest_buy_date": "2026-06-13",
                            "latest_buy_price": 110.0,
                            "gain_vs_first_buy_pct": 20.0,
                            "gain_vs_latest_buy_pct": 9.09,
                            "minervini_pass": True,
                            "obv_macd_cross_up": True,
                            "latest_volume_3x_prev_9d": True,
                        },
                        {
                            "exchange": "NSE",
                            "symbol": "BBB",
                            "name": "B Ltd",
                            "latest_close": 80.0,
                            "latest_close_date": "2026-06-20",
                            "s2_buy_count": 1,
                            "s3_buy_count": 0,
                            "total_buy_count": 1,
                            "first_buy_date": "2026-04-04",
                            "first_buy_price": 90.0,
                            "latest_buy_date": "2026-06-13",
                            "latest_buy_price": 95.0,
                            "gain_vs_first_buy_pct": -11.11,
                            "gain_vs_latest_buy_pct": -15.79,
                            "minervini_pass": False,
                            "obv_macd_cross_up": False,
                            "latest_volume_3x_prev_9d": False,
                        }
                    ]
                ),
                s2_events=pd.DataFrame(),
                s3_events=pd.DataFrame(),
            )
            save_weekly_buy_tracker_outputs(result, data_root / "weekly_buy_tracker")
            loaded = load_weekly_buy_tracker_outputs(data_root / "weekly_buy_tracker")
            self.assertEqual(int(loaded.summary["stocks_with_buy_history"]), 2)

            with (
                patch.dict(os.environ, {"DATA_ROOT": temp_dir}),
                patch("stock_screener.web.main.get_data_root", return_value=data_root),
            ):
                client = TestClient(app)
                response = client.get("/weekly-buy-gains?minervini_only=1&obv_macd_only=1")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Weekly BUY names and current gain", response.text)
        self.assertIn("AAA", response.text)
        self.assertNotIn("BBB", response.text)
        self.assertIn("Stocks Currently in Gain", response.text)
        self.assertIn("All Weekly BUY Stocks", response.text)
        self.assertIn("OBV MACD cross-up only", response.text)
        self.assertIn("Latest volume >= 3x prev 9D avg", response.text)
        self.assertIn("AAA", response.text)
        self.assertIn("weekly-buy-symbols-csv", response.text)

    def test_weekly_buy_gains_page_handles_missing_gain_column(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            result = WeeklyBuyTrackerResult(
                summary={
                    "exchange": "NSE",
                    "start_date": "2026-04-01",
                    "stocks_with_buy_history": 1,
                    "s2_buy_events": 1,
                    "s3_buy_events": 0,
                    "latest_close_date": "2026-06-20",
                },
                stock_stats=pd.DataFrame(
                    [
                        {
                            "exchange": "NSE",
                            "symbol": "AAA",
                            "name": "A Ltd",
                            "latest_close": 120.0,
                            "latest_close_date": "2026-06-20",
                            "s2_buy_count": 1,
                            "s3_buy_count": 0,
                            "total_buy_count": 1,
                            "first_buy_date": "2026-04-04",
                            "first_buy_price": 100.0,
                            "latest_buy_date": "2026-06-13",
                            "latest_buy_price": 110.0,
                        }
                    ]
                ),
                s2_events=pd.DataFrame(),
                s3_events=pd.DataFrame(),
            )
            save_weekly_buy_tracker_outputs(result, data_root / "weekly_buy_tracker")

            with (
                patch.dict(os.environ, {"DATA_ROOT": temp_dir}),
                patch("stock_screener.web.main.get_data_root", return_value=data_root),
            ):
                client = TestClient(app)
                response = client.get("/weekly-buy-gains")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Weekly BUY names and current gain", response.text)
        self.assertIn("AAA", response.text)

    def test_minervini_template_passes_for_strong_uptrend_frame(self) -> None:
        dates = pd.date_range("2025-06-01", periods=260, freq="B")
        close = pd.Series(np.linspace(100.0, 200.0, len(dates)))
        frame = pd.DataFrame(
            {
                "date": dates,
                "open": close,
                "high": close * 1.02,
                "low": close * 0.98,
                "close": close,
                "volume": 100000,
            }
        )

        result = _evaluate_minervini_template(frame)

        self.assertTrue(result["minervini_pass"])
        self.assertEqual(result["minervini_rule_count"], 8)

    def test_minervini_template_fails_for_downtrend_frame(self) -> None:
        dates = pd.date_range("2025-06-01", periods=260, freq="B")
        close = pd.Series(np.linspace(200.0, 100.0, len(dates)))
        frame = pd.DataFrame(
            {
                "date": dates,
                "open": close,
                "high": close * 1.02,
                "low": close * 0.98,
                "close": close,
                "volume": 100000,
            }
        )

        result = _evaluate_minervini_template(frame)

        self.assertFalse(result["minervini_pass"])
        self.assertLess(result["minervini_rule_count"], 8)

    def test_obv_macd_cross_up_detects_cross_above_zero_line(self) -> None:
        rng = np.random.default_rng(1)
        frame = pd.DataFrame()
        result = {}
        for _ in range(200):
            dates = pd.date_range("2025-06-01", periods=120, freq="B")
            close = np.cumsum(rng.normal(0.2, 2.0, len(dates))) + 100.0
            close = np.maximum(close, 1.0)
            volume = rng.integers(50000, 500000, len(dates))
            frame = pd.DataFrame(
                {
                    "date": dates,
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": volume,
                }
            )
            result = _evaluate_obv_macd(frame)
            if result["obv_macd_cross_up"]:
                break

        self.assertIn("obv_macd_line", result)
        self.assertIn("obv_macd_signal", result)
        self.assertIn("obv_macd_hist", result)
        self.assertTrue(result["obv_macd_pass"])

    def test_latest_volume_burst_detects_3x_threshold(self) -> None:
        dates = pd.date_range("2025-06-01", periods=12, freq="B")
        volume = pd.Series([100.0] * 11 + [400.0])
        frame = pd.DataFrame(
            {
                "date": dates,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": volume,
            }
        )

        result = _evaluate_latest_volume_burst(frame)

        self.assertTrue(result["latest_volume_3x_prev_9d"])
        self.assertEqual(float(result["latest_volume_ratio_prev_9d"]), 4.0)


if __name__ == "__main__":
    unittest.main()
