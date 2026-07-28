from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd

from stock_screener.data.storage import Storage
from stock_screener.google_sheets import load_google_oauth_client, save_google_oauth_session, save_google_sheet_target
from stock_screener.minervini_sheet_sync import (
    load_minervini_sheet_sync_outputs,
    run_minervini_sheet_sync,
    save_minervini_sheet_sync_outputs,
)
from stock_screener.web.main import app


class MinerviniSheetSyncTests(unittest.TestCase):
    def test_google_oauth_client_loads_from_env_when_file_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            os.environ,
            {"GOOGLE_CLIENT_ID": "env-client", "GOOGLE_CLIENT_SECRET": "env-secret"},
            clear=False,
        ):
            settings = load_google_oauth_client(Path(temp_dir))

        self.assertTrue(settings.configured)
        self.assertEqual(settings.client_id, "env-client")
        self.assertEqual(settings.client_secret, "env-secret")

    def test_run_minervini_sheet_sync_updates_only_minervini_column(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            storage = Storage(data_root)
            storage.save_instruments(
                pd.DataFrame(
                    [
                        {"exchange": "NSE", "tradingsymbol": "PASS", "name": "Pass Ltd"},
                        {"exchange": "NSE", "tradingsymbol": "FAIL", "name": "Fail Ltd"},
                        {"exchange": "NSE", "tradingsymbol": "SHORT", "name": "Short Ltd"},
                    ]
                )
            )
            storage.save_candles("NSE", "PASS", self._strong_uptrend_frame())
            storage.save_candles("NSE", "FAIL", self._downtrend_frame())
            storage.save_candles("NSE", "SHORT", self._short_frame())

            captured_updates: list[dict[str, object]] = []

            with (
                patch(
                    "stock_screener.minervini_sheet_sync.read_google_sheet_values",
                    return_value=[
                        ["Stock_Symbol", "Signal_Date", "Minervini Filter"],
                        ["PASS", "2026-07-10", ""],
                        ["FAIL", "2026-07-10", ""],
                        ["SHORT", "2026-07-10", ""],
                        ["", "2026-07-10", ""],
                    ],
                ),
                patch(
                    "stock_screener.minervini_sheet_sync.batch_update_google_sheet_values",
                    side_effect=lambda _data_root, _spreadsheet_id, updates: captured_updates.extend(updates) or {"totalUpdatedRanges": len(updates)},
                ),
                patch(
                    "stock_screener.minervini_sheet_sync.google_sheet_worksheet_id",
                    return_value=0,
                ),
            ):
                result = run_minervini_sheet_sync(
                    storage,
                    data_root,
                    spreadsheet_id="sheet123",
                    worksheet_title="Sheet1",
                )

            self.assertEqual(result.summary["sheet_row_count"], 4)
            self.assertEqual(result.summary["rows_updated"], 3)
            self.assertEqual(result.summary["minervini_pass_count"], 1)
            self.assertEqual(result.summary["short_history_count"], 1)
            self.assertEqual(result.summary["blank_symbol_count"], 1)
            self.assertEqual(len(captured_updates), 3)
            self.assertEqual(captured_updates[0]["range"], "Sheet1!C2")
            self.assertEqual(captured_updates[0]["values"], [[1]])
            self.assertEqual(captured_updates[1]["values"], [[0]])
            self.assertEqual(captured_updates[2]["values"], [[0]])
            row_updates = result.row_updates.set_index("input_symbol")
            self.assertEqual(int(row_updates.loc["PASS", "minervini_filter"]), 1)
            self.assertEqual(str(row_updates.loc["SHORT", "status"]), "short_history")

    def test_minervini_sheet_page_renders_saved_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            os.environ,
            {"GOOGLE_CLIENT_ID": "env-client", "GOOGLE_CLIENT_SECRET": "env-secret", "DATA_ROOT": temp_dir},
            clear=False,
        ):
            data_root = Path(temp_dir)
            save_google_sheet_target(data_root, "sheet123", "Sheet1")
            save_google_oauth_session(data_root, "refresh-token", "user@example.com")
            save_minervini_sheet_sync_outputs(
                result=self._sample_result(),
                output_dir=data_root / "minervini_sheet_sync",
            )

            with patch("stock_screener.web.main.get_data_root", return_value=data_root):
                client = TestClient(app)
                response = client.get("/minervini-sheet")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Update Sheet1 with Minervini pass or fail", response.text)
        self.assertIn("Logged in as user@example.com", response.text)
        self.assertIn("PASS", response.text)
        self.assertIn("Run Minervini Sheet Update", response.text)

    @staticmethod
    def _strong_uptrend_frame() -> pd.DataFrame:
        dates = pd.date_range("2025-06-01", periods=260, freq="B")
        close = pd.Series(np.linspace(100.0, 200.0, len(dates)))
        return pd.DataFrame(
            {
                "date": dates,
                "open": close,
                "high": close * 1.02,
                "low": close * 0.98,
                "close": close,
                "volume": 100000,
            }
        )

    @staticmethod
    def _downtrend_frame() -> pd.DataFrame:
        dates = pd.date_range("2025-06-01", periods=260, freq="B")
        close = pd.Series(np.linspace(200.0, 100.0, len(dates)))
        return pd.DataFrame(
            {
                "date": dates,
                "open": close,
                "high": close * 1.02,
                "low": close * 0.98,
                "close": close,
                "volume": 100000,
            }
        )

    @staticmethod
    def _short_frame() -> pd.DataFrame:
        dates = pd.date_range("2026-01-01", periods=120, freq="B")
        close = pd.Series(np.linspace(100.0, 140.0, len(dates)))
        return pd.DataFrame(
            {
                "date": dates,
                "open": close,
                "high": close * 1.02,
                "low": close * 0.98,
                "close": close,
                "volume": 50000,
            }
        )

    @staticmethod
    def _sample_result():
        from stock_screener.minervini_sheet_sync import MinerviniSheetSyncResult

        return MinerviniSheetSyncResult(
            summary={
                "spreadsheet_id": "sheet123",
                "worksheet_title": "Sheet1",
                "sheet_row_count": 1,
                "rows_updated": 1,
                "minervini_pass_count": 1,
                "missing_history_count": 0,
                "short_history_count": 0,
                "blank_symbol_count": 0,
                "updated_ranges": 1,
                "spreadsheet_url": "https://docs.google.com/spreadsheets/d/sheet123/edit#gid=0",
            },
            row_updates=pd.DataFrame(
                [
                    {
                        "sheet_row": 2,
                        "input_symbol": "PASS",
                        "resolved_symbol": "PASS",
                        "exchange": "NSE",
                        "history_rows": 260,
                        "minervini_rule_count": 8,
                        "minervini_filter": 1,
                        "status": "ok",
                    }
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
