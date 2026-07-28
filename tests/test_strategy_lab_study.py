from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from stock_screener.data.storage import Storage
from stock_screener.strategy_lab_study import run_strategy_lab_study, save_strategy_lab_outputs
from stock_screener.web.main import app


class StrategyLabStudyTests(unittest.TestCase):
    def test_strategy_lab_runs_and_produces_strategy_stats(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            storage = Storage(data_root)
            storage.save_instruments(
                pd.DataFrame(
                    [
                        {
                            "exchange": "NSE",
                            "tradingsymbol": "AAA",
                            "instrument_type": "EQ",
                            "segment": "NSE",
                            "instrument_token": 1,
                            "name": "AAA LTD",
                        }
                    ]
                )
            )
            dates = pd.bdate_range("2026-04-20", periods=40)
            closes = [100, 101, 102, 103, 104, 106, 108, 110, 109, 111, 113, 115, 117, 116, 118, 120, 121, 122, 121, 123, 124, 126, 125, 127, 128, 127, 129, 131, 130, 132, 133, 132, 134, 136, 137, 138, 137, 139, 140, 141]
            candles = pd.DataFrame(
                {
                    "date": dates,
                    "open": closes,
                    "high": [c + 1.5 for c in closes],
                    "low": [c - 1.5 for c in closes],
                    "close": closes,
                    "volume": [1000 + i * 20 for i in range(len(closes))],
                }
            )
            storage.save_candles("NSE", "AAA", candles)
            storage.save_signals(
                "latest_raw_signals.csv",
                pd.DataFrame(
                    [
                        {"exchange": "NSE", "symbol": "AAA", "name": "AAA LTD", "date": "2026-04-27", "signal": "BUY"},
                        {"exchange": "NSE", "symbol": "AAA", "name": "AAA LTD", "date": "2026-05-11", "signal": "SELL"},
                        {"exchange": "NSE", "symbol": "AAA", "name": "AAA LTD", "date": "2026-05-25", "signal": "BUY"},
                    ]
                ),
            )

            config = {"data": {"data_root_env": "DATA_ROOT"}}
            result = run_strategy_lab_study(config, storage, exchange="NSE", start_date="2026-04-25")

            self.assertEqual(result.summary["signal_events"], 3)
            self.assertFalse(result.strategy_stats.empty)
            self.assertFalse(result.trade_details.empty)
            self.assertIn("Momentum Capture 3D", set(result.strategy_stats["strategy_name"]))

    def test_strategy_lab_page_renders_saved_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            storage = Storage(data_root)
            storage.save_instruments(
                pd.DataFrame(
                    [
                        {
                            "exchange": "NSE",
                            "tradingsymbol": "AAA",
                            "instrument_type": "EQ",
                            "segment": "NSE",
                            "instrument_token": 1,
                            "name": "AAA LTD",
                        }
                    ]
                )
            )
            dates = pd.bdate_range("2026-04-20", periods=40)
            closes = [100 + i for i in range(40)]
            candles = pd.DataFrame(
                {
                    "date": dates,
                    "open": closes,
                    "high": [c + 1 for c in closes],
                    "low": [c - 1 for c in closes],
                    "close": closes,
                    "volume": [1000 + i * 10 for i in range(len(closes))],
                }
            )
            storage.save_candles("NSE", "AAA", candles)
            storage.save_signals(
                "latest_raw_signals.csv",
                pd.DataFrame(
                    [
                        {"exchange": "NSE", "symbol": "AAA", "name": "AAA LTD", "date": "2026-04-27", "signal": "BUY"},
                        {"exchange": "NSE", "symbol": "AAA", "name": "AAA LTD", "date": "2026-05-11", "signal": "SELL"},
                    ]
                ),
            )
            config = {"data": {"data_root_env": "DATA_ROOT"}}
            result = run_strategy_lab_study(config, storage, exchange="NSE", start_date="2026-04-25")
            save_strategy_lab_outputs(result, data_root / "strategy_lab")

            with (
                patch.dict(os.environ, {"DATA_ROOT": temp_dir}),
                patch("stock_screener.web.main.get_data_root", return_value=data_root),
            ):
                client = TestClient(app)
                response = client.get("/strategy-lab")

            self.assertEqual(response.status_code, 200)
            self.assertIn("Strategy Lab", response.text)
            self.assertIn("Momentum Capture 3D", response.text)
            self.assertIn("AAA", response.text)


if __name__ == "__main__":
    unittest.main()
