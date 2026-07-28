from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import os
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from stock_screener.data.storage import Storage
from stock_screener.weekday_pressure_study import run_weekday_pressure_study, save_weekday_pressure_outputs
from stock_screener.web.main import _latest_weekly_buy_sell_symbols, app


class WeekdayPressureStudyTests(unittest.TestCase):
    def test_study_picks_strongest_buy_and_sell_weekdays_from_daily_pressure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            storage = Storage(data_root)
            storage.save_instruments(
                pd.DataFrame(
                    [
                        {
                            "exchange": "NSE",
                            "tradingsymbol": "TEST",
                            "instrument_type": "EQ",
                            "segment": "NSE",
                            "instrument_token": 123,
                            "name": "TEST LIMITED",
                        }
                    ]
                )
            )
            storage.save_candles(
                "NSE",
                "TEST",
                pd.DataFrame(
                    [
                        {"date": "2026-01-05", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000},   # Mon
                        {"date": "2026-01-06", "open": 100, "high": 102, "low": 99, "close": 101, "volume": 800},    # Tue
                        {"date": "2026-01-07", "open": 101, "high": 101, "low": 99, "close": 100, "volume": 700},    # Wed
                        {"date": "2026-01-08", "open": 100, "high": 100, "low": 98, "close": 99, "volume": 1200},    # Thu
                        {"date": "2026-01-09", "open": 99, "high": 101, "low": 98, "close": 100, "volume": 900},     # Fri
                        {"date": "2026-01-12", "open": 100, "high": 111, "low": 100, "close": 110, "volume": 9000},  # Mon
                        {"date": "2026-01-13", "open": 110, "high": 111, "low": 108, "close": 109, "volume": 850},   # Tue
                        {"date": "2026-01-14", "open": 109, "high": 109, "low": 107, "close": 108, "volume": 750},   # Wed
                        {"date": "2026-01-15", "open": 108, "high": 108, "low": 89, "close": 90, "volume": 12000},   # Thu
                        {"date": "2026-01-16", "open": 90, "high": 92, "low": 89, "close": 91, "volume": 1000},      # Fri
                        {"date": "2026-01-19", "open": 91, "high": 101, "low": 91, "close": 100, "volume": 7000},    # Mon
                        {"date": "2026-01-20", "open": 100, "high": 100, "low": 98, "close": 99, "volume": 900},     # Tue
                        {"date": "2026-01-21", "open": 99, "high": 99, "low": 97, "close": 98, "volume": 850},       # Wed
                        {"date": "2026-01-22", "open": 98, "high": 98, "low": 79, "close": 80, "volume": 13000},     # Thu
                        {"date": "2026-01-23", "open": 80, "high": 82, "low": 79, "close": 81, "volume": 950},       # Fri
                    ]
                ),
            )

            config = {
                "universe": {
                    "mode": "nse_all",
                    "instrument_types": ["EQ"],
                    "restrict_to_metadata_symbols": False,
                }
            }

            result = run_weekday_pressure_study(config, storage, exchange="NSE")

            self.assertEqual(result.summary["symbols_processed"], 1)
            self.assertEqual(result.summary["stocks_with_weekday_profile"], 1)
            self.assertEqual(result.summary["top_buy_weekday"], "Monday")
            self.assertEqual(result.summary["top_sell_weekday"], "Thursday")
            self.assertEqual(len(result.stock_stats), 1)

            row = result.stock_stats.iloc[0]
            self.assertEqual(row["symbol"], "TEST")
            self.assertEqual(row["best_buy_weekday"], "Monday")
            self.assertEqual(row["best_sell_weekday"], "Thursday")
            self.assertGreater(row["best_buy_pressure_score"], 0.0)
            self.assertGreater(row["best_sell_pressure_score"], 0.0)

            monday = result.weekday_details[result.weekday_details["weekday"] == "Monday"].iloc[0]
            thursday = result.weekday_details[result.weekday_details["weekday"] == "Thursday"].iloc[0]
            self.assertEqual(int(monday["up_days"]), 2)
            self.assertEqual(int(thursday["down_days"]), 3)
            self.assertGreater(float(monday["avg_buy_pressure"]), float(thursday["avg_buy_pressure"]))
            self.assertGreater(float(thursday["avg_sell_pressure"]), float(monday["avg_sell_pressure"]))

    def test_latest_weekly_buy_sell_symbols_and_page_render(self) -> None:
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
                        },
                        {
                            "exchange": "NSE",
                            "tradingsymbol": "BBB",
                            "instrument_type": "EQ",
                            "segment": "NSE",
                            "instrument_token": 2,
                            "name": "BBB LTD",
                        },
                    ]
                )
            )
            candles = pd.DataFrame(
                [
                    {"date": "2026-01-05", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000},
                    {"date": "2026-01-06", "open": 100, "high": 102, "low": 99, "close": 101, "volume": 1100},
                    {"date": "2026-01-07", "open": 101, "high": 103, "low": 100, "close": 102, "volume": 1200},
                    {"date": "2026-01-08", "open": 102, "high": 103, "low": 101, "close": 101, "volume": 1300},
                    {"date": "2026-01-09", "open": 101, "high": 105, "low": 100, "close": 104, "volume": 1400},
                    {"date": "2026-01-12", "open": 104, "high": 106, "low": 103, "close": 105, "volume": 1500},
                    {"date": "2026-01-13", "open": 105, "high": 106, "low": 102, "close": 103, "volume": 1600},
                    {"date": "2026-01-14", "open": 103, "high": 104, "low": 100, "close": 101, "volume": 1700},
                    {"date": "2026-01-15", "open": 101, "high": 102, "low": 97, "close": 98, "volume": 1800},
                    {"date": "2026-01-16", "open": 98, "high": 99, "low": 95, "close": 96, "volume": 1900},
                ]
            )
            storage.save_candles("NSE", "AAA", candles)
            storage.save_candles("NSE", "BBB", candles.assign(close=candles["close"] * 1.5))
            storage.save_signals(
                "latest_raw_signals.csv",
                pd.DataFrame(
                    [
                        {"exchange": "NSE", "symbol": "AAA", "name": "AAA LTD", "date": "2026-05-11", "signal": "BUY", "close": 105},
                        {"exchange": "NSE", "symbol": "BBB", "name": "BBB LTD", "date": "2026-05-11", "signal": "SELL", "close": 155},
                        {"exchange": "NSE", "symbol": "OLD", "name": "OLD LTD", "date": "2026-05-04", "signal": "BUY", "close": 80},
                    ]
                ),
            )

            symbols = _latest_weekly_buy_sell_symbols(data_root)
            self.assertEqual(symbols, {"AAA", "BBB"})

            config = {
                "data": {"data_root_env": "DATA_ROOT"},
                "universe": {
                    "mode": "nse_all",
                    "instrument_types": ["EQ"],
                    "restrict_to_metadata_symbols": False,
                    "approximate_nse_traded_universe": {"enabled": False},
                },
            }
            result = run_weekday_pressure_study(config, storage, exchange="NSE", symbols=symbols)
            save_weekday_pressure_outputs(result, data_root / "weekday_pressure_study")

            with (
                patch.dict(os.environ, {"DATA_ROOT": temp_dir}),
                patch("stock_screener.web.main.get_data_root", return_value=data_root),
            ):
                client = TestClient(app)
                response = client.get("/weekday-study")

            self.assertEqual(response.status_code, 200)
            self.assertIn("Weekday Study", response.text)
            self.assertIn("AAA", response.text)
            self.assertIn("BBB", response.text)
            self.assertIn("Latest weekly BUY signals", response.text)


if __name__ == "__main__":
    unittest.main()
