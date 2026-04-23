from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import pandas as pd

from stock_screener.data.storage import Storage
from stock_screener.jobs.daily_scan import daily_signal_config, run_daily_scan


class DailyScanTests(unittest.TestCase):
    def test_daily_signal_config_uses_daily_timeframe_and_age_window(self) -> None:
        config = {
            "data": {"scan_timeframe": "1W"},
            "daily_signals": {"max_signal_age_bars": 2},
            "filters": {"signal": {"max_signal_age_bars": 1}},
        }

        daily_config = daily_signal_config(config)

        self.assertEqual(daily_config["data"]["scan_timeframe"], "1D")
        self.assertEqual(daily_config["filters"]["signal"]["max_signal_age_bars"], 2)
        self.assertEqual(config["data"]["scan_timeframe"], "1W")

    def test_scan_saves_daily_signal_outputs_separately_from_weekly_outputs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            storage = Storage(Path(temp_dir))
            storage.save_candles(
                "NSE",
                "TCS",
                pd.DataFrame(
                    [
                        {"date": "2026-04-13", "open": 100, "high": 104, "low": 99, "close": 103, "volume": 1000},
                        {"date": "2026-04-14", "open": 103, "high": 106, "low": 101, "close": 105, "volume": 1200},
                        {"date": "2026-04-15", "open": 105, "high": 108, "low": 104, "close": 107, "volume": 1100},
                        {"date": "2026-04-16", "open": 107, "high": 109, "low": 105, "close": 106, "volume": 1150},
                        {"date": "2026-04-17", "open": 106, "high": 112, "low": 105, "close": 111, "volume": 1300},
                    ]
                ),
            )

            instruments = pd.DataFrame(
                [
                    {
                        "exchange": "NSE",
                        "tradingsymbol": "TCS",
                        "instrument_type": "EQ",
                        "segment": "NSE",
                        "instrument_token": 123,
                        "name": "TATA CONSULTANCY SERVICES",
                    }
                ]
            )

            class FakeProvider:
                def __init__(self, access_token: str) -> None:
                    self.access_token = access_token

                def validate_session(self) -> None:
                    return None

                def instruments(self) -> pd.DataFrame:
                    return instruments

                def daily_candles(self, token: int, from_date, to_date) -> pd.DataFrame:
                    return pd.DataFrame()

            def fake_strategy(candles: pd.DataFrame, config: dict) -> pd.DataFrame:
                frame = candles.copy().reset_index(drop=True)
                frame["signal"] = "NONE"
                frame["final_buy"] = False
                frame["final_sell"] = False
                frame["upper_level"] = pd.NA
                frame["lower_level"] = pd.NA
                frame["fvg_bull_recent"] = 0
                frame["fvg_bear_recent"] = 0
                frame["volume_confirmation"] = True
                frame["trend_confirmation"] = True
                frame["avg_volume_20"] = frame["volume"]
                frame["avg_traded_value_20"] = frame["close"] * frame["volume"]
                frame.loc[frame.index[-1], "signal"] = "BUY"
                frame.loc[frame.index[-1], "final_buy"] = True
                return frame

            config = {
                "data": {
                    "scan_timeframe": "1W",
                    "history_years": 1,
                    "data_root_env": "DATA_ROOT",
                },
                "daily_signals": {"enabled": True, "max_signal_age_bars": 1},
                "universe": {
                    "mode": "nse_all",
                    "instrument_types": ["EQ"],
                    "restrict_to_metadata_symbols": False,
                },
                "strategy": {"weekly_anchor": "W-FRI", "use_completed_weeks_only": True},
                "filters": {
                    "enabled": True,
                    "signal": {"direction": "BUY", "latest_only": True, "max_signal_age_bars": 1},
                },
                "notifications": {"enabled": False},
            }

            with (
                patch.dict(os.environ, {"DATA_ROOT": temp_dir}),
                patch("stock_screener.jobs.daily_scan.load_access_token", return_value="token"),
                patch("stock_screener.jobs.daily_scan.KiteDataProvider", FakeProvider),
                patch("stock_screener.jobs.daily_scan.run_weekly_buy_sell", side_effect=fake_strategy),
            ):
                summary = run_daily_scan(config)

            weekly = storage.load_signals("latest_filtered.csv")
            daily = storage.load_signals("latest_daily_filtered.csv")
            audit = storage.load_signals("latest_scan_details.csv")

            self.assertEqual(summary["filtered_matches"], 1)
            self.assertEqual(summary["daily_filtered_matches"], 1)
            self.assertEqual(weekly["timeframe"].tolist(), ["1W"])
            self.assertEqual(daily["timeframe"].tolist(), ["1D"])
            self.assertEqual(audit["latest_daily_signal"].tolist(), ["BUY"])

    def test_cached_only_scan_does_not_require_kite_token_or_provider(self) -> None:
        with TemporaryDirectory() as temp_dir:
            storage = Storage(Path(temp_dir))
            storage.save_instruments(
                pd.DataFrame(
                    [
                        {
                            "exchange": "NSE",
                            "tradingsymbol": "TCS",
                            "instrument_type": "EQ",
                            "segment": "NSE",
                            "instrument_token": 123,
                            "name": "TATA CONSULTANCY SERVICES",
                        }
                    ]
                )
            )
            storage.save_candles(
                "NSE",
                "TCS",
                pd.DataFrame(
                    [
                        {"date": "2026-04-13", "open": 100, "high": 104, "low": 99, "close": 103, "volume": 1000},
                        {"date": "2026-04-17", "open": 103, "high": 112, "low": 101, "close": 111, "volume": 1300},
                    ]
                ),
            )

            def fake_strategy(candles: pd.DataFrame, config: dict) -> pd.DataFrame:
                frame = candles.copy().reset_index(drop=True)
                frame["signal"] = "NONE"
                frame["final_buy"] = False
                frame["final_sell"] = False
                frame["avg_volume_20"] = frame["volume"]
                frame["avg_traded_value_20"] = frame["close"] * frame["volume"]
                frame.loc[frame.index[-1], "signal"] = "BUY"
                frame.loc[frame.index[-1], "final_buy"] = True
                return frame

            config = {
                "data": {
                    "scan_timeframe": "1W",
                    "history_years": 1,
                    "data_root_env": "DATA_ROOT",
                    "skip_kite_fetch": True,
                },
                "daily_signals": {"enabled": True, "max_signal_age_bars": 5},
                "universe": {
                    "mode": "nse_all",
                    "instrument_types": ["EQ"],
                    "restrict_to_metadata_symbols": False,
                },
                "strategy": {"weekly_anchor": "W-FRI", "use_completed_weeks_only": True},
                "filters": {
                    "enabled": True,
                    "signal": {"direction": "BUY", "latest_only": True, "max_signal_age_bars": 1},
                },
                "notifications": {"enabled": False},
            }

            with (
                patch.dict(os.environ, {"DATA_ROOT": temp_dir}),
                patch("stock_screener.jobs.daily_scan.load_access_token", side_effect=AssertionError("Kite token should not be loaded")),
                patch("stock_screener.jobs.daily_scan.KiteDataProvider", side_effect=AssertionError("Kite provider should not be created")),
                patch("stock_screener.jobs.daily_scan.run_weekly_buy_sell", side_effect=fake_strategy),
            ):
                summary = run_daily_scan(config)

            audit = storage.load_signals("latest_scan_details.csv")

            self.assertEqual(summary["refresh_mode"], "cached_only")
            self.assertEqual(summary["symbols_scanned"], 1)
            self.assertEqual(audit["fetch_status"].tolist(), ["cached"])


if __name__ == "__main__":
    unittest.main()
