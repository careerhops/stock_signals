from __future__ import annotations

from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from stock_screener.data.storage import Storage
from stock_screener.web.main import (
    _latest_completed_nse_calendar_date,
    _refresh_adx_di_candles,
    _refresh_minervini_quality_benchmark,
    _refresh_trader_setup_history,
)


class KiteOhlcFreshnessTests(unittest.TestCase):
    def test_completed_nse_date_excludes_intraday_daily_candle(self) -> None:
        self.assertEqual(
            _latest_completed_nse_calendar_date(pd.Timestamp("2026-08-27 12:45", tz="Asia/Kolkata")),
            date(2026, 8, 26),
        )
        self.assertEqual(
            _latest_completed_nse_calendar_date(pd.Timestamp("2026-08-27 15:45", tz="Asia/Kolkata")),
            date(2026, 8, 27),
        )

    def test_refresh_includes_only_symbols_on_fresh_benchmark_date(self) -> None:
        with TemporaryDirectory() as temp_dir:
            storage = Storage(Path(temp_dir))
            provider = _FakeKiteProvider()
            config = {
                "data": {
                    "history_years": 5,
                    "readiness": {"minimum_symbols_updated_percent": 50},
                }
            }
            with (
                patch("stock_screener.web.main.load_access_token", return_value="token"),
                patch("stock_screener.web.main.KiteDataProvider", return_value=provider),
                patch("stock_screener.web.main.load_config", return_value=config),
            ):
                expected_date = _refresh_minervini_quality_benchmark(storage, "NIFTY 500")
                symbols, audit = _refresh_adx_di_candles(
                    storage,
                    required_date=expected_date,
                )

        self.assertEqual(expected_date, date(2026, 8, 17))
        self.assertEqual(symbols, ["CURRENT"])
        self.assertEqual(audit["refresh_expected_date"], "2026-08-17")
        self.assertEqual(audit["refresh_current_count"], 1)
        self.assertEqual(audit["refresh_stale_count"], 1)
        self.assertEqual(audit["refresh_failed_count"], 0)
        self.assertEqual(audit["refresh_coverage_pct"], 50.0)

    def test_refresh_rejects_run_below_required_coverage(self) -> None:
        with TemporaryDirectory() as temp_dir:
            storage = Storage(Path(temp_dir))
            provider = _FakeKiteProvider()
            config = {
                "data": {
                    "history_years": 5,
                    "readiness": {"minimum_symbols_updated_percent": 80},
                }
            }
            with (
                patch("stock_screener.web.main.load_access_token", return_value="token"),
                patch("stock_screener.web.main.KiteDataProvider", return_value=provider),
                patch("stock_screener.web.main.load_config", return_value=config),
            ):
                with self.assertRaisesRegex(RuntimeError, "previous screener result was preserved"):
                    _refresh_adx_di_candles(
                        storage,
                        required_date=date(2026, 8, 17),
                    )

    def test_trader_setup_refresh_backfills_ten_years_for_all_equity_symbols(self) -> None:
        with TemporaryDirectory() as temp_dir:
            storage = Storage(Path(temp_dir))
            provider = _AllFreshKiteProvider()
            config = {
                "data": {"readiness": {"minimum_symbols_updated_percent": 80}},
                "universe": {"mode": "nse_all", "instrument_types": ["EQ"]},
            }
            with (
                patch("stock_screener.web.main.load_access_token", return_value="token"),
                patch("stock_screener.web.main.KiteDataProvider", return_value=provider),
                patch("stock_screener.web.main.load_config", return_value=config),
                patch("stock_screener.web.main.time.sleep", return_value=None),
            ):
                symbols, audit = _refresh_trader_setup_history(
                    storage,
                    required_date=date(2026, 8, 17),
                    start_date=date(2016, 8, 17),
                )

        self.assertEqual(symbols, ["CURRENT", "SMALL-SM"])
        self.assertEqual(audit["refresh_universe_count"], 2)
        self.assertEqual(audit["refresh_coverage_pct"], 100.0)
        self.assertEqual(min(call[1] for call in provider.calls), date(2016, 8, 17))


class _FakeKiteProvider:
    def validate_session(self) -> dict[str, str]:
        return {"user_id": "TEST"}

    def instruments(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "exchange": "NSE",
                    "segment": "NSE",
                    "instrument_type": "EQ",
                    "tradingsymbol": "CURRENT",
                    "instrument_token": 1,
                    "name": "Current Ltd",
                },
                {
                    "exchange": "NSE",
                    "segment": "NSE",
                    "instrument_type": "EQ",
                    "tradingsymbol": "STALE",
                    "instrument_token": 2,
                    "name": "Stale Ltd",
                },
                {
                    "exchange": "NSE",
                    "segment": "INDICES",
                    "instrument_type": "EQ",
                    "tradingsymbol": "NIFTY 500",
                    "instrument_token": 500,
                    "name": "Nifty 500",
                },
            ]
        )

    def daily_candles(self, instrument_token: int, from_date: date, to_date: date) -> pd.DataFrame:
        latest = date(2026, 8, 17) if instrument_token in {1, 500} else date(2026, 8, 14)
        return pd.DataFrame(
            [
                {
                    "date": latest,
                    "open": 100.0,
                    "high": 105.0,
                    "low": 99.0,
                    "close": 103.0,
                    "volume": 100_000.0,
                }
            ]
        )


class _AllFreshKiteProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[int, date, date]] = []

    def validate_session(self) -> dict[str, str]:
        return {"user_id": "TEST"}

    def instruments(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"exchange": "NSE", "segment": "NSE", "instrument_type": "EQ", "tradingsymbol": "CURRENT", "instrument_token": 1, "name": "Current Ltd"},
                {"exchange": "NSE", "segment": "NSE", "instrument_type": "EQ", "tradingsymbol": "SMALL-SM", "instrument_token": 2, "name": "Small Ltd"},
                {"exchange": "NSE", "segment": "NSE", "instrument_type": "EQ", "tradingsymbol": "BANKETF", "instrument_token": 3, "name": "Example Nifty Bank ETF"},
                {"exchange": "NSE", "segment": "NSE", "instrument_type": "EQ", "tradingsymbol": "FUNDINAV", "instrument_token": 4, "name": "Example Fund Indicative NAV"},
                {"exchange": "NSE", "segment": "NSE", "instrument_type": "EQ", "tradingsymbol": "TRUST-RR", "instrument_token": 5, "name": "Example Real Estate Trust"},
                {"exchange": "NSE", "segment": "NSE", "instrument_type": "EQ", "tradingsymbol": "BANK10BETF", "instrument_token": 6, "name": "Example AMC Bank Fund"},
                {"exchange": "NSE", "segment": "NSE", "instrument_type": "EQ", "tradingsymbol": "AONESILVER", "instrument_token": 7, "name": "Example AMC - Silver Unit"},
            ]
        )

    def daily_candles(self, instrument_token: int, from_date: date, to_date: date) -> pd.DataFrame:
        self.calls.append((instrument_token, from_date, to_date))
        return pd.DataFrame(
            [{"date": to_date, "open": 100.0, "high": 105.0, "low": 99.0, "close": 103.0, "volume": 100_000.0}]
        )

if __name__ == "__main__":
    unittest.main()
