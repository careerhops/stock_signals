from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from stock_screener.adx_di_study import (
    AdxDiStudyResult,
    load_adx_di_outputs,
    run_adx_di_study,
    save_adx_di_outputs,
)
from stock_screener.data.storage import Storage
from stock_screener.web.main import app


class AdxDiStudyTests(unittest.TestCase):
    def test_adx_di_study_flags_recent_cross_above_di_minus(self) -> None:
        with TemporaryDirectory() as temp_dir:
            storage = Storage(Path(temp_dir))
            storage.save_instruments(
                pd.DataFrame(
                    [
                        {"exchange": "NSE", "tradingsymbol": "PASS", "name": "Pass Ltd"},
                        {"exchange": "NSE", "tradingsymbol": "LEAD", "name": "Lead Ltd"},
                        {"exchange": "NSE", "tradingsymbol": "FAIL", "name": "Fail Ltd"},
                        {"exchange": "NSE", "tradingsymbol": "ADXHIGH", "name": "ADX High Ltd"},
                        {"exchange": "NSE", "tradingsymbol": "WRONGSLOPE", "name": "Wrong Slope Ltd"},
                        {"exchange": "NSE", "tradingsymbol": "SKIP-SM", "name": "Skip Ltd"},
                        {"exchange": "NSE", "tradingsymbol": "AXISBNKETF", "name": "Axis Bank ETF"},
                        {"exchange": "NSE", "tradingsymbol": "MOMENTUM50", "name": "Momentum 50"},
                        {"exchange": "NSE", "tradingsymbol": "NIFTYBEES", "name": "Nifty Bees"},
                        {"exchange": "NSE", "tradingsymbol": "GOLDBEES", "name": "Gold Bees"},
                    ]
                )
            )
            storage.save_candles("NSE", "PASS", self._pass_frame(), "1D")
            storage.save_candles("NSE", "LEAD", self._lead_frame(), "1D")
            storage.save_candles("NSE", "FAIL", self._fail_frame(), "1D")
            storage.save_candles("NSE", "ADXHIGH", self._adx_high_frame(), "1D")
            storage.save_candles("NSE", "WRONGSLOPE", self._wrong_slope_frame(), "1D")
            storage.save_candles("NSE", "SKIP-SM", self._pass_frame(), "1D")
            storage.save_candles("NSE", "AXISBNKETF", self._pass_frame(), "1D")
            storage.save_candles("NSE", "MOMENTUM50", self._pass_frame(), "1D")
            storage.save_candles("NSE", "NIFTYBEES", self._pass_frame(), "1D")
            storage.save_candles("NSE", "GOLDBEES", self._pass_frame(), "1D")

            result = run_adx_di_study(
                storage,
                exchange="NSE",
                length=14,
                threshold=20.0,
                cross_lookback_bars=3,
                max_staleness_days=60,
            )
            save_adx_di_outputs(result, Path(temp_dir) / "adx_di")
            loaded = load_adx_di_outputs(Path(temp_dir) / "adx_di")

        self.assertEqual(int(loaded.summary["symbols_processed"]), 5)
        self.assertEqual(int(loaded.summary["adx_cross_matches"]), 1)
        pass_row = loaded.stock_stats[loaded.stock_stats["symbol"] == "PASS"].iloc[0]
        lead_row = loaded.stock_stats[loaded.stock_stats["symbol"] == "LEAD"].iloc[0]
        fail_row = loaded.stock_stats[loaded.stock_stats["symbol"] == "FAIL"].iloc[0]
        adx_high_row = loaded.stock_stats[loaded.stock_stats["symbol"] == "ADXHIGH"].iloc[0]
        wrong_slope_row = loaded.stock_stats[loaded.stock_stats["symbol"] == "WRONGSLOPE"].iloc[0]
        self.assertNotIn("SKIP-SM", loaded.stock_stats["symbol"].tolist())
        self.assertNotIn("AXISBNKETF", loaded.stock_stats["symbol"].tolist())
        self.assertNotIn("MOMENTUM50", loaded.stock_stats["symbol"].tolist())
        self.assertNotIn("NIFTYBEES", loaded.stock_stats["symbol"].tolist())
        self.assertNotIn("GOLDBEES", loaded.stock_stats["symbol"].tolist())
        self.assertFalse(bool(pass_row["di_plus_cross_above_di_minus_recent"]))
        self.assertEqual(pass_row["latest_di_plus_cross_date"], "2026-03-20")
        self.assertTrue(pd.isna(pass_row["recent_di_plus_cross_dates_csv"]) or pass_row["recent_di_plus_cross_dates_csv"] == "")
        self.assertFalse(bool(pass_row["di_plus_cross_above_di_minus_latest"]))
        self.assertTrue(bool(pass_row["di_plus_above_di_minus"]))
        self.assertTrue(bool(lead_row["di_plus_lead_pending"]))
        self.assertEqual(lead_row["latest_di_plus_cross_date"], "2026-03-10")
        self.assertTrue(pd.isna(lead_row["latest_cross_date"]) or lead_row["latest_cross_date"] == "")
        self.assertTrue(bool(lead_row["di_plus_cross_above_di_minus_recent"]))
        self.assertTrue(bool(lead_row["di_plus_cross_over_threshold_recent"]))
        self.assertEqual(lead_row["latest_di_plus_cross_over_threshold_date"], "2026-03-10")
        self.assertFalse(bool(lead_row["adx_bullish_cross_above_di_minus_recent"]))
        self.assertFalse(bool(fail_row["di_plus_cross_above_di_minus_recent"]))
        self.assertFalse(bool(adx_high_row["di_plus_cross_above_di_minus_recent"]))
        self.assertFalse(bool(wrong_slope_row["di_plus_cross_above_di_minus_recent"]))

    def test_adx_di_page_renders_saved_outputs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            result = AdxDiStudyResult(
                summary={
                    "exchange": "NSE",
                    "symbols_processed": 2,
                    "stocks_with_history": 2,
                    "adx_cross_matches": 1,
                    "latest_close_date": "2026-07-23",
                    "avg_latest_adx": 18.4,
                    "length": 14,
                    "threshold": 20.0,
                    "cross_lookback_bars": 3,
                },
                stock_stats=pd.DataFrame(
                    [
                        {
                            "exchange": "NSE",
                            "symbol": "PASS",
                            "name": "Pass Ltd",
                            "latest_close": 121.0,
                            "latest_close_date": "2026-07-23",
                            "latest_di_plus": 26.66,
                            "latest_di_minus": 11.86,
                            "latest_adx": 14.57,
                            "latest_adx_20": 12.25,
                            "adx_3d_ago": 11.42,
                            "adx_minus_di_minus_gap": 2.71,
                            "di_plus_above_di_minus": True,
                            "adx_above_adx20": True,
                            "adx_above_3d_ago": True,
                            "adx_shortlist_pass": True,
                            "di_plus_crosses_in_lookback_bars": 1,
                            "recent_di_plus_cross_dates_csv": "2026-07-21",
                            "latest_di_plus_cross_date": "2026-07-21",
                            "di_plus_cross_above_di_minus_recent": True,
                            "di_plus_cross_above_di_minus_latest": False,
                            "di_plus_divergence_count": 1,
                            "recent_di_plus_divergence_dates_csv": "2026-07-23",
                            "latest_di_plus_divergence_date": "2026-07-23",
                            "di_plus_divergence_recent": True,
                            "di_plus_divergence_expanding_latest": True,
                            "di_plus_cross_over_threshold_count": 1,
                            "recent_di_plus_cross_over_threshold_dates_csv": "2026-07-21",
                            "latest_di_plus_cross_over_threshold_date": "2026-07-21",
                            "di_plus_cross_over_threshold_recent": True,
                            "di_plus_cross_over_threshold_latest": False,
                            "obv_latest": 125000.0,
                            "obv_sma13": 117500.0,
                            "obv_above_sma13": True,
                            "obv_cross_sma13_count": 1,
                            "recent_obv_cross_sma13_dates_csv": "2026-07-21",
                            "latest_obv_cross_sma13_date": "2026-07-21",
                            "obv_cross_sma13_recent": True,
                            "obv_cross_sma13_latest": False,
                            "di_plus_lead_pending": False,
                            "adx_above_threshold": False,
                            "crosses_in_lookback_bars": 1,
                            "recent_cross_dates_csv": "2026-07-22",
                            "latest_cross_date": "2026-07-22",
                            "adx_bullish_cross_above_di_minus_recent": True,
                            "adx_bullish_cross_above_di_minus_latest": False,
                            "support_level": 100.0,
                            "support_level_date": "2026-06-30",
                            "support_distance_from_level_pct": 21.0,
                            "support_filter_pass": True,
                        },
                        {
                            "exchange": "NSE",
                            "symbol": "PRE",
                            "name": "Pre Divergence Ltd",
                            "latest_close": 111.0,
                            "latest_close_date": "2026-07-23",
                            "latest_di_plus": 18.5,
                            "latest_di_minus": 22.0,
                            "latest_adx": 12.0,
                            "latest_adx_20": 10.5,
                            "adx_3d_ago": 10.8,
                            "adx_minus_di_minus_gap": -10.0,
                            "di_plus_above_di_minus": False,
                            "adx_above_adx20": True,
                            "adx_above_3d_ago": True,
                            "adx_shortlist_pass": False,
                            "di_plus_crosses_in_lookback_bars": 0,
                            "recent_di_plus_cross_dates_csv": "",
                            "latest_di_plus_cross_date": "",
                            "di_plus_cross_above_di_minus_recent": False,
                            "di_plus_cross_above_di_minus_latest": False,
                            "di_plus_divergence_count": 0,
                            "recent_di_plus_divergence_dates_csv": "",
                            "latest_di_plus_divergence_date": "",
                            "di_plus_divergence_recent": False,
                            "di_plus_divergence_expanding_latest": False,
                            "di_plus_pre_cross_threshold_divergence_count": 1,
                            "recent_di_plus_pre_cross_threshold_divergence_dates_csv": "2026-07-23",
                            "latest_di_plus_pre_cross_threshold_divergence_date": "2026-07-23",
                            "di_plus_pre_cross_threshold_divergence_recent": True,
                            "di_plus_pre_cross_threshold_divergence_expanding_latest": True,
                            "di_plus_cross_over_threshold_count": 0,
                            "recent_di_plus_cross_over_threshold_dates_csv": "",
                            "latest_di_plus_cross_over_threshold_date": "",
                            "di_plus_cross_over_threshold_recent": False,
                            "di_plus_cross_over_threshold_latest": False,
                            "obv_latest": 90000.0,
                            "obv_sma13": 87000.0,
                            "obv_above_sma13": True,
                            "obv_cross_sma13_count": 0,
                            "recent_obv_cross_sma13_dates_csv": "",
                            "latest_obv_cross_sma13_date": "",
                            "obv_cross_sma13_recent": False,
                            "obv_cross_sma13_latest": False,
                            "di_plus_lead_pending": False,
                            "adx_above_threshold": False,
                            "crosses_in_lookback_bars": 0,
                            "recent_cross_dates_csv": "",
                            "latest_cross_date": "",
                            "adx_bullish_cross_above_di_minus_recent": False,
                            "adx_bullish_cross_above_di_minus_latest": False,
                            "support_level": 103.0,
                            "support_level_date": "2026-06-30",
                            "support_distance_from_level_pct": 7.5,
                            "support_filter_pass": False,
                        },
                        {
                            "exchange": "NSE",
                            "symbol": "FAIL",
                            "name": "Fail Ltd",
                            "latest_close": 98.0,
                            "latest_close_date": "2026-07-23",
                            "latest_di_plus": 11.0,
                            "latest_di_minus": 17.0,
                            "latest_adx": 9.0,
                            "latest_adx_20": 9.5,
                            "adx_3d_ago": 10.0,
                            "adx_minus_di_minus_gap": -8.0,
                            "di_plus_above_di_minus": False,
                            "adx_above_adx20": False,
                            "adx_above_3d_ago": False,
                            "adx_shortlist_pass": False,
                            "di_plus_crosses_in_lookback_bars": 1,
                            "recent_di_plus_cross_dates_csv": "2026-07-22",
                            "latest_di_plus_cross_date": "2026-07-22",
                            "di_plus_cross_above_di_minus_recent": True,
                            "di_plus_cross_above_di_minus_latest": False,
                            "di_plus_divergence_count": 0,
                            "recent_di_plus_divergence_dates_csv": "",
                            "latest_di_plus_divergence_date": "",
                            "di_plus_divergence_recent": False,
                            "di_plus_divergence_expanding_latest": False,
                            "di_plus_cross_over_threshold_count": 1,
                            "recent_di_plus_cross_over_threshold_dates_csv": "2026-07-22",
                            "latest_di_plus_cross_over_threshold_date": "2026-07-22",
                            "di_plus_cross_over_threshold_recent": True,
                            "di_plus_cross_over_threshold_latest": False,
                            "obv_latest": 82000.0,
                            "obv_sma13": 91000.0,
                            "obv_above_sma13": False,
                            "obv_cross_sma13_count": 0,
                            "recent_obv_cross_sma13_dates_csv": "",
                            "latest_obv_cross_sma13_date": "",
                            "obv_cross_sma13_recent": False,
                            "obv_cross_sma13_latest": False,
                            "di_plus_lead_pending": False,
                            "adx_above_threshold": False,
                            "crosses_in_lookback_bars": 0,
                            "recent_cross_dates_csv": "",
                            "latest_cross_date": "",
                            "adx_bullish_cross_above_di_minus_recent": False,
                            "adx_bullish_cross_above_di_minus_latest": False,
                            "support_level": 90.0,
                            "support_level_date": "2026-06-20",
                            "support_distance_from_level_pct": 8.0,
                            "support_filter_pass": False,
                        },
                    ]
                ),
            )
            save_adx_di_outputs(result, data_root / "adx_di")

            with (
                patch.dict(os.environ, {"DATA_ROOT": temp_dir}),
                patch("stock_screener.web.main.get_data_root", return_value=data_root),
                patch(
                    "stock_screener.web.main._combined_symbol_metadata",
                    return_value=pd.DataFrame(
                        [
                            {
                                "symbol": "PASS",
                                "industry": "Information Technology",
                                "market_cap_cr": 1500.0,
                            }
                        ]
                    ),
                ),
            ):
                client = TestClient(app)
                response = client.get("/adx-di?matches_only=1")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Daily DI+ signal scan", response.text)
        self.assertIn("PASS", response.text)
        self.assertNotIn("PRE", response.text)
        self.assertNotIn("FAIL", response.text)
        self.assertIn("Show only shortlisted DI+ setups", response.text)
        self.assertIn("Sector / Industry Mix", response.text)
        self.assertIn("Sector Leaders", response.text)
        self.assertIn("Information Technology", response.text)
        self.assertIn("Close vs Support %", response.text)
        self.assertIn("Support distance filter", response.text)
        self.assertIn("DI+ cross above threshold", response.text)
        self.assertIn("OBV crossed above 13D SMA", response.text)
        self.assertIn("DI+ divergence expanding", response.text)
        self.assertIn("Pre-Cross Threshold Divergence", response.text)
        self.assertIn("ADX(20)", response.text)
        self.assertIn("Shortlist Pass", response.text)

    @staticmethod
    def _pass_frame() -> pd.DataFrame:
        dates = pd.date_range("2026-01-01", periods=60, freq="B")
        close = [
            100.0, 101.0, 104.0, 102.0, 103.0, 100.0, 101.0, 98.0, 97.0, 96.0,
            99.0, 100.0, 102.0, 105.0, 107.0, 110.0, 111.0, 114.0, 116.0, 113.0,
            116.0, 119.0, 118.0, 121.0, 122.0, 120.0, 119.0, 116.0, 118.0, 115.0,
            117.0, 118.0, 117.0, 114.0, 117.0, 118.0, 117.0, 117.0, 117.0, 116.0,
            116.0, 115.0, 115.0, 114.0, 113.0, 113.0, 112.0, 111.0, 111.0, 110.0,
            109.0, 110.0, 110.0, 109.0, 108.0, 112.0, 116.0, 118.0, 121.0, 121.0,
        ]
        open_values = [
            100.0, 100.0, 101.0, 104.0, 102.0, 103.0, 100.0, 101.0, 98.0, 97.0,
            96.0, 99.0, 100.0, 102.0, 105.0, 107.0, 110.0, 111.0, 114.0, 116.0,
            113.0, 116.0, 119.0, 118.0, 121.0, 122.0, 120.0, 119.0, 116.0, 118.0,
            115.0, 117.0, 118.0, 117.0, 114.0, 117.0, 118.0, 117.0, 117.0, 117.0,
            116.0, 116.0, 115.0, 115.0, 114.0, 113.0, 113.0, 112.0, 111.0, 111.0,
            110.0, 109.0, 110.0, 110.0, 109.0, 108.0, 112.0, 116.0, 118.0, 121.0,
        ]
        high = [
            101.1722, 101.6599, 105.3366, 104.7105, 103.9064, 104.2192, 101.5862, 102.1642, 99.1746, 98.4066,
            100.2130, 101.0734, 102.9565, 105.7140, 108.1599, 110.8125, 111.8623, 115.7439, 117.0268, 117.4205,
            117.4046, 119.8651, 119.7757, 122.1836, 123.6389, 122.9354, 121.3291, 119.7069, 119.7198, 119.0091,
            117.9196, 119.7863, 118.7759, 117.6596, 118.6167, 118.8529, 118.7779, 117.5565, 118.4748, 118.7481,
            116.5911, 117.2293, 116.4444, 116.4018, 115.3370, 114.4687, 114.6548, 112.6758, 111.7671, 112.6835,
            110.6134, 111.1552, 110.5252, 110.5195, 109.8604, 112.9971, 116.6013, 118.6453, 121.7894, 122.4481,
        ]
        low = [
            99.3912, 98.4177, 99.2009, 100.6265, 100.2611, 98.6322, 99.2275, 96.8017, 95.6725, 94.8136,
            94.4727, 97.6358, 98.6542, 101.1449, 103.3059, 106.2529, 109.1637, 110.0558, 113.3812, 111.5077,
            112.3003, 114.3333, 117.1358, 117.3589, 119.3786, 118.8867, 117.6294, 115.2426, 114.6748, 113.7028,
            114.4519, 115.4855, 115.2561, 112.9573, 112.5328, 115.8556, 116.2281, 116.3017, 115.5752, 114.3530,
            114.5344, 113.7046, 113.8084, 112.5583, 111.3821, 111.2978, 110.3158, 110.1726, 110.3131, 108.3474,
            107.6713, 107.9247, 109.0525, 107.4578, 106.4166, 107.3580, 110.5695, 115.3581, 116.3810, 120.0535,
        ]
        return pd.DataFrame(
            {
                "date": dates,
                "open": open_values,
                "high": high,
                "low": low,
                "close": close,
                "volume": [100.0] * len(dates),
            }
        )

    @staticmethod
    def _fail_frame() -> pd.DataFrame:
        dates = pd.date_range("2026-01-01", periods=60, freq="B")
        close = [100.0 + float(index) for index in range(len(dates))]
        return pd.DataFrame(
            {
                "date": dates,
                "open": close,
                "high": [value + 1.0 for value in close],
                "low": [value - 1.0 for value in close],
                "close": close,
                "volume": [100.0] * len(dates),
            }
        )

    @staticmethod
    def _wrong_slope_frame() -> pd.DataFrame:
        base = AdxDiStudyTests._pass_frame().copy()
        base["close"] = base["close"].shift(1).fillna(base["close"])
        base["open"] = base["close"]
        base["high"] = base["close"] + 1.0
        base["low"] = base["close"] - 1.0
        return base

    @staticmethod
    def _lead_frame() -> pd.DataFrame:
        dates = pd.date_range("2026-01-01", periods=60, freq="B")
        close: list[float] = []
        price = 100.0
        for _ in range(45):
            price -= 0.6
            close.append(round(price, 2))
        for delta in [0.8, 1.0, 1.2, 1.5, 1.7, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3]:
            price += delta
            close.append(round(price, 2))

        frame = pd.DataFrame({"date": dates, "close": close})
        frame["open"] = frame["close"].shift(1).fillna(frame["close"])
        frame["high"] = frame[["open", "close"]].max(axis=1) + 0.6
        frame["low"] = frame[["open", "close"]].min(axis=1) - 0.6
        frame["volume"] = 100.0
        return frame[frame["date"] <= "2026-03-10"].reset_index(drop=True)

    @staticmethod
    def _adx_high_frame() -> pd.DataFrame:
        base = AdxDiStudyTests._lead_frame().copy()
        base["high"] = base["high"] + 4.0
        base["low"] = base["low"] - 4.0
        return base


if __name__ == "__main__":
    unittest.main()
