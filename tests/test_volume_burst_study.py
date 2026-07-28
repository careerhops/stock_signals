from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from stock_screener.data.storage import Storage
from stock_screener.volume_burst_study import load_volume_burst_outputs, run_volume_burst_study, save_volume_burst_outputs


class VolumeBurstStudyTests(unittest.TestCase):
    def test_volume_burst_study_flags_latest_3x_volume_stock(self) -> None:
        with TemporaryDirectory() as temp_dir:
            storage = Storage(Path(temp_dir))
            dates = pd.date_range("2026-01-01", periods=12, freq="B")
            pass_frame = pd.DataFrame(
                {
                    "date": dates,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": [100.0] * 11 + [400.0],
                }
            )
            fail_frame = pd.DataFrame(
                {
                    "date": dates,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": [100.0] * 11 + [250.0],
                }
            )
            instruments = pd.DataFrame(
                [
                    {"exchange": "NSE", "tradingsymbol": "PASS", "name": "Pass Ltd"},
                    {"exchange": "NSE", "tradingsymbol": "FAIL", "name": "Fail Ltd"},
                ]
            )
            storage.save_instruments(instruments)
            storage.save_candles("NSE", "PASS", pass_frame, "1D")
            storage.save_candles("NSE", "FAIL", fail_frame, "1D")

            result = run_volume_burst_study(storage, exchange="NSE")
            save_volume_burst_outputs(result, Path(temp_dir) / "volume_burst")
            loaded = load_volume_burst_outputs(Path(temp_dir) / "volume_burst")

        self.assertEqual(int(loaded.summary["symbols_processed"]), 2)
        self.assertEqual(int(loaded.summary["volume_burst_matches"]), 1)
        pass_row = loaded.stock_stats[loaded.stock_stats["symbol"] == "PASS"].iloc[0]
        fail_row = loaded.stock_stats[loaded.stock_stats["symbol"] == "FAIL"].iloc[0]
        self.assertTrue(bool(pass_row["latest_volume_3x_prev_9d"]))
        self.assertFalse(bool(fail_row["latest_volume_3x_prev_9d"]))
        self.assertEqual(float(pass_row["latest_volume_ratio_prev_9d"]), 4.0)


if __name__ == "__main__":
    unittest.main()
