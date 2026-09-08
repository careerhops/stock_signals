from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from stock_screener.data.storage import Storage


class StorageTests(unittest.TestCase):
    def test_concurrent_candle_merges_preserve_both_updates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = Storage(Path(temp_dir))
            first = pd.DataFrame(
                [{"date": "2026-08-27", "open": 10, "high": 12, "low": 9, "close": 11, "volume": 100}]
            )
            second = pd.DataFrame(
                [{"date": "2026-08-28", "open": 11, "high": 13, "low": 10, "close": 12, "volume": 200}]
            )

            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(storage.merge_and_save_candles, "NSE", "TEST", frame, "1D")
                    for frame in (first, second)
                ]
                for future in futures:
                    future.result(timeout=2.0)

            candles = storage.load_candles("NSE", "TEST", "1D")
            self.assertEqual(candles["date"].dt.strftime("%Y-%m-%d").tolist(), ["2026-08-27", "2026-08-28"])

    def test_load_candles_repairs_three_digit_year_and_rewrites_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            storage = Storage(data_root)
            path = storage.candle_path("NSE", "ABLBL", "1D")
            path.write_text(
                "\n".join(
                    [
                        "date,open,high,low,close,volume",
                        "2026-05-26,102.55,103.34,101.81,102.12,427878",
                        "025-06-24,159.0,162.0,152.2,153.17,2254942",
                        "2025-06-25,153.0,160.8,152.05,154.51,3239904",
                    ]
                )
            )

            candles = storage.load_candles("NSE", "ABLBL", "1D")
            rewritten_lines = path.read_text().splitlines()

            self.assertEqual(str(candles.iloc[0]["date"].date()), "2025-06-24")
            self.assertEqual(str(candles.iloc[1]["date"].date()), "2025-06-25")
            self.assertEqual(rewritten_lines[1].split(",")[0], "2025-06-24")
            self.assertNotEqual(rewritten_lines[1].split(",")[0], "025-06-24")

    def test_incremental_merge_preserves_history_replaces_overlap_and_appends_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = Storage(Path(temp_dir))
            storage.save_candles(
                "NSE",
                "TEST",
                pd.DataFrame(
                    [
                        {"date": "2026-08-27", "open": 10, "high": 12, "low": 9, "close": 11, "volume": 100},
                        {"date": "2026-08-28", "open": 11, "high": 13, "low": 10, "close": 12, "volume": 200},
                    ]
                ),
            )
            incremental = pd.DataFrame(
                [
                    {"date": "2026-08-28", "open": 11, "high": 14, "low": 10, "close": 13, "volume": 250},
                    {"date": "2026-08-31", "open": 13, "high": 15, "low": 12, "close": 14, "volume": 300},
                    {"date": "2026-08-31", "open": 13, "high": 16, "low": 12, "close": 15, "volume": 350},
                ]
            )

            merged = storage.merge_and_save_candles("NSE", "TEST", incremental, "1D")
            persisted = storage.load_candles("NSE", "TEST", "1D")

            self.assertEqual(
                persisted["date"].dt.strftime("%Y-%m-%d").tolist(),
                ["2026-08-27", "2026-08-28", "2026-08-31"],
            )
            self.assertEqual(float(persisted.loc[persisted["date"].dt.day == 28, "close"].iloc[0]), 13.0)
            self.assertEqual(float(persisted.iloc[-1]["close"]), 15.0)
            pd.testing.assert_frame_equal(merged.reset_index(drop=True), persisted.reset_index(drop=True))

    def test_load_signals_recovers_from_malformed_row_and_rewrites_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            storage = Storage(data_root)
            path = storage.signals_dir / "latest_raw_signals.csv"
            path.write_text(
                "\n".join(
                    [
                        "date,signal,symbol,exchange",
                        "2026-06-23,BUY,AAA,NSE",
                        "2026-06-30,BUY,BBB,NSE,2026-06-30,SELL,CCC,NSE",
                        "2026-06-30,BUY,DDD,NSE",
                    ]
                ),
                encoding="utf-8",
            )

            signals = storage.load_signals("latest_raw_signals.csv")
            rewritten_lines = path.read_text(encoding="utf-8").splitlines()

            self.assertEqual(list(signals["symbol"]), ["AAA", "DDD"])
            self.assertEqual(len(rewritten_lines), 3)
            self.assertNotIn("BBB", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
