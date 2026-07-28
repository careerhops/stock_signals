from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stock_screener.data.storage import Storage


class StorageTests(unittest.TestCase):
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
