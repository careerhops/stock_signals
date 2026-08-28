from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from stock_screener.data.storage import Storage
from stock_screener.minervini_quality_study import (
    MinerviniQualityStudyResult,
    evaluate_minervini_quality,
    load_minervini_quality_outputs,
    run_minervini_quality_study,
    save_minervini_quality_outputs,
)
from stock_screener.web.main import app


class MinerviniQualityStudyTests(unittest.TestCase):
    def test_extended_scores_match_strict_three_score_gate(self) -> None:
        stock, benchmark = self._quality_frames()

        result = evaluate_minervini_quality(stock, benchmark, score_threshold=70.0)
        strict_result = evaluate_minervini_quality(stock, benchmark, score_threshold=90.0)

        self.assertEqual(result["trend_pass_count"], 7)
        self.assertAlmostEqual(result["stock_quality_score"], 85.0)
        self.assertAlmostEqual(result["setup_quality_score"], 79.0)
        self.assertAlmostEqual(result["entry_quality_score"], 90.0)
        expected_high = float(stock.tail(252)["high"].max())
        expected_distance = (expected_high - float(stock.iloc[-1]["close"])) / expected_high * 100.0
        self.assertAlmostEqual(result["latest_52w_high"], expected_high)
        self.assertAlmostEqual(result["distance_below_52w_high_pct"], expected_distance)
        self.assertTrue(result["quality_pass"])
        self.assertFalse(strict_result["quality_pass"])

    def test_short_history_never_qualifies(self) -> None:
        stock, benchmark = self._quality_frames()

        result = evaluate_minervini_quality(stock.tail(200), benchmark, score_threshold=0.0)

        self.assertEqual(result["data_status"], "SHORT_HISTORY")
        self.assertFalse(result["quality_pass"])

    def test_scan_persists_scores_and_summary(self) -> None:
        with TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            storage = Storage(data_root)
            stock, benchmark = self._quality_frames()
            storage.save_instruments(
                pd.DataFrame([{"exchange": "NSE", "tradingsymbol": "QUALITY", "name": "Quality Ltd"}])
            )
            storage.save_candles("NSE", "QUALITY", stock, "1D")
            storage.save_candles("NSE_INDEX", "NIFTY 500", benchmark, "1D")

            result = run_minervini_quality_study(storage)
            save_minervini_quality_outputs(result, data_root / "minervini_quality")
            loaded = load_minervini_quality_outputs(data_root / "minervini_quality")

        self.assertEqual(int(loaded.summary["symbols_processed"]), 1)
        self.assertEqual(int(loaded.summary["qualified_stocks"]), 1)
        self.assertEqual(loaded.summary["benchmark_symbol"], "NIFTY 500")
        self.assertEqual(loaded.stock_stats.iloc[0]["symbol"], "QUALITY")

    def test_scan_uses_requested_current_universe_and_rejects_stale_stock_date(self) -> None:
        with TemporaryDirectory() as temp_dir:
            storage = Storage(Path(temp_dir))
            stock, benchmark = self._quality_frames()
            storage.save_candles("NSE", "CURRENT", stock, "1D")
            storage.save_candles("NSE", "STALE", stock.iloc[:-1], "1D")
            storage.save_candles("NSE", "OLD-CACHE", stock, "1D")
            storage.save_candles("NSE_INDEX", "NIFTY 500", benchmark, "1D")

            result = run_minervini_quality_study(
                storage,
                symbols=["CURRENT", "STALE"],
            )

        self.assertEqual(set(result.stock_stats["symbol"]), {"CURRENT", "STALE"})
        current = result.stock_stats.set_index("symbol").loc["CURRENT"]
        stale = result.stock_stats.set_index("symbol").loc["STALE"]
        self.assertTrue(bool(current["is_latest_market_date"]))
        self.assertTrue(bool(current["quality_pass"]))
        self.assertFalse(bool(stale["is_latest_market_date"]))
        self.assertFalse(bool(stale["quality_pass"]))
        self.assertEqual(int(result.summary["stale_stock_dates"]), 1)

    def test_page_renders_qualified_results(self) -> None:
        with TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            result = MinerviniQualityStudyResult(
                summary={
                    "symbols_processed": 2,
                    "stocks_scored": 2,
                    "qualified_stocks": 1,
                    "score_threshold": 70.0,
                    "benchmark_symbol": "NIFTY 500",
                    "benchmark_latest_date": "2026-08-07",
                    "latest_stock_date": "2026-08-07",
                },
                stock_stats=pd.DataFrame(
                    [
                        {
                            "exchange": "NSE",
                            "symbol": "QUALITY",
                            "name": "Quality Ltd",
                            "latest_date": "2026-08-07",
                            "latest_close": 150.0,
                            "latest_52w_high": 160.0,
                            "distance_below_52w_high_pct": 6.25,
                            "stock_quality_score": 85.0,
                            "stock_quality_grade": "LEADER",
                            "setup_quality_score": 79.0,
                            "setup_quality_grade": "DEVELOPING",
                            "entry_quality_score": 90.0,
                            "entry_quality_grade": "READY",
                            "quality_pass": True,
                            "trend_pass_count": 7,
                            "data_status": "READY",
                        },
                        {
                            "exchange": "NSE",
                            "symbol": "FAIL",
                            "name": "Fail Ltd",
                            "latest_date": "2026-08-07",
                            "latest_close": 50.0,
                            "stock_quality_score": 60.0,
                            "setup_quality_score": 60.0,
                            "entry_quality_score": 60.0,
                            "quality_pass": False,
                            "data_status": "READY",
                        },
                    ]
                ),
            )
            save_minervini_quality_outputs(result, data_root / "minervini_quality")

            with (
                patch.dict(os.environ, {"DATA_ROOT": temp_dir}),
                patch("stock_screener.web.main.get_data_root", return_value=data_root),
            ):
                response = TestClient(app).get("/minervini-quality?qualified_only=1&score_threshold=70")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Stock, setup, and entry quality", response.text)
        self.assertIn("QUALITY", response.text)
        self.assertNotIn(">FAIL<", response.text)
        self.assertIn("Run Minervini Quality Scan", response.text)
        self.assertIn("Below 52W High %", response.text)
        self.assertIn("6.25", response.text)

    @staticmethod
    def _quality_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
        periods = 320
        dates = pd.bdate_range("2025-01-01", periods=periods)
        base = np.linspace(50.0, 145.0, periods - 20)
        tail = np.array(
            [
                146.0,
                146.8,
                146.2,
                147.0,
                146.6,
                147.4,
                147.0,
                147.8,
                147.4,
                148.1,
                147.8,
                148.5,
                148.2,
                148.8,
                148.5,
                149.0,
                148.9,
                149.2,
                149.1,
                149.4,
            ]
        )
        close = np.concatenate([base, tail])
        spread = np.concatenate([np.full(periods - 20, 1.5), np.linspace(3.0, 0.4, 20)])
        volume = np.concatenate(
            [np.full(periods - 20, 200_000.0), np.full(15, 220_000.0), np.full(5, 70_000.0)]
        )
        stock = pd.DataFrame(
            {
                "date": dates,
                "open": close - 0.15,
                "high": close + spread / 2.0,
                "low": close - spread / 2.0,
                "close": close,
                "volume": volume,
            }
        )
        benchmark = pd.DataFrame({"date": dates, "close": np.linspace(100.0, 125.0, periods)})
        return stock, benchmark


if __name__ == "__main__":
    unittest.main()
