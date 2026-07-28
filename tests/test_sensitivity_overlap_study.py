from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from stock_screener.sensitivity_overlap_study import (
    _build_overlap_outputs,
    load_sensitivity_overlap_outputs,
    save_sensitivity_overlap_outputs,
)
from stock_screener.web.main import app


class SensitivityOverlapStudyTests(unittest.TestCase):
    def test_overlap_outputs_compute_same_week_and_later_conversion(self) -> None:
        s2 = pd.DataFrame(
            [
                {"exchange": "NSE", "symbol": "AAA", "name": "AAA LTD", "date": "2026-01-06", "close": 100.0, "sensitivity": 2},
                {"exchange": "NSE", "symbol": "BBB", "name": "BBB LTD", "date": "2026-01-06", "close": 100.0, "sensitivity": 2},
                {"exchange": "NSE", "symbol": "CCC", "name": "CCC LTD", "date": "2026-01-13", "close": 100.0, "sensitivity": 2},
            ]
        )
        s3 = pd.DataFrame(
            [
                {"exchange": "NSE", "symbol": "AAA", "name": "AAA LTD", "date": "2026-01-06", "close": 100.0, "sensitivity": 3},
                {"exchange": "NSE", "symbol": "BBB", "name": "BBB LTD", "date": "2026-01-13", "close": 100.0, "sensitivity": 3},
                {"exchange": "NSE", "symbol": "DDD", "name": "DDD LTD", "date": "2026-01-13", "close": 100.0, "sensitivity": 3},
            ]
        )

        result = _build_overlap_outputs(
            "NSE",
            pd.Timestamp("2026-01-01"),
            ["AAA", "BBB", "CCC", "DDD"],
            s2,
            s3,
        )

        self.assertEqual(result.summary["s2_buy_events"], 3)
        self.assertEqual(result.summary["s3_buy_events"], 3)
        self.assertEqual(result.summary["same_week_overlap_events"], 1)
        self.assertAlmostEqual(result.summary["same_week_overlap_pct_of_s2"], 33.333333333333336)
        self.assertEqual(result.summary["next_week_overlap_events"], 1)
        self.assertAlmostEqual(result.summary["next_week_overlap_pct_of_s2"], 33.333333333333336)
        self.assertEqual(result.summary["within_4w_overlap_events"], 2)
        self.assertAlmostEqual(result.summary["within_4w_overlap_pct_of_s2"], 66.66666666666667)
        self.assertEqual(result.summary["extra_s2_events"], 2)
        self.assertEqual(result.summary["extra_s2_later_convert_4w_events"], 1)
        self.assertAlmostEqual(result.summary["extra_s2_later_convert_4w_pct"], 50.0)
        self.assertAlmostEqual(result.summary["next_week_avg_return_pct"], 0.0)
        self.assertAlmostEqual(result.summary["next_week_positive_return_pct"], 0.0)

        latest_buckets = result.latest_cohort.set_index("symbol")["bucket"].to_dict()
        self.assertEqual(latest_buckets["BBB"], "Sensitivity 3 only")
        self.assertEqual(latest_buckets["DDD"], "Sensitivity 3 only")
        weekly = result.weekly_breakdown.set_index("week_date")
        self.assertEqual(int(weekly.loc[pd.Timestamp("2026-01-06"), "extra_later_convert_4w_count"]), 1)
        self.assertEqual(int(weekly.loc[pd.Timestamp("2026-01-06"), "next_week_overlap_count"]), 1)
        self.assertEqual(str(weekly.loc[pd.Timestamp("2026-01-06"), "next_week_symbols"]), "BBB")

        details = result.conversion_details.set_index("symbol")
        self.assertTrue(bool(details.loc["BBB", "next_week_match"]))
        self.assertAlmostEqual(float(details.loc["BBB", "next_week_return_pct"]), 0.0)

    def test_page_renders_saved_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            result = _build_overlap_outputs(
                "NSE",
                pd.Timestamp("2026-01-01"),
                ["AAA", "BBB", "DDD"],
                pd.DataFrame(
                    [
                        {"exchange": "NSE", "symbol": "AAA", "name": "AAA LTD", "date": "2026-01-06", "close": 100.0, "sensitivity": 2},
                        {"exchange": "NSE", "symbol": "BBB", "name": "BBB LTD", "date": "2026-01-13", "close": 101.0, "sensitivity": 2},
                    ]
                ),
                pd.DataFrame(
                    [
                        {"exchange": "NSE", "symbol": "AAA", "name": "AAA LTD", "date": "2026-01-06", "close": 100.0, "sensitivity": 3},
                        {"exchange": "NSE", "symbol": "DDD", "name": "DDD LTD", "date": "2026-01-13", "close": 99.0, "sensitivity": 3},
                    ]
                ),
            )
            save_sensitivity_overlap_outputs(result, data_root / "sensitivity_overlap_study")
            loaded = load_sensitivity_overlap_outputs(data_root / "sensitivity_overlap_study")
            self.assertEqual(int(loaded.summary["s2_buy_events"]), 2)

            with (
                patch.dict(os.environ, {"DATA_ROOT": temp_dir}),
                patch("stock_screener.web.main.get_data_root", return_value=data_root),
            ):
                client = TestClient(app)
                response = client.get("/sensitivity-study")

            self.assertEqual(response.status_code, 200)
            self.assertIn("Sensitivity Study", response.text)
            self.assertIn("BBB", response.text)
            self.assertIn("Same-week overlap", response.text)
            self.assertIn("Next-week confirmation %", response.text)


if __name__ == "__main__":
    unittest.main()
