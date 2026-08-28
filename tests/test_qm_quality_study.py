from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from stock_screener.qm_quality_study import QMQualityStudyResult, load_qm_quality_outputs, save_qm_quality_outputs
from stock_screener.web.main import app


class QMQualityStudyTests(unittest.TestCase):
    def test_qm_quality_page_renders_saved_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            result = QMQualityStudyResult(
                summary={
                    "exchange": "NSE",
                    "buy_start_date": "2026-04-01",
                    "buy_end_date": "2026-04-30",
                    "april_buy_symbols": 1,
                    "april_buy_events": 2,
                    "profitable_today": 1,
                    "elite_qm_count": 1,
                    "outlier_pass_count": 1,
                },
                stock_stats=pd.DataFrame(
                    [
                        {
                            "exchange": "NSE",
                            "symbol": "AAA",
                            "name": "A Ltd",
                            "april_buy_count": 2,
                            "s2_april_buy_count": 1,
                            "s3_april_buy_count": 1,
                            "latest_april_buy_date": "2026-04-25",
                            "latest_april_buy_price": 100.0,
                            "latest_close": 120.0,
                            "latest_close_date": "2026-06-22",
                            "current_gain_pct": 20.0,
                            "momentum_12_1_pct": 35.0,
                            "positive_day_pct": 62.0,
                            "top_gap_share_pct": 18.0,
                            "qm_composite_score": 88.0,
                            "qm_quality_bucket": "Elite",
                            "qm_outlier_pass": True,
                        }
                    ]
                ),
                buy_events=pd.DataFrame(),
            )
            save_qm_quality_outputs(result, data_root / "qm_quality")
            loaded = load_qm_quality_outputs(data_root / "qm_quality")
            self.assertEqual(int(loaded.summary["april_buy_symbols"]), 1)

            with (
                patch.dict(os.environ, {"DATA_ROOT": temp_dir}),
                patch("stock_screener.web.main.get_data_root", return_value=data_root),
            ):
                client = TestClient(app)
                response = client.get("/qm-quality")

        self.assertEqual(response.status_code, 404)
        self.assertIn("QM Quality", response.text)
        self.assertIn("temporarily removed from the workspace", response.text)


if __name__ == "__main__":
    unittest.main()
