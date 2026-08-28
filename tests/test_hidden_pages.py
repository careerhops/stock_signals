from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from stock_screener.web import main
from stock_screener.web.main import app


class HiddenPagesTests(unittest.TestCase):
    def test_temporarily_removed_pages_render_placeholder(self) -> None:
        client = TestClient(app)

        for path, page_name in [
            ("/stocks", "Stocks"),
            ("/signal-outcome-study", "Signal Outcome"),
            ("/signal-qa", "Signal QA"),
            ("/backtest", "Backtest"),
            ("/rotation-study", "Rotation Study"),
            ("/sensitivity-study", "Sensitivity Study"),
            ("/weekly-buy-tracker", "Buy Tracker"),
            ("/strategy-lab", "Strategy Lab"),
            ("/weekday-study", "Weekday Study"),
            ("/weekly-buy-gains", "Buy Gains"),
            ("/volume-burst", "Volume Burst"),
            ("/resistance-breaks", "Resistance Breaks"),
            ("/minervini-sheet", "Minervini Sheet"),
            ("/qm-quality", "QM Quality"),
        ]:
            response = client.get(path)

            self.assertEqual(response.status_code, 404)
            self.assertIn(page_name, response.text)
            self.assertIn("temporarily removed from the workspace", response.text)

    def test_removed_study_run_routes_cannot_start_jobs(self) -> None:
        client = TestClient(app)

        for path in [
            "/weekday-study/run",
            "/weekly-buy-gains/run",
            "/volume-burst/run",
            "/resistance-breaks/run",
            "/minervini-sheet/run",
            "/minervini-sheet/google/save",
            "/qm-quality/run",
        ]:
            response = client.post(path)

            self.assertEqual(response.status_code, 404)
            self.assertIn("temporarily removed from the workspace", response.text)

    def test_primary_navigation_hides_removed_studies(self) -> None:
        client = TestClient(app)

        response = client.get("/")

        self.assertEqual(response.status_code, 200)
        for label in [
            "Weekday Study",
            "Buy Gains",
            "Volume Burst",
            "Resistance Breaks",
            "Minervini Sheet",
            "QM Quality",
        ]:
            self.assertNotIn(f">{label}</a>", response.text)

    def test_main_screener_does_not_run_retired_follow_up_studies(self) -> None:
        with (
            patch.object(main, "run_daily_scan", return_value={"symbols_scanned": 12}),
            patch.object(main, "_set_scan_job"),
            patch.object(main, "_build_latest_weekday_pressure_cache") as weekday_study,
            patch.object(main, "_maybe_run_minervini_sheet_sync_after_screener") as minervini_sheet,
        ):
            main._run_screener_job("job-1", {}, "")

        weekday_study.assert_not_called()
        minervini_sheet.assert_not_called()

    def test_login_page_hides_shared_filter_panel(self) -> None:
        client = TestClient(app)

        response = client.get("/login")

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("Common Filters", response.text)
        self.assertNotIn("Active shared filters", response.text)


if __name__ == "__main__":
    unittest.main()
