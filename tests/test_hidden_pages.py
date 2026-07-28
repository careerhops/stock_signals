from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

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
        ]:
            response = client.get(path)

            self.assertEqual(response.status_code, 404)
            self.assertIn(page_name, response.text)
            self.assertIn("temporarily removed from the workspace", response.text)

    def test_login_page_hides_shared_filter_panel(self) -> None:
        client = TestClient(app)

        response = client.get("/login")

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("Common Filters", response.text)
        self.assertNotIn("Active shared filters", response.text)


if __name__ == "__main__":
    unittest.main()
