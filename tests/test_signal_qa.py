from __future__ import annotations

import unittest

import pandas as pd

from stock_screener.signal_qa import build_signal_quality_report, explain_strategy_row
from stock_screener.web.main import _selected_signal_qa_symbol, _signal_qa_candidates


class SignalQATests(unittest.TestCase):
    def test_explain_strategy_row_describes_buy_confluence(self) -> None:
        row = pd.Series(
            {
                "signal": "BUY",
                "close": 120,
                "upper_level": 115,
                "fvg_bull_recent": 2,
                "final_buy": True,
            }
        )

        explanation = explain_strategy_row(row)

        self.assertIn("BUY", explanation)
        self.assertIn("crossed above", explanation)
        self.assertIn("bullish FVG", explanation)

    def test_quality_report_flags_stale_filtered_buy(self) -> None:
        raw = pd.DataFrame(
            [
                {"exchange": "NSE", "symbol": "JIOFIN", "date": "2026-03-13", "signal": "BUY"},
                {"exchange": "NSE", "symbol": "JIOFIN", "date": "2026-04-10", "signal": "SELL"},
            ]
        )
        filtered = pd.DataFrame(
            [
                {"exchange": "NSE", "symbol": "JIOFIN", "date": "2026-03-13", "signal": "BUY"},
            ]
        )
        scan_details = pd.DataFrame(
            [
                {
                    "exchange": "NSE",
                    "symbol": "JIOFIN",
                    "latest_signal": "SELL",
                    "latest_signal_date": "2026-04-10",
                    "daily_rows": 100,
                }
            ]
        )

        report = build_signal_quality_report(raw, filtered, scan_details)

        self.assertGreater(report["summary"]["issue_count"], 0)
        self.assertTrue(
            any(
                issue["problem"].startswith("Filtered BUY does not match latest raw signal")
                for issue in report["issues"]
            )
        )

    def test_signal_qa_search_can_select_symbol_from_loaded_instruments(self) -> None:
        filtered = pd.DataFrame()
        scan_details = pd.DataFrame()
        instruments = pd.DataFrame(
            [
                {
                    "exchange": "NSE",
                    "tradingsymbol": "GLOBALE",
                    "name": "GLOBALE TESSILE",
                }
            ]
        )

        candidates = _signal_qa_candidates(filtered, scan_details, instruments, "GLOBALE")

        class RequestStub:
            query_params: dict[str, str] = {}

        selected_exchange, selected_symbol = _selected_signal_qa_symbol(
            RequestStub(),
            filtered,
            candidates,
            "GLOBALE",
        )

        self.assertEqual(candidates["symbol"].tolist(), ["GLOBALE"])
        self.assertEqual((selected_exchange, selected_symbol), ("NSE", "GLOBALE"))


if __name__ == "__main__":
    unittest.main()
