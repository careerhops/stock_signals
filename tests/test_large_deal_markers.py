from __future__ import annotations

import unittest

import pandas as pd

from stock_screener.web.main import _apply_large_deal_markers


class LargeDealMarkerTests(unittest.TestCase):
    def test_marks_filtered_stocks_with_recent_large_deals(self) -> None:
        filtered = pd.DataFrame(
            [
                {"exchange": "NSE", "symbol": "RIIL", "name": "Reliance Industrial Infrastructure"},
                {"exchange": "NSE", "symbol": "HEG", "name": "HEG"},
            ]
        )
        deals = pd.DataFrame(
            [
                {"deal_date": "2026-04-17", "symbol": "RIIL", "action": "BUY"},
                {"deal_date": "2026-04-18", "symbol": "RIIL", "action": "SELL"},
            ]
        )

        marked = _apply_large_deal_markers(filtered, deals)

        riil = marked[marked["symbol"] == "RIIL"].iloc[0]
        heg = marked[marked["symbol"] == "HEG"].iloc[0]
        self.assertTrue(bool(riil["has_large_deal"]))
        self.assertEqual(riil["large_deal_summary"], "1 BUY, 1 SELL")
        self.assertEqual(riil["large_deal_latest_date"], "2026-04-18")
        self.assertFalse(bool(heg["has_large_deal"]))

    def test_large_deal_marker_uses_base_symbol_for_nse_series_suffixes(self) -> None:
        filtered = pd.DataFrame([{"exchange": "NSE", "symbol": "E2E-BE", "name": "E2E Networks"}])
        deals = pd.DataFrame([{"deal_date": "2026-04-17", "symbol": "E2E", "action": "BUY"}])

        marked = _apply_large_deal_markers(filtered, deals)

        self.assertTrue(bool(marked.iloc[0]["has_large_deal"]))
        self.assertEqual(marked.iloc[0]["large_deal_summary"], "1 BUY")


if __name__ == "__main__":
    unittest.main()
