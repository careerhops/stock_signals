from __future__ import annotations

import unittest

import pandas as pd

from stock_screener.filters import apply_filters


class SignalFilterTests(unittest.TestCase):
    def test_latest_only_uses_latest_overall_signal_per_stock_before_buy_filter(self) -> None:
        raw_signals = pd.DataFrame(
            [
                {"date": "2025-04-18", "exchange": "NSE", "symbol": "JIOFIN", "signal": "BUY", "close": 246.47},
                {"date": "2025-07-25", "exchange": "NSE", "symbol": "JIOFIN", "signal": "SELL", "close": 311.25},
                {"date": "2026-04-10", "exchange": "NSE", "symbol": "TCS", "signal": "BUY", "close": 2524.30},
            ]
        )
        config = {
            "filters": {
                "enabled": True,
                "signal": {
                    "direction": "BUY",
                    "latest_only": True,
                },
            }
        }

        filtered = apply_filters(raw_signals, config)

        self.assertEqual(filtered["symbol"].tolist(), ["TCS"])
        self.assertNotIn("JIOFIN", set(filtered["symbol"]))

    def test_latest_buy_must_also_be_inside_configured_signal_age(self) -> None:
        raw_signals = pd.DataFrame(
            [
                {"date": "2026-01-09", "exchange": "NSE", "symbol": "OLDWIN", "signal": "BUY", "close": 100.0},
                {"date": "2026-03-13", "exchange": "NSE", "symbol": "JIOFIN", "signal": "BUY", "close": 246.47},
                {"date": "2026-04-10", "exchange": "NSE", "symbol": "JIOFIN", "signal": "SELL", "close": 311.25},
                {"date": "2026-04-10", "exchange": "NSE", "symbol": "FRESHBUY", "signal": "BUY", "close": 2524.30},
            ]
        )
        config = {
            "data": {
                "scan_timeframe": "1W",
            },
            "filters": {
                "enabled": True,
                "signal": {
                    "direction": "BUY",
                    "latest_only": True,
                    "max_signal_age_bars": 1,
                },
            },
        }

        filtered = apply_filters(raw_signals, config)

        self.assertEqual(filtered["symbol"].tolist(), ["FRESHBUY"])
        self.assertNotIn("OLDWIN", set(filtered["symbol"]))
        self.assertNotIn("JIOFIN", set(filtered["symbol"]))


if __name__ == "__main__":
    unittest.main()
