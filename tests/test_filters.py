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


if __name__ == "__main__":
    unittest.main()
