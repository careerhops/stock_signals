from __future__ import annotations

import unittest

import pandas as pd

from stock_screener.notifications.telegram import build_buy_signal_list_message, buy_signals_to_csv_bytes


class TelegramNotificationTests(unittest.TestCase):
    def test_buy_signal_list_message_includes_date_name_and_signal_price(self) -> None:
        signals = pd.DataFrame(
            [
                {
                    "date": "2026-04-17",
                    "exchange": "NSE",
                    "symbol": "E2E-BE",
                    "company_name": "E2E Networks Limited",
                    "signal": "BUY",
                    "close": 5239.75,
                    "has_large_deal": True,
                    "large_deal_summary": "1 BUY",
                }
            ]
        )

        message = build_buy_signal_list_message(signals)

        self.assertIn("2026-04-17", message)
        self.assertIn("NSE:E2E-BE", message)
        self.assertIn("E2E Networks Limited", message)
        self.assertIn("Close: 5239.75", message)
        self.assertIn("Large Deal: Yes (1 BUY)", message)

    def test_buy_signal_list_message_includes_selected_filters(self) -> None:
        signals = pd.DataFrame(
            [
                {
                    "date": "2026-04-17",
                    "exchange": "NSE",
                    "symbol": "TCS",
                    "company_name": "Tata Consultancy Services",
                    "signal": "BUY",
                    "close": 2524.30,
                }
            ]
        )

        message = build_buy_signal_list_message(
            signals,
            filters_text="Market cap bucket: Large Cap; Min market cap: 1000 Cr",
        )

        self.assertIn("Filters: Market cap bucket: Large Cap; Min market cap: 1000 Cr", message)

    def test_long_buy_signal_message_limits_inline_rows_and_exports_full_csv(self) -> None:
        signals = pd.DataFrame(
            [
                {
                    "date": f"2026-04-{day:02d}",
                    "exchange": "NSE",
                    "symbol": f"STOCK{day}",
                    "company_name": f"Stock {day} Limited",
                    "signal": "BUY",
                    "close": day * 10,
                }
                for day in range(1, 13)
            ]
        )

        message = build_buy_signal_list_message(signals, inline_limit=10)
        csv_text = buy_signals_to_csv_bytes(signals).decode("utf-8")

        self.assertIn("Showing top 10", message)
        self.assertIn("STOCK10", message)
        self.assertNotIn("STOCK11", message)
        self.assertIn("STOCK12", csv_text)
        self.assertIn("signal_close_price", csv_text)

    def test_buy_signal_csv_includes_large_deal_marker(self) -> None:
        signals = pd.DataFrame(
            [
                {
                    "date": "2026-04-17",
                    "exchange": "NSE",
                    "symbol": "RIIL",
                    "company_name": "Reliance Industrial Infrastructure",
                    "signal": "BUY",
                    "close": 798.90,
                    "has_large_deal": True,
                    "large_deal_summary": "2 BUY",
                    "large_deal_latest_date": "2026-04-17",
                }
            ]
        )

        csv_text = buy_signals_to_csv_bytes(signals).decode("utf-8")

        self.assertIn("recent_large_deal", csv_text)
        self.assertIn("large_deal_summary", csv_text)
        self.assertIn("Yes", csv_text)
        self.assertIn("2 BUY", csv_text)


if __name__ == "__main__":
    unittest.main()
