from __future__ import annotations

import unittest

import pandas as pd

from stock_screener.notifications.telegram import (
    build_buy_signal_list_message,
    build_gtt_stock_list_message,
    buy_signals_to_csv_bytes,
    buy_signals_to_html_bytes,
    gtt_stock_list_to_csv_bytes,
    gtt_stock_list_to_html_bytes,
)


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

    def test_buy_signal_html_report_is_readable_and_interactive(self) -> None:
        signals = pd.DataFrame(
            [
                {
                    "date": "2026-04-17",
                    "exchange": "NSE",
                    "symbol": "NETWEB",
                    "company_name": "Netweb Technologies India",
                    "close": 2418.25,
                    "daily_ema_stack_confirmation": True,
                    "daily_obv_confirmation": True,
                    "volume_confirmation": True,
                    "has_large_deal": True,
                    "large_deal_summary": "1 BUY",
                }
            ]
        )

        html = buy_signals_to_html_bytes(signals, filters_text="Daily EMA stack; OBV rising").decode("utf-8")

        self.assertIn("Weekly BUY Signal Report", html)
        self.assertIn("NETWEB", html)
        self.assertIn("Netweb Technologies India", html)
        self.assertIn("Copy TradingView symbols", html)
        self.assertIn("NSE:NETWEB", html)
        self.assertIn("Daily EMA stack; OBV rising", html)

    def test_gtt_html_report_contains_gtt_metrics(self) -> None:
        stocks = pd.DataFrame(
            [
                {
                    "exchange": "NSE",
                    "symbol": "MASFIN",
                    "company_name": "MAS Financial Services",
                    "latest_close": 312.5,
                    "weekly_technical_rating_status": "Strong Buy",
                    "daily_ema_stack_confirmation": True,
                    "daily_obv_confirmation": True,
                    "valid_pairs": 4,
                    "median_max_gain_pct": 18.5,
                    "hit_10pct_rate_pct": 75.0,
                    "suggested_conservative_gtt_pct": 10.0,
                    "low_sample": False,
                }
            ]
        )

        html = gtt_stock_list_to_html_bytes(stocks, filters_text="Fresh weekly BUY").decode("utf-8")

        self.assertIn("GTT Filtered Stock Report", html)
        self.assertIn("MASFIN", html)
        self.assertIn("Strong Buy", html)
        self.assertIn("Median Gain", html)
        self.assertIn("NSE:MASFIN", html)
        self.assertIn("High Samples Yes", html)

    def test_gtt_telegram_message_sorts_by_median_return_ascending(self) -> None:
        stocks = pd.DataFrame(
            [
                {
                    "exchange": "NSE",
                    "symbol": "HIGH",
                    "company_name": "Higher Median Return",
                    "median_pair_return_last_3_pct": 12.0,
                },
                {
                    "exchange": "NSE",
                    "symbol": "LOW",
                    "company_name": "Lower Median Return",
                    "median_pair_return_last_3_pct": -3.0,
                },
                {
                    "exchange": "NSE",
                    "symbol": "MID",
                    "company_name": "Middle Median Return",
                    "median_pair_return_last_3_pct": 4.0,
                },
            ]
        )

        message = build_gtt_stock_list_message(stocks)

        self.assertLess(message.index("NSE:LOW"), message.index("NSE:MID"))
        self.assertLess(message.index("NSE:MID"), message.index("NSE:HIGH"))

    def test_gtt_csv_uses_high_samples_label(self) -> None:
        stocks = pd.DataFrame(
            [
                {
                    "exchange": "NSE",
                    "symbol": "LOWSAMPLE",
                    "company_name": "Low Sample Co",
                    "low_sample": True,
                },
                {
                    "exchange": "NSE",
                    "symbol": "HIGHSAMPLE",
                    "company_name": "High Sample Co",
                    "low_sample": False,
                },
            ]
        )

        csv_text = gtt_stock_list_to_csv_bytes(stocks).decode("utf-8")

        self.assertIn("high_samples", csv_text)
        self.assertIn("LOWSAMPLE,Low Sample Co", csv_text)
        self.assertIn("No", csv_text)
        self.assertIn("Yes", csv_text)

    def test_html_report_caps_at_30_stocks(self) -> None:
        signals = pd.DataFrame(
            [
                {
                    "date": f"2026-04-{(day % 28) + 1:02d}",
                    "exchange": "NSE",
                    "symbol": f"STOCK{day}",
                    "company_name": f"Stock {day} Limited",
                    "close": 100 + day,
                }
                for day in range(1, 36)
            ]
        )

        html = buy_signals_to_html_bytes(signals).decode("utf-8")

        self.assertIn("Showing top 30 stocks in this HTML report.", html)
        self.assertIn("<span>Total stocks</span><strong>30</strong>", html)
        self.assertEqual(html.count('<article class="card"'), 30)


if __name__ == "__main__":
    unittest.main()
