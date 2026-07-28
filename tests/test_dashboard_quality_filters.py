from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, MagicMock

import pandas as pd

from stock_screener.strategy.daily_confirmation import add_latest_daily_confirmation_columns
from stock_screener.strategy.weekly_shortlist import benchmark_symbol_for_industry
from stock_screener.data.storage import Storage
from stock_screener.web.main import (
    _apply_cmp_filters,
    _apply_signal_quality_filters,
    _apply_weekly_shortlist_filters,
    _enrich_with_latest_daily_close,
    _manual_screener_config,
    _refresh_live_cmp,
)


class DashboardQualityFilterTests(unittest.TestCase):
    def test_manual_screener_config_does_not_narrow_scan_universe_with_ui_filters(self) -> None:
        base_config = {
            "universe": {"filters": {"min_market_cap_cr": 100.0, "stock_search": "OLD"}},
            "filters": {"signal": {}},
            "strategy": {"sensitivity": 3},
            "notifications": {"enabled": True},
        }

        with TemporaryDirectory() as temp_dir:
            storage = Storage(Path(temp_dir))
            config = _manual_screener_config(
                base_config,
                storage,
                min_market_cap=1000.0,
                max_market_cap=5000.0,
                market_cap_bucket="Large Cap",
                stock_search="SUNPHARMA",
                sensitivity=5,
            )

        universe_filters = config["universe"]["filters"]
        self.assertIsNone(universe_filters["min_market_cap_cr"])
        self.assertIsNone(universe_filters["max_market_cap_cr"])
        self.assertIsNone(universe_filters["market_cap_bucket"])
        self.assertIsNone(universe_filters["stock_search"])
        self.assertEqual(config["strategy"]["sensitivity"], 5)
        self.assertTrue(config["filters"]["signal"]["latest_only"])
        self.assertEqual(config["filters"]["signal"]["direction"], "BUY")

    def test_cmp_filter_applies_min_and_max_to_selected_price_column(self) -> None:
        frame = pd.DataFrame(
            [
                {"symbol": "CHEAP", "close": 75.0, "latest_close": 75.0},
                {"symbol": "PASS", "close": 500.0, "latest_close": 500.0},
                {"symbol": "EXPENSIVE", "close": 1250.0, "latest_close": 1250.0},
            ]
        )

        screener_filtered = _apply_cmp_filters(frame, 100.0, 1000.0, "close")
        gtt_filtered = _apply_cmp_filters(frame, 100.0, 1000.0, "latest_close")

        self.assertEqual(screener_filtered["symbol"].tolist(), ["PASS"])
        self.assertEqual(gtt_filtered["symbol"].tolist(), ["PASS"])

    def test_latest_daily_confirmation_adds_latest_close_fields(self) -> None:
        daily = pd.DataFrame(
            [
                {"date": "2026-05-04", "close": 100.0, "volume": 1000},
                {"date": "2026-05-05", "close": 105.5, "volume": 1200},
            ]
        )
        frame = pd.DataFrame([{"symbol": "PASS", "signal": "BUY"}])

        enriched = add_latest_daily_confirmation_columns(frame, daily)

        self.assertEqual(float(enriched.iloc[0]["latest_close"]), 105.5)
        self.assertIn("2026-05-05", str(enriched.iloc[0]["latest_close_date"]))

    def test_enrich_with_latest_daily_close_backfills_from_scan_details(self) -> None:
        frame = pd.DataFrame([{"exchange": "NSE", "symbol": "PASS", "close": 100.0}])
        scan_details = pd.DataFrame(
            [
                {
                    "exchange": "NSE",
                    "symbol": "PASS",
                    "latest_close": 105.5,
                    "latest_close_date": "2026-05-05",
                }
            ]
        )

        enriched = _enrich_with_latest_daily_close(frame, scan_details)

        self.assertEqual(float(enriched.iloc[0]["latest_close"]), 105.5)
        self.assertEqual(str(enriched.iloc[0]["latest_close_date"]), "2026-05-05")

    def test_refresh_live_cmp_overrides_latest_close_from_live_quote(self) -> None:
        frame = pd.DataFrame(
            [
                {"exchange": "NSE", "symbol": "PASS", "latest_close": 105.5, "latest_close_date": "2026-05-05"},
            ]
        )

        with TemporaryDirectory() as temp_dir:
            with patch("stock_screener.web.main.load_access_token", return_value="token"), patch(
                "stock_screener.web.main.KiteDataProvider"
            ) as provider_cls:
                provider = MagicMock()
                provider.ltp.return_value = {"NSE:PASS": 111.25}
                provider_cls.return_value = provider

                refreshed = _refresh_live_cmp(frame, Path(temp_dir))

        self.assertEqual(float(refreshed.iloc[0]["latest_close"]), 111.25)
        self.assertEqual(str(refreshed.iloc[0]["cmp_source"]), "live")

    def test_quality_filters_require_volume_trend_and_return_threshold(self) -> None:
        signals = pd.DataFrame(
            [
                {
                    "symbol": "PASS",
                    "volume_confirmation": True,
                    "daily_ema_stack_confirmation": True,
                    "daily_obv_confirmation": True,
                    "median_pair_return_last_3_pct": 12.5,
                },
                {
                    "symbol": "LOWRET",
                    "volume_confirmation": True,
                    "daily_ema_stack_confirmation": True,
                    "daily_obv_confirmation": True,
                    "median_pair_return_last_3_pct": 2.0,
                },
                {
                    "symbol": "NOVOL",
                    "volume_confirmation": False,
                    "daily_ema_stack_confirmation": True,
                    "daily_obv_confirmation": True,
                    "median_pair_return_last_3_pct": 20.0,
                },
                {
                    "symbol": "NOTREND",
                    "volume_confirmation": True,
                    "daily_ema_stack_confirmation": False,
                    "daily_obv_confirmation": True,
                    "median_pair_return_last_3_pct": 20.0,
                },
                {
                    "symbol": "NOOBV",
                    "volume_confirmation": True,
                    "daily_ema_stack_confirmation": True,
                    "daily_obv_confirmation": False,
                    "median_pair_return_last_3_pct": 20.0,
                },
            ]
        )

        filtered = _apply_signal_quality_filters(
            signals,
            require_volume_confirmation=True,
            require_trend_confirmation=True,
            require_obv_confirmation=True,
            return_metric="median_3",
            min_pair_return=5.0,
        )

        self.assertEqual(filtered["symbol"].tolist(), ["PASS"])

    def test_quality_filters_can_use_last_completed_pair_return(self) -> None:
        signals = pd.DataFrame(
            [
                {
                    "symbol": "PASS",
                    "volume_confirmation": True,
                    "trend_confirmation": True,
                    "prior_pair_return_last_1_pct": 8.0,
                    "median_pair_return_last_3_pct": -2.0,
                },
            ]
        )

        filtered = _apply_signal_quality_filters(
            signals,
            require_volume_confirmation=False,
            require_trend_confirmation=False,
            require_obv_confirmation=False,
            return_metric="last_1",
            min_pair_return=5.0,
        )

        self.assertEqual(filtered["symbol"].tolist(), ["PASS"])

    def test_weekly_shortlist_filters_require_htf_rs_location_and_rr(self) -> None:
        signals = pd.DataFrame(
            [
                {
                    "symbol": "PASS",
                    "htf_alignment_confirmation": True,
                    "volume_confirmation_ratio": 1.8,
                    "relative_strength_12w_pct": 4.2,
                    "distance_from_demand_pct": 5.0,
                    "risk_reward_ratio": 2.6,
                },
                {
                    "symbol": "LOWVOL",
                    "htf_alignment_confirmation": True,
                    "volume_confirmation_ratio": 1.2,
                    "relative_strength_12w_pct": 5.0,
                    "distance_from_demand_pct": 4.0,
                    "risk_reward_ratio": 2.8,
                },
                {
                    "symbol": "LATE",
                    "htf_alignment_confirmation": True,
                    "volume_confirmation_ratio": 1.9,
                    "relative_strength_12w_pct": -1.0,
                    "distance_from_demand_pct": 14.0,
                    "risk_reward_ratio": 1.1,
                },
            ]
        )

        filtered = _apply_weekly_shortlist_filters(
            signals,
            require_htf_alignment=True,
            min_breakout_volume_ratio=1.5,
            require_relative_strength=True,
            min_relative_strength_pct=0.0,
            max_distance_from_demand_pct=8.0,
            min_risk_reward_ratio=2.0,
        )

        self.assertEqual(filtered["symbol"].tolist(), ["PASS"])

    def test_benchmark_symbol_for_industry_uses_sector_when_available(self) -> None:
        self.assertEqual(benchmark_symbol_for_industry("Information Technology"), "NIFTY IT")
        self.assertEqual(benchmark_symbol_for_industry("Financial Services"), "NIFTY FIN SERVICE")
        self.assertEqual(benchmark_symbol_for_industry("Unknown Theme"), "NIFTY 50")


if __name__ == "__main__":
    unittest.main()
