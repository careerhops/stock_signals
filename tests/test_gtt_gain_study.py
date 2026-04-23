from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd
from fastapi.testclient import TestClient

from stock_screener.data.storage import Storage
from stock_screener.gtt_gain_study import (
    build_stock_gtt_stats,
    build_symbol_gtt_pairs,
    load_gtt_gain_outputs,
    max_gain_between_dates,
    run_gtt_gain_study,
)
from stock_screener.web.charts import build_gtt_opportunity_chart
from stock_screener.web.main import (
    _align_gtt_stock_stats_to_latest_universe,
    _apply_gtt_stock_filters,
    _ensure_gtt_weekly_technical_ratings,
    _apply_peak_speed_bucket_filter,
    _build_gtt_universe_audit,
    _gtt_display_summary,
    app,
)


class GttGainStudyTests(unittest.TestCase):
    def test_pair_max_gain_uses_daily_high_between_buy_and_sell_only(self) -> None:
        buy_date = pd.Timestamp("2025-04-21")
        sell_date = pd.Timestamp("2025-08-25")
        daily = pd.DataFrame(
            [
                {"date": buy_date, "open": 800.0, "high": 1400.0, "low": 790.0, "close": 824.0, "volume": 1000},
                {"date": "2025-05-02", "open": 830.0, "high": 950.0, "low": 820.0, "close": 900.0, "volume": 1000},
                {"date": "2025-06-10", "open": 950.0, "high": 1022.0, "low": 930.0, "close": 990.0, "volume": 1000},
                {"date": sell_date, "open": 890.0, "high": 1500.0, "low": 880.0, "close": 890.0, "volume": 1000},
            ]
        )

        result = max_gain_between_dates(daily, buy_date, sell_date, 824.0)

        self.assertTrue(result["valid_daily_window"])
        self.assertEqual(result["highest_price_between_buy_sell"], 1022.0)
        self.assertEqual(pd.Timestamp(result["highest_price_date"]), pd.Timestamp("2025-06-10"))
        self.assertEqual(result["days_to_peak_from_buy"], 50)
        self.assertAlmostEqual(result["max_gain_pct"], 24.029126213592235)

    def test_sell_close_does_not_control_interim_max_gain(self) -> None:
        strategy_output = pd.DataFrame(
            [
                {"date": "2025-04-21", "close": 824.0, "final_buy": True, "final_sell": False},
                {"date": "2025-08-25", "close": 890.0, "final_buy": False, "final_sell": True},
            ]
        )
        daily = pd.DataFrame(
            [
                {"date": "2025-04-21", "open": 800.0, "high": 824.0, "low": 790.0, "close": 824.0, "volume": 1000},
                {"date": "2025-06-10", "open": 950.0, "high": 1022.0, "low": 930.0, "close": 990.0, "volume": 1000},
                {"date": "2025-08-25", "open": 890.0, "high": 900.0, "low": 880.0, "close": 890.0, "volume": 1000},
            ]
        )

        pairs, open_position = build_symbol_gtt_pairs(daily, strategy_output, "NSE", "NATCOPHARM", "NATCO PHARMA")

        self.assertIsNone(open_position)
        self.assertEqual(len(pairs), 1)
        self.assertAlmostEqual(pairs.iloc[0]["buy_to_sell_return_pct"], 8.009708737864077)
        self.assertAlmostEqual(pairs.iloc[0]["max_gain_pct"], 24.029126213592235)
        self.assertEqual(pairs.iloc[0]["days_to_peak_from_buy"], 50)
        self.assertTrue(bool(pairs.iloc[0]["hit_20pct"]))

    def test_pair_without_daily_window_is_kept_but_excluded_from_stock_median(self) -> None:
        strategy_output = pd.DataFrame(
            [
                {"date": "2025-04-21", "close": 824.0, "final_buy": True, "final_sell": False},
                {"date": "2025-04-28", "close": 830.0, "final_buy": False, "final_sell": True},
            ]
        )
        daily = pd.DataFrame(
            [
                {"date": "2025-04-21", "open": 800.0, "high": 900.0, "low": 790.0, "close": 824.0, "volume": 1000},
                {"date": "2025-04-28", "open": 830.0, "high": 1000.0, "low": 820.0, "close": 830.0, "volume": 1000},
            ]
        )

        pairs, _ = build_symbol_gtt_pairs(daily, strategy_output, "NSE", "TEST", "Test Ltd")
        stats = build_stock_gtt_stats(pairs)

        self.assertEqual(len(pairs), 1)
        self.assertFalse(bool(pairs.iloc[0]["valid_daily_window"]))
        self.assertTrue(pd.isna(pairs.iloc[0]["days_to_peak_from_buy"]))
        self.assertEqual(stats.iloc[0]["closed_pairs"], 1)
        self.assertEqual(stats.iloc[0]["valid_pairs"], 0)
        self.assertTrue(pd.isna(stats.iloc[0]["median_max_gain_pct"]))

    def test_stock_aggregation_counts_threshold_rates_and_low_sample(self) -> None:
        pairs = pd.DataFrame(
            [
                {
                    "exchange": "NSE",
                    "symbol": "AAA",
                    "name": "A Ltd",
                    "valid_daily_window": True,
                    "max_gain_pct": 8.0,
                    "days_to_peak_from_buy": 10,
                },
                {
                    "exchange": "NSE",
                    "symbol": "AAA",
                    "name": "A Ltd",
                    "valid_daily_window": True,
                    "max_gain_pct": 12.0,
                    "days_to_peak_from_buy": 20,
                },
                {
                    "exchange": "NSE",
                    "symbol": "AAA",
                    "name": "A Ltd",
                    "valid_daily_window": True,
                    "max_gain_pct": 30.0,
                    "days_to_peak_from_buy": 40,
                },
                {
                    "exchange": "NSE",
                    "symbol": "BBB",
                    "name": "B Ltd",
                    "valid_daily_window": True,
                    "max_gain_pct": 20.0,
                    "days_to_peak_from_buy": 15,
                },
            ]
        )

        stats = build_stock_gtt_stats(pairs)
        aaa = stats[stats["symbol"] == "AAA"].iloc[0]
        bbb = stats[stats["symbol"] == "BBB"].iloc[0]

        self.assertEqual(aaa["valid_pairs"], 3)
        self.assertAlmostEqual(aaa["median_max_gain_pct"], 12.0)
        self.assertAlmostEqual(aaa["median_days_to_peak"], 20.0)
        self.assertAlmostEqual(aaa["avg_days_to_peak"], 23.333333333333332)
        self.assertEqual(aaa["peak_speed_bucket"], "Within 30 days")
        self.assertAlmostEqual(aaa["hit_10pct_rate_pct"], 66.66666666666666)
        self.assertFalse(bool(aaa["low_sample"]))
        self.assertTrue(bool(bbb["low_sample"]))
        self.assertAlmostEqual(aaa["suggested_conservative_gtt_pct"], 10.0)
        self.assertAlmostEqual(aaa["suggested_moderate_gtt_pct"], 20.0)
        self.assertAlmostEqual(bbb["suggested_conservative_gtt_pct"], 10.0)
        self.assertAlmostEqual(bbb["suggested_moderate_gtt_pct"], 20.0)

    def test_stock_aggregation_backfills_days_to_peak_from_saved_dates(self) -> None:
        pairs = pd.DataFrame(
            [
                {
                    "exchange": "NSE",
                    "symbol": "AAA",
                    "name": "A Ltd",
                    "buy_date": "2025-01-01",
                    "highest_price_date": "2025-01-31",
                    "valid_daily_window": True,
                    "max_gain_pct": 12.0,
                },
                {
                    "exchange": "NSE",
                    "symbol": "AAA",
                    "name": "A Ltd",
                    "buy_date": "2025-04-01",
                    "highest_price_date": "2025-05-31",
                    "valid_daily_window": True,
                    "max_gain_pct": 18.0,
                },
            ]
        )

        stats = build_stock_gtt_stats(pairs)

        self.assertAlmostEqual(stats.iloc[0]["median_days_to_peak"], 45.0)
        self.assertAlmostEqual(stats.iloc[0]["avg_days_to_peak"], 45.0)
        self.assertEqual(stats.iloc[0]["peak_speed_bucket"], "31-60 days")

    def test_load_gtt_outputs_rebuilds_days_to_peak_for_older_saved_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            pd.DataFrame([{"symbols_processed": 1}]).to_csv(output_dir / "latest_summary.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "exchange": "NSE",
                        "symbol": "AAA",
                        "name": "A Ltd",
                        "latest_week_signal": "BUY",
                        "latest_signal": "BUY",
                        "is_latest_signal_buy": True,
                    }
                ]
            ).to_csv(output_dir / "latest_stock_gtt_stats.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "exchange": "NSE",
                        "symbol": "AAA",
                        "name": "A Ltd",
                        "buy_date": "2025-01-01",
                        "buy_close": 100.0,
                        "sell_date": "2025-03-01",
                        "sell_close": 112.0,
                        "buy_to_sell_return_pct": 12.0,
                        "valid_daily_window": True,
                        "highest_price_between_buy_sell": 130.0,
                        "highest_price_date": "2025-01-31",
                        "max_gain_pct": 30.0,
                    }
                ]
            ).to_csv(output_dir / "latest_pair_details.csv", index=False)
            pd.DataFrame().to_csv(output_dir / "latest_open_positions.csv", index=False)

            result = load_gtt_gain_outputs(output_dir)

        self.assertEqual(result.pair_details.iloc[0]["days_to_peak_from_buy"], 30)
        self.assertEqual(result.stock_stats.iloc[0]["median_days_to_peak"], 30)
        self.assertEqual(result.stock_stats.iloc[0]["peak_speed_bucket"], "Within 30 days")
        self.assertEqual(result.stock_stats.iloc[0]["latest_signal"], "BUY")

    def test_stock_aggregation_adds_latest_signal_and_trend_context(self) -> None:
        pairs = pd.DataFrame(
            [
                {"exchange": "NSE", "symbol": "AAA", "name": "A Ltd", "valid_daily_window": True, "max_gain_pct": 12.0},
            ]
        )
        latest_context = pd.DataFrame(
            [
                {
                    "exchange": "NSE",
                    "symbol": "AAA",
                    "name": "A Ltd",
                    "latest_week_date": pd.Timestamp("2026-04-17"),
                    "latest_close": 120.0,
                    "ema_20": 110.0,
                    "ema_50": 100.0,
                    "close_above_ema20": True,
                    "ema20_above_ema50": True,
                    "trend_confirmation": True,
                    "volume_confirmation": True,
                    "volume_confirmation_ratio": 1.8,
                    "latest_week_signal": "BUY",
                    "latest_signal": "BUY",
                    "latest_signal_date": pd.Timestamp("2026-04-17"),
                    "is_latest_signal_buy": True,
                    "weekly_technical_rating": 0.72,
                    "weekly_technical_rating_status": "Strong Buy",
                    "weekly_ma_rating": 0.86,
                    "weekly_oscillator_rating": 0.57,
                }
            ]
        )

        stats = build_stock_gtt_stats(pairs, latest_context)

        self.assertEqual(stats.iloc[0]["latest_signal"], "BUY")
        self.assertTrue(bool(stats.iloc[0]["is_latest_signal_buy"]))
        self.assertTrue(bool(stats.iloc[0]["close_above_ema20"]))
        self.assertTrue(bool(stats.iloc[0]["ema20_above_ema50"]))
        self.assertTrue(bool(stats.iloc[0]["volume_confirmation"]))
        self.assertAlmostEqual(stats.iloc[0]["volume_confirmation_ratio"], 1.8)
        self.assertAlmostEqual(stats.iloc[0]["weekly_technical_rating"], 0.72)
        self.assertEqual(stats.iloc[0]["weekly_technical_rating_status"], "Strong Buy")

    def test_gtt_gain_study_page_loads(self) -> None:
        client = TestClient(app)
        response = client.get("/gtt-gain-study")

        self.assertEqual(response.status_code, 200)
        self.assertIn("GTT Gain Study", response.text)
        self.assertIn("Peak Speed Bucket", response.text)
        self.assertIn("Days To Peak", response.text)
        self.assertIn("Volume confirmed", response.text)
        self.assertIn("Fresh daily BUY only", response.text)
        self.assertIn("Weekly technical rating", response.text)
        self.assertIn("Apply Fresh weekly BUY only", response.text)

    def test_gtt_gain_study_page_shows_selected_filters(self) -> None:
        client = TestClient(app)
        response = client.get(
            "/gtt-gain-study?open_buy_regime_only=1&dashboard_buy_only=1&fresh_weekly_buy_only=1&fresh_daily_buy_only=1&trend_only=1&technical_rating_status=Strong%20Buy"
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Filter: open BUY regime", response.text)
        self.assertIn("Filter: dashboard BUY signals only", response.text)
        self.assertIn("Filter: fresh weekly BUY only", response.text)
        self.assertIn("Filter: fresh daily BUY only", response.text)
        self.assertIn("Filter: weekly technical rating is Strong Buy", response.text)
        self.assertIn("20W EMA", response.text)
        self.assertIn("GTT Peak Speed Buckets", response.text)
        self.assertNotIn("Apply Fresh weekly BUY only", response.text)

    def test_gtt_opportunity_chart_uses_filtered_buy_trend_rows(self) -> None:
        chart_html = build_gtt_opportunity_chart(
            pd.DataFrame(
                [
                    {
                        "symbol": "AAA",
                        "name": "A Ltd",
                        "valid_pairs": 5,
                        "hit_10pct_rate_pct": 80.0,
                        "median_max_gain_pct": 18.0,
                        "avg_max_gain_pct": 21.0,
                        "best_max_gain_pct": 34.0,
                        "median_days_to_peak": 45,
                        "suggested_conservative_gtt_pct": 10.0,
                        "suggested_moderate_gtt_pct": 18.0,
                    }
                ]
            )
        )

        self.assertIn("GTT Peak Speed Buckets", chart_html)
        self.assertIn("AAA", chart_html)
        self.assertIn("31-60 days", chart_html)
        self.assertIn("Stocks in bucket", chart_html)

    def test_peak_speed_bucket_filter_limits_gtt_stock_table(self) -> None:
        frame = pd.DataFrame(
            [
                {"symbol": "AAA", "peak_speed_bucket": "Within 30 days", "median_max_gain_pct": 12.0},
                {"symbol": "BBB", "peak_speed_bucket": "31-60 days", "median_max_gain_pct": 18.0},
                {"symbol": "CCC", "peak_speed_bucket": "31-60 days", "median_max_gain_pct": 22.0},
            ]
        )

        filtered = _apply_peak_speed_bucket_filter(frame, "31-60 days")

        self.assertEqual(set(filtered["symbol"]), {"BBB", "CCC"})
        self.assertEqual(list(filtered["median_max_gain_pct"]), [18.0, 22.0])

    def test_gtt_stock_filters_have_distinct_buy_semantics(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "symbol": "AAA",
                    "is_latest_signal_buy": True,
                    "latest_week_signal": "NONE",
                    "close_above_ema20": True,
                    "ema20_above_ema50": True,
                    "volume_confirmation": True,
                    "weekly_technical_rating_status": "Strong Buy",
                },
                {
                    "symbol": "BBB",
                    "is_latest_signal_buy": False,
                    "latest_week_signal": "BUY",
                    "close_above_ema20": True,
                    "ema20_above_ema50": True,
                    "volume_confirmation": False,
                    "weekly_technical_rating_status": "Buy",
                },
                {
                    "symbol": "CCC",
                    "is_latest_signal_buy": True,
                    "latest_week_signal": "BUY",
                    "close_above_ema20": False,
                    "ema20_above_ema50": True,
                    "volume_confirmation": True,
                    "weekly_technical_rating_status": "Sell",
                },
            ]
        )

        open_regime = _apply_gtt_stock_filters(frame, open_buy_regime_only=True, trend_only=False)
        fresh_buy = _apply_gtt_stock_filters(frame, open_buy_regime_only=False, trend_only=False, fresh_weekly_buy_only=True)
        dashboard_buy = _apply_gtt_stock_filters(
            frame,
            open_buy_regime_only=False,
            trend_only=False,
            dashboard_buy_only=True,
            dashboard_buy_symbols={"BBB"},
        )
        volume_confirmed = _apply_gtt_stock_filters(
            frame,
            open_buy_regime_only=False,
            trend_only=False,
            require_volume_confirmation=True,
        )
        fresh_daily_buy = _apply_gtt_stock_filters(
            frame,
            open_buy_regime_only=False,
            trend_only=False,
            fresh_daily_buy_only=True,
            fresh_daily_buy_symbols={"AAA", "BBB"},
        )
        buy_rated = _apply_gtt_stock_filters(
            frame,
            open_buy_regime_only=False,
            trend_only=False,
            technical_rating_status="Buy",
        )

        self.assertEqual(set(open_regime["symbol"]), {"AAA", "CCC"})
        self.assertEqual(set(fresh_buy["symbol"]), {"BBB", "CCC"})
        self.assertEqual(set(dashboard_buy["symbol"]), {"BBB"})
        self.assertEqual(set(volume_confirmed["symbol"]), {"AAA", "CCC"})
        self.assertEqual(set(fresh_daily_buy["symbol"]), {"AAA", "BBB"})
        self.assertEqual(set(buy_rated["symbol"]), {"BBB"})

    def test_gtt_universe_audit_uses_kite_instruments_as_single_universe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            storage = Storage(data_root)
            signals_dir = data_root / "signals"
            candles_dir = data_root / "candles" / "NSE" / "1D"
            signals_dir.mkdir(parents=True, exist_ok=True)
            candles_dir.mkdir(parents=True)
            storage.save_instruments(
                pd.DataFrame(
                    [
                        {"exchange": "NSE", "tradingsymbol": "AAA", "name": "A Ltd", "instrument_type": "EQ", "segment": "NSE"},
                        {"exchange": "NSE", "tradingsymbol": "BBB", "name": "B Ltd", "instrument_type": "EQ", "segment": "NSE"},
                        {"exchange": "NSE", "tradingsymbol": "CCC", "name": "C Ltd", "instrument_type": "EQ", "segment": "NSE"},
                        {"exchange": "NSE", "tradingsymbol": "NONAME", "name": "", "instrument_type": "EQ", "segment": "NSE"},
                    ]
                )
            )

            pd.DataFrame(
                [
                    {"symbol": "AAA", "signal": "BUY"},
                    {"symbol": "BBB", "signal": "SELL"},
                ]
            ).to_csv(signals_dir / "latest_filtered.csv", index=False)
            pd.DataFrame(
                [
                    {"symbol": "AAA"},
                    {"symbol": "BBB"},
                    {"symbol": "CCC"},
                ]
            ).to_csv(signals_dir / "latest_scan_details.csv", index=False)
            for symbol in ("AAA", "ZZZ"):
                (candles_dir / f"{symbol}.csv").write_text("date,open,high,low,close,volume\n", encoding="utf-8")

            stock_stats = pd.DataFrame(
                [
                    {"symbol": "AAA", "is_latest_signal_buy": True, "latest_week_signal": "BUY"},
                    {"symbol": "ZZZ", "is_latest_signal_buy": True, "latest_week_signal": "NONE"},
                ]
            )

            audit = _build_gtt_universe_audit(data_root, stock_stats, {})

        self.assertEqual(audit["dashboard_scanned_symbols"], 3)
        self.assertEqual(audit["home_filtered_buy_symbols"], 1)
        self.assertEqual(audit["latest_nse_universe_symbols"], 4)
        self.assertEqual(audit["gtt_rows_in_latest_universe"], 1)
        self.assertEqual(audit["gtt_open_buy_regime_symbols"], 2)
        self.assertEqual(audit["gtt_fresh_weekly_buy_symbols"], 1)
        self.assertEqual(audit["excluded_cached_sample"], ["ZZZ"])
        self.assertEqual(audit["missing_gtt_rows_sample"], ["BBB", "CCC", "NONAME"])

    def test_gtt_page_alignment_adds_placeholder_rows_for_kite_symbols(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            storage = Storage(data_root)
            storage.save_instruments(
                pd.DataFrame(
                    [
                        {"exchange": "NSE", "tradingsymbol": "AAA", "name": "A Ltd", "instrument_type": "EQ", "segment": "NSE"},
                        {"exchange": "NSE", "tradingsymbol": "BBB", "name": "B Ltd", "instrument_type": "EQ", "segment": "NSE"},
                    ]
                )
            )

            stock_stats = pd.DataFrame(
                [
                    {"exchange": "NSE", "symbol": "AAA", "name": "A Ltd", "is_latest_signal_buy": True},
                    {"exchange": "NSE", "symbol": "ZZZ", "name": "Stale Ltd", "is_latest_signal_buy": True},
                ]
            )

            aligned = _align_gtt_stock_stats_to_latest_universe(data_root, stock_stats, {})

        self.assertEqual(set(aligned["symbol"]), {"AAA", "BBB"})
        self.assertNotIn("ZZZ", set(aligned["symbol"]))
        self.assertEqual(len(aligned), 2)
        bbb = aligned[aligned["symbol"] == "BBB"].iloc[0]
        self.assertEqual(bbb["latest_signal"], "NONE")
        self.assertEqual(bbb["closed_pairs"], 0)

    def test_gtt_weekly_technical_ratings_backfill_from_cached_candles(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            storage = Storage(data_root)
            storage.save_instruments(
                pd.DataFrame(
                    [
                        {"exchange": "NSE", "tradingsymbol": "AAA", "name": "A Ltd", "instrument_type": "EQ", "segment": "NSE"},
                    ]
                )
            )
            candles = pd.DataFrame(
                [
                    {
                        "date": (pd.Timestamp("2024-01-01") + pd.Timedelta(days=index)).strftime("%Y-%m-%d"),
                        "open": 100 + (index * 1.1),
                        "high": 101 + (index * 1.15),
                        "low": 99 + (index * 1.05),
                        "close": 100 + (index * 1.12),
                        "volume": 100000 + (index * 100),
                    }
                    for index in range(320)
                ]
            )
            storage.save_candles("NSE", "AAA", candles)

            stock_stats = pd.DataFrame(
                [
                    {
                        "exchange": "NSE",
                        "symbol": "AAA",
                        "name": "A Ltd",
                        "weekly_technical_rating": pd.NA,
                        "weekly_technical_rating_status": pd.NA,
                    }
                ]
            )

            enriched = _ensure_gtt_weekly_technical_ratings(data_root, stock_stats, {"strategy": {"weekly_anchor": "W-FRI", "use_completed_weeks_only": True}})

        self.assertTrue(pd.notna(enriched.iloc[0]["weekly_technical_rating"]))
        self.assertIn(enriched.iloc[0]["weekly_technical_rating_status"], {"Strong Buy", "Buy"})

    def test_gtt_study_processes_kite_instruments_not_stale_cached_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            storage = Storage(data_root)
            storage.save_instruments(
                pd.DataFrame(
                    [
                        {
                            "exchange": "NSE",
                            "tradingsymbol": "AAA",
                            "name": "A Ltd",
                            "instrument_type": "EQ",
                            "segment": "NSE",
                        },
                        {
                            "exchange": "NSE",
                            "tradingsymbol": "BBB",
                            "name": "B Ltd",
                            "instrument_type": "EQ",
                            "segment": "NSE",
                        },
                        {
                            "exchange": "NSE",
                            "tradingsymbol": "MISSINGNAME",
                            "name": "",
                            "instrument_type": "EQ",
                            "segment": "NSE",
                        },
                    ]
                ),
            )
            candles = pd.DataFrame(
                [
                    {"date": "2025-01-03", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 100},
                    {"date": "2025-01-10", "open": 10, "high": 12, "low": 9, "close": 11, "volume": 100},
                    {"date": "2025-01-17", "open": 11, "high": 13, "low": 10, "close": 12, "volume": 100},
                ]
            )
            storage.save_candles("NSE", "AAA", candles)
            storage.save_candles("NSE", "ZZZ", candles)

            result = run_gtt_gain_study({"strategy": {"use_completed_weeks_only": False}}, storage, exchange="NSE")

        self.assertEqual(result.summary["symbols_processed"], 3)
        self.assertEqual(set(result.stock_stats["symbol"]), {"AAA", "BBB", "MISSINGNAME"})
        self.assertNotIn("ZZZ", set(result.stock_stats["symbol"]))
        missing_name_row = result.stock_stats[result.stock_stats["symbol"] == "MISSINGNAME"].iloc[0]
        self.assertEqual(missing_name_row["name"], "MISSINGNAME")

    def test_gtt_display_summary_uses_aligned_kite_universe_count_not_saved_summary_count(self) -> None:
        summary = _gtt_display_summary(
            {"symbols_processed": 2853},
            pd.DataFrame([{"symbol": "AAA"}, {"symbol": "BBB"}]),
            pd.DataFrame(
                [
                    {"symbol": "AAA", "valid_daily_window": True, "max_gain_pct": 12.0, "hit_10pct": True},
                    {"symbol": "BBB", "valid_daily_window": False, "max_gain_pct": pd.NA, "hit_10pct": False},
                ]
            ),
            pd.DataFrame([{"symbol": "AAA"}]),
        )

        self.assertEqual(summary["symbols_processed"], 2)
        self.assertEqual(summary["closed_pairs"], 2)
        self.assertEqual(summary["valid_pairs"], 1)
        self.assertEqual(summary["open_buy_positions"], 1)
        self.assertEqual(summary["hit_10pct_rate_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
