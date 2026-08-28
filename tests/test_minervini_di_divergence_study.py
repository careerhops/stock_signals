from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from stock_screener.data.storage import Storage
from stock_screener.minervini_di_divergence_study import (
    MinerviniDiDivergenceStudyResult,
    _passes_pre_breakout_watchlist,
    evaluate_di_divergence,
    load_minervini_di_divergence_outputs,
    run_minervini_di_divergence_study,
    save_minervini_di_divergence_outputs,
)
from stock_screener.web.main import _parse_nse_symbol_list, app


class MinerviniDiDivergenceStudyTests(unittest.TestCase):
    def test_target_symbol_parser_accepts_commas_spaces_lines_and_nse_prefix(self) -> None:
        symbols = _parse_nse_symbol_list("NSE:RELIANCE, infy\nHDFCBANK;INFY")

        self.assertEqual(symbols, ["RELIANCE", "INFY", "HDFCBANK"])

    def test_two_day_divergence_requires_plus_up_minus_down_and_plus_above(self) -> None:
        dates = pd.bdate_range("2026-08-03", periods=3)
        passing = pd.DataFrame(
            {
                "date": dates,
                "di_plus": [18.0, 21.0, 25.0],
                "di_minus": [17.0, 14.0, 11.0],
            }
        )
        plus_stalled = passing.copy()
        plus_stalled["di_plus"] = [18.0, 18.0, 25.0]
        minus_rose = passing.copy()
        minus_rose["di_minus"] = [17.0, 18.0, 11.0]
        plus_below = passing.copy()
        plus_below["di_plus"] = [8.0, 9.0, 10.0]

        passed = evaluate_di_divergence(passing, divergence_days=2)

        self.assertTrue(passed["di_divergence_pass"])
        self.assertEqual(passed["latest_di_spread"], 14.0)
        self.assertEqual(passed["spread_change_2d"], 13.0)
        self.assertFalse(evaluate_di_divergence(plus_stalled, divergence_days=2)["di_divergence_pass"])
        self.assertFalse(evaluate_di_divergence(minus_rose, divergence_days=2)["di_divergence_pass"])
        self.assertFalse(evaluate_di_divergence(plus_below, divergence_days=2)["di_divergence_pass"])

    def test_combined_scan_accepts_scores_equal_to_70(self) -> None:
        with TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            storage = Storage(data_root)
            daily = self._daily_frame()
            storage.save_instruments(
                pd.DataFrame([{"exchange": "NSE", "tradingsymbol": "PASS", "name": "Pass Ltd"}])
            )
            storage.save_candles("NSE", "PASS", daily, "1D")
            storage.save_candles(
                "NSE_INDEX",
                "NIFTY 500",
                pd.DataFrame({"date": daily["date"], "close": 100.0}),
                "1D",
            )

            with (
                patch(
                    "stock_screener.minervini_di_divergence_study.calculate_adx_di",
                    return_value=self._adx_frame(daily["date"]),
                ),
                patch(
                    "stock_screener.minervini_di_divergence_study.evaluate_minervini_quality",
                    return_value=self._quality_metrics(70.0, 70.0, 70.0),
                ),
            ):
                result = run_minervini_di_divergence_study(storage, symbols=["PASS"], min_score=70.0)
            save_minervini_di_divergence_outputs(result, data_root / "minervini_di_divergence")
            loaded = load_minervini_di_divergence_outputs(data_root / "minervini_di_divergence")

        row = loaded.stock_stats.iloc[0]
        self.assertTrue(bool(row["di_divergence_pass"]))
        self.assertTrue(bool(row["minervini_threshold_pass"]))
        self.assertTrue(bool(row["combined_pass"]))
        self.assertEqual(int(loaded.summary["combined_matches"]), 1)

    def test_combined_scan_rejects_one_quality_score_below_threshold(self) -> None:
        with TemporaryDirectory() as temp_dir:
            storage = Storage(Path(temp_dir))
            daily = self._daily_frame()
            storage.save_candles("NSE", "FAIL", daily, "1D")
            storage.save_candles(
                "NSE_INDEX",
                "NIFTY 500",
                pd.DataFrame({"date": daily["date"], "close": 100.0}),
                "1D",
            )

            with (
                patch(
                    "stock_screener.minervini_di_divergence_study.calculate_adx_di",
                    return_value=self._adx_frame(daily["date"]),
                ),
                patch(
                    "stock_screener.minervini_di_divergence_study.evaluate_minervini_quality",
                    return_value=self._quality_metrics(85.0, 69.9, 90.0),
                ),
            ):
                result = run_minervini_di_divergence_study(storage, symbols=["FAIL"], min_score=70.0)

        self.assertFalse(bool(result.stock_stats.iloc[0]["minervini_threshold_pass"]))
        self.assertFalse(bool(result.stock_stats.iloc[0]["combined_pass"]))

    def test_combined_scan_rejects_stock_date_that_trails_benchmark(self) -> None:
        with TemporaryDirectory() as temp_dir:
            storage = Storage(Path(temp_dir))
            daily = self._daily_frame()
            storage.save_candles("NSE", "STALE", daily, "1D")
            storage.save_candles(
                "NSE_INDEX",
                "NIFTY 500",
                pd.DataFrame({"date": daily["date"], "close": 100.0}),
                "1D",
            )
            stale_quality = {**self._quality_metrics(90.0, 90.0, 90.0), "latest_date": "2026-06-26"}

            with (
                patch(
                    "stock_screener.minervini_di_divergence_study.calculate_adx_di",
                    return_value=self._adx_frame(daily["date"]),
                ),
                patch(
                    "stock_screener.minervini_di_divergence_study.evaluate_minervini_quality",
                    return_value=stale_quality,
                ),
            ):
                result = run_minervini_di_divergence_study(storage, symbols=["STALE"])

        row = result.stock_stats.iloc[0]
        self.assertFalse(bool(row["is_latest_market_date"]))
        self.assertFalse(bool(row["combined_pass"]))
        self.assertEqual(int(result.summary["stale_stock_dates"]), 1)

    def test_pre_breakout_watchlist_requires_every_hard_condition(self) -> None:
        divergence = {"di_divergence_pass": True}
        quality = self._quality_metrics(90.0, 85.0, 80.0)

        self.assertTrue(_passes_pre_breakout_watchlist(divergence, quality))

        failed_market = {**quality, "market_regime": "NEUTRAL"}
        failed_liquidity = {**quality, "avg_turnover_cr": 24.9}
        failed_extension = {**quality, "atr_extension": 2.01}
        self.assertFalse(_passes_pre_breakout_watchlist(divergence, failed_market))
        self.assertFalse(_passes_pre_breakout_watchlist(divergence, failed_liquidity))
        self.assertFalse(_passes_pre_breakout_watchlist(divergence, failed_extension))

    def test_page_renders_only_combined_matches_by_default(self) -> None:
        with TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            result = MinerviniDiDivergenceStudyResult(
                summary={
                    "symbols_processed": 2,
                    "stocks_evaluated": 2,
                    "di_divergence_matches": 2,
                    "minervini_threshold_matches": 1,
                    "combined_matches": 1,
                    "pre_breakout_matches": 1,
                    "adx_length": 14,
                    "divergence_days": 2,
                    "min_score": 70.0,
                },
                stock_stats=pd.DataFrame(
                    [
                        {
                            "symbol": "PASS",
                            "name": "Pass Ltd",
                            "latest_date": "2026-08-10",
                            "latest_close": 150.0,
                            "di_divergence_pass": True,
                            "stock_quality_score": 90.0,
                            "stock_quality_grade": "LEADER",
                            "setup_quality_score": 85.0,
                            "setup_quality_grade": "READY",
                            "entry_quality_score": 80.0,
                            "entry_quality_grade": "GOOD",
                            "data_status": "READY",
                            "combined_pass": True,
                            "pre_breakout_pass": True,
                            "is_latest_market_date": True,
                            "market_regime": "BULLISH",
                            "trend_pass_count": 7,
                            "relative_performance_pct": 25.0,
                            "rs_line_near_high": True,
                            "avg_turnover_cr": 30.0,
                            "vcp_score": 5,
                            "volume_dry_ratio": 0.6,
                            "pressure_pct": 65.0,
                            "distribution_count_20d": 2,
                            "obv_state": "ACCUMULATING",
                            "pivot_distance_pct": -2.0,
                            "distance_from_sma50_pct": 6.0,
                            "atr_extension": 1.0,
                            "latest_di_plus": 25.0,
                            "latest_di_minus": 11.0,
                            "spread_change_2d": 13.0,
                        },
                        {
                            "symbol": "FAIL",
                            "name": "Fail Ltd",
                            "latest_date": "2026-08-10",
                            "latest_close": 100.0,
                            "di_divergence_pass": True,
                            "stock_quality_score": 69.0,
                            "setup_quality_score": 75.0,
                            "entry_quality_score": 80.0,
                            "data_status": "READY",
                            "combined_pass": False,
                            "pre_breakout_pass": False,
                            "is_latest_market_date": True,
                        },
                    ]
                ),
            )
            save_minervini_di_divergence_outputs(result, data_root / "minervini_di_divergence")
            targeted_result = MinerviniDiDivergenceStudyResult(
                summary={
                    **result.summary,
                    "requested_symbols_csv": "PASS,FAIL",
                    "refresh_requested_count": 2,
                    "refresh_unavailable_count": 0,
                },
                stock_stats=result.stock_stats,
            )
            save_minervini_di_divergence_outputs(
                targeted_result,
                data_root / "minervini_di_divergence_targeted",
            )

            with (
                patch.dict(os.environ, {"DATA_ROOT": temp_dir}),
                patch("stock_screener.web.main.get_data_root", return_value=data_root),
            ):
                response = TestClient(app).get("/minervini-di-divergence")

        self.assertEqual(response.status_code, 200)
        self.assertIn("DI divergence with quality confirmation", response.text)
        self.assertIn(">PASS<", response.text)
        self.assertNotIn(">FAIL<", response.text)
        self.assertIn("Run DI + Minervini Scan", response.text)
        self.assertIn("Pre-Breakout Watchlist", response.text)
        self.assertIn('id="pre-breakout-symbols-csv"', response.text)
        self.assertIn("Scan Selected Stocks", response.text)
        self.assertIn('action="/minervini-di-divergence/run-targeted"', response.text)
        self.assertIn('id="targeted-minervini-symbols"', response.text)

    def test_targeted_run_passes_only_submitted_symbols_to_background_job(self) -> None:
        with TemporaryDirectory() as temp_dir:
            with (
                patch("stock_screener.web.main.get_data_root", return_value=Path(temp_dir)),
                patch("stock_screener.web.main._run_minervini_di_divergence_job") as run_job,
            ):
                response = TestClient(app).post(
                    "/minervini-di-divergence/run-targeted",
                    data={
                        "target_symbols": "RELIANCE, INFY\nHDFCBANK",
                        "adx_length": "14",
                        "divergence_days": "2",
                        "min_score": "70",
                    },
                    follow_redirects=False,
                )

        self.assertEqual(response.status_code, 303)
        self.assertIn("targeted_job=", response.headers["location"])
        run_job.assert_called_once()
        args = run_job.call_args.args
        self.assertEqual(args[6], ["RELIANCE", "INFY", "HDFCBANK"])
        self.assertTrue(args[7])

    @staticmethod
    def _daily_frame() -> pd.DataFrame:
        dates = pd.bdate_range("2025-07-01", periods=260)
        return pd.DataFrame(
            {
                "date": dates,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 100_000.0,
            }
        )

    @staticmethod
    def _adx_frame(dates: pd.Series) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(dates),
                "di_plus": 18.0,
                "di_minus": 17.0,
            }
        )
        frame.loc[frame.index[-3:], "di_plus"] = [18.0, 21.0, 25.0]
        frame.loc[frame.index[-3:], "di_minus"] = [17.0, 14.0, 11.0]
        return frame

    @staticmethod
    def _quality_metrics(stock: float, setup: float, entry: float) -> dict[str, object]:
        return {
            "latest_date": "2026-06-29",
            "latest_close": 100.0,
            "data_status": "READY",
            "stock_quality_score": stock,
            "stock_quality_grade": "STRONG",
            "setup_quality_score": setup,
            "setup_quality_grade": "DEVELOPING",
            "entry_quality_score": entry,
            "entry_quality_grade": "GOOD",
            "market_regime": "BULLISH",
            "trend_pass_count": 7,
            "vcp_score": 4,
            "relative_performance_pct": 25.0,
            "rs_line_near_high": True,
            "avg_turnover_cr": 30.0,
            "pressure_pct": 65.0,
            "distribution_count_20d": 2,
            "volume_dry_ratio": 0.6,
            "pivot_distance_pct": -1.0,
            "distance_from_sma50_pct": 6.0,
            "atr_extension": 1.0,
            "obv_state": "ACCUMULATING",
        }


if __name__ == "__main__":
    unittest.main()
