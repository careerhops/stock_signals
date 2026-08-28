from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from stock_screener.data.storage import Storage
from stock_screener.knox_envelope_study import (
    _annualized_sharpe,
    _confirmation_matches,
    _rsi_extreme_between,
    calculate_knox_envelope,
    load_knox_envelope_outputs,
    run_knox_envelope_study,
    save_knox_envelope_outputs,
)
from stock_screener.web.main import app


def _bullish_knox_frame() -> pd.DataFrame:
    close = [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 80, 82, 90]
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=len(close), freq="B"),
            "open": close,
            "high": [value + 1 for value in close],
            "low": [value - 1 for value in close],
            "close": close,
            "volume": 1000,
        }
    )
    frame.loc[len(frame) - 1, "low"] = 70
    return frame


class KnoxEnvelopeStudyTests(unittest.TestCase):
    def test_annualized_sharpe_uses_trailing_daily_excess_returns(self) -> None:
        close = pd.Series([100.0, 101.0, 100.5, 102.0, 101.5, 103.0])
        returns = close.pct_change(fill_method=None).dropna()
        expected = float(returns.mean() / returns.std(ddof=1) * (252**0.5))

        actual, observations = _annualized_sharpe(
            close,
            lookback_days=5,
            annual_risk_free_rate_pct=0.0,
        )

        self.assertEqual(observations, 5)
        self.assertAlmostEqual(actual, expected)

    def test_rsi_window_does_not_reach_before_knox_reference(self) -> None:
        rsi = pd.Series([25.0] + [40.0] * 101)

        self.assertFalse(
            _rsi_extreme_between(
                rsi,
                row_index=101,
                reference_bars=100,
                threshold=30.0,
                above=False,
            )
        )

    def test_confirmation_links_first_bullish_close_after_setup(self) -> None:
        matches, setup_indexes = _confirmation_matches(
            pd.Series([False, True, False, False, False]),
            pd.Series([False, False, True, False, True]),
            window_bars=3,
        )

        self.assertEqual(matches.tolist(), [False, False, True, False, False])
        self.assertEqual(setup_indexes.tolist(), [-1, -1, 1, -1, -1])

    def test_strong_bullish_close_requires_close_in_selected_range_portion(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=3, freq="B"),
                "open": [100.0, 100.0, 101.0],
                "high": [102.0, 102.0, 110.0],
                "low": [98.0, 98.0, 100.0],
                "close": [100.0, 100.0, 109.0],
                "volume": 1000.0,
            }
        )

        result = calculate_knox_envelope(
            frame,
            confirmation_close_location_pct=80.0,
        )

        self.assertTrue(bool(result.iloc[-1]["strong_bullish_close"]))
        self.assertAlmostEqual(float(result.iloc[-1]["close_location_pct"]), 90.0)

    def test_calculation_matches_published_bullish_knox_loop(self) -> None:
        result = calculate_knox_envelope(
            _bullish_knox_frame(),
            knox_lookback=10,
            rsi_length=3,
            momentum_length=2,
            envelope_length=5,
            envelope_percent=10.0,
            envelope_ma_type="SMA",
        )

        latest = result.iloc[-1]
        self.assertTrue(bool(latest["knox_bullish"]))
        self.assertFalse(bool(latest["knox_bearish"]))
        self.assertEqual(int(latest["knox_reference_bars"]), 10)
        self.assertAlmostEqual(float(latest["momentum"]), 10.0)

    def test_envelope_uses_tradingview_percentage_formula(self) -> None:
        frame = _bullish_knox_frame().iloc[:5].copy()
        result = calculate_knox_envelope(
            frame,
            knox_lookback=5,
            rsi_length=3,
            momentum_length=2,
            envelope_length=5,
            envelope_percent=10.0,
            envelope_ma_type="SMA",
        )

        latest = result.iloc[-1]
        self.assertAlmostEqual(float(latest["envelope_basis"]), 98.0)
        self.assertAlmostEqual(float(latest["envelope_upper"]), 107.8)
        self.assertAlmostEqual(float(latest["envelope_lower"]), 88.2)

    def test_cmf_uses_standard_money_flow_volume_formula(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=20, freq="B"),
                "open": 100.0,
                "high": 110.0,
                "low": 90.0,
                "close": 109.0,
                "volume": 100.0,
            }
        )

        result = calculate_knox_envelope(frame, cmf_length=20)

        self.assertAlmostEqual(float(result.iloc[-1]["cmf"]), 0.9)

    def test_lower_band_proximity_uses_endpoint_candle_low(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=20, freq="B"),
                "open": 100.0,
                "high": 101.0,
                "low": 90.0,
                "close": 100.0,
                "volume": 1000,
            }
        )
        near_low = calculate_knox_envelope(
            frame,
            envelope_length=20,
            envelope_percent=10.0,
            envelope_proximity_pct=2.0,
        )
        self.assertTrue(bool(near_low.iloc[-1]["envelope_lower_support"]))
        self.assertAlmostEqual(float(near_low.iloc[-1]["low_distance_from_lower_pct"]), 0.0)

        frame.loc[frame.index[-1], "low"] = 100.0
        far_low = calculate_knox_envelope(
            frame,
            envelope_length=20,
            envelope_percent=10.0,
            envelope_proximity_pct=2.0,
        )
        self.assertFalse(bool(far_low.iloc[-1]["envelope_lower_support"]))

    def test_study_requires_knox_and_envelope_on_same_recent_bar(self) -> None:
        with TemporaryDirectory() as temp_dir:
            storage = Storage(Path(temp_dir))
            storage.save_instruments(
                pd.DataFrame([{"exchange": "NSE", "tradingsymbol": "PASS", "name": "Pass Ltd"}])
            )
            storage.save_candles("NSE", "PASS", _bullish_knox_frame(), "1D")

            result = run_knox_envelope_study(
                storage,
                symbols=["PASS"],
                knox_lookback=10,
                rsi_length=3,
                momentum_length=2,
                signal_lookback_bars=2,
                signal_direction="bullish",
                envelope_length=5,
                envelope_percent=10.0,
                envelope_mode="inside_envelope",
                confirmation_mode="disabled",
            )
            save_knox_envelope_outputs(result, Path(temp_dir) / "knox_envelope")
            loaded = load_knox_envelope_outputs(Path(temp_dir) / "knox_envelope")

        self.assertEqual(int(loaded.summary["symbols_processed"]), 1)
        self.assertEqual(int(loaded.summary["combined_matches"]), 1)
        row = loaded.stock_stats.iloc[0]
        self.assertTrue(bool(row["combined_match"]))
        self.assertEqual(row["match_side"], "BULLISH")
        self.assertEqual(int(row["signal_age_bars"]), 0)
        self.assertEqual(float(row["reference_low"]), 90.0)
        self.assertEqual(float(row["signal_low"]), 70.0)
        self.assertIn("sharpe_ratio", row.index)
        self.assertTrue(bool(row["technical_match"]))
        self.assertTrue(bool(row["sharpe_pass"]))
        self.assertFalse(bool(row["sharpe_available"]))
        self.assertEqual(int(row["sharpe_observations"]), 0)

    def test_minimum_sharpe_filters_an_otherwise_valid_technical_match(self) -> None:
        with TemporaryDirectory() as temp_dir:
            storage = Storage(Path(temp_dir))
            storage.save_candles("NSE", "PASS", _bullish_knox_frame(), "1D")

            result = run_knox_envelope_study(
                storage,
                symbols=["PASS"],
                knox_lookback=10,
                rsi_length=3,
                momentum_length=2,
                signal_lookback_bars=2,
                signal_direction="bullish",
                envelope_length=5,
                envelope_percent=10.0,
                envelope_mode="inside_envelope",
                confirmation_mode="disabled",
                use_sharpe_filter=True,
                sharpe_lookback_days=10,
                min_sharpe_ratio=100.0,
            )

        row = result.stock_stats.iloc[0]
        self.assertTrue(bool(row["technical_match"]))
        self.assertFalse(bool(row["sharpe_pass"]))
        self.assertFalse(bool(row["combined_match"]))

    def test_study_reports_endpoint_and_later_bullish_confirmation_separately(self) -> None:
        with TemporaryDirectory() as temp_dir:
            storage = Storage(Path(temp_dir))
            frame = _bullish_knox_frame()
            endpoint_date = frame.iloc[-1]["date"]
            confirmation_date = endpoint_date + pd.offsets.BDay(1)
            frame.loc[len(frame)] = {
                "date": confirmation_date,
                "open": 89.0,
                "high": 96.0,
                "low": 88.0,
                "close": 95.0,
                "volume": 1000,
            }
            storage.save_candles("NSE", "PASS", frame, "1D")

            result = run_knox_envelope_study(
                storage,
                symbols=["PASS"],
                knox_lookback=10,
                rsi_length=3,
                momentum_length=2,
                signal_lookback_bars=2,
                signal_direction="bullish",
                envelope_length=5,
                envelope_percent=10.0,
                envelope_mode="inside_envelope",
                cmf_condition="disabled",
                confirmation_mode="strong_bullish_close",
                confirmation_window_bars=3,
                confirmation_close_location_pct=80.0,
            )

        row = result.stock_stats.iloc[0]
        self.assertTrue(bool(row["combined_match"]))
        self.assertEqual(pd.Timestamp(row["signal_date"]), endpoint_date)
        self.assertEqual(pd.Timestamp(row["confirmation_date"]), confirmation_date)
        self.assertEqual(int(row["confirmation_delay_bars"]), 1)
        self.assertTrue(bool(row["confirmation_pass"]))

    def test_study_excludes_confirmation_after_as_of_date(self) -> None:
        with TemporaryDirectory() as temp_dir:
            storage = Storage(Path(temp_dir))
            frame = _bullish_knox_frame()
            endpoint_date = pd.Timestamp(frame.iloc[-1]["date"])
            frame.loc[len(frame)] = {
                "date": endpoint_date + pd.offsets.BDay(1),
                "open": 89.0,
                "high": 96.0,
                "low": 88.0,
                "close": 95.0,
                "volume": 1000,
            }
            storage.save_candles("NSE", "PASS", frame, "1D")

            result = run_knox_envelope_study(
                storage,
                symbols=["PASS"],
                knox_lookback=10,
                rsi_length=3,
                momentum_length=2,
                signal_lookback_bars=1,
                signal_direction="bullish",
                envelope_length=5,
                envelope_percent=10.0,
                envelope_mode="inside_envelope",
                cmf_condition="disabled",
                confirmation_mode="strong_bullish_close",
                confirmation_window_bars=3,
                confirmation_close_location_pct=80.0,
                as_of_date=endpoint_date,
            )

        self.assertFalse(bool(result.stock_stats.iloc[0]["combined_match"]))

    def test_direction_filter_rejects_opposite_knox_signal(self) -> None:
        with TemporaryDirectory() as temp_dir:
            storage = Storage(Path(temp_dir))
            storage.save_candles("NSE", "PASS", _bullish_knox_frame(), "1D")

            result = run_knox_envelope_study(
                storage,
                symbols=["PASS"],
                knox_lookback=10,
                rsi_length=3,
                momentum_length=2,
                signal_direction="bearish",
                envelope_length=5,
                envelope_percent=10.0,
                envelope_mode="inside_envelope",
                confirmation_mode="disabled",
            )

        self.assertEqual(int(result.summary["combined_matches"]), 0)
        self.assertFalse(bool(result.stock_stats.iloc[0]["combined_match"]))

    def test_positive_cmf_condition_is_required_on_knox_endpoint(self) -> None:
        with TemporaryDirectory() as temp_dir:
            storage = Storage(Path(temp_dir))
            frame = _bullish_knox_frame()
            frame["high"] = frame["close"] + 10.0
            frame["low"] = frame["close"] - 1.0
            frame.loc[len(frame) - 1, "low"] = 70.0
            storage.save_candles("NSE", "PASS", frame, "1D")

            filtered = run_knox_envelope_study(
                storage,
                symbols=["PASS"],
                knox_lookback=10,
                rsi_length=3,
                momentum_length=2,
                envelope_length=5,
                envelope_percent=10.0,
                envelope_mode="inside_envelope",
                cmf_length=5,
                cmf_condition="greater_than_zero",
                confirmation_mode="disabled",
            )
            unfiltered = run_knox_envelope_study(
                storage,
                symbols=["PASS"],
                knox_lookback=10,
                rsi_length=3,
                momentum_length=2,
                envelope_length=5,
                envelope_percent=10.0,
                envelope_mode="inside_envelope",
                cmf_length=5,
                cmf_condition="disabled",
                confirmation_mode="disabled",
            )

        self.assertEqual(int(filtered.summary["combined_matches"]), 0)
        self.assertEqual(int(unfiltered.summary["combined_matches"]), 1)
        self.assertLess(float(calculate_knox_envelope(frame, cmf_length=5).iloc[-1]["cmf"]), 0.0)

    def test_recent_knox_and_later_envelope_touch_do_not_combine(self) -> None:
        with TemporaryDirectory() as temp_dir:
            storage = Storage(Path(temp_dir))
            frame = _bullish_knox_frame()
            frame.loc[len(frame)] = {
                "date": frame["date"].iloc[-1] + pd.offsets.BDay(1),
                "open": 85.0,
                "high": 86.0,
                "low": 75.78,
                "close": 85.0,
                "volume": 1000,
            }
            storage.save_candles("NSE", "PASS", frame, "1D")

            result = run_knox_envelope_study(
                storage,
                symbols=["PASS"],
                knox_lookback=10,
                rsi_length=3,
                momentum_length=2,
                signal_lookback_bars=2,
                signal_direction="bullish",
                envelope_length=5,
                envelope_percent=10.0,
                envelope_mode="lower_support",
                envelope_proximity_pct=2.0,
                confirmation_mode="disabled",
            )

        self.assertEqual(int(result.summary["combined_matches"]), 0)
        self.assertFalse(bool(result.stock_stats.iloc[0]["combined_match"]))

    def test_default_window_only_keeps_confirmation_on_latest_bar(self) -> None:
        with TemporaryDirectory() as temp_dir:
            storage = Storage(Path(temp_dir))
            frame = _bullish_knox_frame()
            frame.loc[len(frame)] = {
                "date": frame["date"].iloc[-1] + pd.offsets.BDay(1),
                "open": 90.0,
                "high": 91.0,
                "low": 89.0,
                "close": 90.0,
                "volume": 1000,
            }
            storage.save_candles("NSE", "PASS", frame, "1D")

            fresh_only = run_knox_envelope_study(
                storage,
                symbols=["PASS"],
                knox_lookback=10,
                rsi_length=3,
                momentum_length=2,
                envelope_length=5,
                envelope_percent=10.0,
                envelope_mode="inside_envelope",
                confirmation_mode="disabled",
            )
            rolling = run_knox_envelope_study(
                storage,
                symbols=["PASS"],
                knox_lookback=10,
                rsi_length=3,
                momentum_length=2,
                signal_lookback_bars=2,
                envelope_length=5,
                envelope_percent=10.0,
                envelope_mode="inside_envelope",
                confirmation_mode="disabled",
            )

        self.assertFalse(bool(fresh_only.stock_stats.iloc[0]["combined_match"]))
        self.assertTrue(bool(rolling.stock_stats.iloc[0]["combined_match"]))

    def test_page_renders_saved_matches_and_pinescript_download(self) -> None:
        with TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            storage = Storage(data_root)
            storage.save_instruments(
                pd.DataFrame([{"exchange": "NSE", "tradingsymbol": "PASS", "name": "Pass Ltd"}])
            )
            storage.save_candles("NSE", "PASS", _bullish_knox_frame(), "1D")
            result = run_knox_envelope_study(
                storage,
                symbols=["PASS"],
                knox_lookback=10,
                rsi_length=3,
                momentum_length=2,
                envelope_length=5,
                envelope_percent=10.0,
                envelope_mode="inside_envelope",
                confirmation_mode="disabled",
            )
            save_knox_envelope_outputs(result, data_root / "knox_envelope")

            with patch("stock_screener.web.main.get_data_root", return_value=data_root):
                client = TestClient(app)
                page = client.get("/knox-envelope")
                pine = client.get("/knox-envelope/pinescript")

        self.assertEqual(page.status_code, 200)
        self.assertIn("Daily reversal confluence", page.text)
        self.assertIn('name="cmf_length"', page.text)
        self.assertIn('name="cmf_condition"', page.text)
        self.assertIn('name="confirmation_mode"', page.text)
        self.assertIn('name="confirmation_window_bars"', page.text)
        self.assertIn('name="use_sharpe_filter"', page.text)
        self.assertIn('name="sharpe_lookback_days"', page.text)
        self.assertIn('name="annual_risk_free_rate_pct"', page.text)
        self.assertIn('name="min_sharpe_ratio"', page.text)
        self.assertIn("Sharpe Ratio", page.text)
        self.assertIn("Strong bullish close", page.text)
        self.assertIn("Greater than 0", page.text)
        self.assertIn("PASS", page.text)
        self.assertIn("Download PineScript", page.text)
        self.assertEqual(pine.status_code, 200)
        self.assertIn("Knoxville + Envelope Confluence", pine.text)
        self.assertIn("line.new", pine.text)

    def test_run_form_passes_confirmation_settings_to_background_job(self) -> None:
        with TemporaryDirectory() as temp_dir:
            with (
                patch("stock_screener.web.main.get_data_root", return_value=Path(temp_dir)),
                patch("stock_screener.web.main._run_knox_envelope_job") as run_job,
            ):
                client = TestClient(app)
                response = client.post(
                    "/knox-envelope/run",
                    data={
                        "confirmation_mode": "strong_bullish_close",
                        "confirmation_window_bars": "3",
                        "confirmation_close_location_pct": "80",
                        "cmf_condition": "disabled",
                        "use_sharpe_filter": "1",
                        "sharpe_lookback_days": "126",
                        "annual_risk_free_rate_pct": "6.5",
                        "min_sharpe_ratio": "0.75",
                    },
                    follow_redirects=False,
                )

        self.assertEqual(response.status_code, 303)
        self.assertIn("confirmation_mode=strong_bullish_close", response.headers["location"])
        self.assertIn("confirmation_window_bars=3", response.headers["location"])
        self.assertIn("confirmation_close_location_pct=80.0", response.headers["location"])
        self.assertIn("use_sharpe_filter=1", response.headers["location"])
        self.assertIn("sharpe_lookback_days=126", response.headers["location"])
        self.assertIn("annual_risk_free_rate_pct=6.5", response.headers["location"])
        self.assertIn("min_sharpe_ratio=0.75", response.headers["location"])
        run_job.assert_called_once()


if __name__ == "__main__":
    unittest.main()
