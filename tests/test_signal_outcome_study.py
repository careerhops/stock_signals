from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from stock_screener.data.storage import Storage
from stock_screener.signal_outcome_study import (
    SignalOutcomeStudyResult,
    build_signal_outcome_pairs,
    build_stock_outcome_stats,
    run_signal_outcome_study,
)


class SignalOutcomeStudyTests(unittest.TestCase):
    def test_build_signal_outcome_pairs_tracks_target_and_peak_before_sell(self) -> None:
        strategy_output = pd.DataFrame(
            [
                {"date": "2025-04-21", "close": 100.0, "final_buy": True, "final_sell": False, "signal": "BUY"},
                {"date": "2025-05-19", "close": 108.0, "final_buy": False, "final_sell": True, "signal": "SELL"},
            ]
        )
        daily = pd.DataFrame(
            [
                {"date": "2025-04-21", "open": 100, "high": 100, "low": 99, "close": 100, "volume": 1000},
                {"date": "2025-04-24", "open": 101, "high": 111, "low": 100, "close": 108, "volume": 1000},
                {"date": "2025-05-02", "open": 110, "high": 118, "low": 105, "close": 116, "volume": 1000},
                {"date": "2025-05-19", "open": 108, "high": 109, "low": 107, "close": 108, "volume": 1000},
            ]
        )

        pairs = build_signal_outcome_pairs(
            daily=daily,
            strategy_output=strategy_output,
            exchange="NSE",
            symbol="AAA",
            name="A Ltd",
            target_gain_pct=10.0,
        )

        self.assertEqual(len(pairs), 1)
        row = pairs.iloc[0]
        self.assertTrue(bool(row["hit_target_pct"]))
        self.assertEqual(int(row["days_to_target"]), 3)
        self.assertEqual(int(row["days_to_peak"]), 11)
        self.assertAlmostEqual(float(row["max_gain_pct_before_sell"]), 18.0)
        self.assertFalse(bool(row["failed_buy_flag"]))

    def test_build_stock_outcome_stats_keeps_current_signal_universe_even_without_pairs(self) -> None:
        signal_universe = pd.DataFrame(
            [
                {
                    "exchange": "NSE",
                    "symbol": "AAA",
                    "name": "A Ltd",
                    "latest_week_date": "2026-05-01",
                    "current_signal": "BUY",
                    "current_signal_date": "2026-05-01",
                }
            ]
        )

        stats = build_stock_outcome_stats(pd.DataFrame(), signal_universe, 10.0)

        self.assertEqual(len(stats), 1)
        self.assertEqual(stats.iloc[0]["symbol"], "AAA")
        self.assertEqual(int(stats.iloc[0]["historical_buy_count"]), 0)
        self.assertEqual(float(stats.iloc[0]["target_hit_rate_pct"]), 0.0)

    def test_run_signal_outcome_study_limits_universe_to_current_scope(self) -> None:
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
            aaa_daily = pd.DataFrame(
                [
                    {"date": "2026-04-28", "open": 100, "high": 103, "low": 99, "close": 101, "volume": 1000},
                    {"date": "2026-04-29", "open": 101, "high": 104, "low": 100, "close": 102, "volume": 1000},
                    {"date": "2026-04-30", "open": 102, "high": 105, "low": 101, "close": 103, "volume": 1000},
                    {"date": "2026-05-01", "open": 103, "high": 108, "low": 102, "close": 200, "volume": 1000},
                ]
            )
            bbb_daily = pd.DataFrame(
                [
                    {"date": "2026-04-28", "open": 100, "high": 103, "low": 99, "close": 101, "volume": 1000},
                    {"date": "2026-04-29", "open": 101, "high": 104, "low": 100, "close": 102, "volume": 1000},
                    {"date": "2026-04-30", "open": 102, "high": 105, "low": 101, "close": 103, "volume": 1000},
                    {"date": "2026-05-01", "open": 103, "high": 108, "low": 102, "close": 300, "volume": 1000},
                ]
            )
            storage.save_candles("NSE", "AAA", aaa_daily)
            storage.save_candles("NSE", "BBB", bbb_daily)
            storage.save_signals(
                "latest_raw_signals.csv",
                pd.DataFrame(
                    [
                        {"exchange": "NSE", "symbol": "AAA", "name": "A Ltd", "date": "2026-05-01", "signal": "BUY"},
                        {"exchange": "NSE", "symbol": "BBB", "name": "B Ltd", "date": "2026-05-01", "signal": "SELL"},
                    ]
                ),
            )
            storage.save_signals(
                "latest_scan_details.csv",
                pd.DataFrame(
                    [
                        {"exchange": "NSE", "symbol": "AAA", "name": "A Ltd", "latest_signal": "BUY", "latest_signal_date": "2026-05-01"},
                        {"exchange": "NSE", "symbol": "BBB", "name": "B Ltd", "latest_signal": "SELL", "latest_signal_date": "2026-05-01"},
                    ]
                ),
            )

            def fake_strategy(candles: pd.DataFrame, config: dict) -> pd.DataFrame:
                frame = candles.copy().reset_index(drop=True)
                frame["final_buy"] = False
                frame["final_sell"] = False
                frame["signal"] = "NONE"
                frame["ema_20"] = frame["close"]
                frame["ema_50"] = frame["close"]
                frame["volume_confirmation"] = False
                frame["volume_confirmation_ratio"] = pd.NA
                if float(frame.iloc[-1]["close"]) == 200.0:
                    frame.loc[frame.index[-1], "final_buy"] = True
                    frame.loc[frame.index[-1], "signal"] = "BUY"
                else:
                    frame.loc[frame.index[-1], "final_sell"] = True
                    frame.loc[frame.index[-1], "signal"] = "SELL"
                return frame

            config = {
                "universe": {
                    "mode": "nse_all",
                    "instrument_types": ["EQ"],
                    "restrict_to_metadata_symbols": False,
                    "approximate_nse_traded_universe": {"enabled": False},
                },
                "strategy": {"weekly_anchor": "W-FRI", "use_completed_weeks_only": True},
            }

            with patch("stock_screener.signal_outcome_study.run_weekly_buy_sell", side_effect=fake_strategy):
                result = run_signal_outcome_study(config, storage, exchange="NSE", signal_scope="buy", target_gain_pct=10.0)

        self.assertEqual(set(result.signal_universe["symbol"]), {"AAA"})
        self.assertEqual(set(result.stock_stats["symbol"]), {"AAA"})

    def test_signal_outcome_page_gracefully_handles_broken_saved_outputs(self) -> None:
        from stock_screener.web.main import app

        client = TestClient(app)
        with patch(
            "stock_screener.web.main.load_signal_outcome_outputs",
            side_effect=RuntimeError("broken csv"),
        ):
            response = client.get("/signal-outcome-study")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Could not load Signal Outcome Study", response.text)

    def test_signal_outcome_page_handles_shared_filters_without_type_error(self) -> None:
        from stock_screener.web.main import app

        client = TestClient(app)
        response = client.get(
            "/signal-outcome-study?market_cap_bucket=Large%20Cap&min_market_cap_cr=1000&min_cmp=100&stock_search=A"
        )

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("Invalid comparison between dtype=float64 and str", response.text)


if __name__ == "__main__":
    unittest.main()
