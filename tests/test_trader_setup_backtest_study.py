from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import os
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from stock_screener.data.storage import Storage
from stock_screener.trader_setup_backtest_study import (
    backtest_trader_setup_frame,
    calculate_trader_setup_signals,
    load_trader_setup_backtest_outputs,
    run_trader_setup_backtest_study,
    save_trader_setup_backtest_outputs,
    TraderSetupBacktestResult,
)
from stock_screener.web.main import app


class TraderSetupBacktestStudyTests(unittest.TestCase):
    def test_pivot_level_is_not_available_before_right_bar_confirmation(self) -> None:
        frame = self._base_frame(45)
        frame["high"] = 101.0
        frame["low"] = 99.0
        frame.loc[10, "high"] = 120.0
        frame.loc[11:12, "high"] = [110.0, 108.0]

        signals = calculate_trader_setup_signals(frame, left_bars=2, right_bars=2)

        self.assertTrue(pd.isna(signals.loc[11, "resistance_level"]))
        self.assertEqual(signals.loc[12, "resistance_level"], 120.0)

    def test_trade_enters_next_open_and_uses_conservative_stop_when_both_hit(self) -> None:
        frame = self._base_frame(35)
        frame["pivot_breakout"] = False
        frame["resistance_break"] = False
        frame["support_break"] = False
        frame["combined_long_breakout"] = False
        frame["rvol"] = 2.0
        frame["volume_oscillator"] = 25.0
        frame["momentum_pivot"] = 100.0
        frame.loc[10, ["pivot_breakout", "combined_long_breakout"]] = True
        frame.loc[11, "open"] = 100.0
        frame.loc[11, "high"] = 112.0
        frame.loc[11, "low"] = 94.0

        trades = backtest_trader_setup_frame(
            frame,
            symbol="TEST",
            start_date=frame.iloc[0]["date"],
            end_date=frame.iloc[-1]["date"],
            holding_days=20,
            profit_target_pct=10.0,
            stop_loss_pct=5.0,
            round_trip_cost_pct=0.2,
        )

        pivot_trade = trades[trades["strategy"] == "Momentum Pivot Breakout"].iloc[0]
        self.assertEqual(pivot_trade["entry_date"], frame.iloc[11]["date"].strftime("%Y-%m-%d"))
        self.assertEqual(pivot_trade["entry_price"], 100.0)
        self.assertEqual(pivot_trade["exit_reason"], "STOP")
        self.assertAlmostEqual(pivot_trade["net_return_pct"], -5.2)

    def test_study_persists_strategy_win_rate_outputs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            storage = Storage(Path(temp_dir))
            frame = self._base_frame(320)
            storage.save_instruments(
                pd.DataFrame([{"exchange": "NSE", "tradingsymbol": "TEST", "name": "Test Ltd"}])
            )
            storage.save_candles("NSE", "TEST", frame, "1D")
            result = run_trader_setup_backtest_study(
                storage,
                symbols=["TEST"],
                start_date=frame.iloc[0]["date"],
                end_date=frame.iloc[-1]["date"],
            )
            save_trader_setup_backtest_outputs(result, Path(temp_dir) / "trader_setup_backtest")
            loaded = load_trader_setup_backtest_outputs(Path(temp_dir) / "trader_setup_backtest")

        self.assertEqual(int(loaded.summary["symbols_processed"]), 1)
        self.assertIn("total_trades", loaded.summary)
        self.assertTrue(loaded.strategy_stats.empty or "win_rate_pct" in loaded.strategy_stats.columns)

    def test_signal_without_full_holding_window_is_not_counted(self) -> None:
        frame = self._base_frame(25)
        frame["pivot_breakout"] = False
        frame["resistance_break"] = False
        frame["support_break"] = False
        frame["combined_long_breakout"] = False
        frame["rvol"] = 2.0
        frame["volume_oscillator"] = 25.0
        frame["momentum_pivot"] = 100.0
        frame.loc[20, ["pivot_breakout", "combined_long_breakout"]] = True

        trades = backtest_trader_setup_frame(
            frame,
            symbol="OPEN",
            start_date=frame.iloc[0]["date"],
            end_date=frame.iloc[-1]["date"],
            holding_days=20,
        )

        self.assertTrue(trades.empty)

    def test_web_page_displays_saved_win_rate(self) -> None:
        with TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            result = TraderSetupBacktestResult(
                summary={
                    "symbols_processed": 100,
                    "symbols_with_9y_history": 80,
                    "refresh_coverage_pct": 98.0,
                    "refresh_expected_date": "2026-08-18",
                    "holding_days": 20,
                    "round_trip_cost_pct": 0.2,
                    "total_trades": 10,
                },
                strategy_stats=pd.DataFrame(
                    [{"strategy": "Combined Long Breakout", "trades": 10, "wins": 6, "losses": 4, "win_rate_pct": 60.0, "avg_return_pct": 2.1, "median_return_pct": 1.0, "payoff_ratio": 1.5, "profit_factor": 2.0, "avg_mfe_pct": 7.0, "avg_mae_pct": -2.0}]
                ),
                yearly_stats=pd.DataFrame(),
                stock_stats=pd.DataFrame(),
                trades=pd.DataFrame(),
            )
            save_trader_setup_backtest_outputs(result, data_root / "trader_setup_backtest")
            with (
                patch.dict(os.environ, {"DATA_ROOT": temp_dir}),
                patch("stock_screener.web.main.get_data_root", return_value=data_root),
            ):
                response = TestClient(app).get("/trader-setup-backtest")

        self.assertEqual(response.status_code, 200)
        self.assertIn("TRADERSETUP 10-YEAR STUDY", response.text)
        self.assertIn("Combined Long Breakout", response.text)
        self.assertIn("60.00", response.text)

    @staticmethod
    def _base_frame(periods: int) -> pd.DataFrame:
        dates = pd.bdate_range("2025-01-01", periods=periods)
        close = 100.0 + np.linspace(0.0, 8.0, periods)
        return pd.DataFrame(
            {
                "date": dates,
                "open": close - 0.2,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 100_000.0,
            }
        )


if __name__ == "__main__":
    unittest.main()
