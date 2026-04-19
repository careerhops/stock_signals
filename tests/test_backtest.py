from __future__ import annotations

import unittest

import pandas as pd

from stock_screener.backtest import closed_trades_from_strategy, overall_summary, stock_level_stats


class BuySellBacktestTests(unittest.TestCase):
    def test_pairs_buy_with_next_sell_and_tracks_open_buy(self) -> None:
        strategy_output = pd.DataFrame(
            [
                {"date": "2026-01-02", "close": 100.0, "high": 101.0, "final_buy": True, "final_sell": False},
                {"date": "2026-01-09", "close": 112.0, "high": 118.0, "final_buy": False, "final_sell": True},
                {"date": "2026-01-16", "close": 200.0, "high": 202.0, "final_buy": True, "final_sell": False},
                {"date": "2026-01-23", "close": 180.0, "high": 205.0, "final_buy": False, "final_sell": True},
                {"date": "2026-01-30", "close": 150.0, "high": 152.0, "final_buy": True, "final_sell": False},
            ]
        )

        trades, open_position = closed_trades_from_strategy(
            strategy_output,
            exchange="NSE",
            symbol="TEST",
            name="Test Ltd",
        )

        self.assertEqual(len(trades), 2)
        self.assertEqual(trades["outcome"].tolist(), ["GAIN", "LOSS"])
        self.assertAlmostEqual(trades.iloc[0]["return_pct"], 12.0)
        self.assertAlmostEqual(trades.iloc[1]["return_pct"], -10.0)
        self.assertAlmostEqual(trades.iloc[0]["max_gain_before_sell_pct"], 18.0)
        self.assertAlmostEqual(trades.iloc[1]["max_gain_before_sell_pct"], 2.5)
        self.assertTrue(bool(trades.iloc[0]["hit_10pct_before_sell"]))
        self.assertFalse(bool(trades.iloc[1]["hit_10pct_before_sell"]))
        self.assertIsNotNone(open_position)
        assert open_position is not None
        self.assertEqual(open_position["symbol"], "TEST")
        self.assertAlmostEqual(open_position["open_return_pct"], 0.0)

    def test_summary_counts_wins_losses_and_breakeven_only_for_closed_trades(self) -> None:
        trades = pd.DataFrame(
            [
                {"exchange": "NSE", "symbol": "A", "name": "A Ltd", "return_pct": 10.0},
                {"exchange": "NSE", "symbol": "A", "name": "A Ltd", "return_pct": -5.0},
                {"exchange": "NSE", "symbol": "B", "name": "B Ltd", "return_pct": 0.0},
            ]
        )
        open_positions = pd.DataFrame([{"exchange": "NSE", "symbol": "C"}])

        summary = overall_summary(
            trades,
            open_positions,
            exchange="NSE",
            symbols_processed=3,
            symbols_with_closed_trades=2,
        )
        stats = stock_level_stats(trades)

        self.assertEqual(summary["closed_trades"], 3)
        self.assertEqual(summary["winning_trades"], 1)
        self.assertEqual(summary["losing_trades"], 1)
        self.assertEqual(summary["breakeven_trades"], 1)
        self.assertEqual(summary["open_positions"], 1)
        self.assertAlmostEqual(summary["win_rate_pct"], 33.33333333333333)
        self.assertEqual(set(stats["symbol"]), {"A", "B"})


if __name__ == "__main__":
    unittest.main()
