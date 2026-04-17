from __future__ import annotations

import unittest

import pandas as pd

from stock_screener.strategy.weekly_buy_sell import run_weekly_buy_sell


def _config(sensitivity: int = 3, fvg_lookback: int = 5) -> dict:
    return {
        "strategy": {
            "sensitivity": sensitivity,
            "fvg_lookback": fvg_lookback,
            "prevent_repeated_direction": True,
        }
    }


def _weekly_candles(rows: list[tuple[str, float, float, float, float, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])


class WeeklyBuySellParityTests(unittest.TestCase):
    def test_buy_and_sell_match_pine_break_fvg_and_exclusivity_rules(self) -> None:
        candles = _weekly_candles(
            [
                ("2024-01-05", 9, 10, 8, 9, 1000),
                ("2024-01-12", 10, 11, 9, 10, 1100),
                ("2024-01-19", 11, 12, 10, 11, 1200),
                ("2024-01-26", 11.5, 12.5, 10.5, 11.5, 1300),
                ("2024-02-02", 13, 14, 13, 14, 1400),
                ("2024-02-09", 8.2, 9, 7, 8, 1500),
            ]
        )

        result = run_weekly_buy_sell(candles, _config())
        signals = result[result["signal"].isin(["BUY", "SELL"])][["date", "signal"]].copy()
        signals["date"] = signals["date"].dt.strftime("%Y-%m-%d")

        self.assertEqual(
            signals.to_dict(orient="records"),
            [
                {"date": "2024-02-02", "signal": "BUY"},
                {"date": "2024-02-09", "signal": "SELL"},
            ],
        )

    def test_breakout_uses_previous_structure_levels_to_avoid_repaint(self) -> None:
        candles = _weekly_candles(
            [
                ("2024-01-05", 9, 10, 8, 9, 1000),
                ("2024-01-12", 10, 11, 9, 10, 1000),
                ("2024-01-19", 11, 12, 10, 11, 1000),
                ("2024-01-26", 11.5, 12.5, 10.5, 11.5, 1000),
                ("2024-02-02", 13, 14, 13, 14, 1000),
            ]
        )

        result = run_weekly_buy_sell(candles, _config())
        buy_row = result.iloc[4]

        self.assertEqual(buy_row["upper_level"], 12.5)
        self.assertTrue(bool(buy_row["bull_break"]))
        self.assertTrue(bool(buy_row["final_buy"]))

    def test_repeated_buy_is_suppressed_until_sell_occurs(self) -> None:
        candles = _weekly_candles(
            [
                ("2024-01-05", 9, 10, 8, 9, 1000),
                ("2024-01-12", 10, 11, 9, 10, 1000),
                ("2024-01-19", 11, 12, 10, 11, 1000),
                ("2024-01-26", 11.5, 12.5, 10.5, 11.5, 1000),
                ("2024-02-02", 13, 14, 13, 14, 1000),
                ("2024-02-09", 13, 14.5, 12, 13, 1000),
                ("2024-02-16", 14.8, 15, 14.6, 15, 1000),
            ]
        )

        result = run_weekly_buy_sell(candles, _config())

        self.assertTrue(bool(result.iloc[4]["buy_signal"]))
        self.assertTrue(bool(result.iloc[4]["final_buy"]))
        self.assertTrue(bool(result.iloc[6]["buy_signal"]))
        self.assertFalse(bool(result.iloc[6]["final_buy"]))

    def test_fvg_must_exist_inside_configured_lookback_window(self) -> None:
        candles = _weekly_candles(
            [
                ("2024-01-05", 9, 10, 8, 9, 1000),
                ("2024-01-12", 10, 11, 9, 10, 1000),
                ("2024-01-19", 11, 12, 10, 11, 1000),
                ("2024-01-26", 11.5, 12.5, 10.5, 11.5, 1000),
                ("2024-02-02", 12, 14, 11, 14, 1000),
            ]
        )

        result = run_weekly_buy_sell(candles, _config(fvg_lookback=1))

        self.assertTrue(bool(result.iloc[4]["bull_break"]))
        self.assertEqual(result.iloc[4]["fvg_bull_recent"], 0)
        self.assertFalse(bool(result.iloc[4]["final_buy"]))


if __name__ == "__main__":
    unittest.main()
