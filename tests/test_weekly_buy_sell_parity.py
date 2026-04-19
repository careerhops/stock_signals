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

    def test_signal_quality_metrics_are_added_without_changing_core_signals(self) -> None:
        candles = _weekly_candles(
            [
                ("2024-01-05", 9, 10, 8, 9, 1000),
                ("2024-01-12", 10, 11, 9, 10, 1000),
                ("2024-01-19", 11, 12, 10, 11, 1000),
                ("2024-01-26", 11.5, 12.5, 10.5, 11.5, 1000),
                ("2024-02-02", 13, 14, 13, 14, 3000),
                ("2024-02-09", 8.2, 9, 7, 8, 1500),
                ("2024-02-16", 10, 11, 9, 10, 1000),
                ("2024-02-23", 11, 12, 10, 11, 1000),
                ("2024-03-01", 12, 13, 11, 12, 1000),
                ("2024-03-08", 14, 15, 14, 15, 2200),
            ]
        )
        config = _config()
        config["strategy"]["volume_confirmation_lookback"] = 3
        config["strategy"]["volume_confirmation_multiplier"] = 1.25

        result = run_weekly_buy_sell(candles, config)

        buy_row = result[result["final_buy"]].iloc[0]
        sell_row = result[result["final_sell"]].iloc[0]
        second_buy_row = result[result["final_buy"]].iloc[1]

        self.assertTrue(bool(buy_row["volume_confirmation"]))
        self.assertIn("trend_confirmation", result.columns)
        self.assertAlmostEqual(sell_row["sell_pair_return_pct"], -42.8571428571)
        self.assertAlmostEqual(second_buy_row["prior_pair_return_last_1_pct"], -42.8571428571)
        self.assertTrue(pd.isna(second_buy_row["median_pair_return_last_3_pct"]))

    def test_median_pair_return_requires_three_completed_pairs(self) -> None:
        from stock_screener.strategy.weekly_buy_sell import _add_completed_trade_return_metrics

        result = pd.DataFrame(
            [
                {"date": pd.Timestamp("2026-01-02"), "close": 100.0, "final_buy": True, "final_sell": False},
                {"date": pd.Timestamp("2026-01-09"), "close": 110.0, "final_buy": False, "final_sell": True},
                {"date": pd.Timestamp("2026-01-16"), "close": 100.0, "final_buy": True, "final_sell": False},
                {"date": pd.Timestamp("2026-01-23"), "close": 90.0, "final_buy": False, "final_sell": True},
                {"date": pd.Timestamp("2026-01-30"), "close": 100.0, "final_buy": True, "final_sell": False},
                {"date": pd.Timestamp("2026-02-06"), "close": 120.0, "final_buy": False, "final_sell": True},
                {"date": pd.Timestamp("2026-02-13"), "close": 100.0, "final_buy": True, "final_sell": False},
            ]
        )

        _add_completed_trade_return_metrics(result)
        fourth_buy = result.iloc[-1]

        self.assertAlmostEqual(fourth_buy["prior_pair_return_last_1_pct"], 20.0)
        self.assertAlmostEqual(fourth_buy["median_pair_return_last_3_pct"], 10.0)

    def test_pair_returns_ignore_completed_pairs_started_before_return_lookback(self) -> None:
        from stock_screener.strategy.weekly_buy_sell import _add_completed_trade_return_metrics

        result = pd.DataFrame(
            [
                {"date": pd.Timestamp("2023-06-02"), "close": 260.7, "final_buy": True, "final_sell": False},
                {"date": pd.Timestamp("2025-11-21"), "close": 503.45, "final_buy": False, "final_sell": True},
                {"date": pd.Timestamp("2026-04-17"), "close": 633.35, "final_buy": True, "final_sell": False},
            ]
        )
        lookback_start = result["date"].max() - pd.Timedelta(weeks=104)
        _add_completed_trade_return_metrics(result, lookback_start)

        current_buy = result[result["date"] == pd.Timestamp("2026-04-17")].iloc[0]

        self.assertTrue(pd.isna(current_buy["prior_pair_return_last_1_pct"]))
        self.assertTrue(pd.isna(current_buy["median_pair_return_last_3_pct"]))


if __name__ == "__main__":
    unittest.main()
