from __future__ import annotations

import argparse

import pandas as pd

from stock_screener.config import get_data_root, load_config
from stock_screener.data.storage import Storage
from stock_screener.resample import resample_daily_to_weekly
from stock_screener.strategy.weekly_buy_sell import run_weekly_buy_sell


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug strategy calculations for one symbol.")
    parser.add_argument("--exchange", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--tail", type=int, default=30)
    args = parser.parse_args()

    config = load_config()
    storage = Storage(get_data_root(config))
    daily = storage.load_candles(args.exchange, args.symbol, "1D")

    scan_timeframe = config.get("data", {}).get("scan_timeframe", "1W")
    strategy_cfg = config.get("strategy", {})
    weekly_anchor = strategy_cfg.get("weekly_anchor", "W-FRI")
    use_completed_weeks_only = bool(strategy_cfg.get("use_completed_weeks_only", True))

    strategy_input = daily
    if scan_timeframe == "1W":
        strategy_input = resample_daily_to_weekly(daily, weekly_anchor, use_completed_weeks_only)

    output = run_weekly_buy_sell(strategy_input, config)
    columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "upper_level",
        "lower_level",
        "bull_break",
        "bear_break",
        "fvg_bull",
        "fvg_bear",
        "fvg_bull_recent",
        "fvg_bear_recent",
        "buy_signal",
        "sell_signal",
        "final_buy",
        "final_sell",
        "signal",
    ]

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    print(output[columns].tail(args.tail).to_string(index=False))


if __name__ == "__main__":
    main()
