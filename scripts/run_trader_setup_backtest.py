from __future__ import annotations

import argparse
import json

import pandas as pd

from stock_screener.config import get_data_root, load_config
from stock_screener.data.storage import Storage
from stock_screener.minervini_quality_study import DEFAULT_BENCHMARK_SYMBOL
from stock_screener.trader_setup_backtest_study import (
    DEFAULT_ATR_BUFFER,
    DEFAULT_HOLDING_DAYS,
    DEFAULT_LEFT_BARS,
    DEFAULT_PROFIT_TARGET_PCT,
    DEFAULT_RIGHT_BARS,
    DEFAULT_ROUND_TRIP_COST_PCT,
    DEFAULT_RVOL_THRESHOLD,
    DEFAULT_STOP_LOSS_PCT,
    DEFAULT_VOLUME_OSC_THRESHOLD,
    run_trader_setup_backtest_study,
    save_trader_setup_backtest_outputs,
)
from stock_screener.web.main import (
    _refresh_minervini_quality_benchmark,
    _refresh_trader_setup_history,
    _trader_setup_backtest_dir,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh Kite NSE history and backtest TraderSetup signals.")
    parser.add_argument("--start-date", default=(pd.Timestamp.today() - pd.DateOffset(years=10)).strftime("%Y-%m-%d"))
    parser.add_argument("--holding-days", type=int, default=DEFAULT_HOLDING_DAYS)
    parser.add_argument("--profit-target-pct", type=float, default=DEFAULT_PROFIT_TARGET_PCT)
    parser.add_argument("--stop-loss-pct", type=float, default=DEFAULT_STOP_LOSS_PCT)
    parser.add_argument("--round-trip-cost-pct", type=float, default=DEFAULT_ROUND_TRIP_COST_PCT)
    parser.add_argument("--left-bars", type=int, default=DEFAULT_LEFT_BARS)
    parser.add_argument("--right-bars", type=int, default=DEFAULT_RIGHT_BARS)
    parser.add_argument("--volume-osc-threshold", type=float, default=DEFAULT_VOLUME_OSC_THRESHOLD)
    parser.add_argument("--rvol-threshold", type=float, default=DEFAULT_RVOL_THRESHOLD)
    parser.add_argument("--atr-buffer", type=float, default=DEFAULT_ATR_BUFFER)
    args = parser.parse_args()

    config = load_config()
    data_root = get_data_root(config)
    storage = Storage(data_root)
    start_date = pd.Timestamp(args.start_date).date()
    expected_date = _refresh_minervini_quality_benchmark(storage, DEFAULT_BENCHMARK_SYMBOL)
    print(f"Kite benchmark date: {expected_date}", flush=True)

    def refresh_progress(payload: dict[str, object]) -> None:
        completed = int(payload.get("completed") or 0)
        total = int(payload.get("total") or 0)
        if completed == 1 or completed == total or completed % 25 == 0:
            symbol = str(payload.get("current_symbol") or "")
            print(f"OHLC refresh: {completed}/{total} {symbol}", flush=True)

    symbols, refresh_audit = _refresh_trader_setup_history(
        storage,
        required_date=expected_date,
        start_date=start_date,
        progress_callback=refresh_progress,
    )
    print(json.dumps(refresh_audit, indent=2), flush=True)

    def backtest_progress(payload: dict[str, object]) -> None:
        completed = int(payload.get("completed") or 0)
        total = int(payload.get("total") or 0)
        if completed == 1 or completed == total or completed % 100 == 0:
            symbol = str(payload.get("current_symbol") or "")
            print(f"Backtest: {completed}/{total} {symbol}", flush=True)

    result = run_trader_setup_backtest_study(
        storage,
        exchange="NSE",
        symbols=symbols,
        start_date=start_date,
        end_date=expected_date,
        left_bars=max(args.left_bars, 1),
        right_bars=max(args.right_bars, 1),
        volume_osc_threshold=args.volume_osc_threshold,
        rvol_threshold=max(args.rvol_threshold, 0.0),
        atr_buffer=max(args.atr_buffer, 0.0),
        holding_days=max(args.holding_days, 1),
        profit_target_pct=max(args.profit_target_pct, 0.0),
        stop_loss_pct=max(args.stop_loss_pct, 0.0),
        round_trip_cost_pct=max(args.round_trip_cost_pct, 0.0),
        progress_callback=backtest_progress,
    )
    result.summary.update(refresh_audit)
    output_dir = _trader_setup_backtest_dir(data_root)
    save_trader_setup_backtest_outputs(result, output_dir)
    print(f"Saved results to {output_dir}", flush=True)
    print(result.strategy_stats.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
