from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stock_screener.config import load_config
from stock_screener.resample import resample_daily_to_weekly
from stock_screener.strategy.weekly_buy_sell import run_weekly_buy_sell


SIGNAL_COLUMNS = ("expected_signal", "tv_signal", "signal")


def _read_input(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    normalized_columns = {column: column.strip().lower() for column in frame.columns}
    frame = frame.rename(columns=normalized_columns)

    if "time" in frame.columns and "date" not in frame.columns:
        frame = frame.rename(columns={"time": "date"})

    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required OHLC columns: {', '.join(sorted(missing))}")

    return frame


def _expected_signals(frame: pd.DataFrame) -> pd.DataFrame:
    signal_column = next((column for column in SIGNAL_COLUMNS if column in frame.columns), None)
    if not signal_column:
        raise ValueError(
            "Expected one TradingView signal column named one of: "
            + ", ".join(SIGNAL_COLUMNS)
        )

    expected = frame[["date", signal_column]].copy()
    expected = expected.rename(columns={signal_column: "signal"})
    expected["date"] = pd.to_datetime(expected["date"]).dt.strftime("%Y-%m-%d")
    expected["signal"] = expected["signal"].astype(str).str.upper().str.strip()
    expected = expected[expected["signal"].isin(["BUY", "SELL"])]
    return expected.sort_values(["date", "signal"]).reset_index(drop=True)


def _python_signals(frame: pd.DataFrame, timeframe: str, config: dict) -> pd.DataFrame:
    candles = frame[["date", "open", "high", "low", "close", "volume"]].copy()
    if timeframe == "daily":
        strategy_cfg = config.get("strategy", {})
        candles = resample_daily_to_weekly(
            candles,
            strategy_cfg.get("weekly_anchor", "W-FRI"),
            bool(strategy_cfg.get("use_completed_weeks_only", True)),
        )

    output = run_weekly_buy_sell(candles, config)
    signals = output[output["signal"].isin(["BUY", "SELL"])][["date", "signal"]].copy()
    signals["date"] = pd.to_datetime(signals["date"]).dt.strftime("%Y-%m-%d")
    return signals.sort_values(["date", "signal"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare TradingView exported signals with Python strategy output.")
    parser.add_argument("csv", help="CSV with OHLCV and expected_signal/tv_signal/signal column.")
    parser.add_argument("--timeframe", choices=["weekly", "daily"], default="weekly")
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    frame = _read_input(Path(args.csv))
    expected = _expected_signals(frame)
    actual = _python_signals(frame, args.timeframe, config)

    expected_pairs = set(map(tuple, expected[["date", "signal"]].to_records(index=False)))
    actual_pairs = set(map(tuple, actual[["date", "signal"]].to_records(index=False)))
    missing = sorted(expected_pairs - actual_pairs)
    extra = sorted(actual_pairs - expected_pairs)

    print(f"TradingView signals: {len(expected_pairs)}")
    print(f"Python signals:      {len(actual_pairs)}")

    if not missing and not extra:
        print("MATCH: TradingView and Python signals are identical.")
        return

    if missing:
        print("\nMissing in Python:")
        for date, signal in missing:
            print(f"  {date} {signal}")

    if extra:
        print("\nExtra in Python:")
        for date, signal in extra:
            print(f"  {date} {signal}")

    raise SystemExit(1)


if __name__ == "__main__":
    main()
