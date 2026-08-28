from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stock_screener.config import get_data_root, load_config
from stock_screener.data.storage import Storage
from stock_screener.knox_recovery_study import (
    DEFAULT_PROXIMITY_PCT,
    DEFAULT_RECENT_ENDPOINT_BARS,
    DEFAULT_ROUND_TRIP_COST_PCT,
    run_knox_recovery_study,
    save_knox_recovery_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure five-year recoveries after bullish Knoxville endpoints near the lower Envelope."
    )
    parser.add_argument(
        "--start-date",
        default=(pd.Timestamp.today().normalize() - pd.DateOffset(years=5)).strftime("%Y-%m-%d"),
    )
    parser.add_argument("--end-date", default="")
    parser.add_argument("--knox-lookback", type=int, default=100)
    parser.add_argument("--rsi-length", type=int, default=20)
    parser.add_argument("--momentum-length", type=int, default=20)
    parser.add_argument("--envelope-length", type=int, default=50)
    parser.add_argument("--envelope-percent", type=float, default=18.0)
    parser.add_argument("--envelope-ma-type", choices=("SMA", "EMA"), default="SMA")
    parser.add_argument("--proximity-pct", type=float, default=DEFAULT_PROXIMITY_PCT)
    parser.add_argument("--recent-endpoint-bars", type=int, default=DEFAULT_RECENT_ENDPOINT_BARS)
    parser.add_argument("--round-trip-cost-pct", type=float, default=DEFAULT_ROUND_TRIP_COST_PCT)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--required-latest-date",
        default="",
        help="CPU-only rerun using symbols whose stored candle reaches this date.",
    )
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    data_root = get_data_root(load_config())
    storage = Storage(data_root)
    requested_start = pd.Timestamp(args.start_date).date()
    selected_symbols = [value.strip().upper() for value in args.symbols.split(",") if value.strip()] or None
    refresh_audit: dict[str, object] = {}

    def progress(payload: dict[str, object]) -> None:
        completed = int(payload.get("completed") or 0)
        total = int(payload.get("total") or 0)
        if completed == total or completed % 50 == 0:
            print(
                f"{payload.get('phase', 'Running')}: {completed}/{total} "
                f"{payload.get('current_symbol', '')}",
                flush=True,
            )

    if args.refresh:
        from stock_screener.web.main import (
            _refresh_minervini_quality_benchmark,
            _refresh_trader_setup_history,
        )

        expected_date = _refresh_minervini_quality_benchmark(storage, "NIFTY 50")
        warmup_start = (pd.Timestamp(requested_start) - pd.DateOffset(years=1)).date()
        selected_symbols, refresh_audit = _refresh_trader_setup_history(
            storage,
            required_date=expected_date,
            start_date=warmup_start,
            progress_callback=progress,
        )
        end_date = expected_date
    else:
        end_date = pd.Timestamp(args.end_date).date() if args.end_date else pd.Timestamp.today().date()
        if args.required_latest_date and selected_symbols is None:
            from stock_screener.knox_envelope_pair_backtest import _candidate_symbols

            required_latest = pd.Timestamp(args.required_latest_date).normalize()
            selected_symbols = []
            for symbol in _candidate_symbols(storage, "NSE", None):
                daily = storage.load_candles("NSE", symbol, "1D")
                latest = pd.to_datetime(daily.get("date"), errors="coerce").max()
                if pd.notna(latest) and pd.Timestamp(latest).normalize() == required_latest:
                    selected_symbols.append(symbol)

    result = run_knox_recovery_study(
        storage,
        symbols=selected_symbols,
        start_date=requested_start,
        end_date=end_date,
        knox_lookback=args.knox_lookback,
        rsi_length=args.rsi_length,
        momentum_length=args.momentum_length,
        envelope_length=args.envelope_length,
        envelope_percent=args.envelope_percent,
        envelope_ma_type=args.envelope_ma_type,
        envelope_proximity_pct=args.proximity_pct,
        recent_endpoint_bars=args.recent_endpoint_bars,
        round_trip_cost_pct=args.round_trip_cost_pct,
        progress_callback=progress,
    )
    result.summary.update(refresh_audit)
    output_dir = Path(args.output_dir) if args.output_dir else data_root / "knox_recovery"
    paths = save_knox_recovery_outputs(result, output_dir)

    print("\nSummary")
    for key, value in result.summary.items():
        print(f"{key}: {value}")
    print("\nFiles")
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
