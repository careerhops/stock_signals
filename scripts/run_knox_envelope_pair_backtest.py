from __future__ import annotations

import argparse
from pathlib import Path

from stock_screener.config import get_data_root, load_config
from stock_screener.data.storage import Storage
from stock_screener.knox_envelope_pair_backtest import (
    BASELINE_PARAMETERS,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    default_search_parameters,
    run_knox_envelope_pair_backtest,
    save_knox_envelope_pair_backtest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest paired Knoxville divergence endpoints near Envelope bands."
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--proximity-pct", type=float, default=5.0)
    parser.add_argument("--target-pct", type=float, default=10.0)
    parser.add_argument("--round-trip-cost-pct", type=float, default=0.35)
    parser.add_argument("--entry-cmf-length", type=int, default=0)
    parser.add_argument("--min-entry-cmf", type=float, default=0.0)
    parser.add_argument("--min-entry-rvol20", type=float, default=None)
    parser.add_argument("--entry-obv-accumulation-days", type=int, default=0)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument(
        "--single-parameters",
        default="",
        help="Optional K,R,M,E,P values, for example 200,20,30,200,18",
    )
    parser.add_argument("--symbols", default="")
    parser.add_argument("--output-dir", default="data/knox_envelope_pair_backtest")
    args = parser.parse_args()

    symbols = [value.strip().upper() for value in args.symbols.split(",") if value.strip()] or None
    if args.single_parameters:
        from stock_screener.knox_envelope_pair_backtest import PairStrategyParameters

        values = [float(value.strip()) for value in args.single_parameters.split(",")]
        if len(values) != 5:
            parser.error("--single-parameters requires K,R,M,E,P")
        parameters = (
            PairStrategyParameters(
                int(values[0]), int(values[1]), int(values[2]), int(values[3]), values[4]
            ),
        )
    else:
        parameters = (BASELINE_PARAMETERS,) if args.baseline_only else default_search_parameters()
    storage = Storage(get_data_root(load_config()))

    def progress(payload: dict[str, object]) -> None:
        completed = int(payload.get("completed") or 0)
        total = int(payload.get("total") or 0)
        if completed == total or completed % 100 == 0:
            print(f"{completed}/{total}: {payload.get('current_symbol', '')}", flush=True)

    result = run_knox_envelope_pair_backtest(
        storage,
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        parameters=parameters,
        include_baseline=not bool(args.single_parameters),
        proximity_pct=args.proximity_pct,
        target_pct=args.target_pct,
        round_trip_cost_pct=args.round_trip_cost_pct,
        entry_cmf_length=args.entry_cmf_length or None,
        min_entry_cmf=args.min_entry_cmf,
        min_entry_rvol20=args.min_entry_rvol20,
        entry_obv_accumulation_days=args.entry_obv_accumulation_days or None,
        progress_callback=progress,
    )
    paths = save_knox_envelope_pair_backtest(result, Path(args.output_dir))
    print("\nSummary")
    for key, value in result.summary.items():
        print(f"{key}: {value}")
    if not result.parameter_stats.empty:
        print("\nTop parameter sets")
        print(result.parameter_stats.head(15).to_string(index=False))
    print("\nFiles")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
