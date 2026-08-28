from __future__ import annotations

import argparse
from pathlib import Path

from stock_screener.config import get_data_root, load_config
from stock_screener.data.storage import Storage
from stock_screener.knox_envelope_reversal_optimization import (
    run_knox_envelope_reversal_optimization,
    save_reversal_optimization,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find a fast reversal confirmation for Knoxville and lower Envelope setups."
    )
    parser.add_argument("--start-date", default="2016-08-20")
    parser.add_argument("--end-date", default="2026-08-20")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--round-trip-cost-pct", type=float, default=0.35)
    parser.add_argument("--minimum-validation-trades", type=int, default=100)
    parser.add_argument("--symbols", default="")
    parser.add_argument(
        "--output-dir",
        default="data/knox_envelope_reversal_optimization",
    )
    args = parser.parse_args()
    symbols = [value.strip().upper() for value in args.symbols.split(",") if value.strip()] or None
    storage = Storage(get_data_root(load_config()))

    def progress(payload: dict[str, object]) -> None:
        completed = int(payload.get("completed") or 0)
        total = int(payload.get("total") or 0)
        if completed == total or completed % 100 == 0:
            print(
                f"{payload.get('stage', '')}: {completed}/{total} "
                f"{payload.get('current_symbol', '')}",
                flush=True,
            )

    result = run_knox_envelope_reversal_optimization(
        storage,
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        horizon=args.horizon,
        round_trip_cost_pct=args.round_trip_cost_pct,
        minimum_validation_trades=args.minimum_validation_trades,
        progress_callback=progress,
    )
    paths = save_reversal_optimization(result, Path(args.output_dir))
    print("\nSummary")
    for key, value in result.summary.items():
        print(f"{key}: {value}")
    print("\nFiles")
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
