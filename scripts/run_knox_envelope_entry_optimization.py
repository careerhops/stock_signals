from __future__ import annotations

import argparse
from pathlib import Path

from stock_screener.config import get_data_root, load_config
from stock_screener.data.storage import Storage
from stock_screener.knox_envelope_entry_optimization import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    run_knox_envelope_entry_optimization,
    save_entry_optimization,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize Knoxville, lower Envelope, and CMF for fixed-horizon NSE entries."
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--round-trip-cost-pct", type=float, default=0.35)
    parser.add_argument("--minimum-validation-trades", type=int, default=100)
    parser.add_argument("--symbols", default="")
    parser.add_argument(
        "--output-dir",
        default="data/knox_envelope_entry_optimization",
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

    result = run_knox_envelope_entry_optimization(
        storage,
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        horizon=args.horizon,
        round_trip_cost_pct=args.round_trip_cost_pct,
        minimum_validation_trades=args.minimum_validation_trades,
        progress_callback=progress,
    )
    paths = save_entry_optimization(result, Path(args.output_dir))
    print("\nSummary")
    for key, value in result.summary.items():
        print(f"{key}: {value}")
    print("\nFiles")
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
