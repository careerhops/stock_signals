from __future__ import annotations

import argparse
from pathlib import Path

from stock_screener.config import get_data_root, load_config
from stock_screener.data.storage import Storage
from stock_screener.knox_minervini_backtest_study import (
    DEFAULT_END_DATE,
    DEFAULT_EXIT_VARIANTS,
    DEFAULT_START_DATE,
    ExitVariant,
    run_knox_minervini_backtest,
    save_knox_minervini_backtest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Research Knoxville + Envelope pullbacks inside a Minervini trend."
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--benchmark", default="NIFTY 50")
    parser.add_argument("--ignore-minervini", action="store_true")
    parser.add_argument("--compact-exits", action="store_true")
    parser.add_argument("--output-dir", default="data/knox_minervini_backtest")
    args = parser.parse_args()

    storage = Storage(get_data_root(load_config()))

    def progress(payload: dict[str, object]) -> None:
        completed = int(payload.get("completed") or 0)
        total = int(payload.get("total") or 0)
        if completed == total or completed % 100 == 0:
            print(f"{completed}/{total}: {payload.get('current_symbol', '')}", flush=True)

    compact_exits = tuple(
        ExitVariant(target, stop, 10)
        for target in (5.0, 6.0, 7.0)
        for stop in (3.0, 4.0)
    )
    result = run_knox_minervini_backtest(
        storage,
        start_date=args.start_date,
        end_date=args.end_date,
        benchmark_symbol=args.benchmark,
        use_minervini_filter=not args.ignore_minervini,
        exit_variants=compact_exits if args.compact_exits else DEFAULT_EXIT_VARIANTS,
        progress_callback=progress,
    )
    paths = save_knox_minervini_backtest(result, Path(args.output_dir))
    print(result.summary)
    if not result.variant_stats.empty:
        print(result.variant_stats.head(15).to_string(index=False))
    print({name: str(path) for name, path in paths.items()})


if __name__ == "__main__":
    main()
