from __future__ import annotations

from pathlib import Path

from stock_screener.config import get_data_root, load_config
from stock_screener.data.storage import Storage
from stock_screener.knox_alpha_filter_research import (
    run_knox_alpha_filter_research,
    save_knox_alpha_filter_research,
)


def main() -> None:
    storage = Storage(get_data_root(load_config()))
    output_dir = Path("data/knox_alpha_filter_research")
    ranking, stats = run_knox_alpha_filter_research(
        storage,
        Path("data/knox_envelope_pair_backtest"),
    )
    paths = save_knox_alpha_filter_research(ranking, stats, output_dir)
    print(ranking.to_string(index=False))
    print({name: str(path) for name, path in paths.items()})


if __name__ == "__main__":
    main()
