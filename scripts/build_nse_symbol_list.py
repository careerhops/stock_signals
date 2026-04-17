from __future__ import annotations

import argparse
from pathlib import Path

from stock_screener.auth.kite_token import load_access_token
from stock_screener.config import get_data_root, load_config
from stock_screener.data.kite import KiteDataProvider
from stock_screener.universe import build_universe


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an NSE symbol list from Kite instruments.")
    parser.add_argument("--output", default="data/instruments/nse_symbols.csv")
    parser.add_argument("--mode", default="nse_all", choices=["nse_all", "nse_bse_all", "configured"])
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=0,
        help="Limit output for testing. Default 0 means all symbols.",
    )
    args = parser.parse_args()

    config = load_config()
    config["universe"]["mode"] = args.mode
    if args.max_symbols and args.max_symbols > 0:
        config["universe"]["max_symbols"] = args.max_symbols
    else:
        config["universe"]["max_symbols"] = None

    data_root = get_data_root(config)
    access_token = load_access_token(data_root)
    if not access_token:
        raise RuntimeError("Kite access token not found. Start dashboard and use Kite Login first.")

    provider = KiteDataProvider(access_token=access_token)
    instruments = provider.instruments()
    universe = build_universe(instruments, config)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    universe.to_csv(output, index=False)

    print(f"Saved {len(universe)} symbols to {output}")
    print(universe[["exchange", "tradingsymbol", "name"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
