from __future__ import annotations

import argparse
from pathlib import Path

from stock_screener.config import get_data_root, load_config
from stock_screener.data.nse_market_cap import (
    DEFAULT_NSE_MARKET_CAP_URL,
    fetch_market_caps_from_nse_excel,
    load_nse_market_cap_excel,
)
from stock_screener.data.storage import Storage


def main() -> None:
    parser = argparse.ArgumentParser(description="Import NSE average market-cap Excel into symbol metadata.")
    parser.add_argument(
        "--input",
        default="",
        help="Optional local .xlsx path. If omitted, downloads the NSE archive URL from config/settings.yaml.",
    )
    args = parser.parse_args()

    config = load_config()
    data_root = get_data_root(config)
    storage = Storage(data_root)
    universe_cfg = config.get("universe", {})
    market_cap_cfg = universe_cfg.get("market_cap_source", {})
    bucket_cfg = universe_cfg.get("market_cap_buckets", {})
    small_max_cr = float(bucket_cfg.get("small_max_cr", 5000))
    mid_max_cr = float(bucket_cfg.get("mid_max_cr", 20000))
    market_cap_divisor = market_cap_cfg.get("market_cap_divisor")
    market_cap_divisor = float(market_cap_divisor) if market_cap_divisor else None

    if args.input:
        workbook_path = Path(args.input)
        metadata = load_nse_market_cap_excel(workbook_path, small_max_cr, mid_max_cr, market_cap_divisor)
    else:
        local_path = Path(str(market_cap_cfg.get("local_path", "")))
        if not local_path.is_absolute():
            local_path = Path.cwd() / local_path
        if local_path.exists():
            workbook_path = local_path
            metadata = load_nse_market_cap_excel(workbook_path, small_max_cr, mid_max_cr, market_cap_divisor)
        else:
            source_url = market_cap_cfg.get("url", DEFAULT_NSE_MARKET_CAP_URL)
            local_file = market_cap_cfg.get("local_file", "Average_MCAP_July2025ToDecember2025_20260102201101.xlsx")
            workbook_path = data_root / "instruments" / local_file
            metadata = fetch_market_caps_from_nse_excel(
                source_url,
                workbook_path,
                small_max_cr,
                mid_max_cr,
                market_cap_divisor,
            )

    storage.save_symbol_metadata(metadata)
    output_path = storage.symbol_metadata_path()
    print(f"Saved {len(metadata)} market-cap rows to {output_path}")
    print(metadata[["symbol", "company_name", "industry", "market_cap_cr", "market_cap_bucket"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
