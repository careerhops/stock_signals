from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from stock_screener.universe import build_universe


class UniverseTests(unittest.TestCase):
    def test_nse_series_suffix_matches_base_metadata_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metadata_path = Path(directory) / "symbol_metadata.csv"
            pd.DataFrame(
                [
                    {
                        "symbol": "E2E",
                        "company_name": "E2E Networks Limited",
                        "market_cap_cr": 5239.77,
                    }
                ]
            ).to_csv(metadata_path, index=False)

            instruments = pd.DataFrame(
                [
                    {
                        "instrument_token": 2288641,
                        "exchange": "NSE",
                        "tradingsymbol": "E2E-BE",
                        "name": "E2E NETWORKS",
                        "instrument_type": "EQ",
                        "segment": "NSE",
                    }
                ]
            )
            config = {
                "universe": {
                    "mode": "nse_all",
                    "instrument_types": ["EQ"],
                    "restrict_to_metadata_symbols": True,
                    "metadata_file": str(metadata_path),
                    "filters": {
                        "stock_search": "E2E",
                    },
                }
            }

            universe = build_universe(instruments, config)

        self.assertEqual(universe["tradingsymbol"].tolist(), ["E2E-BE"])
        self.assertEqual(universe["symbol"].tolist(), ["E2E"])


if __name__ == "__main__":
    unittest.main()
