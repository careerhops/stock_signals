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
                    "exclude_series_suffixes": [],
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

    def test_excludes_configured_nse_series_suffixes_for_mainboard_only_universe(self) -> None:
        instruments = pd.DataFrame(
            [
                {
                    "instrument_token": 1,
                    "exchange": "NSE",
                    "tradingsymbol": "ASLIND-SM",
                    "name": "ASL INDUSTRIES",
                    "instrument_type": "EQ",
                    "segment": "NSE",
                },
                {
                    "instrument_token": 2,
                    "exchange": "NSE",
                    "tradingsymbol": "MANAV-ST",
                    "name": "MANAV INFRA PROJECTS",
                    "instrument_type": "EQ",
                    "segment": "NSE",
                },
                {
                    "instrument_token": 3,
                    "exchange": "NSE",
                    "tradingsymbol": "TCS",
                    "name": "TATA CONSULTANCY SERV LT",
                    "instrument_type": "EQ",
                    "segment": "NSE",
                },
            ]
        )

        universe = build_universe(
            instruments,
            {
                "universe": {
                    "mode": "nse_all",
                    "instrument_types": ["EQ"],
                    "exclude_series_suffixes": ["-BE", "-BZ", "-BL", "-BT", "-SM", "-ST"],
                    "restrict_to_metadata_symbols": False,
                    "filters": {},
                }
            },
        )

        self.assertEqual(universe["tradingsymbol"].tolist(), ["TCS"])

    def test_blank_kite_name_uses_symbol_as_display_name_without_dropping_symbol(self) -> None:
        instruments = pd.DataFrame(
            [
                {
                    "instrument_token": 1,
                    "exchange": "NSE",
                    "tradingsymbol": "VALID",
                    "name": "",
                    "instrument_type": "EQ",
                    "segment": "NSE",
                }
            ]
        )

        universe = build_universe(
            instruments,
            {
                "universe": {
                    "mode": "nse_all",
                    "instrument_types": ["EQ"],
                    "restrict_to_metadata_symbols": False,
                    "filters": {},
                }
            },
        )

        self.assertEqual(universe["tradingsymbol"].tolist(), ["VALID"])
        self.assertEqual(universe["name"].tolist(), ["VALID"])

    def test_approximate_nse_traded_universe_keeps_equity_like_suffixes_and_excludes_debt_like_rows(self) -> None:
        instruments = pd.DataFrame(
            [
                {
                    "instrument_token": 1,
                    "exchange": "NSE",
                    "tradingsymbol": "TCS",
                    "name": "TATA CONSULTANCY SERV LT",
                    "instrument_type": "EQ",
                    "segment": "NSE",
                },
                {
                    "instrument_token": 2,
                    "exchange": "NSE",
                    "tradingsymbol": "ABC-SM",
                    "name": "ABC SME LIMITED",
                    "instrument_type": "EQ",
                    "segment": "NSE",
                },
                {
                    "instrument_token": 3,
                    "exchange": "NSE",
                    "tradingsymbol": "PGINVIT-IV",
                    "name": "POWERGRID INFRA. INVITS",
                    "instrument_type": "EQ",
                    "segment": "NSE",
                },
                {
                    "instrument_token": 4,
                    "exchange": "NSE",
                    "tradingsymbol": "ATLPP-E1",
                    "name": "ATL RE.0.50 PPD UP",
                    "instrument_type": "EQ",
                    "segment": "NSE",
                },
                {
                    "instrument_token": 5,
                    "exchange": "NSE",
                    "tradingsymbol": "A2ZINFRA-BE",
                    "name": "A2Z INFRA ENGINEERING",
                    "instrument_type": "EQ",
                    "segment": "NSE",
                },
                {
                    "instrument_token": 6,
                    "exchange": "NSE",
                    "tradingsymbol": "656KA30-SG",
                    "name": "SDL KA 6.56% 2030",
                    "instrument_type": "EQ",
                    "segment": "NSE",
                },
                {
                    "instrument_token": 7,
                    "exchange": "NSE",
                    "tradingsymbol": "850NHAI29-N5",
                    "name": "",
                    "instrument_type": "EQ",
                    "segment": "NSE",
                },
            ]
        )

        universe = build_universe(
            instruments,
            {
                "universe": {
                    "mode": "nse_all",
                    "instrument_types": ["EQ"],
                    "restrict_to_metadata_symbols": False,
                    "approximate_nse_traded_universe": {
                        "enabled": True,
                        "require_nonblank_name": True,
                        "allowed_series_suffixes": ["", "-SM", "-ST", "-BZ", "-IV", "-E1", "-P1", "-RR"],
                    },
                    "filters": {},
                }
            },
        )

        self.assertEqual(
            universe["tradingsymbol"].tolist(),
            ["ABC-SM", "ATLPP-E1", "PGINVIT-IV", "TCS"],
        )


if __name__ == "__main__":
    unittest.main()
