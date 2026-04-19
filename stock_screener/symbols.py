from __future__ import annotations


NSE_SERIES_SUFFIXES = ("-BE", "-BZ", "-BL", "-BT", "-SM", "-ST")


def normalize_nse_symbol(value: object) -> str:
    symbol = str(value or "").strip().upper()
    for suffix in NSE_SERIES_SUFFIXES:
        if symbol.endswith(suffix):
            return symbol[: -len(suffix)]
    return symbol
