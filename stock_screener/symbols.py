from __future__ import annotations


NSE_SERIES_SUFFIXES = ("-BE", "-BZ", "-BL", "-BT", "-SM", "-ST")


def has_nse_series_suffix(value: object, suffixes: tuple[str, ...] | list[str] = NSE_SERIES_SUFFIXES) -> bool:
    symbol = str(value or "").strip().upper()
    return any(symbol.endswith(str(suffix).upper()) for suffix in suffixes)


def normalize_nse_symbol(value: object) -> str:
    symbol = str(value or "").strip().upper()
    for suffix in NSE_SERIES_SUFFIXES:
        if symbol.endswith(suffix):
            return symbol[: -len(suffix)]
    return symbol
