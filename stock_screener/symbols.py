from __future__ import annotations

import re

import pandas as pd


NSE_SERIES_SUFFIXES = ("-BE", "-BZ", "-BL", "-BT", "-SM", "-ST")
NSE_TRADED_ALLOWED_SUFFIXES = ("", "-SM", "-ST", "-BZ", "-IV", "-E1", "-P1", "-RR")
_NSE_DEBT_LIKE_SYMBOL_RE = re.compile(
    r"-SG|-GB|-N[0-9A-Z]+$|-Y[0-9A-Z]+$|-Z[0-9A-Z]+$|-A[0-9A-Z]+$|-P[0-9A-Z]+$|-W$|-NV$|-YW$"
)


def has_nse_series_suffix(value: object, suffixes: tuple[str, ...] | list[str] = NSE_SERIES_SUFFIXES) -> bool:
    symbol = str(value or "").strip().upper()
    return any(symbol.endswith(str(suffix).upper()) for suffix in suffixes)


def normalize_nse_symbol(value: object) -> str:
    symbol = str(value or "").strip().upper()
    for suffix in NSE_SERIES_SUFFIXES:
        if symbol.endswith(suffix):
            return symbol[: -len(suffix)]
    return symbol


def nse_series_suffix(value: object) -> str:
    symbol = str(value or "").strip().upper()
    match = re.search(r"(-[A-Z0-9]+)$", symbol)
    return match.group(1) if match else ""


def is_nse_debt_like_symbol(value: object) -> bool:
    symbol = str(value or "").strip().upper()
    return bool(_NSE_DEBT_LIKE_SYMBOL_RE.search(symbol))


def is_nse_traded_equity_style(
    symbol: object,
    name: object,
    allowed_suffixes: tuple[str, ...] | list[str] = NSE_TRADED_ALLOWED_SUFFIXES,
) -> bool:
    normalized_name = str(name or "").strip()
    if not normalized_name:
        return False
    if is_nse_debt_like_symbol(symbol):
        return False
    return nse_series_suffix(symbol) in {str(suffix).upper() for suffix in allowed_suffixes}


def is_nse_traded_equity_style_series(
    symbols: pd.Series,
    names: pd.Series,
    allowed_suffixes: tuple[str, ...] | list[str] = NSE_TRADED_ALLOWED_SUFFIXES,
) -> pd.Series:
    normalized_symbols = symbols.fillna("").astype(str).str.strip().str.upper()
    normalized_names = names.fillna("").astype(str).str.strip()
    suffixes = normalized_symbols.str.extract(r"(-[A-Z0-9]+)$", expand=False).fillna("")
    allowed_suffix_set = {str(suffix).upper() for suffix in allowed_suffixes}
    has_name = normalized_names.ne("")
    is_not_debt_like = ~normalized_symbols.str.contains(_NSE_DEBT_LIKE_SYMBOL_RE, regex=True, na=False)
    has_allowed_suffix = suffixes.isin(allowed_suffix_set)
    return has_name & is_not_debt_like & has_allowed_suffix
