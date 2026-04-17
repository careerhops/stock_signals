from __future__ import annotations

from typing import Any

import pandas as pd


def _load_metadata(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()

    try:
        metadata = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()

    if "symbol" not in metadata.columns:
        return pd.DataFrame()

    metadata = metadata.copy()
    metadata["symbol"] = metadata["symbol"].astype(str).str.upper()
    return metadata


def _apply_metadata_filters(frame: pd.DataFrame, universe_cfg: dict[str, Any]) -> pd.DataFrame:
    filters_cfg = universe_cfg.get("filters", {}) or {}
    metadata_file = universe_cfg.get("metadata_file")
    restrict_to_metadata_symbols = bool(universe_cfg.get("restrict_to_metadata_symbols", False))
    stock_search = str(filters_cfg.get("stock_search", "") or "").strip().upper()
    industries = filters_cfg.get("industries", []) or []
    min_market_cap = filters_cfg.get("min_market_cap_cr")
    max_market_cap = filters_cfg.get("max_market_cap_cr")
    market_cap_bucket = filters_cfg.get("market_cap_bucket")

    if (
        not restrict_to_metadata_symbols
        and not industries
        and min_market_cap is None
        and max_market_cap is None
        and not market_cap_bucket
        and not stock_search
    ):
        return frame

    metadata = _load_metadata(metadata_file)
    if metadata.empty:
        return frame.iloc[0:0].copy()

    merged = frame.copy()
    merged["symbol_key"] = merged["tradingsymbol"].astype(str).str.upper()
    merge_type = "inner" if restrict_to_metadata_symbols else "left"
    merged = merged.merge(metadata, left_on="symbol_key", right_on="symbol", how=merge_type, suffixes=("", "_metadata"))

    if stock_search:
        exact_symbol_match = merged["tradingsymbol"].astype(str).str.upper() == stock_search
        if exact_symbol_match.any():
            merged = merged[exact_symbol_match]
        else:
            search_columns = ["tradingsymbol", "name", "company_name"]
            search_mask = pd.Series(False, index=merged.index)
            for column in search_columns:
                if column in merged.columns:
                    search_mask = search_mask | merged[column].astype(str).str.upper().str.contains(stock_search, na=False)
            merged = merged[search_mask]

    if industries:
        industry_set = {str(industry).strip().lower() for industry in industries}
        merged = merged[merged["industry"].astype(str).str.strip().str.lower().isin(industry_set)]

    if market_cap_bucket and "market_cap_bucket" in merged.columns:
        merged = merged[merged["market_cap_bucket"].astype(str).str.strip().str.lower() == str(market_cap_bucket).strip().lower()]

    if min_market_cap is not None:
        merged["market_cap_cr"] = pd.to_numeric(merged["market_cap_cr"], errors="coerce")
        merged = merged[merged["market_cap_cr"] >= float(min_market_cap)]

    if max_market_cap is not None:
        merged["market_cap_cr"] = pd.to_numeric(merged["market_cap_cr"], errors="coerce")
        merged = merged[merged["market_cap_cr"] <= float(max_market_cap)]

    return merged.drop(columns=["symbol_key"], errors="ignore")


def build_universe(instruments: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    universe_cfg = config.get("universe", {})
    mode = universe_cfg.get("mode", "configured")
    if mode == "nse_all":
        exchanges = {"NSE"}
        universe_cfg = {**universe_cfg, "allow_symbols": []}
    elif mode == "nse_bse_all":
        exchanges = {"NSE", "BSE"}
        universe_cfg = {**universe_cfg, "allow_symbols": []}
    else:
        exchanges = set(universe_cfg.get("exchanges", ["NSE", "BSE"]))

    instrument_types = set(universe_cfg.get("instrument_types", ["EQ"]))
    allow_symbols = set(universe_cfg.get("allow_symbols", []) or [])
    block_symbols = set(universe_cfg.get("block_symbols", []) or [])
    max_symbols = universe_cfg.get("max_symbols")

    if instruments.empty:
        return instruments

    frame = instruments.copy()
    frame = frame[frame["exchange"].isin(exchanges)]

    if "instrument_type" in frame.columns:
        frame = frame[frame["instrument_type"].isin(instrument_types)]

    if "segment" in frame.columns:
        frame = frame[frame["segment"].astype(str).str.upper() != "INDICES"]

    if "tradingsymbol" in frame.columns:
        frame = frame[~frame["tradingsymbol"].isin(block_symbols)]
        if allow_symbols:
            frame = frame[frame["tradingsymbol"].isin(allow_symbols)]

    frame = _apply_metadata_filters(frame, universe_cfg)
    frame = frame.sort_values(["exchange", "tradingsymbol"])

    if max_symbols:
        frame = frame.head(int(max_symbols))

    return frame.reset_index(drop=True)
