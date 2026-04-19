from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import quote
from uuid import uuid4

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from kiteconnect import KiteConnect

from stock_screener.auth.kite_token import load_access_token, save_access_token, token_status
from stock_screener.backtest import run_buy_sell_backtest, save_backtest_outputs
from stock_screener.backtest_report import write_backtest_workbook
from stock_screener.config import get_data_root, load_config, require_env
from stock_screener.data.kite import KiteDataProvider
from stock_screener.data.nse_market_cap import (
    DEFAULT_NSE_MARKET_CAP_URL,
    fetch_market_caps_from_nse_excel,
    load_nse_market_cap_excel,
)
from stock_screener.data.storage import Storage
from stock_screener.data.supabase_store import SupabaseStore
from stock_screener.gtt_gain_report import write_gtt_gain_workbook
from stock_screener.gtt_gain_study import load_gtt_gain_outputs, run_gtt_gain_study, save_gtt_gain_outputs
from stock_screener.jobs.daily_scan import run_daily_scan
from stock_screener.notifications.telegram import send_buy_signal_list_to_telegram, send_gtt_stock_list_to_telegram
from stock_screener.resample import resample_daily_to_weekly
from stock_screener.signal_qa import build_signal_quality_report, strategy_rows_for_display
from stock_screener.strategy.weekly_buy_sell import run_weekly_buy_sell
from stock_screener.symbols import normalize_nse_symbol
from stock_screener.universe import build_universe
from stock_screener.jobs.large_deals import (
    default_last_7_days_range,
    fetch_and_store_current_large_deals,
)
from stock_screener.web.charts import build_signal_chart, latest_signal_summary


app = FastAPI(title="NSE/BSE Investment Signal Screener")

BASE_DIR = Path(__file__).resolve().parents[2]
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def _template_number(value: Any, digits: int = 2) -> str:
    if value is None or value == "":
        return ""
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.{int(digits)}f}"
    except (TypeError, ValueError):
        return str(value)


templates.env.filters["number"] = _template_number

SCAN_JOBS: dict[str, dict[str, Any]] = {}
SCAN_JOBS_LOCK = Lock()


def _set_scan_job(job_id: str, **updates: Any) -> None:
    with SCAN_JOBS_LOCK:
        current = SCAN_JOBS.setdefault(job_id, {})
        current.update(updates)


def _get_scan_job(job_id: str) -> dict[str, Any]:
    with SCAN_JOBS_LOCK:
        return dict(SCAN_JOBS.get(job_id, {}))


def _is_allowed(request: Request) -> bool:
    expected = os.getenv("DASHBOARD_TOKEN")
    if not expected:
        return True
    return request.query_params.get("token") == expected


def _load_symbol_metadata(config: dict) -> pd.DataFrame:
    metadata_file = config.get("universe", {}).get("metadata_file", "config/symbol_metadata.csv")
    path = BASE_DIR / metadata_file
    if not path.exists():
        return pd.DataFrame()

    metadata = pd.read_csv(path)
    if metadata.empty or "symbol" not in metadata.columns:
        return pd.DataFrame()

    metadata = metadata.copy()
    metadata["symbol"] = metadata["symbol"].astype(str).str.upper()
    if "market_cap_cr" in metadata.columns:
        metadata["market_cap_cr"] = pd.to_numeric(metadata["market_cap_cr"], errors="coerce")
    return metadata


def _combined_symbol_metadata(config: dict, storage: Storage) -> pd.DataFrame:
    metadata_frames = []
    config_metadata = _load_symbol_metadata(config)
    stored_metadata = storage.load_symbol_metadata()

    if not config_metadata.empty:
        metadata_frames.append(config_metadata)
    if not stored_metadata.empty:
        metadata_frames.append(stored_metadata)

    if not metadata_frames:
        return pd.DataFrame()

    metadata = pd.concat(metadata_frames, ignore_index=True)
    if metadata.empty or "symbol" not in metadata.columns:
        return pd.DataFrame()

    metadata = metadata.copy()
    metadata["symbol"] = metadata["symbol"].astype(str).str.upper()
    if "market_cap_cr" in metadata.columns:
        metadata["market_cap_cr"] = pd.to_numeric(metadata["market_cap_cr"], errors="coerce")
    if "free_float_market_cap_cr" in metadata.columns:
        metadata["free_float_market_cap_cr"] = pd.to_numeric(
            metadata["free_float_market_cap_cr"],
            errors="coerce",
        )
    return metadata.drop_duplicates(subset=["symbol"], keep="last")


def _enrich_with_symbol_metadata(frame: pd.DataFrame, metadata: pd.DataFrame, symbol_column: str) -> pd.DataFrame:
    if frame.empty or metadata.empty or symbol_column not in frame.columns:
        return frame

    enriched = frame.copy()
    metadata_for_merge = metadata.copy()
    metadata_for_merge["metadata_symbol_key"] = metadata_for_merge["symbol"].apply(normalize_nse_symbol)
    metadata_for_merge = metadata_for_merge.drop(columns=["symbol"], errors="ignore")
    enriched["symbol_key"] = enriched[symbol_column].apply(normalize_nse_symbol)
    enriched = enriched.merge(metadata_for_merge, left_on="symbol_key", right_on="metadata_symbol_key", how="left")
    return enriched.drop(columns=["symbol_key", "metadata_symbol_key"], errors="ignore")


def _request_float(request: Request, name: str) -> float | None:
    value = request.query_params.get(name, "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _request_bool(request: Request, name: str) -> bool:
    return request.query_params.get(name, "").strip().lower() in {"1", "true", "on", "yes"}


def _optional_float(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _apply_market_cap_filters(
    frame: pd.DataFrame,
    min_market_cap: float | None,
    max_market_cap: float | None,
    market_cap_bucket: str,
) -> pd.DataFrame:
    if frame.empty:
        return frame

    filtered = frame.copy()
    if market_cap_bucket and market_cap_bucket != "All" and "market_cap_bucket" in filtered.columns:
        filtered = filtered[filtered["market_cap_bucket"] == market_cap_bucket]

    if min_market_cap is not None and "market_cap_cr" in filtered.columns:
        filtered = filtered[pd.to_numeric(filtered["market_cap_cr"], errors="coerce") >= min_market_cap]

    if max_market_cap is not None and "market_cap_cr" in filtered.columns:
        filtered = filtered[pd.to_numeric(filtered["market_cap_cr"], errors="coerce") <= max_market_cap]

    return filtered


def _truthy_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _apply_signal_quality_filters(
    frame: pd.DataFrame,
    require_volume_confirmation: bool,
    require_trend_confirmation: bool,
    return_metric: str,
    min_pair_return: float | None,
) -> pd.DataFrame:
    if frame.empty:
        return frame

    filtered = frame.copy()
    if require_volume_confirmation:
        if "volume_confirmation" not in filtered.columns:
            return filtered.iloc[0:0].copy()
        filtered = filtered[_truthy_series(filtered["volume_confirmation"])]

    if require_trend_confirmation:
        if "trend_confirmation" not in filtered.columns:
            return filtered.iloc[0:0].copy()
        filtered = filtered[_truthy_series(filtered["trend_confirmation"])]

    if min_pair_return is not None:
        metric_column = (
            "prior_pair_return_last_1_pct"
            if return_metric == "last_1"
            else "median_pair_return_last_3_pct"
        )
        if metric_column not in filtered.columns:
            return filtered.iloc[0:0].copy()
        filtered = filtered[pd.to_numeric(filtered[metric_column], errors="coerce") >= min_pair_return]

    return filtered


def _signal_quality_filter_warning(
    frame: pd.DataFrame,
    require_volume_confirmation: bool,
    require_trend_confirmation: bool,
    min_pair_return: float | None,
) -> str:
    missing_columns = []
    if require_volume_confirmation and "volume_confirmation" not in frame.columns:
        missing_columns.append("volume confirmation")
    if require_trend_confirmation and "trend_confirmation" not in frame.columns:
        missing_columns.append("trend confirmation")
    if min_pair_return is not None and not {
        "prior_pair_return_last_1_pct",
        "median_pair_return_last_3_pct",
    }.intersection(frame.columns):
        missing_columns.append("BUY-to-SELL return history")
    if not missing_columns:
        return ""
    return (
        "Signal quality columns are missing from the saved BUY list. "
        "Run the Weekly BUY Screener once more so these new fields are written."
    )


def _apply_stock_search(frame: pd.DataFrame, stock_search: str) -> pd.DataFrame:
    stock_search = stock_search.strip().upper()
    if frame.empty or not stock_search:
        return frame

    filtered = frame.copy()
    symbol_column = _symbol_column(filtered)
    if symbol_column:
        exact_symbol_match = filtered[symbol_column].astype(str).str.upper() == stock_search
        if exact_symbol_match.any():
            return filtered[exact_symbol_match]

    search_mask = pd.Series(False, index=filtered.index)
    for column in ("symbol", "tradingsymbol", "name", "company_name"):
        if column in filtered.columns:
            search_mask = search_mask | filtered[column].astype(str).str.upper().str.contains(stock_search, na=False)
    return filtered[search_mask]


def _apply_gtt_stock_filters(
    frame: pd.DataFrame,
    open_buy_regime_only: bool,
    trend_only: bool,
    dashboard_buy_only: bool = False,
    dashboard_buy_symbols: set[str] | None = None,
    fresh_weekly_buy_only: bool = False,
) -> pd.DataFrame:
    if frame.empty:
        return frame

    filtered = frame.copy()
    if open_buy_regime_only:
        if "is_latest_signal_buy" not in filtered.columns:
            return filtered.iloc[0:0].copy()
        filtered = filtered[_truthy_series(filtered["is_latest_signal_buy"])]

    if dashboard_buy_only:
        filtered = _filter_by_symbols(filtered, dashboard_buy_symbols or set())

    if fresh_weekly_buy_only:
        if "latest_week_signal" not in filtered.columns:
            return filtered.iloc[0:0].copy()
        filtered = filtered[filtered["latest_week_signal"].astype(str).str.upper() == "BUY"]

    if trend_only:
        required = {"close_above_ema20", "ema20_above_ema50"}
        if not required.issubset(filtered.columns):
            return filtered.iloc[0:0].copy()
        filtered = filtered[
            _truthy_series(filtered["close_above_ema20"])
            & _truthy_series(filtered["ema20_above_ema50"])
        ]

    return filtered


def _gtt_filter_warning(
    frame: pd.DataFrame,
    open_buy_regime_only: bool,
    trend_only: bool,
    dashboard_buy_only: bool = False,
    fresh_weekly_buy_only: bool = False,
    dashboard_buy_symbols: set[str] | None = None,
) -> str:
    missing = []
    if open_buy_regime_only and "is_latest_signal_buy" not in frame.columns:
        missing.append("open BUY regime")
    if fresh_weekly_buy_only and "latest_week_signal" not in frame.columns:
        missing.append("fresh weekly BUY")
    if trend_only and not {"close_above_ema20", "ema20_above_ema50"}.issubset(frame.columns):
        missing.append("EMA trend")
    if dashboard_buy_only and not dashboard_buy_symbols:
        missing.append("dashboard BUY symbols")
    if not missing:
        return ""
    return (
        "The saved GTT Study rows do not include "
        + " and ".join(missing)
        + " data yet. Run the Weekly BUY Screener and GTT Gain Study once more."
    )


def _filter_by_symbols(frame: pd.DataFrame, symbols: set[str]) -> pd.DataFrame:
    if frame.empty or not symbols:
        return frame.iloc[0:0].copy() if not symbols else frame
    symbol_column = _symbol_column(frame)
    if not symbol_column:
        return frame
    normalized_symbols = {str(symbol).upper() for symbol in symbols}
    return frame[frame[symbol_column].astype(str).str.upper().isin(normalized_symbols)]


def _symbols_from_frame(frame: pd.DataFrame) -> set[str]:
    symbol_column = _symbol_column(frame)
    if frame.empty or not symbol_column:
        return set()
    return set(frame[symbol_column].dropna().astype(str).str.upper())


def _dashboard_buy_symbols(data_root: Path) -> set[str]:
    filtered = Storage(data_root).load_signals("latest_filtered.csv")
    if filtered.empty:
        return set()
    if "signal" in filtered.columns:
        filtered = filtered[filtered["signal"].astype(str).str.upper() == "BUY"]
    return _symbols_from_frame(filtered)


def _latest_kite_universe_symbols(data_root: Path, config: dict[str, Any]) -> set[str]:
    return _symbols_from_frame(_latest_kite_universe_frame(data_root, config))


def _latest_kite_universe_frame(data_root: Path, config: dict[str, Any]) -> pd.DataFrame:
    storage = Storage(data_root)
    instruments = storage.load_instruments()
    if instruments.empty:
        return pd.DataFrame(columns=["exchange", "symbol", "name"])

    universe = build_universe(instruments, config)
    if universe.empty or "tradingsymbol" not in universe.columns:
        return pd.DataFrame(columns=["exchange", "symbol", "name"])

    frame = universe.copy()
    if "exchange" in frame.columns:
        frame = frame[frame["exchange"].astype(str).str.upper() == "NSE"]
    else:
        frame["exchange"] = "NSE"
    if frame.empty:
        return pd.DataFrame(columns=["exchange", "symbol", "name"])

    frame["exchange"] = frame["exchange"].astype(str).str.upper().str.strip()
    frame["symbol"] = frame["tradingsymbol"].astype(str).str.upper().str.strip()
    if "name" not in frame.columns:
        frame["name"] = frame["symbol"]
    frame["name"] = frame["name"].fillna("").astype(str).str.strip()
    frame["name"] = frame["name"].mask(frame["name"] == "", frame["symbol"])
    frame = frame[frame["symbol"] != ""]
    return frame[["exchange", "symbol", "name"]].drop_duplicates(subset=["exchange", "symbol"], keep="last")


def _latest_scan_frame(data_root: Path) -> pd.DataFrame:
    scan_details = Storage(data_root).load_signals("latest_scan_details.csv")
    if scan_details.empty or "symbol" not in scan_details.columns:
        return pd.DataFrame(columns=["exchange", "symbol", "name"])

    frame = scan_details.copy()
    if "exchange" in frame.columns:
        frame = frame[frame["exchange"].astype(str).str.upper() == "NSE"]
    else:
        frame["exchange"] = "NSE"
    if frame.empty:
        return pd.DataFrame(columns=["exchange", "symbol", "name"])

    frame["exchange"] = frame["exchange"].astype(str).str.upper()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    if "name" not in frame.columns:
        frame["name"] = frame["symbol"]
    frame["name"] = frame["name"].fillna(frame["symbol"]).astype(str)
    return frame[["exchange", "symbol", "name"]].drop_duplicates(subset=["exchange", "symbol"], keep="last")


def _align_gtt_stock_stats_to_latest_universe(
    data_root: Path,
    stock_stats: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    latest_universe = _latest_kite_universe_frame(data_root, config)
    if latest_universe.empty:
        return pd.DataFrame(columns=["exchange", "symbol", "name"])

    stats = stock_stats.copy()
    if stats.empty or "symbol" not in stats.columns:
        aligned = latest_universe.copy()
    else:
        stats["symbol"] = stats["symbol"].astype(str).str.upper()
        if "exchange" not in stats.columns:
            stats["exchange"] = "NSE"
        stats["exchange"] = stats["exchange"].astype(str).str.upper()
        stats = stats.drop(columns=["name"], errors="ignore")
        aligned = latest_universe.merge(stats, on=["exchange", "symbol"], how="left")

    for column in ("latest_week_signal", "latest_signal"):
        if column not in aligned.columns:
            aligned[column] = "NONE"
        aligned[column] = aligned[column].fillna("NONE")
    if "is_latest_signal_buy" not in aligned.columns:
        aligned["is_latest_signal_buy"] = False
    aligned["is_latest_signal_buy"] = _truthy_series(aligned["is_latest_signal_buy"])

    count_columns = [
        "closed_pairs",
        "valid_pairs",
        "pairs_without_daily_window",
        "times_went_above_buy_price",
        "hit_5pct_count",
        "hit_10pct_count",
        "hit_15pct_count",
        "hit_20pct_count",
        "hit_25pct_count",
        "hit_30pct_count",
    ]
    rate_columns = [
        "went_above_buy_price_rate_pct",
        "hit_5pct_rate_pct",
        "hit_10pct_rate_pct",
        "hit_15pct_rate_pct",
        "hit_20pct_rate_pct",
        "hit_25pct_rate_pct",
        "hit_30pct_rate_pct",
    ]
    for column in count_columns:
        if column not in aligned.columns:
            aligned[column] = 0
        aligned[column] = pd.to_numeric(aligned[column], errors="coerce").fillna(0).astype(int)
    for column in rate_columns:
        if column not in aligned.columns:
            aligned[column] = 0.0
        aligned[column] = pd.to_numeric(aligned[column], errors="coerce").fillna(0.0)
    if "low_sample" not in aligned.columns:
        aligned["low_sample"] = True
    aligned["low_sample"] = aligned["low_sample"].map(
        lambda value: True if pd.isna(value) else str(value).strip().lower() in {"1", "true", "yes", "y"}
    )
    return aligned


def _gtt_cached_symbols(data_root: Path, exchange: str = "NSE") -> set[str]:
    candle_dir = data_root / "candles" / exchange / "1D"
    if not candle_dir.exists():
        return set()
    return {path.stem.upper() for path in candle_dir.glob("*.csv")}


def _build_gtt_universe_audit(data_root: Path, stock_stats: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    latest_universe_symbols = _latest_kite_universe_symbols(data_root, config)
    dashboard_scan_symbols = _symbols_from_frame(_latest_scan_frame(data_root))
    dashboard_buy_symbols = _dashboard_buy_symbols(data_root)
    gtt_cached_symbols = _gtt_cached_symbols(data_root)
    gtt_stock_symbols = _symbols_from_frame(stock_stats)

    open_buy_count = 0
    fresh_buy_count = 0
    if not stock_stats.empty and "is_latest_signal_buy" in stock_stats.columns:
        open_buy_count = int(_truthy_series(stock_stats["is_latest_signal_buy"]).sum())
    if not stock_stats.empty and "latest_week_signal" in stock_stats.columns:
        fresh_buy_count = int((stock_stats["latest_week_signal"].astype(str).str.upper() == "BUY").sum())

    excluded_cached = sorted(gtt_cached_symbols - latest_universe_symbols)[:20]
    missing_gtt_rows = sorted(latest_universe_symbols - gtt_stock_symbols)[:20]

    return {
        "dashboard_scanned_symbols": len(dashboard_scan_symbols),
        "home_filtered_buy_symbols": len(dashboard_buy_symbols),
        "latest_nse_universe_symbols": len(latest_universe_symbols),
        "gtt_rows_in_latest_universe": len(gtt_stock_symbols & latest_universe_symbols),
        "gtt_open_buy_regime_symbols": open_buy_count,
        "gtt_fresh_weekly_buy_symbols": fresh_buy_count,
        "excluded_cached_symbol_count": len(gtt_cached_symbols - latest_universe_symbols),
        "missing_gtt_row_count": len(latest_universe_symbols - gtt_stock_symbols),
        "excluded_cached_sample": excluded_cached,
        "missing_gtt_rows_sample": missing_gtt_rows,
    }


def _gtt_display_summary(
    saved_summary: dict[str, Any],
    stock_stats: pd.DataFrame,
    pair_details: pd.DataFrame,
    open_positions: pd.DataFrame,
) -> dict[str, Any]:
    summary = dict(saved_summary or {})
    summary["symbols_processed"] = len(stock_stats)
    summary["open_buy_positions"] = len(open_positions)
    summary["closed_pairs"] = len(pair_details)

    if pair_details.empty:
        summary.update(
            {
                "valid_pairs": 0,
                "pairs_without_daily_data": 0,
                "overall_median_max_gain_pct": 0.0,
                "overall_avg_max_gain_pct": 0.0,
                "went_above_buy_price_rate_pct": 0.0,
                "hit_5pct_rate_pct": 0.0,
                "hit_10pct_rate_pct": 0.0,
                "hit_15pct_rate_pct": 0.0,
                "hit_20pct_rate_pct": 0.0,
                "hit_25pct_rate_pct": 0.0,
                "hit_30pct_rate_pct": 0.0,
            }
        )
        return summary

    valid_mask = _truthy_series(pair_details.get("valid_daily_window", pd.Series(False, index=pair_details.index)))
    valid_pairs = pair_details[valid_mask].copy()
    summary["valid_pairs"] = len(valid_pairs)
    summary["pairs_without_daily_data"] = len(pair_details) - len(valid_pairs)
    if valid_pairs.empty:
        summary["overall_median_max_gain_pct"] = 0.0
        summary["overall_avg_max_gain_pct"] = 0.0
        summary["went_above_buy_price_rate_pct"] = 0.0
        for threshold in (5, 10, 15, 20, 25, 30):
            summary[f"hit_{threshold}pct_rate_pct"] = 0.0
        return summary

    max_gain = pd.to_numeric(valid_pairs["max_gain_pct"], errors="coerce").dropna()
    summary["overall_median_max_gain_pct"] = float(max_gain.median()) if not max_gain.empty else 0.0
    summary["overall_avg_max_gain_pct"] = float(max_gain.mean()) if not max_gain.empty else 0.0
    summary["went_above_buy_price_rate_pct"] = float((max_gain > 0).mean() * 100) if not max_gain.empty else 0.0
    for threshold in (5, 10, 15, 20, 25, 30):
        column = f"hit_{threshold}pct"
        if column in valid_pairs.columns:
            summary[f"hit_{threshold}pct_rate_pct"] = float(_truthy_series(valid_pairs[column]).mean() * 100)
        else:
            summary[f"hit_{threshold}pct_rate_pct"] = float((max_gain >= threshold).mean() * 100) if not max_gain.empty else 0.0
    return summary


def _gtt_filter_query(
    token: str = "",
    stock_search: str = "",
    market_cap_bucket: str = "",
    min_market_cap_cr: str = "",
    max_market_cap_cr: str = "",
    open_buy_regime_only: bool = False,
    dashboard_buy_only: bool = False,
    fresh_weekly_buy_only: bool = False,
    trend_only: bool = False,
    require_volume_confirmation: bool = False,
    require_screener_trend_confirmation: bool = False,
    return_metric: str = "",
    min_pair_return_pct: str = "",
) -> str:
    params = []
    if token:
        params.append(f"token={quote(token)}")
    if stock_search:
        params.append(f"stock_search={quote(stock_search)}")
    if market_cap_bucket:
        params.append(f"market_cap_bucket={quote(market_cap_bucket)}")
    if min_market_cap_cr:
        params.append(f"min_market_cap_cr={quote(min_market_cap_cr)}")
    if max_market_cap_cr:
        params.append(f"max_market_cap_cr={quote(max_market_cap_cr)}")
    if open_buy_regime_only:
        params.append("open_buy_regime_only=1")
    if dashboard_buy_only:
        params.append("dashboard_buy_only=1")
    if fresh_weekly_buy_only:
        params.append("fresh_weekly_buy_only=1")
    if trend_only:
        params.append("trend_only=1")
    if require_volume_confirmation:
        params.append("require_volume_confirmation=1")
    if require_screener_trend_confirmation:
        params.append("require_trend_confirmation=1")
    if return_metric:
        params.append(f"return_metric={quote(return_metric)}")
    if min_pair_return_pct:
        params.append(f"min_pair_return_pct={quote(min_pair_return_pct)}")
    return "&".join(params)


def _gtt_filter_summary(
    stock_search: str,
    market_cap_bucket: str,
    min_market_cap_text: str,
    max_market_cap_text: str,
    open_buy_regime_only: bool = False,
    dashboard_buy_only: bool = False,
    fresh_weekly_buy_only: bool = False,
    trend_only: bool = False,
    require_volume_confirmation: bool = False,
    require_screener_trend_confirmation: bool = False,
    return_metric: str = "",
    min_pair_return_text: str = "",
) -> str:
    filters = []
    if stock_search:
        filters.append(f"Search: {stock_search}")
    if market_cap_bucket:
        filters.append(f"Market cap bucket: {market_cap_bucket}")
    if min_market_cap_text:
        filters.append(f"Min market cap: {min_market_cap_text} Cr")
    if max_market_cap_text:
        filters.append(f"Max market cap: {max_market_cap_text} Cr")
    if open_buy_regime_only:
        filters.append("Open BUY regime")
    if dashboard_buy_only:
        filters.append("Dashboard BUY signals only")
    if fresh_weekly_buy_only:
        filters.append("Fresh weekly BUY only")
    if trend_only:
        filters.append("Close > 20W EMA and 20W EMA > 50W EMA")
    if require_volume_confirmation:
        filters.append("Home screener volume confirmation")
    if require_screener_trend_confirmation:
        filters.append("Home screener trend confirmation")
    if min_pair_return_text:
        metric_label = "Home last completed BUY-SELL return" if return_metric == "last_1" else "Home median last 3 BUY-SELL returns"
        filters.append(f"{metric_label} >= {min_pair_return_text}%")
    return "; ".join(filters) if filters else "None"


def _records(frame: pd.DataFrame) -> list[dict]:
    if frame.empty:
        return []
    return frame.where(pd.notna(frame), "").to_dict(orient="records")


def _comma_separated_symbols(frame: pd.DataFrame) -> str:
    symbol_column = _symbol_column(frame)
    if frame.empty or not symbol_column:
        return ""
    symbols = frame[symbol_column].dropna().astype(str).str.upper().str.strip()
    symbols = [symbol for symbol in symbols if symbol]
    return ",".join(dict.fromkeys(symbols))


def _symbol_column(frame: pd.DataFrame) -> str | None:
    if "symbol" in frame.columns:
        return "symbol"
    if "tradingsymbol" in frame.columns:
        return "tradingsymbol"
    return None


def _row_symbol(row: pd.Series) -> str:
    for column in ("symbol", "tradingsymbol"):
        value = row.get(column, "")
        if pd.notna(value) and str(value).strip():
            return str(value)
    return ""


def _dashboard_link_suffix(request: Request) -> str:
    params = []
    for name in (
        "token",
        "stock_search",
        "market_cap_bucket",
        "min_market_cap_cr",
        "max_market_cap_cr",
        "require_volume_confirmation",
        "require_trend_confirmation",
        "return_metric",
        "min_pair_return_pct",
    ):
        value = request.query_params.get(name, "").strip()
        if value:
            params.append(f"{name}={quote(value)}")
    return ("&" + "&".join(params)) if params else ""


def _dashboard_filter_query(
    token: str = "",
    stock_search: str = "",
    market_cap_bucket: str = "",
    min_market_cap_cr: str = "",
    max_market_cap_cr: str = "",
    require_volume_confirmation: bool = False,
    require_trend_confirmation: bool = False,
    return_metric: str = "",
    min_pair_return_pct: str = "",
) -> str:
    params = []
    if token:
        params.append(f"token={quote(token)}")
    if stock_search:
        params.append(f"stock_search={quote(stock_search)}")
    if market_cap_bucket:
        params.append(f"market_cap_bucket={quote(market_cap_bucket)}")
    if min_market_cap_cr:
        params.append(f"min_market_cap_cr={quote(min_market_cap_cr)}")
    if max_market_cap_cr:
        params.append(f"max_market_cap_cr={quote(max_market_cap_cr)}")
    if require_volume_confirmation:
        params.append("require_volume_confirmation=1")
    if require_trend_confirmation:
        params.append("require_trend_confirmation=1")
    if return_metric:
        params.append(f"return_metric={quote(return_metric)}")
    if min_pair_return_pct:
        params.append(f"min_pair_return_pct={quote(min_pair_return_pct)}")
    return "&".join(params)


def _buy_signal_filter_summary(
    stock_search: str,
    market_cap_bucket: str,
    min_market_cap_text: str,
    max_market_cap_text: str,
    require_volume_confirmation: bool = False,
    require_trend_confirmation: bool = False,
    return_metric: str = "",
    min_pair_return_text: str = "",
) -> str:
    filters = []
    if stock_search:
        filters.append(f"Search: {stock_search}")
    if market_cap_bucket:
        filters.append(f"Market cap bucket: {market_cap_bucket}")
    if min_market_cap_text:
        filters.append(f"Min market cap: {min_market_cap_text} Cr")
    if max_market_cap_text:
        filters.append(f"Max market cap: {max_market_cap_text} Cr")
    if require_volume_confirmation:
        filters.append("Volume confirmation: Yes")
    if require_trend_confirmation:
        filters.append("Trend confirmation: Yes")
    if min_pair_return_text:
        metric_label = "Last completed BUY-SELL return" if return_metric == "last_1" else "Median last 3 BUY-SELL returns"
        filters.append(f"{metric_label} >= {min_pair_return_text}%")
    return "; ".join(filters) if filters else "None"


def _manual_screener_config(
    base_config: dict,
    storage: Storage,
    min_market_cap: float | None,
    max_market_cap: float | None,
    market_cap_bucket: str,
    stock_search: str,
) -> dict:
    config = deepcopy(base_config)
    universe_cfg = config.setdefault("universe", {})
    filters_cfg = universe_cfg.setdefault("filters", {})

    metadata_path = storage.symbol_metadata_path()
    if metadata_path.exists():
        universe_cfg["metadata_file"] = str(metadata_path)

    filters_cfg["min_market_cap_cr"] = min_market_cap
    filters_cfg["max_market_cap_cr"] = max_market_cap
    filters_cfg["market_cap_bucket"] = market_cap_bucket or None
    filters_cfg["stock_search"] = stock_search.strip() or None

    signal_cfg = config.setdefault("filters", {}).setdefault("signal", {})
    signal_cfg["direction"] = "BUY"
    signal_cfg["latest_only"] = True

    config.setdefault("notifications", {})["enabled"] = False
    return config


def _load_visible_buy_signals(
    config: dict[str, Any],
    storage: Storage,
    stock_search: str,
    min_market_cap: float | None,
    max_market_cap: float | None,
    market_cap_bucket: str,
    require_volume_confirmation: bool = False,
    require_trend_confirmation: bool = False,
    return_metric: str = "",
    min_pair_return: float | None = None,
) -> pd.DataFrame:
    metadata = _combined_symbol_metadata(config, storage)
    filtered = storage.load_signals("latest_filtered.csv")
    filtered = _enrich_with_symbol_metadata(filtered, metadata, "symbol")
    filtered = _apply_market_cap_filters(filtered, min_market_cap, max_market_cap, market_cap_bucket)
    filtered = _apply_stock_search(filtered, stock_search)
    filtered = _apply_signal_quality_filters(
        filtered,
        require_volume_confirmation,
        require_trend_confirmation,
        return_metric,
        min_pair_return,
    )

    if not filtered.empty and "date" in filtered.columns:
        filtered = filtered.copy()
        filtered["date_sort"] = pd.to_datetime(filtered["date"], errors="coerce")
        sort_columns = ["date_sort"]
        sort_ascending = [False]
        symbol_column = _symbol_column(filtered)
        if symbol_column:
            sort_columns.append(symbol_column)
            sort_ascending.append(True)
        filtered = filtered.sort_values(sort_columns, ascending=sort_ascending).drop(columns=["date_sort"], errors="ignore")
    return filtered.reset_index(drop=True)


def _load_visible_gtt_stock_stats(
    config: dict[str, Any],
    storage: Storage,
    data_root: Path,
    stock_search: str,
    min_market_cap: float | None,
    max_market_cap: float | None,
    market_cap_bucket: str,
    open_buy_regime_only: bool = False,
    dashboard_buy_only: bool = False,
    fresh_weekly_buy_only: bool = False,
    trend_only: bool = False,
) -> pd.DataFrame:
    latest = load_gtt_gain_outputs(_gtt_gain_dir(data_root))
    stock_stats = _align_gtt_stock_stats_to_latest_universe(data_root, latest.stock_stats, config)
    stock_stats = _enrich_with_symbol_metadata(stock_stats, _combined_symbol_metadata(config, storage), "symbol")
    stock_stats = _apply_market_cap_filters(stock_stats, min_market_cap, max_market_cap, market_cap_bucket)
    stock_stats = _apply_stock_search(stock_stats, stock_search)
    stock_stats = _apply_gtt_stock_filters(
        stock_stats,
        open_buy_regime_only,
        trend_only,
        dashboard_buy_only,
        _dashboard_buy_symbols(data_root),
        fresh_weekly_buy_only,
    )

    if stock_stats.empty:
        return stock_stats

    sorted_stats = stock_stats.copy()
    sort_columns = []
    sort_ascending = []
    for column, ascending in (
        ("valid_pairs", False),
        ("hit_10pct_rate_pct", False),
        ("median_max_gain_pct", False),
        ("symbol", True),
    ):
        if column in sorted_stats.columns:
            if column != "symbol":
                sorted_stats[column] = pd.to_numeric(sorted_stats[column], errors="coerce")
            sort_columns.append(column)
            sort_ascending.append(ascending)
    if sort_columns:
        sorted_stats = sorted_stats.sort_values(sort_columns, ascending=sort_ascending, na_position="last")
    return sorted_stats.reset_index(drop=True)


def _signal_qa_candidates(
    filtered: pd.DataFrame,
    scan_details: pd.DataFrame,
    instruments: pd.DataFrame,
    symbol_search: str,
) -> pd.DataFrame:
    candidate_frames = []
    for source_priority, frame in enumerate((scan_details, filtered, instruments)):
        if frame.empty:
            continue
        candidate = frame.copy()
        if "symbol" not in candidate.columns and "tradingsymbol" in candidate.columns:
            candidate["symbol"] = candidate["tradingsymbol"]
        candidate["qa_source_priority"] = source_priority
        candidate_frames.append(candidate)

    if not candidate_frames:
        return pd.DataFrame()

    candidates = pd.concat(candidate_frames, ignore_index=True, sort=False)
    if candidates.empty:
        return candidates

    candidates = candidates.drop_duplicates(subset=[column for column in ("exchange", "symbol") if column in candidates.columns])
    candidates = _apply_stock_search(candidates, symbol_search)

    if not candidates.empty:
        sort_columns = [column for column in ("qa_source_priority", "exchange", "symbol") if column in candidates.columns]
        if sort_columns:
            candidates = candidates.sort_values(sort_columns)
    return candidates.drop(columns=["qa_source_priority"], errors="ignore").reset_index(drop=True)


def _selected_signal_qa_symbol(
    request: Request,
    filtered: pd.DataFrame,
    candidates: pd.DataFrame,
    symbol_search: str,
) -> tuple[str, str]:
    selected_exchange = request.query_params.get("exchange", "").strip()
    selected_symbol = request.query_params.get("symbol", "").strip()

    if selected_exchange and selected_symbol:
        return selected_exchange, selected_symbol

    if symbol_search and not candidates.empty:
        first = candidates.iloc[0]
        return str(first.get("exchange", "")), _row_symbol(first)

    if not filtered.empty:
        first = filtered.iloc[0]
        return str(first.get("exchange", "")), _row_symbol(first)

    if not candidates.empty:
        first = candidates.iloc[0]
        return str(first.get("exchange", "")), _row_symbol(first)

    return "", ""


def _scan_redirect_url(summary: dict[str, Any], query_suffix: str) -> str:
    return (
        "/?"
        f"scan_ran=1&symbols_scanned={summary.get('symbols_scanned', 0)}"
        f"&filtered_matches={summary.get('filtered_matches', 0)}"
        f"{query_suffix}"
    )


def _scan_error_url(error: Exception, query_suffix: str) -> str:
    return f"/?scan_error={quote(str(error)[:500])}{query_suffix}"


def _gtt_gain_redirect_url(summary: dict[str, Any], query_suffix: str) -> str:
    return (
        "/gtt-gain-study?"
        f"study_ran=1&valid_pairs={summary.get('valid_pairs', 0)}"
        f"&symbols_processed={summary.get('symbols_processed', 0)}"
        f"{query_suffix}"
    )


def _gtt_gain_error_url(error: Exception, query_suffix: str) -> str:
    return f"/gtt-gain-study?study_error={quote(str(error)[:500])}{query_suffix}"


def _run_screener_job(job_id: str, scan_config: dict[str, Any], query_suffix: str) -> None:
    _set_scan_job(
        job_id,
        status="running",
        phase="Starting",
        completed=0,
        total=0,
        percent=0,
        current_symbol="",
        current_exchange="",
    )

    def progress_callback(payload: dict[str, Any]) -> None:
        total = int(payload.get("total") or 0)
        completed = int(payload.get("completed") or 0)
        percent = int((completed / total) * 100) if total else 0
        _set_scan_job(
            job_id,
            status="running",
            phase=payload.get("phase", "Running"),
            completed=completed,
            total=total,
            percent=max(0, min(percent, 100)),
            current_symbol=payload.get("current_symbol", ""),
            current_exchange=payload.get("current_exchange", ""),
        )

    try:
        summary = run_daily_scan(scan_config, progress_callback=progress_callback)
        _set_scan_job(
            job_id,
            status="completed",
            phase="Complete",
            completed=int(summary.get("symbols_scanned", 0)),
            total=int(summary.get("symbols_scanned", 0)),
            percent=100,
            current_symbol="",
            current_exchange="",
            summary=summary,
            redirect_url=_scan_redirect_url(summary, query_suffix),
        )
    except Exception as exc:
        _set_scan_job(
            job_id,
            status="failed",
            phase="Failed",
            error=str(exc),
            redirect_url=_scan_error_url(exc, query_suffix),
        )


def _run_gtt_gain_job(job_id: str, config: dict[str, Any], data_root: Path, query_suffix: str) -> None:
    storage = Storage(data_root)
    _set_scan_job(
        job_id,
        status="running",
        phase="Starting GTT Gain Study",
        completed=0,
        total=0,
        percent=0,
        current_symbol="",
        current_exchange="NSE",
    )

    def progress_callback(payload: dict[str, Any]) -> None:
        total = int(payload.get("total") or 0)
        completed = int(payload.get("completed") or 0)
        percent = int((completed / total) * 100) if total else 0
        _set_scan_job(
            job_id,
            status="running",
            phase=payload.get("phase", "Running"),
            completed=completed,
            total=total,
            percent=max(0, min(percent, 100)),
            current_symbol=payload.get("current_symbol", ""),
            current_exchange=payload.get("current_exchange", ""),
        )

    try:
        result = run_gtt_gain_study(config, storage, exchange="NSE", progress_callback=progress_callback)
        save_gtt_gain_outputs(result, _gtt_gain_dir(data_root))
        _set_scan_job(
            job_id,
            status="running",
            phase="Writing Excel report",
            completed=int(result.summary.get("symbols_processed", 0)),
            total=int(result.summary.get("symbols_processed", 0)),
            percent=99,
            current_symbol="",
            current_exchange="NSE",
        )
        write_gtt_gain_workbook(result, _latest_gtt_gain_paths(data_root)["workbook"])
        _set_scan_job(
            job_id,
            status="completed",
            phase="Complete",
            completed=int(result.summary.get("symbols_processed", 0)),
            total=int(result.summary.get("symbols_processed", 0)),
            percent=100,
            current_symbol="",
            current_exchange="",
            summary=result.summary,
            redirect_url=_gtt_gain_redirect_url(result.summary, query_suffix),
        )
    except Exception as exc:
        _set_scan_job(
            job_id,
            status="failed",
            phase="Failed",
            error=str(exc),
            redirect_url=_gtt_gain_error_url(exc, query_suffix),
        )


def _has_market_cap_metadata(storage: Storage) -> bool:
    metadata = storage.load_symbol_metadata()
    return (
        not metadata.empty
        and "market_cap_cr" in metadata.columns
        and pd.to_numeric(metadata["market_cap_cr"], errors="coerce").notna().any()
        and not _market_cap_metadata_needs_refresh(storage)
    )


def _market_cap_metadata_needs_refresh(storage: Storage) -> bool:
    metadata = storage.load_symbol_metadata()
    if metadata.empty or "market_cap_cr" not in metadata.columns:
        return True

    market_caps = pd.to_numeric(metadata["market_cap_cr"], errors="coerce")
    if not market_caps.notna().any():
        return True

    # Full NSE market-cap files should contain very large companies. If the
    # maximum is tiny, the file was imported with the wrong unit divisor.
    return float(market_caps.max()) < 1000


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def _ensure_market_cap_metadata(config: dict, storage: Storage) -> None:
    if not _market_cap_metadata_needs_refresh(storage):
        return

    universe_cfg = config.get("universe", {})
    market_cap_cfg = universe_cfg.get("market_cap_source", {})
    local_path_value = str(market_cap_cfg.get("local_path", "")).strip()
    if not local_path_value:
        return

    local_path = _resolve_project_path(local_path_value)
    if not local_path.exists():
        return

    bucket_cfg = universe_cfg.get("market_cap_buckets", {})
    small_max_cr = float(bucket_cfg.get("small_max_cr", 5000))
    mid_max_cr = float(bucket_cfg.get("mid_max_cr", 20000))
    market_cap_divisor = market_cap_cfg.get("market_cap_divisor")
    market_cap_divisor = float(market_cap_divisor) if market_cap_divisor else None

    metadata = load_nse_market_cap_excel(local_path, small_max_cr, mid_max_cr, market_cap_divisor)
    storage.save_symbol_metadata(metadata)


def _load_big_bull_deals(data_root: Path) -> pd.DataFrame:
    default_from, default_to = default_last_7_days_range()
    try:
        rows = SupabaseStore().list_large_deals(
            limit=1000,
            from_date=default_from.isoformat(),
            to_date=default_to.isoformat(),
        )
        if rows:
            return pd.DataFrame(rows)
    except Exception as exc:
        print(f"Supabase large deals unavailable; falling back to CSV: {exc}")

    path = data_root / "deals" / "big_bull_trades.csv"
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "date",
                "exchange",
                "symbol",
                "investor",
                "category",
                "action",
                "quantity",
                "price",
                "value_cr",
                "source",
            ]
        )
    return pd.read_csv(path)


def _large_deal_markers(deals: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if deals.empty or "symbol" not in deals.columns:
        return {}

    frame = deals.copy()
    frame["symbol_key"] = frame["symbol"].apply(normalize_nse_symbol)
    markers: dict[str, dict[str, Any]] = {}

    for symbol_key, group in frame.groupby("symbol_key", dropna=True):
        actions = group["action"].astype(str).str.upper() if "action" in group.columns else pd.Series(dtype=str)
        buy_count = int((actions == "BUY").sum())
        sell_count = int((actions == "SELL").sum())
        latest_date = ""
        if "deal_date" in group.columns:
            latest_date_value = pd.to_datetime(group["deal_date"], errors="coerce").max()
            latest_date = str(latest_date_value.date()) if pd.notna(latest_date_value) else ""
        elif "date" in group.columns:
            latest_date_value = pd.to_datetime(group["date"], errors="coerce").max()
            latest_date = str(latest_date_value.date()) if pd.notna(latest_date_value) else ""

        summary_parts = []
        if buy_count:
            summary_parts.append(f"{buy_count} BUY")
        if sell_count:
            summary_parts.append(f"{sell_count} SELL")
        if not summary_parts:
            summary_parts.append(f"{len(group)} deal")

        markers[str(symbol_key)] = {
            "has_large_deal": True,
            "large_deal_count": int(len(group)),
            "large_deal_buy_count": buy_count,
            "large_deal_sell_count": sell_count,
            "large_deal_latest_date": latest_date,
            "large_deal_summary": ", ".join(summary_parts),
        }
    return markers


def _apply_large_deal_markers(frame: pd.DataFrame, deals: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    marked = frame.copy()
    markers = _large_deal_markers(deals)
    symbol_column = _symbol_column(marked)
    if not markers or not symbol_column:
        marked["has_large_deal"] = False
        marked["large_deal_summary"] = ""
        marked["large_deal_latest_date"] = ""
        return marked

    keys = marked[symbol_column].apply(normalize_nse_symbol)
    marked["has_large_deal"] = keys.map(lambda key: bool(markers.get(key, {}).get("has_large_deal", False)))
    marked["large_deal_summary"] = keys.map(lambda key: markers.get(key, {}).get("large_deal_summary", ""))
    marked["large_deal_latest_date"] = keys.map(lambda key: markers.get(key, {}).get("large_deal_latest_date", ""))
    marked["large_deal_count"] = keys.map(lambda key: markers.get(key, {}).get("large_deal_count", 0))
    return marked


def _backtest_dir(data_root: Path) -> Path:
    path = data_root / "backtests"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _latest_backtest_paths(data_root: Path) -> dict[str, Path]:
    directory = _backtest_dir(data_root)
    return {
        "summary": directory / "latest_summary.csv",
        "stock_stats": directory / "latest_stock_stats.csv",
        "trades": directory / "latest_trades.csv",
        "open_positions": directory / "latest_open_positions.csv",
        "workbook": directory / "buy_sell_backtest_report.xlsx",
    }


def _gtt_gain_dir(data_root: Path) -> Path:
    path = data_root / "gtt_gain_study"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _latest_gtt_gain_paths(data_root: Path) -> dict[str, Path]:
    directory = _gtt_gain_dir(data_root)
    return {
        "summary": directory / "latest_summary.csv",
        "stock_stats": directory / "latest_stock_gtt_stats.csv",
        "pair_details": directory / "latest_pair_details.csv",
        "open_positions": directory / "latest_open_positions.csv",
        "workbook": directory / "gtt_gain_study_report.xlsx",
    }


def _load_latest_backtest(data_root: Path) -> dict[str, Any]:
    paths = _latest_backtest_paths(data_root)
    summary = {}
    stock_stats = pd.DataFrame()
    trades = pd.DataFrame()
    open_positions = pd.DataFrame()

    if paths["summary"].exists():
        frame = pd.read_csv(paths["summary"])
        if not frame.empty:
            summary = frame.iloc[0].to_dict()
    if paths["stock_stats"].exists():
        stock_stats = pd.read_csv(paths["stock_stats"])
    if paths["trades"].exists():
        trades = pd.read_csv(paths["trades"])
    if paths["open_positions"].exists():
        open_positions = pd.read_csv(paths["open_positions"])

    return {
        "summary": summary,
        "stock_stats": stock_stats,
        "trades": trades,
        "open_positions": open_positions,
        "workbook_exists": paths["workbook"].exists(),
        "workbook_path": paths["workbook"],
    }


def _fetch_and_store_big_bull_deals() -> RedirectResponse:
    try:
        result = fetch_and_store_current_large_deals()
        return RedirectResponse(
            (
                "/big-bull-deals?"
                f"refreshed=1&rows={result['stored']}"
                f"&fetched={result['fetched']}"
                f"&skipped_existing_dates={result.get('skipped_existing_dates', 0)}"
            ),
            status_code=303,
        )
    except Exception as exc:
        message = quote(str(exc)[:500])
        return RedirectResponse(f"/big-bull-deals?fetch_error={message}", status_code=303)


@app.post("/big-bull-deals/fetch")
def fetch_big_bull_deals_post() -> RedirectResponse:
    return _fetch_and_store_big_bull_deals()


@app.get("/big-bull-deals/fetch")
def fetch_big_bull_deals_get() -> RedirectResponse:
    return _fetch_and_store_big_bull_deals()


@app.get("/health", response_class=PlainTextResponse)
def health() -> str:
    return "ok"


@app.get("/backtest", response_class=HTMLResponse)
def backtest_page(request: Request) -> HTMLResponse:
    if not _is_allowed(request):
        return templates.TemplateResponse(
            "locked.html",
            {"request": request, "app_name": "Investment Screener"},
            status_code=401,
        )

    config = load_config()
    data_root = get_data_root(config)
    latest = _load_latest_backtest(data_root)
    stock_stats = latest["stock_stats"].head(100)
    trades = latest["trades"].head(100)

    return templates.TemplateResponse(
        "backtest.html",
        {
            "request": request,
            "app_name": config.get("app", {}).get("name", "Investment Screener"),
            "dashboard_token": request.query_params.get("token", ""),
            "summary": latest["summary"],
            "stock_stats": _records(stock_stats),
            "trades": _records(trades),
            "workbook_exists": latest["workbook_exists"],
            "backtest_ran": request.query_params.get("backtest_ran", ""),
            "backtest_error": request.query_params.get("backtest_error", ""),
        },
    )


@app.post("/backtest/run")
def run_backtest_from_dashboard(request: Request) -> RedirectResponse:
    config = load_config()
    data_root = get_data_root(config)
    storage = Storage(data_root)
    dashboard_token = request.query_params.get("token", "").strip()

    try:
        result = run_buy_sell_backtest(config, storage, exchange="NSE")
        save_backtest_outputs(result, _backtest_dir(data_root), run_id="latest")
        write_backtest_workbook(result, _latest_backtest_paths(data_root)["workbook"])
        redirect_url = (
            "/backtest?"
            f"backtest_ran=1&closed_trades={result.summary.get('closed_trades', 0)}"
            f"&symbols_processed={result.summary.get('symbols_processed', 0)}"
        )
        if dashboard_token:
            redirect_url += f"&token={quote(dashboard_token)}"
        return RedirectResponse(redirect_url, status_code=303)
    except Exception as exc:
        redirect_url = f"/backtest?backtest_error={quote(str(exc)[:500])}"
        if dashboard_token:
            redirect_url += f"&token={quote(dashboard_token)}"
        return RedirectResponse(redirect_url, status_code=303)


@app.get("/backtest/report")
def download_backtest_report() -> FileResponse:
    config = load_config()
    workbook_path = _latest_backtest_paths(get_data_root(config))["workbook"]
    if not workbook_path.exists():
        raise HTTPException(status_code=404, detail="Backtest report has not been generated yet.")
    return FileResponse(
        workbook_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="buy_sell_backtest_report.xlsx",
    )


@app.get("/gtt-gain-study", response_class=HTMLResponse)
def gtt_gain_study_page(request: Request) -> HTMLResponse:
    if not _is_allowed(request):
        return templates.TemplateResponse(
            "locked.html",
            {"request": request, "app_name": "Investment Screener"},
            status_code=401,
        )

    config = load_config()
    data_root = get_data_root(config)
    latest = load_gtt_gain_outputs(_gtt_gain_dir(data_root))
    latest_stock_stats = _align_gtt_stock_stats_to_latest_universe(data_root, latest.stock_stats, config)
    latest_universe_symbols = _latest_kite_universe_symbols(data_root, config)
    metadata = _combined_symbol_metadata(config, Storage(data_root))
    latest_stock_stats = _enrich_with_symbol_metadata(latest_stock_stats, metadata, "symbol")
    stock_search = request.query_params.get("stock_search", "").strip()
    selected_market_cap_bucket = request.query_params.get("market_cap_bucket", "").strip()
    min_market_cap_text = request.query_params.get("min_market_cap_cr", "").strip()
    max_market_cap_text = request.query_params.get("max_market_cap_cr", "").strip()
    min_market_cap = _optional_float(min_market_cap_text)
    max_market_cap = _optional_float(max_market_cap_text)
    require_volume_confirmation = _request_bool(request, "require_volume_confirmation")
    require_screener_trend_confirmation = _request_bool(request, "require_trend_confirmation")
    selected_return_metric = request.query_params.get("return_metric", "median_3").strip() or "median_3"
    if selected_return_metric not in {"last_1", "median_3"}:
        selected_return_metric = "median_3"
    min_pair_return_text = request.query_params.get("min_pair_return_pct", "").strip()
    open_buy_regime_only = _request_bool(request, "open_buy_regime_only") or _request_bool(request, "latest_buy_only")
    dashboard_buy_only = _request_bool(request, "dashboard_buy_only")
    fresh_weekly_buy_only = _request_bool(request, "fresh_weekly_buy_only")
    trend_only = _request_bool(request, "trend_only")
    dashboard_buy_symbols = _dashboard_buy_symbols(data_root)
    universe_audit = _build_gtt_universe_audit(data_root, latest_stock_stats, config)
    universe_pair_details = latest.pair_details
    universe_open_positions = latest.open_positions
    universe_pair_details = _filter_by_symbols(universe_pair_details, latest_universe_symbols)
    universe_open_positions = _filter_by_symbols(universe_open_positions, latest_universe_symbols)
    display_summary = _gtt_display_summary(
        latest.summary,
        latest_stock_stats,
        universe_pair_details,
        universe_open_positions,
    )

    stock_stats = _apply_market_cap_filters(
        latest_stock_stats,
        min_market_cap,
        max_market_cap,
        selected_market_cap_bucket,
    )
    stock_stats = _apply_stock_search(stock_stats, stock_search)
    stock_stats_after_screener_filter_count = len(stock_stats)
    stock_stats_before_filter_count = len(stock_stats)
    gtt_filter_warning = _gtt_filter_warning(
        stock_stats,
        open_buy_regime_only,
        trend_only,
        dashboard_buy_only,
        fresh_weekly_buy_only,
        dashboard_buy_symbols,
    )
    pair_details = universe_pair_details
    open_positions = universe_open_positions
    pair_details = _apply_stock_search(pair_details, stock_search)
    open_positions = _apply_stock_search(open_positions, stock_search)
    stock_stats = _apply_gtt_stock_filters(
        stock_stats,
        open_buy_regime_only,
        trend_only,
        dashboard_buy_only,
        dashboard_buy_symbols,
        fresh_weekly_buy_only,
    )
    if open_buy_regime_only or dashboard_buy_only or fresh_weekly_buy_only or trend_only:
        visible_symbols = set(stock_stats["symbol"].astype(str)) if "symbol" in stock_stats.columns else set()
        pair_details = _filter_by_symbols(pair_details, visible_symbols)
        open_positions = _filter_by_symbols(open_positions, visible_symbols)
    workbook_path = _latest_gtt_gain_paths(data_root)["workbook"]
    gtt_filter_query = _gtt_filter_query(
        token=request.query_params.get("token", ""),
        stock_search=stock_search,
        market_cap_bucket=selected_market_cap_bucket,
        min_market_cap_cr=min_market_cap_text,
        max_market_cap_cr=max_market_cap_text,
        open_buy_regime_only=open_buy_regime_only,
        dashboard_buy_only=dashboard_buy_only,
        fresh_weekly_buy_only=fresh_weekly_buy_only,
        trend_only=trend_only,
        require_volume_confirmation=require_volume_confirmation,
        require_screener_trend_confirmation=require_screener_trend_confirmation,
        return_metric=selected_return_metric if min_pair_return_text else "",
        min_pair_return_pct=min_pair_return_text,
    )
    active_gtt_filter_summary = _gtt_filter_summary(
        stock_search,
        selected_market_cap_bucket,
        min_market_cap_text,
        max_market_cap_text,
        open_buy_regime_only,
        dashboard_buy_only,
        fresh_weekly_buy_only,
        trend_only,
        require_volume_confirmation,
        require_screener_trend_confirmation,
        selected_return_metric,
        min_pair_return_text,
    )

    return templates.TemplateResponse(
        "gtt_gain_study.html",
        {
            "request": request,
            "app_name": config.get("app", {}).get("name", "Investment Screener"),
            "dashboard_token": request.query_params.get("token", ""),
            "summary": display_summary,
            "stock_stats": _records(stock_stats),
            "stock_symbols_csv": _comma_separated_symbols(stock_stats),
            "pair_details": _records(pair_details.head(150)),
            "open_positions": _records(open_positions.head(100)),
            "stock_search": stock_search,
            "open_buy_regime_only": open_buy_regime_only,
            "dashboard_buy_only": dashboard_buy_only,
            "fresh_weekly_buy_only": fresh_weekly_buy_only,
            "trend_only": trend_only,
            "stock_stats_count": len(stock_stats),
            "stock_stats_before_filter_count": stock_stats_before_filter_count,
            "stock_stats_after_screener_filter_count": stock_stats_after_screener_filter_count,
            "selected_market_cap_bucket": selected_market_cap_bucket,
            "selected_min_market_cap": min_market_cap_text,
            "selected_max_market_cap": max_market_cap_text,
            "require_volume_confirmation": require_volume_confirmation,
            "require_screener_trend_confirmation": require_screener_trend_confirmation,
            "selected_return_metric": selected_return_metric,
            "selected_min_pair_return": min_pair_return_text,
            "active_gtt_filter_summary": active_gtt_filter_summary,
            "gtt_filter_query": gtt_filter_query,
            "gtt_filter_warning": gtt_filter_warning,
            "universe_audit": universe_audit,
            "workbook_exists": workbook_path.exists(),
            "gtt_job": request.query_params.get("gtt_job", ""),
            "study_ran": request.query_params.get("study_ran", ""),
            "study_error": request.query_params.get("study_error", ""),
            "telegram_sent": request.query_params.get("telegram_sent", ""),
            "telegram_error": request.query_params.get("telegram_error", ""),
        },
    )


@app.post("/gtt-gain-study/run")
def run_gtt_gain_study_from_dashboard(request: Request, background_tasks: BackgroundTasks) -> RedirectResponse:
    config = load_config()
    data_root = get_data_root(config)
    dashboard_token = request.query_params.get("token", "").strip()
    params = []
    if dashboard_token:
        params.append(f"token={quote(dashboard_token)}")
    query_suffix = ("&" + "&".join(params)) if params else ""

    try:
        job_id = uuid4().hex
        _set_scan_job(job_id, status="queued", phase="Queued", completed=0, total=0, percent=0)
        background_tasks.add_task(_run_gtt_gain_job, job_id, config, data_root, query_suffix)
        redirect_url = f"/gtt-gain-study?gtt_job={job_id}{query_suffix}"
    except Exception as exc:
        redirect_url = _gtt_gain_error_url(exc, query_suffix)
    return RedirectResponse(redirect_url, status_code=303)


@app.get("/gtt-gain-study/report")
def download_gtt_gain_study_report() -> FileResponse:
    config = load_config()
    workbook_path = _latest_gtt_gain_paths(get_data_root(config))["workbook"]
    if not workbook_path.exists():
        raise HTTPException(status_code=404, detail="GTT gain study report has not been generated yet.")
    return FileResponse(
        workbook_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="gtt_gain_study_report.xlsx",
    )


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    if not _is_allowed(request):
        return templates.TemplateResponse(
            "locked.html",
            {"request": request},
            status_code=401,
        )

    config = load_config()
    data_root = get_data_root(config)
    storage = Storage(data_root)
    _ensure_market_cap_metadata(config, storage)
    filtered = storage.load_signals("latest_filtered.csv")
    raw = storage.load_signals("latest_raw_signals.csv")
    scan_details = storage.load_signals("latest_scan_details.csv")
    metadata = _combined_symbol_metadata(config, storage)
    stock_search = request.query_params.get("stock_search", "").strip()
    selected_market_cap_bucket = request.query_params.get("market_cap_bucket", "").strip()
    min_market_cap = _request_float(request, "min_market_cap_cr")
    max_market_cap = _request_float(request, "max_market_cap_cr")
    require_volume_confirmation = _request_bool(request, "require_volume_confirmation")
    require_trend_confirmation = _request_bool(request, "require_trend_confirmation")
    selected_return_metric = request.query_params.get("return_metric", "median_3").strip() or "median_3"
    if selected_return_metric not in {"last_1", "median_3"}:
        selected_return_metric = "median_3"
    min_pair_return = _request_float(request, "min_pair_return_pct")
    filter_link_suffix = _dashboard_link_suffix(request)
    active_filter_parts = []
    if stock_search:
        active_filter_parts.append(f"Search: {stock_search}")
    if selected_market_cap_bucket:
        active_filter_parts.append(selected_market_cap_bucket)
    if min_market_cap is not None:
        active_filter_parts.append(f"Min market cap: {request.query_params.get('min_market_cap_cr')} Cr")
    if max_market_cap is not None:
        active_filter_parts.append(f"Max market cap: {request.query_params.get('max_market_cap_cr')} Cr")
    if require_volume_confirmation:
        active_filter_parts.append("Volume confirmed")
    if require_trend_confirmation:
        active_filter_parts.append("Trend confirmed")
    if min_pair_return is not None:
        metric_label = "last pair return" if selected_return_metric == "last_1" else "median last 3 pair return"
        active_filter_parts.append(f"{metric_label} >= {request.query_params.get('min_pair_return_pct')}%")

    filtered = _enrich_with_symbol_metadata(filtered, metadata, "symbol")
    raw = _enrich_with_symbol_metadata(raw, metadata, "symbol")
    scan_details = _enrich_with_symbol_metadata(scan_details, metadata, "symbol")

    filtered = _apply_market_cap_filters(filtered, min_market_cap, max_market_cap, selected_market_cap_bucket)
    raw = _apply_market_cap_filters(raw, min_market_cap, max_market_cap, selected_market_cap_bucket)
    scan_details = _apply_market_cap_filters(scan_details, min_market_cap, max_market_cap, selected_market_cap_bucket)

    filtered = _apply_stock_search(filtered, stock_search)
    raw = _apply_stock_search(raw, stock_search)
    scan_details = _apply_stock_search(scan_details, stock_search)

    signal_quality_warning = _signal_quality_filter_warning(
        filtered,
        require_volume_confirmation,
        require_trend_confirmation,
        min_pair_return,
    )
    filtered = _apply_signal_quality_filters(
        filtered,
        require_volume_confirmation,
        require_trend_confirmation,
        selected_return_metric,
        min_pair_return,
    )
    large_deals = _load_big_bull_deals(data_root)
    filtered = _apply_large_deal_markers(filtered, large_deals)

    market_cap_bounds = {"min": "", "max": ""}
    if not metadata.empty and "market_cap_cr" in metadata.columns and metadata["market_cap_cr"].notna().any():
        market_cap_bounds = {
            "min": int(metadata["market_cap_cr"].min()),
            "max": int(metadata["market_cap_cr"].max()),
        }

    filtered_symbols = filtered.copy()
    if not filtered_symbols.empty:
        sort_columns = []
        sort_ascending = []
        if "date" in filtered_symbols.columns:
            filtered_symbols["date_sort"] = pd.to_datetime(filtered_symbols["date"], errors="coerce")
            sort_columns.append("date_sort")
            sort_ascending.append(False)
        symbol_sort_column = _symbol_column(filtered_symbols)
        if symbol_sort_column:
            sort_columns.append(symbol_sort_column)
            sort_ascending.append(True)
        if sort_columns:
            filtered_symbols = filtered_symbols.sort_values(sort_columns, ascending=sort_ascending)

    selected_exchange = request.query_params.get("exchange")
    selected_symbol = request.query_params.get("symbol")

    if (not selected_exchange or not selected_symbol) and not filtered_symbols.empty:
        first = filtered_symbols.iloc[0]
        selected_exchange = str(first.get("exchange", ""))
        selected_symbol = _row_symbol(first)

    if (not selected_exchange or not selected_symbol) and not filtered.empty:
        first = filtered.iloc[0]
        selected_exchange = str(first.get("exchange", ""))
        selected_symbol = _row_symbol(first)

    if (not selected_exchange or not selected_symbol) and not scan_details.empty:
        first = scan_details.iloc[0]
        selected_exchange = str(first.get("exchange", ""))
        selected_symbol = _row_symbol(first)

    chart_html = ""
    chart_message = "Choose filters and run the weekly BUY screener to load charts."
    latest_summary = {"signal": "NONE", "date": "", "close": ""}

    if selected_exchange and selected_symbol:
        daily = storage.load_candles(selected_exchange, selected_symbol, "1D")
        if daily.empty:
            chart_message = f"No local OHLC candles found for {selected_exchange}:{selected_symbol}. Update OHLC data first."
        else:
            scan_timeframe = config.get("data", {}).get("scan_timeframe", "1W")
            strategy_cfg = config.get("strategy", {})
            weekly_anchor = strategy_cfg.get("weekly_anchor", "W-FRI")
            use_completed_weeks_only = bool(strategy_cfg.get("use_completed_weeks_only", True))

            strategy_input = daily
            if scan_timeframe == "1W":
                strategy_input = resample_daily_to_weekly(daily, weekly_anchor, use_completed_weeks_only)

            strategy_output = run_weekly_buy_sell(strategy_input, config)
            chart_html = build_signal_chart(strategy_output, selected_exchange, selected_symbol, height=620)
            latest_summary = latest_signal_summary(strategy_output)

    raw_table = raw.copy()
    if not raw_table.empty:
        raw_table["date"] = raw_table["date"].astype(str)
        raw_table = raw_table.sort_values("date", ascending=False)

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "app_name": config.get("app", {}).get("name", "Investment Screener"),
            "filtered": _records(filtered),
            "raw_count": len(raw),
            "filtered_count": len(filtered),
            "token_status": token_status(data_root),
            "scan_details": _records(scan_details),
            "scan_details_count": len(scan_details),
            "filtered_symbols": _records(filtered_symbols.drop(columns=["date_sort"], errors="ignore")),
            "dashboard_token": request.query_params.get("token", ""),
            "filter_link_suffix": filter_link_suffix,
            "selected_exchange": selected_exchange or "",
            "selected_symbol": selected_symbol or "",
            "stock_search": stock_search,
            "latest_summary": latest_summary,
            "chart_html": chart_html,
            "chart_message": chart_message,
            "all_signals": _records(raw_table),
            "selected_market_cap_bucket": selected_market_cap_bucket,
            "selected_min_market_cap": request.query_params.get("min_market_cap_cr", ""),
            "selected_max_market_cap": request.query_params.get("max_market_cap_cr", ""),
            "require_volume_confirmation": require_volume_confirmation,
            "require_trend_confirmation": require_trend_confirmation,
            "selected_return_metric": selected_return_metric,
            "selected_min_pair_return": request.query_params.get("min_pair_return_pct", ""),
            "signal_quality_warning": signal_quality_warning,
            "base_filter_query": _dashboard_filter_query(
                token=request.query_params.get("token", ""),
                stock_search=stock_search,
                market_cap_bucket=selected_market_cap_bucket,
                min_market_cap_cr=request.query_params.get("min_market_cap_cr", ""),
                max_market_cap_cr=request.query_params.get("max_market_cap_cr", ""),
            ),
            "full_filter_query": _dashboard_filter_query(
                token=request.query_params.get("token", ""),
                stock_search=stock_search,
                market_cap_bucket=selected_market_cap_bucket,
                min_market_cap_cr=request.query_params.get("min_market_cap_cr", ""),
                max_market_cap_cr=request.query_params.get("max_market_cap_cr", ""),
                require_volume_confirmation=require_volume_confirmation,
                require_trend_confirmation=require_trend_confirmation,
                return_metric=selected_return_metric if request.query_params.get("min_pair_return_pct", "") else "",
                min_pair_return_pct=request.query_params.get("min_pair_return_pct", ""),
            ),
            "market_cap_bounds": market_cap_bounds,
            "has_metadata": not metadata.empty,
            "scan_ran": request.query_params.get("scan_ran", ""),
            "scan_error": request.query_params.get("scan_error", ""),
            "scan_job": request.query_params.get("scan_job", ""),
            "telegram_sent": request.query_params.get("telegram_sent", ""),
            "telegram_error": request.query_params.get("telegram_error", ""),
            "symbols_scanned": request.query_params.get("symbols_scanned", ""),
            "active_filter_summary": " · ".join(active_filter_parts),
        },
    )


@app.get("/scan-status/{job_id}")
def scan_status(job_id: str) -> JSONResponse:
    job = _get_scan_job(job_id)
    if not job:
        return JSONResponse({"status": "missing", "error": "Scan job not found."}, status_code=404)
    return JSONResponse(job)


@app.post("/run-screener")
async def run_screener_from_dashboard(request: Request, background_tasks: BackgroundTasks) -> RedirectResponse:
    config = load_config()
    data_root = get_data_root(config)
    storage = Storage(data_root)
    form = await request.form()
    _ensure_market_cap_metadata(config, storage)

    dashboard_token = str(form.get("token", "")).strip()
    stock_search = str(form.get("stock_search", "")).strip()
    market_cap_bucket = str(form.get("market_cap_bucket", "")).strip()
    min_market_cap_text = str(form.get("min_market_cap_cr", "")).strip()
    max_market_cap_text = str(form.get("max_market_cap_cr", "")).strip()
    min_market_cap = _optional_float(min_market_cap_text)
    max_market_cap = _optional_float(max_market_cap_text)
    market_cap_filter_requested = bool(market_cap_bucket or min_market_cap_text or max_market_cap_text)

    params = []
    if dashboard_token:
        params.append(f"token={quote(dashboard_token)}")
    if stock_search:
        params.append(f"stock_search={quote(stock_search)}")
    if market_cap_bucket:
        params.append(f"market_cap_bucket={quote(market_cap_bucket)}")
    if min_market_cap_text:
        params.append(f"min_market_cap_cr={quote(min_market_cap_text)}")
    if max_market_cap_text:
        params.append(f"max_market_cap_cr={quote(max_market_cap_text)}")
    query_suffix = ("&" + "&".join(params)) if params else ""

    try:
        if market_cap_filter_requested and not _has_market_cap_metadata(storage):
            raise RuntimeError(
                "Market-cap metadata is missing. Open /stocks and click Fetch NSE Market Caps, "
                "or run python scripts/import_nse_market_caps.py."
            )

        scan_config = _manual_screener_config(
            config,
            storage,
            min_market_cap,
            max_market_cap,
            market_cap_bucket,
            stock_search,
        )
        job_id = uuid4().hex
        _set_scan_job(job_id, status="queued", phase="Queued", completed=0, total=0, percent=0)
        background_tasks.add_task(_run_screener_job, job_id, scan_config, query_suffix)
        redirect_url = f"/?scan_job={job_id}{query_suffix}"
    except Exception as exc:
        redirect_url = _scan_error_url(exc, query_suffix)
    return RedirectResponse(redirect_url, status_code=303)


@app.post("/telegram/send-buy-signals")
async def send_buy_signals_to_telegram(request: Request) -> RedirectResponse:
    config = load_config()
    data_root = get_data_root(config)
    storage = Storage(data_root)
    form = await request.form()
    _ensure_market_cap_metadata(config, storage)

    dashboard_token = str(form.get("token", "")).strip()
    stock_search = str(form.get("stock_search", "")).strip()
    market_cap_bucket = str(form.get("market_cap_bucket", "")).strip()
    min_market_cap_text = str(form.get("min_market_cap_cr", "")).strip()
    max_market_cap_text = str(form.get("max_market_cap_cr", "")).strip()
    require_volume_confirmation = str(form.get("require_volume_confirmation", "")).strip().lower() in {"1", "true", "on", "yes"}
    require_trend_confirmation = str(form.get("require_trend_confirmation", "")).strip().lower() in {"1", "true", "on", "yes"}
    return_metric = str(form.get("return_metric", "median_3")).strip() or "median_3"
    if return_metric not in {"last_1", "median_3"}:
        return_metric = "median_3"
    min_pair_return_text = str(form.get("min_pair_return_pct", "")).strip()
    min_market_cap = _optional_float(min_market_cap_text)
    max_market_cap = _optional_float(max_market_cap_text)
    min_pair_return = _optional_float(min_pair_return_text)

    filter_query = _dashboard_filter_query(
        token=dashboard_token,
        stock_search=stock_search,
        market_cap_bucket=market_cap_bucket,
        min_market_cap_cr=min_market_cap_text,
        max_market_cap_cr=max_market_cap_text,
        require_volume_confirmation=require_volume_confirmation,
        require_trend_confirmation=require_trend_confirmation,
        return_metric=return_metric if min_pair_return_text else "",
        min_pair_return_pct=min_pair_return_text,
    )

    try:
        visible_buy_signals = _load_visible_buy_signals(
            config,
            storage,
            stock_search,
            min_market_cap,
            max_market_cap,
            market_cap_bucket,
            require_volume_confirmation,
            require_trend_confirmation,
            return_metric,
            min_pair_return,
        )
        if visible_buy_signals.empty:
            raise RuntimeError("No weekly BUY signals are available to send.")

        visible_buy_signals = _apply_large_deal_markers(
            visible_buy_signals,
            _load_big_bull_deals(data_root),
        )
        filters_text = _buy_signal_filter_summary(
            stock_search,
            market_cap_bucket,
            min_market_cap_text,
            max_market_cap_text,
            require_volume_confirmation,
            require_trend_confirmation,
            return_metric,
            min_pair_return_text,
        )
        send_buy_signal_list_to_telegram(config, visible_buy_signals, filters_text=filters_text)
        status_query = "telegram_sent=1"
    except Exception as exc:
        status_query = f"telegram_error={quote(str(exc)[:500])}"

    redirect_query = "&".join([part for part in (status_query, filter_query) if part])
    return RedirectResponse(f"/?{redirect_query}", status_code=303)


@app.post("/telegram/send-gtt-list")
async def send_gtt_list_to_telegram(request: Request) -> RedirectResponse:
    config = load_config()
    data_root = get_data_root(config)
    storage = Storage(data_root)
    form = await request.form()
    _ensure_market_cap_metadata(config, storage)

    dashboard_token = str(form.get("token", "")).strip()
    stock_search = str(form.get("stock_search", "")).strip()
    market_cap_bucket = str(form.get("market_cap_bucket", "")).strip()
    min_market_cap_text = str(form.get("min_market_cap_cr", "")).strip()
    max_market_cap_text = str(form.get("max_market_cap_cr", "")).strip()
    min_market_cap = _optional_float(min_market_cap_text)
    max_market_cap = _optional_float(max_market_cap_text)
    open_buy_regime_only = str(form.get("open_buy_regime_only", "")).strip().lower() in {"1", "true", "on", "yes"}
    dashboard_buy_only = str(form.get("dashboard_buy_only", "")).strip().lower() in {"1", "true", "on", "yes"}
    fresh_weekly_buy_only = str(form.get("fresh_weekly_buy_only", "")).strip().lower() in {"1", "true", "on", "yes"}
    trend_only = str(form.get("trend_only", "")).strip().lower() in {"1", "true", "on", "yes"}

    filter_query = _gtt_filter_query(
        token=dashboard_token,
        stock_search=stock_search,
        market_cap_bucket=market_cap_bucket,
        min_market_cap_cr=min_market_cap_text,
        max_market_cap_cr=max_market_cap_text,
        open_buy_regime_only=open_buy_regime_only,
        dashboard_buy_only=dashboard_buy_only,
        fresh_weekly_buy_only=fresh_weekly_buy_only,
        trend_only=trend_only,
    )

    try:
        visible_gtt_stocks = _load_visible_gtt_stock_stats(
            config,
            storage,
            data_root,
            stock_search,
            min_market_cap,
            max_market_cap,
            market_cap_bucket,
            open_buy_regime_only,
            dashboard_buy_only,
            fresh_weekly_buy_only,
            trend_only,
        )
        if visible_gtt_stocks.empty:
            raise RuntimeError("No GTT stocks are available to send with the selected filters.")

        filters_text = _gtt_filter_summary(
            stock_search,
            market_cap_bucket,
            min_market_cap_text,
            max_market_cap_text,
            open_buy_regime_only,
            dashboard_buy_only,
            fresh_weekly_buy_only,
            trend_only,
        )
        send_gtt_stock_list_to_telegram(config, visible_gtt_stocks, filters_text=filters_text)
        status_query = "telegram_sent=1"
    except Exception as exc:
        status_query = f"telegram_error={quote(str(exc)[:500])}"

    redirect_query = "&".join([part for part in (status_query, filter_query) if part])
    return RedirectResponse(f"/gtt-gain-study?{redirect_query}", status_code=303)


@app.get("/signal-qa", response_class=HTMLResponse)
def signal_qa_page(request: Request) -> HTMLResponse:
    if not _is_allowed(request):
        return templates.TemplateResponse(
            "locked.html",
            {"request": request, "app_name": "Investment Screener"},
            status_code=401,
        )

    config = load_config()
    data_root = get_data_root(config)
    storage = Storage(data_root)
    _ensure_market_cap_metadata(config, storage)

    filtered = storage.load_signals("latest_filtered.csv")
    raw = storage.load_signals("latest_raw_signals.csv")
    scan_details = storage.load_signals("latest_scan_details.csv")
    instruments = storage.load_instruments()
    metadata = _combined_symbol_metadata(config, storage)

    filtered = _enrich_with_symbol_metadata(filtered, metadata, "symbol")
    raw = _enrich_with_symbol_metadata(raw, metadata, "symbol")
    scan_details = _enrich_with_symbol_metadata(scan_details, metadata, "symbol")

    report = build_signal_quality_report(raw, filtered, scan_details)
    symbol_search = request.query_params.get("symbol_search", "").strip()
    candidates = _signal_qa_candidates(filtered, scan_details, instruments, symbol_search)
    selected_exchange, selected_symbol = _selected_signal_qa_symbol(request, filtered, candidates, symbol_search)

    chart_html = ""
    latest_summary = {"signal": "NONE", "date": "", "close": ""}
    strategy_rows = pd.DataFrame()
    signal_rows = pd.DataFrame()
    qa_message = "Choose a stock to inspect signal generation."

    if selected_exchange and selected_symbol:
        daily = storage.load_candles(selected_exchange, selected_symbol, "1D")
        if daily.empty:
            qa_message = f"No local OHLC candles found for {selected_exchange}:{selected_symbol}."
        else:
            scan_timeframe = config.get("data", {}).get("scan_timeframe", "1W")
            strategy_cfg = config.get("strategy", {})
            weekly_anchor = strategy_cfg.get("weekly_anchor", "W-FRI")
            use_completed_weeks_only = bool(strategy_cfg.get("use_completed_weeks_only", True))
            strategy_input = daily
            if scan_timeframe == "1W":
                strategy_input = resample_daily_to_weekly(daily, weekly_anchor, use_completed_weeks_only)

            strategy_output = run_weekly_buy_sell(strategy_input, config)
            chart_html = build_signal_chart(strategy_output, selected_exchange, selected_symbol, height=480)
            latest_summary = latest_signal_summary(strategy_output)
            strategy_rows = strategy_rows_for_display(strategy_output, limit=120)
            signal_rows = strategy_rows[strategy_rows["signal"].isin(["BUY", "SELL"])].copy()
            qa_message = ""

    return templates.TemplateResponse(
        "signal_qa.html",
        {
            "request": request,
            "app_name": config.get("app", {}).get("name", "Investment Screener"),
            "dashboard_token": request.query_params.get("token", ""),
            "report": report,
            "checks": report["checks"],
            "issues": report["issues"][:200],
            "symbol_search": symbol_search,
            "candidates": _records(candidates.head(200)),
            "candidate_count": len(candidates),
            "selected_exchange": selected_exchange,
            "selected_symbol": selected_symbol,
            "latest_summary": latest_summary,
            "chart_html": chart_html,
            "qa_message": qa_message,
            "strategy_rows": _records(strategy_rows),
            "signal_rows": _records(signal_rows),
            "raw_count": len(raw),
            "filtered_count": len(filtered),
            "scan_details_count": len(scan_details),
        },
    )


@app.post("/watchlist/add/{exchange}/{symbol}")
def add_watchlist(exchange: str, symbol: str) -> RedirectResponse:
    config = load_config()
    storage = Storage(get_data_root(config))
    storage.add_to_watchlist(exchange, symbol)
    return RedirectResponse("/stocks?watchlist_added=1", status_code=303)


@app.post("/watchlist/remove/{exchange}/{symbol}")
def remove_watchlist(exchange: str, symbol: str) -> RedirectResponse:
    config = load_config()
    storage = Storage(get_data_root(config))
    storage.remove_from_watchlist(exchange, symbol)
    return RedirectResponse("/?watchlist_removed=1", status_code=303)


@app.post("/stocks/fetch")
def fetch_stocks() -> RedirectResponse:
    config = load_config()
    data_root = get_data_root(config)
    access_token = load_access_token(data_root)
    if not access_token:
        return RedirectResponse("/login?message=kite_token_missing", status_code=303)

    provider = KiteDataProvider(access_token=access_token)
    provider.validate_session()
    instruments = provider.instruments()
    storage = Storage(data_root)
    storage.save_instruments(instruments)

    return RedirectResponse("/stocks?refreshed=1", status_code=303)


@app.post("/stocks/fetch-market-caps")
async def fetch_market_caps(request: Request) -> RedirectResponse:
    config = load_config()
    data_root = get_data_root(config)
    storage = Storage(data_root)
    universe_cfg = config.get("universe", {})
    form = await request.form()
    dashboard_token = str(form.get("token", "")).strip()
    market_cap_cfg = universe_cfg.get("market_cap_source", {})
    local_path = _resolve_project_path(str(market_cap_cfg.get("local_path", "")))
    source_url = market_cap_cfg.get("url", DEFAULT_NSE_MARKET_CAP_URL)
    local_file = market_cap_cfg.get("local_file", "Average_MCAP_July2025ToDecember2025_20260102201101.xlsx")
    workbook_path = data_root / "instruments" / local_file

    try:
        bucket_cfg = universe_cfg.get("market_cap_buckets", {})
        small_max_cr = float(bucket_cfg.get("small_max_cr", 5000))
        mid_max_cr = float(bucket_cfg.get("mid_max_cr", 20000))
        market_cap_divisor = market_cap_cfg.get("market_cap_divisor")
        market_cap_divisor = float(market_cap_divisor) if market_cap_divisor else None
        if local_path.exists():
            metadata = load_nse_market_cap_excel(local_path, small_max_cr, mid_max_cr, market_cap_divisor)
        else:
            metadata = fetch_market_caps_from_nse_excel(
                source_url,
                workbook_path,
                small_max_cr,
                mid_max_cr,
                market_cap_divisor,
            )
        storage.save_symbol_metadata(metadata)
        redirect_url = f"/stocks?market_caps_refreshed=1&market_cap_rows={len(metadata)}"
    except Exception as exc:
        redirect_url = f"/stocks?market_cap_error={quote(str(exc)[:500])}"

    if dashboard_token:
        redirect_url += f"&token={quote(dashboard_token)}"
    return RedirectResponse(redirect_url, status_code=303)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request) -> HTMLResponse:
    if not _is_allowed(request):
        return templates.TemplateResponse(
            "locked.html",
            {"request": request, "app_name": "Investment Screener"},
            status_code=401,
        )

    config = load_config()
    data_root = get_data_root(config)

    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "app_name": config.get("app", {}).get("name", "Investment Screener"),
            "token_status": token_status(data_root),
            "dashboard_token": request.query_params.get("token", ""),
        },
    )


@app.get("/stocks", response_class=HTMLResponse)
def stocks_page(request: Request) -> HTMLResponse:
    if not _is_allowed(request):
        return templates.TemplateResponse(
            "locked.html",
            {"request": request, "app_name": "Investment Screener"},
            status_code=401,
        )

    config = load_config()
    data_root = get_data_root(config)
    storage = Storage(data_root)
    _ensure_market_cap_metadata(config, storage)
    instruments = storage.load_instruments()

    stocks = instruments.copy()
    watchlist = storage.load_watchlist()
    watchlist_keys = set(zip(watchlist["exchange"], watchlist["symbol"]))
    active_stock_filters = False
    market_cap_filter_requested = bool(
        request.query_params.get("market_cap_bucket", "").strip()
        or request.query_params.get("min_market_cap_cr", "").strip()
        or request.query_params.get("max_market_cap_cr", "").strip()
    )
    metadata = _combined_symbol_metadata(config, storage)
    if not stocks.empty:
        universe_cfg = config.get("universe", {})
        exchanges = set(universe_cfg.get("exchanges", ["NSE"]))
        instrument_types = set(universe_cfg.get("instrument_types", ["EQ"]))
        restrict_to_metadata_symbols = bool(universe_cfg.get("restrict_to_metadata_symbols", False))
        stocks = stocks[stocks["exchange"].isin(exchanges)]
        if "instrument_type" in stocks.columns:
            stocks = stocks[stocks["instrument_type"].isin(instrument_types)]
        if "segment" in stocks.columns:
            stocks = stocks[stocks["segment"].astype(str).str.upper() != "INDICES"]

        if not metadata.empty:
            stocks = stocks.copy()
            metadata_for_merge = metadata.copy()
            metadata_for_merge["metadata_symbol_key"] = metadata_for_merge["symbol"].apply(normalize_nse_symbol)
            stocks["symbol_key"] = stocks["tradingsymbol"].apply(normalize_nse_symbol)
            merge_type = "inner" if restrict_to_metadata_symbols else "left"
            stocks = stocks.merge(metadata_for_merge, left_on="symbol_key", right_on="metadata_symbol_key", how=merge_type)
            stocks = stocks.drop(columns=["symbol_key", "metadata_symbol_key", "symbol"], errors="ignore")
        elif restrict_to_metadata_symbols:
            stocks = stocks.iloc[0:0].copy()

        search = request.query_params.get("search", "").strip().upper()
        selected_industries = request.query_params.getlist("industry")
        selected_market_cap_bucket = request.query_params.get("market_cap_bucket", "").strip()
        min_market_cap = request.query_params.get("min_market_cap_cr", "").strip()
        max_market_cap = request.query_params.get("max_market_cap_cr", "").strip()
        active_stock_filters = bool(search or selected_industries or selected_market_cap_bucket or min_market_cap or max_market_cap)

        if search:
            symbol_match = stocks["tradingsymbol"].astype(str).str.upper().str.contains(search, na=False)
            name_match = stocks["name"].astype(str).str.upper().str.contains(search, na=False)
            stocks = stocks[symbol_match | name_match]

        if selected_industries and "industry" in stocks.columns:
            stocks = stocks[stocks["industry"].isin(selected_industries)]

        if selected_market_cap_bucket and "market_cap_bucket" in stocks.columns:
            stocks = stocks[stocks["market_cap_bucket"] == selected_market_cap_bucket]

        if min_market_cap and "market_cap_cr" in stocks.columns:
            stocks = stocks[pd.to_numeric(stocks["market_cap_cr"], errors="coerce") >= float(min_market_cap)]

        if max_market_cap and "market_cap_cr" in stocks.columns:
            stocks = stocks[pd.to_numeric(stocks["market_cap_cr"], errors="coerce") <= float(max_market_cap)]

        stocks["is_watchlisted"] = [
            (str(row.exchange).upper(), str(row.tradingsymbol).upper()) in watchlist_keys
            for row in stocks.itertuples()
        ]
        stocks = stocks.sort_values(["exchange", "tradingsymbol"])

    industry_options = []
    market_cap_bucket_options = []
    market_cap_bounds = {"min": "", "max": ""}
    has_market_cap_metadata = False
    if not metadata.empty:
        if "industry" in metadata.columns:
            industry_options = sorted([industry for industry in metadata["industry"].dropna().unique() if str(industry).strip()])
        if "market_cap_bucket" in metadata.columns:
            market_cap_bucket_options = sorted(
                [bucket for bucket in metadata["market_cap_bucket"].dropna().unique() if str(bucket).strip()]
            )
        if "market_cap_cr" in metadata.columns and metadata["market_cap_cr"].notna().any():
            has_market_cap_metadata = True
            market_cap_bounds = {
                "min": int(metadata["market_cap_cr"].min()),
                "max": int(metadata["market_cap_cr"].max()),
            }

    return templates.TemplateResponse(
        "stocks.html",
        {
            "request": request,
            "app_name": config.get("app", {}).get("name", "Investment Screener"),
            "stocks": _records(stocks),
            "stock_count": len(stocks),
            "dashboard_token": request.query_params.get("token", ""),
            "industry_options": industry_options,
            "selected_industries": request.query_params.getlist("industry"),
            "market_cap_bucket_options": market_cap_bucket_options,
            "selected_market_cap_bucket": request.query_params.get("market_cap_bucket", ""),
            "market_cap_bounds": market_cap_bounds,
            "selected_min_market_cap": request.query_params.get("min_market_cap_cr", ""),
            "selected_max_market_cap": request.query_params.get("max_market_cap_cr", ""),
            "search": request.query_params.get("search", ""),
            "has_metadata": not metadata.empty,
            "has_market_cap_metadata": has_market_cap_metadata,
            "active_stock_filters": active_stock_filters,
            "market_cap_filter_requested": market_cap_filter_requested,
        },
    )


@app.get("/big-bull-deals", response_class=HTMLResponse)
def big_bull_deals_page(request: Request) -> HTMLResponse:
    if not _is_allowed(request):
        return templates.TemplateResponse(
            "locked.html",
            {"request": request, "app_name": "Investment Screener"},
            status_code=401,
        )

    config = load_config()
    data_root = get_data_root(config)
    default_from, default_to = default_last_7_days_range()
    from_date = request.query_params.get("from_date", default_from.isoformat())
    to_date = request.query_params.get("to_date", default_to.isoformat())
    action = request.query_params.get("action", "").strip().upper()
    investor = request.query_params.get("investor", "").strip()
    symbol = request.query_params.get("symbol", "").strip()

    try:
        all_recent_rows = SupabaseStore().list_large_deals(limit=5000)
        investor_options = sorted(
            {
                str(row.get("client_name", "")).strip()
                for row in all_recent_rows
                if str(row.get("client_name", "")).strip()
            }
        )
        rows = SupabaseStore().list_large_deals(
            limit=1000,
            from_date=from_date,
            to_date=to_date,
            action=action if action in {"BUY", "SELL"} else None,
            investor=investor or None,
            symbol=symbol or None,
        )
        deals = pd.DataFrame(rows)
    except Exception as exc:
        print(f"Supabase large deals unavailable; falling back to CSV: {exc}")
        investor_options = []
        deals = _load_big_bull_deals(data_root)

    if not deals.empty:
        if "date" in deals.columns:
            deals = deals.sort_values("date", ascending=False)
        if "deal_date" in deals.columns:
            deals = deals.sort_values("deal_date", ascending=False)

    return templates.TemplateResponse(
        "big_bull_deals.html",
        {
            "request": request,
            "app_name": config.get("app", {}).get("name", "Investment Screener"),
            "dashboard_token": request.query_params.get("token", ""),
            "deals": _records(deals),
            "deal_count": len(deals),
            "action": request.query_params.get("action", ""),
            "investor": request.query_params.get("investor", ""),
            "symbol": request.query_params.get("symbol", ""),
            "from_date": from_date,
            "to_date": to_date,
            "investor_options": investor_options,
        },
    )




@app.get("/charts/{exchange}/{symbol}", response_class=HTMLResponse)
def stock_chart(request: Request, exchange: str, symbol: str) -> HTMLResponse:
    if not _is_allowed(request):
        return templates.TemplateResponse(
            "locked.html",
            {"request": request, "app_name": "Investment Screener"},
            status_code=401,
        )

    config = load_config()
    data_root = get_data_root(config)
    storage = Storage(data_root)
    daily = storage.load_candles(exchange, symbol, "1D")

    scan_timeframe = config.get("data", {}).get("scan_timeframe", "1W")
    strategy_cfg = config.get("strategy", {})
    weekly_anchor = strategy_cfg.get("weekly_anchor", "W-FRI")
    use_completed_weeks_only = bool(strategy_cfg.get("use_completed_weeks_only", True))

    strategy_input = daily
    if scan_timeframe == "1W":
        strategy_input = resample_daily_to_weekly(daily, weekly_anchor, use_completed_weeks_only)

    strategy_output = run_weekly_buy_sell(strategy_input, config)
    chart_html = build_signal_chart(strategy_output, exchange, symbol)
    latest_summary = latest_signal_summary(strategy_output)

    return templates.TemplateResponse(
        "chart.html",
        {
            "request": request,
            "app_name": config.get("app", {}).get("name", "Investment Screener"),
            "exchange": exchange,
            "symbol": symbol,
            "scan_timeframe": scan_timeframe,
            "chart_html": chart_html,
            "latest_summary": latest_summary,
            "dashboard_token": request.query_params.get("token", ""),
        },
    )


@app.get("/auth/kite/login")
def kite_login() -> RedirectResponse:
    api_key = require_env("KITE_API_KEY")
    kite = KiteConnect(api_key=api_key)
    return RedirectResponse(kite.login_url())


@app.get("/auth/kite/callback", response_class=HTMLResponse)
def kite_callback(request: Request, request_token: str | None = None, status: str | None = None) -> HTMLResponse:
    if status and status != "success":
        return templates.TemplateResponse(
            "auth_result.html",
            {
                "request": request,
                "app_name": "Investment Screener",
                "success": False,
                "message": f"Kite login did not complete successfully. Status: {status}",
            },
            status_code=400,
        )

    if not request_token:
        return templates.TemplateResponse(
            "auth_result.html",
            {
                "request": request,
                "app_name": "Investment Screener",
                "success": False,
                "message": "Kite callback did not include request_token.",
            },
            status_code=400,
        )

    config = load_config()
    data_root = get_data_root(config)
    api_key = require_env("KITE_API_KEY")
    api_secret = require_env("KITE_API_SECRET")

    kite = KiteConnect(api_key=api_key)
    session = kite.generate_session(request_token, api_secret=api_secret)
    access_token = session["access_token"]
    kite.set_access_token(access_token)
    profile = kite.profile()
    path = save_access_token(data_root, access_token, profile)

    return templates.TemplateResponse(
        "auth_result.html",
        {
            "request": request,
            "app_name": config.get("app", {}).get("name", "Investment Screener"),
            "success": True,
            "message": f"Kite access token saved to {path}. The next scan will use it automatically.",
        },
    )
