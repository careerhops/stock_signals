from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
import json
import os
from pathlib import Path
from threading import Lock
import time
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from kiteconnect import KiteConnect

from stock_screener.auth.kite_token import load_access_token, save_access_token, token_status
from stock_screener.backtest import run_buy_sell_backtest, run_buy_sell_backtest_for_symbols, save_backtest_outputs
from stock_screener.backtest_report import write_backtest_workbook
from stock_screener.config import get_data_root, load_config, require_env
from stock_screener.data.kite import KiteDataProvider
from stock_screener.data.nse_market_cap import (
    DEFAULT_NSE_MARKET_CAP_URL,
    fetch_market_caps_from_nse_excel,
    load_nse_market_cap_excel,
)
from stock_screener.adx_di_study import (
    _is_excluded_adx_symbol,
    load_adx_di_outputs,
    run_adx_di_study,
    save_adx_di_outputs,
)
from stock_screener.data.storage import Storage
from stock_screener.data.supabase_store import SupabaseStore
from stock_screener.gtt_gain_report import write_gtt_gain_workbook
from stock_screener.gtt_gain_study import (
    _latest_signal_context,
    _merge_latest_context,
    _prepare_daily,
    load_gtt_gain_outputs,
    run_gtt_gain_study,
    save_gtt_gain_outputs,
)
from stock_screener.google_sheets import (
    DEFAULT_WORKSHEET_TITLE,
    batch_update_google_sheet_values,
    build_google_oauth_login_url,
    exchange_google_oauth_code,
    export_weekly_buy_tracker_to_google_sheet,
    google_oauth_status,
    has_google_sheets_credentials,
    load_google_sheets_settings,
    load_google_oauth_client,
    read_google_sheet_values,
    save_google_oauth_client,
    save_google_sheet_target,
)
from stock_screener.jobs.daily_scan import daily_signal_config, run_daily_scan
from stock_screener.notifications.telegram import send_buy_signal_list_to_telegram, send_gtt_stock_list_to_telegram
from stock_screener.resample import resample_daily_to_weekly
from stock_screener.rotation_study import load_rotation_study_outputs, run_rotation_study, save_rotation_study_outputs
from stock_screener.signal_outcome_report import write_signal_outcome_workbook
from stock_screener.signal_outcome_study import (
    load_signal_outcome_outputs,
    run_signal_outcome_study,
    save_signal_outcome_outputs,
)
from stock_screener.swing_trade_study import (
    load_swing_trade_outputs,
    run_swing_trade_study,
    save_swing_trade_outputs,
)
from stock_screener.signal_qa import build_signal_quality_report, strategy_rows_for_display
from stock_screener.strategy.technical_ratings import latest_technical_rating
from stock_screener.strategy.weekly_shortlist import (
    DEFAULT_BENCHMARK_SYMBOL as SHORTLIST_DEFAULT_BENCHMARK_SYMBOL,
    benchmark_symbol_for_industry,
    enrich_weekly_signal_shortlist_frame,
)
from stock_screener.strategy.weekly_buy_sell import run_weekly_buy_sell
from stock_screener.strategy_lab_study import (
    DEFAULT_START_DATE as STRATEGY_LAB_DEFAULT_START_DATE,
    load_strategy_lab_outputs,
    run_strategy_lab_study,
    save_strategy_lab_outputs,
)
from stock_screener.sensitivity_overlap_study import (
    DEFAULT_START_DATE as SENSITIVITY_OVERLAP_DEFAULT_START_DATE,
    build_next_week_conversion_markers,
    load_sensitivity_overlap_outputs,
    run_sensitivity_overlap_study,
    save_sensitivity_overlap_outputs,
)
from stock_screener.qm_quality_study import (
    DEFAULT_BUY_END_DATE as QM_QUALITY_DEFAULT_END_DATE,
    DEFAULT_BUY_START_DATE as QM_QUALITY_DEFAULT_START_DATE,
    load_qm_quality_outputs,
    run_qm_quality_study,
    save_qm_quality_outputs,
)
from stock_screener.resistance_breaks_study import (
    load_resistance_breaks_outputs,
    run_resistance_breaks_study,
    save_resistance_breaks_outputs,
)
from stock_screener.symbols import normalize_nse_symbol
from stock_screener.minervini_sheet_sync import (
    DEFAULT_WORKSHEET_TITLE as MINERVINI_SHEET_DEFAULT_WORKSHEET_TITLE,
    load_minervini_sheet_sync_outputs,
    run_minervini_sheet_sync,
    save_minervini_sheet_sync_outputs,
)
from stock_screener.minervini_quality_study import (
    DEFAULT_BENCHMARK_SYMBOL as MINERVINI_QUALITY_DEFAULT_BENCHMARK,
    DEFAULT_SCORE_THRESHOLD as MINERVINI_QUALITY_DEFAULT_THRESHOLD,
    load_minervini_quality_outputs,
    run_minervini_quality_study,
    save_minervini_quality_outputs,
)
from stock_screener.minervini_di_divergence_study import (
    DEFAULT_ADX_LENGTH as MINERVINI_DI_DEFAULT_ADX_LENGTH,
    DEFAULT_DIVERGENCE_DAYS as MINERVINI_DI_DEFAULT_DIVERGENCE_DAYS,
    DEFAULT_MIN_SCORE as MINERVINI_DI_DEFAULT_MIN_SCORE,
    load_minervini_di_divergence_outputs,
    run_minervini_di_divergence_study,
    save_minervini_di_divergence_outputs,
)
from stock_screener.universe import build_universe
from stock_screener.weekday_pressure_study import (
    WEEKDAY_ORDER,
    WeekdayPressureStudyResult,
    load_weekday_pressure_outputs,
    run_weekday_pressure_study,
    save_weekday_pressure_outputs,
)
from stock_screener.weekly_buy_tracker_study import (
    DEFAULT_START_DATE as WEEKLY_BUY_TRACKER_DEFAULT_START_DATE,
    load_weekly_buy_tracker_outputs,
    run_weekly_buy_tracker_study,
    save_weekly_buy_tracker_outputs,
)
from stock_screener.volume_burst_study import (
    load_volume_burst_outputs,
    run_volume_burst_study,
    save_volume_burst_outputs,
)


WEEKLY_BUY_GAINS_DEFAULT_START_DATE = "2026-04-01"
from stock_screener.jobs.large_deals import (
    default_last_7_days_range,
    fetch_and_store_current_large_deals,
)
from stock_screener.web.charts import (
    build_adx_di_chart,
    build_sector_mix_pie_chart,
    build_gtt_opportunity_chart,
    build_rotation_group_chart,
    build_signal_chart,
    latest_signal_summary,
)


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


def _google_oauth_redirect_uri(request: Request) -> str:
    env_redirect = str(os.getenv("GOOGLE_OAUTH_REDIRECT_URI", "")).strip()
    if env_redirect:
        return env_redirect
    callback = str(request.url_for("google_sheets_callback"))
    parts = urlsplit(callback)
    hostname = parts.hostname or ""
    if hostname == "0.0.0.0":
        netloc = parts.netloc.replace("0.0.0.0", "127.0.0.1")
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    return callback


def _append_query_param(url: str, param: str) -> str:
    separator = "&" if "?" in str(url) else "?"
    return f"{url}{separator}{param}"


templates.env.filters["number"] = _template_number


def _template_ratio(value: Any, digits: int = 2) -> str:
    if value is None or value == "":
        return ""
    try:
        if pd.isna(value):
            return ""
        numeric = float(value)
        if numeric == float("inf"):
            return "No losses"
        if numeric == float("-inf"):
            return "No wins"
        return f"{numeric:.{int(digits)}f}"
    except (TypeError, ValueError):
        return str(value)


templates.env.filters["ratio"] = _template_ratio

SCAN_JOBS: dict[str, dict[str, Any]] = {}
SCAN_JOBS_LOCK = Lock()
SCAN_JOBS_DIR = BASE_DIR / "data" / "scan_jobs"
SCAN_JOBS_DIR.mkdir(parents=True, exist_ok=True)
BIG_BULL_DEALS_CACHE: dict[str, Any] = {
    "fetched_at": 0.0,
    "rows": pd.DataFrame(),
}
BIG_BULL_DEALS_CACHE_LOCK = Lock()
BIG_BULL_DEALS_CACHE_TTL_SECONDS = 300

GTT_PEAK_SPEED_BUCKETS = [
    "Within 30 days",
    "31-60 days",
    "61-90 days",
    "91-180 days",
    "181-365 days",
    "Over 1 year",
    "NA",
]
GTT_TECHNICAL_RATING_FILTERS = ["Strong Buy", "Buy", "Neutral", "Sell", "Strong Sell"]
GTT_TABLE_RENDER_LIMIT = 250


def _normalize_gtt_technical_rating_statuses(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        raw_values = [values]
    else:
        try:
            raw_values = list(values)
        except TypeError:
            raw_values = [str(values)]

    allowed = {value.upper(): value for value in GTT_TECHNICAL_RATING_FILTERS}
    normalized: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        text = str(value or "").strip()
        if not text:
            continue
        canonical = allowed.get(text.upper())
        if canonical and canonical not in seen:
            normalized.append(canonical)
            seen.add(canonical)
    return normalized


def _scan_job_path(job_id: str) -> Path:
    return SCAN_JOBS_DIR / f"{job_id}.json"


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _set_scan_job(job_id: str, **updates: Any) -> None:
    with SCAN_JOBS_LOCK:
        current = SCAN_JOBS.setdefault(job_id, {})
        current.update(updates)
        safe_payload = _json_safe(current)
    _scan_job_path(job_id).write_text(json.dumps(safe_payload), encoding="utf-8")


def _get_scan_job(job_id: str) -> dict[str, Any]:
    with SCAN_JOBS_LOCK:
        current = SCAN_JOBS.get(job_id, {})
        if current:
            return dict(current)
    path = _scan_job_path(job_id)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    with SCAN_JOBS_LOCK:
        SCAN_JOBS[job_id] = dict(payload)
    return dict(payload)


def _has_meaningful_text(series: pd.Series) -> pd.Series:
    return ~series.astype(str).str.strip().str.upper().isin({"", "NA", "NAN", "NONE", "<NA>"})


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

    try:
        metadata = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
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


def _sector_label_from_industry(industry: Any) -> str:
    industry_text = str(industry or "").strip()
    if not industry_text:
        return "Unclassified"
    benchmark_symbol = benchmark_symbol_for_industry(industry_text)
    if benchmark_symbol and benchmark_symbol != SHORTLIST_DEFAULT_BENCHMARK_SYMBOL:
        return benchmark_symbol.replace("NIFTY ", "").strip() or industry_text
    return industry_text


def _build_adx_di_sector_views(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return frame.copy(), pd.DataFrame(), pd.DataFrame()

    working = frame.copy()
    if "industry" not in working.columns:
        working["industry"] = ""
    working["industry"] = working["industry"].fillna("").astype(str).str.strip()
    working["sector_label"] = working["industry"].apply(_sector_label_from_industry)

    for column in ("quality_score", "relative_strength_spread_pct", "cross_volume_ratio", "market_cap_cr"):
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
    di_plus = pd.to_numeric(working.get("latest_di_plus"), errors="coerce")
    di_minus = pd.to_numeric(working.get("latest_di_minus"), errors="coerce")
    working["di_plus_minus_range"] = di_plus - di_minus
    if "symbol_display" not in working.columns and "symbol" in working.columns:
        working["symbol_display"] = working["symbol"].map(_display_symbol)

    sorted_working = working.sort_values(
        [
            "sector_label",
            "quality_score",
            "relative_strength_spread_pct",
            "cross_volume_ratio",
            "di_plus_minus_range",
            "market_cap_cr",
            "symbol",
        ],
        ascending=[True, False, False, False, False, False, True],
        na_position="last",
    ).copy()
    sorted_working["sector_rank"] = sorted_working.groupby("sector_label").cumcount() + 1

    leaders = sorted_working[sorted_working["sector_rank"] <= 3].copy()
    leaders["leader_label"] = leaders.apply(
        lambda row: f"{row.get('symbol_display', row.get('symbol', ''))} ({int(row.get('quality_score', 0)) if pd.notna(row.get('quality_score')) else 0})",
        axis=1,
    )
    leader_rollup = (
        leaders.groupby("sector_label", dropna=False)["leader_label"]
        .apply(lambda values: ", ".join([str(value) for value in values if str(value).strip()]))
        .reset_index(name="leading_symbols_csv")
    )

    sector_summary = (
        working.groupby("sector_label", dropna=False)
        .agg(
            stock_count=("symbol", "size"),
            avg_quality_score=("quality_score", "mean"),
            avg_rs_spread_pct=("relative_strength_spread_pct", "mean"),
            avg_cross_volume_ratio=("cross_volume_ratio", "mean"),
        )
        .reset_index()
    )
    total_count = int(len(working))
    sector_summary["share_pct"] = np.where(
        total_count > 0,
        sector_summary["stock_count"].astype(float) * 100.0 / float(total_count),
        np.nan,
    )
    sector_summary = sector_summary.merge(leader_rollup, on="sector_label", how="left")
    sector_summary = sector_summary.sort_values(
        ["stock_count", "avg_quality_score", "avg_rs_spread_pct", "sector_label"],
        ascending=[False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)

    leaders = leaders.sort_values(["sector_label", "sector_rank"], ascending=[True, True], na_position="last").reset_index(drop=True)
    return working, sector_summary, leaders


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


def _request_int(
    request: Request,
    name: str,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    value = request.query_params.get(name, "").strip()
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _apply_request_sensitivity(config: dict[str, Any], request: Request) -> tuple[dict[str, Any], int, int]:
    base_sensitivity = int(config.get("strategy", {}).get("sensitivity", 3))
    selected_sensitivity = _request_int(request, "sensitivity", base_sensitivity, minimum=1, maximum=20)
    if selected_sensitivity == base_sensitivity:
        return config, base_sensitivity, selected_sensitivity
    adjusted = deepcopy(config)
    adjusted.setdefault("strategy", {})["sensitivity"] = selected_sensitivity
    return adjusted, base_sensitivity, selected_sensitivity


def _parse_sensitivity_text(value: str, default: int | None = None) -> int | None:
    value = value.strip()
    if not value:
        return default
    try:
        return max(1, min(20, int(value)))
    except ValueError:
        return default


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


def _apply_cmp_filters(
    frame: pd.DataFrame,
    min_cmp: float | None,
    max_cmp: float | None,
    price_column: str = "close",
) -> pd.DataFrame:
    if frame.empty or (min_cmp is None and max_cmp is None):
        return frame
    if price_column not in frame.columns:
        return frame.iloc[0:0].copy()

    filtered = frame.copy()
    prices = pd.to_numeric(filtered[price_column], errors="coerce")
    if min_cmp is not None:
        filtered = filtered[prices >= float(min_cmp)]
        prices = pd.to_numeric(filtered[price_column], errors="coerce")
    if max_cmp is not None:
        filtered = filtered[prices <= float(max_cmp)]
    return filtered


def _enrich_with_latest_daily_close(
    frame: pd.DataFrame,
    scan_details: pd.DataFrame,
    storage: Storage | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    symbol_column = _symbol_column(frame)
    if not symbol_column or "exchange" not in frame.columns:
        return frame

    if "latest_close" in frame.columns and "latest_close_date" in frame.columns:
        latest_close = pd.to_numeric(frame["latest_close"], errors="coerce")
        latest_date = frame["latest_close_date"].astype(str).str.strip()
        if latest_close.notna().all() and latest_date.ne("").all() and latest_date.ne("NaT").all():
            return frame

    merged = frame.copy()
    if not scan_details.empty and "symbol" in scan_details.columns and "exchange" in scan_details.columns:
        available = scan_details.copy()
        merge_columns = ["exchange", "symbol"]
        extra_columns = [column for column in ("latest_close", "latest_close_date") if column in available.columns]
        if extra_columns:
            available["exchange"] = available["exchange"].astype(str).str.upper()
            available["symbol"] = available["symbol"].astype(str).str.upper()
            available = available[merge_columns + extra_columns].drop_duplicates(subset=merge_columns, keep="last")

            working = frame.copy()
            working["exchange"] = working["exchange"].astype(str).str.upper()
            working[symbol_column] = working[symbol_column].astype(str).str.upper()
            merged = working.merge(
                available.rename(columns={"symbol": symbol_column}),
                on=["exchange", symbol_column],
                how="left",
                suffixes=("", "_scan"),
            )

            for column in extra_columns:
                scan_column = f"{column}_scan"
                if column not in merged.columns:
                    merged[column] = pd.NA
                if scan_column in merged.columns:
                    merged[column] = merged[column].combine_first(merged[scan_column])
                    merged = merged.drop(columns=[scan_column], errors="ignore")

    if storage is None or merged.empty:
        return merged

    if "latest_close" not in merged.columns:
        merged["latest_close"] = pd.NA
    if "latest_close_date" not in merged.columns:
        merged["latest_close_date"] = pd.NA

    latest_close = pd.to_numeric(merged["latest_close"], errors="coerce")
    latest_date = merged["latest_close_date"].astype(str).str.strip()
    missing_mask = latest_close.isna() | latest_date.eq("") | latest_date.eq("NaT")
    if not missing_mask.any():
        return merged

    symbol_column = _symbol_column(merged)
    if not symbol_column or "exchange" not in merged.columns:
        return merged

    missing_rows = merged.loc[missing_mask, ["exchange", symbol_column]].copy()
    missing_rows["exchange"] = missing_rows["exchange"].astype(str).str.strip().str.upper()
    missing_rows[symbol_column] = missing_rows[symbol_column].astype(str).str.strip().str.upper()
    unique_pairs = missing_rows.drop_duplicates(subset=["exchange", symbol_column])

    latest_map: dict[tuple[str, str], tuple[object, object]] = {}
    for _, row in unique_pairs.iterrows():
        exchange = str(row.get("exchange", "")).strip().upper()
        symbol = str(row.get(symbol_column, "")).strip().upper()
        if not exchange or not symbol:
            continue
        daily = storage.load_candles(exchange, symbol, "1D")
        if daily.empty or not {"date", "close"}.issubset(daily.columns):
            continue
        latest_daily = daily.copy()
        latest_daily["date"] = pd.to_datetime(latest_daily["date"], errors="coerce")
        latest_daily = latest_daily.dropna(subset=["date"]).sort_values("date")
        if latest_daily.empty:
            continue
        latest_row = latest_daily.iloc[-1]
        latest_close_value = pd.to_numeric(pd.Series([latest_row.get("close")]), errors="coerce").iloc[0]
        latest_date_value = latest_row.get("date", pd.NA)
        latest_map[(exchange, symbol)] = (latest_close_value, latest_date_value)

    if not latest_map:
        return merged

    for index, row in merged.loc[missing_mask].iterrows():
        key = (
            str(row.get("exchange", "")).strip().upper(),
            str(row.get(symbol_column, "")).strip().upper(),
        )
        if key not in latest_map:
            continue
        latest_close_value, latest_date_value = latest_map[key]
        merged.at[index, "latest_close"] = latest_close_value
        merged.at[index, "latest_close_date"] = latest_date_value

    return merged


def _refresh_live_cmp(
    frame: pd.DataFrame,
    data_root: Path,
    max_symbols: int = 250,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    symbol_column = _symbol_column(frame)
    if not symbol_column or "exchange" not in frame.columns:
        return frame

    working = frame.copy()
    pairs = working[["exchange", symbol_column]].dropna(subset=["exchange", symbol_column]).copy()
    if pairs.empty:
        return working
    pairs["exchange"] = pairs["exchange"].astype(str).str.strip().str.upper()
    pairs[symbol_column] = pairs[symbol_column].astype(str).str.strip().str.upper()
    pairs = pairs.drop_duplicates(subset=["exchange", symbol_column])
    if len(pairs) > max_symbols:
        return working

    access_token = load_access_token(data_root)
    if not access_token:
        return working

    instruments = [f"{row['exchange']}:{row[symbol_column]}" for _, row in pairs.iterrows()]
    try:
        provider = KiteDataProvider(access_token=access_token)
        live_prices = provider.ltp(instruments)
    except Exception:
        return working
    if not live_prices:
        return working

    quote_map = {(key.split(":", 1)[0].upper(), key.split(":", 1)[1].upper()): value for key, value in live_prices.items() if ":" in key}
    if not quote_map:
        return working

    if "latest_close" not in working.columns:
        working["latest_close"] = pd.NA
    if "latest_close_date" not in working.columns:
        working["latest_close_date"] = pd.NA
    working["cmp_source"] = working.get("cmp_source", pd.Series(index=working.index, dtype="object"))

    now = pd.Timestamp.now()
    for index, row in working.iterrows():
        key = (
            str(row.get("exchange", "")).strip().upper(),
            str(row.get(symbol_column, "")).strip().upper(),
        )
        if key not in quote_map:
            continue
        working.at[index, "latest_close"] = float(quote_map[key])
        working.at[index, "latest_close_date"] = now
        working.at[index, "cmp_source"] = "live"
    return working


def _truthy_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _truthy_param(values: Any, default: bool = False) -> bool:
    if values is None:
        return default
    if isinstance(values, str):
        candidates = [values]
    else:
        try:
            candidates = list(values)
        except TypeError:
            candidates = [values]
    if not candidates:
        return default
    return any(str(value).strip().lower() in {"1", "true", "yes", "y", "on"} for value in candidates)


def _apply_signal_quality_filters(
    frame: pd.DataFrame,
    require_volume_confirmation: bool,
    require_trend_confirmation: bool,
    require_obv_confirmation: bool,
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
        required = {"daily_ema_stack_confirmation", "trend_confirmation"}
        if not required.intersection(filtered.columns):
            return filtered.iloc[0:0].copy()
        column = "daily_ema_stack_confirmation" if "daily_ema_stack_confirmation" in filtered.columns else "trend_confirmation"
        filtered = filtered[_truthy_series(filtered[column])]

    if require_obv_confirmation:
        required = {"daily_obv_confirmation", "obv_confirmation"}
        if not required.intersection(filtered.columns):
            return filtered.iloc[0:0].copy()
        column = "daily_obv_confirmation" if "daily_obv_confirmation" in filtered.columns else "obv_confirmation"
        filtered = filtered[_truthy_series(filtered[column])]

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


def _apply_weekly_shortlist_filters(
    frame: pd.DataFrame,
    require_htf_alignment: bool,
    min_breakout_volume_ratio: float | None,
    require_relative_strength: bool,
    min_relative_strength_pct: float | None,
    max_distance_from_demand_pct: float | None,
    min_risk_reward_ratio: float | None,
) -> pd.DataFrame:
    if frame.empty:
        return frame

    filtered = frame.copy()
    if require_htf_alignment:
        if "htf_alignment_confirmation" not in filtered.columns:
            return filtered.iloc[0:0].copy()
        filtered = filtered[_truthy_series(filtered["htf_alignment_confirmation"])]

    if min_breakout_volume_ratio is not None:
        if "volume_confirmation_ratio" not in filtered.columns:
            return filtered.iloc[0:0].copy()
        filtered = filtered[pd.to_numeric(filtered["volume_confirmation_ratio"], errors="coerce") >= float(min_breakout_volume_ratio)]

    if require_relative_strength or min_relative_strength_pct is not None:
        if "relative_strength_12w_pct" not in filtered.columns:
            return filtered.iloc[0:0].copy()
        threshold = float(min_relative_strength_pct) if min_relative_strength_pct is not None else 0.0
        filtered = filtered[pd.to_numeric(filtered["relative_strength_12w_pct"], errors="coerce") >= threshold]

    if max_distance_from_demand_pct is not None:
        if "distance_from_demand_pct" not in filtered.columns:
            return filtered.iloc[0:0].copy()
        distance = pd.to_numeric(filtered["distance_from_demand_pct"], errors="coerce")
        filtered = filtered[distance <= float(max_distance_from_demand_pct)]

    if min_risk_reward_ratio is not None:
        if "risk_reward_ratio" not in filtered.columns:
            return filtered.iloc[0:0].copy()
        rr = pd.to_numeric(filtered["risk_reward_ratio"], errors="coerce")
        filtered = filtered[rr >= float(min_risk_reward_ratio)]

    return filtered


def _weekly_shortlist_filter_warning(
    frame: pd.DataFrame,
    require_htf_alignment: bool,
    min_breakout_volume_ratio: float | None,
    require_relative_strength: bool,
    min_relative_strength_pct: float | None,
    max_distance_from_demand_pct: float | None,
    min_risk_reward_ratio: float | None,
) -> str:
    missing_columns = []
    if require_htf_alignment and "htf_alignment_confirmation" not in frame.columns:
        missing_columns.append("higher timeframe alignment")
    if min_breakout_volume_ratio is not None and "volume_confirmation_ratio" not in frame.columns:
        missing_columns.append("breakout volume ratio")
    if (require_relative_strength or min_relative_strength_pct is not None) and "relative_strength_12w_pct" not in frame.columns:
        missing_columns.append("relative strength vs benchmark")
    if max_distance_from_demand_pct is not None and "distance_from_demand_pct" not in frame.columns:
        missing_columns.append("distance from demand zone")
    if min_risk_reward_ratio is not None and "risk_reward_ratio" not in frame.columns:
        missing_columns.append("risk-reward ratio")
    if not missing_columns:
        return ""
    return (
        "Shortlist columns are missing from the weekly BUY list. "
        "Run the screener after a Kite refresh so the shortlist metrics are rebuilt."
    )


def _signal_quality_filter_warning(
    frame: pd.DataFrame,
    require_volume_confirmation: bool,
    require_trend_confirmation: bool,
    require_obv_confirmation: bool,
    min_pair_return: float | None,
) -> str:
    missing_columns = []
    if require_volume_confirmation and "volume_confirmation" not in frame.columns:
        missing_columns.append("volume confirmation")
    if require_trend_confirmation and not {"daily_ema_stack_confirmation", "trend_confirmation"}.intersection(frame.columns):
        missing_columns.append("daily EMA stack confirmation")
    if require_obv_confirmation and not {"daily_obv_confirmation", "obv_confirmation"}.intersection(frame.columns):
        missing_columns.append("OBV confirmation")
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
    fresh_daily_buy_only: bool = False,
    fresh_daily_buy_symbols: set[str] | None = None,
    require_volume_confirmation: bool = False,
    require_obv_confirmation: bool = False,
    technical_rating_statuses: list[str] | None = None,
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

    if fresh_daily_buy_only:
        filtered = _filter_by_symbols(filtered, fresh_daily_buy_symbols or set())

    if require_volume_confirmation:
        if "volume_confirmation" not in filtered.columns:
            return filtered.iloc[0:0].copy()
        filtered = filtered[_truthy_series(filtered["volume_confirmation"])]

    if require_obv_confirmation:
        required = {"daily_obv_confirmation", "obv_confirmation"}
        if not required.intersection(filtered.columns):
            return filtered.iloc[0:0].copy()
        column = "daily_obv_confirmation" if "daily_obv_confirmation" in filtered.columns else "obv_confirmation"
        filtered = filtered[_truthy_series(filtered[column])]

    normalized_statuses = _normalize_gtt_technical_rating_statuses(technical_rating_statuses or [])
    if normalized_statuses:
        if "weekly_technical_rating_status" not in filtered.columns:
            return filtered.iloc[0:0].copy()
        filtered = filtered[
            filtered["weekly_technical_rating_status"].astype(str).str.strip().str.upper().isin(
                [status.upper() for status in normalized_statuses]
            )
        ]

    if trend_only:
        required = {"daily_ema_stack_confirmation", "trend_confirmation"}
        if not required.intersection(filtered.columns):
            return filtered.iloc[0:0].copy()
        column = "daily_ema_stack_confirmation" if "daily_ema_stack_confirmation" in filtered.columns else "trend_confirmation"
        filtered = filtered[_truthy_series(filtered[column])]

    return filtered


def _apply_peak_speed_bucket_filter(frame: pd.DataFrame, selected_bucket: str) -> pd.DataFrame:
    selected_bucket = selected_bucket.strip()
    if frame.empty or not selected_bucket:
        return frame
    if selected_bucket not in GTT_PEAK_SPEED_BUCKETS or "peak_speed_bucket" not in frame.columns:
        return frame.iloc[0:0].copy()
    return frame[frame["peak_speed_bucket"].astype(str) == selected_bucket].copy()


def _gtt_filter_warning(
    frame: pd.DataFrame,
    open_buy_regime_only: bool,
    trend_only: bool,
    dashboard_buy_only: bool = False,
    fresh_weekly_buy_only: bool = False,
    dashboard_buy_symbols: set[str] | None = None,
    fresh_daily_buy_only: bool = False,
    fresh_daily_buy_symbols: set[str] | None = None,
    require_volume_confirmation: bool = False,
    require_obv_confirmation: bool = False,
    technical_rating_statuses: list[str] | None = None,
) -> str:
    missing = []
    if open_buy_regime_only and "is_latest_signal_buy" not in frame.columns:
        missing.append("open BUY regime")
    if fresh_weekly_buy_only and "latest_week_signal" not in frame.columns:
        missing.append("fresh weekly BUY")
    if trend_only and not {"daily_ema_stack_confirmation", "trend_confirmation"}.intersection(frame.columns):
        missing.append("daily EMA stack")
    if require_volume_confirmation and (
        "volume_confirmation" not in frame.columns
        or not frame["volume_confirmation"].notna().any()
    ):
        missing.append("volume confirmation")
    if require_obv_confirmation and not {"daily_obv_confirmation", "obv_confirmation"}.intersection(frame.columns):
        missing.append("OBV confirmation")
    if dashboard_buy_only and not dashboard_buy_symbols:
        missing.append("dashboard BUY symbols")
    if fresh_daily_buy_only and not fresh_daily_buy_symbols:
        missing.append("daily BUY symbols")
    if _normalize_gtt_technical_rating_statuses(technical_rating_statuses or []) and (
        "weekly_technical_rating_status" not in frame.columns
        or not _has_meaningful_text(frame["weekly_technical_rating_status"]).any()
    ):
        missing.append("weekly technical rating")
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


def _daily_buy_symbols(data_root: Path) -> set[str]:
    filtered = Storage(data_root).load_signals("latest_daily_filtered.csv")
    if filtered.empty:
        return set()
    if "signal" in filtered.columns:
        filtered = filtered[filtered["signal"].astype(str).str.upper() == "BUY"]
    return _symbols_from_frame(filtered)


def _latest_weekly_buy_sell_frame(data_root: Path) -> pd.DataFrame:
    raw = Storage(data_root).load_signals("latest_raw_signals.csv")
    if raw.empty or "date" not in raw.columns:
        return pd.DataFrame()
    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    latest_date = frame["date"].max()
    if pd.isna(latest_date):
        return pd.DataFrame()
    frame = frame[frame["date"] == latest_date].copy()
    if "signal" in frame.columns:
        frame = frame[frame["signal"].astype(str).str.upper().isin({"BUY", "SELL"})]
    symbol_column = _symbol_column(frame)
    if not symbol_column:
        return pd.DataFrame()
    frame[symbol_column] = frame[symbol_column].astype(str).str.upper().str.strip()
    frame = frame[frame[symbol_column] != ""]
    if "exchange" not in frame.columns:
        frame["exchange"] = "NSE"
    frame["exchange"] = frame["exchange"].astype(str).str.upper().str.strip()
    if "name" not in frame.columns:
        frame["name"] = frame[symbol_column]
    frame["name"] = frame["name"].fillna("").astype(str).str.strip().mask(lambda s: s == "", frame[symbol_column])
    frame = frame.drop_duplicates(subset=["exchange", symbol_column], keep="last").reset_index(drop=True)
    return frame


def _latest_weekly_buy_sell_symbols(data_root: Path) -> set[str]:
    frame = _latest_weekly_buy_sell_frame(data_root)
    return _symbols_from_frame(frame)


def _load_signal_universe_for_strategy_lab(storage: Storage, exchange: str, start_date: str) -> pd.DataFrame:
    raw = storage.load_signals("latest_raw_signals.csv")
    if raw.empty or "date" not in raw.columns or "signal" not in raw.columns:
        return pd.DataFrame()
    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame[frame["date"].notna()].copy()
    frame = frame[frame["date"] >= pd.Timestamp(start_date)]
    frame = frame[frame["signal"].astype(str).str.upper().isin({"BUY", "SELL"})]
    if "exchange" in frame.columns:
        frame = frame[frame["exchange"].astype(str).str.upper() == exchange.upper()]
    return frame


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
    if "peak_speed_bucket" not in aligned.columns:
        aligned["peak_speed_bucket"] = pd.NA
    aligned["peak_speed_bucket"] = aligned.apply(
        lambda row: row["peak_speed_bucket"]
        if pd.notna(row.get("peak_speed_bucket"))
        and str(row.get("peak_speed_bucket")).strip().upper() not in {"", "NA", "NAN", "NONE"}
        else _gtt_peak_speed_bucket(row.get("median_days_to_peak")),
        axis=1,
    )
    return aligned


def _ensure_gtt_weekly_technical_ratings(
    data_root: Path,
    stock_stats: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    if stock_stats.empty or "symbol" not in stock_stats.columns:
        return stock_stats

    required_columns = {
        "weekly_technical_rating",
        "weekly_technical_rating_status",
        "weekly_ma_rating",
        "weekly_oscillator_rating",
    }
    if required_columns.issubset(stock_stats.columns) and _has_meaningful_text(stock_stats["weekly_technical_rating_status"]).all():
        return stock_stats

    storage = Storage(data_root)
    strategy_cfg = config.get("strategy", {})
    weekly_anchor = strategy_cfg.get("weekly_anchor", "W-FRI")
    use_completed_weeks_only = bool(strategy_cfg.get("use_completed_weeks_only", True))

    ratings_rows: list[dict[str, Any]] = []
    if required_columns.issubset(stock_stats.columns):
        symbols_to_refresh = stock_stats.loc[
            ~_has_meaningful_text(stock_stats["weekly_technical_rating_status"]),
            ["exchange", "symbol"],
        ].drop_duplicates()
    else:
        symbols_to_refresh = stock_stats[["exchange", "symbol"]].drop_duplicates()

    unique_symbols = symbols_to_refresh
    if unique_symbols.empty:
        return stock_stats
    for _, row in unique_symbols.iterrows():
        exchange = str(row.get("exchange") or "NSE").upper()
        symbol = str(row.get("symbol") or "").upper()
        rating_row = {
            "exchange": exchange,
            "symbol": symbol,
            "weekly_technical_rating": pd.NA,
            "weekly_technical_rating_status": "NA",
            "weekly_ma_rating": pd.NA,
            "weekly_oscillator_rating": pd.NA,
        }
        if not symbol:
            ratings_rows.append(rating_row)
            continue
        daily = storage.load_candles(exchange, symbol, "1D")
        if daily.empty:
            ratings_rows.append(rating_row)
            continue
        weekly = resample_daily_to_weekly(daily, weekly_anchor, use_completed_weeks_only)
        technical = latest_technical_rating(weekly)
        rating_row.update(
            {
                "weekly_technical_rating": technical.get("rating", pd.NA),
                "weekly_technical_rating_status": str(technical.get("rating_status", "NA")),
                "weekly_ma_rating": technical.get("ma_rating", pd.NA),
                "weekly_oscillator_rating": technical.get("oscillator_rating", pd.NA),
            }
        )
        ratings_rows.append(rating_row)

    ratings = pd.DataFrame(ratings_rows)
    merged = stock_stats.merge(ratings, on=["exchange", "symbol"], how="left", suffixes=("", "_fresh"))
    for column in required_columns:
        fresh_column = f"{column}_fresh"
        if column not in merged.columns:
            merged[column] = pd.NA
        if fresh_column in merged.columns:
            merged[column] = merged[fresh_column].combine_first(merged[column])
            merged = merged.drop(columns=[fresh_column], errors="ignore")

    stock_stats_path = _latest_gtt_gain_paths(data_root)["stock_stats"]
    if stock_stats_path.exists():
        merged.to_csv(stock_stats_path, index=False)
    return merged


def _expected_weekday_for_anchor(weekly_anchor: str) -> int | None:
    anchor = str(weekly_anchor or "").strip().upper()
    mapping = {
        "W-MON": 0,
        "W-TUE": 1,
        "W-WED": 2,
        "W-THU": 3,
        "W-FRI": 4,
        "W-SAT": 5,
        "W-SUN": 6,
    }
    return mapping.get(anchor)


def _ensure_gtt_latest_signal_context(
    data_root: Path,
    stock_stats: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    if stock_stats.empty or "symbol" not in stock_stats.columns:
        return stock_stats

    strategy_cfg = config.get("strategy", {})
    weekly_anchor = strategy_cfg.get("weekly_anchor", "W-FRI")
    use_completed_weeks_only = bool(strategy_cfg.get("use_completed_weeks_only", True))

    storage = Storage(data_root)
    context_rows: list[dict[str, Any]] = []
    symbols_to_refresh = stock_stats[["exchange", "symbol", "name"]].drop_duplicates()

    for _, row in symbols_to_refresh.iterrows():
        exchange = str(row.get("exchange") or "NSE").upper()
        symbol = str(row.get("symbol") or "").upper()
        name = str(row.get("name") or symbol)
        if not symbol:
            continue
        daily = storage.load_candles(exchange, symbol, "1D")
        if daily.empty:
            context_rows.append(
                _latest_signal_context(
                    pd.DataFrame(),
                    pd.DataFrame(),
                    exchange,
                    symbol,
                    name,
                    include_daily_quality_metrics=True,
                    include_technical_rating=False,
                )
            )
            continue
        daily = _prepare_daily(daily)
        weekly = resample_daily_to_weekly(daily, weekly_anchor, use_completed_weeks_only)
        strategy_output = run_weekly_buy_sell(weekly, config) if not weekly.empty else pd.DataFrame()
        context_rows.append(
            _latest_signal_context(
                strategy_output,
                daily,
                exchange,
                symbol,
                name,
                include_daily_quality_metrics=True,
                include_technical_rating=False,
            )
        )

    if not context_rows:
        return stock_stats

    refreshed = _merge_latest_context(stock_stats, pd.DataFrame(context_rows))
    preserve_columns = [
        "weekly_technical_rating",
        "weekly_technical_rating_status",
        "weekly_ma_rating",
        "weekly_oscillator_rating",
    ]
    existing = stock_stats[["exchange", "symbol"] + [column for column in preserve_columns if column in stock_stats.columns]].copy()
    refreshed = refreshed.merge(existing, on=["exchange", "symbol"], how="left", suffixes=("", "_existing"))
    for column in preserve_columns:
        existing_column = f"{column}_existing"
        if existing_column not in refreshed.columns:
            continue
        if column == "weekly_technical_rating_status":
            current = refreshed[column].astype(str).str.strip()
            placeholder_mask = current.str.upper().isin({"", "NA", "NAN", "NONE", "<NA>"})
            refreshed.loc[placeholder_mask, column] = refreshed.loc[placeholder_mask, existing_column]
        else:
            refreshed[column] = refreshed[column].combine_first(refreshed[existing_column])
        refreshed = refreshed.drop(columns=[existing_column], errors="ignore")
    stock_stats_path = _latest_gtt_gain_paths(data_root)["stock_stats"]
    if stock_stats_path.exists():
        refreshed.to_csv(stock_stats_path, index=False)
    return refreshed


def _gtt_peak_speed_bucket(days_to_peak: Any) -> str:
    days = pd.to_numeric(pd.Series([days_to_peak]), errors="coerce").iloc[0]
    if pd.isna(days):
        return "NA"
    if days <= 30:
        return "Within 30 days"
    if days <= 60:
        return "31-60 days"
    if days <= 90:
        return "61-90 days"
    if days <= 180:
        return "91-180 days"
    if days <= 365:
        return "181-365 days"
    return "Over 1 year"


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
    sensitivity: str = "",
    market_cap_bucket: str = "",
    min_market_cap_cr: str = "",
    max_market_cap_cr: str = "",
    min_cmp: str = "",
    max_cmp: str = "",
    open_buy_regime_only: bool = False,
    dashboard_buy_only: bool = False,
    fresh_weekly_buy_only: bool = False,
    fresh_daily_buy_only: bool = False,
    trend_only: bool = False,
    require_volume_confirmation: bool = False,
    require_obv_confirmation: bool = False,
    require_screener_trend_confirmation: bool = False,
    return_metric: str = "",
    min_pair_return_pct: str = "",
    peak_speed_bucket: str = "",
    technical_rating_statuses: list[str] | None = None,
) -> str:
    params = []
    if token:
        params.append(f"token={quote(token)}")
    if stock_search:
        params.append(f"stock_search={quote(stock_search)}")
    if sensitivity:
        params.append(f"sensitivity={quote(sensitivity)}")
    if market_cap_bucket:
        params.append(f"market_cap_bucket={quote(market_cap_bucket)}")
    if min_market_cap_cr:
        params.append(f"min_market_cap_cr={quote(min_market_cap_cr)}")
    if max_market_cap_cr:
        params.append(f"max_market_cap_cr={quote(max_market_cap_cr)}")
    if min_cmp:
        params.append(f"min_cmp={quote(min_cmp)}")
    if max_cmp:
        params.append(f"max_cmp={quote(max_cmp)}")
    if open_buy_regime_only:
        params.append("open_buy_regime_only=1")
    if dashboard_buy_only:
        params.append("dashboard_buy_only=1")
    if fresh_weekly_buy_only:
        params.append("fresh_weekly_buy_only=1")
    if fresh_daily_buy_only:
        params.append("fresh_daily_buy_only=1")
    if trend_only:
        params.append("trend_only=1")
    if require_volume_confirmation:
        params.append("require_volume_confirmation=1")
    if require_obv_confirmation:
        params.append("require_obv_confirmation=1")
    if require_screener_trend_confirmation:
        params.append("require_trend_confirmation=1")
    if return_metric:
        params.append(f"return_metric={quote(return_metric)}")
    if min_pair_return_pct:
        params.append(f"min_pair_return_pct={quote(min_pair_return_pct)}")
    if peak_speed_bucket:
        params.append(f"peak_speed_bucket={quote(peak_speed_bucket)}")
    for technical_rating_status in _normalize_gtt_technical_rating_statuses(technical_rating_statuses or []):
        params.append(f"technical_rating_status={quote(technical_rating_status)}")
    return "&".join(params)


def _gtt_filter_summary(
    stock_search: str,
    sensitivity_text: str,
    market_cap_bucket: str,
    min_market_cap_text: str,
    max_market_cap_text: str,
    min_cmp_text: str = "",
    max_cmp_text: str = "",
    open_buy_regime_only: bool = False,
    dashboard_buy_only: bool = False,
    fresh_weekly_buy_only: bool = False,
    fresh_daily_buy_only: bool = False,
    trend_only: bool = False,
    require_volume_confirmation: bool = False,
    require_obv_confirmation: bool = False,
    require_screener_trend_confirmation: bool = False,
    return_metric: str = "",
    min_pair_return_text: str = "",
    peak_speed_bucket: str = "",
    technical_rating_statuses: list[str] | None = None,
) -> str:
    filters = []
    if stock_search:
        filters.append(f"Search: {stock_search}")
    if sensitivity_text:
        filters.append(f"Sensitivity: {sensitivity_text}")
    if market_cap_bucket:
        filters.append(f"Market cap bucket: {market_cap_bucket}")
    if min_market_cap_text:
        filters.append(f"Min market cap: {min_market_cap_text} Cr")
    if max_market_cap_text:
        filters.append(f"Max market cap: {max_market_cap_text} Cr")
    if min_cmp_text:
        filters.append(f"Min CMP: ₹{min_cmp_text}")
    if max_cmp_text:
        filters.append(f"Max CMP: ₹{max_cmp_text}")
    if open_buy_regime_only:
        filters.append("Open BUY regime")
    if dashboard_buy_only:
        filters.append("Dashboard BUY signals only")
    if fresh_weekly_buy_only:
        filters.append("Fresh weekly BUY only")
    if fresh_daily_buy_only:
        filters.append("Fresh daily BUY only")
    if trend_only:
        filters.append("Daily EMA stack confirmed")
    if require_volume_confirmation:
        filters.append("Volume confirmed")
    if require_obv_confirmation:
        filters.append("OBV rising over last 20 days")
    normalized_statuses = _normalize_gtt_technical_rating_statuses(technical_rating_statuses or [])
    if normalized_statuses:
        filters.append(f"Weekly technical rating: {', '.join(normalized_statuses)}")
    if require_screener_trend_confirmation:
        filters.append("Home screener daily EMA stack")
    if min_pair_return_text:
        metric_label = "Home last completed BUY-SELL return" if return_metric == "last_1" else "Home median last 3 BUY-SELL returns"
        filters.append(f"{metric_label} >= {min_pair_return_text}%")
    if peak_speed_bucket:
        filters.append(f"Peak speed bucket: {peak_speed_bucket}")
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


def _display_symbol(value: Any) -> str:
    text = str(value or "").strip().upper()
    return normalize_nse_symbol(text) if text else ""


def _comma_separated_display_symbols(frame: pd.DataFrame) -> str:
    symbol_column = _symbol_column(frame)
    if frame.empty or not symbol_column:
        return ""
    symbols = frame[symbol_column].dropna().astype(str).map(_display_symbol)
    symbols = [symbol for symbol in symbols if symbol]
    return ",".join(dict.fromkeys(symbols))


def _adx_di_sorted_display_symbols(frame: pd.DataFrame) -> str:
    symbol_column = _symbol_column(frame)
    if frame.empty or not symbol_column:
        return ""

    working = frame.copy()
    di_plus = pd.to_numeric(working.get("latest_di_plus"), errors="coerce")
    di_minus = pd.to_numeric(working.get("latest_di_minus"), errors="coerce")
    working["di_plus_minus_range"] = di_plus - di_minus
    working = working.sort_values(
        ["di_plus_minus_range", symbol_column],
        ascending=[False, True],
        na_position="last",
    )
    return _comma_separated_display_symbols(working)


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
        "sensitivity",
        "market_cap_bucket",
        "min_market_cap_cr",
        "max_market_cap_cr",
        "min_cmp",
        "max_cmp",
        "require_volume_confirmation",
        "require_trend_confirmation",
        "require_obv_confirmation",
        "return_metric",
        "min_pair_return_pct",
        "require_htf_alignment",
        "min_breakout_volume_ratio",
        "require_relative_strength",
        "min_relative_strength_pct",
        "max_distance_from_demand_pct",
        "min_risk_reward_ratio",
    ):
        value = request.query_params.get(name, "").strip()
        if value:
            params.append(f"{name}={quote(value)}")
    return ("&" + "&".join(params)) if params else ""


def _dashboard_filter_query(
    token: str = "",
    stock_search: str = "",
    sensitivity: str = "",
    market_cap_bucket: str = "",
    min_market_cap_cr: str = "",
    max_market_cap_cr: str = "",
    min_cmp: str = "",
    max_cmp: str = "",
    require_volume_confirmation: bool = False,
    require_trend_confirmation: bool = False,
    require_obv_confirmation: bool = False,
    return_metric: str = "",
    min_pair_return_pct: str = "",
    require_htf_alignment: bool = False,
    min_breakout_volume_ratio: str = "",
    require_relative_strength: bool = False,
    min_relative_strength_pct: str = "",
    max_distance_from_demand_pct: str = "",
    min_risk_reward_ratio: str = "",
) -> str:
    params = []
    if token:
        params.append(f"token={quote(token)}")
    if stock_search:
        params.append(f"stock_search={quote(stock_search)}")
    if sensitivity:
        params.append(f"sensitivity={quote(sensitivity)}")
    if market_cap_bucket:
        params.append(f"market_cap_bucket={quote(market_cap_bucket)}")
    if min_market_cap_cr:
        params.append(f"min_market_cap_cr={quote(min_market_cap_cr)}")
    if max_market_cap_cr:
        params.append(f"max_market_cap_cr={quote(max_market_cap_cr)}")
    if min_cmp:
        params.append(f"min_cmp={quote(min_cmp)}")
    if max_cmp:
        params.append(f"max_cmp={quote(max_cmp)}")
    if require_volume_confirmation:
        params.append("require_volume_confirmation=1")
    if require_trend_confirmation:
        params.append("require_trend_confirmation=1")
    if require_obv_confirmation:
        params.append("require_obv_confirmation=1")
    if return_metric:
        params.append(f"return_metric={quote(return_metric)}")
    if min_pair_return_pct:
        params.append(f"min_pair_return_pct={quote(min_pair_return_pct)}")
    if require_htf_alignment:
        params.append("require_htf_alignment=1")
    if min_breakout_volume_ratio:
        params.append(f"min_breakout_volume_ratio={quote(min_breakout_volume_ratio)}")
    if require_relative_strength:
        params.append("require_relative_strength=1")
    if min_relative_strength_pct:
        params.append(f"min_relative_strength_pct={quote(min_relative_strength_pct)}")
    if max_distance_from_demand_pct:
        params.append(f"max_distance_from_demand_pct={quote(max_distance_from_demand_pct)}")
    if min_risk_reward_ratio:
        params.append(f"min_risk_reward_ratio={quote(min_risk_reward_ratio)}")
    return "&".join(params)


def _common_filter_context(
    request: Request,
    selected_sensitivity: int | str | None,
    config: dict[str, Any],
    data_root: Path,
) -> dict[str, Any]:
    token = request.query_params.get("token", "").strip()
    stock_search = request.query_params.get("stock_search", "").strip()
    sensitivity_text = str(selected_sensitivity or request.query_params.get("sensitivity", "").strip() or "")
    market_cap_bucket = request.query_params.get("market_cap_bucket", "").strip()
    min_market_cap_cr = request.query_params.get("min_market_cap_cr", "").strip()
    max_market_cap_cr = request.query_params.get("max_market_cap_cr", "").strip()
    min_cmp = request.query_params.get("min_cmp", "").strip()
    max_cmp = request.query_params.get("max_cmp", "").strip()
    require_volume_confirmation = _request_bool(request, "require_volume_confirmation")
    require_trend_confirmation = _request_bool(request, "require_trend_confirmation")
    require_obv_confirmation = _request_bool(request, "require_obv_confirmation")
    query = _dashboard_filter_query(
        token=token,
        stock_search=stock_search,
        sensitivity=sensitivity_text,
        market_cap_bucket=market_cap_bucket,
        min_market_cap_cr=min_market_cap_cr,
        max_market_cap_cr=max_market_cap_cr,
        min_cmp=min_cmp,
        max_cmp=max_cmp,
        require_volume_confirmation=require_volume_confirmation,
        require_trend_confirmation=require_trend_confirmation,
        require_obv_confirmation=require_obv_confirmation,
    )
    summary = _gtt_filter_summary(
        stock_search=stock_search,
        sensitivity_text=sensitivity_text,
        market_cap_bucket=market_cap_bucket,
        min_market_cap_text=min_market_cap_cr,
        max_market_cap_text=max_market_cap_cr,
        min_cmp_text=min_cmp,
        max_cmp_text=max_cmp,
        require_volume_confirmation=require_volume_confirmation,
        require_obv_confirmation=require_obv_confirmation,
        require_screener_trend_confirmation=require_trend_confirmation,
    )
    storage = Storage(data_root)
    metadata = _combined_symbol_metadata(config, storage)
    market_cap_bucket_options: list[str] = []
    market_cap_bounds = {"min": "", "max": ""}
    if not metadata.empty:
        if "market_cap_bucket" in metadata.columns:
            market_cap_bucket_options = sorted(
                [bucket for bucket in metadata["market_cap_bucket"].dropna().unique() if str(bucket).strip()]
            )
        if "market_cap_cr" in metadata.columns and metadata["market_cap_cr"].notna().any():
            market_cap_bounds = {
                "min": int(metadata["market_cap_cr"].min()),
                "max": int(metadata["market_cap_cr"].max()),
            }
    return {
        "common_filter_query": query,
        "common_filter_summary": summary,
        "show_shared_filter_form": True,
        "shared_filter_action": request.url.path,
        "shared_token": token,
        "shared_stock_search": stock_search,
        "shared_sensitivity": sensitivity_text,
        "shared_market_cap_bucket": market_cap_bucket,
        "shared_min_market_cap_cr": min_market_cap_cr,
        "shared_max_market_cap_cr": max_market_cap_cr,
        "shared_min_cmp": min_cmp,
        "shared_max_cmp": max_cmp,
        "shared_require_volume_confirmation": require_volume_confirmation,
        "shared_require_trend_confirmation": require_trend_confirmation,
        "shared_require_obv_confirmation": require_obv_confirmation,
        "shared_market_cap_bucket_options": market_cap_bucket_options,
        "shared_market_cap_bounds": market_cap_bounds,
    }


def _common_filtered_symbols_from_request(
    data_root: Path,
    config: dict[str, Any],
    request: Request,
) -> set[tuple[str, str]] | None:
    stock_search = request.query_params.get("stock_search", "").strip()
    market_cap_bucket = request.query_params.get("market_cap_bucket", "").strip()
    min_market_cap = _request_float(request, "min_market_cap_cr")
    max_market_cap = _request_float(request, "max_market_cap_cr")
    min_cmp = _request_float(request, "min_cmp")
    max_cmp = _request_float(request, "max_cmp")
    require_volume_confirmation = _request_bool(request, "require_volume_confirmation")
    require_trend_confirmation = _request_bool(request, "require_trend_confirmation")
    require_obv_confirmation = _request_bool(request, "require_obv_confirmation")

    common_filters_active = any(
        [
            stock_search,
            market_cap_bucket,
            min_market_cap is not None,
            max_market_cap is not None,
            min_cmp is not None,
            max_cmp is not None,
            require_volume_confirmation,
            require_trend_confirmation,
            require_obv_confirmation,
        ]
    )
    if not common_filters_active:
        return None

    storage = Storage(data_root)
    raw = storage.load_signals("latest_raw_signals.csv")
    if raw.empty or not {"exchange", "symbol", "date"}.issubset(raw.columns):
        return set()

    latest = raw.copy()
    latest["date_sort"] = pd.to_datetime(latest["date"], errors="coerce")
    latest = latest.sort_values("date_sort").groupby(["exchange", "symbol"], dropna=False).tail(1).copy()
    latest = latest.drop(columns=["date_sort"], errors="ignore")
    latest = _enrich_with_latest_daily_close(latest, storage.load_signals("latest_scan_details.csv"), storage)

    metadata = _combined_symbol_metadata(config, storage)
    latest = _enrich_with_symbol_metadata(latest, metadata, "symbol")
    latest = _apply_market_cap_filters(latest, min_market_cap, max_market_cap, market_cap_bucket)
    latest = _apply_cmp_filters(latest, min_cmp, max_cmp, "latest_close")
    latest = _apply_stock_search(latest, stock_search)
    latest = _apply_signal_quality_filters(
        latest,
        require_volume_confirmation,
        require_trend_confirmation,
        require_obv_confirmation,
        "median_3",
        None,
    )
    if latest.empty:
        return set()
    return {
        (str(row.get("exchange", "")).upper(), str(row.get("symbol", "")).upper())
        for _, row in latest.iterrows()
        if str(row.get("symbol", "")).strip()
    }


def _filter_frame_by_symbol_scope(frame: pd.DataFrame, symbol_scope: set[tuple[str, str]] | None) -> pd.DataFrame:
    if symbol_scope is None or frame.empty:
        return frame
    symbol_column = _symbol_column(frame)
    if not symbol_column or "exchange" not in frame.columns:
        return frame
    working = frame.copy()
    keys = list(
        zip(
            working["exchange"].astype(str).str.upper(),
            working[symbol_column].astype(str).str.upper(),
        )
    )
    mask = pd.Series([key in symbol_scope for key in keys], index=working.index)
    return working[mask].reset_index(drop=True)


def _buy_signal_filter_summary(
    stock_search: str,
    market_cap_bucket: str,
    min_market_cap_text: str,
    max_market_cap_text: str,
    min_cmp_text: str = "",
    max_cmp_text: str = "",
    require_volume_confirmation: bool = False,
    require_trend_confirmation: bool = False,
    require_obv_confirmation: bool = False,
    return_metric: str = "",
    min_pair_return_text: str = "",
    require_htf_alignment: bool = False,
    min_breakout_volume_ratio_text: str = "",
    require_relative_strength: bool = False,
    min_relative_strength_pct_text: str = "",
    max_distance_from_demand_pct_text: str = "",
    min_risk_reward_ratio_text: str = "",
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
    if min_cmp_text:
        filters.append(f"Min CMP: ₹{min_cmp_text}")
    if max_cmp_text:
        filters.append(f"Max CMP: ₹{max_cmp_text}")
    if require_volume_confirmation:
        filters.append("Volume confirmation: Yes")
    if require_trend_confirmation:
        filters.append("Daily EMA stack: Yes")
    if require_obv_confirmation:
        filters.append("OBV rising 20D: Yes")
    if min_pair_return_text:
        metric_label = "Last completed BUY-SELL return" if return_metric == "last_1" else "Median last 3 BUY-SELL returns"
        filters.append(f"{metric_label} >= {min_pair_return_text}%")
    if require_htf_alignment:
        filters.append("Monthly structure aligned: Yes")
    if min_breakout_volume_ratio_text:
        filters.append(f"Breakout volume >= {min_breakout_volume_ratio_text}x")
    if require_relative_strength or min_relative_strength_pct_text:
        threshold = min_relative_strength_pct_text or "0"
        filters.append(f"Relative strength vs benchmark >= {threshold}%")
    if max_distance_from_demand_pct_text:
        filters.append(f"Distance from demand <= {max_distance_from_demand_pct_text}%")
    if min_risk_reward_ratio_text:
        filters.append(f"Risk-reward >= {min_risk_reward_ratio_text}")
    return "; ".join(filters) if filters else "None"


def _manual_screener_config(
    base_config: dict,
    storage: Storage,
    min_market_cap: float | None,
    max_market_cap: float | None,
    market_cap_bucket: str,
    stock_search: str,
    sensitivity: int | None = None,
) -> dict:
    config = deepcopy(base_config)
    universe_cfg = config.setdefault("universe", {})

    metadata_path = storage.symbol_metadata_path()
    if metadata_path.exists():
        universe_cfg["metadata_file"] = str(metadata_path)

    # The screener run should always build weekly BUY/SELL signals for the full
    # active universe first. UI filters like market cap, CMP, and search are
    # applied later to the saved signal list for display, charts, and exports.
    filters_cfg = universe_cfg.setdefault("filters", {})
    filters_cfg["min_market_cap_cr"] = None
    filters_cfg["max_market_cap_cr"] = None
    filters_cfg["market_cap_bucket"] = None
    filters_cfg["stock_search"] = None

    signal_cfg = config.setdefault("filters", {}).setdefault("signal", {})
    signal_cfg["direction"] = "BUY"
    signal_cfg["latest_only"] = True
    if sensitivity is not None:
        config.setdefault("strategy", {})["sensitivity"] = max(1, min(20, int(sensitivity)))

    config.setdefault("notifications", {})["enabled"] = False
    return config


def _load_visible_buy_signals(
    config: dict[str, Any],
    storage: Storage,
    stock_search: str,
    min_market_cap: float | None,
    max_market_cap: float | None,
    market_cap_bucket: str,
    min_cmp: float | None = None,
    max_cmp: float | None = None,
    require_volume_confirmation: bool = False,
    require_trend_confirmation: bool = False,
    require_obv_confirmation: bool = False,
    return_metric: str = "",
    min_pair_return: float | None = None,
    require_htf_alignment: bool = False,
    min_breakout_volume_ratio: float | None = None,
    require_relative_strength: bool = False,
    min_relative_strength_pct: float | None = None,
    max_distance_from_demand_pct: float | None = None,
    min_risk_reward_ratio: float | None = None,
) -> pd.DataFrame:
    metadata = _combined_symbol_metadata(config, storage)
    filtered = storage.load_signals("latest_filtered.csv")
    filtered = _enrich_with_latest_daily_close(filtered, storage.load_signals("latest_scan_details.csv"), storage)
    filtered = _enrich_with_symbol_metadata(filtered, metadata, "symbol")
    filtered = _apply_market_cap_filters(filtered, min_market_cap, max_market_cap, market_cap_bucket)
    filtered = _refresh_live_cmp(filtered, storage.data_root)
    filtered = _apply_cmp_filters(filtered, min_cmp, max_cmp, "latest_close")
    filtered = _apply_stock_search(filtered, stock_search)
    filtered = _apply_signal_quality_filters(
        filtered,
        require_volume_confirmation,
        require_trend_confirmation,
        require_obv_confirmation,
        return_metric,
        min_pair_return,
    )
    filtered = enrich_weekly_signal_shortlist_frame(filtered, storage, config)
    filtered = _apply_weekly_shortlist_filters(
        filtered,
        require_htf_alignment,
        min_breakout_volume_ratio,
        require_relative_strength,
        min_relative_strength_pct,
        max_distance_from_demand_pct,
        min_risk_reward_ratio,
    )

    if not filtered.empty and "date" in filtered.columns:
        filtered = filtered.copy()
        filtered["date_sort"] = pd.to_datetime(filtered["date"], errors="coerce")
        sort_columns = []
        sort_ascending = []
        if "shortlist_score" in filtered.columns:
            filtered["shortlist_score"] = pd.to_numeric(filtered["shortlist_score"], errors="coerce")
            sort_columns.append("shortlist_score")
            sort_ascending.append(False)
        if "relative_strength_12w_pct" in filtered.columns:
            filtered["relative_strength_12w_pct"] = pd.to_numeric(filtered["relative_strength_12w_pct"], errors="coerce")
            sort_columns.append("relative_strength_12w_pct")
            sort_ascending.append(False)
        sort_columns.append("date_sort")
        sort_ascending.append(False)
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
    min_cmp: float | None = None,
    max_cmp: float | None = None,
    open_buy_regime_only: bool = False,
    dashboard_buy_only: bool = False,
    fresh_weekly_buy_only: bool = False,
    fresh_daily_buy_only: bool = False,
    trend_only: bool = False,
    peak_speed_bucket: str = "",
    require_volume_confirmation: bool = False,
    require_obv_confirmation: bool = False,
    technical_rating_statuses: list[str] | None = None,
) -> pd.DataFrame:
    latest = load_gtt_gain_outputs(_gtt_gain_dir(data_root))
    stock_stats = _align_gtt_stock_stats_to_latest_universe(data_root, latest.stock_stats, config)
    stock_stats = _ensure_gtt_weekly_technical_ratings(data_root, stock_stats, config)
    stock_stats = _enrich_with_symbol_metadata(stock_stats, _combined_symbol_metadata(config, storage), "symbol")
    stock_stats = _ensure_gtt_s2_to_s3_markers(data_root, stock_stats, config)
    stock_stats = _apply_market_cap_filters(stock_stats, min_market_cap, max_market_cap, market_cap_bucket)
    stock_stats = _apply_cmp_filters(stock_stats, min_cmp, max_cmp, "latest_close")
    stock_stats = _apply_stock_search(stock_stats, stock_search)
    stock_stats = _apply_gtt_stock_filters(
        stock_stats,
        open_buy_regime_only,
        trend_only,
        dashboard_buy_only,
        _dashboard_buy_symbols(data_root),
        fresh_weekly_buy_only,
        fresh_daily_buy_only,
        _daily_buy_symbols(data_root),
        require_volume_confirmation,
        require_obv_confirmation,
        technical_rating_statuses,
    )
    stock_stats = _apply_peak_speed_bucket_filter(stock_stats, peak_speed_bucket)

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


def _latest_signal_week_date(data_root: Path) -> str:
    raw = Storage(data_root).load_signals("latest_raw_signals.csv")
    if raw.empty or "date" not in raw.columns:
        return ""
    dates = pd.to_datetime(raw["date"], errors="coerce").dropna()
    if dates.empty:
        return ""
    return dates.max().strftime("%Y-%m-%d")


def _ensure_gtt_s2_to_s3_markers(
    data_root: Path,
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    if frame.empty or "symbol" not in frame.columns:
        return frame

    enriched = frame.copy()
    symbols = {str(symbol).upper().strip() for symbol in enriched["symbol"] if str(symbol).strip()}
    if not symbols:
        enriched["s2_to_s3_next_week_seen"] = False
        enriched["s2_to_s3_next_week_count"] = 0
        return enriched

    expected_start = SENSITIVITY_OVERLAP_DEFAULT_START_DATE
    latest_week_date = _latest_signal_week_date(data_root)
    cached = load_sensitivity_overlap_outputs(_sensitivity_overlap_dir(data_root))
    use_cached = (
        bool(cached.summary)
        and str(cached.summary.get("start_date", "")) == expected_start
        and str(cached.summary.get("latest_week_date", "")) == latest_week_date
        and not cached.conversion_details.empty
    )

    if use_cached:
        details = cached.conversion_details.copy()
        details["symbol"] = details["symbol"].astype(str).str.upper().str.strip()
        details = details[details["symbol"].isin(symbols)]
        details = details[details["next_week_match"] == True]
        grouped = (
            details.groupby("symbol", dropna=False)
            .agg(
                s2_to_s3_next_week_count=("next_week_match", "sum"),
                s2_to_s3_first_s2_date=("s2_date", "min"),
                s2_to_s3_latest_s2_date=("s2_date", "max"),
            )
            .reset_index()
        )
        markers = pd.DataFrame({"symbol": sorted(symbols)}).merge(grouped, on="symbol", how="left")
    else:
        markers = pd.DataFrame({"symbol": sorted(symbols)})
        markers["s2_to_s3_next_week_count"] = 0
        markers["s2_to_s3_first_s2_date"] = pd.NA
        markers["s2_to_s3_latest_s2_date"] = pd.NA

    if "symbol" not in markers.columns:
        markers["symbol"] = pd.Series(dtype="object")
    markers["symbol"] = markers["symbol"].astype(str).str.upper().str.strip()
    markers["s2_to_s3_next_week_seen"] = pd.to_numeric(
        pd.Series(markers.get("s2_to_s3_next_week_count", 0)),
        errors="coerce",
    ).fillna(0).astype(int) > 0
    markers["s2_to_s3_next_week_count"] = pd.to_numeric(markers.get("s2_to_s3_next_week_count", 0), errors="coerce").fillna(0).astype(int)
    merged = enriched.merge(
        markers[[
            "symbol",
            "s2_to_s3_next_week_seen",
            "s2_to_s3_next_week_count",
            "s2_to_s3_first_s2_date",
            "s2_to_s3_latest_s2_date",
        ]],
        on="symbol",
        how="left",
    )
    merged["s2_to_s3_next_week_seen"] = merged["s2_to_s3_next_week_seen"].fillna(False).astype(bool)
    merged["s2_to_s3_next_week_count"] = pd.to_numeric(merged["s2_to_s3_next_week_count"], errors="coerce").fillna(0).astype(int)
    return merged


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
        f"&refresh_mode={quote(str(summary.get('refresh_mode', 'kite_refresh')))}"
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


def _weekday_study_redirect_url(summary: dict[str, Any], query_suffix: str) -> str:
    return (
        "/weekday-study?"
        f"study_ran=1&symbols_processed={summary.get('symbols_processed', 0)}"
        f"&stocks_with_weekday_profile={summary.get('stocks_with_weekday_profile', 0)}"
        f"{query_suffix}"
    )


def _weekday_study_error_url(error: Exception, query_suffix: str) -> str:
    return f"/weekday-study?study_error={quote(str(error)[:500])}{query_suffix}"


def _strategy_lab_redirect_url(summary: dict[str, Any], query_suffix: str) -> str:
    return (
        "/strategy-lab?"
        f"study_ran=1&signal_events={summary.get('signal_events', 0)}"
        f"&best_strategy={quote(str(summary.get('best_strategy_name', '')))}"
        f"{query_suffix}"
    )


def _strategy_lab_error_url(error: Exception, query_suffix: str) -> str:
    return f"/strategy-lab?study_error={quote(str(error)[:500])}{query_suffix}"


def _sensitivity_overlap_redirect_url(summary: dict[str, Any], query_suffix: str) -> str:
    return (
        "/sensitivity-study?"
        f"study_ran=1&s2_events={summary.get('s2_buy_events', 0)}"
        f"&same_week_overlap={summary.get('same_week_overlap_events', 0)}"
        f"{query_suffix}"
    )


def _sensitivity_overlap_error_url(error: Exception, query_suffix: str) -> str:
    return f"/sensitivity-study?study_error={quote(str(error)[:500])}{query_suffix}"


def _weekly_buy_gains_redirect_url(summary: dict[str, Any], query_suffix: str) -> str:
    return (
        "/weekly-buy-gains?"
        f"study_ran=1&stocks={summary.get('stocks_with_buy_history', 0)}"
        f"&s2_events={summary.get('s2_buy_events', 0)}"
        f"&s3_events={summary.get('s3_buy_events', 0)}"
        f"{query_suffix}"
    )


def _weekly_buy_gains_error_url(error: Exception, query_suffix: str) -> str:
    return f"/weekly-buy-gains?study_error={quote(str(error)[:500])}{query_suffix}"


def _qm_quality_redirect_url(summary: dict[str, Any], query_suffix: str) -> str:
    return (
        "/qm-quality?"
        f"study_ran=1&symbols={summary.get('april_buy_symbols', 0)}"
        f"&events={summary.get('april_buy_events', 0)}"
        f"{query_suffix}"
    )


def _qm_quality_error_url(error: Exception, query_suffix: str) -> str:
    return f"/qm-quality?study_error={quote(str(error)[:500])}{query_suffix}"


def _volume_burst_redirect_url(summary: dict[str, Any], query_suffix: str) -> str:
    return (
        "/volume-burst?"
        f"study_ran=1&symbols={summary.get('symbols_processed', 0)}"
        f"&matches={summary.get('volume_burst_matches', 0)}"
        f"{query_suffix}"
    )


def _volume_burst_error_url(error: Exception, query_suffix: str) -> str:
    return f"/volume-burst?study_error={quote(str(error)[:500])}{query_suffix}"


def _rotation_study_redirect_url(summary: dict[str, Any], query_suffix: str) -> str:
    return (
        "/rotation-study?"
        f"study_ran=1&groups_found={summary.get('groups_found', 0)}"
        f"&candidates={summary.get('catch_up_candidates', 0)}"
        f"{query_suffix}"
    )


def _rotation_study_error_url(error: Exception, query_suffix: str) -> str:
    return f"/rotation-study?study_error={quote(str(error)[:500])}{query_suffix}"


def _signal_outcome_redirect_url(summary: dict[str, Any], query_suffix: str) -> str:
    return (
        "/signal-outcome-study?"
        f"study_ran=1&signals={summary.get('current_signal_universe_count', 0)}"
        f"&pairs={summary.get('historical_pairs_analyzed', 0)}"
        f"{query_suffix}"
    )


def _signal_outcome_error_url(error: Exception, query_suffix: str) -> str:
    return f"/signal-outcome-study?study_error={quote(str(error)[:500])}{query_suffix}"


def _swing_trade_redirect_url(summary: dict[str, Any], query_suffix: str) -> str:
    return (
        "/swing-trade-study?"
        f"study_ran=1&candidates={summary.get('candidates_found', 0)}"
        f"&ready_now={summary.get('ready_now_count', 0)}"
        f"{query_suffix}"
    )


def _swing_trade_error_url(error: Exception, query_suffix: str) -> str:
    return f"/swing-trade-study?study_error={quote(str(error)[:500])}{query_suffix}"


def _run_screener_job(job_id: str, scan_config: dict[str, Any], query_suffix: str) -> None:
    data_root = get_data_root(scan_config)
    storage = Storage(data_root)
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
            status="running",
            phase="Preparing weekday profiles",
            completed=int(summary.get("symbols_scanned", 0)),
            total=int(summary.get("symbols_scanned", 0)),
            percent=100,
            current_symbol="",
            current_exchange="",
        )
        _build_latest_weekday_pressure_cache(scan_config, storage, data_root)
        try:
            _set_scan_job(
                job_id,
                status="running",
                phase="Updating Minervini Google Sheet",
                completed=int(summary.get("symbols_scanned", 0)),
                total=int(summary.get("symbols_scanned", 0)),
                percent=100,
                current_symbol="",
                current_exchange="",
            )
            summary["minervini_sheet_sync"] = _maybe_run_minervini_sheet_sync_after_screener(storage, data_root)
        except Exception as minervini_exc:
            summary["minervini_sheet_sync"] = {
                "status": "failed",
                "error": str(minervini_exc),
            }
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


def _run_weekday_study_job(job_id: str, config: dict[str, Any], data_root: Path, query_suffix: str) -> None:
    storage = Storage(data_root)
    signal_rows = _latest_weekly_buy_sell_frame(data_root)
    signal_symbols = _symbols_from_frame(signal_rows)
    _set_scan_job(
        job_id,
        status="running",
        phase="Starting Weekday Study",
        completed=0,
        total=len(signal_symbols),
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
        result = _build_latest_weekday_pressure_cache(
            config,
            storage,
            data_root,
            progress_callback=progress_callback,
        )
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
            redirect_url=_weekday_study_redirect_url(result.summary, query_suffix),
        )
    except Exception as exc:
        _set_scan_job(
            job_id,
            status="failed",
            phase="Failed",
            error=str(exc),
            redirect_url=_weekday_study_error_url(exc, query_suffix),
        )


def _run_strategy_lab_job(
    job_id: str,
    config: dict[str, Any],
    data_root: Path,
    query_suffix: str,
    start_date: str,
) -> None:
    storage = Storage(data_root)
    signal_rows = _load_signal_universe_for_strategy_lab(storage, "NSE", start_date)
    _set_scan_job(
        job_id,
        status="running",
        phase="Starting Strategy Lab",
        completed=0,
        total=len(signal_rows),
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
        result = run_strategy_lab_study(
            config,
            storage,
            exchange="NSE",
            start_date=start_date,
            progress_callback=progress_callback,
        )
        save_strategy_lab_outputs(result, _strategy_lab_dir(data_root))
        _set_scan_job(
            job_id,
            status="completed",
            phase="Complete",
            completed=int(result.summary.get("signal_events", 0)),
            total=int(result.summary.get("signal_events", 0)),
            percent=100,
            current_symbol="",
            current_exchange="",
            summary=result.summary,
            redirect_url=_strategy_lab_redirect_url(result.summary, query_suffix),
        )
    except Exception as exc:
        _set_scan_job(
            job_id,
            status="failed",
            phase="Failed",
            error=str(exc),
            redirect_url=_strategy_lab_error_url(exc, query_suffix),
        )


def _run_sensitivity_overlap_job(
    job_id: str,
    config: dict[str, Any],
    data_root: Path,
    query_suffix: str,
    start_date: str,
) -> None:
    storage = Storage(data_root)
    symbol_total = len(list((data_root / "candles" / "NSE" / "1D").glob("*.csv")))
    _set_scan_job(
        job_id,
        status="running",
        phase="Starting Sensitivity Study",
        completed=0,
        total=symbol_total,
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
        result = run_sensitivity_overlap_study(
            config,
            storage,
            exchange="NSE",
            start_date=start_date,
            progress_callback=progress_callback,
        )
        save_sensitivity_overlap_outputs(result, _sensitivity_overlap_dir(data_root))
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
            redirect_url=_sensitivity_overlap_redirect_url(result.summary, query_suffix),
        )
    except Exception as exc:
        _set_scan_job(
            job_id,
            status="failed",
            phase="Failed",
            error=str(exc),
            redirect_url=_sensitivity_overlap_error_url(exc, query_suffix),
        )


def _run_weekly_buy_tracker_job(
    job_id: str,
    config: dict[str, Any],
    data_root: Path,
    query_suffix: str,
    start_date: str,
) -> None:
    storage = Storage(data_root)
    symbol_total = len(list((data_root / "candles" / "NSE" / "1D").glob("*.csv")))
    _set_scan_job(
        job_id,
        status="running",
        phase="Starting Weekly Buy Tracker",
        completed=0,
        total=symbol_total,
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
        result = run_weekly_buy_tracker_study(
            config,
            storage,
            exchange="NSE",
            start_date=start_date,
            progress_callback=progress_callback,
        )
        save_weekly_buy_tracker_outputs(result, _weekly_buy_tracker_dir(data_root))
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
            redirect_url=_weekly_buy_gains_redirect_url(result.summary, query_suffix),
        )
    except Exception as exc:
        _set_scan_job(
            job_id,
            status="failed",
            phase="Failed",
            error=str(exc),
            redirect_url=_weekly_buy_gains_error_url(exc, query_suffix),
        )


def _run_qm_quality_job(
    job_id: str,
    config: dict[str, Any],
    data_root: Path,
    query_suffix: str,
    buy_start_date: str,
    buy_end_date: str,
    run_mode: str,
    price_as_of_date: str,
) -> None:
    storage = Storage(data_root)
    symbol_total = len(list((data_root / "candles" / "NSE" / "1D").glob("*.csv")))
    signal_frame = _latest_weekly_buy_sell_frame(data_root) if run_mode == "latest" else None
    _set_scan_job(
        job_id,
        status="running",
        phase="Starting Quantitative Momentum Quality",
        completed=0,
        total=symbol_total,
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
        result = run_qm_quality_study(
            config,
            storage,
            exchange="NSE",
            buy_start_date=buy_start_date,
            buy_end_date=buy_end_date,
            price_as_of_date=price_as_of_date,
            signal_frame=signal_frame,
            progress_callback=progress_callback,
        )
        save_qm_quality_outputs(result, _qm_quality_dir(data_root))
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
            redirect_url=_qm_quality_redirect_url(result.summary, query_suffix),
        )
    except Exception as exc:
        _set_scan_job(
            job_id,
            status="failed",
            phase="Failed",
            error=str(exc),
            redirect_url=_qm_quality_error_url(exc, query_suffix),
        )


def _run_volume_burst_job(
    job_id: str,
    data_root: Path,
    query_suffix: str,
) -> None:
    storage = Storage(data_root)
    symbol_total = len(list((data_root / "candles" / "NSE" / "1D").glob("*.csv")))
    _set_scan_job(
        job_id,
        status="running",
        phase="Starting Volume Burst Study",
        completed=0,
        total=symbol_total,
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
        result = run_volume_burst_study(storage, exchange="NSE", progress_callback=progress_callback)
        save_volume_burst_outputs(result, _volume_burst_dir(data_root))
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
            redirect_url=_volume_burst_redirect_url(result.summary, query_suffix),
        )
    except Exception as exc:
        _set_scan_job(
            job_id,
            status="failed",
            phase="Failed",
            error=str(exc),
            redirect_url=_volume_burst_error_url(exc, query_suffix),
        )


def _run_resistance_breaks_job(
    job_id: str,
    data_root: Path,
    query_suffix: str,
    left_bars: int,
    right_bars: int,
    volume_avg_window: int,
    volume_multiplier: float,
    min_break_count: int,
    recent_window_days: int,
) -> None:
    storage = Storage(data_root)
    symbol_total = len(list((data_root / "candles" / "NSE" / "1D").glob("*.csv")))
    _set_scan_job(
        job_id,
        status="running",
        phase="Starting resistance break scan",
        completed=0,
        total=symbol_total,
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
        result = run_resistance_breaks_study(
            storage,
            exchange="NSE",
            left_bars=left_bars,
            right_bars=right_bars,
            volume_avg_window=volume_avg_window,
            volume_multiplier=volume_multiplier,
            min_break_count=min_break_count,
            recent_window_days=recent_window_days,
            progress_callback=progress_callback,
        )
        save_resistance_breaks_outputs(result, _resistance_breaks_dir(data_root))
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
            redirect_url=f"/resistance-breaks?study_ran=1{query_suffix}",
        )
    except Exception as exc:
        _set_scan_job(
            job_id,
            status="failed",
            phase="Failed",
            error=str(exc),
            redirect_url=f"/resistance-breaks?study_error={quote(str(exc)[:500])}{query_suffix}",
        )


def _refresh_minervini_quality_benchmark(storage: Storage, benchmark_symbol: str) -> None:
    access_token = load_access_token(storage.data_root)
    if not access_token:
        if not storage.load_candles("NSE_INDEX", benchmark_symbol, "1D").empty:
            return
        raise RuntimeError("Kite access token not found. Refresh the Kite login before the first quality scan.")

    provider = KiteDataProvider(access_token=access_token)
    provider.validate_session()
    instruments = storage.load_instruments()
    if instruments.empty:
        instruments = provider.instruments()
        storage.save_instruments(instruments)

    rows = instruments[
        (instruments["exchange"].astype(str).str.upper() == "NSE")
        & (instruments["tradingsymbol"].astype(str).str.upper() == benchmark_symbol.upper())
    ]
    if rows.empty:
        instruments = provider.instruments()
        storage.save_instruments(instruments)
        rows = instruments[
            (instruments["exchange"].astype(str).str.upper() == "NSE")
            & (instruments["tradingsymbol"].astype(str).str.upper() == benchmark_symbol.upper())
        ]
    if rows.empty:
        raise RuntimeError(f"Kite instrument {benchmark_symbol} was not found.")

    history_years = int(load_config().get("data", {}).get("history_years", 10))
    existing = storage.load_candles("NSE_INDEX", benchmark_symbol, "1D")
    from_date = _fetch_incremental_start_date(existing, history_years)
    today = date.today()
    if from_date <= today:
        new_daily = provider.daily_candles(int(rows.iloc[0]["instrument_token"]), from_date, today)
        storage.merge_and_save_candles("NSE_INDEX", benchmark_symbol, new_daily, "1D")


def _refresh_minervini_quality_candles(
    storage: Storage,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[str]:
    config = load_config()
    history_years = int(config.get("data", {}).get("history_years", 10))
    access_token = load_access_token(storage.data_root)
    if not access_token:
        raise RuntimeError("Kite access token not found. Refresh the Kite login before running Minervini Quality.")

    provider = KiteDataProvider(access_token=access_token)
    provider.validate_session()
    instruments = provider.instruments()
    storage.save_instruments(instruments)
    universe = build_universe(instruments, config)
    candidates = universe[
        universe["exchange"].astype(str).str.upper().eq("NSE")
    ].drop_duplicates(subset=["tradingsymbol"]).sort_values("tradingsymbol").reset_index(drop=True)
    today = date.today()
    refreshed_symbols: list[str] = []

    if progress_callback:
        progress_callback(
            {
                "phase": "Refreshing Minervini NSE candles",
                "completed": 0,
                "total": len(candidates),
                "current_symbol": "",
                "current_exchange": "NSE",
            }
        )

    for completed, (_, instrument) in enumerate(candidates.iterrows(), start=1):
        symbol = str(instrument["tradingsymbol"]).strip().upper()
        existing = storage.load_candles("NSE", symbol, "1D")
        from_date = _fetch_incremental_start_date(existing, history_years)
        if from_date <= today:
            new_daily = provider.daily_candles(
                int(instrument["instrument_token"]),
                from_date,
                today,
            )
            daily = storage.merge_and_save_candles("NSE", symbol, new_daily, "1D")
        else:
            daily = existing
        if not daily.empty:
            refreshed_symbols.append(symbol)
        if progress_callback:
            progress_callback(
                {
                    "phase": "Refreshing Minervini NSE candles",
                    "completed": completed,
                    "total": len(candidates),
                    "current_symbol": symbol,
                    "current_exchange": "NSE",
                }
            )

    return refreshed_symbols


def _run_minervini_quality_job(
    job_id: str,
    data_root: Path,
    query_suffix: str,
    score_threshold: float,
) -> None:
    storage = Storage(data_root)
    symbol_total = len(list((data_root / "candles" / "NSE" / "1D").glob("*.csv")))
    _set_scan_job(
        job_id,
        status="running",
        phase="Refreshing Minervini NSE candles",
        completed=0,
        total=symbol_total,
        percent=0,
        current_symbol="",
        current_exchange="NSE",
    )

    def refresh_progress_callback(payload: dict[str, Any]) -> None:
        total = int(payload.get("total") or 0)
        completed = int(payload.get("completed") or 0)
        percent = int((completed / total) * 50) if total else 0
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

    def scan_progress_callback(payload: dict[str, Any]) -> None:
        total = int(payload.get("total") or 0)
        completed = int(payload.get("completed") or 0)
        percent = 50 + (int((completed / total) * 50) if total else 0)
        _set_scan_job(
            job_id,
            status="running",
            phase=payload.get("phase", "Running"),
            completed=completed,
            total=total,
            percent=max(50, min(percent, 100)),
            current_symbol=payload.get("current_symbol", ""),
            current_exchange=payload.get("current_exchange", ""),
        )

    try:
        refreshed_symbols = _refresh_minervini_quality_candles(
            storage,
            progress_callback=refresh_progress_callback,
        )
        if not refreshed_symbols:
            raise RuntimeError("No fresh NSE daily candles were available for the Minervini Quality scan.")
        _set_scan_job(
            job_id,
            status="running",
            phase="Refreshing NIFTY 500 benchmark",
            completed=0,
            total=len(refreshed_symbols),
            percent=50,
            current_symbol="",
            current_exchange="NSE",
        )
        _refresh_minervini_quality_benchmark(storage, MINERVINI_QUALITY_DEFAULT_BENCHMARK)
        result = run_minervini_quality_study(
            storage,
            exchange="NSE",
            symbols=refreshed_symbols,
            benchmark_symbol=MINERVINI_QUALITY_DEFAULT_BENCHMARK,
            score_threshold=score_threshold,
            progress_callback=scan_progress_callback,
        )
        save_minervini_quality_outputs(result, _minervini_quality_dir(data_root))
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
            redirect_url=f"/minervini-quality?study_ran=1{query_suffix}",
        )
    except Exception as exc:
        _set_scan_job(
            job_id,
            status="failed",
            phase="Failed",
            error=str(exc),
            redirect_url=f"/minervini-quality?study_error={quote(str(exc)[:500])}{query_suffix}",
        )


def _fetch_incremental_start_date(existing: pd.DataFrame, history_years: int) -> date:
    if existing.empty:
        return date.today() - timedelta(days=365 * int(history_years))
    last_date = pd.to_datetime(existing["date"], errors="coerce").max()
    if pd.isna(last_date):
        return date.today() - timedelta(days=365 * int(history_years))
    # Overlap the latest saved day so corrected EOD candles replace stale values.
    return pd.Timestamp(last_date).date()


def _refresh_adx_di_candles(
    storage: Storage,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[str]:
    config = load_config()
    history_years = int(config.get("data", {}).get("history_years", 10))
    access_token = load_access_token(storage.data_root)
    if not access_token:
        raise RuntimeError("Kite access token not found. Refresh the Kite login before running the ADX screener.")

    provider = KiteDataProvider(access_token=access_token)
    provider.validate_session()
    instruments = provider.instruments()
    storage.save_instruments(instruments)

    nse = instruments[
        (instruments["exchange"].astype(str).str.upper() == "NSE")
        & (instruments["segment"].astype(str).str.upper() != "INDICES")
    ].copy()
    if "instrument_type" in nse.columns:
        nse = nse[nse["instrument_type"].astype(str).str.upper().isin({"EQ"})].copy()
    nse["tradingsymbol"] = nse["tradingsymbol"].astype(str).str.upper().str.strip()
    nse = nse[~nse["tradingsymbol"].apply(_is_excluded_adx_symbol)].drop_duplicates(subset=["tradingsymbol"])
    candidates = nse.sort_values("tradingsymbol").reset_index(drop=True)
    today = date.today()
    refreshed_symbols: list[str] = []

    if progress_callback:
        progress_callback(
            {
                "phase": "Refreshing ADX / DI candles",
                "completed": 0,
                "total": len(candidates),
                "current_symbol": "",
                "current_exchange": "NSE",
            }
        )

    for completed, (_, instrument) in enumerate(candidates.iterrows(), start=1):
        symbol = str(instrument["tradingsymbol"]).upper()
        token = int(instrument["instrument_token"])
        existing_daily = storage.load_candles("NSE", symbol, "1D")
        from_date = _fetch_incremental_start_date(existing_daily, history_years)
        if from_date <= today:
            new_daily = provider.daily_candles(token, from_date, today)
            daily = storage.merge_and_save_candles("NSE", symbol, new_daily, "1D")
        else:
            daily = existing_daily
        if daily.empty:
            if progress_callback:
                progress_callback(
                    {
                        "phase": "Refreshing ADX / DI candles",
                        "completed": completed,
                        "total": len(candidates),
                        "current_symbol": symbol,
                        "current_exchange": "NSE",
                    }
                )
            continue
        refreshed_symbols.append(symbol)
        if progress_callback:
            progress_callback(
                {
                    "phase": "Refreshing ADX / DI candles",
                    "completed": completed,
                    "total": len(candidates),
                    "current_symbol": symbol,
                    "current_exchange": "NSE",
                }
            )

    benchmark_rows = instruments[
        (instruments["exchange"].astype(str).str.upper() == "NSE")
        & (instruments["tradingsymbol"].astype(str).str.upper() == "NIFTY 50")
    ].drop_duplicates(subset=["tradingsymbol"])
    if not benchmark_rows.empty:
        benchmark = benchmark_rows.iloc[0]
        existing_benchmark = storage.load_candles("NSE_INDEX", "NIFTY 50", "1D")
        from_date = _fetch_incremental_start_date(existing_benchmark, history_years)
        if from_date <= today:
            new_benchmark = provider.daily_candles(int(benchmark["instrument_token"]), from_date, today)
            storage.merge_and_save_candles("NSE_INDEX", "NIFTY 50", new_benchmark, "1D")

    return refreshed_symbols


def _run_adx_di_job(
    job_id: str,
    data_root: Path,
    query_suffix: str,
    length: int,
    threshold: float,
    cross_lookback_bars: int,
    trend_fast_ma_length: int,
    trend_slow_ma_length: int,
    volume_avg_lookback: int,
    min_volume_ratio: float,
    breakout_lookback_days: int,
    rs_lookback_days: int,
    min_rs_spread_pct: float,
    atr_channel_ma_length: int,
    atr_channel_atr_length: int,
    atr_channel_ma_type: str,
    atr_lower1_proximity_pct: float,
) -> None:
    storage = Storage(data_root)
    _set_scan_job(
        job_id,
        status="running",
        phase="Refreshing ADX / DI candles",
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
        refreshed_symbols = _refresh_adx_di_candles(storage, progress_callback=progress_callback)
        if not refreshed_symbols:
            raise RuntimeError("No fresh NSE daily candles were available for the ADX screener run.")
        result = run_adx_di_study(
            storage,
            exchange="NSE",
            symbols=refreshed_symbols,
            length=length,
            threshold=threshold,
            cross_lookback_bars=cross_lookback_bars,
            trend_fast_ma_length=trend_fast_ma_length,
            trend_slow_ma_length=trend_slow_ma_length,
            volume_avg_lookback=volume_avg_lookback,
            min_volume_ratio=min_volume_ratio,
            breakout_lookback_days=breakout_lookback_days,
            rs_lookback_days=rs_lookback_days,
            min_rs_spread_pct=min_rs_spread_pct,
            atr_channel_ma_length=atr_channel_ma_length,
            atr_channel_atr_length=atr_channel_atr_length,
            atr_channel_ma_type=atr_channel_ma_type,
            atr_lower1_proximity_pct=atr_lower1_proximity_pct,
            progress_callback=progress_callback,
        )
        save_adx_di_outputs(result, _adx_di_dir(data_root))
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
            redirect_url=f"/adx-di?study_ran=1{query_suffix}",
        )
    except Exception as exc:
        _set_scan_job(
            job_id,
            status="failed",
            phase="Failed",
            error=str(exc),
            redirect_url=f"/adx-di?study_error={quote(str(exc)[:500])}{query_suffix}",
        )


def _run_minervini_di_divergence_job(
    job_id: str,
    data_root: Path,
    query_suffix: str,
    adx_length: int,
    divergence_days: int,
    min_score: float,
) -> None:
    storage = Storage(data_root)
    _set_scan_job(
        job_id,
        status="running",
        phase="Refreshing NSE daily candles",
        completed=0,
        total=0,
        percent=0,
        current_symbol="",
        current_exchange="NSE",
    )

    def refresh_progress_callback(payload: dict[str, Any]) -> None:
        total = int(payload.get("total") or 0)
        completed = int(payload.get("completed") or 0)
        percent = int((completed / total) * 50) if total else 0
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

    def scan_progress_callback(payload: dict[str, Any]) -> None:
        total = int(payload.get("total") or 0)
        completed = int(payload.get("completed") or 0)
        percent = 50 + (int((completed / total) * 50) if total else 0)
        _set_scan_job(
            job_id,
            status="running",
            phase=payload.get("phase", "Running"),
            completed=completed,
            total=total,
            percent=max(50, min(percent, 100)),
            current_symbol=payload.get("current_symbol", ""),
            current_exchange=payload.get("current_exchange", ""),
        )

    try:
        refreshed_symbols = _refresh_adx_di_candles(storage, progress_callback=refresh_progress_callback)
        if not refreshed_symbols:
            raise RuntimeError("No fresh NSE daily candles were available for the combined screener run.")
        _set_scan_job(
            job_id,
            status="running",
            phase="Refreshing NIFTY 500 benchmark",
            completed=0,
            total=len(refreshed_symbols),
            percent=50,
            current_symbol="",
            current_exchange="NSE",
        )
        _refresh_minervini_quality_benchmark(storage, MINERVINI_QUALITY_DEFAULT_BENCHMARK)
        result = run_minervini_di_divergence_study(
            storage,
            exchange="NSE",
            symbols=refreshed_symbols,
            adx_length=adx_length,
            divergence_days=divergence_days,
            min_score=min_score,
            benchmark_symbol=MINERVINI_QUALITY_DEFAULT_BENCHMARK,
            progress_callback=scan_progress_callback,
        )
        save_minervini_di_divergence_outputs(result, _minervini_di_divergence_dir(data_root))
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
            redirect_url=f"/minervini-di-divergence?study_ran=1{query_suffix}",
        )
    except Exception as exc:
        _set_scan_job(
            job_id,
            status="failed",
            phase="Failed",
            error=str(exc),
            redirect_url=f"/minervini-di-divergence?study_error={quote(str(exc)[:500])}{query_suffix}",
        )


def _run_minervini_sheet_job(job_id: str, data_root: Path, query_suffix: str) -> None:
    storage = Storage(data_root)
    _set_scan_job(
        job_id,
        status="running",
        phase="Reading Google Sheet",
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
        result = run_minervini_sheet_sync(storage, data_root, progress_callback=progress_callback)
        save_minervini_sheet_sync_outputs(result, _minervini_sheet_sync_dir(data_root))
        _set_scan_job(
            job_id,
            status="completed",
            phase="Complete",
            completed=int(result.summary.get("sheet_row_count", 0)),
            total=int(result.summary.get("sheet_row_count", 0)),
            percent=100,
            current_symbol="",
            current_exchange="",
            summary=result.summary,
            redirect_url=f"/minervini-sheet?study_ran=1{query_suffix}",
        )
    except Exception as exc:
        _set_scan_job(
            job_id,
            status="failed",
            phase="Failed",
            error=str(exc),
            redirect_url=f"/minervini-sheet?study_error={quote(str(exc)[:500])}{query_suffix}",
        )


def _maybe_run_minervini_sheet_sync_after_screener(storage: Storage, data_root: Path) -> dict[str, Any]:
    settings = load_google_sheets_settings(data_root)
    oauth = google_oauth_status(data_root)
    if not settings.spreadsheet_id:
        return {"status": "skipped", "reason": "sheet_not_configured"}
    if not oauth.get("logged_in"):
        return {"status": "skipped", "reason": "google_login_required"}
    result = run_minervini_sheet_sync(storage, data_root)
    save_minervini_sheet_sync_outputs(result, _minervini_sheet_sync_dir(data_root))
    return {"status": "completed", **result.summary}


def _run_rotation_study_job(job_id: str, config: dict[str, Any], data_root: Path, query_suffix: str) -> None:
    storage = Storage(data_root)
    _set_scan_job(
        job_id,
        status="running",
        phase="Starting Rotation Study",
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
        result = run_rotation_study(config, storage, exchange="NSE", progress_callback=progress_callback)
        save_rotation_study_outputs(result, _rotation_study_dir(data_root))
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
            redirect_url=_rotation_study_redirect_url(result.summary, query_suffix),
        )
    except Exception as exc:
        _set_scan_job(
            job_id,
            status="failed",
            phase="Failed",
            error=str(exc),
            redirect_url=_rotation_study_error_url(exc, query_suffix),
        )


def _run_signal_outcome_job(
    job_id: str,
    config: dict[str, Any],
    data_root: Path,
    query_suffix: str,
    signal_scope: str,
    target_gain_pct: float,
) -> None:
    storage = Storage(data_root)
    _set_scan_job(
        job_id,
        status="running",
        phase="Starting Signal Outcome Study",
        completed=0,
        total=0,
        percent=0,
        current_symbol="",
        current_exchange="NSE",
    )

    def outcome_progress_callback(payload: dict[str, Any]) -> None:
        total = int(payload.get("total") or 0)
        completed = int(payload.get("completed") or 0)
        raw_percent = int((completed / total) * 100) if total else 0
        _set_scan_job(
            job_id,
            status="running",
            phase=payload.get("phase", "Running"),
            completed=completed,
            total=total,
            percent=max(0, min(raw_percent, 100)),
            current_symbol=payload.get("current_symbol", ""),
            current_exchange=payload.get("current_exchange", ""),
        )

    try:
        result = run_signal_outcome_study(
            config,
            storage,
            exchange="NSE",
            signal_scope=signal_scope,
            target_gain_pct=target_gain_pct,
            progress_callback=outcome_progress_callback,
        )
        save_signal_outcome_outputs(result, _signal_outcome_study_dir(data_root))
        _set_scan_job(
            job_id,
            status="completed",
            phase="Complete",
            completed=int(result.summary.get("current_signal_universe_count", 0)),
            total=int(result.summary.get("current_signal_universe_count", 0)),
            percent=100,
            current_symbol="",
            current_exchange="",
            summary=result.summary,
            redirect_url=_signal_outcome_redirect_url(result.summary, query_suffix),
        )
    except Exception as exc:
        _set_scan_job(
            job_id,
            status="failed",
            phase="Failed",
            error=str(exc),
            redirect_url=_signal_outcome_error_url(exc, query_suffix),
        )


def _run_swing_trade_job(job_id: str, config: dict[str, Any], data_root: Path, query_suffix: str) -> None:
    storage = Storage(data_root)
    _set_scan_job(
        job_id,
        status="running",
        phase="Starting Swing Trade Study",
        completed=0,
        total=0,
        percent=0,
        current_symbol="",
        current_exchange="NSE",
    )

    def swing_progress_callback(payload: dict[str, Any]) -> None:
        total = int(payload.get("total") or 0)
        completed = int(payload.get("completed") or 0)
        raw_percent = int((completed / total) * 100) if total else 0
        _set_scan_job(
            job_id,
            status="running",
            phase=payload.get("phase", "Running"),
            completed=completed,
            total=total,
            percent=max(0, min(raw_percent, 100)),
            current_symbol=payload.get("current_symbol", ""),
            current_exchange=payload.get("current_exchange", ""),
        )

    try:
        result = run_swing_trade_study(config, storage, exchange="NSE", progress_callback=swing_progress_callback)
        save_swing_trade_outputs(result, _swing_trade_study_dir(data_root))
        _set_scan_job(
            job_id,
            status="completed",
            phase="Complete",
            completed=int(result.summary.get("symbols_scored", 0)),
            total=int(result.summary.get("symbols_scored", 0)),
            percent=100,
            current_symbol="",
            current_exchange="",
            summary=result.summary,
            redirect_url=_swing_trade_redirect_url(result.summary, query_suffix),
        )
    except Exception as exc:
        _set_scan_job(
            job_id,
            status="failed",
            phase="Failed",
            error=str(exc),
            redirect_url=_swing_trade_error_url(exc, query_suffix),
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
    with BIG_BULL_DEALS_CACHE_LOCK:
        fetched_at = float(BIG_BULL_DEALS_CACHE.get("fetched_at") or 0.0)
        cached_rows = BIG_BULL_DEALS_CACHE.get("rows")
        if (
            isinstance(cached_rows, pd.DataFrame)
            and time.monotonic() - fetched_at < BIG_BULL_DEALS_CACHE_TTL_SECONDS
        ):
            return cached_rows.copy()

    default_from, default_to = default_last_7_days_range()
    try:
        rows = SupabaseStore().list_large_deals(
            limit=1000,
            from_date=default_from.isoformat(),
            to_date=default_to.isoformat(),
            timeout=3,
        )
        if rows:
            deals = pd.DataFrame(rows)
            with BIG_BULL_DEALS_CACHE_LOCK:
                BIG_BULL_DEALS_CACHE["fetched_at"] = time.monotonic()
                BIG_BULL_DEALS_CACHE["rows"] = deals.copy()
            return deals
    except Exception as exc:
        print(f"Supabase large deals unavailable; falling back to CSV: {exc}")

    path = data_root / "deals" / "big_bull_trades.csv"
    if not path.exists():
        deals = pd.DataFrame(
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
    else:
        deals = pd.read_csv(path)

    with BIG_BULL_DEALS_CACHE_LOCK:
        BIG_BULL_DEALS_CACHE["fetched_at"] = time.monotonic()
        BIG_BULL_DEALS_CACHE["rows"] = deals.copy()
    return deals


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


def _weekday_pressure_dir(data_root: Path) -> Path:
    path = data_root / "weekday_pressure_study"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _latest_weekday_pressure_paths(data_root: Path) -> dict[str, Path]:
    directory = _weekday_pressure_dir(data_root)
    return {
        "summary": directory / "latest_summary.csv",
        "stock_stats": directory / "latest_stock_stats.csv",
        "weekday_details": directory / "latest_weekday_details.csv",
    }


def _weekday_pressure_cache_is_fresh(
    cached: WeekdayPressureStudyResult,
    latest_week_date: str,
    signal_symbols: set[str],
) -> bool:
    if not cached.summary:
        return False
    if str(cached.summary.get("latest_week_date", "")) != latest_week_date:
        return False
    source_symbol_count = int(pd.to_numeric(cached.summary.get("source_symbol_count", 0), errors="coerce") or 0)
    return source_symbol_count == len(signal_symbols)


def _build_latest_weekday_pressure_cache(
    config: dict[str, Any],
    storage: Storage,
    data_root: Path,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> WeekdayPressureStudyResult:
    signal_rows = _latest_weekly_buy_sell_frame(data_root)
    signal_symbols = _symbols_from_frame(signal_rows)
    latest_week_date = ""
    if not signal_rows.empty and "date" in signal_rows.columns:
        latest_dates = pd.to_datetime(signal_rows["date"], errors="coerce").dropna()
        if not latest_dates.empty:
            latest_week_date = latest_dates.max().strftime("%Y-%m-%d")
    result = run_weekday_pressure_study(
        config,
        storage,
        exchange="NSE",
        symbols=signal_symbols,
        progress_callback=progress_callback,
    )
    summary = dict(result.summary)
    summary["latest_week_date"] = latest_week_date
    summary["source_signal_count"] = int(len(signal_rows))
    summary["source_symbol_count"] = int(len(signal_symbols))
    enriched_result = WeekdayPressureStudyResult(
        summary=summary,
        stock_stats=result.stock_stats,
        weekday_details=result.weekday_details,
    )
    save_weekday_pressure_outputs(enriched_result, _weekday_pressure_dir(data_root))
    return enriched_result


def _ensure_latest_weekday_pressure_cache(
    config: dict[str, Any],
    storage: Storage,
    data_root: Path,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> WeekdayPressureStudyResult:
    signal_rows = _latest_weekly_buy_sell_frame(data_root)
    signal_symbols = _symbols_from_frame(signal_rows)
    latest_week_date = ""
    if not signal_rows.empty and "date" in signal_rows.columns:
        latest_dates = pd.to_datetime(signal_rows["date"], errors="coerce").dropna()
        if not latest_dates.empty:
            latest_week_date = latest_dates.max().strftime("%Y-%m-%d")
    cached = load_weekday_pressure_outputs(_weekday_pressure_dir(data_root))
    if _weekday_pressure_cache_is_fresh(cached, latest_week_date, signal_symbols):
        return cached
    return _build_latest_weekday_pressure_cache(
        config,
        storage,
        data_root,
        progress_callback=progress_callback,
    )


def _merge_cached_weekday_profiles(
    config: dict[str, Any],
    storage: Storage,
    data_root: Path,
    frame: pd.DataFrame,
    exchange_column: str = "exchange",
    symbol_column: str = "symbol",
) -> pd.DataFrame:
    if frame.empty or symbol_column not in frame.columns:
        return frame.copy()

    signal_rows = _latest_weekly_buy_sell_frame(data_root)
    signal_symbols = _symbols_from_frame(signal_rows)
    latest_week_date = ""
    if not signal_rows.empty and "date" in signal_rows.columns:
        latest_dates = pd.to_datetime(signal_rows["date"], errors="coerce").dropna()
        if not latest_dates.empty:
            latest_week_date = latest_dates.max().strftime("%Y-%m-%d")
    cached = load_weekday_pressure_outputs(_weekday_pressure_dir(data_root))
    if not _weekday_pressure_cache_is_fresh(cached, latest_week_date, signal_symbols):
        merged = frame.copy()
        if "best_buy_weekday" not in merged.columns:
            merged["best_buy_weekday"] = pd.NA
        if "best_sell_weekday" not in merged.columns:
            merged["best_sell_weekday"] = pd.NA
        return merged
    if cached.stock_stats.empty or "symbol" not in cached.stock_stats.columns:
        merged = frame.copy()
        if "best_buy_weekday" not in merged.columns:
            merged["best_buy_weekday"] = pd.NA
        if "best_sell_weekday" not in merged.columns:
            merged["best_sell_weekday"] = pd.NA
        return merged

    profiles = cached.stock_stats.copy()
    profiles["symbol"] = profiles["symbol"].astype(str).str.upper().str.strip()
    if "exchange" not in profiles.columns:
        profiles["exchange"] = "NSE"
    profiles["exchange"] = profiles["exchange"].astype(str).str.upper().str.strip()

    merged = frame.copy()
    if exchange_column not in merged.columns:
        merged[exchange_column] = "NSE"
    merged[exchange_column] = merged[exchange_column].astype(str).str.upper().str.strip()
    merged[symbol_column] = merged[symbol_column].astype(str).str.upper().str.strip()
    merged = merged.merge(
        profiles[["exchange", "symbol", "best_buy_weekday", "best_sell_weekday"]],
        left_on=[exchange_column, symbol_column],
        right_on=["exchange", "symbol"],
        how="left",
        suffixes=("", "_weekday"),
    )
    return merged.drop(columns=["exchange_weekday", "symbol_weekday"], errors="ignore")


def _strategy_lab_dir(data_root: Path) -> Path:
    path = data_root / "strategy_lab"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sensitivity_overlap_dir(data_root: Path) -> Path:
    path = data_root / "sensitivity_overlap_study"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _weekly_buy_tracker_dir(data_root: Path) -> Path:
    path = data_root / "weekly_buy_tracker"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _qm_quality_dir(data_root: Path) -> Path:
    path = data_root / "qm_quality"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _volume_burst_dir(data_root: Path) -> Path:
    path = data_root / "volume_burst"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resistance_breaks_dir(data_root: Path) -> Path:
    path = data_root / "resistance_breaks"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _adx_di_dir(data_root: Path) -> Path:
    path = data_root / "adx_di"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _minervini_sheet_sync_dir(data_root: Path) -> Path:
    path = data_root / "minervini_sheet_sync"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _minervini_quality_dir(data_root: Path) -> Path:
    path = data_root / "minervini_quality"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _minervini_di_divergence_dir(data_root: Path) -> Path:
    path = data_root / "minervini_di_divergence"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _latest_strategy_lab_paths(data_root: Path) -> dict[str, Path]:
    directory = _strategy_lab_dir(data_root)
    return {
        "summary": directory / "latest_summary.csv",
        "strategy_stats": directory / "latest_strategy_stats.csv",
        "trade_details": directory / "latest_trade_details.csv",
        "signal_universe": directory / "latest_signal_universe.csv",
    }


def _latest_sensitivity_overlap_paths(data_root: Path) -> dict[str, Path]:
    directory = _sensitivity_overlap_dir(data_root)
    return {
        "summary": directory / "latest_summary.csv",
        "weekly_breakdown": directory / "latest_weekly_breakdown.csv",
        "latest_cohort": directory / "latest_latest_cohort.csv",
        "conversion_details": directory / "latest_conversion_details.csv",
    }


def _rotation_study_dir(data_root: Path) -> Path:
    path = data_root / "rotation_study"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _latest_rotation_study_paths(data_root: Path) -> dict[str, Path]:
    directory = _rotation_study_dir(data_root)
    return {
        "summary": directory / "latest_summary.csv",
        "groups": directory / "latest_groups.csv",
        "members": directory / "latest_members.csv",
        "candidates": directory / "latest_candidates.csv",
    }


def _signal_outcome_study_dir(data_root: Path) -> Path:
    path = data_root / "signal_outcome_study"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _latest_signal_outcome_paths(data_root: Path) -> dict[str, Path]:
    directory = _signal_outcome_study_dir(data_root)
    return {
        "summary": directory / "latest_summary.csv",
        "signal_universe": directory / "latest_signal_universe.csv",
        "stock_stats": directory / "latest_stock_stats.csv",
        "pair_details": directory / "latest_pair_details.csv",
        "workbook": directory / "signal_outcome_study_report.xlsx",
    }


def _swing_trade_study_dir(data_root: Path) -> Path:
    path = data_root / "swing_trade_study"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _rotation_group_chart_rows(
    data_root: Path,
    config: dict[str, Any],
    groups: pd.DataFrame,
    members: pd.DataFrame,
    selected_group_id: str,
) -> list[dict[str, Any]]:
    if groups.empty or members.empty:
        return []

    chart_groups = groups.copy()
    if selected_group_id:
        chart_groups = chart_groups[chart_groups["group_id"].astype(str).str.upper() == selected_group_id.upper()]
    else:
        chart_groups = chart_groups.head(6)

    rows: list[dict[str, Any]] = []
    for _, group_row in chart_groups.iterrows():
        group_id = str(group_row.get("group_id", ""))
        if not group_id:
            continue
        group_members = members[members["group_id"].astype(str) == group_id].copy()
        if group_members.empty:
            continue
        chart_html = build_rotation_group_chart(data_root, config, group_id, group_members)
        if not chart_html:
            continue
        rows.append(
            {
                "group_id": group_id,
                "group_size": group_row.get("group_size", ""),
                "leaders_count": group_row.get("leaders_count", ""),
                "catch_up_candidates_count": group_row.get("catch_up_candidates_count", ""),
                "latest_weekly_buy_count": group_row.get("latest_weekly_buy_count", ""),
                "latest_weekly_sell_count": group_row.get("latest_weekly_sell_count", ""),
                "chart_html": chart_html,
            }
        )
    return rows


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


def _fetch_and_store_big_bull_deals(dashboard_token: str = "", sensitivity: str = "") -> RedirectResponse:
    params: list[str] = []
    try:
        result = fetch_and_store_current_large_deals()
        params.extend(
            [
                "refreshed=1",
                f"rows={result['stored']}",
                f"fetched={result['fetched']}",
                f"skipped_existing_dates={result.get('skipped_existing_dates', 0)}",
            ]
        )
    except Exception as exc:
        params.append(f"fetch_error={quote(str(exc)[:500])}")
    if dashboard_token:
        params.append(f"token={quote(dashboard_token)}")
    if sensitivity:
        params.append(f"sensitivity={quote(sensitivity)}")
    return RedirectResponse(f"/big-bull-deals?{'&'.join(params)}", status_code=303)


@app.post("/big-bull-deals/fetch")
async def fetch_big_bull_deals_post(request: Request) -> RedirectResponse:
    form = await request.form()
    dashboard_token = str(form.get("token", "")).strip()
    sensitivity = str(form.get("sensitivity", "")).strip()
    return _fetch_and_store_big_bull_deals(dashboard_token, sensitivity)


@app.get("/big-bull-deals/fetch")
def fetch_big_bull_deals_get(request: Request) -> RedirectResponse:
    dashboard_token = request.query_params.get("token", "").strip()
    sensitivity = request.query_params.get("sensitivity", "").strip()
    return _fetch_and_store_big_bull_deals(dashboard_token, sensitivity)


@app.get("/health", response_class=PlainTextResponse)
def health() -> str:
    return "ok"


def _temporarily_removed_response(request: Request, page_name: str) -> HTMLResponse:
    if not _is_allowed(request):
        return templates.TemplateResponse(
            "locked.html",
            {"request": request, "app_name": "Investment Screener"},
            status_code=401,
        )

    config = load_config()
    _, base_sensitivity, selected_sensitivity = _apply_request_sensitivity(config, request)
    data_root = get_data_root(config)
    common_filter_context = _common_filter_context(request, selected_sensitivity, config, data_root)
    return templates.TemplateResponse(
        "page_unavailable.html",
        {
            "request": request,
            "app_name": config.get("app", {}).get("name", "Investment Screener"),
            "dashboard_token": request.query_params.get("token", ""),
            "selected_sensitivity": selected_sensitivity,
            "default_sensitivity": base_sensitivity,
            "page_name": page_name,
            "message": f"{page_name} is temporarily removed from the workspace for now.",
            **common_filter_context,
            "show_shared_filter_form": False,
            "show_shared_filter_status": False,
        },
        status_code=404,
    )


@app.get("/backtest", response_class=HTMLResponse)
def backtest_page(request: Request) -> HTMLResponse:
    return _temporarily_removed_response(request, "Backtest")


@app.post("/backtest/run")
async def run_backtest_from_dashboard(request: Request) -> RedirectResponse:
    raise HTTPException(status_code=404, detail="Backtest is temporarily removed from the workspace for now.")


@app.get("/backtest/report")
def download_backtest_report() -> FileResponse:
    raise HTTPException(status_code=404, detail="Backtest is temporarily removed from the workspace for now.")


@app.get("/gtt-gain-study", response_class=HTMLResponse)
def gtt_gain_study_page(request: Request) -> HTMLResponse:
    if not _is_allowed(request):
        return templates.TemplateResponse(
            "locked.html",
            {"request": request, "app_name": "Investment Screener"},
            status_code=401,
        )

    config = load_config()
    config, base_sensitivity, selected_sensitivity = _apply_request_sensitivity(config, request)
    data_root = get_data_root(config)
    common_filter_context = _common_filter_context(request, selected_sensitivity, config, data_root)
    latest = load_gtt_gain_outputs(_gtt_gain_dir(data_root))
    latest_stock_stats = _align_gtt_stock_stats_to_latest_universe(data_root, latest.stock_stats, config)
    latest_universe_symbols = _latest_kite_universe_symbols(data_root, config)
    metadata = _combined_symbol_metadata(config, Storage(data_root))
    latest_stock_stats = _enrich_with_symbol_metadata(latest_stock_stats, metadata, "symbol")
    stock_search = request.query_params.get("stock_search", "").strip()
    sensitivity_text = str(selected_sensitivity)
    selected_market_cap_bucket = request.query_params.get("market_cap_bucket", "").strip()
    min_market_cap_text = request.query_params.get("min_market_cap_cr", "").strip()
    max_market_cap_text = request.query_params.get("max_market_cap_cr", "").strip()
    min_cmp_text = request.query_params.get("min_cmp", "").strip()
    max_cmp_text = request.query_params.get("max_cmp", "").strip()
    min_market_cap = _optional_float(min_market_cap_text)
    max_market_cap = _optional_float(max_market_cap_text)
    min_cmp = _optional_float(min_cmp_text)
    max_cmp = _optional_float(max_cmp_text)
    require_volume_confirmation = _request_bool(request, "require_volume_confirmation")
    require_obv_confirmation = _request_bool(request, "require_obv_confirmation")
    require_screener_trend_confirmation = _request_bool(request, "require_trend_confirmation")
    selected_return_metric = request.query_params.get("return_metric", "median_3").strip() or "median_3"
    if selected_return_metric not in {"last_1", "median_3"}:
        selected_return_metric = "median_3"
    min_pair_return_text = request.query_params.get("min_pair_return_pct", "").strip()
    selected_peak_speed_bucket = request.query_params.get("peak_speed_bucket", "").strip()
    if selected_peak_speed_bucket not in GTT_PEAK_SPEED_BUCKETS:
        selected_peak_speed_bucket = ""
    selected_technical_rating_statuses = _normalize_gtt_technical_rating_statuses(
        request.query_params.getlist("technical_rating_status")
    )
    open_buy_regime_only = _request_bool(request, "open_buy_regime_only") or _request_bool(request, "latest_buy_only")
    dashboard_buy_only = _request_bool(request, "dashboard_buy_only")
    fresh_weekly_buy_only = _request_bool(request, "fresh_weekly_buy_only")
    fresh_daily_buy_only = _request_bool(request, "fresh_daily_buy_only")
    trend_only = _request_bool(request, "trend_only")
    dashboard_buy_symbols = _dashboard_buy_symbols(data_root)
    fresh_daily_buy_symbols = _daily_buy_symbols(data_root)
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
    stock_stats = _apply_cmp_filters(stock_stats, min_cmp, max_cmp, "latest_close")
    stock_stats = _apply_stock_search(stock_stats, stock_search)
    stock_stats = _apply_gtt_stock_filters(
        stock_stats,
        open_buy_regime_only,
        False,
        dashboard_buy_only,
        dashboard_buy_symbols,
        fresh_weekly_buy_only,
        fresh_daily_buy_only,
        fresh_daily_buy_symbols,
        False,
        False,
        [],
    )
    if selected_technical_rating_statuses:
        stock_stats = _ensure_gtt_weekly_technical_ratings(data_root, stock_stats, config)
    if trend_only or require_volume_confirmation or require_obv_confirmation:
        stock_stats = _ensure_gtt_latest_signal_context(data_root, stock_stats, config)
    stock_stats_after_screener_filter_count = len(stock_stats)
    stock_stats_before_filter_count = len(stock_stats)
    gtt_filter_warning = _gtt_filter_warning(
        stock_stats,
        open_buy_regime_only,
        trend_only,
        dashboard_buy_only,
        fresh_weekly_buy_only,
        dashboard_buy_symbols,
        fresh_daily_buy_only,
        fresh_daily_buy_symbols,
        require_volume_confirmation,
        require_obv_confirmation,
        selected_technical_rating_statuses,
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
        fresh_daily_buy_only,
        fresh_daily_buy_symbols,
        require_volume_confirmation,
        require_obv_confirmation,
        selected_technical_rating_statuses,
    )
    bucket_chart_stock_stats = stock_stats.copy()
    stock_stats_before_bucket_count = len(bucket_chart_stock_stats)
    stock_stats = _apply_peak_speed_bucket_filter(stock_stats, selected_peak_speed_bucket)
    stock_stats = _ensure_gtt_s2_to_s3_markers(data_root, stock_stats, config)
    if (
        open_buy_regime_only
        or dashboard_buy_only
        or fresh_weekly_buy_only
        or fresh_daily_buy_only
        or trend_only
        or require_volume_confirmation
        or require_obv_confirmation
        or selected_peak_speed_bucket
        or selected_technical_rating_statuses
        or min_cmp is not None
        or max_cmp is not None
    ):
        visible_symbols = set(stock_stats["symbol"].astype(str)) if "symbol" in stock_stats.columns else set()
        pair_details = _filter_by_symbols(pair_details, visible_symbols)
        open_positions = _filter_by_symbols(open_positions, visible_symbols)
    gtt_opportunity_chart_html = ""
    if fresh_weekly_buy_only and trend_only:
        gtt_opportunity_chart_html = build_gtt_opportunity_chart(bucket_chart_stock_stats)
        gtt_opportunity_chart_message = (
            "No chartable stocks remain after the Fresh weekly BUY and daily EMA stack filters."
            if not gtt_opportunity_chart_html
            else ""
        )
    else:
        gtt_opportunity_chart_message = (
            "Apply Fresh weekly BUY only and Daily EMA stack to plot the bucket chart."
        )
    workbook_path = _latest_gtt_gain_paths(data_root)["workbook"]
    gtt_filter_query = _gtt_filter_query(
        token=request.query_params.get("token", ""),
        stock_search=stock_search,
        sensitivity=sensitivity_text,
        market_cap_bucket=selected_market_cap_bucket,
        min_market_cap_cr=min_market_cap_text,
        max_market_cap_cr=max_market_cap_text,
        min_cmp=min_cmp_text,
        max_cmp=max_cmp_text,
        open_buy_regime_only=open_buy_regime_only,
        dashboard_buy_only=dashboard_buy_only,
        fresh_weekly_buy_only=fresh_weekly_buy_only,
        fresh_daily_buy_only=fresh_daily_buy_only,
        trend_only=trend_only,
        require_volume_confirmation=require_volume_confirmation,
        require_obv_confirmation=require_obv_confirmation,
        require_screener_trend_confirmation=require_screener_trend_confirmation,
        return_metric=selected_return_metric if min_pair_return_text else "",
        min_pair_return_pct=min_pair_return_text,
        peak_speed_bucket=selected_peak_speed_bucket,
        technical_rating_statuses=selected_technical_rating_statuses,
    )
    active_gtt_filter_summary = _gtt_filter_summary(
        stock_search,
        sensitivity_text,
        selected_market_cap_bucket,
        min_market_cap_text,
        max_market_cap_text,
        min_cmp_text,
        max_cmp_text,
        open_buy_regime_only,
        dashboard_buy_only,
        fresh_weekly_buy_only,
        fresh_daily_buy_only,
        trend_only,
        require_volume_confirmation,
        require_obv_confirmation,
        require_screener_trend_confirmation,
        selected_return_metric,
        min_pair_return_text,
        selected_peak_speed_bucket,
        selected_technical_rating_statuses,
    )
    rendered_stock_stats = stock_stats.head(GTT_TABLE_RENDER_LIMIT).copy()
    stock_stats_truncated = len(rendered_stock_stats) < len(stock_stats)

    return templates.TemplateResponse(
        "gtt_gain_study.html",
        {
            "request": request,
            "app_name": config.get("app", {}).get("name", "Investment Screener"),
            "dashboard_token": request.query_params.get("token", ""),
            "selected_sensitivity": selected_sensitivity,
            "default_sensitivity": base_sensitivity,
            "summary": display_summary,
            "stock_stats": _records(rendered_stock_stats),
            "stock_symbols_csv": _comma_separated_symbols(stock_stats),
            "gtt_peak_speed_buckets": GTT_PEAK_SPEED_BUCKETS,
            "selected_peak_speed_bucket": selected_peak_speed_bucket,
            "selected_technical_rating_statuses": selected_technical_rating_statuses,
            "gtt_technical_rating_filters": GTT_TECHNICAL_RATING_FILTERS,
            "stock_stats_before_bucket_count": stock_stats_before_bucket_count,
            "gtt_opportunity_chart_html": gtt_opportunity_chart_html,
            "gtt_opportunity_chart_message": gtt_opportunity_chart_message,
            "pair_details": _records(pair_details.head(150)),
            "open_positions": _records(open_positions.head(100)),
            "stock_search": stock_search,
            "open_buy_regime_only": open_buy_regime_only,
            "dashboard_buy_only": dashboard_buy_only,
            "fresh_weekly_buy_only": fresh_weekly_buy_only,
            "fresh_daily_buy_only": fresh_daily_buy_only,
            "trend_only": trend_only,
            "stock_stats_count": len(stock_stats),
            "rendered_stock_stats_count": len(rendered_stock_stats),
            "stock_stats_truncated": stock_stats_truncated,
            "stock_stats_render_limit": GTT_TABLE_RENDER_LIMIT,
            "stock_stats_before_filter_count": stock_stats_before_filter_count,
            "stock_stats_after_screener_filter_count": stock_stats_after_screener_filter_count,
            "selected_market_cap_bucket": selected_market_cap_bucket,
            "selected_min_market_cap": min_market_cap_text,
            "selected_max_market_cap": max_market_cap_text,
            "selected_min_cmp": min_cmp_text,
            "selected_max_cmp": max_cmp_text,
            "require_volume_confirmation": require_volume_confirmation,
            "require_obv_confirmation": require_obv_confirmation,
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
            "telegram_sent_count": request.query_params.get("telegram_sent_count", ""),
            "telegram_error": request.query_params.get("telegram_error", ""),
            **common_filter_context,
            "show_shared_filter_form": False,
            "show_shared_filter_status": False,
        },
    )


@app.post("/gtt-gain-study/run")
async def run_gtt_gain_study_from_dashboard(request: Request, background_tasks: BackgroundTasks) -> RedirectResponse:
    config = load_config()
    form = await request.form()
    selected_sensitivity = _parse_sensitivity_text(
        str(form.get("sensitivity", "")),
        int(config.get("strategy", {}).get("sensitivity", 3)),
    ) or int(config.get("strategy", {}).get("sensitivity", 3))
    if selected_sensitivity != int(config.get("strategy", {}).get("sensitivity", 3)):
        config = deepcopy(config)
        config.setdefault("strategy", {})["sensitivity"] = selected_sensitivity
    data_root = get_data_root(config)
    dashboard_token = request.query_params.get("token", "").strip()
    params = []
    if dashboard_token:
        params.append(f"token={quote(dashboard_token)}")
    params.append(f"sensitivity={selected_sensitivity}")
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
    data_root = get_data_root(config)
    gtt_dir = _gtt_gain_dir(data_root)
    workbook_path = _latest_gtt_gain_paths(data_root)["workbook"]
    result = load_gtt_gain_outputs(gtt_dir)
    if result.stock_stats.empty and result.pair_details.empty and not workbook_path.exists():
        raise HTTPException(status_code=404, detail="GTT gain study report has not been generated yet.")
    if not result.stock_stats.empty or not result.pair_details.empty:
        write_gtt_gain_workbook(result, workbook_path)
    return FileResponse(
        workbook_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="gtt_gain_study_report.xlsx",
    )


@app.get("/weekday-study", response_class=HTMLResponse)
def weekday_study_page(request: Request) -> HTMLResponse:
    if not _is_allowed(request):
        return templates.TemplateResponse(
            "locked.html",
            {"request": request, "app_name": "Investment Screener"},
            status_code=401,
        )

    config = load_config()
    _, base_sensitivity, selected_sensitivity = _apply_request_sensitivity(config, request)
    data_root = get_data_root(config)
    common_filter_context = _common_filter_context(request, selected_sensitivity, config, data_root)
    latest = load_weekday_pressure_outputs(_weekday_pressure_dir(data_root))
    signal_frame = _latest_weekly_buy_sell_frame(data_root)

    signal_symbol_column = _symbol_column(signal_frame)
    if signal_frame.empty or not signal_symbol_column:
        merged = latest.stock_stats.copy()
        if not merged.empty and "signal" not in merged.columns:
            merged["signal"] = ""
        if not merged.empty and "signal_date" not in merged.columns:
            merged["signal_date"] = ""
    else:
        base = signal_frame.copy()
        base["symbol"] = base[signal_symbol_column].astype(str).str.upper().str.strip()
        base["signal"] = base["signal"].astype(str).str.upper().str.strip()
        base["signal_date"] = pd.to_datetime(base["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        keep_columns = ["exchange", "symbol", "name", "signal", "signal_date"]
        if "close" in base.columns:
            keep_columns.append("close")
            base = base.rename(columns={"close": "signal_close"})
            keep_columns[-1] = "signal_close"
        base = base[keep_columns].drop_duplicates(subset=["exchange", "symbol"], keep="last")
        merged = base
        if not latest.stock_stats.empty:
            merged = merged.merge(
                latest.stock_stats,
                on=["exchange", "symbol"],
                how="left",
                suffixes=("", "_study"),
            )
            if "name_study" in merged.columns:
                merged["name"] = merged["name"].fillna("").astype(str).str.strip().mask(
                    lambda s: s == "",
                    merged["name_study"].fillna("").astype(str).str.strip(),
                )
                merged = merged.drop(columns=["name_study"])

    stock_search = request.query_params.get("stock_search", "").strip()
    selected_signal = request.query_params.get("signal", "").strip().upper()
    if selected_signal not in {"", "BUY", "SELL"}:
        selected_signal = ""
    selected_buy_weekday = request.query_params.get("best_buy_weekday", "").strip()
    if selected_buy_weekday not in WEEKDAY_ORDER:
        selected_buy_weekday = ""
    selected_sell_weekday = request.query_params.get("best_sell_weekday", "").strip()
    if selected_sell_weekday not in WEEKDAY_ORDER:
        selected_sell_weekday = ""

    if not merged.empty:
        merged = _apply_stock_search(merged, stock_search)
        if selected_signal:
            merged = merged[merged["signal"].astype(str).str.upper() == selected_signal]
        if selected_buy_weekday and "best_buy_weekday" in merged.columns:
            merged = merged[merged["best_buy_weekday"].astype(str) == selected_buy_weekday]
        if selected_sell_weekday and "best_sell_weekday" in merged.columns:
            merged = merged[merged["best_sell_weekday"].astype(str) == selected_sell_weekday]
        sort_columns = []
        ascending = []
        if "best_buy_pressure_score" in merged.columns:
            merged["best_buy_pressure_score"] = pd.to_numeric(merged["best_buy_pressure_score"], errors="coerce")
            sort_columns.append("best_buy_pressure_score")
            ascending.append(False)
        if "symbol" in merged.columns:
            sort_columns.append("symbol")
            ascending.append(True)
        if sort_columns:
            merged = merged.sort_values(sort_columns, ascending=ascending, na_position="last")

    selected_symbol = request.query_params.get("symbol", "").strip().upper()
    if not selected_symbol and not merged.empty and "symbol" in merged.columns:
        selected_symbol = str(merged.iloc[0].get("symbol") or "").upper()

    detail_rows = latest.weekday_details.copy()
    if not detail_rows.empty:
        detail_rows["symbol"] = detail_rows["symbol"].astype(str).str.upper().str.strip()
        if selected_symbol:
            detail_rows = detail_rows[detail_rows["symbol"] == selected_symbol]
        if not detail_rows.empty:
            detail_rows["weekday"] = pd.Categorical(detail_rows["weekday"], categories=WEEKDAY_ORDER, ordered=True)
            detail_rows = detail_rows.sort_values("weekday")

    selected_row = {}
    if selected_symbol and not merged.empty and "symbol" in merged.columns:
        selected_match = merged[merged["symbol"].astype(str).str.upper() == selected_symbol]
        if not selected_match.empty:
            selected_row = selected_match.iloc[0].to_dict()

    signal_counts = (
        signal_frame["signal"].astype(str).str.upper().value_counts().to_dict()
        if not signal_frame.empty and "signal" in signal_frame.columns
        else {}
    )

    return templates.TemplateResponse(
        "weekday_study.html",
        {
            "request": request,
            "app_name": config.get("app", {}).get("name", "Investment Screener"),
            "dashboard_token": request.query_params.get("token", ""),
            "selected_sensitivity": selected_sensitivity,
            "default_sensitivity": base_sensitivity,
            "summary": latest.summary,
            "stock_stats": _records(merged),
            "stock_stats_count": len(merged),
            "signal_universe_count": len(signal_frame),
            "buy_signal_count": signal_counts.get("BUY", 0),
            "sell_signal_count": signal_counts.get("SELL", 0),
            "stock_search": stock_search,
            "selected_signal": selected_signal,
            "selected_buy_weekday": selected_buy_weekday,
            "selected_sell_weekday": selected_sell_weekday,
            "weekday_options": WEEKDAY_ORDER,
            "selected_symbol": selected_symbol,
            "selected_row": selected_row,
            "weekday_details": _records(detail_rows),
            "weekday_job": request.query_params.get("weekday_job", ""),
            "study_ran": request.query_params.get("study_ran", ""),
            "study_error": request.query_params.get("study_error", ""),
            **common_filter_context,
            "show_shared_filter_form": False,
            "show_shared_filter_status": False,
        },
    )


@app.post("/weekday-study/run")
async def run_weekday_study_from_dashboard(request: Request, background_tasks: BackgroundTasks) -> RedirectResponse:
    config = load_config()
    data_root = get_data_root(config)
    form = await request.form()

    dashboard_token = str(form.get("token", "")).strip()
    sensitivity_text = str(form.get("sensitivity", "")).strip()
    query_suffix = ""
    if dashboard_token:
        query_suffix += f"&token={quote(dashboard_token)}"
    if sensitivity_text:
        query_suffix += f"&sensitivity={quote(sensitivity_text)}"

    try:
        job_id = uuid4().hex
        _set_scan_job(job_id, status="queued", phase="Queued", completed=0, total=0, percent=0)
        background_tasks.add_task(_run_weekday_study_job, job_id, config, data_root, query_suffix)
        redirect_url = f"/weekday-study?weekday_job={job_id}{query_suffix}"
    except Exception as exc:
        redirect_url = _weekday_study_error_url(exc, query_suffix)
    return RedirectResponse(redirect_url, status_code=303)


@app.get("/strategy-lab", response_class=HTMLResponse)
def strategy_lab_page(request: Request) -> HTMLResponse:
    return _temporarily_removed_response(request, "Strategy Lab")


@app.get("/sensitivity-study", response_class=HTMLResponse)
def sensitivity_overlap_page(request: Request) -> HTMLResponse:
    return _temporarily_removed_response(request, "Sensitivity Study")


@app.post("/sensitivity-study/run")
async def run_sensitivity_overlap_from_dashboard(request: Request, background_tasks: BackgroundTasks) -> RedirectResponse:
    raise HTTPException(status_code=404, detail="Sensitivity Study is temporarily removed from the workspace for now.")


@app.post("/strategy-lab/run")
async def run_strategy_lab_from_dashboard(request: Request, background_tasks: BackgroundTasks) -> RedirectResponse:
    raise HTTPException(status_code=404, detail="Strategy Lab is temporarily removed from the workspace for now.")


@app.get("/weekly-buy-gains", response_class=HTMLResponse)
def weekly_buy_gains_page(request: Request) -> HTMLResponse:
    if not _is_allowed(request):
        return templates.TemplateResponse(
            "locked.html",
            {"request": request, "app_name": "Investment Screener"},
            status_code=401,
        )

    config = load_config()
    _, base_sensitivity, selected_sensitivity = _apply_request_sensitivity(config, request)
    data_root = get_data_root(config)
    common_filter_context = _common_filter_context(request, selected_sensitivity, config, data_root)
    latest = load_weekly_buy_tracker_outputs(_weekly_buy_tracker_dir(data_root))
    start_date = request.query_params.get("start_date", WEEKLY_BUY_GAINS_DEFAULT_START_DATE).strip() or WEEKLY_BUY_GAINS_DEFAULT_START_DATE
    stock_search = request.query_params.get("stock_search", "").strip()
    minervini_only = _truthy_param(request.query_params.getlist("minervini_only"), default=False)
    obv_macd_only = _truthy_param(request.query_params.getlist("obv_macd_only"), default=False)
    latest_volume_only = _truthy_param(request.query_params.getlist("latest_volume_only"), default=False)

    stock_stats = latest.stock_stats.copy()
    if not stock_stats.empty:
        for column in (
            "first_buy_date",
            "latest_buy_date",
            "first_s2_buy_date",
            "latest_s2_buy_date",
            "first_s3_buy_date",
            "latest_s3_buy_date",
            "latest_close_date",
        ):
            if column in stock_stats.columns:
                stock_stats[column] = pd.to_datetime(stock_stats[column], errors="coerce").dt.strftime("%Y-%m-%d")
        stock_stats = _apply_stock_search(stock_stats, stock_search)
        if minervini_only:
            if "minervini_pass" in stock_stats.columns:
                stock_stats = stock_stats[_truthy_series(stock_stats["minervini_pass"])].copy()
            else:
                stock_stats = stock_stats.iloc[0:0].copy()
        if obv_macd_only:
            if "obv_macd_cross_up" in stock_stats.columns:
                stock_stats = stock_stats[_truthy_series(stock_stats["obv_macd_cross_up"])].copy()
            else:
                stock_stats = stock_stats.iloc[0:0].copy()
        if latest_volume_only:
            if "latest_volume_3x_prev_9d" in stock_stats.columns:
                stock_stats = stock_stats[_truthy_series(stock_stats["latest_volume_3x_prev_9d"])].copy()
            else:
                stock_stats = stock_stats.iloc[0:0].copy()
    gainers = stock_stats.copy()
    if not gainers.empty and "gain_vs_latest_buy_pct" in gainers.columns:
        gainers["gain_vs_latest_buy_pct"] = pd.to_numeric(gainers["gain_vs_latest_buy_pct"], errors="coerce")
        gainers = gainers[gainers["gain_vs_latest_buy_pct"] > 0].copy()
        gainers = gainers.sort_values(["gain_vs_latest_buy_pct", "symbol"], ascending=[False, True], na_position="last")

    profitable_count = int(len(gainers))
    total_count = int(len(stock_stats))
    avg_gain = (
        pd.to_numeric(stock_stats["gain_vs_latest_buy_pct"], errors="coerce").dropna()
        if "gain_vs_latest_buy_pct" in stock_stats.columns
        else pd.Series(dtype=float)
    )
    avg_gain_value = round(float(avg_gain.mean()), 2) if not avg_gain.empty else None
    stock_symbols_csv = _comma_separated_symbols(stock_stats)
    obv_macd_count = int(_truthy_series(stock_stats["obv_macd_cross_up"]).sum()) if not stock_stats.empty and "obv_macd_cross_up" in stock_stats.columns else 0
    latest_volume_count = int(_truthy_series(stock_stats["latest_volume_3x_prev_9d"]).sum()) if not stock_stats.empty and "latest_volume_3x_prev_9d" in stock_stats.columns else 0
    minervini_notice = ""
    if minervini_only and latest.stock_stats is not None and not latest.stock_stats.empty and "minervini_pass" not in latest.stock_stats.columns:
        minervini_notice = "Minervini results are not available yet. Run Weekly Buy Gains once more to populate them."
    obv_macd_notice = ""
    if obv_macd_only and latest.stock_stats is not None and not latest.stock_stats.empty and "obv_macd_cross_up" not in latest.stock_stats.columns:
        obv_macd_notice = "OBV MACD results are not available yet. Run Weekly Buy Gains once more to populate them."
    latest_volume_notice = ""
    if latest_volume_only and latest.stock_stats is not None and not latest.stock_stats.empty and "latest_volume_3x_prev_9d" not in latest.stock_stats.columns:
        latest_volume_notice = "Latest-volume burst results are not available yet. Run Weekly Buy Gains once more to populate them."

    return templates.TemplateResponse(
        "weekly_buy_gains.html",
        {
            "request": request,
            "app_name": config.get("app", {}).get("name", "Investment Screener"),
            "dashboard_token": request.query_params.get("token", ""),
            "selected_sensitivity": selected_sensitivity,
            "default_sensitivity": base_sensitivity,
            "summary": latest.summary,
            "quality_benchmark_symbol": str(latest.summary.get("quality_benchmark_symbol", "") or ""),
            "stock_stats": _records(stock_stats),
            "stock_stats_count": len(stock_stats),
            "gainers": _records(gainers),
            "gainers_count": profitable_count,
            "avg_current_gain_pct": avg_gain_value,
            "profitable_share_pct": round((profitable_count / total_count) * 100.0, 2) if total_count else None,
            "stock_symbols_csv": stock_symbols_csv,
            "obv_macd_count": obv_macd_count,
            "latest_volume_count": latest_volume_count,
            "minervini_only": minervini_only,
            "minervini_notice": minervini_notice,
            "obv_macd_only": obv_macd_only,
            "obv_macd_notice": obv_macd_notice,
            "latest_volume_only": latest_volume_only,
            "latest_volume_notice": latest_volume_notice,
            "start_date": start_date,
            "stock_search": stock_search,
            "study_job": request.query_params.get("study_job", ""),
            "study_ran": request.query_params.get("study_ran", ""),
            "study_error": request.query_params.get("study_error", ""),
            **common_filter_context,
            "show_shared_filter_form": False,
            "show_shared_filter_status": False,
        },
    )


@app.post("/weekly-buy-gains/run")
async def run_weekly_buy_gains_from_dashboard(request: Request, background_tasks: BackgroundTasks) -> RedirectResponse:
    config = load_config()
    data_root = get_data_root(config)
    form = await request.form()

    dashboard_token = str(form.get("token", "")).strip()
    sensitivity_text = str(form.get("sensitivity", "")).strip()
    start_date = str(form.get("start_date", WEEKLY_BUY_GAINS_DEFAULT_START_DATE)).strip() or WEEKLY_BUY_GAINS_DEFAULT_START_DATE
    minervini_only = _truthy_param(form.getlist("minervini_only"), default=False)
    obv_macd_only = _truthy_param(form.getlist("obv_macd_only"), default=False)
    latest_volume_only = _truthy_param(form.getlist("latest_volume_only"), default=False)
    params = [f"start_date={quote(start_date)}"]
    if dashboard_token:
        params.append(f"token={quote(dashboard_token)}")
    if sensitivity_text:
        params.append(f"sensitivity={quote(sensitivity_text)}")
    params.append(f"minervini_only={'1' if minervini_only else '0'}")
    params.append(f"obv_macd_only={'1' if obv_macd_only else '0'}")
    params.append(f"latest_volume_only={'1' if latest_volume_only else '0'}")
    query_suffix = "&" + "&".join(params)

    try:
        job_id = uuid4().hex
        _set_scan_job(job_id, status="queued", phase="Queued", completed=0, total=0, percent=0)
        background_tasks.add_task(_run_weekly_buy_tracker_job, job_id, config, data_root, query_suffix, start_date)
        redirect_url = f"/weekly-buy-gains?study_job={job_id}{query_suffix}"
    except Exception as exc:
        redirect_url = _weekly_buy_gains_error_url(exc, query_suffix)
    return RedirectResponse(redirect_url, status_code=303)


@app.get("/volume-burst", response_class=HTMLResponse)
def volume_burst_page(request: Request) -> HTMLResponse:
    if not _is_allowed(request):
        return templates.TemplateResponse(
            "locked.html",
            {"request": request, "app_name": "Investment Screener"},
            status_code=401,
        )

    config = load_config()
    _, base_sensitivity, selected_sensitivity = _apply_request_sensitivity(config, request)
    data_root = get_data_root(config)
    common_filter_context = _common_filter_context(request, selected_sensitivity, config, data_root)
    latest = load_volume_burst_outputs(_volume_burst_dir(data_root))
    stock_search = request.query_params.get("stock_search", "").strip()
    matches_only = _truthy_param(request.query_params.getlist("matches_only"), default=False)

    stock_stats = latest.stock_stats.copy()
    if not stock_stats.empty:
        if "latest_close_date" in stock_stats.columns:
            stock_stats["latest_close_date"] = pd.to_datetime(stock_stats["latest_close_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        stock_stats = _apply_stock_search(stock_stats, stock_search)
        if matches_only:
            if "latest_volume_3x_prev_9d" in stock_stats.columns:
                stock_stats = stock_stats[_truthy_series(stock_stats["latest_volume_3x_prev_9d"])].copy()
            else:
                stock_stats = stock_stats.iloc[0:0].copy()

    ratio_series = (
        pd.to_numeric(stock_stats["latest_volume_ratio_prev_9d"], errors="coerce").dropna()
        if "latest_volume_ratio_prev_9d" in stock_stats.columns
        else pd.Series(dtype=float)
    )
    return templates.TemplateResponse(
        "volume_burst.html",
        {
            "request": request,
            "app_name": config.get("app", {}).get("name", "Investment Screener"),
            "dashboard_token": request.query_params.get("token", ""),
            "selected_sensitivity": selected_sensitivity,
            "default_sensitivity": base_sensitivity,
            "summary": latest.summary,
            "stock_stats": _records(stock_stats),
            "stock_stats_count": len(stock_stats),
            "stock_symbols_csv": _comma_separated_symbols(stock_stats),
            "stock_search": stock_search,
            "matches_only": matches_only,
            "avg_ratio": round(float(ratio_series.mean()), 2) if not ratio_series.empty else None,
            "max_ratio": round(float(ratio_series.max()), 2) if not ratio_series.empty else None,
            "study_job": request.query_params.get("study_job", ""),
            "study_ran": request.query_params.get("study_ran", ""),
            "study_error": request.query_params.get("study_error", ""),
            **common_filter_context,
            "show_shared_filter_form": False,
            "show_shared_filter_status": False,
        },
    )


@app.post("/volume-burst/run")
async def run_volume_burst_from_dashboard(request: Request, background_tasks: BackgroundTasks) -> RedirectResponse:
    config = load_config()
    data_root = get_data_root(config)
    form = await request.form()

    dashboard_token = str(form.get("token", "")).strip()
    sensitivity_text = str(form.get("sensitivity", "")).strip()
    params = []
    if dashboard_token:
        params.append(f"token={quote(dashboard_token)}")
    if sensitivity_text:
        params.append(f"sensitivity={quote(sensitivity_text)}")
    query_suffix = "&" + "&".join(params) if params else ""

    try:
        job_id = uuid4().hex
        _set_scan_job(job_id, status="queued", phase="Queued", completed=0, total=0, percent=0)
        background_tasks.add_task(_run_volume_burst_job, job_id, data_root, query_suffix)
        redirect_url = f"/volume-burst?study_job={job_id}{query_suffix}"
    except Exception as exc:
        redirect_url = _volume_burst_error_url(exc, query_suffix)
    return RedirectResponse(redirect_url, status_code=303)


@app.get("/resistance-breaks", response_class=HTMLResponse)
def resistance_breaks_page(request: Request) -> HTMLResponse:
    if not _is_allowed(request):
        return templates.TemplateResponse(
            "locked.html",
            {"request": request, "app_name": "Investment Screener"},
            status_code=401,
        )

    config = load_config()
    _, base_sensitivity, selected_sensitivity = _apply_request_sensitivity(config, request)
    data_root = get_data_root(config)
    common_filter_context = _common_filter_context(request, selected_sensitivity, config, data_root)
    latest = load_resistance_breaks_outputs(_resistance_breaks_dir(data_root))
    stock_search = request.query_params.get("stock_search", "").strip()
    matches_only = _truthy_param(request.query_params.getlist("matches_only"), default=False)
    left_bars = int(request.query_params.get("left_bars", latest.summary.get("left_bars", 15)) or 15)
    right_bars = int(request.query_params.get("right_bars", latest.summary.get("right_bars", 15)) or 15)
    volume_avg_window = int(request.query_params.get("volume_avg_window", latest.summary.get("volume_avg_window", 20)) or 20)
    volume_multiplier = float(request.query_params.get("volume_multiplier", latest.summary.get("volume_multiplier", 2.0)) or 2.0)
    min_break_count = int(request.query_params.get("min_break_count", latest.summary.get("min_break_count", 2)) or 2)
    recent_window_days = int(request.query_params.get("recent_window_days", latest.summary.get("recent_breakout_window_days", 7)) or 7)

    stock_stats = latest.stock_stats.copy()
    if not stock_stats.empty:
        if "latest_close_date" in stock_stats.columns:
            stock_stats["latest_close_date"] = pd.to_datetime(stock_stats["latest_close_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        stock_stats = _apply_stock_search(stock_stats, stock_search)
        if matches_only:
            if "passes_volume_confirmed_resistance_breaks" in stock_stats.columns:
                stock_stats = stock_stats[_truthy_series(stock_stats["passes_volume_confirmed_resistance_breaks"])].copy()
            else:
                stock_stats = stock_stats.iloc[0:0].copy()

    breakout_events = latest.breakout_events.copy()
    if not breakout_events.empty:
        if "date" in breakout_events.columns:
            breakout_events["date"] = pd.to_datetime(breakout_events["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        if stock_search:
            breakout_events = _apply_stock_search(breakout_events, stock_search)

    break_counts = (
        pd.to_numeric(stock_stats["volume_confirmed_resistance_break_count"], errors="coerce").dropna()
        if "volume_confirmed_resistance_break_count" in stock_stats.columns
        else pd.Series(dtype=float)
    )
    return templates.TemplateResponse(
        "resistance_breaks.html",
        {
            "request": request,
            "app_name": config.get("app", {}).get("name", "Investment Screener"),
            "dashboard_token": request.query_params.get("token", ""),
            "selected_sensitivity": selected_sensitivity,
            "default_sensitivity": base_sensitivity,
            "summary": latest.summary,
            "stock_stats": _records(stock_stats),
            "stock_stats_count": len(stock_stats),
            "breakout_events": _records(breakout_events.head(300)),
            "breakout_events_count": len(breakout_events),
            "stock_symbols_csv": _comma_separated_symbols(stock_stats),
            "stock_search": stock_search,
            "matches_only": matches_only,
            "left_bars": left_bars,
            "right_bars": right_bars,
            "volume_avg_window": volume_avg_window,
            "volume_multiplier": volume_multiplier,
            "min_break_count": min_break_count,
            "recent_window_days": recent_window_days,
            "avg_break_count": round(float(break_counts.mean()), 2) if not break_counts.empty else None,
            "max_break_count": int(break_counts.max()) if not break_counts.empty else None,
            "study_job": request.query_params.get("study_job", ""),
            "study_ran": request.query_params.get("study_ran", ""),
            "study_error": request.query_params.get("study_error", ""),
            **common_filter_context,
            "show_shared_filter_form": False,
            "show_shared_filter_status": False,
        },
    )


@app.post("/resistance-breaks/run")
async def run_resistance_breaks_from_dashboard(request: Request, background_tasks: BackgroundTasks) -> RedirectResponse:
    config = load_config()
    data_root = get_data_root(config)
    form = await request.form()

    dashboard_token = str(form.get("token", "")).strip()
    sensitivity_text = str(form.get("sensitivity", "")).strip()
    left_bars = max(int(str(form.get("left_bars", "15")).strip() or "15"), 1)
    right_bars = max(int(str(form.get("right_bars", "15")).strip() or "15"), 1)
    volume_avg_window = max(int(str(form.get("volume_avg_window", "20")).strip() or "20"), 2)
    volume_multiplier = max(float(str(form.get("volume_multiplier", "2")).strip() or "2"), 0.1)
    min_break_count = max(int(str(form.get("min_break_count", "2")).strip() or "2"), 1)
    recent_window_days = max(int(str(form.get("recent_window_days", "7")).strip() or "7"), 1)
    params = [
        f"left_bars={left_bars}",
        f"right_bars={right_bars}",
        f"volume_avg_window={volume_avg_window}",
        f"volume_multiplier={quote(str(volume_multiplier))}",
        f"min_break_count={min_break_count}",
        f"recent_window_days={recent_window_days}",
    ]
    if dashboard_token:
        params.append(f"token={quote(dashboard_token)}")
    if sensitivity_text:
        params.append(f"sensitivity={quote(sensitivity_text)}")
    query_suffix = "&" + "&".join(params)

    try:
        job_id = uuid4().hex
        _set_scan_job(job_id, status="queued", phase="Queued", completed=0, total=0, percent=0)
        background_tasks.add_task(
            _run_resistance_breaks_job,
            job_id,
            data_root,
            query_suffix,
            left_bars,
            right_bars,
            volume_avg_window,
            volume_multiplier,
            min_break_count,
            recent_window_days,
        )
        redirect_url = f"/resistance-breaks?study_job={job_id}{query_suffix}"
    except Exception as exc:
        redirect_url = f"/resistance-breaks?study_error={quote(str(exc)[:500])}{query_suffix}"
    return RedirectResponse(redirect_url, status_code=303)


@app.get("/minervini-quality", response_class=HTMLResponse)
def minervini_quality_page(request: Request) -> HTMLResponse:
    if not _is_allowed(request):
        return templates.TemplateResponse(
            "locked.html",
            {"request": request, "app_name": "Investment Screener"},
            status_code=401,
        )

    config = load_config()
    _, base_sensitivity, selected_sensitivity = _apply_request_sensitivity(config, request)
    data_root = get_data_root(config)
    common_filter_context = _common_filter_context(request, selected_sensitivity, config, data_root)
    latest = load_minervini_quality_outputs(_minervini_quality_dir(data_root))
    stock_search = request.query_params.get("stock_search", "").strip()
    qualified_only = _truthy_param(request.query_params.getlist("qualified_only"), default=True)
    try:
        score_threshold = float(
            request.query_params.get(
                "score_threshold",
                latest.summary.get("score_threshold", MINERVINI_QUALITY_DEFAULT_THRESHOLD),
            )
            or MINERVINI_QUALITY_DEFAULT_THRESHOLD
        )
    except (TypeError, ValueError):
        score_threshold = MINERVINI_QUALITY_DEFAULT_THRESHOLD
    score_threshold = max(0.0, min(score_threshold, 99.0))

    stock_stats = latest.stock_stats.copy()
    qualified_stock_stats = pd.DataFrame()
    view_summary = dict(latest.summary)
    if not stock_stats.empty:
        stock_stats = _apply_stock_search(stock_stats, stock_search)
        score_columns = ("stock_quality_score", "setup_quality_score", "entry_quality_score")
        for column in score_columns:
            if column not in stock_stats.columns:
                stock_stats[column] = np.nan
            stock_stats[column] = pd.to_numeric(stock_stats[column], errors="coerce")
        if "is_latest_market_date" in stock_stats.columns:
            current_date_mask = _truthy_series(stock_stats["is_latest_market_date"])
        else:
            benchmark_date = pd.to_datetime(
                latest.summary.get("benchmark_latest_date"),
                errors="coerce",
            )
            stock_dates = pd.to_datetime(
                stock_stats.get("latest_date", pd.Series(pd.NaT, index=stock_stats.index)),
                errors="coerce",
            )
            current_date_mask = stock_dates.dt.normalize().eq(benchmark_date)
        ready_mask = stock_stats.get(
            "data_status",
            pd.Series("", index=stock_stats.index),
        ).astype(str).eq("READY")
        threshold_mask = (
            current_date_mask
            & ready_mask
            & (stock_stats["stock_quality_score"] >= 75.0)
            & (stock_stats["setup_quality_score"] >= 70.0)
            & (stock_stats["entry_quality_score"] >= 70.0)
        )
        for column in score_columns:
            threshold_mask &= stock_stats[column] > score_threshold
        stock_stats["quality_pass"] = threshold_mask
        view_summary["qualified_stocks"] = int(threshold_mask.sum())
        qualified_stock_stats = stock_stats[threshold_mask].copy()
        if qualified_only:
            stock_stats = qualified_stock_stats.copy()
        stock_stats = stock_stats.sort_values(
            ["entry_quality_score", "setup_quality_score", "stock_quality_score", "symbol"],
            ascending=[False, False, False, True],
            na_position="last",
        )

    return templates.TemplateResponse(
        "minervini_quality.html",
        {
            "request": request,
            "app_name": config.get("app", {}).get("name", "Investment Screener"),
            "dashboard_token": request.query_params.get("token", ""),
            "selected_sensitivity": selected_sensitivity,
            "default_sensitivity": base_sensitivity,
            "summary": view_summary,
            "stock_stats": _records(stock_stats),
            "stock_stats_count": len(stock_stats),
            "stock_symbols_csv": _comma_separated_symbols(qualified_stock_stats),
            "stock_search": stock_search,
            "qualified_only": qualified_only,
            "score_threshold": score_threshold,
            "study_job": request.query_params.get("study_job", ""),
            "study_ran": request.query_params.get("study_ran", ""),
            "study_error": request.query_params.get("study_error", ""),
            **common_filter_context,
            "show_shared_filter_form": False,
            "show_shared_filter_status": False,
        },
    )


@app.post("/minervini-quality/run")
async def run_minervini_quality_from_dashboard(request: Request, background_tasks: BackgroundTasks) -> RedirectResponse:
    config = load_config()
    data_root = get_data_root(config)
    form = await request.form()
    dashboard_token = str(form.get("token", "")).strip()
    sensitivity_text = str(form.get("sensitivity", "")).strip()
    try:
        score_threshold = float(
            str(form.get("score_threshold", MINERVINI_QUALITY_DEFAULT_THRESHOLD)).strip()
            or MINERVINI_QUALITY_DEFAULT_THRESHOLD
        )
    except ValueError:
        score_threshold = MINERVINI_QUALITY_DEFAULT_THRESHOLD
    score_threshold = max(0.0, min(score_threshold, 99.0))
    params = [f"score_threshold={quote(str(score_threshold))}", "qualified_only=1"]
    if dashboard_token:
        params.append(f"token={quote(dashboard_token)}")
    if sensitivity_text:
        params.append(f"sensitivity={quote(sensitivity_text)}")
    query_suffix = "&" + "&".join(params)

    try:
        job_id = uuid4().hex
        _set_scan_job(job_id, status="queued", phase="Queued", completed=0, total=0, percent=0)
        background_tasks.add_task(
            _run_minervini_quality_job,
            job_id,
            data_root,
            query_suffix,
            score_threshold,
        )
        redirect_url = f"/minervini-quality?study_job={job_id}{query_suffix}"
    except Exception as exc:
        redirect_url = f"/minervini-quality?study_error={quote(str(exc)[:500])}{query_suffix}"
    return RedirectResponse(redirect_url, status_code=303)


@app.get("/minervini-di-divergence", response_class=HTMLResponse)
def minervini_di_divergence_page(request: Request) -> HTMLResponse:
    if not _is_allowed(request):
        return templates.TemplateResponse(
            "locked.html",
            {"request": request, "app_name": "Investment Screener"},
            status_code=401,
        )

    config = load_config()
    _, base_sensitivity, selected_sensitivity = _apply_request_sensitivity(config, request)
    data_root = get_data_root(config)
    common_filter_context = _common_filter_context(request, selected_sensitivity, config, data_root)
    latest = load_minervini_di_divergence_outputs(_minervini_di_divergence_dir(data_root))
    stock_search = request.query_params.get("stock_search", "").strip()
    matches_only = _truthy_param(request.query_params.getlist("matches_only"), default=True)

    try:
        adx_length = max(
            int(request.query_params.get("adx_length", latest.summary.get("adx_length", MINERVINI_DI_DEFAULT_ADX_LENGTH)) or MINERVINI_DI_DEFAULT_ADX_LENGTH),
            2,
        )
    except (TypeError, ValueError):
        adx_length = MINERVINI_DI_DEFAULT_ADX_LENGTH
    try:
        divergence_days = max(
            int(request.query_params.get("divergence_days", latest.summary.get("divergence_days", MINERVINI_DI_DEFAULT_DIVERGENCE_DAYS)) or MINERVINI_DI_DEFAULT_DIVERGENCE_DAYS),
            1,
        )
    except (TypeError, ValueError):
        divergence_days = MINERVINI_DI_DEFAULT_DIVERGENCE_DAYS
    try:
        min_score = float(
            request.query_params.get("min_score", latest.summary.get("min_score", MINERVINI_DI_DEFAULT_MIN_SCORE))
            or MINERVINI_DI_DEFAULT_MIN_SCORE
        )
    except (TypeError, ValueError):
        min_score = MINERVINI_DI_DEFAULT_MIN_SCORE
    min_score = max(0.0, min(min_score, 100.0))

    stock_stats = latest.stock_stats.copy()
    combined_stock_stats = pd.DataFrame()
    pre_breakout_stock_stats = pd.DataFrame()
    view_summary = dict(latest.summary)
    if not stock_stats.empty:
        stock_stats = _apply_stock_search(stock_stats, stock_search)
        score_columns = ("stock_quality_score", "setup_quality_score", "entry_quality_score")
        for column in score_columns:
            if column not in stock_stats.columns:
                stock_stats[column] = np.nan
            stock_stats[column] = pd.to_numeric(stock_stats[column], errors="coerce")
        if "spread_change_2d" not in stock_stats.columns:
            stock_stats["spread_change_2d"] = np.nan
        stock_stats["spread_change_2d"] = pd.to_numeric(stock_stats["spread_change_2d"], errors="coerce")
        divergence_mask = _truthy_series(
            stock_stats.get("di_divergence_pass", pd.Series(False, index=stock_stats.index))
        )
        ready_mask = stock_stats.get("data_status", pd.Series("", index=stock_stats.index)).astype(str).eq("READY")
        current_date_mask = _truthy_series(
            stock_stats.get("is_latest_market_date", pd.Series(True, index=stock_stats.index))
        )
        minervini_mask = ready_mask & current_date_mask
        for column in score_columns:
            minervini_mask &= stock_stats[column] >= min_score
        combined_mask = divergence_mask & minervini_mask & current_date_mask
        stock_stats["minervini_threshold_pass"] = minervini_mask
        stock_stats["combined_pass"] = combined_mask
        pre_breakout_mask = _truthy_series(
            stock_stats.get("pre_breakout_pass", pd.Series(False, index=stock_stats.index))
        ) & current_date_mask
        stock_stats["pre_breakout_pass"] = pre_breakout_mask
        combined_stock_stats = stock_stats[combined_mask].copy()
        pre_breakout_stock_stats = stock_stats[pre_breakout_mask].copy()
        if not pre_breakout_stock_stats.empty:
            pre_breakout_stock_stats = pre_breakout_stock_stats.sort_values(
                ["stock_quality_score", "setup_quality_score", "entry_quality_score", "spread_change_2d", "symbol"],
                ascending=[False, False, False, False, True],
                na_position="last",
            )
        view_summary["di_divergence_matches"] = int((divergence_mask & current_date_mask).sum())
        view_summary["minervini_threshold_matches"] = int(minervini_mask.sum())
        view_summary["combined_matches"] = int(combined_mask.sum())
        view_summary["pre_breakout_matches"] = int(pre_breakout_mask.sum())
        if matches_only:
            stock_stats = combined_stock_stats.copy()
        stock_stats = stock_stats.sort_values(
            ["combined_pass", "spread_change_2d", "entry_quality_score", "setup_quality_score", "stock_quality_score", "symbol"],
            ascending=[False, False, False, False, False, True],
            na_position="last",
        )

    return templates.TemplateResponse(
        "minervini_di_divergence.html",
        {
            "request": request,
            "app_name": config.get("app", {}).get("name", "Investment Screener"),
            "dashboard_token": request.query_params.get("token", ""),
            "selected_sensitivity": selected_sensitivity,
            "default_sensitivity": base_sensitivity,
            "summary": view_summary,
            "stock_stats": _records(stock_stats),
            "stock_stats_count": len(stock_stats),
            "stock_symbols_csv": _comma_separated_symbols(combined_stock_stats),
            "pre_breakout_stock_stats": _records(pre_breakout_stock_stats),
            "pre_breakout_stock_symbols_csv": _comma_separated_symbols(pre_breakout_stock_stats),
            "stock_search": stock_search,
            "matches_only": matches_only,
            "adx_length": adx_length,
            "divergence_days": divergence_days,
            "min_score": min_score,
            "study_job": request.query_params.get("study_job", ""),
            "study_ran": request.query_params.get("study_ran", ""),
            "study_error": request.query_params.get("study_error", ""),
            **common_filter_context,
            "show_shared_filter_form": False,
            "show_shared_filter_status": False,
        },
    )


@app.post("/minervini-di-divergence/run")
async def run_minervini_di_divergence_from_dashboard(
    request: Request,
    background_tasks: BackgroundTasks,
) -> RedirectResponse:
    config = load_config()
    data_root = get_data_root(config)
    form = await request.form()
    dashboard_token = str(form.get("token", "")).strip()
    sensitivity_text = str(form.get("sensitivity", "")).strip()
    try:
        adx_length = max(int(str(form.get("adx_length", MINERVINI_DI_DEFAULT_ADX_LENGTH)).strip()), 2)
    except ValueError:
        adx_length = MINERVINI_DI_DEFAULT_ADX_LENGTH
    try:
        divergence_days = max(int(str(form.get("divergence_days", MINERVINI_DI_DEFAULT_DIVERGENCE_DAYS)).strip()), 1)
    except ValueError:
        divergence_days = MINERVINI_DI_DEFAULT_DIVERGENCE_DAYS
    try:
        min_score = float(str(form.get("min_score", MINERVINI_DI_DEFAULT_MIN_SCORE)).strip())
    except ValueError:
        min_score = MINERVINI_DI_DEFAULT_MIN_SCORE
    min_score = max(0.0, min(min_score, 100.0))

    params = [
        f"adx_length={adx_length}",
        f"divergence_days={divergence_days}",
        f"min_score={quote(str(min_score))}",
        "matches_only=1",
    ]
    if dashboard_token:
        params.append(f"token={quote(dashboard_token)}")
    if sensitivity_text:
        params.append(f"sensitivity={quote(sensitivity_text)}")
    query_suffix = "&" + "&".join(params)

    try:
        job_id = uuid4().hex
        _set_scan_job(job_id, status="queued", phase="Queued", completed=0, total=0, percent=0)
        background_tasks.add_task(
            _run_minervini_di_divergence_job,
            job_id,
            data_root,
            query_suffix,
            adx_length,
            divergence_days,
            min_score,
        )
        redirect_url = f"/minervini-di-divergence?study_job={job_id}{query_suffix}"
    except Exception as exc:
        redirect_url = f"/minervini-di-divergence?study_error={quote(str(exc)[:500])}{query_suffix}"
    return RedirectResponse(redirect_url, status_code=303)


@app.get("/adx-di", response_class=HTMLResponse)
def adx_di_page(request: Request) -> HTMLResponse:
    if not _is_allowed(request):
        return templates.TemplateResponse(
            "locked.html",
            {"request": request, "app_name": "Investment Screener"},
            status_code=401,
        )

    config = load_config()
    _, base_sensitivity, selected_sensitivity = _apply_request_sensitivity(config, request)
    data_root = get_data_root(config)
    common_filter_context = _common_filter_context(request, selected_sensitivity, config, data_root)
    latest = load_adx_di_outputs(_adx_di_dir(data_root))
    stock_search = request.query_params.get("stock_search", "").strip()
    matches_only = _truthy_param(request.query_params.getlist("matches_only"), default=False)
    length = max(int(request.query_params.get("length", latest.summary.get("length", 14)) or 14), 2)
    threshold = float(request.query_params.get("threshold", latest.summary.get("threshold", 20.0)) or 20.0)
    cross_lookback_bars = max(int(request.query_params.get("cross_lookback_bars", latest.summary.get("cross_lookback_bars", 3)) or 3), 1)
    trend_fast_ma_length = max(int(request.query_params.get("trend_fast_ma_length", latest.summary.get("trend_fast_ma_length", 50)) or 50), 2)
    trend_slow_ma_length = max(int(request.query_params.get("trend_slow_ma_length", latest.summary.get("trend_slow_ma_length", 200)) or 200), trend_fast_ma_length + 1)
    volume_avg_lookback = max(int(request.query_params.get("volume_avg_lookback", latest.summary.get("volume_avg_lookback", 20)) or 20), 2)
    min_volume_ratio = max(float(request.query_params.get("min_volume_ratio", latest.summary.get("min_volume_ratio", 1.5)) or 1.5), 0.0)
    breakout_lookback_days = max(int(request.query_params.get("breakout_lookback_days", latest.summary.get("breakout_lookback_days", 20)) or 20), 2)
    rs_lookback_days = max(int(request.query_params.get("rs_lookback_days", latest.summary.get("rs_lookback_days", 20)) or 20), 1)
    min_rs_spread_pct = float(request.query_params.get("min_rs_spread_pct", latest.summary.get("min_rs_spread_pct", 0.0)) or 0.0)
    atr_channel_ma_length = max(int(request.query_params.get("atr_channel_ma_length", latest.summary.get("atr_channel_ma_length", 20)) or 20), 1)
    atr_channel_atr_length = max(int(request.query_params.get("atr_channel_atr_length", latest.summary.get("atr_channel_atr_length", 14)) or 14), 1)
    atr_channel_ma_type = str(request.query_params.get("atr_channel_ma_type", latest.summary.get("atr_channel_ma_type", "EMA")) or "EMA").strip().upper()
    if atr_channel_ma_type not in {"EMA", "SMA"}:
        atr_channel_ma_type = "EMA"
    atr_lower1_proximity_value = str(request.query_params.get("atr_lower1_proximity_pct", latest.summary.get("atr_lower1_proximity_pct", 2.0)) or 2.0).strip()
    min_current_price_value = str(request.query_params.get("min_current_price", "")).strip()
    min_support_distance_value = str(request.query_params.get("min_support_distance_pct", "20")).strip()
    max_support_distance_value = str(request.query_params.get("max_support_distance_pct", "40")).strip()
    try:
        min_current_price = float(min_current_price_value) if min_current_price_value else None
    except ValueError:
        min_current_price = None
    try:
        min_support_distance_pct = float(min_support_distance_value) if min_support_distance_value else 20.0
    except ValueError:
        min_support_distance_pct = 20.0
    try:
        max_support_distance_pct = float(max_support_distance_value) if max_support_distance_value else 40.0
    except ValueError:
        max_support_distance_pct = 40.0
    try:
        atr_lower1_proximity_pct = float(atr_lower1_proximity_value) if atr_lower1_proximity_value else 2.0
    except ValueError:
        atr_lower1_proximity_pct = 2.0
    require_trend_filter = _truthy_param(request.query_params.getlist("require_trend_filter"), default=False)
    require_volume_filter = _truthy_param(request.query_params.getlist("require_volume_filter"), default=False)
    require_breakout_filter = _truthy_param(request.query_params.getlist("require_breakout_filter"), default=False)
    require_rs_filter = _truthy_param(request.query_params.getlist("require_rs_filter"), default=False)
    require_support_filter = _truthy_param(request.query_params.getlist("require_support_filter"), default=False)
    require_threshold_cross_filter = _truthy_param(request.query_params.getlist("require_threshold_cross_filter"), default=False)
    require_atr_lower1_filter = _truthy_param(request.query_params.getlist("require_atr_lower1_filter"), default=False)
    require_obv_cross_filter = _truthy_param(request.query_params.getlist("require_obv_cross_filter"), default=False)
    require_divergence_filter = _truthy_param(request.query_params.getlist("require_divergence_filter"), default=False)
    chart_symbol = str(request.query_params.get("chart_symbol", "")).strip().upper()
    storage = Storage(data_root)

    all_stock_stats = latest.stock_stats.copy()
    stock_stats = all_stock_stats.copy()
    if not stock_stats.empty:
        default_columns: dict[str, Any] = {
            "di_plus_crosses_in_lookback_bars": 0,
            "recent_di_plus_cross_dates_csv": "",
            "latest_di_plus_cross_date": "",
            "di_plus_cross_above_di_minus_recent": False,
            "di_plus_cross_above_di_minus_latest": False,
            "latest_adx_20": pd.NA,
            "adx_3d_ago": pd.NA,
            "adx_above_adx20": False,
            "adx_above_3d_ago": False,
            "adx_shortlist_pass": False,
            "di_plus_divergence_count": 0,
            "recent_di_plus_divergence_dates_csv": "",
            "latest_di_plus_divergence_date": "",
            "di_plus_divergence_recent": False,
            "di_plus_divergence_expanding_latest": False,
            "di_plus_pre_cross_threshold_divergence_count": 0,
            "recent_di_plus_pre_cross_threshold_divergence_dates_csv": "",
            "latest_di_plus_pre_cross_threshold_divergence_date": "",
            "di_plus_pre_cross_threshold_divergence_recent": False,
            "di_plus_pre_cross_threshold_divergence_expanding_latest": False,
            "di_plus_cross_over_threshold_count": 0,
            "recent_di_plus_cross_over_threshold_dates_csv": "",
            "latest_di_plus_cross_over_threshold_date": "",
            "di_plus_cross_over_threshold_recent": False,
            "di_plus_cross_over_threshold_latest": False,
            "obv_latest": pd.NA,
            "obv_sma13": pd.NA,
            "obv_above_sma13": False,
            "obv_cross_sma13_count": 0,
            "recent_obv_cross_sma13_dates_csv": "",
            "latest_obv_cross_sma13_date": "",
            "obv_cross_sma13_recent": False,
            "obv_cross_sma13_latest": False,
            "pine_ema13": pd.NA,
            "pine_ema26": pd.NA,
            "pine_obv": pd.NA,
            "pine_obv_sma100": pd.NA,
            "pine_obv_above_sma100": False,
            "pine_obv_bullish_cross": False,
            "pine_ma50": pd.NA,
            "pine_ma150": pd.NA,
            "pine_ma200": pd.NA,
            "pine_high52week": pd.NA,
            "pine_low52week": pd.NA,
            "pine_trend_template_passed": False,
            "pine_trend_template_text": "",
            "pine_distance_from_50_sma_pct": pd.NA,
            "pine_buy_risk_text": "",
            "pine_buy_risk_pass": False,
            "pine_buy_volume_20": pd.NA,
            "pine_sell_volume_20": pd.NA,
            "pine_buying_pressure": False,
            "pine_pressure_text": "",
            "pine_relative_price_strength": pd.NA,
            "pine_relative_price_strength_text": "",
            "pine_relative_price_strength_pass": False,
            "pine_vcp_range_percent": pd.NA,
            "pine_vcp_triggered": False,
            "pine_vcp_text": "",
            "di_plus_lead_pending": False,
            "trend_fast_ma": pd.NA,
            "trend_slow_ma": pd.NA,
            "trend_fast_ma_slope": pd.NA,
            "trend_slow_ma_slope": pd.NA,
            "trend_filter_pass": False,
            "cross_date": "",
            "cross_close": pd.NA,
            "cross_volume_ratio": pd.NA,
            "volume_filter_pass": False,
            "breakout_level": pd.NA,
            "breakout_extension_pct": pd.NA,
            "breakout_filter_pass": False,
            "support_level": pd.NA,
            "support_level_date": "",
            "support_distance_from_level_pct": pd.NA,
            "support_filter_pass": False,
            "atr_channel_ma": pd.NA,
            "atr_channel_atr": pd.NA,
            "atr_lower1": pd.NA,
            "atr_lower1_distance_pct": pd.NA,
            "atr_lower1_proximity_pass": False,
            "rs_stock_return_pct": pd.NA,
            "rs_benchmark_return_pct": pd.NA,
            "relative_strength_spread_pct": pd.NA,
            "rs_filter_pass": False,
            "quality_score": 0,
        }
        for column, default_value in default_columns.items():
            if column not in stock_stats.columns:
                stock_stats[column] = default_value
        metadata = _combined_symbol_metadata(config, storage)
        stock_stats = _enrich_with_symbol_metadata(stock_stats, metadata, "symbol")
        if "latest_close_date" in stock_stats.columns:
            stock_stats["latest_close_date"] = pd.to_datetime(stock_stats["latest_close_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        stock_stats = _apply_stock_search(stock_stats, stock_search)
        if matches_only:
            di_plus_matches = _truthy_series(stock_stats["di_plus_cross_above_di_minus_recent"]) if "di_plus_cross_above_di_minus_recent" in stock_stats.columns else pd.Series(False, index=stock_stats.index)
            di_plus_still_above = _truthy_series(stock_stats["di_plus_above_di_minus"]) if "di_plus_above_di_minus" in stock_stats.columns else pd.Series(False, index=stock_stats.index)
            adx_shortlist_pass = _truthy_series(stock_stats["adx_shortlist_pass"]) if "adx_shortlist_pass" in stock_stats.columns else pd.Series(False, index=stock_stats.index)
            if (
                len(di_plus_matches) == len(stock_stats)
                and len(di_plus_still_above) == len(stock_stats)
                and len(adx_shortlist_pass) == len(stock_stats)
            ):
                stock_stats = stock_stats[di_plus_matches & di_plus_still_above & adx_shortlist_pass].copy()
            else:
                stock_stats = stock_stats.iloc[0:0].copy()
        if require_trend_filter and "trend_filter_pass" in stock_stats.columns:
            stock_stats = stock_stats[_truthy_series(stock_stats["trend_filter_pass"])].copy()
        if require_volume_filter and "volume_filter_pass" in stock_stats.columns:
            stock_stats = stock_stats[_truthy_series(stock_stats["volume_filter_pass"])].copy()
        if require_breakout_filter and "breakout_filter_pass" in stock_stats.columns:
            stock_stats = stock_stats[_truthy_series(stock_stats["breakout_filter_pass"])].copy()
        if require_rs_filter and "rs_filter_pass" in stock_stats.columns:
            stock_stats = stock_stats[_truthy_series(stock_stats["rs_filter_pass"])].copy()
        if require_threshold_cross_filter and "di_plus_cross_over_threshold_recent" in stock_stats.columns:
            threshold_matches = _truthy_series(stock_stats["di_plus_cross_over_threshold_recent"])
            di_plus_still_above = _truthy_series(stock_stats["di_plus_above_di_minus"]) if "di_plus_above_di_minus" in stock_stats.columns else pd.Series(False, index=stock_stats.index)
            stock_stats = stock_stats[threshold_matches & di_plus_still_above].copy()
        if require_divergence_filter and "di_plus_divergence_recent" in stock_stats.columns:
            divergence_recent = _truthy_series(stock_stats["di_plus_divergence_recent"])
            divergence_expanding = _truthy_series(stock_stats["di_plus_divergence_expanding_latest"]) if "di_plus_divergence_expanding_latest" in stock_stats.columns else pd.Series(False, index=stock_stats.index)
            di_plus_still_above = _truthy_series(stock_stats["di_plus_above_di_minus"]) if "di_plus_above_di_minus" in stock_stats.columns else pd.Series(False, index=stock_stats.index)
            stock_stats = stock_stats[divergence_recent & divergence_expanding & di_plus_still_above].copy()
        if require_obv_cross_filter and "obv_cross_sma13_recent" in stock_stats.columns:
            obv_cross_matches = _truthy_series(stock_stats["obv_cross_sma13_recent"])
            obv_still_above = _truthy_series(stock_stats["obv_above_sma13"]) if "obv_above_sma13" in stock_stats.columns else pd.Series(False, index=stock_stats.index)
            stock_stats = stock_stats[obv_cross_matches & obv_still_above].copy()
        if require_atr_lower1_filter and "atr_lower1_distance_pct" in stock_stats.columns:
            atr_distance = pd.to_numeric(stock_stats["atr_lower1_distance_pct"], errors="coerce")
            stock_stats = stock_stats[
                atr_distance.notna()
                & (atr_distance <= float(atr_lower1_proximity_pct))
            ].copy()
        if require_support_filter and "support_distance_from_level_pct" in stock_stats.columns:
            support_distance = pd.to_numeric(stock_stats["support_distance_from_level_pct"], errors="coerce")
            stock_stats = stock_stats[
                support_distance.notna()
                & (support_distance >= float(min_support_distance_pct))
                & (support_distance <= float(max_support_distance_pct))
            ].copy()
        if min_current_price is not None and "latest_close" in stock_stats.columns:
            latest_close = pd.to_numeric(stock_stats["latest_close"], errors="coerce")
            stock_stats = stock_stats[latest_close >= float(min_current_price)].copy()
        if "symbol" in stock_stats.columns:
            stock_stats["symbol_display"] = stock_stats["symbol"].map(_display_symbol)
        if not stock_stats.empty:
            crossover_signal = _truthy_series(stock_stats["di_plus_cross_above_di_minus_recent"]) if "di_plus_cross_above_di_minus_recent" in stock_stats.columns else pd.Series(False, index=stock_stats.index)
            threshold_divergence_signal = _truthy_series(stock_stats["di_plus_pre_cross_threshold_divergence_recent"]) if "di_plus_pre_cross_threshold_divergence_recent" in stock_stats.columns else pd.Series(False, index=stock_stats.index)
            shortlist_pass = _truthy_series(stock_stats["adx_shortlist_pass"]) if "adx_shortlist_pass" in stock_stats.columns else pd.Series(False, index=stock_stats.index)
            stock_stats["di_plus_signal_text"] = np.select(
                [
                    crossover_signal & shortlist_pass,
                    crossover_signal,
                    threshold_divergence_signal,
                ],
                [
                    "Cross above DI- + shortlist",
                    "Cross above DI-",
                    "Threshold divergence",
                ],
                default="",
            )

    stock_stats, sector_summary, sector_leaders = _build_adx_di_sector_views(stock_stats)
    adx_di_sorted_stats = stock_stats.copy()
    if not adx_di_sorted_stats.empty:
        di_plus = pd.to_numeric(adx_di_sorted_stats.get("latest_di_plus"), errors="coerce")
        di_minus = pd.to_numeric(adx_di_sorted_stats.get("latest_di_minus"), errors="coerce")
        adx_di_sorted_stats["di_plus_minus_range"] = di_plus - di_minus
        adx_di_sorted_stats = adx_di_sorted_stats.sort_values(
            ["di_plus_cross_above_di_minus_recent", "latest_di_plus_cross_date", "di_plus_minus_range", "symbol"],
            ascending=[False, False, False, True],
            na_position="last",
        )

    visible_symbols = adx_di_sorted_stats.get("symbol", pd.Series(dtype=str)).astype(str).tolist() if not adx_di_sorted_stats.empty else []
    selected_chart_symbol = chart_symbol if chart_symbol and chart_symbol in visible_symbols else (visible_symbols[0] if visible_symbols else "")
    chart_symbol_option_rows = [{"value": symbol, "label": _display_symbol(symbol)} for symbol in visible_symbols]

    chart_html = ""
    chart_message = "Run the ADX / DI scan and choose a visible stock to inspect the chart."
    selected_chart_name = ""
    selected_chart_symbol_display = _display_symbol(selected_chart_symbol)
    if selected_chart_symbol:
        daily = storage.load_candles("NSE", selected_chart_symbol, "1D")
        if daily.empty:
            chart_message = f"No local daily candles found for NSE:{selected_chart_symbol}."
        else:
            chart_html = build_adx_di_chart(daily, "NSE", selected_chart_symbol, length=length, threshold=threshold)
            selected_rows = all_stock_stats[all_stock_stats["symbol"].astype(str).str.upper() == selected_chart_symbol.upper()]
            if not selected_rows.empty:
                selected_chart_name = str(selected_rows.iloc[0].get("name", ""))
            chart_message = ""

    sector_chart_html = build_sector_mix_pie_chart(sector_summary, title="Sector / Industry Mix")
    adx_series = (
        pd.to_numeric(stock_stats["latest_adx"], errors="coerce").dropna()
        if "latest_adx" in stock_stats.columns
        else pd.Series(dtype=float)
    )
    return templates.TemplateResponse(
        "adx_di.html",
        {
            "request": request,
            "app_name": config.get("app", {}).get("name", "Investment Screener"),
            "dashboard_token": request.query_params.get("token", ""),
            "selected_sensitivity": selected_sensitivity,
            "default_sensitivity": base_sensitivity,
            "summary": latest.summary,
            "stock_stats": _records(stock_stats),
            "stock_stats_count": len(stock_stats),
            "stock_symbols_csv": _adx_di_sorted_display_symbols(stock_stats),
            "sector_summary": _records(sector_summary),
            "sector_summary_count": len(sector_summary),
            "sector_leaders": _records(sector_leaders),
            "sector_chart_html": sector_chart_html,
            "stock_search": stock_search,
            "matches_only": matches_only,
            "length": length,
            "threshold": threshold,
            "cross_lookback_bars": cross_lookback_bars,
            "trend_fast_ma_length": trend_fast_ma_length,
            "trend_slow_ma_length": trend_slow_ma_length,
            "volume_avg_lookback": volume_avg_lookback,
            "min_volume_ratio": min_volume_ratio,
            "breakout_lookback_days": breakout_lookback_days,
            "rs_lookback_days": rs_lookback_days,
            "min_rs_spread_pct": min_rs_spread_pct,
            "atr_channel_ma_length": atr_channel_ma_length,
            "atr_channel_atr_length": atr_channel_atr_length,
            "atr_channel_ma_type": atr_channel_ma_type,
            "atr_lower1_proximity_pct": atr_lower1_proximity_value,
            "min_current_price": min_current_price_value,
            "min_support_distance_pct": min_support_distance_value,
            "max_support_distance_pct": max_support_distance_value,
            "require_trend_filter": require_trend_filter,
            "require_volume_filter": require_volume_filter,
            "require_breakout_filter": require_breakout_filter,
            "require_rs_filter": require_rs_filter,
            "require_threshold_cross_filter": require_threshold_cross_filter,
            "require_atr_lower1_filter": require_atr_lower1_filter,
            "require_obv_cross_filter": require_obv_cross_filter,
            "require_divergence_filter": require_divergence_filter,
            "require_support_filter": require_support_filter,
            "study_job": request.query_params.get("study_job", ""),
            "study_ran": request.query_params.get("study_ran", ""),
            "study_error": request.query_params.get("study_error", ""),
            "chart_html": chart_html,
            "chart_message": chart_message,
            "chart_symbol": selected_chart_symbol,
            "chart_symbol_display": selected_chart_symbol_display,
            "chart_name": selected_chart_name,
            "chart_symbol_options": chart_symbol_option_rows,
            "avg_latest_adx": round(float(adx_series.mean()), 2) if not adx_series.empty else None,
            "max_latest_adx": round(float(adx_series.max()), 2) if not adx_series.empty else None,
            **common_filter_context,
            "show_shared_filter_form": False,
            "show_shared_filter_status": False,
        },
    )


@app.post("/adx-di/run")
async def run_adx_di_from_dashboard(request: Request, background_tasks: BackgroundTasks) -> RedirectResponse:
    config = load_config()
    data_root = get_data_root(config)
    form = await request.form()

    dashboard_token = str(form.get("token", "")).strip()
    sensitivity_text = str(form.get("sensitivity", "")).strip()
    length = max(int(str(form.get("length", "14")).strip() or "14"), 2)
    threshold = max(float(str(form.get("threshold", "20")).strip() or "20"), 0.0)
    cross_lookback_bars = max(int(str(form.get("cross_lookback_bars", "3")).strip() or "3"), 1)
    trend_fast_ma_length = max(int(str(form.get("trend_fast_ma_length", "50")).strip() or "50"), 2)
    trend_slow_ma_length = max(int(str(form.get("trend_slow_ma_length", "200")).strip() or "200"), trend_fast_ma_length + 1)
    volume_avg_lookback = max(int(str(form.get("volume_avg_lookback", "20")).strip() or "20"), 2)
    min_volume_ratio = max(float(str(form.get("min_volume_ratio", "1.5")).strip() or "1.5"), 0.0)
    breakout_lookback_days = max(int(str(form.get("breakout_lookback_days", "20")).strip() or "20"), 2)
    rs_lookback_days = max(int(str(form.get("rs_lookback_days", "20")).strip() or "20"), 1)
    min_rs_spread_pct = float(str(form.get("min_rs_spread_pct", "0")).strip() or "0")
    atr_channel_ma_length = max(int(str(form.get("atr_channel_ma_length", "20")).strip() or "20"), 1)
    atr_channel_atr_length = max(int(str(form.get("atr_channel_atr_length", "14")).strip() or "14"), 1)
    atr_channel_ma_type = str(form.get("atr_channel_ma_type", "EMA")).strip().upper() or "EMA"
    if atr_channel_ma_type not in {"EMA", "SMA"}:
        atr_channel_ma_type = "EMA"
    atr_lower1_proximity_pct = max(float(str(form.get("atr_lower1_proximity_pct", "2")).strip() or "2"), 0.0)
    require_trend_filter = _truthy_param(form.getlist("require_trend_filter"), default=False)
    require_volume_filter = _truthy_param(form.getlist("require_volume_filter"), default=False)
    require_breakout_filter = _truthy_param(form.getlist("require_breakout_filter"), default=False)
    require_rs_filter = _truthy_param(form.getlist("require_rs_filter"), default=False)
    require_threshold_cross_filter = _truthy_param(form.getlist("require_threshold_cross_filter"), default=False)
    require_atr_lower1_filter = _truthy_param(form.getlist("require_atr_lower1_filter"), default=False)
    require_obv_cross_filter = _truthy_param(form.getlist("require_obv_cross_filter"), default=False)
    require_divergence_filter = _truthy_param(form.getlist("require_divergence_filter"), default=False)
    require_support_filter = _truthy_param(form.getlist("require_support_filter"), default=False)
    params = [
        f"length={length}",
        f"threshold={quote(str(threshold))}",
        f"cross_lookback_bars={cross_lookback_bars}",
        f"trend_fast_ma_length={trend_fast_ma_length}",
        f"trend_slow_ma_length={trend_slow_ma_length}",
        f"volume_avg_lookback={volume_avg_lookback}",
        f"min_volume_ratio={quote(str(min_volume_ratio))}",
        f"breakout_lookback_days={breakout_lookback_days}",
        f"rs_lookback_days={rs_lookback_days}",
        f"min_rs_spread_pct={quote(str(min_rs_spread_pct))}",
        f"atr_channel_ma_length={atr_channel_ma_length}",
        f"atr_channel_atr_length={atr_channel_atr_length}",
        f"atr_channel_ma_type={quote(atr_channel_ma_type)}",
        f"atr_lower1_proximity_pct={quote(str(atr_lower1_proximity_pct))}",
    ]
    if require_trend_filter:
        params.append("require_trend_filter=1")
    if require_volume_filter:
        params.append("require_volume_filter=1")
    if require_breakout_filter:
        params.append("require_breakout_filter=1")
    if require_rs_filter:
        params.append("require_rs_filter=1")
    if require_threshold_cross_filter:
        params.append("require_threshold_cross_filter=1")
    if require_divergence_filter:
        params.append("require_divergence_filter=1")
    if require_atr_lower1_filter:
        params.append("require_atr_lower1_filter=1")
    if require_obv_cross_filter:
        params.append("require_obv_cross_filter=1")
    if require_support_filter:
        params.append("require_support_filter=1")
    if dashboard_token:
        params.append(f"token={quote(dashboard_token)}")
    if sensitivity_text:
        params.append(f"sensitivity={quote(sensitivity_text)}")
    query_suffix = "&" + "&".join(params)

    try:
        job_id = uuid4().hex
        _set_scan_job(job_id, status="queued", phase="Queued", completed=0, total=0, percent=0)
        background_tasks.add_task(
            _run_adx_di_job,
            job_id,
            data_root,
            query_suffix,
            length,
            threshold,
            cross_lookback_bars,
            trend_fast_ma_length,
            trend_slow_ma_length,
            volume_avg_lookback,
            min_volume_ratio,
            breakout_lookback_days,
            rs_lookback_days,
            min_rs_spread_pct,
            atr_channel_ma_length,
            atr_channel_atr_length,
            atr_channel_ma_type,
            atr_lower1_proximity_pct,
        )
        redirect_url = f"/adx-di?study_job={job_id}{query_suffix}"
    except Exception as exc:
        redirect_url = f"/adx-di?study_error={quote(str(exc)[:500])}{query_suffix}"
    return RedirectResponse(redirect_url, status_code=303)


@app.get("/qm-quality", response_class=HTMLResponse)
def qm_quality_page(request: Request) -> HTMLResponse:
    if not _is_allowed(request):
        return templates.TemplateResponse(
            "locked.html",
            {"request": request, "app_name": "Investment Screener"},
            status_code=401,
        )

    config = load_config()
    _, base_sensitivity, selected_sensitivity = _apply_request_sensitivity(config, request)
    data_root = get_data_root(config)
    common_filter_context = _common_filter_context(request, selected_sensitivity, config, data_root)
    latest = load_qm_quality_outputs(_qm_quality_dir(data_root))
    buy_start_date = request.query_params.get("buy_start_date", QM_QUALITY_DEFAULT_START_DATE).strip() or QM_QUALITY_DEFAULT_START_DATE
    buy_end_date = request.query_params.get("buy_end_date", QM_QUALITY_DEFAULT_END_DATE).strip() or QM_QUALITY_DEFAULT_END_DATE
    run_mode = request.query_params.get("run_mode", "date_range").strip() or "date_range"
    price_as_of_date = request.query_params.get("price_as_of_date", "").strip() or str(latest.summary.get("price_as_of_date", "") or "")
    stock_search = request.query_params.get("stock_search", "").strip()

    stock_stats = latest.stock_stats.copy()
    if not stock_stats.empty:
        for column in ("first_april_buy_date", "latest_april_buy_date", "latest_close_date", "as_of_date"):
            if column in stock_stats.columns:
                stock_stats[column] = pd.to_datetime(stock_stats[column], errors="coerce").dt.strftime("%Y-%m-%d")
        stock_stats = _apply_stock_search(stock_stats, stock_search)
    elite = stock_stats.copy()
    if not elite.empty and "qm_quality_bucket" in elite.columns:
        elite = elite[elite["qm_quality_bucket"].astype(str).isin(["Elite", "High"])].copy()
        if "qm_composite_score" in elite.columns:
            elite["qm_composite_score"] = pd.to_numeric(elite["qm_composite_score"], errors="coerce")
            elite = elite.sort_values(["qm_composite_score", "current_gain_pct", "symbol"], ascending=[False, False, True], na_position="last")

    return templates.TemplateResponse(
        "qm_quality.html",
        {
            "request": request,
            "app_name": config.get("app", {}).get("name", "Investment Screener"),
            "dashboard_token": request.query_params.get("token", ""),
            "selected_sensitivity": selected_sensitivity,
            "default_sensitivity": base_sensitivity,
            "summary": latest.summary,
            "stock_stats": _records(stock_stats),
            "stock_stats_count": len(stock_stats),
            "elite_stats": _records(elite),
            "elite_stats_count": len(elite),
            "buy_start_date": buy_start_date,
            "buy_end_date": buy_end_date,
            "run_mode": run_mode,
            "price_as_of_date": price_as_of_date,
            "stock_search": stock_search,
            "study_job": request.query_params.get("study_job", ""),
            "study_ran": request.query_params.get("study_ran", ""),
            "study_error": request.query_params.get("study_error", ""),
            **common_filter_context,
            "show_shared_filter_form": False,
            "show_shared_filter_status": False,
        },
    )


@app.post("/qm-quality/run")
async def run_qm_quality_from_dashboard(request: Request, background_tasks: BackgroundTasks) -> RedirectResponse:
    config = load_config()
    data_root = get_data_root(config)
    form = await request.form()

    dashboard_token = str(form.get("token", "")).strip()
    sensitivity_text = str(form.get("sensitivity", "")).strip()
    buy_start_date = str(form.get("buy_start_date", QM_QUALITY_DEFAULT_START_DATE)).strip() or QM_QUALITY_DEFAULT_START_DATE
    buy_end_date = str(form.get("buy_end_date", QM_QUALITY_DEFAULT_END_DATE)).strip() or QM_QUALITY_DEFAULT_END_DATE
    price_as_of_date = str(form.get("price_as_of_date", "")).strip()
    run_mode = str(form.get("run_mode", "date_range")).strip() or "date_range"
    params = [f"buy_start_date={quote(buy_start_date)}", f"buy_end_date={quote(buy_end_date)}", f"run_mode={quote(run_mode)}"]
    if price_as_of_date:
        params.append(f"price_as_of_date={quote(price_as_of_date)}")
    if dashboard_token:
        params.append(f"token={quote(dashboard_token)}")
    if sensitivity_text:
        params.append(f"sensitivity={quote(sensitivity_text)}")
    query_suffix = "&" + "&".join(params)

    try:
        job_id = uuid4().hex
        _set_scan_job(job_id, status="queued", phase="Queued", completed=0, total=0, percent=0)
        background_tasks.add_task(_run_qm_quality_job, job_id, config, data_root, query_suffix, buy_start_date, buy_end_date, run_mode, price_as_of_date)
        redirect_url = f"/qm-quality?study_job={job_id}{query_suffix}"
    except Exception as exc:
        redirect_url = _qm_quality_error_url(exc, query_suffix)
    return RedirectResponse(redirect_url, status_code=303)


@app.get("/minervini-sheet", response_class=HTMLResponse)
def minervini_sheet_page(request: Request) -> HTMLResponse:
    if not _is_allowed(request):
        return templates.TemplateResponse(
            "locked.html",
            {"request": request, "app_name": "Investment Screener"},
            status_code=401,
        )

    config = load_config()
    _, base_sensitivity, selected_sensitivity = _apply_request_sensitivity(config, request)
    data_root = get_data_root(config)
    common_filter_context = _common_filter_context(request, selected_sensitivity, config, data_root)
    google_settings = load_google_sheets_settings(data_root)
    oauth_status = google_oauth_status(data_root)
    oauth_client = load_google_oauth_client(data_root)
    latest = load_minervini_sheet_sync_outputs(_minervini_sheet_sync_dir(data_root))
    dashboard_token = request.query_params.get("token", "")
    stock_search = request.query_params.get("stock_search", "").strip()
    row_updates = latest.row_updates.copy()
    if not row_updates.empty and stock_search:
        search = stock_search.upper()
        mask = (
            row_updates.get("input_symbol", pd.Series("", index=row_updates.index)).astype(str).str.upper().str.contains(search, na=False)
            | row_updates.get("resolved_symbol", pd.Series("", index=row_updates.index)).astype(str).str.upper().str.contains(search, na=False)
        )
        row_updates = row_updates[mask].copy()

    redirect_uri = _google_oauth_redirect_uri(request)
    return_to = "/minervini-sheet"
    token_suffix = ""
    if dashboard_token:
        return_to = f"{return_to}?token={quote(str(dashboard_token))}"
        token_suffix = f"&token={quote(str(dashboard_token))}"
    if selected_sensitivity:
        separator = "&" if "?" in return_to else "?"
        return_to = f"{return_to}{separator}sensitivity={quote(str(selected_sensitivity))}"
        token_suffix += f"&sensitivity={quote(str(selected_sensitivity))}"
    login_url = f"/auth/google-sheets/login?return_to={quote(return_to, safe='')}{token_suffix}"
    login_error = ""
    if not oauth_client.configured:
        login_error = "Google OAuth client is not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env."

    return templates.TemplateResponse(
        "minervini_sheet.html",
        {
            "request": request,
            "app_name": config.get("app", {}).get("name", "Investment Screener"),
            "dashboard_token": dashboard_token,
            "selected_sensitivity": selected_sensitivity,
            "default_sensitivity": base_sensitivity,
            "google_sheets_settings": google_settings,
            "google_oauth_status": oauth_status,
            "google_oauth_client_configured": oauth_client.configured,
            "google_redirect_uri": redirect_uri,
            "google_login_url": login_url,
            "google_error": request.query_params.get("google_error", "").strip() or login_error,
            "google_saved": request.query_params.get("google_saved", "").strip(),
            "google_login_done": request.query_params.get("google_login_done", "").strip(),
            "stock_search": stock_search,
            "summary": latest.summary,
            "row_updates": _records(row_updates.head(250)),
            "row_updates_count": len(row_updates),
            "study_job": request.query_params.get("study_job", ""),
            "study_ran": request.query_params.get("study_ran", ""),
            "study_error": request.query_params.get("study_error", ""),
            **common_filter_context,
            "show_shared_filter_form": False,
            "show_shared_filter_status": False,
        },
    )


@app.post("/minervini-sheet/google/save")
async def save_minervini_sheet_settings(request: Request) -> RedirectResponse:
    config = load_config()
    data_root = get_data_root(config)
    form = await request.form()
    spreadsheet_id = str(form.get("spreadsheet_id", "")).strip()
    worksheet_title = str(form.get("worksheet_title", MINERVINI_SHEET_DEFAULT_WORKSHEET_TITLE)).strip() or MINERVINI_SHEET_DEFAULT_WORKSHEET_TITLE
    dashboard_token = str(form.get("token", "")).strip()
    sensitivity_text = str(form.get("sensitivity", "")).strip()

    params = ["google_saved=1"]
    if dashboard_token:
        params.append(f"token={quote(dashboard_token)}")
    if sensitivity_text:
        params.append(f"sensitivity={quote(sensitivity_text)}")
    query_suffix = "?" + "&".join(params)

    try:
        save_google_sheet_target(data_root, spreadsheet_id, worksheet_title)
        redirect_url = f"/minervini-sheet{query_suffix}"
    except Exception as exc:
        redirect_url = f"/minervini-sheet?google_error={quote(str(exc)[:500])}"
        if dashboard_token:
            redirect_url += f"&token={quote(dashboard_token)}"
        if sensitivity_text:
            redirect_url += f"&sensitivity={quote(sensitivity_text)}"
    return RedirectResponse(redirect_url, status_code=303)


@app.post("/minervini-sheet/run")
async def run_minervini_sheet_from_dashboard(request: Request, background_tasks: BackgroundTasks) -> RedirectResponse:
    config = load_config()
    data_root = get_data_root(config)
    form = await request.form()
    dashboard_token = str(form.get("token", "")).strip()
    sensitivity_text = str(form.get("sensitivity", "")).strip()
    params = []
    if dashboard_token:
        params.append(f"token={quote(dashboard_token)}")
    if sensitivity_text:
        params.append(f"sensitivity={quote(sensitivity_text)}")
    query_suffix = "&" + "&".join(params) if params else ""

    try:
        job_id = uuid4().hex
        _set_scan_job(job_id, status="queued", phase="Queued", completed=0, total=0, percent=0)
        background_tasks.add_task(_run_minervini_sheet_job, job_id, data_root, query_suffix)
        redirect_url = f"/minervini-sheet?study_job={job_id}{query_suffix}"
    except Exception as exc:
        redirect_url = f"/minervini-sheet?study_error={quote(str(exc)[:500])}{query_suffix}"
    return RedirectResponse(redirect_url, status_code=303)


@app.get("/weekly-buy-tracker", response_class=HTMLResponse)
def weekly_buy_tracker_page(request: Request) -> HTMLResponse:
    return _temporarily_removed_response(request, "Buy Tracker")


@app.post("/weekly-buy-tracker/run")
async def run_weekly_buy_tracker_from_dashboard(request: Request, background_tasks: BackgroundTasks) -> RedirectResponse:
    raise HTTPException(status_code=404, detail="Buy Tracker is temporarily removed from the workspace for now.")


@app.post("/weekly-buy-tracker/google/save")
async def save_weekly_buy_tracker_google_settings(request: Request) -> RedirectResponse:
    raise HTTPException(status_code=404, detail="Buy Tracker is temporarily removed from the workspace for now.")


@app.post("/weekly-buy-tracker/google/oauth-client/save")
async def save_weekly_buy_tracker_google_oauth_client(request: Request) -> RedirectResponse:
    raise HTTPException(status_code=404, detail="Buy Tracker is temporarily removed from the workspace for now.")


@app.get("/auth/google-sheets/login")
def google_sheets_login(request: Request) -> RedirectResponse:
    if not _is_allowed(request):
        raise HTTPException(status_code=401, detail="Dashboard access is required.")
    config = load_config()
    data_root = get_data_root(config)
    redirect_uri = _google_oauth_redirect_uri(request)
    return_to = str(request.query_params.get("return_to", "/minervini-sheet")).strip() or "/minervini-sheet"
    try:
        login_url = build_google_oauth_login_url(data_root, redirect_uri, return_to)
        return RedirectResponse(login_url, status_code=303)
    except Exception as exc:
        return RedirectResponse(_append_query_param(return_to, f"google_error={quote(str(exc)[:500])}"), status_code=303)


@app.get("/auth/google-sheets/callback", response_class=HTMLResponse, name="google_sheets_callback")
def google_sheets_callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
):
    config = load_config()
    data_root = get_data_root(config)
    redirect_uri = _google_oauth_redirect_uri(request)
    return_to = "/minervini-sheet"
    if error:
        return RedirectResponse(_append_query_param(return_to, f"google_error={quote(str(error)[:500])}"), status_code=303)
    if not code or not state:
        return RedirectResponse(_append_query_param(return_to, "google_error=Missing+Google+OAuth+response"), status_code=303)
    try:
        outcome = exchange_google_oauth_code(data_root, code=code, state=state, redirect_uri=redirect_uri)
        return_to = str(outcome.get("return_to", return_to)).strip() or return_to
        return RedirectResponse(_append_query_param(return_to, "google_login_done=1"), status_code=303)
    except Exception as exc:
        return RedirectResponse(_append_query_param(return_to, f"google_error={quote(str(exc)[:500])}"), status_code=303)


@app.post("/weekly-buy-tracker/google/export")
async def export_weekly_buy_tracker_google_sheet(request: Request) -> RedirectResponse:
    raise HTTPException(status_code=404, detail="Buy Tracker is temporarily removed from the workspace for now.")


@app.get("/rotation-study", response_class=HTMLResponse)
def rotation_study_page(request: Request) -> HTMLResponse:
    return _temporarily_removed_response(request, "Rotation Study")


@app.post("/rotation-study/run")
async def run_rotation_study_from_dashboard(request: Request, background_tasks: BackgroundTasks) -> RedirectResponse:
    raise HTTPException(status_code=404, detail="Rotation Study is temporarily removed from the workspace for now.")


@app.get("/signal-outcome-study", response_class=HTMLResponse)
def signal_outcome_study_page(request: Request) -> HTMLResponse:
    return _temporarily_removed_response(request, "Signal Outcome")


@app.post("/signal-outcome-study/run")
async def run_signal_outcome_study_from_dashboard(request: Request, background_tasks: BackgroundTasks) -> RedirectResponse:
    raise HTTPException(status_code=404, detail="Signal Outcome is temporarily removed from the workspace for now.")


@app.get("/signal-outcome-study/report")
def download_signal_outcome_study_report() -> FileResponse:
    raise HTTPException(status_code=404, detail="Signal Outcome is temporarily removed from the workspace for now.")


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    if not _is_allowed(request):
        return templates.TemplateResponse(
            "locked.html",
            {"request": request},
            status_code=401,
        )

    config = load_config()
    config, base_sensitivity, selected_sensitivity = _apply_request_sensitivity(config, request)
    data_root = get_data_root(config)
    common_filter_context = _common_filter_context(request, selected_sensitivity, config, data_root)
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
    min_cmp = _request_float(request, "min_cmp")
    max_cmp = _request_float(request, "max_cmp")
    require_volume_confirmation = _request_bool(request, "require_volume_confirmation")
    require_trend_confirmation = _request_bool(request, "require_trend_confirmation")
    require_obv_confirmation = _request_bool(request, "require_obv_confirmation")
    selected_return_metric = request.query_params.get("return_metric", "median_3").strip() or "median_3"
    if selected_return_metric not in {"last_1", "median_3"}:
        selected_return_metric = "median_3"
    min_pair_return = _request_float(request, "min_pair_return_pct")
    require_htf_alignment = _request_bool(request, "require_htf_alignment")
    min_breakout_volume_ratio_text = request.query_params.get("min_breakout_volume_ratio", "").strip()
    min_breakout_volume_ratio = _optional_float(min_breakout_volume_ratio_text)
    require_relative_strength = _request_bool(request, "require_relative_strength")
    min_relative_strength_pct_text = request.query_params.get("min_relative_strength_pct", "").strip()
    min_relative_strength_pct = _optional_float(min_relative_strength_pct_text)
    max_distance_from_demand_pct_text = request.query_params.get("max_distance_from_demand_pct", "").strip()
    max_distance_from_demand_pct = _optional_float(max_distance_from_demand_pct_text)
    min_risk_reward_ratio_text = request.query_params.get("min_risk_reward_ratio", "").strip()
    min_risk_reward_ratio = _optional_float(min_risk_reward_ratio_text)
    filter_link_suffix = _dashboard_link_suffix(request)
    active_filter_parts = []
    if stock_search:
        active_filter_parts.append(f"Search: {stock_search}")
    if selected_sensitivity != base_sensitivity:
        active_filter_parts.append(f"Sensitivity: {selected_sensitivity}")
    if selected_market_cap_bucket:
        active_filter_parts.append(selected_market_cap_bucket)
    if min_market_cap is not None:
        active_filter_parts.append(f"Min market cap: {request.query_params.get('min_market_cap_cr')} Cr")
    if max_market_cap is not None:
        active_filter_parts.append(f"Max market cap: {request.query_params.get('max_market_cap_cr')} Cr")
    if min_cmp is not None:
        active_filter_parts.append(f"Min CMP: ₹{request.query_params.get('min_cmp')}")
    if max_cmp is not None:
        active_filter_parts.append(f"Max CMP: ₹{request.query_params.get('max_cmp')}")
    if require_volume_confirmation:
        active_filter_parts.append("Volume confirmed")
    if require_trend_confirmation:
        active_filter_parts.append("Daily EMA stack confirmed")
    if require_obv_confirmation:
        active_filter_parts.append("OBV rising over last 20 days")
    if min_pair_return is not None:
        metric_label = "last pair return" if selected_return_metric == "last_1" else "median last 3 pair return"
        active_filter_parts.append(f"{metric_label} >= {request.query_params.get('min_pair_return_pct')}%")
    if require_htf_alignment:
        active_filter_parts.append("Monthly structure aligned")
    if min_breakout_volume_ratio is not None:
        active_filter_parts.append(f"Breakout volume >= {min_breakout_volume_ratio_text}x")
    if require_relative_strength or min_relative_strength_pct is not None:
        threshold = min_relative_strength_pct_text or "0"
        active_filter_parts.append(f"Relative strength vs benchmark >= {threshold}%")
    if max_distance_from_demand_pct is not None:
        active_filter_parts.append(f"Distance from demand <= {max_distance_from_demand_pct_text}%")
    if min_risk_reward_ratio is not None:
        active_filter_parts.append(f"Risk-reward >= {min_risk_reward_ratio_text}")

    filtered = _enrich_with_symbol_metadata(filtered, metadata, "symbol")
    raw = _enrich_with_symbol_metadata(raw, metadata, "symbol")
    scan_details = _enrich_with_symbol_metadata(scan_details, metadata, "symbol")
    filtered = _enrich_with_latest_daily_close(filtered, scan_details, storage)

    filtered = _apply_market_cap_filters(filtered, min_market_cap, max_market_cap, selected_market_cap_bucket)
    raw = _apply_market_cap_filters(raw, min_market_cap, max_market_cap, selected_market_cap_bucket)
    scan_details = _apply_market_cap_filters(scan_details, min_market_cap, max_market_cap, selected_market_cap_bucket)

    filtered = _refresh_live_cmp(filtered, data_root)
    filtered = _apply_cmp_filters(filtered, min_cmp, max_cmp, "latest_close")

    filtered = _apply_stock_search(filtered, stock_search)
    raw = _apply_stock_search(raw, stock_search)
    scan_details = _apply_stock_search(scan_details, stock_search)

    filtered = enrich_weekly_signal_shortlist_frame(filtered, storage, config)

    signal_quality_warning = _signal_quality_filter_warning(
        filtered,
        require_volume_confirmation,
        require_trend_confirmation,
        require_obv_confirmation,
        min_pair_return,
    )
    filtered = _apply_signal_quality_filters(
        filtered,
        require_volume_confirmation,
        require_trend_confirmation,
        require_obv_confirmation,
        selected_return_metric,
        min_pair_return,
    )
    shortlist_warning = _weekly_shortlist_filter_warning(
        filtered,
        require_htf_alignment,
        min_breakout_volume_ratio,
        require_relative_strength,
        min_relative_strength_pct,
        max_distance_from_demand_pct,
        min_risk_reward_ratio,
    )
    filtered = _apply_weekly_shortlist_filters(
        filtered,
        require_htf_alignment,
        min_breakout_volume_ratio,
        require_relative_strength,
        min_relative_strength_pct,
        max_distance_from_demand_pct,
        min_risk_reward_ratio,
    )
    large_deals = _load_big_bull_deals(data_root)
    filtered = _apply_large_deal_markers(filtered, large_deals)

    raw_symbol_column = _symbol_column(raw)
    filtered_symbol_column = _symbol_column(filtered)
    if not raw.empty and raw_symbol_column and filtered_symbol_column:
        filtered_symbol_pairs = filtered[["exchange", filtered_symbol_column]].copy()
        filtered_symbol_pairs = filtered_symbol_pairs.dropna(subset=["exchange", filtered_symbol_column])
        if filtered_symbol_pairs.empty:
            raw = raw.iloc[0:0].copy()
        else:
            filtered_symbol_pairs["exchange"] = filtered_symbol_pairs["exchange"].astype(str).str.upper()
            filtered_symbol_pairs[filtered_symbol_column] = filtered_symbol_pairs[filtered_symbol_column].astype(str).str.upper()
            raw = raw.copy()
            raw["exchange"] = raw["exchange"].astype(str).str.upper()
            raw[raw_symbol_column] = raw[raw_symbol_column].astype(str).str.upper()
            raw = raw.merge(
                filtered_symbol_pairs.rename(columns={filtered_symbol_column: raw_symbol_column}).drop_duplicates(),
                on=["exchange", raw_symbol_column],
                how="inner",
            )

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
        if "shortlist_score" in filtered_symbols.columns:
            filtered_symbols["shortlist_score"] = pd.to_numeric(filtered_symbols["shortlist_score"], errors="coerce")
            sort_columns.append("shortlist_score")
            sort_ascending.append(False)
        if "relative_strength_12w_pct" in filtered_symbols.columns:
            filtered_symbols["relative_strength_12w_pct"] = pd.to_numeric(filtered_symbols["relative_strength_12w_pct"], errors="coerce")
            sort_columns.append("relative_strength_12w_pct")
            sort_ascending.append(False)
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
    latest_daily_summary = {"signal": "NONE", "date": "", "close": ""}

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
            daily_strategy_output = run_weekly_buy_sell(daily, daily_signal_config(config))
            latest_daily_summary = latest_signal_summary(daily_strategy_output)

    raw_table = raw.copy()
    if not raw_table.empty:
        raw_table["date"] = raw_table["date"].astype(str)
        raw_table = raw_table.sort_values("date", ascending=False)
        raw_table = raw_table.head(500)

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "app_name": config.get("app", {}).get("name", "Investment Screener"),
            "filtered": _records(filtered),
            "raw_count": len(raw),
            "filtered_count": len(filtered),
            "daily_filtered": [],
            "daily_raw_count": 0,
            "daily_filtered_count": 0,
            "token_status": token_status(data_root),
            "scan_details": _records(scan_details),
            "scan_details_count": len(scan_details),
            "filtered_symbols": _records(filtered_symbols.drop(columns=["date_sort"], errors="ignore")),
            "dashboard_token": request.query_params.get("token", ""),
            "filter_link_suffix": filter_link_suffix,
            "selected_exchange": selected_exchange or "",
            "selected_symbol": selected_symbol or "",
            "stock_search": stock_search,
            "selected_sensitivity": selected_sensitivity,
            "default_sensitivity": base_sensitivity,
            "latest_summary": latest_summary,
            "latest_daily_summary": latest_daily_summary,
            "chart_html": chart_html,
            "chart_message": chart_message,
            "all_signals": _records(raw_table),
            "all_signals_preview_count": len(raw_table),
            "all_daily_signals": [],
            "selected_market_cap_bucket": selected_market_cap_bucket,
            "selected_min_market_cap": request.query_params.get("min_market_cap_cr", ""),
            "selected_max_market_cap": request.query_params.get("max_market_cap_cr", ""),
            "selected_min_cmp": request.query_params.get("min_cmp", ""),
            "selected_max_cmp": request.query_params.get("max_cmp", ""),
            "require_volume_confirmation": require_volume_confirmation,
            "require_trend_confirmation": require_trend_confirmation,
            "require_obv_confirmation": require_obv_confirmation,
            "selected_return_metric": selected_return_metric,
            "selected_min_pair_return": request.query_params.get("min_pair_return_pct", ""),
            "signal_quality_warning": signal_quality_warning,
            "shortlist_warning": shortlist_warning,
            "require_htf_alignment": require_htf_alignment,
            "selected_min_breakout_volume_ratio": min_breakout_volume_ratio_text,
            "require_relative_strength": require_relative_strength,
            "selected_min_relative_strength_pct": min_relative_strength_pct_text,
            "selected_max_distance_from_demand_pct": max_distance_from_demand_pct_text,
            "selected_min_risk_reward_ratio": min_risk_reward_ratio_text,
            "base_filter_query": _dashboard_filter_query(
                token=request.query_params.get("token", ""),
                stock_search=stock_search,
                sensitivity=str(selected_sensitivity),
                market_cap_bucket=selected_market_cap_bucket,
                min_market_cap_cr=request.query_params.get("min_market_cap_cr", ""),
                max_market_cap_cr=request.query_params.get("max_market_cap_cr", ""),
                min_cmp=request.query_params.get("min_cmp", ""),
                max_cmp=request.query_params.get("max_cmp", ""),
            ),
            "full_filter_query": _dashboard_filter_query(
                token=request.query_params.get("token", ""),
                stock_search=stock_search,
                sensitivity=str(selected_sensitivity),
                market_cap_bucket=selected_market_cap_bucket,
                min_market_cap_cr=request.query_params.get("min_market_cap_cr", ""),
                max_market_cap_cr=request.query_params.get("max_market_cap_cr", ""),
                min_cmp=request.query_params.get("min_cmp", ""),
                max_cmp=request.query_params.get("max_cmp", ""),
                require_volume_confirmation=require_volume_confirmation,
                require_trend_confirmation=require_trend_confirmation,
                require_obv_confirmation=require_obv_confirmation,
                return_metric=selected_return_metric if request.query_params.get("min_pair_return_pct", "") else "",
                min_pair_return_pct=request.query_params.get("min_pair_return_pct", ""),
                require_htf_alignment=require_htf_alignment,
                min_breakout_volume_ratio=min_breakout_volume_ratio_text,
                require_relative_strength=require_relative_strength,
                min_relative_strength_pct=min_relative_strength_pct_text,
                max_distance_from_demand_pct=max_distance_from_demand_pct_text,
                min_risk_reward_ratio=min_risk_reward_ratio_text,
            ),
            "market_cap_bounds": market_cap_bounds,
            "has_metadata": not metadata.empty,
            "scan_ran": request.query_params.get("scan_ran", ""),
            "scan_error": request.query_params.get("scan_error", ""),
            "scan_job": request.query_params.get("scan_job", ""),
            "telegram_sent": request.query_params.get("telegram_sent", ""),
            "telegram_sent_count": request.query_params.get("telegram_sent_count", ""),
            "telegram_error": request.query_params.get("telegram_error", ""),
            "symbols_scanned": request.query_params.get("symbols_scanned", ""),
            "refresh_mode": request.query_params.get("refresh_mode", ""),
            "active_filter_summary": " · ".join(active_filter_parts),
            **common_filter_context,
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
    sensitivity_text = str(form.get("sensitivity", "")).strip()
    market_cap_bucket = str(form.get("market_cap_bucket", "")).strip()
    min_market_cap_text = str(form.get("min_market_cap_cr", "")).strip()
    max_market_cap_text = str(form.get("max_market_cap_cr", "")).strip()
    min_cmp_text = str(form.get("min_cmp", "")).strip()
    max_cmp_text = str(form.get("max_cmp", "")).strip()
    require_volume_confirmation = str(form.get("require_volume_confirmation", "")).strip().lower() in {"1", "true", "on", "yes"}
    require_trend_confirmation = str(form.get("require_trend_confirmation", "")).strip().lower() in {"1", "true", "on", "yes"}
    require_obv_confirmation = str(form.get("require_obv_confirmation", "")).strip().lower() in {"1", "true", "on", "yes"}
    return_metric = str(form.get("return_metric", "median_3")).strip() or "median_3"
    min_pair_return_text = str(form.get("min_pair_return_pct", "")).strip()
    require_htf_alignment = str(form.get("require_htf_alignment", "")).strip().lower() in {"1", "true", "on", "yes"}
    min_breakout_volume_ratio_text = str(form.get("min_breakout_volume_ratio", "")).strip()
    require_relative_strength = str(form.get("require_relative_strength", "")).strip().lower() in {"1", "true", "on", "yes"}
    min_relative_strength_pct_text = str(form.get("min_relative_strength_pct", "")).strip()
    max_distance_from_demand_pct_text = str(form.get("max_distance_from_demand_pct", "")).strip()
    min_risk_reward_ratio_text = str(form.get("min_risk_reward_ratio", "")).strip()
    min_market_cap = _optional_float(min_market_cap_text)
    max_market_cap = _optional_float(max_market_cap_text)
    sensitivity = _parse_sensitivity_text(sensitivity_text)
    market_cap_filter_requested = bool(market_cap_bucket or min_market_cap_text or max_market_cap_text)
    refresh_data = str(form.get("refresh_data", "0")).strip().lower() in {"1", "true", "on", "yes"}

    params = []
    if dashboard_token:
        params.append(f"token={quote(dashboard_token)}")
    if stock_search:
        params.append(f"stock_search={quote(stock_search)}")
    if sensitivity_text:
        params.append(f"sensitivity={quote(sensitivity_text)}")
    if market_cap_bucket:
        params.append(f"market_cap_bucket={quote(market_cap_bucket)}")
    if min_market_cap_text:
        params.append(f"min_market_cap_cr={quote(min_market_cap_text)}")
    if max_market_cap_text:
        params.append(f"max_market_cap_cr={quote(max_market_cap_text)}")
    if min_cmp_text:
        params.append(f"min_cmp={quote(min_cmp_text)}")
    if max_cmp_text:
        params.append(f"max_cmp={quote(max_cmp_text)}")
    if require_volume_confirmation:
        params.append("require_volume_confirmation=1")
    if require_trend_confirmation:
        params.append("require_trend_confirmation=1")
    if require_obv_confirmation:
        params.append("require_obv_confirmation=1")
    if return_metric:
        params.append(f"return_metric={quote(return_metric)}")
    if min_pair_return_text:
        params.append(f"min_pair_return_pct={quote(min_pair_return_text)}")
    if require_htf_alignment:
        params.append("require_htf_alignment=1")
    if min_breakout_volume_ratio_text:
        params.append(f"min_breakout_volume_ratio={quote(min_breakout_volume_ratio_text)}")
    if require_relative_strength:
        params.append("require_relative_strength=1")
    if min_relative_strength_pct_text:
        params.append(f"min_relative_strength_pct={quote(min_relative_strength_pct_text)}")
    if max_distance_from_demand_pct_text:
        params.append(f"max_distance_from_demand_pct={quote(max_distance_from_demand_pct_text)}")
    if min_risk_reward_ratio_text:
        params.append(f"min_risk_reward_ratio={quote(min_risk_reward_ratio_text)}")
    query_suffix = ("&" + "&".join(params)) if params else ""

    try:
        if market_cap_filter_requested and not _has_market_cap_metadata(storage):
            raise RuntimeError(
                "Market-cap metadata is missing. The Stocks page is hidden for now, "
                "so load it with python scripts/import_nse_market_caps.py."
            )

        scan_config = _manual_screener_config(
            config,
            storage,
            min_market_cap,
            max_market_cap,
            market_cap_bucket,
            stock_search,
            sensitivity,
        )
        scan_config.setdefault("data", {})["skip_kite_fetch"] = not refresh_data
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
    min_cmp_text = str(form.get("min_cmp", "")).strip()
    max_cmp_text = str(form.get("max_cmp", "")).strip()
    require_volume_confirmation = str(form.get("require_volume_confirmation", "")).strip().lower() in {"1", "true", "on", "yes"}
    require_trend_confirmation = str(form.get("require_trend_confirmation", "")).strip().lower() in {"1", "true", "on", "yes"}
    require_obv_confirmation = str(form.get("require_obv_confirmation", "")).strip().lower() in {"1", "true", "on", "yes"}
    return_metric = str(form.get("return_metric", "median_3")).strip() or "median_3"
    if return_metric not in {"last_1", "median_3"}:
        return_metric = "median_3"
    min_pair_return_text = str(form.get("min_pair_return_pct", "")).strip()
    require_htf_alignment = str(form.get("require_htf_alignment", "")).strip().lower() in {"1", "true", "on", "yes"}
    min_breakout_volume_ratio_text = str(form.get("min_breakout_volume_ratio", "")).strip()
    require_relative_strength = str(form.get("require_relative_strength", "")).strip().lower() in {"1", "true", "on", "yes"}
    min_relative_strength_pct_text = str(form.get("min_relative_strength_pct", "")).strip()
    max_distance_from_demand_pct_text = str(form.get("max_distance_from_demand_pct", "")).strip()
    min_risk_reward_ratio_text = str(form.get("min_risk_reward_ratio", "")).strip()
    min_market_cap = _optional_float(min_market_cap_text)
    max_market_cap = _optional_float(max_market_cap_text)
    min_cmp = _optional_float(min_cmp_text)
    max_cmp = _optional_float(max_cmp_text)
    min_pair_return = _optional_float(min_pair_return_text)
    min_breakout_volume_ratio = _optional_float(min_breakout_volume_ratio_text)
    min_relative_strength_pct = _optional_float(min_relative_strength_pct_text)
    max_distance_from_demand_pct = _optional_float(max_distance_from_demand_pct_text)
    min_risk_reward_ratio = _optional_float(min_risk_reward_ratio_text)

    filter_query = _dashboard_filter_query(
        token=dashboard_token,
        stock_search=stock_search,
        market_cap_bucket=market_cap_bucket,
        min_market_cap_cr=min_market_cap_text,
        max_market_cap_cr=max_market_cap_text,
        min_cmp=min_cmp_text,
        max_cmp=max_cmp_text,
        require_volume_confirmation=require_volume_confirmation,
        require_trend_confirmation=require_trend_confirmation,
        require_obv_confirmation=require_obv_confirmation,
        return_metric=return_metric if min_pair_return_text else "",
        min_pair_return_pct=min_pair_return_text,
        require_htf_alignment=require_htf_alignment,
        min_breakout_volume_ratio=min_breakout_volume_ratio_text,
        require_relative_strength=require_relative_strength,
        min_relative_strength_pct=min_relative_strength_pct_text,
        max_distance_from_demand_pct=max_distance_from_demand_pct_text,
        min_risk_reward_ratio=min_risk_reward_ratio_text,
    )

    try:
        visible_buy_signals = _load_visible_buy_signals(
            config,
            storage,
            stock_search,
            min_market_cap,
            max_market_cap,
            market_cap_bucket,
            min_cmp,
            max_cmp,
            require_volume_confirmation,
            require_trend_confirmation,
            require_obv_confirmation,
            return_metric,
            min_pair_return,
            require_htf_alignment,
            min_breakout_volume_ratio,
            require_relative_strength,
            min_relative_strength_pct,
            max_distance_from_demand_pct,
            min_risk_reward_ratio,
        )
        if visible_buy_signals.empty:
            raise RuntimeError("No weekly BUY signals are available to send.")

        visible_buy_signals = _apply_large_deal_markers(
            visible_buy_signals,
            _load_big_bull_deals(data_root),
        )
        visible_buy_signals = _merge_cached_weekday_profiles(
            config,
            storage,
            data_root,
            visible_buy_signals,
            exchange_column="exchange",
            symbol_column="symbol" if "symbol" in visible_buy_signals.columns else "tradingsymbol",
        )
        filters_text = _buy_signal_filter_summary(
            stock_search,
            market_cap_bucket,
            min_market_cap_text,
            max_market_cap_text,
            min_cmp_text,
            max_cmp_text,
            require_volume_confirmation,
            require_trend_confirmation,
            require_obv_confirmation,
            return_metric,
            min_pair_return_text,
            require_htf_alignment,
            min_breakout_volume_ratio_text,
            require_relative_strength,
            min_relative_strength_pct_text,
            max_distance_from_demand_pct_text,
            min_risk_reward_ratio_text,
        )
        send_buy_signal_list_to_telegram(config, visible_buy_signals, filters_text=filters_text)
        status_query = f"telegram_sent=1&telegram_sent_count={len(visible_buy_signals)}"
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
    min_cmp_text = str(form.get("min_cmp", "")).strip()
    max_cmp_text = str(form.get("max_cmp", "")).strip()
    min_market_cap = _optional_float(min_market_cap_text)
    max_market_cap = _optional_float(max_market_cap_text)
    min_cmp = _optional_float(min_cmp_text)
    max_cmp = _optional_float(max_cmp_text)
    open_buy_regime_only = str(form.get("open_buy_regime_only", "")).strip().lower() in {"1", "true", "on", "yes"}
    dashboard_buy_only = str(form.get("dashboard_buy_only", "")).strip().lower() in {"1", "true", "on", "yes"}
    fresh_weekly_buy_only = str(form.get("fresh_weekly_buy_only", "")).strip().lower() in {"1", "true", "on", "yes"}
    fresh_daily_buy_only = str(form.get("fresh_daily_buy_only", "")).strip().lower() in {"1", "true", "on", "yes"}
    trend_only = str(form.get("trend_only", "")).strip().lower() in {"1", "true", "on", "yes"}
    require_volume_confirmation = str(form.get("require_volume_confirmation", "")).strip().lower() in {"1", "true", "on", "yes"}
    require_obv_confirmation = str(form.get("require_obv_confirmation", "")).strip().lower() in {"1", "true", "on", "yes"}
    peak_speed_bucket = str(form.get("peak_speed_bucket", "")).strip()
    technical_rating_statuses = _normalize_gtt_technical_rating_statuses(form.getlist("technical_rating_status"))
    if peak_speed_bucket not in GTT_PEAK_SPEED_BUCKETS:
        peak_speed_bucket = ""

    filter_query = _gtt_filter_query(
        token=dashboard_token,
        stock_search=stock_search,
        market_cap_bucket=market_cap_bucket,
        min_market_cap_cr=min_market_cap_text,
        max_market_cap_cr=max_market_cap_text,
        min_cmp=min_cmp_text,
        max_cmp=max_cmp_text,
        open_buy_regime_only=open_buy_regime_only,
        dashboard_buy_only=dashboard_buy_only,
        fresh_weekly_buy_only=fresh_weekly_buy_only,
        fresh_daily_buy_only=fresh_daily_buy_only,
        trend_only=trend_only,
        require_volume_confirmation=require_volume_confirmation,
        require_obv_confirmation=require_obv_confirmation,
        peak_speed_bucket=peak_speed_bucket,
        technical_rating_statuses=technical_rating_statuses,
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
            min_cmp,
            max_cmp,
            open_buy_regime_only,
            dashboard_buy_only,
            fresh_weekly_buy_only,
            fresh_daily_buy_only,
            trend_only,
            peak_speed_bucket,
            require_volume_confirmation,
            require_obv_confirmation,
            technical_rating_statuses,
        )
        if visible_gtt_stocks.empty:
            raise RuntimeError("No GTT stocks are available to send with the selected filters.")

        filters_text = _gtt_filter_summary(
            stock_search=stock_search,
            sensitivity_text="",
            market_cap_bucket=market_cap_bucket,
            min_market_cap_text=min_market_cap_text,
            max_market_cap_text=max_market_cap_text,
            min_cmp_text=min_cmp_text,
            max_cmp_text=max_cmp_text,
            open_buy_regime_only=open_buy_regime_only,
            dashboard_buy_only=dashboard_buy_only,
            fresh_weekly_buy_only=fresh_weekly_buy_only,
            fresh_daily_buy_only=fresh_daily_buy_only,
            trend_only=trend_only,
            require_volume_confirmation=require_volume_confirmation,
            require_obv_confirmation=require_obv_confirmation,
            peak_speed_bucket=peak_speed_bucket,
            technical_rating_statuses=technical_rating_statuses,
        )
        send_gtt_stock_list_to_telegram(config, visible_gtt_stocks, filters_text=filters_text)
        status_query = f"telegram_sent=1&telegram_sent_count={len(visible_gtt_stocks)}"
    except Exception as exc:
        status_query = f"telegram_error={quote(str(exc)[:500])}"

    redirect_query = "&".join([part for part in (status_query, filter_query) if part])
    return RedirectResponse(f"/gtt-gain-study?{redirect_query}", status_code=303)


@app.get("/signal-qa", response_class=HTMLResponse)
def signal_qa_page(request: Request) -> HTMLResponse:
    return _temporarily_removed_response(request, "Signal QA")


@app.post("/watchlist/add/{exchange}/{symbol}")
async def add_watchlist(request: Request, exchange: str, symbol: str) -> RedirectResponse:
    raise HTTPException(status_code=404, detail="Stocks is temporarily removed from the workspace for now.")


@app.post("/watchlist/remove/{exchange}/{symbol}")
async def remove_watchlist(request: Request, exchange: str, symbol: str) -> RedirectResponse:
    raise HTTPException(status_code=404, detail="Stocks is temporarily removed from the workspace for now.")


@app.post("/stocks/fetch")
async def fetch_stocks(request: Request) -> RedirectResponse:
    raise HTTPException(status_code=404, detail="Stocks is temporarily removed from the workspace for now.")


@app.post("/stocks/fetch-market-caps")
async def fetch_market_caps(request: Request) -> RedirectResponse:
    raise HTTPException(status_code=404, detail="Stocks is temporarily removed from the workspace for now.")


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request) -> HTMLResponse:
    if not _is_allowed(request):
        return templates.TemplateResponse(
            "locked.html",
            {"request": request, "app_name": "Investment Screener"},
            status_code=401,
        )

    config = load_config()
    _, base_sensitivity, selected_sensitivity = _apply_request_sensitivity(config, request)
    data_root = get_data_root(config)
    common_filter_context = _common_filter_context(request, selected_sensitivity, config, data_root)

    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "app_name": config.get("app", {}).get("name", "Investment Screener"),
            "token_status": token_status(data_root),
            "dashboard_token": request.query_params.get("token", ""),
            "selected_sensitivity": selected_sensitivity,
            "default_sensitivity": base_sensitivity,
            **common_filter_context,
            "show_shared_filter_form": False,
            "show_shared_filter_status": False,
        },
    )


@app.get("/stocks", response_class=HTMLResponse)
def stocks_page(request: Request) -> HTMLResponse:
    return _temporarily_removed_response(request, "Stocks")


@app.get("/big-bull-deals", response_class=HTMLResponse)
def big_bull_deals_page(request: Request) -> HTMLResponse:
    if not _is_allowed(request):
        return templates.TemplateResponse(
            "locked.html",
            {"request": request, "app_name": "Investment Screener"},
            status_code=401,
        )

    config = load_config()
    _, base_sensitivity, selected_sensitivity = _apply_request_sensitivity(config, request)
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
            "selected_sensitivity": selected_sensitivity,
            "default_sensitivity": base_sensitivity,
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
    config, base_sensitivity, selected_sensitivity = _apply_request_sensitivity(config, request)
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
            "selected_sensitivity": selected_sensitivity,
            "default_sensitivity": base_sensitivity,
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
