from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import quote
from uuid import uuid4

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from kiteconnect import KiteConnect

from stock_screener.auth.kite_token import load_access_token, save_access_token, token_status
from stock_screener.config import get_data_root, load_config, require_env
from stock_screener.data.kite import KiteDataProvider
from stock_screener.data.nse_market_cap import (
    DEFAULT_NSE_MARKET_CAP_URL,
    fetch_market_caps_from_nse_excel,
    load_nse_market_cap_excel,
)
from stock_screener.data.storage import Storage
from stock_screener.data.supabase_store import SupabaseStore
from stock_screener.jobs.daily_scan import run_daily_scan
from stock_screener.resample import resample_daily_to_weekly
from stock_screener.strategy.weekly_buy_sell import run_weekly_buy_sell
from stock_screener.jobs.large_deals import (
    default_last_7_days_range,
    fetch_and_store_current_large_deals,
)
from stock_screener.web.charts import build_signal_chart, latest_signal_summary


app = FastAPI(title="NSE/BSE Investment Signal Screener")

BASE_DIR = Path(__file__).resolve().parents[2]
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

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
    metadata_for_merge["metadata_symbol_key"] = metadata_for_merge["symbol"].astype(str).str.upper()
    metadata_for_merge = metadata_for_merge.drop(columns=["symbol"], errors="ignore")
    enriched["symbol_key"] = enriched[symbol_column].astype(str).str.upper()
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


def _records(frame: pd.DataFrame) -> list[dict]:
    if frame.empty:
        return []
    return frame.where(pd.notna(frame), "").to_dict(orient="records")


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
    for name in ("token", "stock_search", "market_cap_bucket", "min_market_cap_cr", "max_market_cap_cr"):
        value = request.query_params.get(name, "").strip()
        if value:
            params.append(f"{name}={quote(value)}")
    return ("&" + "&".join(params)) if params else ""


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


def _scan_redirect_url(summary: dict[str, Any], query_suffix: str) -> str:
    return (
        "/?"
        f"scan_ran=1&symbols_scanned={summary.get('symbols_scanned', 0)}"
        f"&filtered_matches={summary.get('filtered_matches', 0)}"
        f"{query_suffix}"
    )


def _scan_error_url(error: Exception, query_suffix: str) -> str:
    return f"/?scan_error={quote(str(error)[:500])}{query_suffix}"


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

    filtered = _enrich_with_symbol_metadata(filtered, metadata, "symbol")
    raw = _enrich_with_symbol_metadata(raw, metadata, "symbol")
    scan_details = _enrich_with_symbol_metadata(scan_details, metadata, "symbol")

    filtered = _apply_market_cap_filters(filtered, min_market_cap, max_market_cap, selected_market_cap_bucket)
    raw = _apply_market_cap_filters(raw, min_market_cap, max_market_cap, selected_market_cap_bucket)
    scan_details = _apply_market_cap_filters(scan_details, min_market_cap, max_market_cap, selected_market_cap_bucket)

    filtered = _apply_stock_search(filtered, stock_search)
    raw = _apply_stock_search(raw, stock_search)
    scan_details = _apply_stock_search(scan_details, stock_search)

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
            "market_cap_bounds": market_cap_bounds,
            "has_metadata": not metadata.empty,
            "scan_ran": request.query_params.get("scan_ran", ""),
            "scan_error": request.query_params.get("scan_error", ""),
            "scan_job": request.query_params.get("scan_job", ""),
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
            stocks["symbol_key"] = stocks["tradingsymbol"].astype(str).str.upper()
            merge_type = "inner" if restrict_to_metadata_symbols else "left"
            stocks = stocks.merge(metadata, left_on="symbol_key", right_on="symbol", how=merge_type)
            stocks = stocks.drop(columns=["symbol_key", "symbol"], errors="ignore")
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
