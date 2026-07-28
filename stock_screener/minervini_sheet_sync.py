from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from stock_screener.data.storage import Storage
from stock_screener.google_sheets import (
    batch_update_google_sheet_values,
    google_sheet_worksheet_id,
    load_google_sheets_settings,
    read_google_sheet_values,
)
from stock_screener.symbols import normalize_nse_symbol
from stock_screener.weekly_buy_tracker_study import _evaluate_minervini_template


DEFAULT_WORKSHEET_TITLE = "Sheet1"
STOCK_SYMBOL_HEADER = "stock_symbol"
MINERVINI_FILTER_HEADER = "minervini_filter"


@dataclass(frozen=True)
class MinerviniSheetSyncResult:
    summary: dict[str, Any]
    row_updates: pd.DataFrame


def run_minervini_sheet_sync(
    storage: Storage,
    data_root: Path,
    spreadsheet_id: str | None = None,
    worksheet_title: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> MinerviniSheetSyncResult:
    settings = load_google_sheets_settings(data_root)
    target_spreadsheet_id = str(spreadsheet_id or settings.spreadsheet_id).strip()
    target_worksheet_title = str(worksheet_title or settings.worksheet_title or DEFAULT_WORKSHEET_TITLE).strip() or DEFAULT_WORKSHEET_TITLE
    if not target_spreadsheet_id:
        raise RuntimeError("Google Sheet target is not configured yet.")

    values = read_google_sheet_values(data_root, target_spreadsheet_id, target_worksheet_title)
    headers, data_rows = _split_sheet_rows(values)
    header_map = _header_index_map(headers)
    if STOCK_SYMBOL_HEADER not in header_map:
        raise RuntimeError('The target worksheet is missing the "Stock_Symbol" header.')
    if MINERVINI_FILTER_HEADER not in header_map:
        raise RuntimeError('The target worksheet is missing the "Minervini Filter" header.')

    instruments = storage.load_instruments()
    symbol_index = header_map[STOCK_SYMBOL_HEADER]
    minervini_index = header_map[MINERVINI_FILTER_HEADER]
    minervini_column = _column_letters(minervini_index + 1)

    updates: list[dict[str, Any]] = []
    row_records: list[dict[str, Any]] = []
    total = len(data_rows)
    for offset, row in enumerate(data_rows, start=2):
        raw_symbol = _cell(row, symbol_index)
        if progress_callback:
            progress_callback(
                {
                    "phase": "Evaluating Minervini rules",
                    "completed": offset - 2,
                    "total": total,
                    "current_symbol": raw_symbol,
                    "current_exchange": "NSE",
                }
            )
        if not raw_symbol:
            row_records.append(
                {
                    "sheet_row": offset,
                    "input_symbol": "",
                    "resolved_symbol": "",
                    "exchange": "",
                    "history_rows": 0,
                    "minervini_rule_count": 0,
                    "minervini_filter": 0,
                    "status": "blank_symbol",
                }
            )
            continue

        resolved_exchange, resolved_symbol = _resolve_symbol(storage, instruments, raw_symbol)
        daily = storage.load_candles(resolved_exchange, resolved_symbol, "1D") if resolved_exchange and resolved_symbol else pd.DataFrame()
        result = _evaluate_minervini_template(daily)
        flag = 1 if bool(result.get("minervini_pass")) else 0
        status = "ok"
        if daily.empty:
            status = "missing_history"
        elif len(daily) < 200:
            status = "short_history"

        updates.append(
            {
                "range": f"{target_worksheet_title}!{minervini_column}{offset}",
                "majorDimension": "ROWS",
                "values": [[flag]],
            }
        )
        row_records.append(
            {
                "sheet_row": offset,
                "input_symbol": raw_symbol,
                "resolved_symbol": resolved_symbol,
                "exchange": resolved_exchange,
                "history_rows": int(len(daily)),
                "minervini_rule_count": int(result.get("minervini_rule_count", 0) or 0),
                "minervini_filter": flag,
                "status": status,
            }
        )

    if progress_callback:
        progress_callback(
            {
                "phase": "Writing Google Sheet updates",
                "completed": total,
                "total": total,
                "current_symbol": "",
                "current_exchange": "",
            }
        )
    batch_response = batch_update_google_sheet_values(data_root, target_spreadsheet_id, updates)
    worksheet_id = google_sheet_worksheet_id(data_root, target_spreadsheet_id, target_worksheet_title)
    row_frame = pd.DataFrame(row_records)
    summary = {
        "spreadsheet_id": target_spreadsheet_id,
        "worksheet_title": target_worksheet_title,
        "sheet_row_count": total,
        "symbols_evaluated": int(row_frame["input_symbol"].astype(str).str.strip().ne("").sum()) if not row_frame.empty else 0,
        "rows_updated": len(updates),
        "minervini_pass_count": int((row_frame["minervini_filter"] == 1).sum()) if not row_frame.empty else 0,
        "missing_history_count": int((row_frame["status"] == "missing_history").sum()) if not row_frame.empty else 0,
        "short_history_count": int((row_frame["status"] == "short_history").sum()) if not row_frame.empty else 0,
        "blank_symbol_count": int((row_frame["status"] == "blank_symbol").sum()) if not row_frame.empty else 0,
        "updated_ranges": int(batch_response.get("totalUpdatedRanges", batch_response.get("updatedRanges", 0)) or 0),
        "spreadsheet_url": f"https://docs.google.com/spreadsheets/d/{target_spreadsheet_id}/edit#gid={worksheet_id}",
    }
    return MinerviniSheetSyncResult(summary=summary, row_updates=row_frame)


def save_minervini_sheet_sync_outputs(result: MinerviniSheetSyncResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "latest_summary.json").write_text(json.dumps(result.summary, indent=2), encoding="utf-8")
    result.row_updates.to_csv(output_dir / "latest_row_updates.csv", index=False)


def load_minervini_sheet_sync_outputs(output_dir: Path) -> MinerviniSheetSyncResult:
    summary_path = output_dir / "latest_summary.json"
    rows_path = output_dir / "latest_row_updates.csv"
    summary: dict[str, Any] = {}
    rows = pd.DataFrame(
        columns=[
            "sheet_row",
            "input_symbol",
            "resolved_symbol",
            "exchange",
            "history_rows",
            "minervini_rule_count",
            "minervini_filter",
            "status",
        ]
    )
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            summary = {}
    if rows_path.exists():
        try:
            rows = pd.read_csv(rows_path)
        except pd.errors.EmptyDataError:
            rows = rows.iloc[0:0].copy()
    return MinerviniSheetSyncResult(summary=summary, row_updates=rows)


def _split_sheet_rows(values: list[list[Any]]) -> tuple[list[Any], list[list[Any]]]:
    if not values:
        return [], []
    header_row = values[0] if isinstance(values[0], list) else []
    data_rows = [row if isinstance(row, list) else [] for row in values[1:]]
    return header_row, data_rows


def _header_index_map(headers: list[Any]) -> dict[str, int]:
    return {_normalize_header(header): index for index, header in enumerate(headers)}


def _normalize_header(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def _cell(row: list[Any], index: int) -> str:
    if index < 0 or index >= len(row):
        return ""
    return str(row[index] or "").strip()


def _column_letters(index: int) -> str:
    result = ""
    current = int(index)
    while current > 0:
        current, remainder = divmod(current - 1, 26)
        result = chr(65 + remainder) + result
    return result or "A"


def _resolve_symbol(storage: Storage, instruments: pd.DataFrame, raw_symbol: str) -> tuple[str, str]:
    candidate = str(raw_symbol or "").strip().upper()
    normalized = normalize_nse_symbol(candidate)
    for symbol in _symbol_candidates(candidate, normalized):
        daily = storage.load_candles("NSE", symbol, "1D")
        if not daily.empty:
            return "NSE", symbol

    if instruments.empty or not {"exchange", "tradingsymbol"}.issubset(instruments.columns):
        return "NSE", candidate

    working = instruments.copy()
    working["exchange"] = working["exchange"].astype(str).str.upper()
    working["tradingsymbol"] = working["tradingsymbol"].astype(str).str.upper()
    working["symbol_key"] = working["tradingsymbol"].apply(normalize_nse_symbol)

    exact_matches = working[working["tradingsymbol"] == candidate]
    for _, row in exact_matches.iterrows():
        exchange = str(row.get("exchange", "")).strip().upper()
        symbol = str(row.get("tradingsymbol", "")).strip().upper()
        daily = storage.load_candles(exchange, symbol, "1D")
        if not daily.empty:
            return exchange, symbol

    normalized_matches = working[working["symbol_key"] == normalized]
    preferred = normalized_matches.sort_values(["exchange", "tradingsymbol"])
    for _, row in preferred.iterrows():
        exchange = str(row.get("exchange", "")).strip().upper()
        symbol = str(row.get("tradingsymbol", "")).strip().upper()
        daily = storage.load_candles(exchange, symbol, "1D")
        if not daily.empty:
            return exchange, symbol

    return "NSE", candidate


def _symbol_candidates(candidate: str, normalized: str) -> list[str]:
    values = []
    for value in (candidate, normalized):
        cleaned = str(value or "").strip().upper()
        if cleaned and cleaned not in values:
            values.append(cleaned)
    return values
