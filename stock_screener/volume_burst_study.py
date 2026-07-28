from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from stock_screener.data.storage import Storage
from stock_screener.weekly_buy_tracker_study import _emit_progress, _evaluate_latest_volume_burst, _load_name_map, _to_float


@dataclass(frozen=True)
class VolumeBurstStudyResult:
    summary: dict[str, Any]
    stock_stats: pd.DataFrame


def run_volume_burst_study(
    storage: Storage,
    exchange: str = "NSE",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> VolumeBurstStudyResult:
    data_root = storage.data_root
    all_symbols = sorted(p.stem for p in (data_root / "candles" / exchange / "1D").glob("*.csv"))
    name_map = _load_name_map(storage, exchange)

    rows: list[dict[str, Any]] = []
    _emit_progress(
        progress_callback,
        phase="Scanning latest volume burst",
        completed=0,
        total=len(all_symbols),
        current_symbol="",
        current_exchange=exchange,
    )

    for index, symbol in enumerate(all_symbols, start=1):
        daily = storage.load_candles(exchange, symbol, "1D")
        _emit_progress(
            progress_callback,
            phase="Scanning latest volume burst",
            completed=index,
            total=len(all_symbols),
            current_symbol=symbol,
            current_exchange=exchange,
        )
        if daily.empty or len(daily) < 10:
            continue

        frame = daily.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        if frame.empty or len(frame) < 10:
            continue

        latest_daily = frame.iloc[-1]
        volume_burst = _evaluate_latest_volume_burst(frame)
        rows.append(
            {
                "exchange": exchange,
                "symbol": symbol,
                "name": name_map.get(symbol, symbol),
                "latest_close": _to_float(latest_daily.get("close")),
                "latest_close_date": latest_daily.get("date"),
                **volume_burst,
            }
        )

    stock_stats = pd.DataFrame(rows)
    if not stock_stats.empty:
        if "latest_close_date" in stock_stats.columns:
            stock_stats["latest_close_date"] = pd.to_datetime(stock_stats["latest_close_date"], errors="coerce")
        if "latest_volume_ratio_prev_9d" in stock_stats.columns:
            stock_stats["latest_volume_ratio_prev_9d"] = pd.to_numeric(stock_stats["latest_volume_ratio_prev_9d"], errors="coerce")
        stock_stats = stock_stats.sort_values(
            ["latest_volume_ratio_prev_9d", "symbol"],
            ascending=[False, True],
            na_position="last",
        ).reset_index(drop=True)

    summary = _build_summary(exchange, len(all_symbols), stock_stats)
    return VolumeBurstStudyResult(summary=summary, stock_stats=stock_stats)


def save_volume_burst_outputs(result: VolumeBurstStudyResult, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "latest_summary.csv"
    stock_stats_path = output_dir / "latest_stock_stats.csv"
    pd.DataFrame([result.summary]).to_csv(summary_path, index=False)
    result.stock_stats.to_csv(stock_stats_path, index=False)
    return {"summary": summary_path, "stock_stats": stock_stats_path}


def load_volume_burst_outputs(output_dir: Path) -> VolumeBurstStudyResult:
    def _read(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    summary = {}
    summary_path = output_dir / "latest_summary.csv"
    if summary_path.exists():
        try:
            frame = pd.read_csv(summary_path)
            if not frame.empty:
                summary = frame.iloc[0].to_dict()
        except pd.errors.EmptyDataError:
            summary = {}

    return VolumeBurstStudyResult(summary=summary, stock_stats=_read(output_dir / "latest_stock_stats.csv"))


def _build_summary(exchange: str, symbols_processed: int, stock_stats: pd.DataFrame) -> dict[str, Any]:
    latest_date = ""
    if not stock_stats.empty and "latest_close_date" in stock_stats.columns:
        latest_dates = pd.to_datetime(stock_stats["latest_close_date"], errors="coerce")
        if latest_dates.notna().any():
            latest_date = str(latest_dates.max().date())

    ratio_series = (
        pd.to_numeric(stock_stats["latest_volume_ratio_prev_9d"], errors="coerce").dropna()
        if "latest_volume_ratio_prev_9d" in stock_stats.columns
        else pd.Series(dtype=float)
    )
    match_count = int(stock_stats["latest_volume_3x_prev_9d"].astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"}).sum()) if (not stock_stats.empty and "latest_volume_3x_prev_9d" in stock_stats.columns) else 0
    return {
        "exchange": exchange,
        "symbols_processed": symbols_processed,
        "stocks_with_history": int(len(stock_stats)),
        "volume_burst_matches": match_count,
        "latest_close_date": latest_date,
        "avg_latest_volume_ratio_prev_9d": float(ratio_series.mean()) if not ratio_series.empty else 0.0,
    }
