from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from stock_screener.data.storage import Storage
from stock_screener.universe import build_universe


WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


@dataclass(frozen=True)
class WeekdayPressureStudyResult:
    summary: dict[str, Any]
    stock_stats: pd.DataFrame
    weekday_details: pd.DataFrame


def enrich_with_weekday_profiles(
    storage: Storage,
    frame: pd.DataFrame,
    exchange_column: str = "exchange",
    symbol_column: str = "symbol",
    name_column: str = "name",
    lookback_years: int = 5,
) -> pd.DataFrame:
    if frame.empty or symbol_column not in frame.columns:
        return frame.copy()

    enriched = frame.copy()
    if exchange_column not in enriched.columns:
        enriched[exchange_column] = "NSE"
    if name_column not in enriched.columns:
        enriched[name_column] = enriched[symbol_column]

    markers: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in enriched[[exchange_column, symbol_column, name_column]].itertuples(index=False):
        exchange = str(getattr(row, exchange_column)).upper().strip()
        symbol = str(getattr(row, symbol_column)).upper().strip()
        name = str(getattr(row, name_column)).strip() or symbol
        if not symbol or (exchange, symbol) in seen:
            continue
        seen.add((exchange, symbol))
        daily = storage.load_candles(exchange, symbol, "1D")
        if daily.empty:
            continue
        trimmed = _trim_to_lookback_years(daily, lookback_years)
        _, stock_row = _weekday_pressure_for_stock(trimmed, exchange, symbol, name)
        if stock_row is not None:
            markers.append(stock_row)

    if not markers:
        if "best_buy_weekday" not in enriched.columns:
            enriched["best_buy_weekday"] = pd.NA
        if "best_sell_weekday" not in enriched.columns:
            enriched["best_sell_weekday"] = pd.NA
        return enriched

    marker_frame = pd.DataFrame(markers)
    merged = enriched.merge(
        marker_frame[["exchange", "symbol", "best_buy_weekday", "best_sell_weekday"]],
        left_on=[exchange_column, symbol_column],
        right_on=["exchange", "symbol"],
        how="left",
        suffixes=("", "_weekday"),
    )
    merged = merged.drop(columns=["exchange_weekday", "symbol_weekday"], errors="ignore")
    return merged


def run_weekday_pressure_study(
    config: dict[str, Any],
    storage: Storage,
    exchange: str = "NSE",
    symbols: set[str] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> WeekdayPressureStudyResult:
    universe_rows = _kite_instruments_universe(storage, config, exchange)
    if symbols is not None:
        allowed_symbols = {str(symbol).upper().strip() for symbol in symbols if str(symbol).strip()}
        universe_rows = [row for row in universe_rows if str(row.get("symbol") or "").upper() in allowed_symbols]
    if not universe_rows:
        return WeekdayPressureStudyResult(_empty_summary(exchange, 0), pd.DataFrame(), pd.DataFrame())

    detail_frames: list[pd.DataFrame] = []
    stock_rows: list[dict[str, Any]] = []
    symbols_processed = 0

    _emit_progress(
        progress_callback,
        phase="Analyzing weekday buy/sell pressure",
        completed=0,
        total=len(universe_rows),
        current_symbol="",
        current_exchange=exchange,
    )

    for index, universe_row in enumerate(universe_rows, start=1):
        row_exchange = str(universe_row.get("exchange") or exchange).upper()
        symbol = str(universe_row.get("symbol") or "").upper()
        name = str(universe_row.get("name") or symbol)
        if not symbol:
            continue

        daily = storage.load_candles(row_exchange, symbol, "1D")
        detail, stock_row = _weekday_pressure_for_stock(daily, row_exchange, symbol, name)
        if not detail.empty:
            detail_frames.append(detail)
            stock_rows.append(stock_row)
        symbols_processed += 1
        _emit_progress(
            progress_callback,
            phase="Analyzing weekday buy/sell pressure",
            completed=index,
            total=len(universe_rows),
            current_symbol=symbol,
            current_exchange=row_exchange,
        )

    stock_stats = pd.DataFrame(stock_rows).sort_values(
        ["best_buy_pressure_score", "sample_days", "symbol"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True) if stock_rows else pd.DataFrame()
    weekday_details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    summary = build_weekday_pressure_summary(stock_stats, weekday_details, exchange, symbols_processed)
    return WeekdayPressureStudyResult(summary, stock_stats, weekday_details)


def save_weekday_pressure_outputs(result: WeekdayPressureStudyResult, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "latest_summary.csv"
    stock_stats_path = output_dir / "latest_stock_stats.csv"
    weekday_details_path = output_dir / "latest_weekday_details.csv"
    pd.DataFrame([result.summary]).to_csv(summary_path, index=False)
    result.stock_stats.to_csv(stock_stats_path, index=False)
    result.weekday_details.to_csv(weekday_details_path, index=False)
    return {
        "summary": summary_path,
        "stock_stats": stock_stats_path,
        "weekday_details": weekday_details_path,
    }


def load_weekday_pressure_outputs(output_dir: Path) -> WeekdayPressureStudyResult:
    summary_path = output_dir / "latest_summary.csv"
    stock_stats_path = output_dir / "latest_stock_stats.csv"
    weekday_details_path = output_dir / "latest_weekday_details.csv"

    summary = {}
    if summary_path.exists():
        try:
            summary_frame = pd.read_csv(summary_path)
            if not summary_frame.empty:
                summary = summary_frame.iloc[0].to_dict()
        except pd.errors.EmptyDataError:
            summary = {}

    def _read(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    return WeekdayPressureStudyResult(
        summary=summary,
        stock_stats=_read(stock_stats_path),
        weekday_details=_read(weekday_details_path),
    )


def build_weekday_pressure_summary(
    stock_stats: pd.DataFrame,
    weekday_details: pd.DataFrame,
    exchange: str,
    symbols_processed: int,
) -> dict[str, Any]:
    if stock_stats.empty:
        return _empty_summary(exchange, symbols_processed)

    buy_counts = stock_stats["best_buy_weekday"].value_counts(dropna=False).to_dict() if "best_buy_weekday" in stock_stats.columns else {}
    sell_counts = stock_stats["best_sell_weekday"].value_counts(dropna=False).to_dict() if "best_sell_weekday" in stock_stats.columns else {}
    return {
        "exchange": exchange,
        "symbols_processed": symbols_processed,
        "stocks_with_weekday_profile": len(stock_stats),
        "weekday_rows": len(weekday_details),
        "top_buy_weekday": max(buy_counts, key=buy_counts.get) if buy_counts else "",
        "top_buy_weekday_count": buy_counts.get(max(buy_counts, key=buy_counts.get), 0) if buy_counts else 0,
        "top_sell_weekday": max(sell_counts, key=sell_counts.get) if sell_counts else "",
        "top_sell_weekday_count": sell_counts.get(max(sell_counts, key=sell_counts.get), 0) if sell_counts else 0,
    }


def _weekday_pressure_for_stock(
    daily: pd.DataFrame,
    exchange: str,
    symbol: str,
    name: str,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    if daily.empty:
        return pd.DataFrame(), None

    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame[frame["date"].notna()].copy()
    if frame.empty:
        return pd.DataFrame(), None

    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "close", "volume"]).sort_values("date").reset_index(drop=True)
    if len(frame) < 10:
        return pd.DataFrame(), None

    frame["weekday"] = pd.Categorical(frame["date"].dt.day_name(), categories=WEEKDAY_ORDER, ordered=True)
    frame = frame[frame["weekday"].notna()].copy()
    if frame.empty:
        return pd.DataFrame(), None

    frame["traded_value"] = frame["close"] * frame["volume"]
    frame["return_pct"] = frame["close"].pct_change() * 100.0
    frame["buy_pressure"] = (frame["traded_value"] * frame["return_pct"].clip(lower=0.0).fillna(0.0)) / 100.0
    frame["sell_pressure"] = (frame["traded_value"] * (-frame["return_pct"].clip(upper=0.0).fillna(0.0))) / 100.0
    frame["up_day"] = frame["return_pct"] > 0
    frame["down_day"] = frame["return_pct"] < 0

    grouped = (
        frame.groupby("weekday", observed=False)
        .agg(
            sample_days=("weekday", "size"),
            up_days=("up_day", "sum"),
            down_days=("down_day", "sum"),
            avg_return_pct=("return_pct", "mean"),
            avg_traded_value=("traded_value", "mean"),
            avg_buy_pressure=("buy_pressure", "mean"),
            avg_sell_pressure=("sell_pressure", "mean"),
            median_buy_pressure=("buy_pressure", "median"),
            median_sell_pressure=("sell_pressure", "median"),
        )
        .reset_index()
    )
    grouped["exchange"] = exchange
    grouped["symbol"] = symbol
    grouped["name"] = name
    grouped["sample_days"] = grouped["sample_days"].fillna(0).astype(int)
    grouped["up_days"] = grouped["up_days"].fillna(0).astype(int)
    grouped["down_days"] = grouped["down_days"].fillna(0).astype(int)

    ranked = grouped[grouped["sample_days"] > 0].copy()
    if ranked.empty:
        return pd.DataFrame(), None

    best_buy_row = ranked.sort_values(
        ["avg_buy_pressure", "up_days", "sample_days"],
        ascending=[False, False, False],
        na_position="last",
    ).iloc[0]
    best_sell_row = ranked.sort_values(
        ["avg_sell_pressure", "down_days", "sample_days"],
        ascending=[False, False, False],
        na_position="last",
    ).iloc[0]

    stock_row = {
        "exchange": exchange,
        "symbol": symbol,
        "name": name,
        "sample_days": int(len(frame)),
        "history_start": frame["date"].min(),
        "history_end": frame["date"].max(),
        "best_buy_weekday": str(best_buy_row["weekday"]),
        "best_buy_pressure_score": float(best_buy_row["avg_buy_pressure"]),
        "best_buy_avg_return_pct": float(best_buy_row["avg_return_pct"]),
        "best_buy_up_days": int(best_buy_row["up_days"]),
        "best_buy_sample_days": int(best_buy_row["sample_days"]),
        "best_sell_weekday": str(best_sell_row["weekday"]),
        "best_sell_pressure_score": float(best_sell_row["avg_sell_pressure"]),
        "best_sell_avg_return_pct": float(best_sell_row["avg_return_pct"]),
        "best_sell_down_days": int(best_sell_row["down_days"]),
        "best_sell_sample_days": int(best_sell_row["sample_days"]),
    }
    return grouped, stock_row


def _trim_to_lookback_years(daily: pd.DataFrame, lookback_years: int) -> pd.DataFrame:
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame[frame["date"].notna()].copy()
    if frame.empty or lookback_years <= 0:
        return frame
    max_date = frame["date"].max()
    cutoff = max_date - pd.DateOffset(years=int(lookback_years))
    return frame[frame["date"] >= cutoff].copy()


def _kite_instruments_universe(storage: Storage, config: dict[str, Any], exchange: str) -> list[dict[str, str]]:
    instruments = storage.load_instruments()
    if instruments.empty:
        return []
    universe = build_universe(instruments, config)
    if universe.empty or "tradingsymbol" not in universe.columns:
        return []
    if "exchange" in universe.columns:
        universe = universe[universe["exchange"].astype(str).str.upper() == exchange.upper()]
    else:
        universe["exchange"] = exchange.upper()
    if universe.empty:
        return []
    universe = universe.copy()
    universe["symbol"] = universe["tradingsymbol"].astype(str).str.upper().str.strip()
    if "name" not in universe.columns:
        universe["name"] = universe["symbol"]
    universe["name"] = universe["name"].fillna("").astype(str).str.strip().mask(lambda s: s == "", universe["symbol"])
    universe["exchange"] = universe["exchange"].astype(str).str.upper().str.strip()
    universe = universe[universe["symbol"] != ""]
    universe = universe.drop_duplicates(subset=["exchange", "symbol"], keep="last").sort_values(["exchange", "symbol"])
    return universe[["exchange", "symbol", "name"]].to_dict(orient="records")


def _empty_summary(exchange: str, symbols_processed: int = 0) -> dict[str, Any]:
    return {
        "exchange": exchange,
        "symbols_processed": symbols_processed,
        "stocks_with_weekday_profile": 0,
        "weekday_rows": 0,
        "top_buy_weekday": "",
        "top_buy_weekday_count": 0,
        "top_sell_weekday": "",
        "top_sell_weekday_count": 0,
    }


def _emit_progress(
    progress_callback: Callable[[dict[str, Any]], None] | None,
    *,
    phase: str,
    completed: int,
    total: int,
    current_symbol: str,
    current_exchange: str,
) -> None:
    if not progress_callback:
        return
    progress_callback(
        {
            "phase": phase,
            "completed": completed,
            "total": total,
            "current_symbol": current_symbol,
            "current_exchange": current_exchange,
        }
    )
