from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from stock_screener.data.storage import Storage


DEFAULT_DMA_WINDOWS = (75, 100, 200)
DEFAULT_DMA_TOUCH_TOLERANCE_PCT = 2.0
DEFAULT_PIVOT_ORDER = 3
DEFAULT_RECENT_PLOT_DAYS = 90
DEFAULT_LOOKBACK_MONTHS = 6


@dataclass(frozen=True)
class StockSignatureStudyResult:
    exchange: str
    symbol: str
    daily: pd.DataFrame
    pivots: pd.DataFrame
    legs: pd.DataFrame
    cycles: pd.DataFrame
    dma_rebounds: pd.DataFrame
    dma_summary: pd.DataFrame
    summary: dict[str, Any]
    current: dict[str, Any]
    config: dict[str, Any]


@dataclass(frozen=True)
class StockSignatureScanResult:
    summary: dict[str, Any]
    stock_stats: pd.DataFrame


def run_stock_signature_study(
    daily: pd.DataFrame,
    *,
    exchange: str = "NSE",
    symbol: str,
    pivot_order: int = DEFAULT_PIVOT_ORDER,
    dma_windows: Iterable[int] = DEFAULT_DMA_WINDOWS,
    dma_touch_tolerance_pct: float = DEFAULT_DMA_TOUCH_TOLERANCE_PCT,
    lookback_months: int = DEFAULT_LOOKBACK_MONTHS,
    as_of_date: Any | None = None,
) -> StockSignatureStudyResult:
    order = max(int(pivot_order), 1)
    windows = tuple(sorted({max(int(value), 2) for value in dma_windows}))
    tolerance = max(float(dma_touch_tolerance_pct), 0.0)
    frame = _prepare_daily(daily)
    frame = _clip_analysis_window(frame, lookback_months=max(int(lookback_months), 1), as_of_date=as_of_date)
    frame = _add_indicators(frame, windows)
    pivots = _detect_pivots(frame, order, windows)
    legs = _build_low_to_high_legs(pivots, windows, tolerance)
    cycles = _build_bullish_cycles(pivots)
    dma_rebounds, dma_summary = _summarize_dma_rebounds(legs)
    summary = _build_summary(exchange, symbol, frame, pivots, legs, cycles, order, lookback_months=max(int(lookback_months), 1))
    current = _build_current_state(frame, pivots, windows)
    return StockSignatureStudyResult(
        exchange=str(exchange).upper(),
        symbol=str(symbol).upper(),
        daily=frame,
        pivots=pivots,
        legs=legs,
        cycles=cycles,
        dma_rebounds=dma_rebounds,
        dma_summary=dma_summary,
        summary=summary,
        current=current,
        config={
            "pivot_order": order,
            "dma_windows": list(windows),
            "dma_touch_tolerance_pct": tolerance,
            "lookback_months": max(int(lookback_months), 1),
            "as_of_date": "" if as_of_date is None else str(as_of_date),
        },
    )


def run_stock_signature_scan(
    storage: Storage,
    universe: pd.DataFrame,
    *,
    exchange: str = "NSE",
    universe_name: str = "NIFTY100",
    pivot_order: int = DEFAULT_PIVOT_ORDER,
    dma_windows: Iterable[int] = DEFAULT_DMA_WINDOWS,
    dma_touch_tolerance_pct: float = DEFAULT_DMA_TOUCH_TOLERANCE_PCT,
    lookback_months: int = DEFAULT_LOOKBACK_MONTHS,
    as_of_date: Any | None = None,
    required_latest_date: Any | None = None,
    progress_callback: Any | None = None,
) -> StockSignatureScanResult:
    windows = tuple(sorted({max(int(value), 2) for value in dma_windows}))
    symbols = _universe_symbols(universe)
    metadata = _universe_metadata(universe)
    required_latest_ts = _coerce_normalized_timestamp(required_latest_date)
    rows: list[dict[str, Any]] = []
    latest_dates: list[pd.Timestamp] = []

    _emit_progress(
        progress_callback,
        phase="Scanning Nifty 100 signatures",
        completed=0,
        total=len(symbols),
        current_symbol="",
        current_exchange=exchange,
    )
    for index, symbol in enumerate(symbols, start=1):
        daily = storage.load_candles(exchange, symbol, "1D")
        result = run_stock_signature_study(
            daily,
            exchange=exchange,
            symbol=symbol,
            pivot_order=pivot_order,
            dma_windows=windows,
            dma_touch_tolerance_pct=dma_touch_tolerance_pct,
            lookback_months=lookback_months,
            as_of_date=as_of_date,
        )
        rows.append(_scan_row(result, metadata.get(symbol, {}), required_latest_date=required_latest_ts))
        latest_date = pd.to_datetime(result.current.get("latest_date"), errors="coerce")
        if pd.notna(latest_date):
            latest_dates.append(latest_date)
        _emit_progress(
            progress_callback,
            phase="Scanning Nifty 100 signatures",
            completed=index,
            total=len(symbols),
            current_symbol=symbol,
            current_exchange=exchange,
        )

    stock_stats = pd.DataFrame(rows)
    if not stock_stats.empty:
        stock_stats = stock_stats.sort_values(
            ["historical_rank_score", "sample_confidence_score", "symbol"],
            ascending=[False, False, True],
            na_position="last",
        ).reset_index(drop=True)
        stock_stats["rank"] = np.arange(1, len(stock_stats) + 1)

    latest_market_date = max(latest_dates).strftime("%Y-%m-%d") if latest_dates else ""
    ready_count = int((stock_stats["data_status"] == "READY").sum()) if not stock_stats.empty and "data_status" in stock_stats.columns else 0
    summary = {
        "exchange": str(exchange).upper(),
        "universe": str(universe_name or "CUSTOM").upper(),
        "symbols_requested": int(len(symbols)),
        "symbols_processed": int(len(stock_stats)),
        "stocks_with_ready_signature": ready_count,
        "latest_market_date": latest_market_date,
        "lookback_months": max(int(lookback_months), 1),
        "pivot_order": max(int(pivot_order), 1),
        "dma_windows": ",".join(str(value) for value in windows),
        "dma_touch_tolerance_pct": max(float(dma_touch_tolerance_pct), 0.0),
        "requested_as_of_date": "" if as_of_date is None else str(as_of_date),
        "required_latest_date": "" if required_latest_ts is None else required_latest_ts.strftime("%Y-%m-%d"),
    }
    return StockSignatureScanResult(summary=summary, stock_stats=stock_stats)


def save_stock_signature_scan_outputs(result: StockSignatureScanResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(_json_safe(result.summary), indent=2), encoding="utf-8")
    result.stock_stats.to_csv(output_dir / "latest_stock_stats.csv", index=False)


def load_stock_signature_scan_outputs(output_dir: Path) -> StockSignatureScanResult:
    summary_path = output_dir / "summary.json"
    stock_stats_path = output_dir / "latest_stock_stats.csv"
    summary: dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}
    if stock_stats_path.exists():
        try:
            stock_stats = pd.read_csv(stock_stats_path)
        except pd.errors.EmptyDataError:
            stock_stats = pd.DataFrame()
    else:
        stock_stats = pd.DataFrame()
    return StockSignatureScanResult(summary=summary, stock_stats=stock_stats)


def _clip_analysis_window(
    frame: pd.DataFrame,
    *,
    lookback_months: int,
    as_of_date: Any | None,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    working = frame.copy()
    cutoff = pd.to_datetime(as_of_date, errors="coerce") if as_of_date is not None and str(as_of_date).strip() else pd.NaT
    if pd.notna(cutoff):
        working = working[working["date"].dt.normalize() <= pd.Timestamp(cutoff).normalize()].copy()
    if working.empty:
        return working.reset_index(drop=True)

    end_date = working["date"].max()
    start_date = end_date - pd.DateOffset(months=max(int(lookback_months), 1))
    return working[working["date"] >= start_date].reset_index(drop=True)


def _prepare_daily(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    frame = daily.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    required = ["date", "open", "high", "low", "close"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", format="mixed")
    for column in required[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "volume" not in frame.columns:
        frame["volume"] = np.nan
    else:
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")

    return (
        frame.dropna(subset=required)
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )


def _add_indicators(frame: pd.DataFrame, dma_windows: tuple[int, ...]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    enriched = frame.copy()
    for dma in dma_windows:
        enriched[f"dma_{dma}"] = enriched["close"].rolling(dma, min_periods=dma).mean()

    previous_close = enriched["close"].shift(1)
    true_range = pd.concat(
        [
            enriched["high"] - enriched["low"],
            (enriched["high"] - previous_close).abs(),
            (enriched["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    enriched["atr20"] = true_range.rolling(20, min_periods=20).mean()
    enriched["atr20_pct"] = enriched["atr20"] / enriched["close"] * 100.0
    return enriched


def _detect_pivots(frame: pd.DataFrame, pivot_order: int, dma_windows: tuple[int, ...]) -> pd.DataFrame:
    columns = [
        "row_idx",
        "date",
        "type",
        "price",
        "structure",
        "confirmation_date",
        *[f"dma_{dma}" for dma in dma_windows],
        *[f"distance_to_dma{dma}_pct" for dma in dma_windows],
    ]
    if frame.empty or len(frame) < pivot_order * 2 + 1:
        return pd.DataFrame(columns=columns)

    pivot_window = pivot_order * 2 + 1
    rolling_high = frame["high"].rolling(pivot_window, center=True).max()
    rolling_low = frame["low"].rolling(pivot_window, center=True).min()

    events: list[dict[str, Any]] = []
    for idx, row in frame.iterrows():
        if bool(row["high"] == rolling_high.loc[idx]):
            events.append(
                {
                    "row_idx": int(idx),
                    "date": row["date"],
                    "type": "HIGH",
                    "price": float(row["high"]),
                }
            )
        if bool(row["low"] == rolling_low.loc[idx]):
            events.append(
                {
                    "row_idx": int(idx),
                    "date": row["date"],
                    "type": "LOW",
                    "price": float(row["low"]),
                }
            )

    if not events:
        return pd.DataFrame(columns=columns)

    raw_pivots = pd.DataFrame(events).sort_values(["row_idx", "type"]).reset_index(drop=True)
    compressed: list[dict[str, Any]] = []
    for _, pivot in raw_pivots.iterrows():
        current = pivot.to_dict()
        if not compressed:
            compressed.append(current)
            continue
        previous = compressed[-1]
        if current["type"] == previous["type"]:
            if current["type"] == "HIGH" and current["price"] > previous["price"]:
                compressed[-1] = current
            elif current["type"] == "LOW" and current["price"] < previous["price"]:
                compressed[-1] = current
        else:
            compressed.append(current)

    pivots = pd.DataFrame(compressed)
    pivots["structure"] = ""
    last_high = None
    last_low = None
    for index in range(len(pivots)):
        pivot_type = pivots.loc[index, "type"]
        price = float(pivots.loc[index, "price"])
        if pivot_type == "HIGH":
            if last_high is None:
                structure = "H"
            elif price > last_high:
                structure = "HH"
            else:
                structure = "LH"
            last_high = price
        else:
            if last_low is None:
                structure = "L"
            elif price > last_low:
                structure = "HL"
            else:
                structure = "LL"
            last_low = price
        pivots.loc[index, "structure"] = structure

    confirmation_dates = []
    for _, pivot in pivots.iterrows():
        confirmation_idx = int(pivot["row_idx"]) + pivot_order
        confirmation_dates.append(frame.loc[confirmation_idx, "date"] if confirmation_idx < len(frame) else pd.NaT)
    pivots["confirmation_date"] = confirmation_dates

    for dma in dma_windows:
        pivots[f"dma_{dma}"] = pivots["row_idx"].map(frame[f"dma_{dma}"])
        pivots[f"distance_to_dma{dma}_pct"] = (pivots["price"] / pivots[f"dma_{dma}"] - 1.0) * 100.0

    return pivots[columns]


def _build_low_to_high_legs(
    pivots: pd.DataFrame,
    dma_windows: tuple[int, ...],
    dma_touch_tolerance_pct: float,
) -> pd.DataFrame:
    columns = [
        "low_date",
        "low_structure",
        "low_price",
        "high_date",
        "high_structure",
        "high_price",
        "jump_pct",
        "sessions",
        *[f"dma_{dma}" for dma in dma_windows],
        *[f"distance_dma{dma}_pct" for dma in dma_windows],
        "nearest_dma",
        "nearest_dma_distance_pct",
        "abs_nearest_dma_distance_pct",
        "dma_touch",
    ]
    if pivots.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for index in range(len(pivots) - 1):
        low = pivots.iloc[index]
        high = pivots.iloc[index + 1]
        if not (low["type"] == "LOW" and high["type"] == "HIGH"):
            continue

        row: dict[str, Any] = {
            "low_date": low["date"],
            "low_structure": low["structure"],
            "low_price": float(low["price"]),
            "high_date": high["date"],
            "high_structure": high["structure"],
            "high_price": float(high["price"]),
            "jump_pct": (float(high["price"]) / float(low["price"]) - 1.0) * 100.0,
            "sessions": int(high["row_idx"] - low["row_idx"]),
        }

        distances: dict[int, float] = {}
        for dma in dma_windows:
            dma_value = low.get(f"dma_{dma}", np.nan)
            distance = (float(low["price"]) / float(dma_value) - 1.0) * 100.0 if pd.notna(dma_value) else np.nan
            row[f"dma_{dma}"] = dma_value
            row[f"distance_dma{dma}_pct"] = distance
            distances[dma] = distance

        valid_distances = {dma: abs(value) for dma, value in distances.items() if np.isfinite(value)}
        if valid_distances:
            nearest_dma = min(valid_distances, key=valid_distances.get)
            nearest_distance = distances[nearest_dma]
        else:
            nearest_dma = np.nan
            nearest_distance = np.nan

        row["nearest_dma"] = nearest_dma
        row["nearest_dma_distance_pct"] = nearest_distance
        row["abs_nearest_dma_distance_pct"] = abs(nearest_distance) if np.isfinite(nearest_distance) else np.nan
        row["dma_touch"] = bool(
            abs(nearest_distance) <= dma_touch_tolerance_pct if np.isfinite(nearest_distance) else False
        )
        rows.append(row)

    return pd.DataFrame(rows, columns=columns)


def _build_bullish_cycles(pivots: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "start_date",
        "start_low",
        "peak_date",
        "peak_price",
        "breakdown_date",
        "breakdown_low",
        "higher_high_count",
        "higher_low_count",
        "upward_legs",
        "low_to_peak_gain_pct",
        "peak_to_breakdown_pct",
        "cycle_sessions",
    ]
    if pivots.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    index = 0
    while index < len(pivots):
        if pivots.iloc[index]["type"] != "LOW":
            index += 1
            continue

        start = pivots.iloc[index]
        previous_low = float(start["price"])
        previous_high = None
        hh_count = 0
        hl_count = 0
        upward_legs = 0
        peak_price = float(start["price"])
        peak_date = start["date"]
        breakdown = None

        next_index = index + 1
        while next_index < len(pivots):
            pivot = pivots.iloc[next_index]
            price = float(pivot["price"])
            if pivot["type"] == "HIGH":
                upward_legs += 1
                if previous_high is not None and price > previous_high:
                    hh_count += 1
                previous_high = price
                if price > peak_price:
                    peak_price = price
                    peak_date = pivot["date"]
            else:
                if price > previous_low:
                    hl_count += 1
                    previous_low = price
                else:
                    breakdown = pivot
                    break
            next_index += 1

        if breakdown is None:
            break

        rows.append(
            {
                "start_date": start["date"],
                "start_low": float(start["price"]),
                "peak_date": peak_date,
                "peak_price": peak_price,
                "breakdown_date": breakdown["date"],
                "breakdown_low": float(breakdown["price"]),
                "higher_high_count": hh_count,
                "higher_low_count": hl_count,
                "upward_legs": upward_legs,
                "low_to_peak_gain_pct": (peak_price / float(start["price"]) - 1.0) * 100.0,
                "peak_to_breakdown_pct": (float(breakdown["price"]) / peak_price - 1.0) * 100.0,
                "cycle_sessions": int(breakdown["row_idx"] - start["row_idx"]),
            }
        )
        index = next_index

    return pd.DataFrame(rows, columns=columns)


def _summarize_dma_rebounds(legs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_columns = [
        "nearest_dma",
        "rebounds",
        "median_jump_pct",
        "mean_jump_pct",
        "p25_jump_pct",
        "p75_jump_pct",
        "median_sessions_to_high",
    ]
    if legs.empty or "dma_touch" not in legs.columns:
        return legs.iloc[0:0].copy(), pd.DataFrame(columns=summary_columns)

    dma_rebounds = legs[legs["dma_touch"] == True].copy()
    if dma_rebounds.empty:
        return dma_rebounds, pd.DataFrame(columns=summary_columns)

    summary = (
        dma_rebounds.groupby("nearest_dma")
        .agg(
            rebounds=("jump_pct", "size"),
            median_jump_pct=("jump_pct", "median"),
            mean_jump_pct=("jump_pct", "mean"),
            p25_jump_pct=("jump_pct", lambda values: values.quantile(0.25)),
            p75_jump_pct=("jump_pct", lambda values: values.quantile(0.75)),
            median_sessions_to_high=("sessions", "median"),
        )
        .reset_index()
        .sort_values(["median_jump_pct", "rebounds"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return dma_rebounds, summary


def _build_summary(
    exchange: str,
    symbol: str,
    frame: pd.DataFrame,
    pivots: pd.DataFrame,
    legs: pd.DataFrame,
    cycles: pd.DataFrame,
    pivot_order: int,
    lookback_months: int,
) -> dict[str, Any]:
    if frame.empty:
        return {
            "exchange": str(exchange).upper(),
            "symbol": str(symbol).upper(),
            "data_status": "EMPTY",
            "pivot_order": pivot_order,
            "lookback_months": lookback_months,
        }

    summary: dict[str, Any] = {
        "exchange": str(exchange).upper(),
        "symbol": str(symbol).upper(),
        "data_status": "READY",
        "history_start": frame["date"].min().strftime("%Y-%m-%d"),
        "history_end": frame["date"].max().strftime("%Y-%m-%d"),
        "lookback_months": lookback_months,
        "latest_close": float(frame.iloc[-1]["close"]),
        "pivot_order": pivot_order,
        "confirmed_pivots": int(len(pivots)),
        "completed_bullish_cycles": int(len(cycles)),
        "swing_low_to_high_legs": int(len(legs)),
    }
    if not cycles.empty:
        summary.update(
            {
                "median_higher_highs_before_breakdown": float(cycles["higher_high_count"].median()),
                "median_higher_lows_before_breakdown": float(cycles["higher_low_count"].median()),
                "median_upward_legs": float(cycles["upward_legs"].median()),
                "median_complete_low_to_peak_gain_pct": float(cycles["low_to_peak_gain_pct"].median()),
                "median_cycle_sessions": float(cycles["cycle_sessions"].median()),
                "median_peak_to_breakdown_pct": float(cycles["peak_to_breakdown_pct"].median()),
            }
        )
    if not legs.empty:
        summary.update(
            {
                "median_low_to_high_jump_pct": float(legs["jump_pct"].median()),
                "p25_low_to_high_jump_pct": float(legs["jump_pct"].quantile(0.25)),
                "p75_low_to_high_jump_pct": float(legs["jump_pct"].quantile(0.75)),
                "median_low_to_high_sessions": float(legs["sessions"].median()),
            }
        )
    return summary


def _build_current_state(
    frame: pd.DataFrame,
    pivots: pd.DataFrame,
    dma_windows: tuple[int, ...],
) -> dict[str, Any]:
    if frame.empty:
        return {}

    latest = frame.iloc[-1]
    current: dict[str, Any] = {
        "latest_date": latest["date"].strftime("%Y-%m-%d"),
        "latest_close": float(latest["close"]),
    }
    distances: dict[int, float] = {}
    for dma in dma_windows:
        dma_value = latest.get(f"dma_{dma}", np.nan)
        distance = (float(latest["close"]) / float(dma_value) - 1.0) * 100.0 if pd.notna(dma_value) else np.nan
        slope = np.nan
        if f"dma_{dma}" in frame.columns and len(frame) > 5:
            prior = frame[f"dma_{dma}"].iloc[-6]
            if pd.notna(dma_value) and pd.notna(prior):
                slope = (float(dma_value) / float(prior) - 1.0) * 100.0
        current[f"dma_{dma}"] = dma_value
        current[f"distance_to_dma{dma}_pct"] = distance
        current[f"dma_{dma}_slope_5d_pct"] = slope
        distances[dma] = distance

    valid_distances = {dma: abs(value) for dma, value in distances.items() if np.isfinite(value)}
    if valid_distances:
        nearest_dma = min(valid_distances, key=valid_distances.get)
        current["nearest_dma"] = nearest_dma
        current["nearest_dma_distance_pct"] = distances[nearest_dma]
        current["abs_nearest_dma_distance_pct"] = abs(distances[nearest_dma])
    else:
        current["nearest_dma"] = ""
        current["nearest_dma_distance_pct"] = np.nan
        current["abs_nearest_dma_distance_pct"] = np.nan

    confirmed = pivots[pivots["confirmation_date"].notna()].copy() if not pivots.empty else pd.DataFrame()
    if not confirmed.empty:
        last_pivot = confirmed.iloc[-1]
        current["last_confirmed_pivot_type"] = last_pivot["type"]
        current["last_confirmed_pivot_structure"] = last_pivot["structure"]
        current["last_confirmed_pivot_date"] = last_pivot["date"].strftime("%Y-%m-%d")
        current["last_confirmed_pivot_price"] = float(last_pivot["price"])
        current["last_pivot_confirmation_date"] = last_pivot["confirmation_date"].strftime("%Y-%m-%d")

    lows = confirmed[confirmed["type"] == "LOW"] if not confirmed.empty else pd.DataFrame()
    if not lows.empty:
        last_low = lows.iloc[-1]
        current["last_confirmed_low_date"] = last_low["date"].strftime("%Y-%m-%d")
        current["last_confirmed_low_price"] = float(last_low["price"])
        current["rebound_from_last_low_pct"] = (float(latest["close"]) / float(last_low["price"]) - 1.0) * 100.0

    highs = confirmed[confirmed["type"] == "HIGH"] if not confirmed.empty else pd.DataFrame()
    if not highs.empty:
        last_high = highs.iloc[-1]
        current["last_confirmed_high_date"] = last_high["date"].strftime("%Y-%m-%d")
        current["last_confirmed_high_price"] = float(last_high["price"])
        current["drawdown_from_last_high_pct"] = (float(latest["close"]) / float(last_high["price"]) - 1.0) * 100.0

    return current


def _universe_symbols(universe: pd.DataFrame) -> list[str]:
    if universe.empty:
        return []
    symbol_column = next((column for column in ("Symbol", "symbol", "tradingsymbol", "Tradingsymbol") if column in universe.columns), "")
    if not symbol_column:
        return []
    symbols = universe[symbol_column].dropna().astype(str).str.upper().str.strip()
    return sorted(dict.fromkeys(symbol for symbol in symbols if symbol))


def _universe_metadata(universe: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if universe.empty:
        return {}
    symbol_column = next((column for column in ("Symbol", "symbol", "tradingsymbol", "Tradingsymbol") if column in universe.columns), "")
    if not symbol_column:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for _, row in universe.iterrows():
        symbol = str(row.get(symbol_column, "")).strip().upper()
        if not symbol:
            continue
        result[symbol] = {
            "name": row.get("Company Name", row.get("name", row.get("company_name", symbol))),
            "industry": row.get("Industry", row.get("industry", "")),
            "source_universe": row.get("source_universe", ""),
        }
    return result


def _scan_row(
    result: StockSignatureStudyResult,
    metadata: dict[str, Any],
    *,
    required_latest_date: pd.Timestamp | None = None,
) -> dict[str, Any]:
    summary = result.summary
    current = result.current
    data_status = _scan_data_status(result, required_latest_date=required_latest_date)
    best_dma = _best_dma(result.dma_summary)
    score_parts = _score_signature(result, data_status=data_status, best_dma=best_dma)
    return {
        "exchange": result.exchange,
        "symbol": result.symbol,
        "name": metadata.get("name") or result.symbol,
        "industry": metadata.get("industry") or "",
        "source_universe": metadata.get("source_universe") or "",
        "data_status": data_status,
        "analysis_start": summary.get("history_start", ""),
        "analysis_end": summary.get("history_end", ""),
        "latest_date": current.get("latest_date", ""),
        "latest_close": current.get("latest_close", np.nan),
        "confirmed_pivots": summary.get("confirmed_pivots", 0),
        "completed_bullish_cycles": summary.get("completed_bullish_cycles", 0),
        "swing_low_to_high_legs": summary.get("swing_low_to_high_legs", 0),
        "median_low_to_high_jump_pct": summary.get("median_low_to_high_jump_pct", np.nan),
        "p25_low_to_high_jump_pct": summary.get("p25_low_to_high_jump_pct", np.nan),
        "p75_low_to_high_jump_pct": summary.get("p75_low_to_high_jump_pct", np.nan),
        "median_low_to_high_sessions": summary.get("median_low_to_high_sessions", np.nan),
        "median_higher_highs_before_breakdown": summary.get("median_higher_highs_before_breakdown", np.nan),
        "median_higher_lows_before_breakdown": summary.get("median_higher_lows_before_breakdown", np.nan),
        "median_complete_low_to_peak_gain_pct": summary.get("median_complete_low_to_peak_gain_pct", np.nan),
        "nearest_dma": current.get("nearest_dma", ""),
        "nearest_dma_distance_pct": current.get("nearest_dma_distance_pct", np.nan),
        "abs_nearest_dma_distance_pct": current.get("abs_nearest_dma_distance_pct", np.nan),
        "dma_75": current.get("dma_75", np.nan),
        "distance_to_dma75_pct": current.get("distance_to_dma75_pct", np.nan),
        "dma_75_slope_5d_pct": current.get("dma_75_slope_5d_pct", np.nan),
        "dma_100": current.get("dma_100", np.nan),
        "distance_to_dma100_pct": current.get("distance_to_dma100_pct", np.nan),
        "dma_100_slope_5d_pct": current.get("dma_100_slope_5d_pct", np.nan),
        "dma_200": current.get("dma_200", np.nan),
        "distance_to_dma200_pct": current.get("distance_to_dma200_pct", np.nan),
        "dma_200_slope_5d_pct": current.get("dma_200_slope_5d_pct", np.nan),
        "last_confirmed_pivot_type": current.get("last_confirmed_pivot_type", ""),
        "last_confirmed_pivot_structure": current.get("last_confirmed_pivot_structure", ""),
        "last_confirmed_pivot_date": current.get("last_confirmed_pivot_date", ""),
        "last_confirmed_low_date": current.get("last_confirmed_low_date", ""),
        "last_confirmed_low_price": current.get("last_confirmed_low_price", np.nan),
        "rebound_from_last_low_pct": current.get("rebound_from_last_low_pct", np.nan),
        "last_confirmed_high_date": current.get("last_confirmed_high_date", ""),
        "last_confirmed_high_price": current.get("last_confirmed_high_price", np.nan),
        "drawdown_from_last_high_pct": current.get("drawdown_from_last_high_pct", np.nan),
        "best_dma": best_dma.get("nearest_dma", ""),
        "best_dma_rebounds": best_dma.get("rebounds", 0),
        "best_dma_median_jump_pct": best_dma.get("median_jump_pct", np.nan),
        "best_dma_median_sessions_to_high": best_dma.get("median_sessions_to_high", np.nan),
        **score_parts,
    }


def _scan_data_status(
    result: StockSignatureStudyResult,
    *,
    required_latest_date: pd.Timestamp | None = None,
) -> str:
    if result.daily.empty:
        return "EMPTY"
    if required_latest_date is not None:
        latest = _coerce_normalized_timestamp(result.current.get("latest_date"))
        if latest is None or latest != required_latest_date:
            return "STALE_DATE"
    if result.pivots.empty:
        return "NO_PIVOTS"
    if result.legs.empty:
        return "NO_LOW_HIGH_LEGS"
    return "READY"


def _coerce_normalized_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).normalize()


def _best_dma(dma_summary: pd.DataFrame) -> dict[str, Any]:
    if dma_summary.empty:
        return {"nearest_dma": "", "rebounds": 0}
    ranked = dma_summary.copy()
    ranked["median_jump_pct"] = pd.to_numeric(ranked["median_jump_pct"], errors="coerce")
    ranked["rebounds"] = pd.to_numeric(ranked["rebounds"], errors="coerce").fillna(0)
    ranked = ranked.sort_values(["median_jump_pct", "rebounds"], ascending=[False, False], na_position="last")
    return ranked.iloc[0].to_dict()


def _score_signature(
    result: StockSignatureStudyResult,
    *,
    data_status: str,
    best_dma: dict[str, Any],
) -> dict[str, Any]:
    if data_status != "READY":
        return {
            "historical_rank_score": 0.0,
            "sample_confidence_score": 0.0,
            "rebound_strength_score": 0.0,
            "current_setup_score": 0.0,
            "dma_affinity_score": 0.0,
            "rank_reason": data_status,
        }

    summary = result.summary
    current = result.current
    legs = float(summary.get("swing_low_to_high_legs") or 0)
    cycles = float(summary.get("completed_bullish_cycles") or 0)
    median_jump = _finite(summary.get("median_low_to_high_jump_pct"), 0.0)
    p75_jump = _finite(summary.get("p75_low_to_high_jump_pct"), median_jump)
    current_rebound = _finite(current.get("rebound_from_last_low_pct"), np.nan)
    nearest_distance = abs(_finite(current.get("nearest_dma_distance_pct"), np.nan))
    best_dma_rebounds = _finite(best_dma.get("rebounds"), 0.0)
    best_dma_median = _finite(best_dma.get("median_jump_pct"), 0.0)

    sample_confidence = min(25.0, legs * 4.0 + cycles * 3.0)
    rebound_strength = min(25.0, max(median_jump, 0.0) / 20.0 * 25.0)

    current_setup = 0.0
    if np.isfinite(current_rebound):
        if current_rebound <= 0:
            current_setup = 8.0
        elif median_jump > 0 and current_rebound <= median_jump:
            current_setup = 25.0 - (current_rebound / median_jump * 8.0)
        elif p75_jump > 0 and current_rebound <= p75_jump:
            current_setup = 12.0
        else:
            current_setup = 4.0
    if str(current.get("last_confirmed_pivot_type", "")).upper() == "LOW":
        current_setup += 4.0
    current_setup = min(25.0, current_setup)

    dma_affinity = min(15.0, best_dma_rebounds * 4.0 + max(best_dma_median, 0.0) / 20.0 * 8.0)
    if np.isfinite(nearest_distance):
        dma_affinity += max(0.0, 10.0 - nearest_distance * 2.0)
    dma_affinity = min(25.0, dma_affinity)

    total = sample_confidence + rebound_strength + current_setup + dma_affinity
    reason = (
        f"Median rebound {median_jump:.1f}%, "
        f"{int(legs)} legs, "
        f"current rebound {current_rebound:.1f}%"
        if np.isfinite(current_rebound)
        else f"Median rebound {median_jump:.1f}%, {int(legs)} legs"
    )
    if best_dma.get("nearest_dma"):
        reason += f", best DMA {int(float(best_dma['nearest_dma']))}"

    return {
        "historical_rank_score": round(float(total), 2),
        "sample_confidence_score": round(float(sample_confidence), 2),
        "rebound_strength_score": round(float(rebound_strength), 2),
        "current_setup_score": round(float(current_setup), 2),
        "dma_affinity_score": round(float(dma_affinity), 2),
        "rank_reason": reason,
    }


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if np.isfinite(numeric) else default


def _emit_progress(progress_callback: Any | None, **payload: Any) -> None:
    if progress_callback is None:
        return
    progress_callback(payload)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
