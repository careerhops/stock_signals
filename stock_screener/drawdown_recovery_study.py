from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from stock_screener.data.storage import Storage


DEFAULT_HISTORY_MONTHS = 24
DEFAULT_PRIOR_HIGH_LOOKBACK_SESSIONS = 420
DEFAULT_MIN_DRAWDOWN_FLOOR_PCT = 20.0
DEFAULT_TARGET_RECOVERY_PCT = 10.0
DEFAULT_DMA_WINDOW = 200
DEFAULT_VOLUME_BASELINE_SESSIONS = 60
DEFAULT_MIN_VOLUME_MULTIPLE = 3.0
DEFAULT_MIN_DMA_IMPROVEMENT_PCT_POINTS = 5.0
DEFAULT_MAX_CURRENT_AGE_SESSIONS = 90
DEFAULT_TROUGH_ORDER = 3
RECOVERY_WINDOW_OPTIONS = (5, 10, 15, 20, 30, 40, 60, 90)
DRAWDOWN_RECOVERY_LOGIC_VERSION = "deep_drawdown_dma_recovery_v1"


@dataclass(frozen=True)
class DrawdownRecoveryResult:
    summary: dict[str, Any]
    current_candidates: pd.DataFrame
    stock_stats: pd.DataFrame
    events: pd.DataFrame


def run_drawdown_recovery_study(
    storage: Storage,
    universe: pd.DataFrame,
    *,
    exchange: str = "NSE",
    universe_name: str = "NIFTY100",
    history_months: int = DEFAULT_HISTORY_MONTHS,
    prior_high_lookback_sessions: int = DEFAULT_PRIOR_HIGH_LOOKBACK_SESSIONS,
    min_drawdown_floor_pct: float = DEFAULT_MIN_DRAWDOWN_FLOOR_PCT,
    target_recovery_pct: float = DEFAULT_TARGET_RECOVERY_PCT,
    dma_window: int = DEFAULT_DMA_WINDOW,
    volume_baseline_sessions: int = DEFAULT_VOLUME_BASELINE_SESSIONS,
    min_volume_multiple: float = DEFAULT_MIN_VOLUME_MULTIPLE,
    min_dma_improvement_pct_points: float = DEFAULT_MIN_DMA_IMPROVEMENT_PCT_POINTS,
    max_current_age_sessions: int = DEFAULT_MAX_CURRENT_AGE_SESSIONS,
    trough_order: int = DEFAULT_TROUGH_ORDER,
    as_of_date: Any | None = None,
    required_latest_date: Any | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> DrawdownRecoveryResult:
    symbols = _universe_symbols(universe)
    metadata = _universe_metadata(universe)
    required_latest_ts = _coerce_normalized_timestamp(required_latest_date)
    rows: list[dict[str, Any]] = []
    event_frames: list[pd.DataFrame] = []
    latest_dates: list[pd.Timestamp] = []

    _emit_progress(
        progress_callback,
        phase="Scanning deep drawdown recoveries",
        completed=0,
        total=len(symbols),
        current_symbol="",
        current_exchange=exchange,
    )

    for index, symbol in enumerate(symbols, start=1):
        daily = storage.load_candles(exchange, symbol, "1D")
        stock_events, row = analyze_drawdown_recovery(
            daily,
            exchange=exchange,
            symbol=symbol,
            history_months=history_months,
            prior_high_lookback_sessions=prior_high_lookback_sessions,
            min_drawdown_floor_pct=min_drawdown_floor_pct,
            target_recovery_pct=target_recovery_pct,
            dma_window=dma_window,
            volume_baseline_sessions=volume_baseline_sessions,
            min_volume_multiple=min_volume_multiple,
            min_dma_improvement_pct_points=min_dma_improvement_pct_points,
            max_current_age_sessions=max_current_age_sessions,
            trough_order=trough_order,
            as_of_date=as_of_date,
            required_latest_date=required_latest_ts,
        )
        row.update(metadata.get(symbol, {}))
        rows.append(row)
        if not stock_events.empty:
            for key, value in metadata.get(symbol, {}).items():
                if key not in stock_events.columns:
                    stock_events[key] = value
            event_frames.append(stock_events)
        latest_date = pd.to_datetime(row.get("latest_date"), errors="coerce")
        if pd.notna(latest_date):
            latest_dates.append(latest_date.normalize())
        _emit_progress(
            progress_callback,
            phase="Scanning deep drawdown recoveries",
            completed=index,
            total=len(symbols),
            current_symbol=symbol,
            current_exchange=exchange,
        )

    stock_stats = pd.DataFrame(rows)
    if not stock_stats.empty:
        stock_stats = stock_stats.sort_values(
            ["setup_score", "historical_success_rate", "symbol"],
            ascending=[False, False, True],
            na_position="last",
        ).reset_index(drop=True)
        stock_stats["rank"] = np.arange(1, len(stock_stats) + 1)

    events = pd.concat(event_frames, ignore_index=True) if event_frames else _empty_events()
    if not events.empty:
        events = events.sort_values(
            ["event_match_score", "low_date", "symbol"],
            ascending=[False, False, True],
            na_position="last",
        ).reset_index(drop=True)

    if stock_stats.empty:
        current_candidates = pd.DataFrame()
    else:
        candidate_mask = (
            stock_stats.get("data_status", pd.Series("", index=stock_stats.index)).eq("READY")
            & stock_stats.get("candidate_pass", pd.Series(False, index=stock_stats.index)).fillna(False).astype(bool)
        )
        current_candidates = stock_stats.loc[candidate_mask].copy()

    latest_market_date = max(latest_dates).strftime("%Y-%m-%d") if latest_dates else ""
    ready_count = int((stock_stats["data_status"] == "READY").sum()) if not stock_stats.empty and "data_status" in stock_stats.columns else 0
    current_count = int(len(current_candidates))
    historical_match_count = (
        int(events["event_pass"].fillna(False).astype(bool).sum())
        if not events.empty and "event_pass" in events.columns
        else 0
    )
    summary = {
        "logic_version": DRAWDOWN_RECOVERY_LOGIC_VERSION,
        "exchange": str(exchange).upper(),
        "universe": str(universe_name or "CUSTOM").upper(),
        "symbols_requested": int(len(symbols)),
        "symbols_processed": int(len(stock_stats)),
        "stocks_with_ready_history": ready_count,
        "current_candidates": current_count,
        "historical_matching_events": historical_match_count,
        "latest_market_date": latest_market_date,
        "history_months": max(int(history_months), 1),
        "prior_high_lookback_sessions": max(int(prior_high_lookback_sessions), 20),
        "min_drawdown_floor_pct": max(float(min_drawdown_floor_pct), 0.0),
        "target_recovery_pct": max(float(target_recovery_pct), 0.0),
        "dma_window": max(int(dma_window), 2),
        "volume_baseline_sessions": max(int(volume_baseline_sessions), 5),
        "min_volume_multiple": max(float(min_volume_multiple), 0.0),
        "min_dma_improvement_pct_points": max(float(min_dma_improvement_pct_points), 0.0),
        "max_current_age_sessions": max(int(max_current_age_sessions), 1),
        "trough_order": max(int(trough_order), 1),
        "requested_as_of_date": "" if as_of_date is None else str(as_of_date),
        "required_latest_date": "" if required_latest_ts is None else required_latest_ts.strftime("%Y-%m-%d"),
    }
    return DrawdownRecoveryResult(
        summary=summary,
        current_candidates=current_candidates,
        stock_stats=stock_stats,
        events=events,
    )


def analyze_drawdown_recovery(
    daily: pd.DataFrame,
    *,
    exchange: str = "NSE",
    symbol: str,
    history_months: int = DEFAULT_HISTORY_MONTHS,
    prior_high_lookback_sessions: int = DEFAULT_PRIOR_HIGH_LOOKBACK_SESSIONS,
    min_drawdown_floor_pct: float = DEFAULT_MIN_DRAWDOWN_FLOOR_PCT,
    target_recovery_pct: float = DEFAULT_TARGET_RECOVERY_PCT,
    dma_window: int = DEFAULT_DMA_WINDOW,
    volume_baseline_sessions: int = DEFAULT_VOLUME_BASELINE_SESSIONS,
    min_volume_multiple: float = DEFAULT_MIN_VOLUME_MULTIPLE,
    min_dma_improvement_pct_points: float = DEFAULT_MIN_DMA_IMPROVEMENT_PCT_POINTS,
    max_current_age_sessions: int = DEFAULT_MAX_CURRENT_AGE_SESSIONS,
    trough_order: int = DEFAULT_TROUGH_ORDER,
    as_of_date: Any | None = None,
    required_latest_date: Any | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    exchange_text = str(exchange).upper()
    symbol_text = str(symbol).upper()
    frame = _prepare_daily(daily)
    frame = _clip_analysis_window(
        frame,
        history_months=max(int(history_months), 1),
        as_of_date=as_of_date,
    )
    row = _base_stock_row(exchange_text, symbol_text)
    if frame.empty:
        row["data_status"] = "NO_DATA"
        row["reason"] = "No daily candles available on or before the selected date."
        return _empty_events(), row

    latest_date = frame.iloc[-1]["date"].normalize()
    row.update(
        {
            "history_start": frame.iloc[0]["date"].strftime("%Y-%m-%d"),
            "history_end": latest_date.strftime("%Y-%m-%d"),
            "latest_date": latest_date.strftime("%Y-%m-%d"),
            "latest_close": float(frame.iloc[-1]["close"]),
        }
    )
    required_latest_ts = _coerce_normalized_timestamp(required_latest_date)
    if required_latest_ts is not None and latest_date != required_latest_ts:
        row["data_status"] = "STALE_DATE"
        row["reason"] = f"Latest candle is {latest_date.strftime('%Y-%m-%d')}, expected {required_latest_ts.strftime('%Y-%m-%d')}."
        return _empty_events(), row

    min_rows = max(min(int(prior_high_lookback_sessions), 80), int(volume_baseline_sessions) + 10, int(dma_window) + 5)
    if len(frame) < min_rows:
        row["data_status"] = "INSUFFICIENT_HISTORY"
        row["reason"] = f"Need at least {min_rows} daily candles after the selected date filter."
        return _empty_events(), row

    frame = _add_features(
        frame,
        dma_window=max(int(dma_window), 2),
        volume_baseline_sessions=max(int(volume_baseline_sessions), 5),
    )
    events = _detect_drawdown_events(
        frame,
        exchange=exchange_text,
        symbol=symbol_text,
        prior_high_lookback_sessions=max(int(prior_high_lookback_sessions), 20),
        min_drawdown_floor_pct=max(float(min_drawdown_floor_pct), 0.0),
        target_recovery_pct=max(float(target_recovery_pct), 0.0),
        dma_window=max(int(dma_window), 2),
        min_volume_multiple=max(float(min_volume_multiple), 0.0),
        min_dma_improvement_pct_points=max(float(min_dma_improvement_pct_points), 0.0),
        trough_order=max(int(trough_order), 1),
    )
    if events.empty:
        row["data_status"] = "NO_DRAWDOWN_EVENTS"
        row["reason"] = "No local drawdown trough met the minimum fall threshold."
        return events, row

    optimized = _optimized_thresholds(
        events,
        min_drawdown_floor_pct=max(float(min_drawdown_floor_pct), 0.0),
        target_recovery_pct=max(float(target_recovery_pct), 0.0),
    )
    current = _current_setup_row(
        frame,
        events,
        optimized,
        min_volume_multiple=max(float(min_volume_multiple), 0.0),
        min_dma_improvement_pct_points=max(float(min_dma_improvement_pct_points), 0.0),
        max_current_age_sessions=max(int(max_current_age_sessions), 1),
    )
    row.update(optimized)
    row.update(current)
    row["data_status"] = "READY"
    row["reason"] = _setup_reason(row)
    return events, row


def save_drawdown_recovery_outputs(result: DrawdownRecoveryResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(_json_safe(result.summary), indent=2), encoding="utf-8")
    result.current_candidates.to_csv(output_dir / "current_candidates.csv", index=False)
    result.stock_stats.to_csv(output_dir / "stock_stats.csv", index=False)
    result.events.to_csv(output_dir / "historical_events.csv", index=False)


def load_drawdown_recovery_outputs(output_dir: Path) -> DrawdownRecoveryResult:
    summary_path = output_dir / "summary.json"
    summary: dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}
    return DrawdownRecoveryResult(
        summary=summary,
        current_candidates=_read_csv(output_dir / "current_candidates.csv"),
        stock_stats=_read_csv(output_dir / "stock_stats.csv"),
        events=_read_csv(output_dir / "historical_events.csv"),
    )


def _prepare_daily(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    frame = daily.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    required = ["date", "open", "high", "low", "close"]
    if any(column not in frame.columns for column in required):
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", format="mixed")
    for column in required[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "volume" in frame.columns:
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    else:
        frame["volume"] = np.nan
    return (
        frame.dropna(subset=required)
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )


def _clip_analysis_window(
    frame: pd.DataFrame,
    *,
    history_months: int,
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
    start_date = end_date - pd.DateOffset(months=max(int(history_months), 1))
    return working[working["date"] >= start_date].reset_index(drop=True)


def _add_features(
    frame: pd.DataFrame,
    *,
    dma_window: int,
    volume_baseline_sessions: int,
) -> pd.DataFrame:
    enriched = frame.copy().reset_index(drop=True)
    enriched[f"dma_{dma_window}"] = enriched["close"].rolling(dma_window, min_periods=dma_window).mean()
    enriched["dma_distance_pct"] = (enriched["close"] / enriched[f"dma_{dma_window}"] - 1.0) * 100.0
    enriched["volume_median_baseline"] = (
        enriched["volume"].shift(1).rolling(volume_baseline_sessions, min_periods=max(5, volume_baseline_sessions // 2)).median()
    )
    return enriched


def _detect_drawdown_events(
    frame: pd.DataFrame,
    *,
    exchange: str,
    symbol: str,
    prior_high_lookback_sessions: int,
    min_drawdown_floor_pct: float,
    target_recovery_pct: float,
    dma_window: int,
    min_volume_multiple: float,
    min_dma_improvement_pct_points: float,
    trough_order: int,
) -> pd.DataFrame:
    if frame.empty:
        return _empty_events()

    working = frame.copy().reset_index(drop=True)
    prior_high_prices = np.full(len(working), np.nan)
    prior_high_indexes = np.full(len(working), np.nan)
    min_prior_rows = min(60, max(int(prior_high_lookback_sessions) // 4, 20))
    for idx in range(len(working)):
        start = max(0, idx - int(prior_high_lookback_sessions))
        prior = working.iloc[start:idx]
        if len(prior) < min_prior_rows:
            continue
        high_position = int(prior["high"].idxmax())
        prior_high_prices[idx] = float(working.loc[high_position, "high"])
        prior_high_indexes[idx] = high_position
    working["prior_high_price"] = prior_high_prices
    working["prior_high_idx"] = prior_high_indexes
    working["drawdown_pct"] = (working["low"] / working["prior_high_price"] - 1.0) * 100.0

    pivot_window = int(trough_order) * 2 + 1
    local_drawdown_min = working["drawdown_pct"].rolling(pivot_window, center=True, min_periods=1).min()
    trough_mask = (
        working["drawdown_pct"].notna()
        & (working["drawdown_pct"] <= -abs(float(min_drawdown_floor_pct)))
        & (working["drawdown_pct"] <= local_drawdown_min + 1e-9)
    )
    rows: list[dict[str, Any]] = []
    max_window = max(RECOVERY_WINDOW_OPTIONS)
    for idx in working.index[trough_mask]:
        low_idx = int(idx)
        prior_high_idx = _finite_int(working.loc[low_idx, "prior_high_idx"])
        if prior_high_idx is None or prior_high_idx >= low_idx:
            continue
        low = working.loc[low_idx]
        prior_high = working.loc[prior_high_idx]
        low_price = _finite_float(low.get("low"))
        high_price = _finite_float(prior_high.get("high"))
        if low_price is None or high_price is None or low_price <= 0 or high_price <= 0:
            continue
        target_price = low_price * (1.0 + float(target_recovery_pct) / 100.0)
        future = working.iloc[low_idx + 1 : min(len(working), low_idx + max_window + 1)]
        target_row = None
        if not future.empty:
            reached = future[future["close"] >= target_price]
            if not reached.empty:
                target_row = reached.iloc[0]
        sessions_to_target = None
        target_date = ""
        target_close = np.nan
        target_dma_distance = np.nan
        if target_row is not None:
            target_idx = int(target_row.name)
            sessions_to_target = target_idx - low_idx
            target_date = pd.Timestamp(target_row["date"]).strftime("%Y-%m-%d")
            target_close = _finite_float(target_row.get("close")) or np.nan
            target_dma_distance = _finite_float(target_row.get("dma_distance_pct")) or np.nan

        low_dma_distance = _finite_float(low.get("dma_distance_pct"))
        max_volume_multiple = _window_volume_multiple(
            working,
            start_idx=low_idx + 1,
            end_idx=min(len(working) - 1, low_idx + max_window),
            baseline=_finite_float(low.get("volume_median_baseline")),
        )
        dma_improvement_to_target = (
            target_dma_distance - low_dma_distance
            if _is_finite(target_dma_distance) and low_dma_distance is not None
            else np.nan
        )
        target_reached = sessions_to_target is not None
        event_pass = bool(
            target_reached
            and max_volume_multiple >= float(min_volume_multiple)
            and _is_finite(dma_improvement_to_target)
            and dma_improvement_to_target >= float(min_dma_improvement_pct_points)
        )
        drawdown_pct = (low_price / high_price - 1.0) * 100.0
        drawdown_abs_pct = abs(drawdown_pct)
        rows.append(
            {
                "exchange": exchange,
                "symbol": symbol,
                "prior_high_date": pd.Timestamp(prior_high["date"]).strftime("%Y-%m-%d"),
                "prior_high_price": high_price,
                "low_date": pd.Timestamp(low["date"]).strftime("%Y-%m-%d"),
                "low_price": low_price,
                "drawdown_pct": drawdown_pct,
                "drawdown_abs_pct": drawdown_abs_pct,
                "fall_calendar_days": int((pd.Timestamp(low["date"]).normalize() - pd.Timestamp(prior_high["date"]).normalize()).days),
                "fall_sessions": int(low_idx - prior_high_idx),
                "target_recovery_pct": float(target_recovery_pct),
                "target_price": target_price,
                "target_reached": target_reached,
                "target_date": target_date,
                "target_close": target_close,
                "sessions_to_target": sessions_to_target if sessions_to_target is not None else np.nan,
                "dma_window": int(dma_window),
                "dma_distance_at_low_pct": low_dma_distance if low_dma_distance is not None else np.nan,
                "dma_distance_at_target_pct": target_dma_distance,
                "dma_improvement_to_target_pct_points": dma_improvement_to_target,
                "volume_baseline": _finite_float(low.get("volume_median_baseline")) or np.nan,
                "max_volume_multiple_90d": max_volume_multiple,
                "volume_spike_3x": bool(max_volume_multiple >= float(min_volume_multiple)),
                "event_pass": event_pass,
                "event_match_score": _event_match_score(
                    drawdown_abs_pct=drawdown_abs_pct,
                    target_reached=target_reached,
                    sessions_to_target=sessions_to_target,
                    dma_improvement_pct_points=dma_improvement_to_target,
                    max_volume_multiple=max_volume_multiple,
                    target_recovery_pct=float(target_recovery_pct),
                    min_volume_multiple=float(min_volume_multiple),
                    min_dma_improvement_pct_points=float(min_dma_improvement_pct_points),
                ),
                "_low_idx": low_idx,
            }
        )

    if not rows:
        return _empty_events()
    return pd.DataFrame(rows, columns=_event_columns(include_internal=True))


def _optimized_thresholds(
    events: pd.DataFrame,
    *,
    min_drawdown_floor_pct: float,
    target_recovery_pct: float,
) -> dict[str, Any]:
    if events.empty:
        return {
            "optimized_drawdown_pct": float(min_drawdown_floor_pct),
            "optimized_recovery_sessions": 30,
            "historical_events": 0,
            "successful_events": 0,
            "historical_success_rate": 0.0,
        }
    successful = events[events["target_reached"].fillna(False).astype(bool)].copy()
    if successful.empty:
        optimized_drawdown = max(float(min_drawdown_floor_pct), float(events["drawdown_abs_pct"].quantile(0.75)))
        optimized_window = 30
    else:
        optimized_drawdown = max(float(min_drawdown_floor_pct), float(successful["drawdown_abs_pct"].median()))
        median_sessions = float(pd.to_numeric(successful["sessions_to_target"], errors="coerce").median())
        optimized_window = _nearest_recovery_window(median_sessions)
    success_rate = float(len(successful) / len(events)) if len(events) else 0.0
    return {
        "optimized_drawdown_pct": optimized_drawdown,
        "optimized_recovery_sessions": int(optimized_window),
        "historical_events": int(len(events)),
        "successful_events": int(len(successful)),
        "historical_success_rate": success_rate,
        "median_success_drawdown_pct": float(successful["drawdown_abs_pct"].median()) if not successful.empty else np.nan,
        "median_sessions_to_10pct": float(successful["sessions_to_target"].median()) if not successful.empty else np.nan,
        "target_recovery_pct": float(target_recovery_pct),
    }


def _current_setup_row(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    optimized: dict[str, Any],
    *,
    min_volume_multiple: float,
    min_dma_improvement_pct_points: float,
    max_current_age_sessions: int,
) -> dict[str, Any]:
    latest = frame.iloc[-1]
    latest_idx = len(frame) - 1
    current = events.sort_values("_low_idx").iloc[-1]
    low_idx = int(current["_low_idx"])
    low_dma_distance = _finite_float(current.get("dma_distance_at_low_pct"))
    current_dma_distance = _finite_float(latest.get("dma_distance_pct"))
    sessions_since_low = int(latest_idx - low_idx)
    low_price = float(current["low_price"])
    latest_close = float(latest["close"])
    current_recovery_pct = (latest_close / low_price - 1.0) * 100.0 if low_price > 0 else np.nan
    dma_improvement_current = (
        current_dma_distance - low_dma_distance
        if current_dma_distance is not None and low_dma_distance is not None
        else np.nan
    )
    optimized_y = int(optimized.get("optimized_recovery_sessions") or 30)
    window_end = min(latest_idx, low_idx + optimized_y)
    volume_multiple_current = _window_volume_multiple(
        frame,
        start_idx=low_idx + 1,
        end_idx=window_end,
        baseline=_finite_float(current.get("volume_baseline")),
    )
    target_reached_within_y = bool(
        current.get("target_reached")
        and _is_finite(current.get("sessions_to_target"))
        and float(current["sessions_to_target"]) <= optimized_y
    )
    drawdown_abs_pct = float(current["drawdown_abs_pct"])
    optimized_x = float(optimized.get("optimized_drawdown_pct") or 0.0)
    candidate_pass = bool(
        drawdown_abs_pct >= optimized_x
        and target_reached_within_y
        and sessions_since_low <= int(max_current_age_sessions)
        and _is_finite(dma_improvement_current)
        and dma_improvement_current >= float(min_dma_improvement_pct_points)
        and volume_multiple_current >= float(min_volume_multiple)
    )
    setup_score = _setup_score(
        drawdown_abs_pct=drawdown_abs_pct,
        optimized_drawdown_pct=optimized_x,
        current_recovery_pct=current_recovery_pct,
        target_recovery_pct=float(optimized.get("target_recovery_pct") or DEFAULT_TARGET_RECOVERY_PCT),
        dma_improvement_pct_points=dma_improvement_current,
        min_dma_improvement_pct_points=float(min_dma_improvement_pct_points),
        volume_multiple=volume_multiple_current,
        min_volume_multiple=float(min_volume_multiple),
        target_reached_within_y=target_reached_within_y,
        historical_success_rate=float(optimized.get("historical_success_rate") or 0.0),
    )
    return {
        "prior_high_date": current["prior_high_date"],
        "prior_high_price": float(current["prior_high_price"]),
        "low_date": current["low_date"],
        "low_price": low_price,
        "drawdown_pct": float(current["drawdown_pct"]),
        "drawdown_abs_pct": drawdown_abs_pct,
        "fall_calendar_days": int(current["fall_calendar_days"]),
        "fall_sessions": int(current["fall_sessions"]),
        "sessions_since_low": sessions_since_low,
        "current_recovery_pct": current_recovery_pct,
        "target_reached_date": current.get("target_date", ""),
        "sessions_to_target": float(current["sessions_to_target"]) if _is_finite(current.get("sessions_to_target")) else np.nan,
        "target_reached_within_optimized_y": target_reached_within_y,
        "dma_window": int(current["dma_window"]),
        "dma_distance_at_low_pct": low_dma_distance if low_dma_distance is not None else np.nan,
        "current_dma_distance_pct": current_dma_distance if current_dma_distance is not None else np.nan,
        "dma_improvement_current_pct_points": dma_improvement_current,
        "max_volume_multiple_to_current": volume_multiple_current,
        "min_volume_multiple": float(min_volume_multiple),
        "min_dma_improvement_pct_points": float(min_dma_improvement_pct_points),
        "candidate_pass": candidate_pass,
        "setup_score": setup_score,
    }


def _event_match_score(
    *,
    drawdown_abs_pct: float,
    target_reached: bool,
    sessions_to_target: int | None,
    dma_improvement_pct_points: float,
    max_volume_multiple: float,
    target_recovery_pct: float,
    min_volume_multiple: float,
    min_dma_improvement_pct_points: float,
) -> float:
    drawdown_score = min(30.0, max(float(drawdown_abs_pct), 0.0) / 60.0 * 30.0)
    speed_score = 0.0
    if target_reached and sessions_to_target is not None and sessions_to_target > 0:
        speed_score = max(0.0, 20.0 * (1.0 - min(float(sessions_to_target), 90.0) / 90.0))
    dma_score = min(
        25.0,
        max(float(dma_improvement_pct_points), 0.0)
        / max(float(min_dma_improvement_pct_points), 1.0)
        * 12.5,
    ) if _is_finite(dma_improvement_pct_points) else 0.0
    volume_score = min(
        15.0,
        max(float(max_volume_multiple), 0.0)
        / max(float(min_volume_multiple), 1.0)
        * 10.0,
    ) if _is_finite(max_volume_multiple) else 0.0
    target_score = 10.0 if target_reached and float(target_recovery_pct) > 0 else 0.0
    return round(drawdown_score + speed_score + dma_score + volume_score + target_score, 2)


def _setup_score(
    *,
    drawdown_abs_pct: float,
    optimized_drawdown_pct: float,
    current_recovery_pct: float,
    target_recovery_pct: float,
    dma_improvement_pct_points: float,
    min_dma_improvement_pct_points: float,
    volume_multiple: float,
    min_volume_multiple: float,
    target_reached_within_y: bool,
    historical_success_rate: float,
) -> float:
    drawdown_score = min(25.0, max(float(drawdown_abs_pct), 0.0) / max(float(optimized_drawdown_pct), 1.0) * 20.0)
    recovery_score = min(25.0, max(float(current_recovery_pct), 0.0) / max(float(target_recovery_pct), 1.0) * 25.0)
    dma_score = min(
        20.0,
        max(float(dma_improvement_pct_points), 0.0)
        / max(float(min_dma_improvement_pct_points), 1.0)
        * 15.0,
    ) if _is_finite(dma_improvement_pct_points) else 0.0
    volume_score = min(
        15.0,
        max(float(volume_multiple), 0.0)
        / max(float(min_volume_multiple), 1.0)
        * 15.0,
    ) if _is_finite(volume_multiple) else 0.0
    history_score = min(10.0, max(float(historical_success_rate), 0.0) * 10.0)
    timing_score = 5.0 if target_reached_within_y else 0.0
    return round(drawdown_score + recovery_score + dma_score + volume_score + history_score + timing_score, 2)


def _window_volume_multiple(
    frame: pd.DataFrame,
    *,
    start_idx: int,
    end_idx: int,
    baseline: float | None,
) -> float:
    if baseline is None or baseline <= 0 or start_idx > end_idx:
        return np.nan
    window = frame.iloc[max(start_idx, 0) : min(end_idx, len(frame) - 1) + 1]
    if window.empty:
        return np.nan
    max_volume = _finite_float(window["volume"].max())
    if max_volume is None:
        return np.nan
    return float(max_volume / baseline)


def _setup_reason(row: dict[str, Any]) -> str:
    checks = []
    if bool(row.get("candidate_pass")):
        return "Deep fall recovered by target, DMA distance improved, and volume expanded inside optimized window."
    if float(row.get("drawdown_abs_pct") or 0.0) < float(row.get("optimized_drawdown_pct") or 0.0):
        checks.append("fall below stock-specific threshold")
    if not bool(row.get("target_reached_within_optimized_y")):
        checks.append("target recovery not reached inside optimized sessions")
    if (
        not _is_finite(row.get("dma_improvement_current_pct_points"))
        or float(row.get("dma_improvement_current_pct_points") or 0.0)
        < float(row.get("min_dma_improvement_pct_points") or 0.0)
    ):
        checks.append("DMA distance improvement below threshold")
    if _is_finite(row.get("max_volume_multiple_to_current")) and _is_finite(row.get("min_volume_multiple")):
        if float(row.get("max_volume_multiple_to_current") or 0.0) < float(row.get("min_volume_multiple") or 0.0):
            checks.append("volume expansion below threshold")
    return "; ".join(checks) if checks else "Ready but not a current pass."


def _nearest_recovery_window(value: float) -> int:
    if not _is_finite(value):
        return 30
    for window in RECOVERY_WINDOW_OPTIONS:
        if value <= window:
            return int(window)
    return int(RECOVERY_WINDOW_OPTIONS[-1])


def _base_stock_row(exchange: str, symbol: str) -> dict[str, Any]:
    return {
        "exchange": exchange,
        "symbol": symbol,
        "name": symbol,
        "industry": "",
        "source_universe": "",
        "data_status": "NOT_RUN",
        "reason": "",
        "history_start": "",
        "history_end": "",
        "latest_date": "",
        "latest_close": np.nan,
        "prior_high_date": "",
        "prior_high_price": np.nan,
        "low_date": "",
        "low_price": np.nan,
        "drawdown_pct": np.nan,
        "drawdown_abs_pct": np.nan,
        "fall_calendar_days": np.nan,
        "fall_sessions": np.nan,
        "optimized_drawdown_pct": np.nan,
        "optimized_recovery_sessions": np.nan,
        "target_recovery_pct": np.nan,
        "sessions_since_low": np.nan,
        "current_recovery_pct": np.nan,
        "target_reached_date": "",
        "sessions_to_target": np.nan,
        "target_reached_within_optimized_y": False,
        "dma_window": np.nan,
        "dma_distance_at_low_pct": np.nan,
        "current_dma_distance_pct": np.nan,
        "dma_improvement_current_pct_points": np.nan,
        "max_volume_multiple_to_current": np.nan,
        "historical_events": 0,
        "successful_events": 0,
        "historical_success_rate": 0.0,
        "candidate_pass": False,
        "setup_score": 0.0,
    }


def _universe_symbols(universe: pd.DataFrame) -> list[str]:
    if universe.empty:
        return []
    symbol_column = next((column for column in ("Symbol", "symbol", "tradingsymbol") if column in universe.columns), "")
    if not symbol_column:
        return []
    symbols = universe[symbol_column].dropna().astype(str).str.upper().str.strip()
    return sorted(dict.fromkeys(symbol for symbol in symbols if symbol))


def _universe_metadata(universe: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if universe.empty:
        return {}
    metadata: dict[str, dict[str, Any]] = {}
    symbol_column = next((column for column in ("Symbol", "symbol", "tradingsymbol") if column in universe.columns), "")
    if not symbol_column:
        return metadata
    for _, row in universe.iterrows():
        symbol = str(row.get(symbol_column, "")).strip().upper()
        if not symbol:
            continue
        metadata[symbol] = {
            "name": str(row.get("Company Name", row.get("name", symbol)) or symbol),
            "industry": str(row.get("Industry", row.get("industry", "")) or ""),
            "source_universe": str(row.get("source_universe", "") or ""),
        }
    return metadata


def _emit_progress(
    progress_callback: Callable[[dict[str, Any]], None] | None,
    **payload: Any,
) -> None:
    if progress_callback is not None:
        progress_callback(payload)


def _coerce_normalized_timestamp(value: Any | None) -> pd.Timestamp | None:
    if value is None or str(value).strip() == "":
        return None
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return pd.Timestamp(timestamp).normalize()


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _finite_int(value: Any) -> int | None:
    number = _finite_float(value)
    if number is None:
        return None
    return int(number)


def _is_finite(value: Any) -> bool:
    return _finite_float(value) is not None


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=_event_columns(include_internal=False))


def _event_columns(*, include_internal: bool) -> list[str]:
    columns = [
        "exchange",
        "symbol",
        "prior_high_date",
        "prior_high_price",
        "low_date",
        "low_price",
        "drawdown_pct",
        "drawdown_abs_pct",
        "fall_calendar_days",
        "fall_sessions",
        "target_recovery_pct",
        "target_price",
        "target_reached",
        "target_date",
        "target_close",
        "sessions_to_target",
        "dma_window",
        "dma_distance_at_low_pct",
        "dma_distance_at_target_pct",
        "dma_improvement_to_target_pct_points",
        "volume_baseline",
        "max_volume_multiple_90d",
        "volume_spike_3x",
        "event_pass",
        "event_match_score",
    ]
    if include_internal:
        columns.append("_low_idx")
    return columns


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
