from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from stock_screener.data.storage import Storage
from stock_screener.knox_envelope_pair_backtest import _candidate_symbols
from stock_screener.knox_envelope_study import (
    DEFAULT_ENVELOPE_LENGTH,
    DEFAULT_ENVELOPE_MA_TYPE,
    DEFAULT_ENVELOPE_PERCENT,
    DEFAULT_KNOX_LOOKBACK,
    DEFAULT_MOMENTUM_LENGTH,
    DEFAULT_RSI_LENGTH,
    calculate_knox_envelope,
)
from stock_screener.weekly_buy_tracker_study import _emit_progress, _load_name_map


DEFAULT_PROXIMITY_PCT = 3.0
DEFAULT_RECENT_ENDPOINT_BARS = 5
DEFAULT_ROUND_TRIP_COST_PCT = 0.35
RECOVERY_WINDOWS = (5, 10, 20, 40, 60, 120)
KNOX_RECOVERY_LOGIC_VERSION = "knox_endpoint_drop_target_v3"


@dataclass(frozen=True)
class KnoxRecoveryStudyResult:
    summary: dict[str, Any]
    current_candidates: pd.DataFrame
    stock_stats: pd.DataFrame
    events: pd.DataFrame


def backtest_knox_recovery_frame(
    calculated: pd.DataFrame,
    *,
    symbol: str,
    exchange: str = "NSE",
    name: str = "",
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    envelope_proximity_pct: float = DEFAULT_PROXIMITY_PCT,
    round_trip_cost_pct: float = DEFAULT_ROUND_TRIP_COST_PCT,
) -> pd.DataFrame:
    """Measure recovery after each bullish Knoxville endpoint without same-bar lookahead."""
    frame = calculated.copy().reset_index(drop=True)
    if frame.empty or "knox_bullish" not in frame.columns:
        return _empty_events()

    frame["date"] = pd.to_datetime(frame.get("date"), errors="coerce")
    for column in ("open", "high", "low", "close", "envelope_lower", "knox_reference_bars"):
        if column not in frame.columns:
            continue
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    frame = frame.dropna(subset=["date", "high", "low", "close"]).reset_index(drop=True)
    if frame.empty:
        return _empty_events()

    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    end_position = int(frame.index[frame["date"].dt.normalize() <= end_ts].max()) if bool(
        (frame["date"].dt.normalize() <= end_ts).any()
    ) else -1
    if end_position < 0:
        return _empty_events()

    rows: list[dict[str, Any]] = []
    event_indexes = frame.index[
        frame["knox_bullish"].fillna(False).astype(bool)
        & frame["date"].dt.normalize().between(start_ts, end_ts)
    ]
    for endpoint_index in event_indexes:
        endpoint_index = int(endpoint_index)
        reference_bars_value = frame.iloc[endpoint_index].get("knox_reference_bars")
        if pd.isna(reference_bars_value):
            continue
        reference_bars = int(reference_bars_value)
        reference_index = endpoint_index - reference_bars
        if reference_bars <= 0 or reference_index < 0:
            continue

        first = frame.iloc[reference_index]
        second = frame.iloc[endpoint_index]
        first_low = _finite_float(first.get("low"))
        second_low = _finite_float(second.get("low"))
        lower_band = _finite_float(second.get("envelope_lower"))
        if first_low is None or second_low is None or first_low <= 0 or second_low <= 0:
            continue

        drop_pct = (first_low - second_low) / first_low * 100.0
        equal_bounce_target = second_low * (1.0 + drop_pct / 100.0)
        full_recovery_target = first_low
        full_recovery_gain_required_pct = (first_low / second_low - 1.0) * 100.0
        lower_distance_pct = (
            abs(second_low - lower_band) / lower_band * 100.0
            if lower_band is not None and lower_band > 0
            else None
        )
        proximity_pass = bool(
            lower_distance_pct is not None
            and lower_distance_pct <= max(float(envelope_proximity_pct), 0.0)
        )

        future = frame.iloc[endpoint_index + 1 : end_position + 1]
        endpoint_date = pd.Timestamp(second["date"]).normalize()
        equal_high = _first_recovery(
            future, "high", equal_bounce_target, endpoint_index, endpoint_date
        )
        equal_close = _first_recovery(
            future, "close", equal_bounce_target, endpoint_index, endpoint_date
        )
        full_high = _first_recovery(
            future, "high", full_recovery_target, endpoint_index, endpoint_date
        )
        full_close = _first_recovery(
            future, "close", full_recovery_target, endpoint_index, endpoint_date
        )
        observation_sessions = max(end_position - endpoint_index, 0)
        endpoint_age_bars = observation_sessions
        discontinuity = _has_overnight_discontinuity(
            frame,
            start_index=reference_index,
            end_index=end_position,
        )

        row: dict[str, Any] = {
            "exchange": exchange,
            "symbol": str(symbol).upper(),
            "name": name or str(symbol).upper(),
            "first_endpoint_date": _date_text(first.get("date")),
            "second_endpoint_date": _date_text(second.get("date")),
            "first_endpoint_low": first_low,
            "second_endpoint_low": second_low,
            "line_trading_bars": reference_bars,
            "line_calendar_days": int(
                (pd.Timestamp(second["date"]).normalize() - pd.Timestamp(first["date"]).normalize()).days
            ),
            "drop_pct": drop_pct,
            "envelope_lower": lower_band,
            "envelope_distance_pct": lower_distance_pct,
            "proximity_pass": proximity_pass,
            "data_quality_pass": not discontinuity,
            "data_quality_reason": "" if not discontinuity else "overnight_price_discontinuity",
            "equal_bounce_target": equal_bounce_target,
            "full_recovery_target": full_recovery_target,
            "full_recovery_gain_required_pct": full_recovery_gain_required_pct,
            "observation_sessions": observation_sessions,
            "endpoint_age_bars": endpoint_age_bars,
            "latest_observation_date": _date_text(frame.iloc[end_position].get("date")),
            "latest_close": _finite_float(frame.iloc[end_position].get("close")),
        }
        _add_recovery_columns(row, "equal_high", equal_high)
        _add_recovery_columns(row, "equal_close", equal_close)
        _add_recovery_columns(row, "full_high", full_high)
        _add_recovery_columns(row, "full_close", full_close)
        for window in RECOVERY_WINDOWS:
            row[f"full_close_recovered_{window}d"] = bool(
                full_close is not None and int(full_close[1]) <= window
            )
            row[f"full_close_{window}d_eligible"] = bool(
                observation_sessions >= window
            )
        _add_next_day_entry_metrics(
            row,
            frame,
            endpoint_index=endpoint_index,
            end_index=end_position,
            equal_target=equal_bounce_target,
            full_target=full_recovery_target,
            endpoint_drop_pct=drop_pct,
            round_trip_cost_pct=max(float(round_trip_cost_pct), 0.0),
        )
        rows.append(row)

    return pd.DataFrame(rows) if rows else _empty_events()


def run_knox_recovery_study(
    storage: Storage,
    exchange: str = "NSE",
    *,
    symbols: list[str] | None = None,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    knox_lookback: int = DEFAULT_KNOX_LOOKBACK,
    rsi_length: int = DEFAULT_RSI_LENGTH,
    momentum_length: int = DEFAULT_MOMENTUM_LENGTH,
    envelope_length: int = DEFAULT_ENVELOPE_LENGTH,
    envelope_percent: float = DEFAULT_ENVELOPE_PERCENT,
    envelope_ma_type: str = DEFAULT_ENVELOPE_MA_TYPE,
    envelope_proximity_pct: float = DEFAULT_PROXIMITY_PCT,
    recent_endpoint_bars: int = DEFAULT_RECENT_ENDPOINT_BARS,
    round_trip_cost_pct: float = DEFAULT_ROUND_TRIP_COST_PCT,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> KnoxRecoveryStudyResult:
    candidates = _candidate_symbols(storage, exchange, symbols)
    names = _load_name_map(storage, exchange)
    event_frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []

    _emit_progress(
        progress_callback,
        phase="Backtesting Knoxville recoveries",
        completed=0,
        total=len(candidates),
        current_symbol="",
        current_exchange=exchange,
    )
    min_history = max(int(knox_lookback) + int(momentum_length) + 1, int(envelope_length))
    for completed, symbol in enumerate(candidates, start=1):
        daily = storage.load_candles(exchange, symbol, "1D")
        _emit_progress(
            progress_callback,
            phase="Backtesting Knoxville recoveries",
            completed=completed,
            total=len(candidates),
            current_symbol=symbol,
            current_exchange=exchange,
        )
        if daily.empty or len(daily) < min_history:
            continue
        calculated = calculate_knox_envelope(
            daily,
            knox_lookback=knox_lookback,
            rsi_length=rsi_length,
            momentum_length=momentum_length,
            envelope_length=envelope_length,
            envelope_percent=envelope_percent,
            envelope_ma_type=envelope_ma_type,
            envelope_proximity_pct=envelope_proximity_pct,
        )
        if calculated.empty:
            continue
        coverage_rows.append(
            {
                "symbol": symbol,
                "history_start": calculated.iloc[0]["date"],
                "history_end": calculated.iloc[-1]["date"],
            }
        )
        events = backtest_knox_recovery_frame(
            calculated,
            symbol=symbol,
            exchange=exchange,
            name=names.get(symbol, symbol),
            start_date=start_date,
            end_date=end_date,
            envelope_proximity_pct=envelope_proximity_pct,
            round_trip_cost_pct=round_trip_cost_pct,
        )
        if not events.empty:
            event_frames.append(events)

    events = pd.concat(event_frames, ignore_index=True) if event_frames else _empty_events()
    if not events.empty:
        events = events.sort_values(
            ["second_endpoint_date", "symbol"], ascending=[False, True]
        ).reset_index(drop=True)
    qualifying = (
        events.loc[
            events["proximity_pass"].fillna(False).astype(bool)
            & events["data_quality_pass"].fillna(False).astype(bool)
        ].copy()
        if not events.empty
        else events.copy()
    )
    stock_stats = _aggregate_stock_stats(qualifying)
    recent_endpoint_bars = max(int(recent_endpoint_bars), 0)
    current_candidates = qualifying.loc[
        pd.to_numeric(qualifying.get("endpoint_age_bars"), errors="coerce") <= recent_endpoint_bars
    ].copy() if not qualifying.empty else qualifying.copy()
    if not current_candidates.empty:
        current_candidates = current_candidates.sort_values(
            ["envelope_distance_pct", "second_endpoint_date", "symbol"],
            ascending=[True, False, True],
        ).drop_duplicates("symbol", keep="first").reset_index(drop=True)

    coverage = pd.DataFrame(coverage_rows)
    summary = _build_summary(
        events,
        qualifying,
        coverage,
        exchange=exchange,
        symbols_processed=len(candidates),
        start_date=start_date,
        end_date=end_date,
        knox_lookback=knox_lookback,
        rsi_length=rsi_length,
        momentum_length=momentum_length,
        envelope_length=envelope_length,
        envelope_percent=envelope_percent,
        envelope_ma_type=envelope_ma_type,
        envelope_proximity_pct=envelope_proximity_pct,
        recent_endpoint_bars=recent_endpoint_bars,
        round_trip_cost_pct=round_trip_cost_pct,
        current_candidates=len(current_candidates),
    )
    return KnoxRecoveryStudyResult(summary, current_candidates, stock_stats, events)


def save_knox_recovery_outputs(
    result: KnoxRecoveryStudyResult,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "latest_summary.csv",
        "current_candidates": output_dir / "latest_current_candidates.csv",
        "stock_stats": output_dir / "latest_stock_stats.csv",
        "events": output_dir / "latest_events.csv",
    }
    pd.DataFrame([result.summary]).to_csv(paths["summary"], index=False)
    result.current_candidates.to_csv(paths["current_candidates"], index=False)
    result.stock_stats.to_csv(paths["stock_stats"], index=False)
    result.events.to_csv(paths["events"], index=False)
    return paths


def load_knox_recovery_outputs(output_dir: Path) -> KnoxRecoveryStudyResult:
    def read(name: str) -> pd.DataFrame:
        path = output_dir / name
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    summary_frame = read("latest_summary.csv")
    return KnoxRecoveryStudyResult(
        summary=summary_frame.iloc[0].to_dict() if not summary_frame.empty else {},
        current_candidates=read("latest_current_candidates.csv"),
        stock_stats=read("latest_stock_stats.csv"),
        events=read("latest_events.csv"),
    )


def _first_recovery(
    future: pd.DataFrame,
    price_column: str,
    target: float,
    endpoint_index: int,
    endpoint_date: pd.Timestamp,
) -> tuple[pd.Timestamp, int, int] | None:
    hits = future.index[pd.to_numeric(future[price_column], errors="coerce") >= target]
    if not len(hits):
        return None
    hit_index = int(hits[0])
    hit_date = pd.Timestamp(future.loc[hit_index, "date"]).normalize()
    trading_days = hit_index - endpoint_index
    calendar_days = int((hit_date - endpoint_date).days)
    return hit_date, trading_days, calendar_days


def _add_recovery_columns(
    row: dict[str, Any],
    prefix: str,
    recovery: tuple[pd.Timestamp, int, int] | None,
) -> None:
    row[f"{prefix}_recovered"] = recovery is not None
    row[f"{prefix}_date"] = recovery[0].strftime("%Y-%m-%d") if recovery else ""
    row[f"{prefix}_trading_days"] = recovery[1] if recovery else pd.NA
    row[f"{prefix}_calendar_days"] = recovery[2] if recovery else pd.NA


def _aggregate_stock_stats(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (symbol, name), group in events.groupby(["symbol", "name"], dropna=False):
        row = {
            "symbol": symbol,
            "name": name,
            "events": len(group),
            "median_line_trading_bars": _median(group["line_trading_bars"]),
            "median_line_calendar_days": _median(group["line_calendar_days"]),
            "median_drop_pct": _median(group["drop_pct"]),
            "equal_high_recovery_rate_pct": _rate(group["equal_high_recovered"]),
            "equal_close_recovery_rate_pct": _rate(group["equal_close_recovered"]),
            "full_high_recovery_rate_pct": _rate(group["full_high_recovered"]),
            "full_close_recovery_rate_pct": _rate(group["full_close_recovered"]),
            "median_equal_close_trading_days": _median_recovered(group, "equal_close"),
            "median_full_close_trading_days": _median_recovered(group, "full_close"),
            "unrecovered_full_close": int((~group["full_close_recovered"].astype(bool)).sum()),
        }
        for window in RECOVERY_WINDOWS:
            eligible = group.loc[group[f"full_close_{window}d_eligible"].astype(bool)]
            row[f"full_close_{window}d_recovery_rate_pct"] = (
                _rate(eligible[f"full_close_recovered_{window}d"]) if not eligible.empty else np.nan
            )
            row[f"full_close_{window}d_eligible_events"] = len(eligible)
            trade_eligible = group.loc[group[f"entry_{window}d_eligible"].astype(bool)]
            row[f"entry_{window}d_trades"] = len(trade_eligible)
            row[f"entry_{window}d_win_rate_pct"] = (
                _rate(trade_eligible[f"entry_{window}d_win"])
                if not trade_eligible.empty
                else np.nan
            )
            row[f"entry_{window}d_avg_net_return_pct"] = _mean(
                trade_eligible.get(f"entry_{window}d_net_return_pct", pd.Series(dtype=float))
            )
            row[f"entry_{window}d_median_net_return_pct"] = _median(
                trade_eligible.get(f"entry_{window}d_net_return_pct", pd.Series(dtype=float))
            )
            target_hits = trade_eligible.loc[
                trade_eligible[f"entry_{window}d_drop_target_hit"].astype(bool)
            ]
            row[f"entry_{window}d_drop_target_hit_rate_pct"] = (
                len(target_hits) / len(trade_eligible) * 100.0
                if len(trade_eligible)
                else np.nan
            )
            row[f"entry_{window}d_drop_target_median_exit_sessions"] = _median(
                target_hits.get("drop_target_exit_trading_days", pd.Series(dtype=float))
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["full_close_recovery_rate_pct", "events", "symbol"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def _build_summary(
    events: pd.DataFrame,
    qualifying: pd.DataFrame,
    coverage: pd.DataFrame,
    **settings: Any,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "logic_version": KNOX_RECOVERY_LOGIC_VERSION,
        "knox_events": len(events),
        "proximity_events": int(events["proximity_pass"].fillna(False).astype(bool).sum()) if not events.empty else 0,
        "data_quality_excluded_events": int(
            (
                events["proximity_pass"].fillna(False).astype(bool)
                & ~events["data_quality_pass"].fillna(False).astype(bool)
            ).sum()
        ) if not events.empty else 0,
        "qualifying_events": len(qualifying),
        "stocks_with_history": len(coverage),
        "stocks_with_qualifying_events": int(qualifying["symbol"].nunique()) if not qualifying.empty else 0,
        "median_line_trading_bars": _median(qualifying.get("line_trading_bars", pd.Series(dtype=float))),
        "median_drop_pct": _median(qualifying.get("drop_pct", pd.Series(dtype=float))),
        "full_high_recovery_rate_pct": _rate(qualifying.get("full_high_recovered", pd.Series(dtype=bool))),
        "full_close_recovery_rate_pct": _rate(qualifying.get("full_close_recovered", pd.Series(dtype=bool))),
        "median_full_high_trading_days": _median_recovered(qualifying, "full_high"),
        "median_full_close_trading_days": _median_recovered(qualifying, "full_close"),
        "unrecovered_full_close": int((~qualifying["full_close_recovered"].astype(bool)).sum()) if not qualifying.empty else 0,
        "earliest_history_date": _date_min(coverage.get("history_start", pd.Series(dtype="datetime64[ns]"))),
        "latest_history_date": _date_max(coverage.get("history_end", pd.Series(dtype="datetime64[ns]"))),
        "generated_at_ist": pd.Timestamp.now(tz="Asia/Kolkata").strftime("%Y-%m-%d %H:%M:%S IST"),
        **settings,
    }
    for window in RECOVERY_WINDOWS:
        if qualifying.empty:
            eligible = qualifying
        else:
            eligible = qualifying.loc[qualifying[f"full_close_{window}d_eligible"].astype(bool)]
        summary[f"full_close_{window}d_eligible_events"] = len(eligible)
        summary[f"full_close_{window}d_recovery_rate_pct"] = (
            _rate(eligible[f"full_close_recovered_{window}d"]) if not eligible.empty else np.nan
        )
        trade_eligible = (
            qualifying.loc[qualifying[f"entry_{window}d_eligible"].astype(bool)]
            if not qualifying.empty
            else qualifying
        )
        summary[f"entry_{window}d_trades"] = len(trade_eligible)
        summary[f"entry_{window}d_win_rate_pct"] = (
            _rate(trade_eligible[f"entry_{window}d_win"]) if not trade_eligible.empty else np.nan
        )
        summary[f"entry_{window}d_avg_net_return_pct"] = _mean(
            trade_eligible.get(f"entry_{window}d_net_return_pct", pd.Series(dtype=float))
        )
        summary[f"entry_{window}d_median_net_return_pct"] = _median(
            trade_eligible.get(f"entry_{window}d_net_return_pct", pd.Series(dtype=float))
        )
        summary[f"entry_{window}d_equal_target_profitable_hit_rate_pct"] = (
            _rate(trade_eligible[f"entry_{window}d_equal_target_profitable_hit"])
            if not trade_eligible.empty
            else np.nan
        )
        summary[f"entry_{window}d_full_target_profitable_hit_rate_pct"] = (
            _rate(trade_eligible[f"entry_{window}d_full_target_profitable_hit"])
            if not trade_eligible.empty
            else np.nan
        )
        target_hits = (
            trade_eligible.loc[
                trade_eligible[f"entry_{window}d_drop_target_hit"].astype(bool)
            ]
            if not trade_eligible.empty
            else trade_eligible
        )
        summary[f"entry_{window}d_drop_target_hit_rate_pct"] = (
            len(target_hits) / len(trade_eligible) * 100.0
            if len(trade_eligible)
            else np.nan
        )
        summary[f"entry_{window}d_drop_target_profitable_exit_rate_pct"] = (
            _rate(target_hits["drop_target_profitable_exit"])
            if not target_hits.empty
            else np.nan
        )
        summary[f"entry_{window}d_drop_target_median_exit_sessions"] = _median(
            target_hits.get("drop_target_exit_trading_days", pd.Series(dtype=float))
        )
        non_overlapping = _non_overlapping_trade_events(trade_eligible, window)
        summary[f"entry_{window}d_non_overlapping_trades"] = len(non_overlapping)
        summary[f"entry_{window}d_non_overlapping_win_rate_pct"] = (
            _rate(non_overlapping[f"entry_{window}d_win"])
            if not non_overlapping.empty
            else np.nan
        )
        summary[f"entry_{window}d_non_overlapping_avg_net_return_pct"] = _mean(
            non_overlapping.get(
                f"entry_{window}d_net_return_pct", pd.Series(dtype=float)
            )
        )
        summary[f"entry_{window}d_non_overlapping_median_net_return_pct"] = _median(
            non_overlapping.get(
                f"entry_{window}d_net_return_pct", pd.Series(dtype=float)
            )
        )
        summary[f"entry_{window}d_non_overlapping_drop_target_hit_rate_pct"] = (
            _rate(non_overlapping[f"entry_{window}d_drop_target_hit"])
            if not non_overlapping.empty
            else np.nan
        )
    return summary


def add_non_overlapping_trade_summary(
    summary: dict[str, Any],
    qualifying: pd.DataFrame,
) -> dict[str, Any]:
    """Add one-position-per-stock statistics to an existing saved study summary."""
    updated = dict(summary)
    for window in RECOVERY_WINDOWS:
        eligible_column = f"entry_{window}d_eligible"
        if qualifying.empty or eligible_column not in qualifying.columns:
            selected = pd.DataFrame()
        else:
            eligible = qualifying.loc[_boolean_series(qualifying[eligible_column])]
            selected = _non_overlapping_trade_events(eligible, window)
        updated[f"entry_{window}d_non_overlapping_trades"] = len(selected)
        updated[f"entry_{window}d_non_overlapping_win_rate_pct"] = (
            _rate(selected[f"entry_{window}d_win"]) if not selected.empty else np.nan
        )
        updated[f"entry_{window}d_non_overlapping_avg_net_return_pct"] = _mean(
            selected.get(f"entry_{window}d_net_return_pct", pd.Series(dtype=float))
        )
        updated[f"entry_{window}d_non_overlapping_median_net_return_pct"] = _median(
            selected.get(f"entry_{window}d_net_return_pct", pd.Series(dtype=float))
        )
        updated[f"entry_{window}d_non_overlapping_drop_target_hit_rate_pct"] = (
            _rate(selected[f"entry_{window}d_drop_target_hit"])
            if not selected.empty and f"entry_{window}d_drop_target_hit" in selected.columns
            else np.nan
        )
    return updated


def _non_overlapping_trade_events(events: pd.DataFrame, window: int) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    frame = events.copy()
    frame["entry_date"] = pd.to_datetime(frame.get("entry_date"), errors="coerce")
    exit_column = f"entry_{window}d_exit_date"
    frame[exit_column] = pd.to_datetime(frame.get(exit_column), errors="coerce")
    frame = frame.dropna(subset=["symbol", "entry_date", exit_column]).sort_values(
        ["symbol", "entry_date", "second_endpoint_date"]
    )
    selected_indexes: list[int] = []
    for _, group in frame.groupby("symbol", sort=False):
        last_exit: pd.Timestamp | None = None
        for index, row in group.iterrows():
            entry_date = pd.Timestamp(row["entry_date"])
            if last_exit is None or entry_date > last_exit:
                selected_indexes.append(index)
                last_exit = pd.Timestamp(row[exit_column])
    return frame.loc[selected_indexes].copy()


def _add_next_day_entry_metrics(
    row: dict[str, Any],
    frame: pd.DataFrame,
    *,
    endpoint_index: int,
    end_index: int,
    equal_target: float,
    full_target: float,
    endpoint_drop_pct: float,
    round_trip_cost_pct: float,
) -> None:
    entry_index = endpoint_index + 1
    entry_available = entry_index <= end_index
    entry_price = (
        _finite_float(frame.iloc[entry_index].get("high")) if entry_available else None
    )
    entry_available = bool(entry_available and entry_price is not None and entry_price > 0)
    row["entry_available"] = entry_available
    row["entry_date"] = _date_text(frame.iloc[entry_index].get("date")) if entry_available else ""
    row["entry_price_next_day_high"] = entry_price if entry_available else pd.NA
    latest_close = _finite_float(frame.iloc[end_index].get("close"))
    row["latest_net_return_pct"] = (
        (latest_close - entry_price) / entry_price * 100.0 - round_trip_cost_pct
        if entry_available and latest_close is not None
        else pd.NA
    )
    drop_target_price = (
        entry_price * (1.0 + endpoint_drop_pct / 100.0)
        if entry_available and entry_price is not None
        else None
    )
    target_future = frame.iloc[entry_index + 1 : end_index + 1] if entry_available else frame.iloc[0:0]
    target_hit_indexes = (
        target_future.index[
            pd.to_numeric(target_future.get("high"), errors="coerce") >= drop_target_price
        ]
        if drop_target_price is not None and not target_future.empty
        else pd.Index([])
    )
    target_hit_index = int(target_hit_indexes[0]) if len(target_hit_indexes) else None
    target_hit = target_hit_index is not None
    target_net_return_pct = endpoint_drop_pct - round_trip_cost_pct
    row["drop_target_price_from_entry"] = drop_target_price if drop_target_price is not None else pd.NA
    row["drop_target_hit"] = target_hit
    row["drop_target_exit_date"] = (
        _date_text(frame.iloc[target_hit_index].get("date")) if target_hit_index is not None else ""
    )
    row["drop_target_exit_trading_days"] = (
        target_hit_index - entry_index if target_hit_index is not None else pd.NA
    )
    row["drop_target_exit_calendar_days"] = (
        int(
            (
                pd.Timestamp(frame.iloc[target_hit_index]["date"]).normalize()
                - pd.Timestamp(frame.iloc[entry_index]["date"]).normalize()
            ).days
        )
        if target_hit_index is not None
        else pd.NA
    )
    row["drop_target_exit_net_return_pct"] = target_net_return_pct if target_hit else pd.NA
    row["drop_target_profitable_exit"] = bool(target_hit and target_net_return_pct > 0.0)

    for window in RECOVERY_WINDOWS:
        exit_index = entry_index + window - 1
        eligible = bool(entry_available and exit_index <= end_index)
        row[f"entry_{window}d_eligible"] = eligible
        row[f"entry_{window}d_drop_target_hit"] = bool(
            eligible
            and target_hit_index is not None
            and target_hit_index <= exit_index
        )
        row[f"entry_{window}d_exit_date"] = (
            _date_text(frame.iloc[exit_index].get("date")) if eligible else ""
        )
        if not eligible or entry_price is None:
            row[f"entry_{window}d_exit_close"] = pd.NA
            row[f"entry_{window}d_net_return_pct"] = pd.NA
            row[f"entry_{window}d_win"] = False
            row[f"entry_{window}d_equal_target_profitable_hit"] = False
            row[f"entry_{window}d_full_target_profitable_hit"] = False
            continue

        exit_close = _finite_float(frame.iloc[exit_index].get("close"))
        net_return = (
            (exit_close - entry_price) / entry_price * 100.0 - round_trip_cost_pct
            if exit_close is not None
            else None
        )
        # A next-day-high entry is assumed to occur after that day's range is known,
        # so target checks begin on the following session.
        target_window = frame.iloc[entry_index + 1 : exit_index + 1]
        future_high = pd.to_numeric(target_window.get("high"), errors="coerce")
        equal_target_profitable = equal_target > entry_price * (1.0 + round_trip_cost_pct / 100.0)
        full_target_profitable = full_target > entry_price * (1.0 + round_trip_cost_pct / 100.0)
        row[f"entry_{window}d_exit_close"] = exit_close if exit_close is not None else pd.NA
        row[f"entry_{window}d_net_return_pct"] = net_return if net_return is not None else pd.NA
        row[f"entry_{window}d_win"] = bool(net_return is not None and net_return > 0.0)
        row[f"entry_{window}d_equal_target_profitable_hit"] = bool(
            equal_target_profitable and not future_high.empty and (future_high >= equal_target).any()
        )
        row[f"entry_{window}d_full_target_profitable_hit"] = bool(
            full_target_profitable and not future_high.empty and (future_high >= full_target).any()
        )


def _median_recovered(frame: pd.DataFrame, prefix: str) -> float | None:
    if frame.empty or f"{prefix}_recovered" not in frame.columns:
        return None
    recovered = frame.loc[frame[f"{prefix}_recovered"].astype(bool), f"{prefix}_trading_days"]
    return _median(recovered)


def _median(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.median()) if not numeric.empty else None


def _mean(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else None


def _rate(values: pd.Series) -> float | None:
    if values.empty:
        return None
    return float(_boolean_series(values).mean() * 100.0)


def _boolean_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.fillna("").astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _has_overnight_discontinuity(
    frame: pd.DataFrame,
    *,
    start_index: int,
    end_index: int,
) -> bool:
    if "open" not in frame.columns:
        return False
    window = frame.iloc[max(start_index, 0) : end_index + 1]
    previous_close = pd.to_numeric(frame["close"], errors="coerce").shift(1).iloc[
        max(start_index, 0) : end_index + 1
    ]
    overnight_ratio = pd.to_numeric(window["open"], errors="coerce") / previous_close.replace(
        0.0, np.nan
    )
    return bool(((overnight_ratio < 0.55) | (overnight_ratio > 1.80)).fillna(False).any())


def _date_text(value: Any) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(parsed) else pd.Timestamp(parsed).strftime("%Y-%m-%d")


def _date_min(values: pd.Series) -> str:
    parsed = pd.to_datetime(values, errors="coerce").dropna()
    return parsed.min().strftime("%Y-%m-%d") if not parsed.empty else ""


def _date_max(values: pd.Series) -> str:
    parsed = pd.to_datetime(values, errors="coerce").dropna()
    return parsed.max().strftime("%Y-%m-%d") if not parsed.empty else ""


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "exchange",
            "symbol",
            "name",
            "first_endpoint_date",
            "second_endpoint_date",
            "first_endpoint_low",
            "second_endpoint_low",
            "line_trading_bars",
            "line_calendar_days",
            "drop_pct",
            "envelope_lower",
            "envelope_distance_pct",
            "proximity_pass",
            "data_quality_pass",
            "data_quality_reason",
            "equal_bounce_target",
            "full_recovery_target",
            "full_recovery_gain_required_pct",
            "observation_sessions",
            "endpoint_age_bars",
            "latest_observation_date",
            "latest_close",
        ]
    )
