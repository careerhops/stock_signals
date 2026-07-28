from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from stock_screener.data.storage import Storage
from stock_screener.weekly_buy_tracker_study import _emit_progress, _load_name_map, _to_float


@dataclass(frozen=True)
class ResistanceBreaksStudyResult:
    summary: dict[str, Any]
    stock_stats: pd.DataFrame
    breakout_events: pd.DataFrame


def run_resistance_breaks_study(
    storage: Storage,
    exchange: str = "NSE",
    *,
    left_bars: int = 15,
    right_bars: int = 15,
    volume_avg_window: int = 20,
    volume_multiplier: float = 2.0,
    min_break_count: int = 2,
    recent_window_days: int = 7,
    reference_date: str | pd.Timestamp | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> ResistanceBreaksStudyResult:
    data_root = storage.data_root
    all_symbols = sorted(p.stem for p in (data_root / "candles" / exchange / "1D").glob("*.csv"))
    name_map = _load_name_map(storage, exchange)

    stock_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    _emit_progress(
        progress_callback,
        phase="Scanning resistance breaks",
        completed=0,
        total=len(all_symbols),
        current_symbol="",
        current_exchange=exchange,
    )

    min_history = max(left_bars + right_bars + 5, volume_avg_window + 2)
    for index, symbol in enumerate(all_symbols, start=1):
        daily = storage.load_candles(exchange, symbol, "1D")
        _emit_progress(
            progress_callback,
            phase="Scanning resistance breaks",
            completed=index,
            total=len(all_symbols),
            current_symbol=symbol,
            current_exchange=exchange,
        )
        if daily.empty or len(daily) < min_history:
            continue

        frame = _prepare_daily(daily)
        if frame.empty or len(frame) < min_history:
            continue

        metrics = _evaluate_resistance_breaks(
            frame,
            left_bars=left_bars,
            right_bars=right_bars,
            volume_avg_window=volume_avg_window,
            volume_multiplier=volume_multiplier,
            min_break_count=min_break_count,
            recent_window_days=recent_window_days,
            reference_date=reference_date,
        )
        stock_rows.append(
            {
                "exchange": exchange,
                "symbol": symbol,
                "name": name_map.get(symbol, symbol),
                "latest_close": _to_float(frame.iloc[-1].get("close")),
                "latest_close_date": frame.iloc[-1].get("date"),
                **metrics,
            }
        )
        if metrics["breakout_events"]:
            for event in metrics["breakout_events"]:
                event_rows.append(
                    {
                        "exchange": exchange,
                        "symbol": symbol,
                        "name": name_map.get(symbol, symbol),
                        **event,
                    }
                )

    stock_stats = pd.DataFrame(stock_rows)
    if not stock_stats.empty:
        if "latest_close_date" in stock_stats.columns:
            stock_stats["latest_close_date"] = pd.to_datetime(stock_stats["latest_close_date"], errors="coerce")
        for column in ("volume_confirmed_resistance_break_count", "latest_break_volume_ratio", "latest_resistance_level"):
            if column in stock_stats.columns:
                stock_stats[column] = pd.to_numeric(stock_stats[column], errors="coerce")
        stock_stats = stock_stats.sort_values(
            ["volume_confirmed_resistance_break_count", "latest_break_volume_ratio", "symbol"],
            ascending=[False, False, True],
            na_position="last",
        ).reset_index(drop=True)

    breakout_events = pd.DataFrame(event_rows)
    if not breakout_events.empty:
        if "date" in breakout_events.columns:
            breakout_events["date"] = pd.to_datetime(breakout_events["date"], errors="coerce")
        for column in ("close", "resistance_level", "volume", "avg_volume", "volume_ratio"):
            if column in breakout_events.columns:
                breakout_events[column] = pd.to_numeric(breakout_events[column], errors="coerce")
        breakout_events = breakout_events.sort_values(
            ["date", "symbol"],
            ascending=[False, True],
            na_position="last",
        ).reset_index(drop=True)

    summary = _build_summary(
        exchange,
        len(all_symbols),
        stock_stats,
        left_bars=left_bars,
        right_bars=right_bars,
        volume_avg_window=volume_avg_window,
        volume_multiplier=volume_multiplier,
        min_break_count=min_break_count,
        recent_window_days=recent_window_days,
    )
    return ResistanceBreaksStudyResult(summary=summary, stock_stats=stock_stats, breakout_events=breakout_events)


def save_resistance_breaks_outputs(result: ResistanceBreaksStudyResult, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "latest_summary.csv"
    stock_stats_path = output_dir / "latest_stock_stats.csv"
    breakout_events_path = output_dir / "latest_breakout_events.csv"
    pd.DataFrame([result.summary]).to_csv(summary_path, index=False)
    result.stock_stats.to_csv(stock_stats_path, index=False)
    result.breakout_events.to_csv(breakout_events_path, index=False)
    return {
        "summary": summary_path,
        "stock_stats": stock_stats_path,
        "breakout_events": breakout_events_path,
    }


def load_resistance_breaks_outputs(output_dir: Path) -> ResistanceBreaksStudyResult:
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

    return ResistanceBreaksStudyResult(
        summary=summary,
        stock_stats=_read(output_dir / "latest_stock_stats.csv"),
        breakout_events=_read(output_dir / "latest_breakout_events.csv"),
    )


def _prepare_daily(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "open", "high", "low", "close", "volume"]).sort_values("date").reset_index(drop=True)
    return frame


def _evaluate_resistance_breaks(
    frame: pd.DataFrame,
    *,
    left_bars: int,
    right_bars: int,
    volume_avg_window: int,
    volume_multiplier: float,
    min_break_count: int,
    recent_window_days: int,
    reference_date: str | pd.Timestamp | None,
) -> dict[str, Any]:
    highs = frame["high"].reset_index(drop=True)
    closes = frame["close"].reset_index(drop=True)
    volumes = frame["volume"].reset_index(drop=True)
    dates = pd.to_datetime(frame["date"], errors="coerce").reset_index(drop=True)
    reference_ts = pd.Timestamp(reference_date).normalize() if reference_date is not None else pd.Timestamp.today().normalize()
    eligible_mask = dates.notna() & (dates.dt.normalize() <= reference_ts)
    if not eligible_mask.any():
        return _empty_metrics(recent_window_days)

    eligible = frame.loc[eligible_mask].reset_index(drop=True)
    highs = eligible["high"].reset_index(drop=True)
    closes = eligible["close"].reset_index(drop=True)
    volumes = eligible["volume"].reset_index(drop=True)
    dates = pd.to_datetime(eligible["date"], errors="coerce").reset_index(drop=True)

    left_max = highs.shift(1).rolling(left_bars, min_periods=left_bars).max()
    right_max = highs.iloc[::-1].shift(1).rolling(right_bars, min_periods=right_bars).max().iloc[::-1]
    pivot_mask = left_max.notna() & right_max.notna() & highs.notna() & (highs >= left_max) & (highs > right_max)
    pivot_highs = highs.where(pivot_mask)

    avg_volume = volumes.shift(1).rolling(volume_avg_window, min_periods=volume_avg_window).mean()
    six_month_cutoff = reference_ts - pd.DateOffset(months=6)
    pivot_candidates = pd.DataFrame(
        {
            "date": dates,
            "resistance_level": pivot_highs,
        }
    )
    pivot_candidates = pivot_candidates[
        pivot_candidates["resistance_level"].notna()
        & pivot_candidates["date"].notna()
        & (pivot_candidates["date"].dt.normalize() >= six_month_cutoff.normalize())
        & (pivot_candidates["date"].dt.normalize() <= reference_ts)
    ].copy()
    if pivot_candidates.empty:
        return _empty_metrics(recent_window_days)

    pivot_candidates = pivot_candidates.sort_values(
        ["resistance_level", "date"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)
    chosen_resistance_level = _to_float(pivot_candidates.iloc[0]["resistance_level"])
    chosen_resistance_ts = pd.Timestamp(pivot_candidates.iloc[0]["date"]).normalize() if chosen_resistance_level is not None else pd.NaT
    chosen_resistance_date = chosen_resistance_ts.strftime("%Y-%m-%d") if pd.notna(chosen_resistance_ts) else ""
    if chosen_resistance_level is None:
        return _empty_metrics(recent_window_days)

    prev_close = closes.shift(1)
    crosses_resistance = (
        prev_close.notna()
        & (prev_close <= float(chosen_resistance_level))
        & (closes > float(chosen_resistance_level))
        & dates.notna()
        & (dates.dt.normalize() > chosen_resistance_ts)
    )
    volume_ratio = volumes / avg_volume
    volume_confirmed = crosses_resistance & avg_volume.notna() & (volumes > (avg_volume * float(volume_multiplier)))

    breakout_events: list[dict[str, Any]] = []
    for idx in range(len(eligible)):
        if not bool(volume_confirmed.fillna(False).iloc[idx]):
            continue
        breakout_events.append(
            {
                "date": dates.iloc[idx],
                "close": _to_float(closes.iloc[idx]),
                "resistance_level": chosen_resistance_level,
                "resistance_zone_date": chosen_resistance_date,
                "volume": _to_float(volumes.iloc[idx]),
                "avg_volume": _to_float(avg_volume.iloc[idx]),
                "volume_ratio": _to_float(volume_ratio.iloc[idx]),
            }
        )

    recent_cutoff = reference_ts - pd.Timedelta(days=int(recent_window_days) - 1) if pd.notna(reference_ts) else pd.NaT
    recent_breakout_events = [
        event for event in breakout_events
        if (
            pd.notna(pd.Timestamp(event["date"]))
            and pd.notna(recent_cutoff)
            and pd.Timestamp(event["date"]).normalize() >= recent_cutoff
            and pd.Timestamp(event["date"]).normalize() <= reference_ts
        )
    ]
    latest_event = recent_breakout_events[-1] if recent_breakout_events else {}
    recent_dates = [pd.Timestamp(event["date"]).strftime("%Y-%m-%d") for event in recent_breakout_events[-5:]] if recent_breakout_events else []
    latest_close = _to_float(closes.iloc[-1]) if not closes.empty else None
    high_52w = _to_float(highs.rolling(252, min_periods=min(180, len(highs))).max().iloc[-1]) if not highs.empty else None
    close_above_recent_resistance = bool(
        latest_close is not None
        and chosen_resistance_level is not None
        and latest_close > chosen_resistance_level
    )
    resistance_within_25pct_of_ath = bool(
        chosen_resistance_level is not None
        and high_52w is not None
        and high_52w > 0
        and chosen_resistance_level < high_52w
        and chosen_resistance_level >= (0.75 * high_52w)
    )
    return {
        "latest_resistance_level": chosen_resistance_level,
        "selected_resistance_zone_date": chosen_resistance_date,
        "latest_52w_high": high_52w,
        "resistance_break_count_all": int(crosses_resistance.fillna(False).sum()),
        "volume_confirmed_resistance_break_count_all": int(len(breakout_events)),
        "volume_confirmed_resistance_break_count": int(len(recent_breakout_events)),
        "close_above_recent_resistance": close_above_recent_resistance,
        "resistance_within_25pct_of_ath": resistance_within_25pct_of_ath,
        "passes_volume_confirmed_resistance_breaks": bool(
            len(recent_breakout_events) >= int(min_break_count)
            and close_above_recent_resistance
            and resistance_within_25pct_of_ath
        ),
        "latest_break_date": pd.Timestamp(latest_event["date"]).strftime("%Y-%m-%d") if latest_event else "",
        "latest_break_close": latest_event.get("close") if latest_event else None,
        "latest_break_volume_ratio": latest_event.get("volume_ratio") if latest_event else None,
        "recent_break_dates_csv": ",".join(recent_dates),
        "recent_breakout_window_days": int(recent_window_days),
        "breakout_events": breakout_events,
    }


def _empty_metrics(recent_window_days: int) -> dict[str, Any]:
    return {
        "latest_resistance_level": None,
        "selected_resistance_zone_date": "",
        "latest_52w_high": None,
        "resistance_break_count_all": 0,
        "volume_confirmed_resistance_break_count_all": 0,
        "volume_confirmed_resistance_break_count": 0,
        "close_above_recent_resistance": False,
        "resistance_within_25pct_of_ath": False,
        "passes_volume_confirmed_resistance_breaks": False,
        "latest_break_date": "",
        "latest_break_close": None,
        "latest_break_volume_ratio": None,
        "recent_break_dates_csv": "",
        "recent_breakout_window_days": int(recent_window_days),
        "breakout_events": [],
    }


def _build_summary(
    exchange: str,
    symbols_processed: int,
    stock_stats: pd.DataFrame,
    *,
    left_bars: int,
    right_bars: int,
    volume_avg_window: int,
    volume_multiplier: float,
    min_break_count: int,
    recent_window_days: int,
) -> dict[str, Any]:
    latest_date = ""
    if not stock_stats.empty and "latest_close_date" in stock_stats.columns:
        latest_dates = pd.to_datetime(stock_stats["latest_close_date"], errors="coerce")
        if latest_dates.notna().any():
            latest_date = str(latest_dates.max().date())

    break_counts = (
        pd.to_numeric(stock_stats["volume_confirmed_resistance_break_count"], errors="coerce").dropna()
        if "volume_confirmed_resistance_break_count" in stock_stats.columns
        else pd.Series(dtype=float)
    )
    match_count = int(
        stock_stats["passes_volume_confirmed_resistance_breaks"].astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"}).sum()
    ) if (not stock_stats.empty and "passes_volume_confirmed_resistance_breaks" in stock_stats.columns) else 0
    return {
        "exchange": exchange,
        "symbols_processed": symbols_processed,
        "stocks_with_history": int(len(stock_stats)),
        "resistance_break_matches": match_count,
        "latest_close_date": latest_date,
        "avg_volume_confirmed_break_count": float(break_counts.mean()) if not break_counts.empty else 0.0,
        "left_bars": int(left_bars),
        "right_bars": int(right_bars),
        "volume_avg_window": int(volume_avg_window),
        "volume_multiplier": float(volume_multiplier),
        "min_break_count": int(min_break_count),
        "recent_breakout_window_days": int(recent_window_days),
    }
