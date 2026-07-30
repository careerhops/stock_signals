from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from stock_screener.data.storage import Storage
from stock_screener.weekly_buy_tracker_study import _emit_progress, _load_name_map, _to_float


@dataclass(frozen=True)
class AdxDiStudyResult:
    summary: dict[str, Any]
    stock_stats: pd.DataFrame


def _is_excluded_adx_symbol(symbol: str) -> bool:
    value = str(symbol or "").strip().upper()
    if not value:
        return True
    if "-" in value:
        return True
    if "NIFTY" in value or "BEES" in value:
        return True
    if value.endswith("ETF"):
        return True
    return any(character.isdigit() for character in value)


def calculate_adx_di(frame: pd.DataFrame, length: int = 14, threshold: float = 20.0) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    prepared = prepared.dropna(subset=["date", "high", "low", "close"]).sort_values("date").reset_index(drop=True)
    if prepared.empty:
        return prepared

    high = prepared["high"].astype(float)
    low = prepared["low"].astype(float)
    close = prepared["close"].astype(float)

    prev_close = close.shift(1).fillna(0.0)
    prev_high = high.shift(1).fillna(0.0)
    prev_low = low.shift(1).fillna(0.0)

    true_range = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    up_move = high - prev_high
    down_move = prev_low - low
    directional_movement_plus = np.where(up_move > down_move, np.maximum(up_move, 0.0), 0.0)
    directional_movement_minus = np.where(down_move > up_move, np.maximum(down_move, 0.0), 0.0)

    smoothed_true_range = _pine_recursive_smooth(true_range, int(length))
    smoothed_dm_plus = _pine_recursive_smooth(pd.Series(directional_movement_plus, index=prepared.index), int(length))
    smoothed_dm_minus = _pine_recursive_smooth(pd.Series(directional_movement_minus, index=prepared.index), int(length))

    with np.errstate(divide="ignore", invalid="ignore"):
        di_plus = np.where(smoothed_true_range > 0, smoothed_dm_plus / smoothed_true_range * 100.0, np.nan)
        di_minus = np.where(smoothed_true_range > 0, smoothed_dm_minus / smoothed_true_range * 100.0, np.nan)
        di_sum = di_plus + di_minus
        dx = np.where(di_sum > 0, np.abs(di_plus - di_minus) / di_sum * 100.0, np.nan)

    adx = pd.Series(dx, index=prepared.index).rolling(int(length), min_periods=int(length)).mean()

    prepared["true_range"] = true_range
    prepared["dm_plus"] = directional_movement_plus
    prepared["dm_minus"] = directional_movement_minus
    prepared["smoothed_true_range"] = smoothed_true_range
    prepared["smoothed_dm_plus"] = smoothed_dm_plus
    prepared["smoothed_dm_minus"] = smoothed_dm_minus
    prepared["di_plus"] = di_plus
    prepared["di_minus"] = di_minus
    prepared["dx"] = dx
    prepared["adx"] = adx
    prepared["threshold"] = float(threshold)
    prepared["adx_slope_positive"] = prepared["adx"] > prepared["adx"].shift(1)
    prepared["di_minus_slope_negative"] = prepared["di_minus"] < prepared["di_minus"].shift(1)
    prepared["di_plus_crossed_above_di_minus"] = (
        prepared["di_plus"].shift(1).notna()
        & prepared["di_minus"].shift(1).notna()
        & prepared["di_plus"].notna()
        & prepared["di_minus"].notna()
        & prepared["adx"].notna()
        & (prepared["di_plus"].shift(1) <= prepared["di_minus"].shift(1))
        & (prepared["di_plus"] > prepared["di_minus"])
        & (prepared["adx"] < prepared["di_plus"])
        & (prepared["adx"] < prepared["di_minus"])
    )
    prepared["di_plus_crossed_above_di_minus_over_threshold"] = (
        prepared["di_plus_crossed_above_di_minus"]
        & (prepared["di_plus"] > float(threshold))
        & (prepared["di_minus"] > float(threshold))
    )
    prepared["adx_crossed_above_di_minus"] = (
        prepared["adx"].shift(1).notna()
        & prepared["di_minus"].shift(1).notna()
        & prepared["adx"].notna()
        & prepared["di_minus"].notna()
        & (prepared["adx"].shift(1) <= prepared["di_minus"].shift(1))
        & (prepared["adx"] > prepared["di_minus"])
        & prepared["adx_slope_positive"]
        & prepared["di_minus_slope_negative"]
    )
    prepared["adx_bullish_cross_above_di_minus"] = (
        prepared["adx"].shift(1).notna()
        & prepared["di_minus"].shift(1).notna()
        & prepared["di_plus"].shift(1).notna()
        & prepared["adx"].notna()
        & prepared["di_minus"].notna()
        & prepared["di_plus"].notna()
        & (prepared["di_plus"] > prepared["di_minus"])
        & (prepared["adx"].shift(1) <= prepared["di_minus"].shift(1))
        & (prepared["adx"].shift(1) <= prepared["di_plus"].shift(1))
        & (prepared["adx"] > prepared["di_minus"])
        & (prepared["adx"] < prepared["di_plus"])
        & prepared["adx_slope_positive"]
        & prepared["di_minus_slope_negative"]
    )
    return prepared


def run_adx_di_study(
    storage: Storage,
    exchange: str = "NSE",
    *,
    symbols: list[str] | None = None,
    length: int = 14,
    threshold: float = 20.0,
    cross_lookback_bars: int = 3,
    trend_fast_ma_length: int = 50,
    trend_slow_ma_length: int = 200,
    volume_avg_lookback: int = 20,
    min_volume_ratio: float = 1.5,
    breakout_lookback_days: int = 20,
    rs_lookback_days: int = 20,
    min_rs_spread_pct: float = 0.0,
    support_left_bars: int = 15,
    support_right_bars: int = 15,
    atr_channel_ma_length: int = 20,
    atr_channel_atr_length: int = 14,
    atr_channel_ma_type: str = "EMA",
    atr_lower1_proximity_pct: float = 2.0,
    max_staleness_days: int = 10,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> AdxDiStudyResult:
    data_root = storage.data_root
    benchmark_daily = _prepare_benchmark_daily(storage.load_candles("NSE_INDEX", "NIFTY 50", "1D"))
    if symbols is None:
        all_symbols = sorted(
            p.stem
            for p in (data_root / "candles" / exchange / "1D").glob("*.csv")
            if not _is_excluded_adx_symbol(p.stem)
        )
    else:
        all_symbols = sorted(
            {
                str(symbol or "").strip().upper()
                for symbol in symbols
                if not _is_excluded_adx_symbol(str(symbol or "").strip().upper())
            }
        )
    name_map = _load_name_map(storage, exchange)

    rows: list[dict[str, Any]] = []
    _emit_progress(
        progress_callback,
        phase="Scanning ADX/DI crossover",
        completed=0,
        total=len(all_symbols),
        current_symbol="",
        current_exchange=exchange,
    )

    min_history = max(int(length) * 3, 40)
    latest_dates_seen: list[pd.Timestamp] = []
    for index, symbol in enumerate(all_symbols, start=1):
        daily = storage.load_candles(exchange, symbol, "1D")
        _emit_progress(
            progress_callback,
            phase="Scanning ADX/DI crossover",
            completed=index,
            total=len(all_symbols),
            current_symbol=symbol,
            current_exchange=exchange,
        )
        if daily.empty or len(daily) < min_history:
            continue

        frame = calculate_adx_di(daily, length=length, threshold=threshold)
        if frame.empty or len(frame) < min_history:
            continue
        latest_symbol_date = pd.to_datetime(frame.iloc[-1].get("date"), errors="coerce")
        if pd.notna(latest_symbol_date):
            latest_dates_seen.append(pd.Timestamp(latest_symbol_date).normalize())

        latest = frame.iloc[-1]
        di_plus_cross_series = frame["di_plus_crossed_above_di_minus"].fillna(False)
        di_plus_cross_over_threshold_series = frame["di_plus_crossed_above_di_minus_over_threshold"].fillna(False)
        cross_series = frame["adx_bullish_cross_above_di_minus"].fillna(False)
        recent_di_plus_cross_rows = frame.iloc[-int(cross_lookback_bars):].loc[di_plus_cross_series.iloc[-int(cross_lookback_bars):]]
        all_di_plus_cross_rows = frame.loc[di_plus_cross_series]
        latest_di_plus_cross_row = all_di_plus_cross_rows.iloc[-1] if not all_di_plus_cross_rows.empty else None
        recent_di_plus_cross_over_threshold_rows = frame.iloc[-int(cross_lookback_bars):].loc[
            di_plus_cross_over_threshold_series.iloc[-int(cross_lookback_bars):]
        ]
        all_di_plus_cross_over_threshold_rows = frame.loc[di_plus_cross_over_threshold_series]
        latest_di_plus_cross_over_threshold_row = (
            all_di_plus_cross_over_threshold_rows.iloc[-1] if not all_di_plus_cross_over_threshold_rows.empty else None
        )
        recent_cross_rows = frame.iloc[-int(cross_lookback_bars):].loc[cross_series.iloc[-int(cross_lookback_bars):]]
        all_cross_rows = frame.loc[cross_series]
        latest_cross_row = all_cross_rows.iloc[-1] if not all_cross_rows.empty else None
        recent_di_plus_cross_dates = [pd.Timestamp(value).strftime("%Y-%m-%d") for value in recent_di_plus_cross_rows["date"].tolist()]
        recent_di_plus_cross_over_threshold_dates = [
            pd.Timestamp(value).strftime("%Y-%m-%d") for value in recent_di_plus_cross_over_threshold_rows["date"].tolist()
        ]
        recent_cross_dates = [pd.Timestamp(value).strftime("%Y-%m-%d") for value in recent_cross_rows["date"].tolist()]

        latest_di_plus_cross_ts = pd.Timestamp(latest_di_plus_cross_row["date"]) if latest_di_plus_cross_row is not None else pd.NaT
        latest_adx_cross_ts = pd.Timestamp(latest_cross_row["date"]) if latest_cross_row is not None else pd.NaT
        di_plus_lead_pending = bool(
            pd.notna(latest_di_plus_cross_ts)
            and (pd.isna(latest_adx_cross_ts) or latest_di_plus_cross_ts > latest_adx_cross_ts)
            and pd.notna(latest.get("di_plus"))
            and pd.notna(latest.get("di_minus"))
            and pd.notna(latest.get("adx"))
            and float(latest.get("di_plus")) > float(latest.get("di_minus"))
            and float(latest.get("adx")) <= float(latest.get("di_minus"))
        )
        quality = _evaluate_quality_metrics(
            frame,
            latest_di_plus_cross_row,
            benchmark_daily,
            cross_lookback_bars=int(cross_lookback_bars),
            trend_fast_ma_length=int(trend_fast_ma_length),
            trend_slow_ma_length=int(trend_slow_ma_length),
            volume_avg_lookback=int(volume_avg_lookback),
            min_volume_ratio=float(min_volume_ratio),
            breakout_lookback_days=int(breakout_lookback_days),
            rs_lookback_days=int(rs_lookback_days),
            min_rs_spread_pct=float(min_rs_spread_pct),
            support_left_bars=int(support_left_bars),
            support_right_bars=int(support_right_bars),
            atr_channel_ma_length=int(atr_channel_ma_length),
            atr_channel_atr_length=int(atr_channel_atr_length),
            atr_channel_ma_type=str(atr_channel_ma_type or "EMA"),
            atr_lower1_proximity_pct=float(atr_lower1_proximity_pct),
        )

        rows.append(
            {
                "exchange": exchange,
                "symbol": symbol,
                "name": name_map.get(symbol, symbol),
                "latest_close": _to_float(latest.get("close")),
                "latest_close_date": latest.get("date"),
                "latest_di_plus": _to_float(latest.get("di_plus")),
                "latest_di_minus": _to_float(latest.get("di_minus")),
                "latest_adx": _to_float(latest.get("adx")),
                "threshold": float(threshold),
                "adx_minus_di_minus_gap": _to_float((latest.get("adx") - latest.get("di_minus")) if pd.notna(latest.get("adx")) and pd.notna(latest.get("di_minus")) else None),
                "di_plus_above_di_minus": bool(pd.notna(latest.get("di_plus")) and pd.notna(latest.get("di_minus")) and float(latest.get("di_plus")) > float(latest.get("di_minus"))),
                "di_plus_crosses_in_lookback_bars": int(len(recent_di_plus_cross_rows)),
                "recent_di_plus_cross_dates_csv": ",".join(recent_di_plus_cross_dates),
                "latest_di_plus_cross_date": pd.Timestamp(latest_di_plus_cross_row["date"]).strftime("%Y-%m-%d") if latest_di_plus_cross_row is not None else "",
                "di_plus_cross_above_di_minus_recent": bool(not recent_di_plus_cross_rows.empty),
                "di_plus_cross_above_di_minus_latest": bool(di_plus_cross_series.iloc[-1]) if len(di_plus_cross_series) else False,
                "di_plus_cross_over_threshold_count": int(len(recent_di_plus_cross_over_threshold_rows)),
                "recent_di_plus_cross_over_threshold_dates_csv": ",".join(recent_di_plus_cross_over_threshold_dates),
                "latest_di_plus_cross_over_threshold_date": (
                    pd.Timestamp(latest_di_plus_cross_over_threshold_row["date"]).strftime("%Y-%m-%d")
                    if latest_di_plus_cross_over_threshold_row is not None
                    else ""
                ),
                "di_plus_cross_over_threshold_recent": bool(not recent_di_plus_cross_over_threshold_rows.empty),
                "di_plus_cross_over_threshold_latest": (
                    bool(di_plus_cross_over_threshold_series.iloc[-1]) if len(di_plus_cross_over_threshold_series) else False
                ),
                "di_plus_lead_pending": di_plus_lead_pending,
                "adx_above_threshold": bool(pd.notna(latest.get("adx")) and float(latest.get("adx")) >= float(threshold)),
                "crosses_in_lookback_bars": int(len(recent_cross_rows)),
                "recent_cross_dates_csv": ",".join(recent_cross_dates),
                "latest_cross_date": pd.Timestamp(latest_cross_row["date"]).strftime("%Y-%m-%d") if latest_cross_row is not None else "",
                "adx_bullish_cross_above_di_minus_recent": bool(not recent_cross_rows.empty),
                "adx_bullish_cross_above_di_minus_latest": bool(cross_series.iloc[-1]) if len(cross_series) else False,
                "is_stale_symbol": False,
                **quality,
            }
        )

    stock_stats = pd.DataFrame(rows)
    if not stock_stats.empty:
        if "latest_close_date" in stock_stats.columns:
            stock_stats["latest_close_date"] = pd.to_datetime(stock_stats["latest_close_date"], errors="coerce")
        latest_market_date = max(latest_dates_seen) if latest_dates_seen else pd.NaT
        if pd.notna(latest_market_date):
            staleness_cutoff = pd.Timestamp(latest_market_date) - pd.Timedelta(days=int(max_staleness_days))
            stock_stats["is_stale_symbol"] = pd.to_datetime(stock_stats["latest_close_date"], errors="coerce") < staleness_cutoff
            stale_mask = stock_stats["is_stale_symbol"].fillna(False)
            stock_stats.loc[stale_mask, "di_plus_crosses_in_lookback_bars"] = 0
            stock_stats.loc[stale_mask, "recent_di_plus_cross_dates_csv"] = ""
            stock_stats.loc[stale_mask, "di_plus_cross_above_di_minus_recent"] = False
            stock_stats.loc[stale_mask, "di_plus_cross_above_di_minus_latest"] = False
            stock_stats.loc[stale_mask, "di_plus_cross_over_threshold_count"] = 0
            stock_stats.loc[stale_mask, "recent_di_plus_cross_over_threshold_dates_csv"] = ""
            stock_stats.loc[stale_mask, "di_plus_cross_over_threshold_recent"] = False
            stock_stats.loc[stale_mask, "di_plus_cross_over_threshold_latest"] = False
            stock_stats.loc[stale_mask, "obv_cross_sma13_count"] = 0
            stock_stats.loc[stale_mask, "recent_obv_cross_sma13_dates_csv"] = ""
            stock_stats.loc[stale_mask, "obv_cross_sma13_recent"] = False
            stock_stats.loc[stale_mask, "obv_cross_sma13_latest"] = False
            stock_stats.loc[stale_mask, "di_plus_lead_pending"] = False
            stock_stats.loc[stale_mask, "crosses_in_lookback_bars"] = 0
            stock_stats.loc[stale_mask, "recent_cross_dates_csv"] = ""
            stock_stats.loc[stale_mask, "adx_bullish_cross_above_di_minus_recent"] = False
            stock_stats.loc[stale_mask, "adx_bullish_cross_above_di_minus_latest"] = False
        numeric_columns = [
            "latest_di_plus",
            "latest_di_minus",
            "latest_adx",
            "adx_minus_di_minus_gap",
            "di_plus_crosses_in_lookback_bars",
            "di_plus_cross_over_threshold_count",
            "obv_latest",
            "obv_sma13",
            "obv_cross_sma13_count",
            "crosses_in_lookback_bars",
            "trend_fast_ma",
            "trend_slow_ma",
            "trend_fast_ma_slope",
            "trend_slow_ma_slope",
            "cross_volume_ratio",
            "breakout_level",
            "breakout_extension_pct",
            "support_level",
            "support_distance_from_level_pct",
            "atr_channel_ma",
            "atr_channel_atr",
            "atr_lower1",
            "atr_lower1_distance_pct",
            "rs_stock_return_pct",
            "rs_benchmark_return_pct",
            "relative_strength_spread_pct",
            "quality_score",
        ]
        for column in numeric_columns:
            if column in stock_stats.columns:
                stock_stats[column] = pd.to_numeric(stock_stats[column], errors="coerce")
        stock_stats = stock_stats.sort_values(
            ["quality_score", "di_plus_cross_above_di_minus_recent", "latest_di_plus_cross_date", "relative_strength_spread_pct", "cross_volume_ratio", "symbol"],
            ascending=[False, False, False, False, False, True],
            na_position="last",
        ).reset_index(drop=True)

    summary = _build_summary(
        exchange,
        len(all_symbols),
        stock_stats,
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
        support_left_bars=support_left_bars,
        support_right_bars=support_right_bars,
        atr_channel_ma_length=atr_channel_ma_length,
        atr_channel_atr_length=atr_channel_atr_length,
        atr_channel_ma_type=atr_channel_ma_type,
        atr_lower1_proximity_pct=atr_lower1_proximity_pct,
        max_staleness_days=max_staleness_days,
    )
    return AdxDiStudyResult(summary=summary, stock_stats=stock_stats)


def save_adx_di_outputs(result: AdxDiStudyResult, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "latest_summary.csv"
    stock_stats_path = output_dir / "latest_stock_stats.csv"
    pd.DataFrame([result.summary]).to_csv(summary_path, index=False)
    result.stock_stats.to_csv(stock_stats_path, index=False)
    return {"summary": summary_path, "stock_stats": stock_stats_path}


def load_adx_di_outputs(output_dir: Path) -> AdxDiStudyResult:
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

    return AdxDiStudyResult(summary=summary, stock_stats=_read(output_dir / "latest_stock_stats.csv"))


def _pine_recursive_smooth(values: pd.Series, length: int) -> pd.Series:
    series = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    if series.empty:
        return series
    smoothed = np.zeros(len(series), dtype=float)
    previous = 0.0
    for index, value in enumerate(series.to_numpy(dtype=float)):
        current = previous - (previous / float(length)) + value
        smoothed[index] = current
        previous = current
    return pd.Series(smoothed, index=series.index)


def _prepare_benchmark_daily(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["date", "close"])
    prepared = frame.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    prepared["close"] = pd.to_numeric(prepared["close"], errors="coerce")
    prepared = prepared.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return prepared[["date", "close"]]


def _evaluate_quality_metrics(
    frame: pd.DataFrame,
    latest_di_plus_cross_row: pd.Series | None,
    benchmark_daily: pd.DataFrame,
    *,
    cross_lookback_bars: int,
    trend_fast_ma_length: int,
    trend_slow_ma_length: int,
    volume_avg_lookback: int,
    min_volume_ratio: float,
    breakout_lookback_days: int,
    rs_lookback_days: int,
    min_rs_spread_pct: float,
    support_left_bars: int,
    support_right_bars: int,
    atr_channel_ma_length: int,
    atr_channel_atr_length: int,
    atr_channel_ma_type: str,
    atr_lower1_proximity_pct: float,
) -> dict[str, Any]:
    close = pd.to_numeric(frame["close"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    close_delta = close.diff()
    obv_step = pd.Series(
        np.where(close_delta > 0, volume, np.where(close_delta < 0, -volume, 0.0)),
        index=frame.index,
        dtype=float,
    )
    obv = obv_step.cumsum()
    obv_sma13 = obv.rolling(13, min_periods=13).mean()
    obv_cross_sma13_series = (
        obv.shift(1).notna()
        & obv_sma13.shift(1).notna()
        & obv.notna()
        & obv_sma13.notna()
        & (obv.shift(1) <= obv_sma13.shift(1))
        & (obv > obv_sma13)
    )
    obv_window = max(int(cross_lookback_bars), 1)
    recent_obv_cross_rows = frame.iloc[-obv_window:].loc[obv_cross_sma13_series.iloc[-obv_window:]]
    all_obv_cross_rows = frame.loc[obv_cross_sma13_series]
    latest_obv_cross_row = all_obv_cross_rows.iloc[-1] if not all_obv_cross_rows.empty else None
    recent_obv_cross_dates = [pd.Timestamp(value).strftime("%Y-%m-%d") for value in recent_obv_cross_rows["date"].tolist()]

    fast_ma = close.rolling(int(trend_fast_ma_length), min_periods=int(trend_fast_ma_length)).mean()
    slow_ma = close.rolling(int(trend_slow_ma_length), min_periods=int(trend_slow_ma_length)).mean()
    fast_slope = fast_ma.diff()
    slow_slope = slow_ma.diff()

    latest_close = _to_float(close.iloc[-1]) if len(close) else None
    latest_fast_ma = _to_float(fast_ma.iloc[-1]) if len(fast_ma) else None
    latest_slow_ma = _to_float(slow_ma.iloc[-1]) if len(slow_ma) else None
    latest_fast_slope = _to_float(fast_slope.iloc[-1]) if len(fast_slope) else None
    latest_slow_slope = _to_float(slow_slope.iloc[-1]) if len(slow_slope) else None
    latest_obv = _to_float(obv.iloc[-1]) if len(obv) else None
    latest_obv_sma13 = _to_float(obv_sma13.iloc[-1]) if len(obv_sma13) else None
    obv_above_sma13 = bool(latest_obv is not None and latest_obv_sma13 is not None and latest_obv > latest_obv_sma13)
    trend_filter_pass = bool(
        latest_close is not None
        and latest_fast_ma is not None
        and latest_slow_ma is not None
        and latest_fast_slope is not None
        and latest_slow_slope is not None
        and latest_close > latest_fast_ma > latest_slow_ma
        and latest_fast_slope > 0
        and latest_slow_slope > 0
    )

    cross_volume_ratio = pd.NA
    volume_filter_pass = False
    breakout_level = pd.NA
    breakout_extension_pct = pd.NA
    breakout_filter_pass = False
    cross_close = pd.NA
    cross_date = ""
    if latest_di_plus_cross_row is not None:
        cross_index = int(latest_di_plus_cross_row.name)
        cross_date = pd.Timestamp(latest_di_plus_cross_row["date"]).strftime("%Y-%m-%d")
        volume_avg = volume.shift(1).rolling(int(volume_avg_lookback), min_periods=max(5, min(int(volume_avg_lookback), 20))).mean()
        avg_volume_value = _to_float(volume_avg.iloc[cross_index]) if cross_index < len(volume_avg) else None
        cross_volume_value = _to_float(volume.iloc[cross_index]) if cross_index < len(volume) else None
        if avg_volume_value is not None and avg_volume_value > 0 and cross_volume_value is not None:
            cross_volume_ratio = cross_volume_value / avg_volume_value
            volume_filter_pass = bool(cross_volume_ratio >= float(min_volume_ratio))
        breakout_series = high.shift(1).rolling(int(breakout_lookback_days), min_periods=max(5, min(int(breakout_lookback_days), 20))).max()
        breakout_level_value = _to_float(breakout_series.iloc[cross_index]) if cross_index < len(breakout_series) else None
        cross_close = _to_float(close.iloc[cross_index]) if cross_index < len(close) else None
        if breakout_level_value is not None:
            breakout_level = breakout_level_value
        if breakout_level_value is not None and breakout_level_value > 0 and cross_close is not None:
            breakout_extension_pct = ((cross_close - breakout_level_value) / breakout_level_value) * 100.0
            breakout_filter_pass = bool(cross_close > breakout_level_value)

    rs_stock_return_pct, rs_benchmark_return_pct, relative_strength_spread_pct = _relative_strength_snapshot(frame, benchmark_daily, int(rs_lookback_days))
    rs_filter_pass = bool(
        pd.notna(relative_strength_spread_pct) and float(relative_strength_spread_pct) >= float(min_rs_spread_pct)
    )
    support_level, support_level_date, support_distance_from_level_pct, support_filter_pass = _support_snapshot(
        frame,
        left_bars=int(support_left_bars),
        right_bars=int(support_right_bars),
    )
    (
        atr_channel_ma,
        atr_channel_atr,
        atr_lower1,
        atr_lower1_distance_pct,
        atr_lower1_proximity_pass,
    ) = _atr_lower1_snapshot(
        frame,
        ma_length=int(atr_channel_ma_length),
        atr_length=int(atr_channel_atr_length),
        ma_type=str(atr_channel_ma_type or "EMA"),
        proximity_pct=float(atr_lower1_proximity_pct),
    )
    quality_score = int(sum([trend_filter_pass, volume_filter_pass, breakout_filter_pass, rs_filter_pass]))

    return {
        "trend_fast_ma": latest_fast_ma,
        "trend_slow_ma": latest_slow_ma,
        "trend_fast_ma_slope": latest_fast_slope,
        "trend_slow_ma_slope": latest_slow_slope,
        "trend_filter_pass": trend_filter_pass,
        "cross_date": cross_date,
        "cross_close": cross_close,
        "cross_volume_ratio": cross_volume_ratio,
        "volume_filter_pass": volume_filter_pass,
        "obv_latest": latest_obv,
        "obv_sma13": latest_obv_sma13,
        "obv_above_sma13": obv_above_sma13,
        "obv_cross_sma13_count": int(len(recent_obv_cross_rows)),
        "recent_obv_cross_sma13_dates_csv": ",".join(recent_obv_cross_dates),
        "latest_obv_cross_sma13_date": (
            pd.Timestamp(latest_obv_cross_row["date"]).strftime("%Y-%m-%d") if latest_obv_cross_row is not None else ""
        ),
        "obv_cross_sma13_recent": bool(not recent_obv_cross_rows.empty),
        "obv_cross_sma13_latest": bool(obv_cross_sma13_series.iloc[-1]) if len(obv_cross_sma13_series) else False,
        "breakout_level": breakout_level,
        "breakout_extension_pct": breakout_extension_pct,
        "breakout_filter_pass": breakout_filter_pass,
        "rs_stock_return_pct": rs_stock_return_pct,
        "rs_benchmark_return_pct": rs_benchmark_return_pct,
        "relative_strength_spread_pct": relative_strength_spread_pct,
        "rs_filter_pass": rs_filter_pass,
        "support_level": support_level,
        "support_level_date": support_level_date,
        "support_distance_from_level_pct": support_distance_from_level_pct,
        "support_filter_pass": support_filter_pass,
        "atr_channel_ma": atr_channel_ma,
        "atr_channel_atr": atr_channel_atr,
        "atr_lower1": atr_lower1,
        "atr_lower1_distance_pct": atr_lower1_distance_pct,
        "atr_lower1_proximity_pass": atr_lower1_proximity_pass,
        "quality_score": quality_score,
    }


def _support_snapshot(
    frame: pd.DataFrame,
    *,
    left_bars: int = 15,
    right_bars: int = 15,
) -> tuple[float | pd.NA, str, float | pd.NA, bool]:
    if frame.empty or int(left_bars) < 1 or int(right_bars) < 1:
        return (pd.NA, "", pd.NA, False)

    lows = pd.to_numeric(frame["low"], errors="coerce").reset_index(drop=True)
    closes = pd.to_numeric(frame["close"], errors="coerce").reset_index(drop=True)
    dates = pd.to_datetime(frame["date"], errors="coerce").reset_index(drop=True)
    if lows.empty or closes.empty or dates.empty:
        return (pd.NA, "", pd.NA, False)

    left_min = lows.shift(1).rolling(int(left_bars), min_periods=int(left_bars)).min()
    right_min = lows.iloc[::-1].shift(1).rolling(int(right_bars), min_periods=int(right_bars)).min().iloc[::-1]
    pivot_mask = left_min.notna() & right_min.notna() & lows.notna() & (lows <= left_min) & (lows < right_min)
    pivot_lows = lows.where(pivot_mask)
    pivot_dates = dates.where(pivot_mask)

    # Mirrors fixnan(ta.pivotlow(leftBars, rightBars)[1]) as a forward-held latest pivot support.
    support_series = pivot_lows.shift(1).ffill()
    support_date_series = pivot_dates.shift(1).ffill()

    latest_close = _to_float(closes.iloc[-1]) if len(closes) else None
    latest_support = _to_float(support_series.iloc[-1]) if len(support_series) else None
    latest_support_date = ""
    latest_support_ts = support_date_series.iloc[-1] if len(support_date_series) else pd.NaT
    if pd.notna(latest_support_ts):
        latest_support_date = pd.Timestamp(latest_support_ts).strftime("%Y-%m-%d")
    support_distance_from_level_pct: float | pd.NA = pd.NA
    support_filter_pass = False
    if latest_support is not None and latest_support > 0 and latest_close is not None:
        support_distance_from_level_pct = ((latest_close - latest_support) / latest_support) * 100.0
        support_filter_pass = bool(20.0 <= float(support_distance_from_level_pct) <= 40.0)
    return (latest_support if latest_support is not None else pd.NA, latest_support_date, support_distance_from_level_pct, support_filter_pass)


def _atr_lower1_snapshot(
    frame: pd.DataFrame,
    *,
    ma_length: int = 20,
    atr_length: int = 14,
    ma_type: str = "EMA",
    proximity_pct: float = 2.0,
) -> tuple[float | pd.NA, float | pd.NA, float | pd.NA, float | pd.NA, bool]:
    if frame.empty or int(ma_length) < 1 or int(atr_length) < 1:
        return (pd.NA, pd.NA, pd.NA, pd.NA, False)

    close = pd.to_numeric(frame["close"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    if close.empty or high.empty or low.empty:
        return (pd.NA, pd.NA, pd.NA, pd.NA, False)

    ma_type_value = str(ma_type or "EMA").strip().upper()
    if ma_type_value == "SMA":
        ma = close.rolling(int(ma_length), min_periods=int(ma_length)).mean()
    else:
        ma = close.ewm(span=int(ma_length), adjust=False, min_periods=int(ma_length)).mean()

    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = _pine_rma(true_range, int(atr_length))
    lower1 = ma - atr

    latest_close = _to_float(close.iloc[-1]) if len(close) else None
    latest_ma = _to_float(ma.iloc[-1]) if len(ma) else None
    latest_atr = _to_float(atr.iloc[-1]) if len(atr) else None
    latest_lower1 = _to_float(lower1.iloc[-1]) if len(lower1) else None
    atr_lower1_distance_pct: float | pd.NA = pd.NA
    atr_lower1_proximity_pass = False
    if latest_close is not None and latest_lower1 is not None and latest_lower1 > 0:
        atr_lower1_distance_pct = abs((latest_close - latest_lower1) / latest_lower1) * 100.0
        atr_lower1_proximity_pass = bool(float(atr_lower1_distance_pct) <= float(proximity_pct))
    return (
        latest_ma if latest_ma is not None else pd.NA,
        latest_atr if latest_atr is not None else pd.NA,
        latest_lower1 if latest_lower1 is not None else pd.NA,
        atr_lower1_distance_pct,
        atr_lower1_proximity_pass,
    )


def _relative_strength_snapshot(frame: pd.DataFrame, benchmark_daily: pd.DataFrame, lookback_days: int) -> tuple[float | pd.NA, float | pd.NA, float | pd.NA]:
    if frame.empty or benchmark_daily.empty or int(lookback_days) < 1:
        return (pd.NA, pd.NA, pd.NA)
    stock = frame[["date", "close"]].copy()
    stock["date"] = pd.to_datetime(stock["date"], errors="coerce")
    stock["close"] = pd.to_numeric(stock["close"], errors="coerce")
    stock = stock.dropna(subset=["date", "close"]).sort_values("date")
    benchmark = benchmark_daily.copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"], errors="coerce")
    benchmark["close"] = pd.to_numeric(benchmark["close"], errors="coerce")
    benchmark = benchmark.dropna(subset=["date", "close"]).sort_values("date")
    merged = pd.merge(stock, benchmark, on="date", how="inner", suffixes=("_stock", "_benchmark"))
    if len(merged) <= int(lookback_days):
        return (pd.NA, pd.NA, pd.NA)
    latest = merged.iloc[-1]
    past = merged.iloc[-(int(lookback_days) + 1)]
    if float(past["close_stock"]) <= 0 or float(past["close_benchmark"]) <= 0:
        return (pd.NA, pd.NA, pd.NA)
    stock_return = ((float(latest["close_stock"]) / float(past["close_stock"])) - 1.0) * 100.0
    benchmark_return = ((float(latest["close_benchmark"]) / float(past["close_benchmark"])) - 1.0) * 100.0
    return (stock_return, benchmark_return, stock_return - benchmark_return)


def _pine_rma(values: pd.Series, length: int) -> pd.Series:
    series = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=series.index, dtype=float)
    if series.empty or int(length) < 1:
        return result
    if len(series) < int(length):
        return result
    seed = series.iloc[: int(length)].mean()
    result.iloc[int(length) - 1] = seed
    previous = seed
    for index in range(int(length), len(series)):
        value = series.iloc[index]
        if pd.isna(value):
            result.iloc[index] = previous
            continue
        previous = ((previous * (float(length) - 1.0)) + float(value)) / float(length)
        result.iloc[index] = previous
    return result


def _build_summary(
    exchange: str,
    symbols_processed: int,
    stock_stats: pd.DataFrame,
    *,
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
    support_left_bars: int,
    support_right_bars: int,
    atr_channel_ma_length: int,
    atr_channel_atr_length: int,
    atr_channel_ma_type: str,
    atr_lower1_proximity_pct: float,
    max_staleness_days: int,
) -> dict[str, Any]:
    latest_date = ""
    if not stock_stats.empty and "latest_close_date" in stock_stats.columns:
        latest_dates = pd.to_datetime(stock_stats["latest_close_date"], errors="coerce")
        if latest_dates.notna().any():
            latest_date = str(latest_dates.max().date())

    matches = (
        stock_stats["di_plus_cross_above_di_minus_recent"].astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"}).sum()
        if (not stock_stats.empty and "di_plus_cross_above_di_minus_recent" in stock_stats.columns)
        else 0
    )
    adx_series = (
        pd.to_numeric(stock_stats["latest_adx"], errors="coerce").dropna()
        if "latest_adx" in stock_stats.columns
        else pd.Series(dtype=float)
    )
    return {
        "exchange": exchange,
        "symbols_processed": symbols_processed,
        "stocks_with_history": int(len(stock_stats)),
        "adx_cross_matches": int(matches),
        "latest_close_date": latest_date,
        "avg_latest_adx": float(adx_series.mean()) if not adx_series.empty else 0.0,
        "length": int(length),
        "threshold": float(threshold),
        "cross_lookback_bars": int(cross_lookback_bars),
        "trend_fast_ma_length": int(trend_fast_ma_length),
        "trend_slow_ma_length": int(trend_slow_ma_length),
        "volume_avg_lookback": int(volume_avg_lookback),
        "min_volume_ratio": float(min_volume_ratio),
        "breakout_lookback_days": int(breakout_lookback_days),
        "rs_lookback_days": int(rs_lookback_days),
        "min_rs_spread_pct": float(min_rs_spread_pct),
        "support_left_bars": int(support_left_bars),
        "support_right_bars": int(support_right_bars),
        "atr_channel_ma_length": int(atr_channel_ma_length),
        "atr_channel_atr_length": int(atr_channel_atr_length),
        "atr_channel_ma_type": str(atr_channel_ma_type or "EMA"),
        "atr_lower1_proximity_pct": float(atr_lower1_proximity_pct),
        "max_staleness_days": int(max_staleness_days),
    }
