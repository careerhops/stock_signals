from __future__ import annotations

# Score formulas are derived from the user-provided "TraderSetup V2.2" Pine
# indicator and its MPL-2.0 Minervini/VCP source attribution to noam73.

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from stock_screener.data.storage import Storage
from stock_screener.weekly_buy_tracker_study import _emit_progress, _load_name_map, _to_float


DEFAULT_BENCHMARK_SYMBOL = "NIFTY 500"
DEFAULT_SCORE_THRESHOLD = 70.0


@dataclass(frozen=True)
class MinerviniQualityStudyResult:
    summary: dict[str, Any]
    stock_stats: pd.DataFrame


def run_minervini_quality_study(
    storage: Storage,
    exchange: str = "NSE",
    *,
    benchmark_symbol: str = DEFAULT_BENCHMARK_SYMBOL,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> MinerviniQualityStudyResult:
    benchmark = _prepare_benchmark(storage.load_candles("NSE_INDEX", benchmark_symbol, "1D"))
    if benchmark.empty:
        raise RuntimeError(
            f"{benchmark_symbol} daily candles are unavailable. Refresh Kite data and run the scan again."
        )

    data_root = storage.data_root
    all_symbols = sorted(path.stem for path in (data_root / "candles" / exchange / "1D").glob("*.csv"))
    name_map = _load_name_map(storage, exchange)
    rows: list[dict[str, Any]] = []

    _emit_progress(
        progress_callback,
        phase="Scoring extended Minervini quality",
        completed=0,
        total=len(all_symbols),
        current_symbol="",
        current_exchange=exchange,
    )

    for index, symbol in enumerate(all_symbols, start=1):
        daily = _prepare_daily(storage.load_candles(exchange, symbol, "1D"))
        _emit_progress(
            progress_callback,
            phase="Scoring extended Minervini quality",
            completed=index,
            total=len(all_symbols),
            current_symbol=symbol,
            current_exchange=exchange,
        )
        if daily.empty:
            continue

        metrics = evaluate_minervini_quality(
            daily,
            benchmark,
            score_threshold=score_threshold,
        )
        rows.append(
            {
                "exchange": exchange,
                "symbol": symbol,
                "name": name_map.get(symbol, symbol),
                **metrics,
            }
        )

    stock_stats = pd.DataFrame(rows)
    if not stock_stats.empty:
        numeric_columns = (
            "latest_close",
            "stock_quality_score",
            "setup_quality_score",
            "entry_quality_score",
            "trend_pass_count",
            "relative_performance_pct",
            "avg_turnover_cr",
            "vcp_score",
            "pressure_pct",
            "distribution_count_20d",
            "volume_dry_ratio",
            "rvol",
            "pivot_distance_pct",
            "distance_from_sma50_pct",
            "atr_extension",
        )
        for column in numeric_columns:
            if column in stock_stats.columns:
                stock_stats[column] = pd.to_numeric(stock_stats[column], errors="coerce")
        stock_stats = stock_stats.sort_values(
            ["quality_pass", "entry_quality_score", "setup_quality_score", "stock_quality_score", "symbol"],
            ascending=[False, False, False, False, True],
            na_position="last",
        ).reset_index(drop=True)

    latest_dates = pd.to_datetime(stock_stats.get("latest_date", pd.Series(dtype="object")), errors="coerce").dropna()
    qualified = (
        stock_stats[stock_stats["quality_pass"].fillna(False).astype(bool)]
        if not stock_stats.empty and "quality_pass" in stock_stats.columns
        else pd.DataFrame()
    )
    summary = {
        "exchange": exchange,
        "benchmark_symbol": benchmark_symbol,
        "benchmark_latest_date": benchmark.iloc[-1]["date"].strftime("%Y-%m-%d"),
        "symbols_processed": len(all_symbols),
        "stocks_scored": len(stock_stats),
        "qualified_stocks": len(qualified),
        "missing_or_short_history": int(
            (stock_stats.get("data_status", pd.Series(dtype="object")) != "READY").sum()
        ),
        "score_threshold": float(score_threshold),
        "latest_stock_date": latest_dates.max().strftime("%Y-%m-%d") if not latest_dates.empty else "",
    }
    return MinerviniQualityStudyResult(summary=summary, stock_stats=stock_stats)


def evaluate_minervini_quality(
    daily: pd.DataFrame,
    benchmark_daily: pd.DataFrame,
    *,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> dict[str, Any]:
    frame = _prepare_daily(daily)
    benchmark = _prepare_benchmark(benchmark_daily)
    if frame.empty:
        return _empty_metrics(score_threshold, "NO_HISTORY")

    close = frame["close"]
    open_ = frame["open"]
    high = frame["high"]
    low = frame["low"]
    volume = frame["volume"]

    benchmark_aligned = pd.merge_asof(
        frame[["date"]].sort_values("date"),
        benchmark.rename(columns={"close": "benchmark_close"}).sort_values("date"),
        on="date",
        direction="backward",
    )["benchmark_close"]

    sma50 = close.rolling(50, min_periods=50).mean()
    sma150 = close.rolling(150, min_periods=150).mean()
    sma200 = close.rolling(200, min_periods=200).mean()
    high52 = high.rolling(252, min_periods=252).max()
    low52 = low.rolling(252, min_periods=252).min()

    trend_flags = (
        (close > sma150) & (close > sma200),
        sma150 > sma200,
        sma200 > sma200.shift(20),
        (sma50 > sma150) & (sma50 > sma200),
        close > sma50,
        close > low52 * 1.25,
        close > high52 * 0.75,
    )
    trend_pass_count = sum(int(bool(flag.iloc[-1])) for flag in trend_flags)

    true_range = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = _pine_rma(true_range, 14)
    atr_ma = close.ewm(span=20, adjust=False).mean()

    distance_from_sma50 = _safe_pct_difference(close, sma50)
    atr_extension = (close - atr_ma) / atr.replace(0, np.nan)

    up_volume = volume.where(close > close.shift(1), 0.0)
    down_volume = volume.where(close < close.shift(1), 0.0)
    buy_volume20 = up_volume.rolling(20, min_periods=20).sum()
    sell_volume20 = down_volume.rolling(20, min_periods=20).sum()
    pressure_total = buy_volume20 + sell_volume20
    pressure_pct = 100.0 * buy_volume20 / pressure_total.replace(0, np.nan)
    pressure_pct = pressure_pct.fillna(50.0)

    stock_weighted = _weighted_performance(close)
    benchmark_weighted = _weighted_performance(benchmark_aligned)
    relative_performance = stock_weighted - benchmark_weighted

    rs_line = close / benchmark_aligned.replace(0, np.nan)
    rs_line_high252 = rs_line.rolling(252, min_periods=252).max()
    rs_line_distance_pct = _safe_pct_difference(rs_line, rs_line_high252)
    rs_line_near_high = rs_line.notna() & rs_line_high252.notna() & (rs_line >= rs_line_high252 * 0.98)

    benchmark_ma50 = benchmark_aligned.rolling(50, min_periods=50).mean()
    benchmark_ma200 = benchmark_aligned.rolling(200, min_periods=200).mean()
    market_bullish = (benchmark_aligned > benchmark_ma50) & (benchmark_ma50 > benchmark_ma200)
    market_caution = (benchmark_aligned > benchmark_ma200) & ~market_bullish

    range20 = (high.rolling(20, min_periods=20).max() - low.rolling(20, min_periods=20).min()) / close.replace(0, np.nan) * 100.0
    range10 = (high.rolling(10, min_periods=10).max() - low.rolling(10, min_periods=10).min()) / close.replace(0, np.nan) * 100.0
    range5 = (high.rolling(5, min_periods=5).max() - low.rolling(5, min_periods=5).min()) / close.replace(0, np.nan) * 100.0
    contraction_sequence = (range5 < range10) & (range10 < range20)
    tight5_day_range = range5 <= 4.0

    atr_pct = atr / close.replace(0, np.nan) * 100.0
    atr_pct5 = atr_pct.rolling(5, min_periods=5).mean()
    atr_pct20 = atr_pct.rolling(20, min_periods=20).mean()
    atr_tight = atr_pct5 < atr_pct20 * 0.80

    vol5 = volume.rolling(5, min_periods=5).mean()
    vol20 = volume.rolling(20, min_periods=20).mean()
    volume_dry_ratio = vol5 / vol20.replace(0, np.nan)
    volume_dry = volume_dry_ratio < 0.70
    volume_nearly_dry = volume_dry_ratio <= 0.85
    near_52w_high = close >= high52 * 0.85
    vcp_score = (
        contraction_sequence.fillna(False).astype(int)
        + tight5_day_range.fillna(False).astype(int)
        + atr_tight.fillna(False).astype(int)
        + volume_dry.fillna(False).astype(int)
        + near_52w_high.fillna(False).astype(int)
    )

    momentum_pivot = high.rolling(20, min_periods=20).max().shift(1)
    pivot_distance = _safe_pct_difference(close, momentum_pivot)
    near_pivot = pivot_distance.between(-5.0, 0.0, inclusive="both")
    just_above_pivot = pivot_distance.gt(0.0) & pivot_distance.le(2.0)
    pivot_breakout = (
        (close.shift(1) <= momentum_pivot.shift(1))
        & (close > momentum_pivot)
    )

    body = (close - open_).abs()
    upper_wick = high - pd.concat([open_, close], axis=1).max(axis=1)
    good_bull_candle = (close > open_) & body.gt(0) & (upper_wick <= body * 0.50)
    rvol = volume / vol20.replace(0, np.nan)
    pivot_breakout_confirmed = (
        pivot_breakout
        & good_bull_candle
        & (rvol >= 1.40)
        & (close > momentum_pivot + atr * 0.10)
    )

    distribution_day = (close < close.shift(1)) & (volume > volume.shift(1))
    distribution_count = distribution_day.astype(float).rolling(20, min_periods=20).sum()

    avg_turnover_cr = (close * volume).rolling(20, min_periods=20).mean() / 10_000_000.0

    volume_direction = pd.Series(
        np.where(close > close.shift(1), volume, np.where(close < close.shift(1), -volume, 0.0)),
        index=frame.index,
        dtype=float,
    )
    obv = volume_direction.cumsum()
    obv_sma100 = obv.rolling(100, min_periods=100).mean()
    obv_bullish = (obv > obv_sma100) & (obv_sma100 > obv_sma100.shift(5))
    obv_bearish = (obv < obv_sma100) & (obv_sma100 < obv_sma100.shift(5))

    i = len(frame) - 1
    latest_relative = _to_float(relative_performance.iloc[i])
    latest_turnover = _to_float(avg_turnover_cr.iloc[i])
    latest_rs_distance = _to_float(rs_line_distance_pct.iloc[i])
    latest_vcp_score = int(vcp_score.iloc[i])
    latest_pressure = _to_float(pressure_pct.iloc[i])
    latest_distribution = _to_float(distribution_count.iloc[i])

    stock_trend_points = trend_pass_count / 7.0 * 40.0
    stock_relative_points = _relative_points(latest_relative)
    stock_rs_points = 20.0 if bool(rs_line_near_high.iloc[i]) else 10.0 if latest_rs_distance is not None and latest_rs_distance >= -5.0 else 0.0
    stock_liquidity_points = _liquidity_points(latest_turnover, 10.0)
    stock_score = _clamp_score(stock_trend_points + stock_relative_points + stock_rs_points + stock_liquidity_points)

    setup_vcp_points = latest_vcp_score / 5.0 * 50.0
    setup_pressure_points = _pressure_points(latest_pressure)
    setup_distribution_points = _distribution_points(latest_distribution)
    latest_obv_bullish = bool(obv_bullish.iloc[i])
    latest_obv_bearish = bool(obv_bearish.iloc[i])
    setup_obv_points = 15.0 if latest_obv_bullish else 0.0 if latest_obv_bearish else 7.0
    latest_volume_dry = bool(volume_dry.iloc[i])
    latest_volume_nearly_dry = bool(volume_nearly_dry.iloc[i])
    setup_volume_points = 5.0 if latest_volume_dry else 2.0 if latest_volume_nearly_dry else 0.0
    setup_score = _clamp_score(
        setup_vcp_points
        + setup_pressure_points
        + setup_distribution_points
        + setup_obv_points
        + setup_volume_points
    )

    latest_pivot_distance = _to_float(pivot_distance.iloc[i])
    latest_distance50 = _to_float(distance_from_sma50.iloc[i])
    latest_atr_extension = _to_float(atr_extension.iloc[i])
    latest_rvol = _to_float(rvol.iloc[i])
    entry_pivot_points = _entry_pivot_points(latest_pivot_distance)
    entry_extension_points = _entry_extension_points(latest_distance50, latest_atr_extension)
    entry_confirmation_points = _entry_confirmation_points(
        pivot_breakout_confirmed=bool(pivot_breakout_confirmed.iloc[i]),
        just_above_pivot=bool(just_above_pivot.iloc[i]),
        near_pivot=bool(near_pivot.iloc[i]),
        rvol=latest_rvol,
        bullish_close=bool(close.iloc[i] > open_.iloc[i]),
        good_bull_candle=bool(good_bull_candle.iloc[i]),
    )
    entry_score = _clamp_score(entry_pivot_points + entry_extension_points + entry_confirmation_points)

    required_values = (
        latest_relative,
        latest_turnover,
        latest_rs_distance,
        latest_distribution,
        latest_pivot_distance,
        latest_distance50,
        latest_atr_extension,
    )
    data_status = "READY" if len(frame) >= 253 and all(value is not None for value in required_values) else "SHORT_HISTORY"
    stock_quality_green = stock_score >= 75.0
    setup_quality_green = setup_score >= 70.0
    entry_quality_green = entry_score >= 70.0
    quality_pass = bool(
        data_status == "READY"
        and stock_quality_green
        and setup_quality_green
        and entry_quality_green
        and stock_score > float(score_threshold)
        and setup_score > float(score_threshold)
        and entry_score > float(score_threshold)
    )

    market_regime = "BULLISH" if bool(market_bullish.iloc[i]) else "CAUTION" if bool(market_caution.iloc[i]) else "WEAK"
    obv_state = "ACCUMULATING" if latest_obv_bullish else "DISTRIBUTING" if latest_obv_bearish else "NEUTRAL"
    return {
        "latest_date": frame.iloc[i]["date"].strftime("%Y-%m-%d"),
        "latest_close": _to_float(close.iloc[i]),
        "data_status": data_status,
        "market_regime": market_regime,
        "stock_quality_score": stock_score,
        "stock_quality_grade": _stock_grade(stock_score),
        "stock_quality_green": stock_quality_green,
        "setup_quality_score": setup_score,
        "setup_quality_grade": _setup_grade(setup_score),
        "setup_quality_green": setup_quality_green,
        "entry_quality_score": entry_score,
        "entry_quality_grade": _entry_grade(entry_score, latest_pivot_distance, latest_distance50, latest_atr_extension),
        "entry_quality_green": entry_quality_green,
        "quality_pass": quality_pass,
        "score_threshold": float(score_threshold),
        "trend_pass_count": trend_pass_count,
        "relative_performance_pct": latest_relative,
        "rs_line_near_high": bool(rs_line_near_high.iloc[i]),
        "avg_turnover_cr": latest_turnover,
        "vcp_score": latest_vcp_score,
        "pressure_pct": latest_pressure,
        "distribution_count_20d": latest_distribution,
        "obv_state": obv_state,
        "volume_dry_ratio": _to_float(volume_dry_ratio.iloc[i]),
        "rvol": latest_rvol,
        "pivot_distance_pct": latest_pivot_distance,
        "distance_from_sma50_pct": latest_distance50,
        "atr_extension": latest_atr_extension,
        "pivot_breakout_confirmed": bool(pivot_breakout_confirmed.iloc[i]),
    }


def save_minervini_quality_outputs(result: MinerviniQualityStudyResult, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "latest_summary.csv"
    stock_stats_path = output_dir / "latest_stock_stats.csv"
    pd.DataFrame([result.summary]).to_csv(summary_path, index=False)
    result.stock_stats.to_csv(stock_stats_path, index=False)
    return {"summary": summary_path, "stock_stats": stock_stats_path}


def load_minervini_quality_outputs(output_dir: Path) -> MinerviniQualityStudyResult:
    summary: dict[str, Any] = {}
    summary_path = output_dir / "latest_summary.csv"
    if summary_path.exists():
        try:
            frame = pd.read_csv(summary_path)
            if not frame.empty:
                summary = frame.iloc[0].to_dict()
        except pd.errors.EmptyDataError:
            pass

    stock_stats_path = output_dir / "latest_stock_stats.csv"
    try:
        stock_stats = pd.read_csv(stock_stats_path) if stock_stats_path.exists() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        stock_stats = pd.DataFrame()
    return MinerviniQualityStudyResult(summary=summary, stock_stats=stock_stats)


def _prepare_daily(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    prepared = frame.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    return prepared.dropna(subset=["date", "open", "high", "low", "close", "volume"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _prepare_benchmark(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["date", "close"])
    prepared = frame.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    prepared["close"] = pd.to_numeric(prepared["close"], errors="coerce")
    return prepared.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last")[["date", "close"]].reset_index(drop=True)


def _pine_rma(series: pd.Series, length: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    result = pd.Series(np.nan, index=values.index, dtype=float)
    if len(values) < length:
        return result
    seed = values.iloc[:length].mean()
    if pd.isna(seed):
        return result
    result.iloc[length - 1] = seed
    previous = float(seed)
    for index in range(length, len(values)):
        value = values.iloc[index]
        if pd.isna(value):
            continue
        previous = (previous * (length - 1) + float(value)) / float(length)
        result.iloc[index] = previous
    return result


def _weighted_performance(series: pd.Series) -> pd.Series:
    def roc(length: int) -> pd.Series:
        prior = series.shift(length)
        return (series - prior) / prior.replace(0, np.nan) * 100.0

    return roc(63) * 0.40 + roc(126) * 0.20 + roc(189) * 0.20 + roc(252) * 0.20


def _safe_pct_difference(value: pd.Series, reference: pd.Series) -> pd.Series:
    return (value - reference) / reference.replace(0, np.nan) * 100.0


def _relative_points(value: float | None) -> float:
    if value is None:
        return 0.0
    if value >= 25.0:
        return 25.0
    if value >= 15.0:
        return 21.0
    if value >= 5.0:
        return 14.0
    if value > 0.0:
        return 8.0
    return 0.0


def _liquidity_points(value: float | None, minimum: float) -> float:
    if value is None:
        return 0.0
    if value >= minimum * 2.0:
        return 15.0
    if value >= minimum:
        return 10.0
    if value >= minimum * 0.5:
        return 5.0
    return 0.0


def _pressure_points(value: float | None) -> float:
    if value is None:
        return 0.0
    if value >= 65.0:
        return 15.0
    if value >= 60.0:
        return 12.0
    if value >= 55.0:
        return 8.0
    if value >= 45.0:
        return 4.0
    return 0.0


def _distribution_points(value: float | None) -> float:
    if value is None:
        return 0.0
    if value <= 2.0:
        return 15.0
    if value <= 3.0:
        return 12.0
    if value <= 4.0:
        return 8.0
    if value <= 5.0:
        return 4.0
    return 0.0


def _entry_pivot_points(distance: float | None) -> float:
    if distance is None:
        return 0.0
    if -3.0 <= distance <= 0.0:
        return 50.0
    if 0.0 < distance <= 2.0:
        return 45.0
    if -5.0 <= distance < -3.0:
        return 40.0
    if 2.0 < distance <= 5.0 or -10.0 <= distance < -5.0:
        return 20.0
    return 0.0


def _entry_extension_points(distance50: float | None, atr_extension: float | None) -> float:
    if distance50 is None or atr_extension is None:
        return 0.0
    if 0.0 <= distance50 <= 5.0 and atr_extension <= 1.5:
        return 30.0
    if 5.0 < distance50 <= 10.0 and atr_extension <= 2.0:
        return 25.0
    if 10.0 < distance50 <= 15.0 and atr_extension <= 3.0:
        return 10.0
    return 0.0


def _entry_confirmation_points(
    *,
    pivot_breakout_confirmed: bool,
    just_above_pivot: bool,
    near_pivot: bool,
    rvol: float | None,
    bullish_close: bool,
    good_bull_candle: bool,
) -> float:
    if pivot_breakout_confirmed:
        return 20.0
    if just_above_pivot and rvol is not None and rvol >= 1.40 and good_bull_candle:
        return 16.0
    if near_pivot:
        return 10.0
    if rvol is not None and rvol >= 1.40 and bullish_close:
        return 6.0
    return 0.0


def _clamp_score(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def _stock_grade(score: float) -> str:
    return "LEADER" if score >= 85 else "STRONG" if score >= 75 else "WATCH" if score >= 65 else "WEAK"


def _setup_grade(score: float) -> str:
    return "READY" if score >= 85 else "DEVELOPING" if score >= 70 else "WATCH" if score >= 55 else "WEAK"


def _entry_grade(score: float, pivot: float | None, distance50: float | None, atr_extension: float | None) -> str:
    if score >= 85:
        return "READY"
    if score >= 70:
        return "GOOD"
    if score >= 55:
        return "WATCH"
    if (pivot is not None and pivot > 5.0) or (distance50 is not None and distance50 > 15.0) or (atr_extension is not None and atr_extension > 3.0):
        return "CHASE"
    return "EARLY"


def _empty_metrics(score_threshold: float, status: str) -> dict[str, Any]:
    return {
        "latest_date": "",
        "latest_close": None,
        "data_status": status,
        "stock_quality_score": None,
        "setup_quality_score": None,
        "entry_quality_score": None,
        "quality_pass": False,
        "score_threshold": float(score_threshold),
    }
