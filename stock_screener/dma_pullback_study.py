from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from stock_screener.adx_di_study import calculate_adx_di
from stock_screener.data.storage import Storage
from stock_screener.minervini_di_divergence_study import evaluate_di_divergence
from stock_screener.minervini_quality_study import (
    DEFAULT_BENCHMARK_SYMBOL,
    _prepare_benchmark,
    evaluate_minervini_quality,
)
from stock_screener.weekly_buy_tracker_study import _emit_progress, _load_name_map


DEFAULT_DMA_LENGTHS = (75, 100, 200)
DEFAULT_TOUCH_LOOKBACK_BARS = 5
DEFAULT_PROXIMITY_PCT = 2.0
DEFAULT_MAX_DISTANCE_ABOVE_DMA_PCT = 5.0
DEFAULT_ADX_LENGTH = 14
DEFAULT_DIVERGENCE_DAYS = 2
DEFAULT_MIN_QUALITY_SCORE = 70.0


@dataclass(frozen=True)
class DmaPullbackStudyResult:
    summary: dict[str, Any]
    stock_stats: pd.DataFrame


def evaluate_dma_pullback(
    daily: pd.DataFrame,
    *,
    dma_lengths: Iterable[int] = DEFAULT_DMA_LENGTHS,
    touch_lookback_bars: int = DEFAULT_TOUCH_LOOKBACK_BARS,
    proximity_pct: float = DEFAULT_PROXIMITY_PCT,
    max_distance_above_dma_pct: float = DEFAULT_MAX_DISTANCE_ABOVE_DMA_PCT,
) -> dict[str, Any]:
    frame = _prepare_daily(daily)
    lengths = tuple(sorted({max(int(value), 2) for value in dma_lengths}))
    minimum_history = max(lengths, default=0) + max(int(touch_lookback_bars), 1)
    if frame.empty or len(frame) < minimum_history:
        return _empty_pullback_metrics(lengths, "SHORT_HISTORY")

    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    dates = frame["date"]
    latest_close = float(close.iloc[-1])
    previous_close = float(close.iloc[-2])
    upward_turn = bool(latest_close > previous_close)
    touch_window = max(int(touch_lookback_bars), 1)
    zone_fraction = max(float(proximity_pct), 0.0) / 100.0
    max_distance = max(float(max_distance_above_dma_pct), 0.0)

    result: dict[str, Any] = {
        "latest_date": dates.iloc[-1].strftime("%Y-%m-%d"),
        "latest_close": latest_close,
        "upward_turn": upward_turn,
        "data_status": "READY",
    }
    matched_lengths: list[int] = []
    matched_distances: list[float] = []
    latest_touch_date = pd.NaT

    for length in lengths:
        moving_average = close.rolling(length, min_periods=length).mean()
        lower_zone = moving_average * (1.0 - zone_fraction)
        upper_zone = moving_average * (1.0 + zone_fraction)
        # A candle touches the support zone whenever its price range intersects
        # the band around the moving average.
        touch = (low <= upper_zone) & (high >= lower_zone)
        recent_touch = touch.iloc[-touch_window:]
        touched = bool(recent_touch.fillna(False).any())
        touch_date = (
            dates.loc[recent_touch.index[recent_touch.fillna(False)]].iloc[-1]
            if touched
            else pd.NaT
        )
        latest_dma = float(moving_average.iloc[-1])
        distance_pct = (
            (latest_close / latest_dma - 1.0) * 100.0
            if np.isfinite(latest_dma) and latest_dma > 0.0
            else np.nan
        )
        recovered_above = bool(
            np.isfinite(distance_pct)
            and 0.0 <= distance_pct <= max_distance
            and upward_turn
        )
        passed = bool(touched and recovered_above)
        if passed:
            matched_lengths.append(length)
            matched_distances.append(distance_pct)
            if pd.isna(latest_touch_date) or touch_date > latest_touch_date:
                latest_touch_date = touch_date

        result[f"dma_{length}"] = latest_dma
        result[f"distance_to_{length}dma_pct"] = distance_pct
        result[f"touch_{length}dma_recent"] = touched
        result[f"touch_{length}dma_date"] = (
            touch_date.strftime("%Y-%m-%d") if pd.notna(touch_date) else ""
        )
        result[f"pullback_from_{length}dma"] = passed

    result["dma_pullback_pass"] = bool(matched_lengths)
    result["matched_dma"] = ", ".join(str(value) for value in matched_lengths)
    result["nearest_dma_distance_pct"] = (
        min(matched_distances) if matched_distances else np.nan
    )
    result["latest_touch_date"] = (
        latest_touch_date.strftime("%Y-%m-%d") if pd.notna(latest_touch_date) else ""
    )
    return result


def run_dma_pullback_study(
    storage: Storage,
    exchange: str = "NSE",
    *,
    symbols: list[str] | None = None,
    dma_lengths: Iterable[int] = DEFAULT_DMA_LENGTHS,
    touch_lookback_bars: int = DEFAULT_TOUCH_LOOKBACK_BARS,
    proximity_pct: float = DEFAULT_PROXIMITY_PCT,
    max_distance_above_dma_pct: float = DEFAULT_MAX_DISTANCE_ABOVE_DMA_PCT,
    require_di_minervini: bool = False,
    adx_length: int = DEFAULT_ADX_LENGTH,
    divergence_days: int = DEFAULT_DIVERGENCE_DAYS,
    min_quality_score: float = DEFAULT_MIN_QUALITY_SCORE,
    benchmark_symbol: str = DEFAULT_BENCHMARK_SYMBOL,
    as_of_date: Any | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> DmaPullbackStudyResult:
    cutoff = pd.to_datetime(as_of_date, errors="coerce") if as_of_date is not None else pd.NaT
    benchmark_source = storage.load_candles("NSE_INDEX", benchmark_symbol, "1D")
    if pd.notna(cutoff) and not benchmark_source.empty:
        benchmark_dates = pd.to_datetime(benchmark_source.get("date"), errors="coerce", format="mixed")
        benchmark_source = benchmark_source.loc[
            benchmark_dates.dt.normalize() <= pd.Timestamp(cutoff).normalize()
        ].copy()
    benchmark = _prepare_benchmark(benchmark_source)
    if benchmark.empty:
        raise RuntimeError(
            f"{benchmark_symbol} daily candles are unavailable. Refresh Kite data and run the scan again."
        )

    if symbols is None:
        all_symbols = sorted(
            path.stem
            for path in (storage.data_root / "candles" / exchange / "1D").glob("*.csv")
        )
    else:
        all_symbols = sorted({str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()})
    name_map = _load_name_map(storage, exchange)
    rows: list[dict[str, Any]] = []
    short_history = 0

    _emit_progress(
        progress_callback,
        phase="Scanning DMA pullbacks",
        completed=0,
        total=len(all_symbols),
        current_symbol="",
        current_exchange=exchange,
    )
    for index, symbol in enumerate(all_symbols, start=1):
        daily = storage.load_candles(exchange, symbol, "1D")
        if pd.notna(cutoff) and not daily.empty:
            dates = pd.to_datetime(daily.get("date"), errors="coerce", format="mixed")
            daily = daily.loc[dates.dt.normalize() <= pd.Timestamp(cutoff).normalize()].copy()
        pullback = evaluate_dma_pullback(
            daily,
            dma_lengths=dma_lengths,
            touch_lookback_bars=touch_lookback_bars,
            proximity_pct=proximity_pct,
            max_distance_above_dma_pct=max_distance_above_dma_pct,
        )
        _emit_progress(
            progress_callback,
            phase="Scanning DMA pullbacks",
            completed=index,
            total=len(all_symbols),
            current_symbol=symbol,
            current_exchange=exchange,
        )
        if pullback.get("data_status") != "READY":
            short_history += 1
            continue

        di_minervini_pass = False
        quality: dict[str, Any] = {}
        divergence: dict[str, Any] = {}
        if pullback["dma_pullback_pass"] and require_di_minervini:
            adx_frame = calculate_adx_di(daily, length=int(adx_length), threshold=20.0)
            divergence = evaluate_di_divergence(
                adx_frame, divergence_days=int(divergence_days)
            )
            quality = evaluate_minervini_quality(
                daily, benchmark, score_threshold=float(min_quality_score)
            )
            quality_pass = bool(
                quality.get("data_status") == "READY"
                and _score_at_least(quality.get("stock_quality_score"), min_quality_score)
                and _score_at_least(quality.get("setup_quality_score"), min_quality_score)
                and _score_at_least(quality.get("entry_quality_score"), min_quality_score)
            )
            di_minervini_pass = bool(
                divergence.get("di_divergence_pass", False) and quality_pass
            )

        combined_pass = bool(
            pullback["dma_pullback_pass"]
            and (not require_di_minervini or di_minervini_pass)
        )
        rows.append(
            {
                "exchange": exchange,
                "symbol": symbol,
                "name": name_map.get(symbol, symbol),
                **pullback,
                "latest_di_plus": divergence.get("latest_di_plus"),
                "latest_di_minus": divergence.get("latest_di_minus"),
                "di_divergence_pass": divergence.get("di_divergence_pass", False),
                "stock_quality_score": quality.get("stock_quality_score"),
                "setup_quality_score": quality.get("setup_quality_score"),
                "entry_quality_score": quality.get("entry_quality_score"),
                "di_minervini_pass": di_minervini_pass,
                "combined_pass": combined_pass,
            }
        )

    stock_stats = pd.DataFrame(rows)
    benchmark_latest_date = pd.Timestamp(benchmark.iloc[-1]["date"]).normalize()
    if not stock_stats.empty:
        latest_dates = pd.to_datetime(stock_stats["latest_date"], errors="coerce").dt.normalize()
        stock_stats["is_latest_market_date"] = latest_dates.eq(benchmark_latest_date)
        stock_stats["combined_pass"] = (
            stock_stats["combined_pass"].fillna(False).astype(bool)
            & stock_stats["is_latest_market_date"]
        )
        stock_stats = stock_stats.sort_values(
            ["combined_pass", "dma_pullback_pass", "latest_touch_date", "symbol"],
            ascending=[False, False, False, True],
            na_position="last",
        ).reset_index(drop=True)

    matches = (
        stock_stats.loc[stock_stats["combined_pass"].fillna(False).astype(bool)]
        if not stock_stats.empty
        else pd.DataFrame()
    )
    latest_stock_date = pd.to_datetime(
        stock_stats.get("latest_date", pd.Series(dtype="object")),
        errors="coerce",
    ).max()
    summary = {
        "exchange": exchange,
        "symbols_processed": len(all_symbols),
        "stocks_evaluated": len(stock_stats),
        "short_or_missing_history": short_history,
        "dma_pullback_matches": int(
            stock_stats["dma_pullback_pass"].fillna(False).astype(bool).sum()
        ) if not stock_stats.empty else 0,
        "di_minervini_matches": int(
            stock_stats["di_minervini_pass"].fillna(False).astype(bool).sum()
        ) if not stock_stats.empty else 0,
        "combined_matches": len(matches),
        "require_di_minervini": bool(require_di_minervini),
        "dma_lengths": ",".join(str(value) for value in dma_lengths),
        "touch_lookback_bars": int(touch_lookback_bars),
        "proximity_pct": float(proximity_pct),
        "max_distance_above_dma_pct": float(max_distance_above_dma_pct),
        "adx_length": int(adx_length),
        "divergence_days": int(divergence_days),
        "min_quality_score": float(min_quality_score),
        "benchmark_symbol": benchmark_symbol,
        "benchmark_latest_date": benchmark_latest_date.strftime("%Y-%m-%d"),
        "analysis_as_of_date": benchmark_latest_date.strftime("%Y-%m-%d"),
        "latest_stock_date": latest_stock_date.strftime("%Y-%m-%d") if pd.notna(latest_stock_date) else "",
        "generated_at_ist": pd.Timestamp.now(tz="Asia/Kolkata").strftime(
            "%Y-%m-%d %H:%M:%S IST"
        ),
    }
    return DmaPullbackStudyResult(summary=summary, stock_stats=stock_stats)


def save_dma_pullback_outputs(result: DmaPullbackStudyResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result.stock_stats.to_csv(output_dir / "stock_stats.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(result.summary, indent=2, default=str), encoding="utf-8"
    )


def load_dma_pullback_outputs(output_dir: Path) -> DmaPullbackStudyResult:
    summary_path = output_dir / "summary.json"
    stats_path = output_dir / "stock_stats.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    stock_stats = pd.read_csv(stats_path) if stats_path.exists() else pd.DataFrame()
    return DmaPullbackStudyResult(summary=summary, stock_stats=stock_stats)


def _prepare_daily(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame.get("date"), errors="coerce", format="mixed")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    return (
        frame.dropna(subset=["date", "open", "high", "low", "close"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )


def _score_at_least(value: Any, threshold: float) -> bool:
    try:
        return bool(float(value) >= float(threshold))
    except (TypeError, ValueError):
        return False


def _empty_pullback_metrics(lengths: tuple[int, ...], status: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "latest_date": "",
        "latest_close": np.nan,
        "upward_turn": False,
        "dma_pullback_pass": False,
        "matched_dma": "",
        "nearest_dma_distance_pct": np.nan,
        "latest_touch_date": "",
        "data_status": status,
    }
    for length in lengths:
        result.update(
            {
                f"dma_{length}": np.nan,
                f"distance_to_{length}dma_pct": np.nan,
                f"touch_{length}dma_recent": False,
                f"touch_{length}dma_date": "",
                f"pullback_from_{length}dma": False,
            }
        )
    return result
