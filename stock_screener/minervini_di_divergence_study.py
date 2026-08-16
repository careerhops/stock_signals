from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from stock_screener.adx_di_study import _is_excluded_adx_symbol, calculate_adx_di
from stock_screener.data.storage import Storage
from stock_screener.minervini_quality_study import (
    DEFAULT_BENCHMARK_SYMBOL,
    _prepare_benchmark,
    evaluate_minervini_quality,
)
from stock_screener.weekly_buy_tracker_study import _emit_progress, _load_name_map, _to_float


DEFAULT_ADX_LENGTH = 14
DEFAULT_DIVERGENCE_DAYS = 2
DEFAULT_MIN_SCORE = 70.0


@dataclass(frozen=True)
class MinerviniDiDivergenceStudyResult:
    summary: dict[str, Any]
    stock_stats: pd.DataFrame


def run_minervini_di_divergence_study(
    storage: Storage,
    exchange: str = "NSE",
    *,
    symbols: list[str] | None = None,
    adx_length: int = DEFAULT_ADX_LENGTH,
    divergence_days: int = DEFAULT_DIVERGENCE_DAYS,
    min_score: float = DEFAULT_MIN_SCORE,
    benchmark_symbol: str = DEFAULT_BENCHMARK_SYMBOL,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> MinerviniDiDivergenceStudyResult:
    benchmark = _prepare_benchmark(storage.load_candles("NSE_INDEX", benchmark_symbol, "1D"))
    if benchmark.empty:
        raise RuntimeError(
            f"{benchmark_symbol} daily candles are unavailable. Refresh Kite data and run the scan again."
        )

    data_root = storage.data_root
    if symbols is None:
        all_symbols = sorted(
            path.stem
            for path in (data_root / "candles" / exchange / "1D").glob("*.csv")
            if not _is_excluded_adx_symbol(path.stem)
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
    short_history_count = 0
    minimum_history = max(253, int(adx_length) * 3, int(divergence_days) + 2)

    _emit_progress(
        progress_callback,
        phase="Scanning DI divergence and Minervini quality",
        completed=0,
        total=len(all_symbols),
        current_symbol="",
        current_exchange=exchange,
    )

    for index, symbol in enumerate(all_symbols, start=1):
        daily = storage.load_candles(exchange, symbol, "1D")
        _emit_progress(
            progress_callback,
            phase="Scanning DI divergence and Minervini quality",
            completed=index,
            total=len(all_symbols),
            current_symbol=symbol,
            current_exchange=exchange,
        )
        if daily.empty:
            short_history_count += 1
            continue

        adx_frame = calculate_adx_di(daily, length=int(adx_length), threshold=20.0)
        if adx_frame.empty or len(adx_frame) < minimum_history:
            short_history_count += 1
            continue

        divergence = evaluate_di_divergence(adx_frame, divergence_days=int(divergence_days))
        quality = evaluate_minervini_quality(daily, benchmark, score_threshold=float(min_score))
        minervini_threshold_pass = bool(
            quality.get("data_status") == "READY"
            and _score_at_least(quality.get("stock_quality_score"), min_score)
            and _score_at_least(quality.get("setup_quality_score"), min_score)
            and _score_at_least(quality.get("entry_quality_score"), min_score)
        )
        combined_pass = bool(divergence["di_divergence_pass"] and minervini_threshold_pass)
        pre_breakout_pass = _passes_pre_breakout_watchlist(divergence, quality)
        rows.append(
            {
                "exchange": exchange,
                "symbol": symbol,
                "name": name_map.get(symbol, symbol),
                "latest_date": quality.get("latest_date", ""),
                "latest_close": quality.get("latest_close"),
                **divergence,
                "stock_quality_score": quality.get("stock_quality_score"),
                "stock_quality_grade": quality.get("stock_quality_grade", ""),
                "setup_quality_score": quality.get("setup_quality_score"),
                "setup_quality_grade": quality.get("setup_quality_grade", ""),
                "entry_quality_score": quality.get("entry_quality_score"),
                "entry_quality_grade": quality.get("entry_quality_grade", ""),
                "minervini_threshold_pass": minervini_threshold_pass,
                "combined_pass": combined_pass,
                "pre_breakout_pass": pre_breakout_pass,
                "data_status": quality.get("data_status", ""),
                "market_regime": quality.get("market_regime", ""),
                "trend_pass_count": quality.get("trend_pass_count"),
                "vcp_score": quality.get("vcp_score"),
                "relative_performance_pct": quality.get("relative_performance_pct"),
                "rs_line_near_high": quality.get("rs_line_near_high", False),
                "avg_turnover_cr": quality.get("avg_turnover_cr"),
                "pressure_pct": quality.get("pressure_pct"),
                "distribution_count_20d": quality.get("distribution_count_20d"),
                "volume_dry_ratio": quality.get("volume_dry_ratio"),
                "pivot_distance_pct": quality.get("pivot_distance_pct"),
                "distance_from_sma50_pct": quality.get("distance_from_sma50_pct"),
                "atr_extension": quality.get("atr_extension"),
                "obv_state": quality.get("obv_state", ""),
            }
        )

    stock_stats = pd.DataFrame(rows)
    if not stock_stats.empty:
        latest_date_series = pd.to_datetime(stock_stats["latest_date"], errors="coerce")
        latest_market_date = latest_date_series.max()
        stock_stats["is_latest_market_date"] = latest_date_series.eq(latest_market_date)
        stock_stats["combined_pass"] = (
            stock_stats["combined_pass"].fillna(False).astype(bool)
            & stock_stats["is_latest_market_date"]
        )
        stock_stats["pre_breakout_pass"] = (
            stock_stats["pre_breakout_pass"].fillna(False).astype(bool)
            & stock_stats["is_latest_market_date"]
        )
        numeric_columns = (
            "latest_close",
            "latest_di_plus",
            "di_plus_1d_ago",
            "di_plus_2d_ago",
            "latest_di_minus",
            "di_minus_1d_ago",
            "di_minus_2d_ago",
            "latest_di_spread",
            "spread_change_2d",
            "stock_quality_score",
            "setup_quality_score",
            "entry_quality_score",
            "trend_pass_count",
            "vcp_score",
            "relative_performance_pct",
            "avg_turnover_cr",
            "pressure_pct",
            "distribution_count_20d",
            "volume_dry_ratio",
            "pivot_distance_pct",
            "distance_from_sma50_pct",
            "atr_extension",
        )
        for column in numeric_columns:
            if column in stock_stats.columns:
                stock_stats[column] = pd.to_numeric(stock_stats[column], errors="coerce")
        stock_stats = stock_stats.sort_values(
            [
                "pre_breakout_pass",
                "combined_pass",
                "spread_change_2d",
                "entry_quality_score",
                "setup_quality_score",
                "stock_quality_score",
                "symbol",
            ],
            ascending=[False, False, False, False, False, False, True],
            na_position="last",
        ).reset_index(drop=True)

    current_date_mask = (
        stock_stats["is_latest_market_date"].fillna(False).astype(bool)
        if not stock_stats.empty
        else pd.Series(dtype=bool)
    )
    combined_matches = int(stock_stats["combined_pass"].fillna(False).astype(bool).sum()) if not stock_stats.empty else 0
    pre_breakout_matches = int(stock_stats["pre_breakout_pass"].fillna(False).astype(bool).sum()) if not stock_stats.empty else 0
    divergence_matches = int((stock_stats["di_divergence_pass"].fillna(False).astype(bool) & current_date_mask).sum()) if not stock_stats.empty else 0
    quality_matches = int((stock_stats["minervini_threshold_pass"].fillna(False).astype(bool) & current_date_mask).sum()) if not stock_stats.empty else 0
    latest_dates = pd.to_datetime(stock_stats.get("latest_date", pd.Series(dtype="object")), errors="coerce").dropna()
    summary = {
        "exchange": exchange,
        "symbols_processed": len(all_symbols),
        "stocks_evaluated": len(stock_stats),
        "short_or_missing_history": short_history_count,
        "stale_stock_dates": int((~current_date_mask).sum()) if not stock_stats.empty else 0,
        "di_divergence_matches": divergence_matches,
        "minervini_threshold_matches": quality_matches,
        "combined_matches": combined_matches,
        "pre_breakout_matches": pre_breakout_matches,
        "adx_length": int(adx_length),
        "divergence_days": int(divergence_days),
        "min_score": float(min_score),
        "benchmark_symbol": benchmark_symbol,
        "benchmark_latest_date": benchmark.iloc[-1]["date"].strftime("%Y-%m-%d"),
        "latest_stock_date": latest_dates.max().strftime("%Y-%m-%d") if not latest_dates.empty else "",
    }
    return MinerviniDiDivergenceStudyResult(summary=summary, stock_stats=stock_stats)


def evaluate_di_divergence(adx_frame: pd.DataFrame, *, divergence_days: int = DEFAULT_DIVERGENCE_DAYS) -> dict[str, Any]:
    days = max(int(divergence_days), 1)
    frame = adx_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["di_plus"] = pd.to_numeric(frame["di_plus"], errors="coerce")
    frame["di_minus"] = pd.to_numeric(frame["di_minus"], errors="coerce")
    frame = frame.dropna(subset=["date", "di_plus", "di_minus"]).sort_values("date").reset_index(drop=True)
    if len(frame) < days + 1:
        return _empty_divergence(days)

    recent = frame.iloc[-(days + 1):].copy()
    di_plus_deltas = recent["di_plus"].diff().iloc[1:]
    di_minus_deltas = recent["di_minus"].diff().iloc[1:]
    plus_rising_each_day = bool((di_plus_deltas > 0).all())
    minus_falling_each_day = bool((di_minus_deltas < 0).all())
    latest = recent.iloc[-1]
    latest_plus = _to_float(latest.get("di_plus"))
    latest_minus = _to_float(latest.get("di_minus"))
    di_plus_above_di_minus = bool(
        latest_plus is not None and latest_minus is not None and latest_plus > latest_minus
    )
    divergence_pass = bool(plus_rising_each_day and minus_falling_each_day and di_plus_above_di_minus)

    result = {
        "di_divergence_pass": divergence_pass,
        "di_plus_rising_each_day": plus_rising_each_day,
        "di_minus_falling_each_day": minus_falling_each_day,
        "di_plus_above_di_minus": di_plus_above_di_minus,
        "divergence_days": days,
        "divergence_window_start": recent.iloc[0]["date"].strftime("%Y-%m-%d"),
        "divergence_window_end": latest["date"].strftime("%Y-%m-%d"),
        "latest_di_plus": latest_plus,
        "latest_di_minus": latest_minus,
        "latest_di_spread": _to_float(latest_plus - latest_minus) if latest_plus is not None and latest_minus is not None else None,
        "spread_change_2d": _to_float(
            (latest_plus - latest_minus)
            - (float(recent.iloc[0]["di_plus"]) - float(recent.iloc[0]["di_minus"]))
        ) if latest_plus is not None and latest_minus is not None else None,
        "di_plus_1d_ago": _series_offset(recent["di_plus"], 1),
        "di_plus_2d_ago": _series_offset(recent["di_plus"], 2),
        "di_minus_1d_ago": _series_offset(recent["di_minus"], 1),
        "di_minus_2d_ago": _series_offset(recent["di_minus"], 2),
    }
    return result


def save_minervini_di_divergence_outputs(
    result: MinerviniDiDivergenceStudyResult,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "latest_summary.csv"
    stock_stats_path = output_dir / "latest_stock_stats.csv"
    pd.DataFrame([result.summary]).to_csv(summary_path, index=False)
    result.stock_stats.to_csv(stock_stats_path, index=False)
    return {"summary": summary_path, "stock_stats": stock_stats_path}


def load_minervini_di_divergence_outputs(output_dir: Path) -> MinerviniDiDivergenceStudyResult:
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
    return MinerviniDiDivergenceStudyResult(summary=summary, stock_stats=stock_stats)


def _score_at_least(value: Any, threshold: float) -> bool:
    numeric = _to_float(value)
    return bool(numeric is not None and numeric >= float(threshold))


def _passes_pre_breakout_watchlist(divergence: dict[str, Any], quality: dict[str, Any]) -> bool:
    pivot_distance = _to_float(quality.get("pivot_distance_pct"))
    distance_from_sma50 = _to_float(quality.get("distance_from_sma50_pct"))
    atr_extension = _to_float(quality.get("atr_extension"))
    turnover = _to_float(quality.get("avg_turnover_cr"))
    distribution_count = _to_float(quality.get("distribution_count_20d"))
    return bool(
        divergence.get("di_divergence_pass")
        and quality.get("data_status") == "READY"
        and str(quality.get("market_regime", "")).upper() == "BULLISH"
        and int(quality.get("trend_pass_count") or 0) == 7
        and _score_at_least(quality.get("stock_quality_score"), 85.0)
        and _score_at_least(quality.get("setup_quality_score"), 80.0)
        and _score_at_least(quality.get("entry_quality_score"), 70.0)
        and bool(quality.get("rs_line_near_high"))
        and turnover is not None
        and turnover >= 25.0
        and int(quality.get("vcp_score") or 0) >= 4
        and distribution_count is not None
        and distribution_count <= 3.0
        and str(quality.get("obv_state", "")).upper() == "ACCUMULATING"
        and pivot_distance is not None
        and -5.0 <= pivot_distance <= 0.0
        and distance_from_sma50 is not None
        and 0.0 <= distance_from_sma50 <= 10.0
        and atr_extension is not None
        and atr_extension <= 2.0
    )


def _series_offset(series: pd.Series, offset: int) -> float | None:
    return _to_float(series.iloc[-(int(offset) + 1)]) if len(series) > int(offset) else None


def _empty_divergence(days: int) -> dict[str, Any]:
    return {
        "di_divergence_pass": False,
        "di_plus_rising_each_day": False,
        "di_minus_falling_each_day": False,
        "di_plus_above_di_minus": False,
        "divergence_days": days,
        "divergence_window_start": "",
        "divergence_window_end": "",
        "latest_di_plus": None,
        "latest_di_minus": None,
        "latest_di_spread": None,
        "spread_change_2d": None,
        "di_plus_1d_ago": None,
        "di_plus_2d_ago": None,
        "di_minus_1d_ago": None,
        "di_minus_2d_ago": None,
    }
