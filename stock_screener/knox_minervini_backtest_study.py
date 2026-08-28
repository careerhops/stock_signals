from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from stock_screener.data.storage import Storage
from stock_screener.knox_envelope_study import _pine_rsi


DEFAULT_START_DATE = "2017-01-01"
DEFAULT_END_DATE = "2026-08-20"
DEFAULT_ROUND_TRIP_COST_PCT = 0.35
DEFAULT_MIN_TURNOVER_CR = 5.0
DEFAULT_BAND_PROXIMITY_PCT = 2.0
DEFAULT_MINERVINI_RECENCY_DAYS = 60


@dataclass(frozen=True)
class SignalVariant:
    name: str
    knox_lookback: int
    envelope_length: int
    envelope_percent: float
    knox_role: str = "bullish_pullback"


@dataclass(frozen=True)
class ExitVariant:
    target_pct: float
    stop_pct: float
    max_holding_days: int

    @property
    def name(self) -> str:
        return f"T{self.target_pct:g}_S{self.stop_pct:g}_H{self.max_holding_days}"


@dataclass(frozen=True)
class KnoxMinerviniBacktestResult:
    summary: dict[str, Any]
    variant_stats: pd.DataFrame
    period_stats: pd.DataFrame
    trades: pd.DataFrame


DEFAULT_SIGNAL_VARIANTS = (
    SignalVariant("K20_E20_3", 20, 20, 3.0),
    SignalVariant("K40_E20_3", 40, 20, 3.0),
    SignalVariant("K40_E20_5", 40, 20, 5.0),
    SignalVariant("K40_E50_5", 40, 50, 5.0),
    SignalVariant("K60_E20_5", 60, 20, 5.0),
    SignalVariant("K60_E50_5", 60, 50, 5.0),
    SignalVariant("BEARLEAD_K20_E20_3", 20, 20, 3.0, "bearish_lead"),
    SignalVariant("BEARLEAD_K40_E20_3", 40, 20, 3.0, "bearish_lead"),
    SignalVariant("BEARLEAD_K40_E20_5", 40, 20, 5.0, "bearish_lead"),
    SignalVariant("BEARLEAD_K40_E50_5", 40, 50, 5.0, "bearish_lead"),
    SignalVariant("BEARLEAD_K60_E20_5", 60, 20, 5.0, "bearish_lead"),
    SignalVariant("BEARLEAD_K60_E50_5", 60, 50, 5.0, "bearish_lead"),
)

DEFAULT_EXIT_VARIANTS = tuple(
    ExitVariant(target, stop, hold)
    for target in (5.0, 6.0, 7.0)
    for stop in (3.0, 4.0)
    for hold in (7, 10, 15)
)


def run_knox_minervini_backtest(
    storage: Storage,
    *,
    exchange: str = "NSE",
    benchmark_exchange: str = "NSE_INDEX",
    benchmark_symbol: str = "NIFTY 50",
    symbols: list[str] | None = None,
    start_date: str | pd.Timestamp = DEFAULT_START_DATE,
    end_date: str | pd.Timestamp = DEFAULT_END_DATE,
    signal_variants: Iterable[SignalVariant] = DEFAULT_SIGNAL_VARIANTS,
    exit_variants: Iterable[ExitVariant] = DEFAULT_EXIT_VARIANTS,
    band_proximity_pct: float = DEFAULT_BAND_PROXIMITY_PCT,
    min_turnover_cr: float = DEFAULT_MIN_TURNOVER_CR,
    round_trip_cost_pct: float = DEFAULT_ROUND_TRIP_COST_PCT,
    use_minervini_filter: bool = True,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> KnoxMinerviniBacktestResult:
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    signal_variants = tuple(signal_variants)
    exit_variants = tuple(exit_variants)
    benchmark = _prepare_daily(storage.load_candles(benchmark_exchange, benchmark_symbol, "1D"))
    if benchmark.empty:
        raise RuntimeError(f"No daily benchmark history found for {benchmark_exchange}:{benchmark_symbol}.")
    benchmark_features = _benchmark_features(benchmark)

    candidates = _candidate_symbols(storage, exchange, symbols)
    trade_frames: list[pd.DataFrame] = []
    symbols_with_history = 0
    signal_count = 0
    for completed, symbol in enumerate(candidates, start=1):
        daily = _prepare_daily(storage.load_candles(exchange, symbol, "1D"))
        if len(daily) < 280:
            _emit_progress(progress_callback, completed, len(candidates), symbol)
            continue
        daily = daily[daily["date"] <= end_ts].reset_index(drop=True)
        if len(daily) < 280 or daily["date"].max() < start_ts:
            _emit_progress(progress_callback, completed, len(candidates), symbol)
            continue
        symbols_with_history += 1
        features = calculate_knox_minervini_features(
            daily,
            benchmark_features,
            signal_variants=signal_variants,
            band_proximity_pct=band_proximity_pct,
            min_turnover_cr=min_turnover_cr,
            use_minervini_filter=use_minervini_filter,
        )
        for signal_variant in signal_variants:
            signal_column = f"signal_{signal_variant.name}"
            signal_indexes = features.index[
                features[signal_column].fillna(False)
                & features["date"].between(start_ts, end_ts)
            ].tolist()
            signal_count += len(signal_indexes)
            if not signal_indexes:
                continue
            for exit_variant in exit_variants:
                trades = _backtest_signal_frame(
                    features,
                    symbol=symbol,
                    exchange=exchange,
                    signal_indexes=signal_indexes,
                    signal_variant=signal_variant,
                    exit_variant=exit_variant,
                    end_ts=end_ts,
                    round_trip_cost_pct=round_trip_cost_pct,
                )
                if not trades.empty:
                    trade_frames.append(trades)
        _emit_progress(progress_callback, completed, len(candidates), symbol)

    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else _empty_trades()
    if not trades.empty:
        trades["period"] = trades["entry_date"].map(_period_name)
    period_stats = _aggregate(trades, ["signal_variant", "exit_variant", "period"])
    variant_stats = _build_variant_stats(period_stats)
    summary = {
        "exchange": exchange,
        "benchmark": benchmark_symbol,
        "start_date": start_ts.strftime("%Y-%m-%d"),
        "end_date": end_ts.strftime("%Y-%m-%d"),
        "symbols_considered": len(candidates),
        "symbols_with_history": symbols_with_history,
        "raw_signal_count_across_variants": signal_count,
        "trade_rows": len(trades),
        "signal_variants": len(signal_variants),
        "exit_variants": len(exit_variants),
        "round_trip_cost_pct": float(round_trip_cost_pct),
        "min_turnover_cr": float(min_turnover_cr),
        "band_proximity_pct": float(band_proximity_pct),
        "use_minervini_filter": bool(use_minervini_filter),
        "minervini_sequence": (
            "Strict template passed in prior 60 sessions; current 50/150/200 MA stack, "
            "rising 200 SMA, positive relative performance, liquidity, and bullish market retained"
            if use_minervini_filter
            else "Disabled; signals use Knoxville and Envelope only"
        ),
        "best_robust_variant": variant_stats.iloc[0]["variant"] if not variant_stats.empty else "",
    }
    return KnoxMinerviniBacktestResult(summary, variant_stats, period_stats, trades)


def calculate_knox_minervini_features(
    daily: pd.DataFrame,
    benchmark_features: pd.DataFrame,
    *,
    signal_variants: Iterable[SignalVariant] = DEFAULT_SIGNAL_VARIANTS,
    band_proximity_pct: float = DEFAULT_BAND_PROXIMITY_PCT,
    min_turnover_cr: float = DEFAULT_MIN_TURNOVER_CR,
    use_minervini_filter: bool = True,
) -> pd.DataFrame:
    frame = _prepare_daily(daily)
    if frame.empty:
        return frame
    signal_variants = tuple(signal_variants)
    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    volume = frame["volume"]

    sma50 = close.rolling(50, min_periods=50).mean()
    sma150 = close.rolling(150, min_periods=150).mean()
    sma200 = close.rolling(200, min_periods=200).mean()
    high252 = high.rolling(252, min_periods=252).max()
    low252 = low.rolling(252, min_periods=252).min()
    trend_flags = pd.concat(
        [
            close > sma150,
            close > sma200,
            sma150 > sma200,
            sma200 > sma200.shift(20),
            (sma50 > sma150) & (sma50 > sma200),
            close > sma50,
            close > low252 * 1.25,
            close > high252 * 0.75,
        ],
        axis=1,
    )
    trend_pass_count = trend_flags.fillna(False).sum(axis=1)

    aligned_benchmark = pd.merge_asof(
        frame[["date"]],
        benchmark_features.sort_values("date"),
        on="date",
        direction="backward",
    )
    stock_weighted = _weighted_performance(close)
    relative_performance = stock_weighted - aligned_benchmark["weighted_performance"]
    avg_turnover_cr = (close * volume).rolling(20, min_periods=20).mean() / 10_000_000.0
    market_bullish = aligned_benchmark["market_bullish"].eq(True)
    strict_minervini_pass = (
        (trend_pass_count == 8)
        & (relative_performance > 0.0)
        & market_bullish
        & (avg_turnover_cr >= float(min_turnover_cr))
    )
    recent_minervini_leader = (
        strict_minervini_pass.shift(1)
        .rolling(DEFAULT_MINERVINI_RECENCY_DAYS, min_periods=1)
        .max()
        .fillna(0.0)
        .astype(bool)
    )
    current_trend_intact = (
        (close > sma200)
        & (sma200 > sma200.shift(20))
        & (sma50 > sma150)
        & (sma150 > sma200)
    )
    minervini_pullback_eligible = (
        recent_minervini_leader
        & current_trend_intact
        & (relative_performance > 0.0)
        & market_bullish
        & (avg_turnover_cr >= float(min_turnover_cr))
    )

    frame["minervini_trend_count"] = trend_pass_count
    frame["relative_performance_pct"] = relative_performance
    frame["market_bullish"] = market_bullish.to_numpy()
    frame["avg_turnover_cr"] = avg_turnover_cr
    frame["strict_minervini_pass"] = strict_minervini_pass.fillna(False)
    frame["recent_minervini_leader"] = recent_minervini_leader
    frame["current_minervini_trend_intact"] = current_trend_intact.fillna(False)
    frame["minervini_pullback_eligible"] = minervini_pullback_eligible.fillna(False)
    frame["rsi14"] = _pine_rsi(close, 14)

    momentum_cache: dict[int, pd.Series] = {}
    knox_cache: dict[int, pd.Series] = {}
    bearish_knox_cache: dict[int, pd.Series] = {}
    envelope_cache: dict[tuple[int, float], tuple[pd.Series, pd.Series]] = {}
    for variant in signal_variants:
        if variant.knox_lookback not in momentum_cache:
            momentum_cache[variant.knox_lookback] = close - close.shift(20)
            knox_cache[variant.knox_lookback] = _bullish_knoxville(
                low,
                momentum_cache[variant.knox_lookback],
                frame["rsi14"],
                variant.knox_lookback,
            )
            bearish_knox_cache[variant.knox_lookback] = _bearish_knoxville(
                high,
                momentum_cache[variant.knox_lookback],
                frame["rsi14"],
                variant.knox_lookback,
            )
        envelope_key = (variant.envelope_length, variant.envelope_percent)
        if envelope_key not in envelope_cache:
            basis = close.ewm(
                span=variant.envelope_length,
                adjust=False,
                min_periods=variant.envelope_length,
            ).mean()
            lower = basis * (1.0 - variant.envelope_percent / 100.0)
            distance = (close - lower) / lower.replace(0.0, np.nan) * 100.0
            envelope_cache[envelope_key] = (lower, distance)
        lower, distance = envelope_cache[envelope_key]
        near_reclaimed_lower = (
            lower.notna()
            & (close >= lower)
            & (distance >= 0.0)
            & (distance <= float(band_proximity_pct))
        )
        frame[f"knox_{variant.name}"] = knox_cache[variant.knox_lookback]
        frame[f"bearish_knox_{variant.name}"] = bearish_knox_cache[variant.knox_lookback]
        frame[f"lower_{variant.name}"] = lower
        frame[f"distance_lower_{variant.name}"] = distance
        if variant.knox_role == "bearish_lead":
            bearish_leader_event = frame[f"bearish_knox_{variant.name}"]
            if use_minervini_filter:
                bearish_leader_event &= frame["strict_minervini_pass"]
            bearish_warning_3_to_20_days_ago = (
                bearish_leader_event.shift(3)
                .rolling(18, min_periods=1)
                .max()
                .fillna(0.0)
                .astype(bool)
            )
            context_filter = (
                frame["current_minervini_trend_intact"]
                & (relative_performance > 0.0)
                & market_bullish
                & (avg_turnover_cr >= float(min_turnover_cr))
                if use_minervini_filter
                else pd.Series(True, index=frame.index)
            )
            frame[f"signal_{variant.name}"] = (
                bearish_warning_3_to_20_days_ago
                & context_filter
                & near_reclaimed_lower
                & (close > high.shift(1))
            ).fillna(False)
        else:
            context_filter = (
                frame["minervini_pullback_eligible"]
                if use_minervini_filter
                else pd.Series(True, index=frame.index)
            )
            frame[f"signal_{variant.name}"] = (
                context_filter
                & frame[f"knox_{variant.name}"]
                & near_reclaimed_lower
            ).fillna(False)
    return frame


def save_knox_minervini_backtest(
    result: KnoxMinerviniBacktestResult,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "summary.csv",
        "variant_stats": output_dir / "variant_stats.csv",
        "period_stats": output_dir / "period_stats.csv",
        "trades": output_dir / "trades.csv",
    }
    pd.DataFrame([result.summary]).to_csv(paths["summary"], index=False)
    result.variant_stats.to_csv(paths["variant_stats"], index=False)
    result.period_stats.to_csv(paths["period_stats"], index=False)
    result.trades.to_csv(paths["trades"], index=False)
    return paths


def _bullish_knoxville(
    low: pd.Series,
    momentum: pd.Series,
    rsi: pd.Series,
    lookback: int,
) -> pd.Series:
    values_low = low.to_numpy(dtype=float)
    values_momentum = momentum.to_numpy(dtype=float)
    values_rsi = rsi.to_numpy(dtype=float)
    rolling_low = low.rolling(lookback, min_periods=lookback).min().to_numpy(dtype=float)
    result = np.zeros(len(low), dtype=bool)
    candidate_indexes = np.flatnonzero(
        np.isfinite(rolling_low) & np.isclose(values_low, rolling_low, rtol=0.0, atol=1e-10)
    )
    for index in candidate_indexes:
        current_momentum = values_momentum[index]
        if not np.isfinite(current_momentum):
            continue
        bar_down = 0
        for offset in range(5, lookback + 1):
            reference = index - offset
            if reference < 0:
                break
            reference_momentum = values_momentum[reference]
            if np.isfinite(reference_momentum) and current_momentum > reference_momentum:
                bar_down = offset
        if bar_down <= 0 or not values_low[index] < values_low[index - bar_down]:
            continue
        rsi_start = max(0, index - (bar_down + 1))
        if np.any(values_rsi[rsi_start : index + 1] < 30.0):
            result[index] = True
    return pd.Series(result, index=low.index, dtype=bool)


def _bearish_knoxville(
    high: pd.Series,
    momentum: pd.Series,
    rsi: pd.Series,
    lookback: int,
) -> pd.Series:
    values_high = high.to_numpy(dtype=float)
    values_momentum = momentum.to_numpy(dtype=float)
    values_rsi = rsi.to_numpy(dtype=float)
    rolling_high = high.rolling(lookback, min_periods=lookback).max().to_numpy(dtype=float)
    result = np.zeros(len(high), dtype=bool)
    candidate_indexes = np.flatnonzero(
        np.isfinite(rolling_high) & np.isclose(values_high, rolling_high, rtol=0.0, atol=1e-10)
    )
    for index in candidate_indexes:
        current_momentum = values_momentum[index]
        if not np.isfinite(current_momentum):
            continue
        bar_up = 0
        for offset in range(5, lookback + 1):
            reference = index - offset
            if reference < 0:
                break
            reference_momentum = values_momentum[reference]
            if np.isfinite(reference_momentum) and current_momentum < reference_momentum:
                bar_up = offset
        if bar_up <= 0 or not values_high[index] > values_high[index - bar_up]:
            continue
        rsi_start = max(0, index - (bar_up + 1))
        if np.any(values_rsi[rsi_start : index + 1] > 70.0):
            result[index] = True
    return pd.Series(result, index=high.index, dtype=bool)


def _backtest_signal_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    exchange: str,
    signal_indexes: list[int],
    signal_variant: SignalVariant,
    exit_variant: ExitVariant,
    end_ts: pd.Timestamp,
    round_trip_cost_pct: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    next_allowed_entry = 0
    for signal_index in signal_indexes:
        entry_index = int(signal_index) + 1
        planned_exit_index = entry_index + exit_variant.max_holding_days - 1
        if entry_index < next_allowed_entry or planned_exit_index >= len(frame):
            continue
        if frame.iloc[planned_exit_index]["date"] > end_ts:
            continue
        trade = _simulate_long_trade(
            frame,
            signal_index=int(signal_index),
            entry_index=entry_index,
            planned_exit_index=planned_exit_index,
            target_pct=exit_variant.target_pct,
            stop_pct=exit_variant.stop_pct,
            round_trip_cost_pct=round_trip_cost_pct,
        )
        if trade is None:
            continue
        trade["signal_envelope_distance_pct"] = _finite_float(
            frame.iloc[signal_index].get(f"distance_lower_{signal_variant.name}")
        )
        rows.append(
            {
                "exchange": exchange,
                "symbol": symbol,
                "signal_variant": signal_variant.name,
                "exit_variant": exit_variant.name,
                "knox_lookback": signal_variant.knox_lookback,
                "envelope_length": signal_variant.envelope_length,
                "envelope_percent": signal_variant.envelope_percent,
                "knox_role": signal_variant.knox_role,
                "target_pct": exit_variant.target_pct,
                "stop_pct": exit_variant.stop_pct,
                "max_holding_days": exit_variant.max_holding_days,
                **trade,
            }
        )
        next_allowed_entry = int(trade["exit_index"]) + 1
    return pd.DataFrame(rows)


def _simulate_long_trade(
    frame: pd.DataFrame,
    *,
    signal_index: int,
    entry_index: int,
    planned_exit_index: int,
    target_pct: float,
    stop_pct: float,
    round_trip_cost_pct: float,
) -> dict[str, Any] | None:
    entry_price = _finite_float(frame.iloc[entry_index].get("open"))
    if entry_price is None or entry_price <= 0:
        return None
    target_price = entry_price * (1.0 + target_pct / 100.0)
    stop_price = entry_price * (1.0 - stop_pct / 100.0)
    exit_index = planned_exit_index
    exit_price = _finite_float(frame.iloc[exit_index].get("close"))
    exit_reason = "TIME"
    for index in range(entry_index, planned_exit_index + 1):
        bar = frame.iloc[index]
        bar_open = float(bar["open"])
        if bar_open <= stop_price:
            exit_index, exit_price, exit_reason = index, bar_open, "STOP_GAP"
            break
        if bar_open >= target_price:
            exit_index, exit_price, exit_reason = index, bar_open, "TARGET_GAP"
            break
        # Daily bars do not reveal intraday path. If both levels trade, assume the stop first.
        if float(bar["low"]) <= stop_price:
            exit_index, exit_price, exit_reason = index, stop_price, "STOP"
            break
        if float(bar["high"]) >= target_price:
            exit_index, exit_price, exit_reason = index, target_price, "TARGET"
            break
    if exit_price is None:
        return None
    gross_return = (exit_price - entry_price) / entry_price * 100.0
    net_return = gross_return - float(round_trip_cost_pct)
    window = frame.iloc[entry_index : exit_index + 1]
    signal = frame.iloc[signal_index]
    return {
        "signal_date": signal["date"],
        "entry_date": frame.iloc[entry_index]["date"],
        "exit_date": frame.iloc[exit_index]["date"],
        "entry_price": entry_price,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "bars_held": int(exit_index - entry_index + 1),
        "gross_return_pct": gross_return,
        "net_return_pct": net_return,
        "win_flag": net_return > 0.0,
        "target_hit": str(exit_reason).startswith("TARGET"),
        "mfe_pct": (float(window["high"].max()) - entry_price) / entry_price * 100.0,
        "mae_pct": (float(window["low"].min()) - entry_price) / entry_price * 100.0,
        "signal_close": float(signal["close"]),
        "signal_rsi": _finite_float(signal.get("rsi14")),
        "signal_relative_performance_pct": _finite_float(signal.get("relative_performance_pct")),
        "signal_turnover_cr": _finite_float(signal.get("avg_turnover_cr")),
        "exit_index": int(exit_index),
    }


def _aggregate(trades: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in trades.groupby(groups, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        returns = pd.to_numeric(group["net_return_pct"], errors="coerce").dropna()
        wins = returns[returns > 0]
        losses = returns[returns <= 0]
        gross_profit = float(wins.sum())
        gross_loss = abs(float(losses.sum()))
        row = dict(zip(groups, key_values))
        row.update(
            {
                "trades": len(returns),
                "win_rate_pct": float((returns > 0).mean() * 100.0),
                "target_rate_pct": float(group["target_hit"].fillna(False).mean() * 100.0),
                "avg_net_return_pct": float(returns.mean()),
                "median_net_return_pct": float(returns.median()),
                "profit_factor": gross_profit / gross_loss if gross_loss > 0 else np.nan,
                "avg_bars_held": float(group["bars_held"].mean()),
                "avg_mfe_pct": float(group["mfe_pct"].mean()),
                "avg_mae_pct": float(group["mae_pct"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _build_variant_stats(period_stats: pd.DataFrame) -> pd.DataFrame:
    if period_stats.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (signal_variant, exit_variant), group in period_stats.groupby(
        ["signal_variant", "exit_variant"]
    ):
        by_period = {str(row["period"]): row for row in group.to_dict(orient="records")}
        row: dict[str, Any] = {
            "signal_variant": signal_variant,
            "exit_variant": exit_variant,
            "variant": f"{signal_variant}__{exit_variant}",
        }
        for period in ("DEVELOPMENT", "VALIDATION", "TEST"):
            stats = by_period.get(period, {})
            prefix = period.lower()
            for metric in (
                "trades",
                "win_rate_pct",
                "target_rate_pct",
                "avg_net_return_pct",
                "median_net_return_pct",
                "profit_factor",
            ):
                row[f"{prefix}_{metric}"] = stats.get(metric, 0 if metric == "trades" else np.nan)
        valid_returns = [
            row.get("validation_avg_net_return_pct"),
            row.get("test_avg_net_return_pct"),
        ]
        valid_returns = [float(value) for value in valid_returns if pd.notna(value)]
        row["robust_score"] = min(valid_returns) if len(valid_returns) == 2 else -999.0
        row["adequate_sample"] = bool(
            int(row.get("validation_trades", 0)) >= 20
            and int(row.get("test_trades", 0)) >= 20
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["adequate_sample", "robust_score", "test_profit_factor", "variant"],
        ascending=[False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def _benchmark_features(benchmark: pd.DataFrame) -> pd.DataFrame:
    close = benchmark["close"]
    ma50 = close.rolling(50, min_periods=50).mean()
    ma200 = close.rolling(200, min_periods=200).mean()
    return pd.DataFrame(
        {
            "date": benchmark["date"],
            "weighted_performance": _weighted_performance(close),
            "market_bullish": (close > ma50) & (ma50 > ma200),
        }
    )


def _weighted_performance(series: pd.Series) -> pd.Series:
    def roc(length: int) -> pd.Series:
        prior = series.shift(length)
        return (series - prior) / prior.replace(0.0, np.nan) * 100.0

    return roc(63) * 0.40 + roc(126) * 0.20 + roc(189) * 0.20 + roc(252) * 0.20


def _period_name(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp <= pd.Timestamp("2022-12-31"):
        return "DEVELOPMENT"
    if timestamp <= pd.Timestamp("2024-12-31"):
        return "VALIDATION"
    return "TEST"


def _candidate_symbols(storage: Storage, exchange: str, symbols: list[str] | None) -> list[str]:
    if symbols is None:
        values = [path.stem for path in (storage.data_root / "candles" / exchange / "1D").glob("*.csv")]
    else:
        values = symbols
    return sorted(
        {
            str(symbol).strip().upper()
            for symbol in values
            if str(symbol).strip() and not _excluded_symbol(str(symbol).strip().upper())
        }
    )


def _excluded_symbol(symbol: str) -> bool:
    return (
        "-" in symbol
        or any(character.isdigit() for character in symbol)
        or any(token in symbol for token in ("NIFTY", "BEES", "ETF"))
    )


def _prepare_daily(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    prepared = frame.copy()
    prepared["date"] = pd.to_datetime(prepared.get("date"), errors="coerce").dt.tz_localize(None)
    for column in ("open", "high", "low", "close", "volume"):
        prepared[column] = pd.to_numeric(prepared.get(column), errors="coerce")
    return (
        prepared.dropna(subset=["date", "open", "high", "low", "close", "volume"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _emit_progress(
    callback: Callable[[dict[str, Any]], None] | None,
    completed: int,
    total: int,
    symbol: str,
) -> None:
    if callback:
        callback(
            {
                "phase": "Backtesting Knoxville + Envelope + Minervini",
                "completed": completed,
                "total": total,
                "current_symbol": symbol,
                "current_exchange": "NSE",
            }
        )


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "exchange",
            "symbol",
            "signal_variant",
            "exit_variant",
            "signal_date",
            "entry_date",
            "exit_date",
            "net_return_pct",
            "win_flag",
            "target_hit",
        ]
    )
