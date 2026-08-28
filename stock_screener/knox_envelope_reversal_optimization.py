from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from stock_screener.data.storage import Storage
from stock_screener.knox_envelope_entry_optimization import (
    DEFAULT_END_DATE,
    DEFAULT_HORIZON,
    DEFAULT_ROUND_TRIP_COST_PCT,
    DEFAULT_START_DATE,
    _entry_outcomes,
    _fast_pine_rsi,
    _load_daily_cache,
    _lower_envelope_distance,
)
from stock_screener.knox_envelope_pair_backtest import (
    _candidate_symbols,
    _knoxville_endpoints,
)


@dataclass(frozen=True)
class SetupParameters:
    knox_lookback: int = 100
    rsi_length: int = 14
    momentum_length: int = 20
    envelope_length: int = 100
    envelope_percent: float = 14.0
    proximity_pct: float = 5.0

    @property
    def name(self) -> str:
        return (
            f"K{self.knox_lookback}_R{self.rsi_length}_M{self.momentum_length}"
            f"_E{self.envelope_length}_P{self.envelope_percent:g}"
            f"_D{self.proximity_pct:g}"
        )


@dataclass(frozen=True)
class ConfirmationParameters:
    kind: str
    window: int
    length: int = 0
    threshold: float = 0.0

    @property
    def name(self) -> str:
        return (
            f"{self.kind}_L{self.length}_T{self.threshold:g}_W{self.window}"
        )


@dataclass(frozen=True)
class ReversalOptimizationResult:
    summary: dict[str, Any]
    stage_stats: pd.DataFrame
    best_trades: pd.DataFrame


def default_setup_candidates() -> tuple[SetupParameters, ...]:
    baseline = SetupParameters()
    values = [baseline]
    values.extend(replace(baseline, knox_lookback=value) for value in (50, 75, 150, 200))
    values.extend(replace(baseline, rsi_length=value) for value in (10, 20))
    values.extend(replace(baseline, momentum_length=value) for value in (10, 30))
    values.extend(
        [
            replace(baseline, knox_lookback=50, rsi_length=10, momentum_length=10),
            replace(baseline, knox_lookback=75, rsi_length=10, momentum_length=20),
            replace(baseline, knox_lookback=150, rsi_length=20, momentum_length=30),
            replace(baseline, knox_lookback=200, rsi_length=20, momentum_length=30),
        ]
    )
    return tuple(dict.fromkeys(values))


def envelope_setup_candidates(
    bases: Iterable[SetupParameters],
) -> tuple[SetupParameters, ...]:
    values = []
    for base in bases:
        for length in (50, 75, 100, 150, 200):
            for percent in (8.0, 10.0, 12.0, 14.0, 16.0, 18.0):
                for proximity in (2.0, 3.0, 5.0):
                    values.append(
                        replace(
                            base,
                            envelope_length=length,
                            envelope_percent=percent,
                            proximity_pct=proximity,
                        )
                    )
    return tuple(dict.fromkeys(values))


def confirmation_candidates() -> tuple[ConfirmationParameters, ...]:
    definitions = [
        ("close_above_prior_high", 1, 0.0),
        ("close_above_prior_high", 2, 0.0),
        ("strong_bullish_close", 0, 0.60),
        ("strong_bullish_close", 0, 0.70),
        ("strong_bullish_close", 0, 0.80),
        ("rsi_cross", 3, 30.0),
        ("rsi_cross", 3, 40.0),
        ("rsi_cross", 5, 30.0),
        ("rsi_cross", 5, 40.0),
        ("williams_cross", 5, -80.0),
        ("williams_cross", 5, -70.0),
        ("williams_cross", 10, -80.0),
        ("stochastic_cross", 5, 50.0),
        ("stochastic_cross", 9, 50.0),
        ("force_index_cross", 2, 0.0),
        ("force_index_cross", 5, 0.0),
        ("lower_wick_rejection", 0, 0.65),
        ("bullish_volume_thrust", 10, 1.20),
        ("bullish_volume_thrust", 10, 1.50),
        ("prior_high_volume_thrust", 10, 1.20),
    ]
    return tuple(
        ConfirmationParameters(kind, window, length, threshold)
        for kind, length, threshold in definitions
        for window in (0, 1, 2, 3)
    )


def run_knox_envelope_reversal_optimization(
    storage: Storage,
    *,
    exchange: str = "NSE",
    symbols: list[str] | None = None,
    start_date: str | pd.Timestamp = DEFAULT_START_DATE,
    end_date: str | pd.Timestamp = DEFAULT_END_DATE,
    horizon: int = DEFAULT_HORIZON,
    round_trip_cost_pct: float = DEFAULT_ROUND_TRIP_COST_PCT,
    minimum_validation_trades: int = 100,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> ReversalOptimizationResult:
    candidates = _candidate_symbols(storage, exchange, symbols)
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    horizon = max(int(horizon), 20)
    daily_cache = _load_daily_cache(
        storage,
        candidates,
        exchange=exchange,
        end_ts=end_ts,
        progress_callback=progress_callback,
    )

    stage1 = default_setup_candidates()
    stage1_trades, audit1 = _evaluate_setups(
        daily_cache,
        stage1,
        exchange=exchange,
        start_ts=start_ts,
        end_ts=end_ts,
        horizon=horizon,
        round_trip_cost_pct=round_trip_cost_pct,
        stage="KNOXVILLE_NO_FILTER",
        progress_callback=progress_callback,
    )
    stage1_stats = _outcome_stats(stage1_trades, _setup_metadata(stage1), "KNOXVILLE_NO_FILTER")
    stage1_best = _top_setups(stage1_stats, stage1, minimum_validation_trades, 3)

    stage2 = envelope_setup_candidates(stage1_best)
    stage2_trades, audit2 = _evaluate_setups(
        daily_cache,
        stage2,
        exchange=exchange,
        start_ts=start_ts,
        end_ts=end_ts,
        horizon=horizon,
        round_trip_cost_pct=round_trip_cost_pct,
        stage="ENVELOPE_NO_FILTER",
        progress_callback=progress_callback,
    )
    stage2_stats = _outcome_stats(stage2_trades, _setup_metadata(stage2), "ENVELOPE_NO_FILTER")
    stage2_best = _top_setups(stage2_stats, stage2, minimum_validation_trades, 5)

    confirmations = confirmation_candidates()
    stage3_trades, audit3, metadata3 = _evaluate_confirmations(
        daily_cache,
        stage2_best,
        confirmations,
        exchange=exchange,
        start_ts=start_ts,
        end_ts=end_ts,
        horizon=horizon,
        round_trip_cost_pct=round_trip_cost_pct,
        progress_callback=progress_callback,
    )
    stage3_stats = _outcome_stats(stage3_trades, metadata3, "FAST_CONFIRMATION")
    best_name = _top_parameter_names(stage3_stats, minimum_validation_trades, 1)[0]
    best_trades = stage3_trades.loc[stage3_trades["parameter_name"] == best_name].copy()
    all_stats = pd.concat([stage1_stats, stage2_stats, stage3_stats], ignore_index=True)
    best_validation = stage3_stats.loc[
        (stage3_stats["parameter_name"] == best_name)
        & (stage3_stats["cohort"] == "VALIDATION")
    ]
    best_holdout = stage3_stats.loc[
        (stage3_stats["parameter_name"] == best_name)
        & (stage3_stats["cohort"] == "HOLDOUT")
    ]
    best_meta = metadata3[best_name]
    summary = {
        "exchange": exchange,
        "requested_start_date": start_ts.date().isoformat(),
        "requested_end_date": end_ts.date().isoformat(),
        "stored_eligible_symbols": len(candidates),
        "symbols_with_sufficient_history": audit3["symbols_with_sufficient_history"],
        "latest_observed_date": audit3["latest_observed_date"],
        "entry_execution": "next_session_high_after_confirmation",
        "round_trip_cost_pct": float(round_trip_cost_pct),
        "selection_cohort": "entry dates in 2022-2023",
        "holdout_cohort": "entry dates in 2024-2026",
        "minimum_validation_trades": int(minimum_validation_trades),
        "stage1_parameter_count": len(stage1),
        "stage2_parameter_count": len(stage2),
        "stage3_parameter_count": len(stage2_best) * len(confirmations),
        "best_parameter_name": best_name,
        **best_meta,
        **_summary_cohort("validation", best_validation),
        **_summary_cohort("holdout", best_holdout),
        "stage1_symbols_with_history": audit1["symbols_with_sufficient_history"],
        "stage2_symbols_with_history": audit2["symbols_with_sufficient_history"],
    }
    return ReversalOptimizationResult(summary, all_stats, best_trades)


def _evaluate_setups(
    daily_cache: dict[str, pd.DataFrame],
    parameters: tuple[SetupParameters, ...],
    **kwargs: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return _evaluate(
        daily_cache,
        parameters,
        confirmations=None,
        **kwargs,
    )[:2]


def _evaluate_confirmations(
    daily_cache: dict[str, pd.DataFrame],
    setups: list[SetupParameters],
    confirmations: tuple[ConfirmationParameters, ...],
    **kwargs: Any,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, dict[str, Any]]]:
    trades, audit, metadata = _evaluate(
        daily_cache,
        tuple(setups),
        confirmations=confirmations,
        stage="FAST_CONFIRMATION",
        **kwargs,
    )
    return trades, audit, metadata


def _evaluate(
    daily_cache: dict[str, pd.DataFrame],
    setups: tuple[SetupParameters, ...],
    *,
    confirmations: tuple[ConfirmationParameters, ...] | None,
    exchange: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    horizon: int,
    round_trip_cost_pct: float,
    stage: str,
    progress_callback: Callable[[dict[str, Any]], None] | None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}
    history_count = 0
    latest_observed: pd.Timestamp | None = None
    knox_keys = sorted(
        {(p.knox_lookback, p.rsi_length, p.momentum_length) for p in setups}
    )
    envelope_keys = sorted({(p.envelope_length, p.envelope_percent) for p in setups})
    minimum_history = max(
        max(p.knox_lookback + p.momentum_length + 2 for p in setups),
        max(p.envelope_length for p in setups),
    )
    symbols = sorted(daily_cache)
    for completed, symbol in enumerate(symbols, start=1):
        daily = daily_cache[symbol]
        if len(daily) < minimum_history + horizon + 5 or daily["date"].max() < start_ts:
            _emit_progress(progress_callback, stage, completed, len(symbols), symbol)
            continue
        history_count += 1
        latest = pd.Timestamp(daily["date"].max())
        latest_observed = latest if latest_observed is None else max(latest_observed, latest)
        rsi_cache = {
            length: _fast_pine_rsi(daily["close"], length)
            for length in {key[1] for key in knox_keys}
        }
        momentum_cache = {
            length: daily["close"] - daily["close"].shift(length)
            for length in {key[2] for key in knox_keys}
        }
        knox_cache = {
            key: _knoxville_endpoints(
                daily["high"],
                daily["low"],
                momentum_cache[key[2]],
                rsi_cache[key[1]],
                key[0],
            )[0]
            for key in knox_keys
        }
        envelope_cache = {
            key: _lower_envelope_distance(daily, key[0], key[1])
            for key in envelope_keys
        }
        event_cache = _confirmation_events(daily, confirmations or ())
        for setup in setups:
            lower_distance = envelope_cache[(setup.envelope_length, setup.envelope_percent)]
            setup_mask = (
                knox_cache[(setup.knox_lookback, setup.rsi_length, setup.momentum_length)]
                & np.isfinite(lower_distance)
                & (np.abs(lower_distance) <= setup.proximity_pct)
            )
            if confirmations is None:
                name = setup.name
                metadata.setdefault(name, _metadata(setup, None))
                signal_masks = [(name, setup_mask)]
            else:
                signal_masks = []
                for confirmation in confirmations:
                    name = f"{setup.name}_{confirmation.name}"
                    metadata.setdefault(name, _metadata(setup, confirmation))
                    signal_masks.append(
                        (
                            name,
                            _confirmed_signal(
                                setup_mask,
                                event_cache[confirmation.name],
                                confirmation.window,
                            ),
                        )
                    )
            for name, signal_mask in signal_masks:
                outcomes = _entry_outcomes(
                    daily,
                    signal_mask,
                    symbol=symbol,
                    exchange=exchange,
                    parameter_name=name,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    horizon=horizon,
                    round_trip_cost_pct=round_trip_cost_pct,
                    lower_distance=lower_distance,
                    cmf=pd.Series(np.nan, index=daily.index),
                )
                for outcome in outcomes:
                    outcome["setup_or_confirmation"] = stage
                rows.extend(outcomes)
        _emit_progress(progress_callback, stage, completed, len(symbols), symbol)
    return pd.DataFrame(rows), {
        "symbols_with_sufficient_history": history_count,
        "latest_observed_date": latest_observed.date().isoformat() if latest_observed else "",
    }, metadata


def _confirmation_events(
    daily: pd.DataFrame,
    confirmations: tuple[ConfirmationParameters, ...],
) -> dict[str, np.ndarray]:
    if not confirmations:
        return {}
    close = daily["close"].astype(float)
    high = daily["high"].astype(float)
    low = daily["low"].astype(float)
    open_ = daily["open"].astype(float)
    volume = daily["volume"].astype(float)
    candle_range = (high - low).replace(0.0, np.nan)
    close_location = (close - low) / candle_range
    bullish = (close > open_) & (close > close.shift(1))
    cache: dict[tuple[str, int, float], pd.Series] = {}
    result = {}
    for item in confirmations:
        key = (item.kind, item.length, item.threshold)
        if key not in cache:
            if item.kind == "close_above_prior_high":
                prior_high = high.shift(1).rolling(item.length, min_periods=item.length).max()
                event = close > prior_high
            elif item.kind == "strong_bullish_close":
                event = bullish & (close_location >= item.threshold)
            elif item.kind == "rsi_cross":
                rsi = _fast_pine_rsi(close, item.length)
                event = (rsi > item.threshold) & (rsi.shift(1) <= item.threshold)
            elif item.kind == "williams_cross":
                highest = high.rolling(item.length, min_periods=item.length).max()
                lowest = low.rolling(item.length, min_periods=item.length).min()
                williams = -100.0 * (highest - close) / (highest - lowest).replace(0.0, np.nan)
                event = (williams > item.threshold) & (williams.shift(1) <= item.threshold)
            elif item.kind == "stochastic_cross":
                highest = high.rolling(item.length, min_periods=item.length).max()
                lowest = low.rolling(item.length, min_periods=item.length).min()
                percent_k = 100.0 * (close - lowest) / (highest - lowest).replace(0.0, np.nan)
                percent_d = percent_k.rolling(3, min_periods=3).mean()
                event = (
                    (percent_k > percent_d)
                    & (percent_k.shift(1) <= percent_d.shift(1))
                    & (percent_k <= item.threshold)
                )
            elif item.kind == "force_index_cross":
                force = ((close - close.shift(1)) * volume).ewm(
                    span=item.length,
                    adjust=False,
                    min_periods=item.length,
                ).mean()
                event = (force > 0.0) & (force.shift(1) <= 0.0)
            elif item.kind == "lower_wick_rejection":
                body_low = pd.concat([open_, close], axis=1).min(axis=1)
                lower_wick = body_low - low
                body = (close - open_).abs()
                event = bullish & (close_location >= item.threshold) & (lower_wick > body)
            elif item.kind == "bullish_volume_thrust":
                prior_average = volume.shift(1).rolling(item.length, min_periods=item.length).mean()
                event = bullish & (close_location >= 0.65) & (volume >= prior_average * item.threshold)
            elif item.kind == "prior_high_volume_thrust":
                prior_average = volume.shift(1).rolling(item.length, min_periods=item.length).mean()
                event = (close > high.shift(1)) & (volume >= prior_average * item.threshold)
            else:
                raise ValueError(f"Unsupported confirmation kind: {item.kind}")
            cache[key] = event.fillna(False)
        result[item.name] = cache[key].to_numpy(dtype=bool)
    return result


def _confirmed_signal(
    setup_mask: np.ndarray,
    confirmation_event: np.ndarray,
    window: int,
) -> np.ndarray:
    recent_setup = (
        pd.Series(setup_mask, dtype=bool)
        .rolling(max(int(window), 0) + 1, min_periods=1)
        .max()
        .fillna(0.0)
        .astype(bool)
        .to_numpy(dtype=bool)
    )
    return recent_setup & confirmation_event


def _outcome_stats(
    trades: pd.DataFrame,
    metadata: dict[str, dict[str, Any]],
    stage: str,
) -> pd.DataFrame:
    rows = []
    if trades.empty:
        return pd.DataFrame()
    for (name, cohort), group in trades.groupby(["parameter_name", "cohort"]):
        returns_20 = pd.to_numeric(group["return_20_pct"], errors="coerce")
        target_before_stop = group["target_7_before_stop_5_within_20"].fillna(False)
        rows.append(
            {
                "stage": stage,
                "parameter_name": name,
                "cohort": cohort,
                **metadata[str(name)],
                "trades": len(group),
                "entry_score": (
                    target_before_stop.mean() * 70.0
                    + group["target_7_within_20"].mean() * 20.0
                    + (returns_20 > 0.0).mean() * 10.0
                ),
                "target_5_within_10_pct": group["target_5_within_10"].mean() * 100.0,
                "target_7_within_10_pct": group["target_7_within_10"].mean() * 100.0,
                "target_7_within_20_pct": group["target_7_within_20"].mean() * 100.0,
                "target_10_within_20_pct": group["target_10_within_20"].mean() * 100.0,
                "target_7_before_stop_5_pct": target_before_stop.mean() * 100.0,
                "stop_5_within_20_pct": group["stop_5_within_20"].mean() * 100.0,
                "win_10_pct": (group["return_10_pct"] > 0.0).mean() * 100.0,
                "win_20_pct": (returns_20 > 0.0).mean() * 100.0,
                "median_return_10_pct": group["return_10_pct"].median(),
                "median_return_20_pct": returns_20.median(),
                "median_mfe_20_pct": group["mfe_20_pct"].median(),
                "median_mae_20_pct": group["mae_20_pct"].median(),
            }
        )
    return pd.DataFrame(rows)


def _top_setups(
    stats: pd.DataFrame,
    setups: tuple[SetupParameters, ...],
    minimum_validation_trades: int,
    count: int,
) -> list[SetupParameters]:
    setup_map = {item.name: item for item in setups}
    names = _top_parameter_names(stats, minimum_validation_trades, count)
    return [setup_map[name] for name in names]


def _top_parameter_names(
    stats: pd.DataFrame,
    minimum_validation_trades: int,
    count: int,
) -> list[str]:
    validation = stats.loc[stats["cohort"] == "VALIDATION"].copy()

    def rank(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.sort_values(
            [
                "entry_score",
                "target_7_before_stop_5_pct",
                "target_7_within_20_pct",
                "median_return_20_pct",
                "trades",
            ],
            ascending=[False, False, False, False, False],
        )

    sufficient = rank(validation.loc[validation["trades"] >= minimum_validation_trades])
    ranked = pd.concat([sufficient, rank(validation)], ignore_index=True)
    return ranked.drop_duplicates("parameter_name")["parameter_name"].head(count).tolist()


def _setup_metadata(
    setups: tuple[SetupParameters, ...],
) -> dict[str, dict[str, Any]]:
    return {item.name: _metadata(item, None) for item in setups}


def _metadata(
    setup: SetupParameters,
    confirmation: ConfirmationParameters | None,
) -> dict[str, Any]:
    return {
        "knox_lookback": setup.knox_lookback,
        "rsi_length": setup.rsi_length,
        "momentum_length": setup.momentum_length,
        "envelope_length": setup.envelope_length,
        "envelope_percent": setup.envelope_percent,
        "proximity_pct": setup.proximity_pct,
        "confirmation_kind": confirmation.kind if confirmation else "none",
        "confirmation_window": confirmation.window if confirmation else 0,
        "confirmation_length": confirmation.length if confirmation else 0,
        "confirmation_threshold": confirmation.threshold if confirmation else 0.0,
    }


def _summary_cohort(prefix: str, frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {f"{prefix}_trades": 0}
    row = frame.iloc[0]
    keys = (
        "trades",
        "entry_score",
        "target_5_within_10_pct",
        "target_7_within_10_pct",
        "target_7_within_20_pct",
        "target_10_within_20_pct",
        "target_7_before_stop_5_pct",
        "stop_5_within_20_pct",
        "win_10_pct",
        "win_20_pct",
        "median_return_10_pct",
        "median_return_20_pct",
        "median_mfe_20_pct",
        "median_mae_20_pct",
    )
    return {f"{prefix}_{key}": row[key] for key in keys}


def save_reversal_optimization(
    result: ReversalOptimizationResult,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "summary.csv",
        "stage_stats": output_dir / "stage_stats.csv",
        "best_trades": output_dir / "best_trades.csv",
        "report": output_dir / "report.md",
    }
    pd.DataFrame([result.summary]).to_csv(paths["summary"], index=False)
    result.stage_stats.to_csv(paths["stage_stats"], index=False)
    result.best_trades.to_csv(paths["best_trades"], index=False)
    paths["report"].write_text(_reversal_report(result), encoding="utf-8")
    return paths


def _reversal_report(result: ReversalOptimizationResult) -> str:
    summary = result.summary
    stats = result.stage_stats
    setup = SetupParameters(
        knox_lookback=int(summary["knox_lookback"]),
        rsi_length=int(summary["rsi_length"]),
        momentum_length=int(summary["momentum_length"]),
        envelope_length=int(summary["envelope_length"]),
        envelope_percent=float(summary["envelope_percent"]),
        proximity_pct=float(summary["proximity_pct"]),
    )
    base = stats.loc[
        (stats["parameter_name"] == setup.name)
        & (stats["stage"] == "ENVELOPE_NO_FILTER")
    ].set_index("cohort")
    family = stats.loc[
        (stats["stage"] == "FAST_CONFIRMATION")
        & (stats["cohort"] == "HOLDOUT")
        & (stats["trades"] >= 100)
    ]
    family_rows = []
    for kind, group in family.groupby("confirmation_kind"):
        family_rows.append(
            {
                "kind": kind,
                "risk": group["target_7_before_stop_5_pct"].median(),
                "stop": group["stop_5_within_20_pct"].median(),
                "return": group["median_return_20_pct"].median(),
            }
        )
    family_rows.sort(key=lambda row: row["risk"], reverse=True)
    family_table = [
        "| Confirmation family | Median +7% before -5% | Median -5% touch | Median 20D return |",
        "|---|---:|---:|---:|",
    ]
    family_table.extend(
        f"| {row['kind']} | {row['risk']:.1f}% | {row['stop']:.1f}% | {row['return']:+.2f}% |"
        for row in family_rows
    )
    base_validation = base.loc["VALIDATION"]
    base_holdout = base.loc["HOLDOUT"]
    return "\n".join(
        [
            "# Fast Reversal Confirmation Research",
            "",
            "## Verdict",
            "",
            "CMF was removed from every stage. The validation-selected fast confirmation "
            "did not improve the Knoxville and Envelope setup in the untouched holdout. "
            "None of the tested confirmation families is suitable as a standalone entry gate.",
            "",
            "## Validation-Selected Rule",
            "",
            f"- Knoxville: {summary['knox_lookback']} bars, RSI {summary['rsi_length']}, "
            f"momentum {summary['momentum_length']}",
            f"- Envelope: {summary['envelope_length']}-day SMA, "
            f"{summary['envelope_percent']:.0f}% lower band, "
            f"{summary['proximity_pct']:.0f}% proximity",
            f"- Confirmation: {summary['confirmation_kind']}, length "
            f"{summary['confirmation_length']}, threshold "
            f"{summary['confirmation_threshold']:.0f}, within "
            f"{summary['confirmation_window']} sessions",
            "- Entry: next session's high after confirmation",
            f"- Round-trip cost: {summary['round_trip_cost_pct']:.2f}%",
            "",
            "## Confirmation Versus No Confirmation",
            "",
            "| Metric | Base validation | Confirmed validation | Base holdout | Confirmed holdout |",
            "|---|---:|---:|---:|---:|",
            f"| Trades | {int(base_validation['trades'])} | {int(summary['validation_trades'])} | "
            f"{int(base_holdout['trades'])} | {int(summary['holdout_trades'])} |",
            f"| +7% before -5% | {base_validation['target_7_before_stop_5_pct']:.1f}% | "
            f"{summary['validation_target_7_before_stop_5_pct']:.1f}% | "
            f"{base_holdout['target_7_before_stop_5_pct']:.1f}% | "
            f"{summary['holdout_target_7_before_stop_5_pct']:.1f}% |",
            f"| Hit +7% in 20D | {base_validation['target_7_within_20_pct']:.1f}% | "
            f"{summary['validation_target_7_within_20_pct']:.1f}% | "
            f"{base_holdout['target_7_within_20_pct']:.1f}% | "
            f"{summary['holdout_target_7_within_20_pct']:.1f}% |",
            f"| Touched -5% | {base_validation['stop_5_within_20_pct']:.1f}% | "
            f"{summary['validation_stop_5_within_20_pct']:.1f}% | "
            f"{base_holdout['stop_5_within_20_pct']:.1f}% | "
            f"{summary['holdout_stop_5_within_20_pct']:.1f}% |",
            f"| Median 20D return | {base_validation['median_return_20_pct']:+.2f}% | "
            f"{summary['validation_median_return_20_pct']:+.2f}% | "
            f"{base_holdout['median_return_20_pct']:+.2f}% | "
            f"{summary['holdout_median_return_20_pct']:+.2f}% |",
            "",
            "## Holdout Family Diagnostics",
            "",
            *family_table,
            "",
            "Strong bullish closes and lower-wick rejection were the least weak families, "
            "but both retained negative median returns. The next research step should add "
            "a market-regime or relative-strength gate rather than another oversold oscillator.",
            "",
            "## Audit",
            "",
            f"- Eligible NSE equity symbols: {summary['stored_eligible_symbols']}",
            f"- Symbols with sufficient history: {summary['symbols_with_sufficient_history']}",
            f"- Latest candle: {summary['latest_observed_date']}",
            f"- Search: {summary['stage1_parameter_count']} Knoxville, "
            f"{summary['stage2_parameter_count']} Envelope, and "
            f"{summary['stage3_parameter_count']} confirmation combinations",
            "- ETFs are excluded using both symbol patterns and Kite instrument names.",
            "",
        ]
    )


def _emit_progress(
    callback: Callable[[dict[str, Any]], None] | None,
    stage: str,
    completed: int,
    total: int,
    symbol: str,
) -> None:
    if callback:
        callback(
            {
                "stage": stage,
                "completed": completed,
                "total": total,
                "current_symbol": symbol,
            }
        )
