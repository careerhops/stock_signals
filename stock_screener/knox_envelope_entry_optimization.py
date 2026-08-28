from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from stock_screener.data.storage import Storage
from stock_screener.knox_envelope_pair_backtest import (
    _candidate_symbols,
    _knoxville_endpoints,
    _prepare_daily,
)


DEFAULT_START_DATE = "2016-08-20"
DEFAULT_END_DATE = "2026-08-20"
DEFAULT_HORIZON = 20
DEFAULT_ROUND_TRIP_COST_PCT = 0.35


@dataclass(frozen=True)
class EntryStrategyParameters:
    knox_lookback: int = 100
    rsi_length: int = 14
    momentum_length: int = 20
    envelope_length: int = 100
    envelope_percent: float = 14.0
    proximity_pct: float = 5.0
    cmf_length: int = 20
    cmf_mode: str = "positive"
    cmf_min: float = 0.0
    cmf_max: float | None = None

    @property
    def name(self) -> str:
        maximum = "ANY" if self.cmf_max is None else f"{self.cmf_max:g}"
        return (
            f"K{self.knox_lookback}_R{self.rsi_length}_M{self.momentum_length}"
            f"_E{self.envelope_length}_P{self.envelope_percent:g}"
            f"_D{self.proximity_pct:g}_C{self.cmf_length}"
            f"_{self.cmf_mode}_{self.cmf_min:g}_{maximum}"
        )


@dataclass(frozen=True)
class EntryOptimizationResult:
    summary: dict[str, Any]
    stage_stats: pd.DataFrame
    best_trades: pd.DataFrame


def default_knoxville_candidates() -> tuple[EntryStrategyParameters, ...]:
    baseline = EntryStrategyParameters()
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


def envelope_candidates(
    bases: Iterable[EntryStrategyParameters],
) -> tuple[EntryStrategyParameters, ...]:
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


def cmf_candidates(
    bases: Iterable[EntryStrategyParameters],
) -> tuple[EntryStrategyParameters, ...]:
    modes = (
        ("positive", 0.0, None),
        ("positive", 0.10, None),
        ("band", 0.0, 0.20),
        ("band", 0.0, 0.40),
        ("band", 0.10, 0.40),
        ("rising_positive", 0.0, None),
        ("crossed_recent", 0.0, None),
    )
    values = []
    for base in bases:
        for length in (10, 20, 30):
            for mode, minimum, maximum in modes:
                values.append(
                    replace(
                        base,
                        cmf_length=length,
                        cmf_mode=mode,
                        cmf_min=minimum,
                        cmf_max=maximum,
                    )
                )
    return tuple(dict.fromkeys(values))


def run_knox_envelope_entry_optimization(
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
) -> EntryOptimizationResult:
    candidates = _candidate_symbols(storage, exchange, symbols)
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    horizon = max(int(horizon), 20)
    stage_frames: list[pd.DataFrame] = []
    daily_cache = _load_daily_cache(
        storage,
        candidates,
        exchange=exchange,
        end_ts=end_ts,
        progress_callback=progress_callback,
    )

    stage1_parameters = default_knoxville_candidates()
    stage1_trades, stage1_audit = _evaluate_parameter_set(
        daily_cache,
        stage1_parameters,
        exchange=exchange,
        start_ts=start_ts,
        end_ts=end_ts,
        horizon=horizon,
        round_trip_cost_pct=round_trip_cost_pct,
        stage="KNOXVILLE",
        progress_callback=progress_callback,
    )
    stage1_stats = _parameter_stats(stage1_trades, stage1_parameters, "KNOXVILLE")
    stage_frames.append(stage1_stats)
    stage1_best = _top_parameters(stage1_stats, stage1_parameters, minimum_validation_trades, 3)

    stage2_parameters = envelope_candidates(stage1_best)
    stage2_trades, stage2_audit = _evaluate_parameter_set(
        daily_cache,
        stage2_parameters,
        exchange=exchange,
        start_ts=start_ts,
        end_ts=end_ts,
        horizon=horizon,
        round_trip_cost_pct=round_trip_cost_pct,
        stage="ENVELOPE",
        progress_callback=progress_callback,
    )
    stage2_stats = _parameter_stats(stage2_trades, stage2_parameters, "ENVELOPE")
    stage_frames.append(stage2_stats)
    stage2_best = _top_parameters(stage2_stats, stage2_parameters, minimum_validation_trades, 5)

    stage3_parameters = cmf_candidates(stage2_best)
    stage3_trades, stage3_audit = _evaluate_parameter_set(
        daily_cache,
        stage3_parameters,
        exchange=exchange,
        start_ts=start_ts,
        end_ts=end_ts,
        horizon=horizon,
        round_trip_cost_pct=round_trip_cost_pct,
        stage="CMF",
        progress_callback=progress_callback,
    )
    stage3_stats = _parameter_stats(stage3_trades, stage3_parameters, "CMF")
    stage_frames.append(stage3_stats)
    best_parameters = _top_parameters(
        stage3_stats,
        stage3_parameters,
        minimum_validation_trades,
        1,
    )[0]
    best_trades = stage3_trades.loc[
        stage3_trades["parameter_name"] == best_parameters.name
    ].copy()

    all_stats = pd.concat(stage_frames, ignore_index=True)
    best_stats = all_stats.loc[all_stats["parameter_name"] == best_parameters.name]
    validation = best_stats.loc[best_stats["cohort"] == "VALIDATION"]
    holdout = best_stats.loc[best_stats["cohort"] == "HOLDOUT"]
    summary = {
        "exchange": exchange,
        "requested_start_date": start_ts.date().isoformat(),
        "requested_end_date": end_ts.date().isoformat(),
        "stored_eligible_symbols": len(candidates),
        "symbols_with_sufficient_history": stage3_audit["symbols_with_sufficient_history"],
        "latest_observed_date": stage3_audit["latest_observed_date"],
        "horizon_sessions": horizon,
        "entry_execution": "next_session_high",
        "outcome_window": "sessions_after_entry_session",
        "round_trip_cost_pct": float(round_trip_cost_pct),
        "selection_cohort": "entry dates in 2022-2023",
        "holdout_cohort": "entry dates in 2024-2026",
        "minimum_validation_trades": int(minimum_validation_trades),
        "stage1_parameter_count": len(stage1_parameters),
        "stage2_parameter_count": len(stage2_parameters),
        "stage3_parameter_count": len(stage3_parameters),
        "best_parameter_name": best_parameters.name,
        "best_knox_lookback": best_parameters.knox_lookback,
        "best_rsi_length": best_parameters.rsi_length,
        "best_momentum_length": best_parameters.momentum_length,
        "best_envelope_length": best_parameters.envelope_length,
        "best_envelope_percent": best_parameters.envelope_percent,
        "best_proximity_pct": best_parameters.proximity_pct,
        "best_cmf_length": best_parameters.cmf_length,
        "best_cmf_mode": best_parameters.cmf_mode,
        "best_cmf_min": best_parameters.cmf_min,
        "best_cmf_max": best_parameters.cmf_max if best_parameters.cmf_max is not None else "",
        **_summary_cohort("validation", validation),
        **_summary_cohort("holdout", holdout),
        "stage1_symbols_with_history": stage1_audit["symbols_with_sufficient_history"],
        "stage2_symbols_with_history": stage2_audit["symbols_with_sufficient_history"],
    }
    return EntryOptimizationResult(summary=summary, stage_stats=all_stats, best_trades=best_trades)


def save_entry_optimization(
    result: EntryOptimizationResult,
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
    paths["report"].write_text(_optimization_report(result), encoding="utf-8")
    return paths


def _optimization_report(result: EntryOptimizationResult) -> str:
    summary = result.summary
    trades = result.best_trades.copy()
    yearly = []
    for year, group in trades.groupby(pd.to_datetime(trades["entry_date"]).dt.year):
        yearly.append(
            {
                "year": int(year),
                "trades": len(group),
                "target7": group["target_7_within_20"].mean() * 100.0,
                "target7_before_stop": group[
                    "target_7_before_stop_5_within_20"
                ].mean()
                * 100.0,
                "stop5": group["stop_5_within_20"].mean() * 100.0,
                "win20": (group["return_20_pct"] > 0.0).mean() * 100.0,
                "median20": group["return_20_pct"].median(),
            }
        )

    holdout = trades.loc[trades["cohort"] == "HOLDOUT"]
    holdout_successes = int(
        holdout["target_7_before_stop_5_within_20"].fillna(False).sum()
    )
    ci_low, ci_high = _wilson_interval(holdout_successes, len(holdout))
    yearly_lines = [
        "| Year | Trades | Hit 7% | 7% before -5% | Hit -5% | Positive 20D | Median 20D |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    yearly_lines.extend(
        "| {year} | {trades} | {target7:.1f}% | {target7_before_stop:.1f}% | "
        "{stop5:.1f}% | {win20:.1f}% | {median20:+.2f}% |".format(**row)
        for row in yearly
    )

    return "\n".join(
        [
            "# Knoxville + Envelope + CMF Entry Optimization",
            "",
            "## Verdict",
            "",
            "The search found a best validation-selected combination, but it did not "
            "remain robust in the untouched 2024-2026 holdout. It is not suitable as a "
            "standalone live entry strategy yet.",
            "",
            "## Selected Rule",
            "",
            f"- Knoxville bars back: {summary['best_knox_lookback']}",
            f"- RSI period: {summary['best_rsi_length']}",
            f"- Momentum period: {summary['best_momentum_length']}",
            f"- Envelope: {summary['best_envelope_length']}-day SMA, "
            f"{summary['best_envelope_percent']:.0f}% lower band",
            f"- Knoxville bullish endpoint low: within {summary['best_proximity_pct']:.0f}% "
            "of the lower envelope",
            f"- CMF: {summary['best_cmf_length']} days, greater than "
            f"{summary['best_cmf_min']:.2f} and at most {summary['best_cmf_max']:.2f}",
            "- Signal is formed at the close; simulated entry is the next session's high.",
            "- Outcomes start on the session after entry, avoiding unknown intraday "
            "target/stop ordering on the entry day.",
            f"- Round-trip cost: {summary['round_trip_cost_pct']:.2f}%.",
            "",
            "## Time-Separated Results",
            "",
            "| Metric | Validation 2022-2023 | Untouched holdout 2024-2026 |",
            "|---|---:|---:|",
            f"| Trades | {summary['validation_trades']} | {summary['holdout_trades']} |",
            f"| Hit +7% within 20 sessions | {summary['validation_target_7_within_20_pct']:.1f}% | "
            f"{summary['holdout_target_7_within_20_pct']:.1f}% |",
            f"| Hit +7% before -5% | {summary['validation_target_7_before_stop_5_pct']:.1f}% | "
            f"{summary['holdout_target_7_before_stop_5_pct']:.1f}% |",
            f"| Touched -5% within 20 sessions | {summary['validation_stop_5_within_20_pct']:.1f}% | "
            f"{summary['holdout_stop_5_within_20_pct']:.1f}% |",
            f"| Positive 20-session close | {summary['validation_win_20_pct']:.1f}% | "
            f"{summary['holdout_win_20_pct']:.1f}% |",
            f"| Median 20-session return | {summary['validation_median_return_20_pct']:+.2f}% | "
            f"{summary['holdout_median_return_20_pct']:+.2f}% |",
            f"| Median favorable excursion | {summary['validation_median_mfe_20_pct']:+.2f}% | "
            f"{summary['holdout_median_mfe_20_pct']:+.2f}% |",
            f"| Median adverse excursion | {summary['validation_median_mae_20_pct']:+.2f}% | "
            f"{summary['holdout_median_mae_20_pct']:+.2f}% |",
            "",
            f"The holdout 95% Wilson interval for hitting +7% before -5% is "
            f"{ci_low * 100.0:.1f}% to {ci_high * 100.0:.1f}%.",
            "",
            "## Yearly Stability",
            "",
            *yearly_lines,
            "",
            "## Universe And Search",
            "",
            f"- Stored eligible NSE symbols: {summary['stored_eligible_symbols']}",
            f"- Symbols with sufficient history in the final stage: "
            f"{summary['symbols_with_sufficient_history']}",
            f"- Latest candle observed: {summary['latest_observed_date']}",
            f"- Search candidates: {summary['stage1_parameter_count']} Knoxville, "
            f"{summary['stage2_parameter_count']} Envelope, and "
            f"{summary['stage3_parameter_count']} CMF combinations",
            "- Existing universe hygiene exclusions remain active: hyphenated symbols, "
            "ETFs, NIFTY/BEES names, and symbols containing digits.",
            "",
            "## Limitations",
            "",
            "- The stored universe can contain survivorship bias; delisted historical "
            "constituents are not guaranteed to be present.",
            "- Daily bars cannot determine the order of a target and stop touched during "
            "the same session. Entry-day outcomes are deliberately excluded.",
            "- The model includes 0.35% round-trip cost but not symbol-specific slippage, "
            "market impact, taxes, or liquidity limits.",
            "- Testing many parameter combinations creates selection bias. The untouched "
            "holdout is therefore the decision result, not the validation winner.",
            "",
        ]
    )


def _wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials <= 0:
        return (float("nan"), float("nan"))
    probability = successes / trials
    denominator = 1.0 + z * z / trials
    center = (probability + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * np.sqrt(
            probability * (1.0 - probability) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return center - margin, center + margin


def _evaluate_parameter_set(
    daily_cache: dict[str, pd.DataFrame],
    parameters: tuple[EntryStrategyParameters, ...],
    *,
    exchange: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    horizon: int,
    round_trip_cost_pct: float,
    stage: str,
    progress_callback: Callable[[dict[str, Any]], None] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    history_count = 0
    latest_observed: pd.Timestamp | None = None
    knox_keys = sorted(
        {(p.knox_lookback, p.rsi_length, p.momentum_length) for p in parameters}
    )
    envelope_keys = sorted({(p.envelope_length, p.envelope_percent) for p in parameters})
    cmf_lengths = sorted({p.cmf_length for p in parameters})
    minimum_history = max(
        max(p.knox_lookback + p.momentum_length + 2 for p in parameters),
        max(p.envelope_length for p in parameters),
        max(cmf_lengths),
    )

    symbols = sorted(daily_cache)
    for completed, symbol in enumerate(symbols, start=1):
        daily = daily_cache[symbol]
        if len(daily) < minimum_history + horizon + 2 or daily["date"].max() < start_ts:
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
        cmf_cache = {length: _cmf(daily, length) for length in cmf_lengths}

        for item in parameters:
            bullish = knox_cache[
                (item.knox_lookback, item.rsi_length, item.momentum_length)
            ]
            lower_distance = envelope_cache[
                (item.envelope_length, item.envelope_percent)
            ]
            cmf = cmf_cache[item.cmf_length]
            cmf_mask = _cmf_mask(cmf, item)
            signal_mask = (
                bullish
                & np.isfinite(lower_distance)
                & (np.abs(lower_distance) <= item.proximity_pct)
                & cmf_mask
            )
            rows.extend(
                _entry_outcomes(
                    daily,
                    signal_mask,
                    symbol=symbol,
                    exchange=exchange,
                    parameter_name=item.name,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    horizon=horizon,
                    round_trip_cost_pct=round_trip_cost_pct,
                    lower_distance=lower_distance,
                    cmf=cmf,
                )
            )
        _emit_progress(progress_callback, stage, completed, len(symbols), symbol)

    return pd.DataFrame(rows), {
        "symbols_with_sufficient_history": history_count,
        "latest_observed_date": latest_observed.date().isoformat() if latest_observed else "",
    }


def _load_daily_cache(
    storage: Storage,
    symbols: list[str],
    *,
    exchange: str,
    end_ts: pd.Timestamp,
    progress_callback: Callable[[dict[str, Any]], None] | None,
) -> dict[str, pd.DataFrame]:
    cache: dict[str, pd.DataFrame] = {}
    for completed, symbol in enumerate(symbols, start=1):
        daily = _prepare_daily(storage.load_candles(exchange, symbol, "1D"))
        daily = daily.loc[daily["date"] <= end_ts].reset_index(drop=True)
        if not daily.empty:
            cache[symbol] = daily
        _emit_progress(progress_callback, "LOAD", completed, len(symbols), symbol)
    return cache


def _entry_outcomes(
    daily: pd.DataFrame,
    signal_mask: np.ndarray,
    *,
    symbol: str,
    exchange: str,
    parameter_name: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    horizon: int,
    round_trip_cost_pct: float,
    lower_distance: np.ndarray,
    cmf: pd.Series,
) -> list[dict[str, Any]]:
    rows = []
    dates = daily["date"]
    eligible = signal_mask & dates.between(start_ts, end_ts).to_numpy(dtype=bool)
    next_allowed_entry = 0
    for signal_index in np.flatnonzero(eligible):
        entry_index = int(signal_index) + 1
        if entry_index < next_allowed_entry:
            continue
        outcome_start = entry_index + 1
        outcome_end = entry_index + horizon
        if outcome_end >= len(daily):
            continue
        entry_price = float(daily.iloc[entry_index]["high"])
        if not np.isfinite(entry_price) or entry_price <= 0.0:
            continue
        future = daily.iloc[outcome_start : outcome_end + 1]
        highs = future["high"].to_numpy(dtype=float)
        lows = future["low"].to_numpy(dtype=float)
        high_returns = (highs / entry_price - 1.0) * 100.0
        low_returns = (lows / entry_price - 1.0) * 100.0
        target_first = {
            target: _first_true(high_returns >= target) for target in (5.0, 7.0, 10.0)
        }
        stop_first = _first_true(low_returns <= -5.0)
        forward_returns = {}
        for sessions in (5, 10, 20):
            close_value = float(daily.iloc[entry_index + sessions]["close"])
            forward_returns[sessions] = (
                (close_value / entry_price - 1.0) * 100.0 - float(round_trip_cost_pct)
            )
        target7_index = target_first[7.0]
        target7_before_stop = target7_index is not None and (
            stop_first is None or target7_index < stop_first
        )
        entry_date = pd.Timestamp(daily.iloc[entry_index]["date"])
        rows.append(
            {
                "exchange": exchange,
                "symbol": symbol,
                "parameter_name": parameter_name,
                "signal_date": daily.iloc[int(signal_index)]["date"],
                "entry_date": entry_date,
                "entry_price": entry_price,
                "cohort": _cohort(entry_date),
                "signal_low_from_lower_pct": float(lower_distance[int(signal_index)]),
                "signal_cmf": float(cmf.iloc[int(signal_index)]),
                "target_5_within_10": target_first[5.0] is not None and target_first[5.0] < 10,
                "target_7_within_10": target7_index is not None and target7_index < 10,
                "target_7_within_20": target7_index is not None,
                "target_10_within_20": target_first[10.0] is not None,
                "target_7_before_stop_5_within_20": target7_before_stop,
                "stop_5_within_20": stop_first is not None,
                "return_5_pct": forward_returns[5],
                "return_10_pct": forward_returns[10],
                "return_20_pct": forward_returns[20],
                "mfe_20_pct": float(np.nanmax(high_returns)),
                "mae_20_pct": float(np.nanmin(low_returns)),
            }
        )
        next_allowed_entry = entry_index + horizon + 1
    return rows


def _parameter_stats(
    trades: pd.DataFrame,
    parameters: tuple[EntryStrategyParameters, ...],
    stage: str,
) -> pd.DataFrame:
    parameter_map = {item.name: item for item in parameters}
    rows = []
    if trades.empty:
        return pd.DataFrame()
    for (name, cohort), group in trades.groupby(["parameter_name", "cohort"]):
        item = parameter_map[str(name)]
        returns_20 = pd.to_numeric(group["return_20_pct"], errors="coerce")
        target7_stop = group["target_7_before_stop_5_within_20"].fillna(False)
        row = {
            "stage": stage,
            "parameter_name": name,
            "cohort": cohort,
            "knox_lookback": item.knox_lookback,
            "rsi_length": item.rsi_length,
            "momentum_length": item.momentum_length,
            "envelope_length": item.envelope_length,
            "envelope_percent": item.envelope_percent,
            "proximity_pct": item.proximity_pct,
            "cmf_length": item.cmf_length,
            "cmf_mode": item.cmf_mode,
            "cmf_min": item.cmf_min,
            "cmf_max": item.cmf_max,
            "trades": len(group),
            "entry_score": (
                float(target7_stop.mean() * 50.0)
                + float(group["target_7_within_20"].mean() * 30.0)
                + float((returns_20 > 0.0).mean() * 20.0)
            ),
            "target_5_within_10_pct": float(group["target_5_within_10"].mean() * 100.0),
            "target_7_within_10_pct": float(group["target_7_within_10"].mean() * 100.0),
            "target_7_within_20_pct": float(group["target_7_within_20"].mean() * 100.0),
            "target_10_within_20_pct": float(group["target_10_within_20"].mean() * 100.0),
            "target_7_before_stop_5_pct": float(target7_stop.mean() * 100.0),
            "stop_5_within_20_pct": float(group["stop_5_within_20"].mean() * 100.0),
            "win_5_pct": float((group["return_5_pct"] > 0.0).mean() * 100.0),
            "win_10_pct": float((group["return_10_pct"] > 0.0).mean() * 100.0),
            "win_20_pct": float((returns_20 > 0.0).mean() * 100.0),
            "median_return_5_pct": float(group["return_5_pct"].median()),
            "median_return_10_pct": float(group["return_10_pct"].median()),
            "median_return_20_pct": float(returns_20.median()),
            "median_mfe_20_pct": float(group["mfe_20_pct"].median()),
            "median_mae_20_pct": float(group["mae_20_pct"].median()),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _top_parameters(
    stats: pd.DataFrame,
    parameters: tuple[EntryStrategyParameters, ...],
    minimum_validation_trades: int,
    count: int,
) -> list[EntryStrategyParameters]:
    parameter_map = {item.name: item for item in parameters}
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

    preferred = rank(
        validation.loc[
            (validation["trades"] >= int(minimum_validation_trades))
            & (validation["median_return_20_pct"] > 0.0)
        ]
    )
    sampled = rank(
        validation.loc[validation["trades"] >= int(minimum_validation_trades)]
    )
    ranked = pd.concat([preferred, sampled, rank(validation)], ignore_index=True)
    ranked = ranked.drop_duplicates("parameter_name", keep="first")
    names = ranked["parameter_name"].head(count)
    return [parameter_map[str(name)] for name in names]


def _lower_envelope_distance(
    daily: pd.DataFrame,
    length: int,
    percent: float,
) -> np.ndarray:
    basis = daily["close"].rolling(length, min_periods=length).mean()
    lower = basis * (1.0 - float(percent) / 100.0)
    return ((daily["low"] - lower) / lower.replace(0.0, np.nan) * 100.0).to_numpy(
        dtype=float
    )


def _cmf(daily: pd.DataFrame, length: int) -> pd.Series:
    volume = pd.to_numeric(daily["volume"], errors="coerce").fillna(0.0)
    candle_range = (daily["high"] - daily["low"]).replace(0.0, np.nan)
    multiplier = (
        ((daily["close"] - daily["low"]) - (daily["high"] - daily["close"]))
        / candle_range
    ).fillna(0.0)
    return (
        (multiplier * volume).rolling(length, min_periods=length).sum()
        / volume.rolling(length, min_periods=length).sum().replace(0.0, np.nan)
    )


def _fast_pine_rsi(close: pd.Series, length: int) -> pd.Series:
    values = pd.to_numeric(close, errors="coerce").to_numpy(dtype=float)
    delta = np.empty(len(values), dtype=float)
    delta[0] = np.nan
    delta[1:] = np.diff(values)
    gains = np.where(np.isnan(delta), np.nan, np.maximum(delta, 0.0))
    losses = np.where(np.isnan(delta), np.nan, np.maximum(-delta, 0.0))
    avg_gain = _fast_pine_rma(gains, length)
    avg_loss = _fast_pine_rma(losses, length)
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
    rsi[avg_loss == 0.0] = 100.0
    rsi[(avg_gain == 0.0) & (avg_loss == 0.0)] = 50.0
    return pd.Series(rsi, index=close.index, dtype=float)


def _fast_pine_rma(values: np.ndarray, length: int) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=float)
    valid_indexes = np.flatnonzero(np.isfinite(values))
    if length < 1 or len(valid_indexes) < length:
        return result
    seed_index = int(valid_indexes[length - 1])
    previous = float(np.mean(values[valid_indexes[:length]]))
    result[seed_index] = previous
    for position in range(seed_index + 1, len(values)):
        value = values[position]
        if np.isfinite(value):
            previous = ((previous * (length - 1.0)) + float(value)) / float(length)
        result[position] = previous
    return result


def _cmf_mask(cmf: pd.Series, item: EntryStrategyParameters) -> np.ndarray:
    if item.cmf_mode == "band":
        mask = cmf > item.cmf_min
        if item.cmf_max is not None:
            mask &= cmf <= item.cmf_max
    elif item.cmf_mode == "rising_positive":
        mask = (cmf > item.cmf_min) & (cmf > cmf.shift(3))
    elif item.cmf_mode == "crossed_recent":
        crossed = (cmf > 0.0) & (cmf.shift(1) <= 0.0)
        mask = (cmf > item.cmf_min) & crossed.rolling(3, min_periods=1).max().fillna(0.0).astype(bool)
    else:
        mask = cmf > item.cmf_min
    return mask.fillna(False).to_numpy(dtype=bool)


def _first_true(values: np.ndarray) -> int | None:
    indexes = np.flatnonzero(values)
    return int(indexes[0]) if len(indexes) else None


def _cohort(entry_date: pd.Timestamp) -> str:
    if entry_date.year <= 2021:
        return "DEVELOPMENT"
    if entry_date.year <= 2023:
        return "VALIDATION"
    return "HOLDOUT"


def _summary_cohort(prefix: str, frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    row = frame.iloc[0]
    metrics = (
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
    return {f"{prefix}_{metric}": row.get(metric) for metric in metrics}


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
