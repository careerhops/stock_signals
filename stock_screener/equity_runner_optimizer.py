from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from stock_screener.data.storage import Storage
from stock_screener.equity_runner_research import (
    calculate_runner_features,
    parse_equity_runner_report,
)


@dataclass(frozen=True)
class ExitPolicy:
    name: str
    target_pct: float
    stop_pct: float
    hold_sessions: int
    trailing: bool = False
    atr_adaptive: bool = False
    target_atr_multiple: float = 0.0
    stop_atr_multiple: float = 0.0
    min_target_pct: float = 5.0
    max_target_pct: float = 15.0
    min_stop_pct: float = 3.0
    max_stop_pct: float = 6.0

    def levels(self, atr_pct: float) -> tuple[float, float]:
        if not self.atr_adaptive:
            return self.target_pct, self.stop_pct
        target = float(np.clip(atr_pct * self.target_atr_multiple, self.min_target_pct, self.max_target_pct))
        stop = float(np.clip(atr_pct * self.stop_atr_multiple, self.min_stop_pct, self.max_stop_pct))
        return target, stop


@dataclass(frozen=True)
class EntryRule:
    family: str
    return20_min: float
    return20_max: float
    return3_min: float
    return3_max: float
    sma20_min: float
    sma20_max: float
    sma50_min: float
    sma50_max: float
    atr_min: float
    atr_max: float
    rsi_min: float
    rsi_max: float
    di_spread_min: float
    adx_min: float
    relative_volume_min: float
    range_position_min: float
    close_location_min: float
    breadth_min: float
    relative_strength20_min: float

    @property
    def name(self) -> str:
        values = asdict(self)
        family = values.pop("family")
        body = "__".join(f"{key}_{value:g}" for key, value in values.items())
        return f"{family.upper()}__{body}"


@dataclass(frozen=True)
class OptimizerResult:
    summary: dict[str, Any]
    entry_search: pd.DataFrame
    policy_search: pd.DataFrame
    split_metrics: pd.DataFrame
    stability_by_year: pd.DataFrame
    stability_by_symbol: pd.DataFrame
    latest_shortlist: pd.DataFrame
    selected_rule: EntryRule | None
    selected_policy: ExitPolicy | None


def build_optimizer_dataset(
    storage: Storage,
    report_path: str | Path,
    *,
    exchange: str = "NSE",
    min_date: str = "2021-08-11",
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    report = parse_equity_runner_report(report_path)
    frames: list[pd.DataFrame] = []
    for symbol in report.symbols:
        candles = storage.load_candles(exchange, symbol, "1D")
        featured = _extended_features(calculate_runner_features(candles))
        if featured.empty:
            continue
        featured["symbol"] = symbol
        featured["bar_index"] = np.arange(len(featured), dtype=int)
        frames.append(featured)
    if not frames:
        raise ValueError("No candle data is available for the Equity Runner cohort")

    data = pd.concat(frames, ignore_index=True)
    breadth = data.groupby("date", as_index=False).agg(
        breadth_above_sma20_pct=("distance_sma20_pct", lambda values: float((values > 0.0).mean() * 100.0)),
        breadth_di_positive_pct=("di_spread", lambda values: float((values > 0.0).mean() * 100.0)),
    )
    data = data.merge(breadth, on="date", how="left", validate="many_to_one")

    benchmark = _benchmark_features(storage.load_candles("NSE_INDEX", "NIFTY 50", "1D"))
    data = data.merge(benchmark, on="date", how="left", validate="many_to_one")
    data["relative_strength20_pct"] = data["return_20d_pct"] - data["nifty_return20_pct"]
    data = data.loc[data["date"] >= pd.Timestamp(min_date)].copy()
    data.reset_index(drop=True, inplace=True)
    data["bar_index"] = data.groupby("symbol", sort=False).cumcount()
    return data, report.symbols


def _extended_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    close = result["close"]
    high = result["high"]
    low = result["low"]
    volume = result["volume"].fillna(0.0)
    result["return_5d_pct"] = (close / close.shift(5) - 1.0) * 100.0
    result["return_10d_pct"] = (close / close.shift(10) - 1.0) * 100.0
    result["return_63d_pct"] = (close / close.shift(63) - 1.0) * 100.0
    result["distance_sma200_pct"] = (close / close.rolling(200, min_periods=200).mean() - 1.0) * 100.0
    sma20 = close.rolling(20, min_periods=20).mean()
    sma50 = close.rolling(50, min_periods=50).mean()
    result["sma20_slope5_pct"] = (sma20 / sma20.shift(5) - 1.0) * 100.0
    result["sma50_slope10_pct"] = (sma50 / sma50.shift(10) - 1.0) * 100.0
    prior_high55 = high.shift(1).rolling(55, min_periods=55).max()
    prior_low55 = low.shift(1).rolling(55, min_periods=55).min()
    result["range_position55_pct"] = (
        (close - prior_low55) / (prior_high55 - prior_low55).replace(0.0, np.nan) * 100.0
    )
    result["breakout20_pct"] = (close / high.shift(1).rolling(20, min_periods=20).max() - 1.0) * 100.0
    result["distance_52w_high_pct"] = (
        close / high.shift(1).rolling(252, min_periods=100).max() - 1.0
    ) * 100.0
    result["volume5_to20_ratio"] = (
        volume.rolling(5, min_periods=5).mean() / volume.rolling(20, min_periods=20).mean().replace(0.0, np.nan)
    )
    result["adx_slope3"] = result["adx14"] - result["adx14"].shift(3)
    result["di_spread_slope3"] = result["di_spread"] - result["di_spread"].shift(3)
    return result


def _benchmark_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=["date", "nifty_return20_pct", "nifty_above_sma200", "nifty_bullish"]
        )
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce", format="mixed")
    close = pd.to_numeric(result["close"], errors="coerce")
    sma50 = close.rolling(50, min_periods=50).mean()
    sma200 = close.rolling(200, min_periods=200).mean()
    result["nifty_return20_pct"] = (close / close.shift(20) - 1.0) * 100.0
    result["nifty_above_sma200"] = close > sma200
    result["nifty_bullish"] = (close > sma200) & (sma50 > sma200) & (sma200 > sma200.shift(20))
    return result[["date", "nifty_return20_pct", "nifty_above_sma200", "nifty_bullish"]]


def run_bruteforce_optimizer(
    storage: Storage,
    report_path: str | Path,
    *,
    random_rules: int = 30_000,
    seed: int = 20260906,
    min_average_traded_value: float = 10_000_000.0,
    development_end: str = "2023-12-31",
    validation_end: str = "2025-12-31",
    expectancy_gate: float = 1.3,
    profit_factor_gate: float = 1.5,
) -> OptimizerResult:
    data, symbols = build_optimizer_dataset(storage, report_path)
    base_mask = _base_eligibility(data, min_average_traded_value)
    development_mask = base_mask & (data["date"] <= pd.Timestamp(development_end))
    validation_mask = base_mask & data["date"].between(
        pd.Timestamp(development_end) + pd.Timedelta(days=1), pd.Timestamp(validation_end)
    )
    test_mask = base_mask & (data["date"] > pd.Timestamp(validation_end))

    reference_policy = ExitPolicy("FIXED_T7_S5_H7", 7.0, 5.0, 7)
    all_indexes = np.flatnonzero(base_mask.to_numpy())
    price_arrays = _price_arrays(data)
    reference = _simulate_indexes(data, all_indexes, reference_policy, price_arrays=price_arrays)
    reference_returns = pd.Series(np.nan, index=data.index, dtype=float)
    reference_returns.loc[reference["data_index"].astype(int)] = reference["return_pct"].to_numpy()

    entry_rows: list[dict[str, Any]] = []
    generated_rules = list(_random_entry_rules(random_rules, seed))
    for rule in generated_rules:
        rule_mask = _entry_rule_mask(data, rule) & base_mask
        development_returns = reference_returns.loc[rule_mask & development_mask].dropna()
        validation_returns = reference_returns.loc[rule_mask & validation_mask].dropna()
        if len(development_returns) < 60 or len(validation_returns) < 30:
            continue
        development = _return_metrics(development_returns)
        validation = _return_metrics(validation_returns)
        entry_rows.append(
            {
                "rule_name": rule.name,
                "rule_json": json.dumps(asdict(rule), sort_keys=True),
                "development_trades": development["trades"],
                "development_expectancy_pct": development["expectancy_pct"],
                "development_profit_factor": development["profit_factor"],
                "validation_trades": validation["trades"],
                "validation_expectancy_pct": validation["expectancy_pct"],
                "validation_profit_factor": validation["profit_factor"],
                "entry_score": _selection_score(development, validation),
            }
        )
    entry_search = pd.DataFrame(entry_rows)
    if entry_search.empty:
        return _empty_result(data, symbols, random_rules)
    entry_search = entry_search.sort_values(
        ["entry_score", "validation_expectancy_pct", "validation_trades"], ascending=[False, False, False]
    ).drop_duplicates("rule_name").reset_index(drop=True)

    finalist_rules = [_rule_from_json(raw) for raw in entry_search.head(50)["rule_json"]]
    union_mask = pd.Series(False, index=data.index)
    rule_masks: dict[str, pd.Series] = {}
    for rule in finalist_rules:
        mask = _entry_rule_mask(data, rule) & base_mask
        rule_masks[rule.name] = mask
        union_mask |= mask
    union_indexes = np.flatnonzero(union_mask.to_numpy())

    policies = list(_exit_policies())
    simulations = {
        policy.name: _simulate_indexes(
            data, union_indexes, policy, price_arrays=price_arrays
        ).set_index("data_index")
        for policy in policies
    }
    simulation_returns: dict[str, np.ndarray] = {}
    for policy in policies:
        values = np.full(len(data), np.nan)
        simulated = simulations[policy.name]
        values[simulated.index.to_numpy(dtype=int)] = simulated["return_pct"].to_numpy(dtype=float)
        simulation_returns[policy.name] = values
    policy_rows: list[dict[str, Any]] = []
    development_array = development_mask.to_numpy(dtype=bool)
    validation_array = validation_mask.to_numpy(dtype=bool)
    development_indexes = set(np.flatnonzero(development_array))
    validation_indexes = set(np.flatnonzero(validation_array))
    for rule in finalist_rules:
        mask = rule_masks[rule.name]
        rule_array = mask.to_numpy(dtype=bool)
        for policy in policies:
            returns = simulation_returns[policy.name]
            development = _return_metrics_array(returns[rule_array & development_array])
            validation = _return_metrics_array(returns[rule_array & validation_array])
            if development["trades"] < 60 or validation["trades"] < 30:
                continue
            policy_rows.append(
                {
                    "rule_name": rule.name,
                    "rule_json": json.dumps(asdict(rule), sort_keys=True),
                    "policy_name": policy.name,
                    "policy_json": json.dumps(asdict(policy), sort_keys=True),
                    **{f"development_{key}": value for key, value in development.items()},
                    **{f"validation_{key}": value for key, value in validation.items()},
                    "selection_score": _selection_score(development, validation),
                }
            )
    policy_search = pd.DataFrame(policy_rows)
    if policy_search.empty:
        return _empty_result(data, symbols, random_rules, entry_search=entry_search)
    policy_search = policy_search.sort_values(
        ["selection_score", "validation_expectancy_pct", "validation_trades"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    exact_rows: list[dict[str, Any]] = []
    for _, candidate in policy_search.head(100).iterrows():
        rule = _rule_from_json(candidate["rule_json"])
        policy = _policy_from_json(candidate["policy_json"])
        indexes = np.flatnonzero(rule_masks[rule.name].to_numpy())
        selected = simulations[policy.name].reindex(indexes).dropna(subset=["return_pct"])
        development = _non_overlapping_metrics(
            data, selected.loc[selected.index.isin(development_indexes)]
        )
        validation = _non_overlapping_metrics(
            data, selected.loc[selected.index.isin(validation_indexes)]
        )
        if development["trades"] < 40 or validation["trades"] < 25:
            continue
        exact_rows.append(
            {
                **candidate.to_dict(),
                **{f"development_{key}": value for key, value in development.items()},
                **{f"validation_{key}": value for key, value in validation.items()},
                "selection_score": _selection_score(development, validation),
            }
        )
    if not exact_rows:
        return _empty_result(data, symbols, random_rules, entry_search=entry_search)
    policy_search = pd.DataFrame(exact_rows).sort_values(
        ["selection_score", "validation_expectancy_pct", "validation_trades"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    selected_row = policy_search.iloc[0]
    selected_rule = _rule_from_json(selected_row["rule_json"])
    selected_policy = _policy_from_json(selected_row["policy_json"])
    selected_mask = rule_masks[selected_rule.name]
    selected_simulation = simulations[selected_policy.name]

    split_rows: list[dict[str, Any]] = []
    split_definitions = (
        ("DEVELOPMENT_2021_2023", development_mask),
        ("VALIDATION_2024_2025", validation_mask),
        ("HOLDOUT_2026", test_mask),
        ("HOLDOUT_2026_H1", test_mask & (data["date"] < pd.Timestamp("2026-07-01"))),
        ("HOLDOUT_2026_H2", test_mask & (data["date"] >= pd.Timestamp("2026-07-01"))),
    )
    chosen_trades: list[pd.DataFrame] = []
    for split_name, split_mask in split_definitions:
        indexes = np.flatnonzero((selected_mask & split_mask).to_numpy())
        trades = _select_non_overlapping(data, selected_simulation.reindex(indexes).dropna(subset=["return_pct"]))
        trades["split"] = split_name
        # H1/H2 are diagnostic views of HOLDOUT_2026. Excluding those slices
        # here prevents the yearly and symbol stability tables double-counting
        # the same holdout trades.
        if split_name in {"DEVELOPMENT_2021_2023", "VALIDATION_2024_2025", "HOLDOUT_2026"}:
            chosen_trades.append(trades)
        split_rows.append({"split": split_name, **_return_metrics(trades["return_pct"])})
    split_metrics = pd.DataFrame(split_rows)
    all_chosen = pd.concat(chosen_trades, ignore_index=True) if chosen_trades else pd.DataFrame()
    stability_by_year = _stability_table(all_chosen, data, "year")
    stability_by_symbol = _stability_table(all_chosen, data, "symbol")

    latest_shortlist = _latest_shortlist(data, selected_rule, selected_policy)
    expansion_ready = _gate_passed(split_metrics, expectancy_gate, profit_factor_gate)
    summary = {
        "cohort_symbols": len(symbols),
        "dataset_rows": len(data),
        "random_rules_requested": random_rules,
        "entry_rules_evaluated": len(entry_search),
        "rule_policy_pairs_evaluated": len(policy_search),
        "selected_rule": selected_rule.name,
        "selected_policy": selected_policy.name,
        "latest_data_date": data["date"].max().strftime("%Y-%m-%d"),
        "latest_shortlist_count": len(latest_shortlist),
        "expectancy_gate_pct": expectancy_gate,
        "profit_factor_gate": profit_factor_gate,
        "expansion_ready": expansion_ready,
    }
    return OptimizerResult(
        summary,
        entry_search,
        policy_search,
        split_metrics,
        stability_by_year,
        stability_by_symbol,
        latest_shortlist,
        selected_rule,
        selected_policy,
    )


def _base_eligibility(data: pd.DataFrame, min_average_traded_value: float) -> pd.Series:
    required = [
        "return_3d_pct",
        "return_20d_pct",
        "distance_sma20_pct",
        "distance_sma50_pct",
        "atr14_pct",
        "rsi14",
        "relative_volume20",
        "adx14",
        "di_spread",
        "range_position20_pct",
        "close_location_pct",
        "breadth_above_sma20_pct",
        "relative_strength20_pct",
    ]
    return (
        data[required].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        & data["data_quality_pass"].fillna(False)
        & (data["average_traded_value20"] >= min_average_traded_value)
    )


def _random_entry_rules(count: int, seed: int) -> Iterable[EntryRule]:
    rng = np.random.default_rng(seed)
    families = ("momentum", "pullback", "breakout", "reacceleration")
    seen: set[str] = set()
    for index in range(count):
        family = families[index % len(families)]
        values: dict[str, float | str] = {
            "family": family,
            "return20_min": float(rng.choice([0, 5, 10, 15, 20])),
            "return20_max": float(rng.choice([25, 35, 50, 80, 200])),
            "return3_min": float(rng.choice([-8, -3, 0, 1, 3])),
            "return3_max": float(rng.choice([3, 6, 10, 20, 100])),
            "sma20_min": float(rng.choice([-5, -2, 0, 2, 5])),
            "sma20_max": float(rng.choice([5, 10, 15, 25, 100])),
            "sma50_min": float(rng.choice([-10, 0, 5, 10])),
            "sma50_max": float(rng.choice([20, 35, 50, 100])),
            "atr_min": float(rng.choice([1.5, 2.5, 3.5, 4.5])),
            "atr_max": float(rng.choice([5, 7, 10, 20])),
            "rsi_min": float(rng.choice([35, 45, 50, 55, 60])),
            "rsi_max": float(rng.choice([60, 65, 70, 75, 80, 90])),
            "di_spread_min": float(rng.choice([-10, 0, 5, 10, 15])),
            "adx_min": float(rng.choice([0, 15, 20, 25, 30])),
            "relative_volume_min": float(rng.choice([0, 0.5, 0.8, 1, 1.2, 1.5, 2])),
            "range_position_min": float(rng.choice([0, 40, 60, 75, 90])),
            "close_location_min": float(rng.choice([0, 30, 50, 70])),
            "breadth_min": float(rng.choice([0, 40, 50, 60])),
            "relative_strength20_min": float(rng.choice([-20, -5, 0, 5, 10])),
        }
        if family == "pullback":
            values.update(
                return20_min=float(rng.choice([0, 5, 10])),
                return20_max=float(rng.choice([25, 35, 50])),
                return3_min=float(rng.choice([-12, -8, -5, -3])),
                return3_max=float(rng.choice([0, 2, 4])),
                sma20_min=float(rng.choice([-6, -4, -2])),
                sma20_max=float(rng.choice([1, 3, 5])),
                rsi_max=float(rng.choice([55, 60, 65])),
            )
        elif family == "breakout":
            values.update(
                relative_volume_min=float(rng.choice([1, 1.2, 1.5, 2])),
                range_position_min=float(rng.choice([85, 90, 95, 100])),
                close_location_min=float(rng.choice([50, 60, 70, 80])),
            )
        elif family == "reacceleration":
            values.update(
                return3_min=float(rng.choice([0, 1, 3])),
                return3_max=float(rng.choice([6, 10, 20])),
                di_spread_min=float(rng.choice([0, 5, 10])),
                adx_min=float(rng.choice([15, 20, 25])),
            )
        rule = EntryRule(**values)
        if not (
            rule.return20_min < rule.return20_max
            and rule.return3_min < rule.return3_max
            and rule.sma20_min < rule.sma20_max
            and rule.sma50_min < rule.sma50_max
            and rule.atr_min < rule.atr_max
            and rule.rsi_min < rule.rsi_max
        ):
            continue
        if rule.name in seen:
            continue
        seen.add(rule.name)
        yield rule


def _entry_rule_mask(data: pd.DataFrame, rule: EntryRule) -> pd.Series:
    mask = (
        data["return_20d_pct"].between(rule.return20_min, rule.return20_max)
        & data["return_3d_pct"].between(rule.return3_min, rule.return3_max)
        & data["distance_sma20_pct"].between(rule.sma20_min, rule.sma20_max)
        & data["distance_sma50_pct"].between(rule.sma50_min, rule.sma50_max)
        & data["atr14_pct"].between(rule.atr_min, rule.atr_max)
        & data["rsi14"].between(rule.rsi_min, rule.rsi_max)
        & (data["di_spread"] >= rule.di_spread_min)
        & (data["adx14"] >= rule.adx_min)
        & (data["relative_volume20"] >= rule.relative_volume_min)
        & (data["range_position20_pct"] >= rule.range_position_min)
        & (data["close_location_pct"] >= rule.close_location_min)
        & (data["breadth_above_sma20_pct"] >= rule.breadth_min)
        & (data["relative_strength20_pct"] >= rule.relative_strength20_min)
    )
    if rule.family == "pullback":
        mask &= data["sma50_slope10_pct"] > 0.0
    elif rule.family == "breakout":
        mask &= data["breakout20_pct"] >= -1.0
    elif rule.family == "reacceleration":
        mask &= (data["di_spread_slope3"] > 0.0) & (data["adx_slope3"] > -3.0)
    return mask.fillna(False)


def _exit_policies() -> Iterable[ExitPolicy]:
    for target in (5.0, 6.0, 7.0, 8.0, 10.0, 12.0):
        for stop in (3.0, 4.0, 5.0, 6.0):
            for hold in (5, 7, 10):
                yield ExitPolicy(f"FIXED_T{target:g}_S{stop:g}_H{hold}", target, stop, hold)
    for target_multiple in (1.5, 2.0, 2.5):
        for stop_multiple in (0.75, 1.0, 1.25):
            for hold in (5, 7, 10):
                yield ExitPolicy(
                    f"ATR_T{target_multiple:g}_S{stop_multiple:g}_H{hold}",
                    0.0,
                    0.0,
                    hold,
                    atr_adaptive=True,
                    target_atr_multiple=target_multiple,
                    stop_atr_multiple=stop_multiple,
                )
    yield ExitPolicy("TRAIL_T7_S5_H7", 7.0, 5.0, 7, trailing=True)
    yield ExitPolicy("TRAIL_T8_S4_H7", 8.0, 4.0, 7, trailing=True)
    yield ExitPolicy("TRAIL_T10_S5_H7", 10.0, 5.0, 7, trailing=True)


def _price_arrays(data: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    return {
        symbol: {
            "open": group.sort_values("bar_index")["open"].to_numpy(dtype=float),
            "high": group.sort_values("bar_index")["high"].to_numpy(dtype=float),
            "low": group.sort_values("bar_index")["low"].to_numpy(dtype=float),
            "close": group.sort_values("bar_index")["close"].to_numpy(dtype=float),
            "date": group.sort_values("bar_index")["date"].to_numpy(),
        }
        for symbol, group in data.groupby("symbol", sort=False)
    }


def _simulate_indexes(
    data: pd.DataFrame,
    indexes: np.ndarray,
    policy: ExitPolicy,
    *,
    price_arrays: dict[str, dict[str, np.ndarray]] | None = None,
) -> pd.DataFrame:
    arrays = price_arrays or _price_arrays(data)
    rows: list[dict[str, Any]] = []
    for data_index in indexes:
        row = data.loc[data_index]
        symbol = str(row["symbol"])
        bar = int(row["bar_index"])
        values = arrays[symbol]
        if bar + 1 >= len(values["open"]):
            continue
        target_pct, stop_pct = policy.levels(float(row["atr14_pct"]))
        outcome = _simulate_one(values, bar, target_pct, stop_pct, policy.hold_sessions, policy.trailing)
        if outcome is None:
            continue
        return_pct, exit_bar, exit_reason = outcome
        rows.append(
            {
                "data_index": int(data_index),
                "return_pct": return_pct - 0.15,
                "exit_bar": exit_bar,
                "exit_date": values["date"][exit_bar],
                "exit_reason": exit_reason,
                "target_pct": target_pct,
                "stop_pct": stop_pct,
                "hold_sessions": policy.hold_sessions,
            }
        )
    columns = [
        "data_index",
        "return_pct",
        "exit_bar",
        "exit_date",
        "exit_reason",
        "target_pct",
        "stop_pct",
        "hold_sessions",
    ]
    return pd.DataFrame(rows, columns=columns)


def _simulate_one(
    values: dict[str, np.ndarray],
    signal_bar: int,
    target_pct: float,
    stop_pct: float,
    hold_sessions: int,
    trailing: bool,
) -> tuple[float, int, str] | None:
    entry_bar = signal_bar + 1
    entry_price = float(values["open"][entry_bar])
    if not np.isfinite(entry_price) or entry_price <= 0.0:
        return None
    target_price = entry_price * (1.0 + target_pct / 100.0)
    stop_floor = entry_price * (1.0 - stop_pct / 100.0)
    active_stop = stop_floor
    peak = entry_price
    last_bar = min(len(values["open"]) - 1, signal_bar + hold_sessions)
    for bar in range(entry_bar, last_bar + 1):
        if values["open"][bar] <= active_stop:
            return (values["open"][bar] / entry_price - 1.0) * 100.0, bar, "GAP_STOP"
        if values["open"][bar] >= target_price:
            return (values["open"][bar] / entry_price - 1.0) * 100.0, bar, "GAP_TARGET"
        if values["low"][bar] <= active_stop:
            return (active_stop / entry_price - 1.0) * 100.0, bar, "STOP"
        if values["high"][bar] >= target_price:
            return (target_price / entry_price - 1.0) * 100.0, bar, "TARGET"
        if bar == last_bar:
            return (values["close"][bar] / entry_price - 1.0) * 100.0, bar, "MAX_HOLD"
        if trailing:
            peak = max(peak, float(values["high"][bar]))
            active_stop = max(stop_floor, peak * (1.0 - stop_pct / 100.0))
    return None


def _select_non_overlapping(data: pd.DataFrame, simulated: pd.DataFrame) -> pd.DataFrame:
    if simulated.empty:
        return simulated.copy()
    joined = simulated.join(data[["symbol", "date", "bar_index"]], how="left")
    selected: list[int] = []
    for _, group in joined.sort_values(["symbol", "bar_index"]).groupby("symbol", sort=False):
        blocked_through = -1
        for index, row in group.iterrows():
            if int(row["bar_index"]) <= blocked_through:
                continue
            selected.append(index)
            blocked_through = int(row["exit_bar"])
    return joined.loc[selected].sort_values(["date", "symbol"]).copy()


def _non_overlapping_metrics(data: pd.DataFrame, simulated: pd.DataFrame) -> dict[str, Any]:
    return _return_metrics(_select_non_overlapping(data, simulated)["return_pct"])


def _return_metrics(returns: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if values.empty:
        return {"trades": 0, "win_rate_pct": math.nan, "expectancy_pct": math.nan, "profit_factor": math.nan}
    wins = values.loc[values > 0.0]
    losses = values.loc[values <= 0.0]
    gross_loss = abs(float(losses.sum()))
    profit_factor = float(wins.sum()) / gross_loss if gross_loss else math.inf
    return {
        "trades": int(len(values)),
        "win_rate_pct": float(len(wins) / len(values) * 100.0),
        "expectancy_pct": float(values.mean()),
        "profit_factor": profit_factor,
        "average_win_pct": float(wins.mean()) if len(wins) else math.nan,
        "average_loss_pct": float(losses.mean()) if len(losses) else math.nan,
    }


def _return_metrics_array(returns: np.ndarray) -> dict[str, Any]:
    values = returns[np.isfinite(returns)]
    if len(values) == 0:
        return {"trades": 0, "win_rate_pct": math.nan, "expectancy_pct": math.nan, "profit_factor": math.nan}
    win_mask = values > 0.0
    gross_profit = float(values[win_mask].sum())
    gross_loss = abs(float(values[~win_mask].sum()))
    return {
        "trades": int(len(values)),
        "win_rate_pct": float(win_mask.mean() * 100.0),
        "expectancy_pct": float(values.mean()),
        "profit_factor": gross_profit / gross_loss if gross_loss else math.inf,
        "average_win_pct": float(values[win_mask].mean()) if win_mask.any() else math.nan,
        "average_loss_pct": float(values[~win_mask].mean()) if (~win_mask).any() else math.nan,
    }


def _selection_score(development: dict[str, Any], validation: dict[str, Any]) -> float:
    expectancies = [development["expectancy_pct"], validation["expectancy_pct"]]
    factors = [development["profit_factor"], validation["profit_factor"]]
    if not all(np.isfinite(value) for value in expectancies + factors):
        return -math.inf
    weakest_expectancy = min(expectancies)
    weakest_factor = min(factors)
    count_weight = min(1.0, math.sqrt(validation["trades"] / 60.0))
    stability = 1.0 / (1.0 + abs(expectancies[0] - expectancies[1]))
    if weakest_expectancy <= 0.0 or weakest_factor <= 1.0:
        return weakest_expectancy - abs(1.0 - weakest_factor)
    return weakest_expectancy * weakest_factor * count_weight * stability


def _rule_from_json(raw: str) -> EntryRule:
    return EntryRule(**json.loads(raw))


def _policy_from_json(raw: str) -> ExitPolicy:
    return ExitPolicy(**json.loads(raw))


def _stability_table(trades: pd.DataFrame, data: pd.DataFrame, grouping: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    frame = trades.copy()
    if grouping == "year":
        frame["group"] = pd.to_datetime(frame["date"]).dt.year.astype(str)
    else:
        frame["group"] = frame["symbol"]
    rows = [{grouping: name, **_return_metrics(group["return_pct"])} for name, group in frame.groupby("group")]
    return pd.DataFrame(rows).sort_values("expectancy_pct", ascending=False).reset_index(drop=True)


def _latest_shortlist(data: pd.DataFrame, rule: EntryRule, policy: ExitPolicy) -> pd.DataFrame:
    latest_date = data["date"].max()
    latest = data.loc[data["date"] == latest_date].copy()
    latest = latest.loc[_base_eligibility(latest, 10_000_000.0) & _entry_rule_mask(latest, rule)].copy()
    levels = latest["atr14_pct"].apply(policy.levels)
    latest["suggested_target_pct"] = [value[0] for value in levels]
    latest["suggested_stop_pct"] = [value[1] for value in levels]
    latest["suggested_hold_sessions"] = policy.hold_sessions
    columns = [
        "symbol",
        "date",
        "close",
        "suggested_target_pct",
        "suggested_stop_pct",
        "suggested_hold_sessions",
        "return_20d_pct",
        "return_3d_pct",
        "distance_sma20_pct",
        "distance_sma50_pct",
        "atr14_pct",
        "rsi14",
        "di_spread",
        "adx14",
        "relative_volume20",
        "relative_strength20_pct",
        "breadth_above_sma20_pct",
    ]
    return latest[columns].sort_values(["relative_strength20_pct", "di_spread"], ascending=False)


def _gate_passed(metrics: pd.DataFrame, expectancy_gate: float, profit_factor_gate: float) -> bool:
    required = metrics.loc[metrics["split"].isin(["VALIDATION_2024_2025", "HOLDOUT_2026"])]
    return bool(
        len(required) == 2
        and (required["trades"] >= 25).all()
        and (required["expectancy_pct"] > expectancy_gate).all()
        and (required["profit_factor"] > profit_factor_gate).all()
    )


def _empty_result(
    data: pd.DataFrame,
    symbols: tuple[str, ...],
    random_rules: int,
    *,
    entry_search: pd.DataFrame | None = None,
) -> OptimizerResult:
    summary = {
        "cohort_symbols": len(symbols),
        "dataset_rows": len(data),
        "random_rules_requested": random_rules,
        "entry_rules_evaluated": 0 if entry_search is None else len(entry_search),
        "rule_policy_pairs_evaluated": 0,
        "selected_rule": "",
        "selected_policy": "",
        "latest_data_date": data["date"].max().strftime("%Y-%m-%d"),
        "latest_shortlist_count": 0,
        "expansion_ready": False,
    }
    return OptimizerResult(
        summary,
        pd.DataFrame() if entry_search is None else entry_search,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        None,
        None,
    )


def save_optimizer_result(result: OptimizerResult, output_dir: str | Path) -> dict[str, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    frames = {
        "entry_search": result.entry_search,
        "policy_search": result.policy_search,
        "split_metrics": result.split_metrics,
        "stability_by_year": result.stability_by_year,
        "stability_by_symbol": result.stability_by_symbol,
        "latest_shortlist": result.latest_shortlist,
    }
    paths: dict[str, Path] = {}
    for name, frame in frames.items():
        path = directory / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    report_path = directory / "report.md"
    report_path.write_text(_report_markdown(result), encoding="utf-8")
    paths["report"] = report_path
    return paths


def _report_markdown(result: OptimizerResult) -> str:
    summary = result.summary
    lines = [
        "# Equity Runner Brute-force Optimizer",
        "",
        "Research output only. The 2026 holdout is never used to select parameters.",
        "",
        f"- Cohort: {summary['cohort_symbols']} symbols",
        f"- Entry rules evaluated: {summary['entry_rules_evaluated']}",
        f"- Rule/exit pairs evaluated: {summary['rule_policy_pairs_evaluated']}",
        f"- Selected entry: {summary['selected_rule'] or 'None'}",
        f"- Selected exit: {summary['selected_policy'] or 'None'}",
        f"- Expansion ready: {summary['expansion_ready']}",
        "",
        "## Walk-forward metrics",
        "",
        "```",
        result.split_metrics.to_string(index=False) if not result.split_metrics.empty else "No eligible result",
        "```",
        "",
        "## Latest cohort shortlist",
        "",
        ", ".join(result.latest_shortlist.get("symbol", pd.Series(dtype=str)).astype(str)) or "None",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Brute-force Equity Runner entries and exits")
    parser.add_argument("--report", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="data/research/equity_runner/optimizer")
    parser.add_argument("--rules", type=int, default=30_000)
    args = parser.parse_args()
    result = run_bruteforce_optimizer(
        Storage(Path(args.data_root)),
        args.report,
        random_rules=max(1_000, args.rules),
    )
    paths = save_optimizer_result(result, args.output)
    print(pd.Series(result.summary).to_string())
    print("\nWalk-forward metrics")
    print(result.split_metrics.to_string(index=False))
    print("\nLatest shortlist")
    print(",".join(result.latest_shortlist.get("symbol", pd.Series(dtype=str)).astype(str)))
    print(f"\nReport: {paths['report']}")


if __name__ == "__main__":
    main()
