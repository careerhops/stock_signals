from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from stock_screener.data.storage import Storage
from stock_screener.resample import resample_daily_to_weekly
from stock_screener.strategy.weekly_buy_sell import run_weekly_buy_sell
from stock_screener.universe import build_universe


@dataclass(frozen=True)
class RotationStudyResult:
    summary: dict[str, Any]
    groups: pd.DataFrame
    members: pd.DataFrame
    candidates: pd.DataFrame


DEFAULT_CONFIG = {
    "lookback_weeks": 260,
    "min_history_weeks": 104,
    "min_overlap_weeks": 52,
    "min_group_size": 5,
    "target_group_size": 12,
    "max_group_size": 20,
    "min_correlation": 0.68,
    "lag_window_weeks": 8,
    "min_lag_correlation": 0.45,
    "catch_up_gap_pct": 8.0,
    "group_strength_min_8w": 5.0,
}


def run_rotation_study(
    config: dict[str, Any],
    storage: Storage,
    exchange: str = "NSE",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> RotationStudyResult:
    study_cfg = {**DEFAULT_CONFIG, **(config.get("rotation_study", {}) or {})}
    universe_rows = _kite_instruments_universe(storage, config, exchange)
    if not universe_rows:
        return RotationStudyResult(_empty_summary(exchange), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    strategy_cfg = config.get("strategy", {})
    weekly_anchor = strategy_cfg.get("weekly_anchor", "W-FRI")
    use_completed_weeks_only = bool(strategy_cfg.get("use_completed_weeks_only", True))

    weekly_close: dict[str, pd.Series] = {}
    symbol_names: dict[str, str] = {}
    latest_signals: dict[str, dict[str, Any]] = {}
    symbols_processed = 0
    symbols_with_history = 0

    _emit_progress(progress_callback, "Preparing weekly price histories", 0, len(universe_rows), "", exchange)
    for index, row in enumerate(universe_rows, start=1):
        symbol = str(row.get("symbol") or "").upper()
        name = str(row.get("name") or symbol)
        if not symbol:
            continue

        symbols_processed += 1
        _emit_progress(progress_callback, "Preparing weekly price histories", index - 1, len(universe_rows), symbol, exchange)
        daily = storage.load_candles(exchange, symbol, "1D")
        if daily.empty:
            continue
        weekly = resample_daily_to_weekly(daily, weekly_anchor, use_completed_weeks_only)
        if weekly.empty or "close" not in weekly.columns:
            continue
        weekly = weekly.copy()
        weekly["date"] = pd.to_datetime(weekly["date"], errors="coerce")
        weekly = weekly.sort_values("date").dropna(subset=["date"])
        strategy_output = run_weekly_buy_sell(weekly, config)
        latest_signals[symbol] = _latest_week_signal_context(strategy_output, weekly)
        closes = pd.to_numeric(weekly["close"], errors="coerce")
        series = pd.Series(closes.values, index=weekly["date"], name=symbol).dropna()
        if len(series) < int(study_cfg["min_history_weeks"]):
            continue
        weekly_close[symbol] = series
        symbol_names[symbol] = name
        symbols_with_history += 1
        _emit_progress(progress_callback, "Preparing weekly price histories", index, len(universe_rows), symbol, exchange)

    if not weekly_close:
        return RotationStudyResult(
            {
                **_empty_summary(exchange),
                "symbols_processed": symbols_processed,
                "symbols_with_history": symbols_with_history,
            },
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    close_matrix = pd.concat(weekly_close.values(), axis=1).sort_index()
    close_matrix = close_matrix.tail(int(study_cfg["lookback_weeks"]))
    returns = close_matrix.pct_change().replace([np.inf, -np.inf], np.nan)

    eligible_symbols = [
        symbol
        for symbol in returns.columns
        if int(returns[symbol].notna().sum()) >= int(study_cfg["min_overlap_weeks"])
    ]
    if not eligible_symbols:
        return RotationStudyResult(
            {
                **_empty_summary(exchange),
                "symbols_processed": symbols_processed,
                "symbols_with_history": symbols_with_history,
            },
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    returns = returns[eligible_symbols]
    close_matrix = close_matrix[eligible_symbols]
    correlation = returns.corr(min_periods=int(study_cfg["min_overlap_weeks"]))
    groups = _build_groups_from_correlation(
        correlation,
        min_correlation=float(study_cfg["min_correlation"]),
        min_group_size=int(study_cfg["min_group_size"]),
        target_group_size=int(study_cfg["target_group_size"]),
        max_group_size=int(study_cfg["max_group_size"]),
    )

    _emit_progress(progress_callback, "Scoring groups and catch-up candidates", len(eligible_symbols), len(eligible_symbols), "", exchange)
    groups_frame, members_frame = _build_group_outputs(
        groups=groups,
        close_matrix=close_matrix,
        returns=returns,
        correlation=correlation,
        symbol_names=symbol_names,
        latest_signals=latest_signals,
        exchange=exchange,
        study_cfg=study_cfg,
    )
    candidates_frame = _build_candidates_frame(members_frame)
    summary = _build_summary(
        exchange=exchange,
        universe_rows=len(universe_rows),
        symbols_processed=symbols_processed,
        symbols_with_history=symbols_with_history,
        eligible_symbols=len(eligible_symbols),
        groups_frame=groups_frame,
        members_frame=members_frame,
        candidates_frame=candidates_frame,
    )
    return RotationStudyResult(summary, groups_frame, members_frame, candidates_frame)


def save_rotation_study_outputs(result: RotationStudyResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result.summary]).to_csv(output_dir / "latest_summary.csv", index=False)
    result.groups.to_csv(output_dir / "latest_groups.csv", index=False)
    result.members.to_csv(output_dir / "latest_members.csv", index=False)
    result.candidates.to_csv(output_dir / "latest_candidates.csv", index=False)


def load_rotation_study_outputs(output_dir: Path) -> RotationStudyResult:
    summary = {}
    groups = _load_csv(output_dir / "latest_groups.csv")
    members = _load_csv(output_dir / "latest_members.csv")
    candidates = _load_csv(output_dir / "latest_candidates.csv")
    summary_path = output_dir / "latest_summary.csv"
    if summary_path.exists():
        frame = _load_csv(summary_path)
        if not frame.empty:
            summary = frame.iloc[0].to_dict()
    return RotationStudyResult(summary, groups, members, candidates)


def _build_group_outputs(
    groups: list[list[str]],
    close_matrix: pd.DataFrame,
    returns: pd.DataFrame,
    correlation: pd.DataFrame,
    symbol_names: dict[str, str],
    latest_signals: dict[str, dict[str, Any]],
    exchange: str,
    study_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    lag_window = int(study_cfg["lag_window_weeks"])
    min_lag_corr = float(study_cfg["min_lag_correlation"])
    catch_up_gap = float(study_cfg["catch_up_gap_pct"])
    group_strength_min_8w = float(study_cfg["group_strength_min_8w"])

    for index, symbols in enumerate(groups, start=1):
        if not symbols:
            continue
        group_id = f"G{index:03d}"
        group_returns = returns[symbols]
        group_close = close_matrix[symbols]
        group_mean_returns = group_returns.mean(axis=1, skipna=True)
        group_r4 = _period_return_from_prices(group_close, 4).median(skipna=True)
        group_r8 = _period_return_from_prices(group_close, 8).median(skipna=True)
        group_r13 = _period_return_from_prices(group_close, 13).median(skipna=True)
        group_r26 = _period_return_from_prices(group_close, 26).median(skipna=True)
        pair_corr = _upper_triangle_mean(correlation.loc[symbols, symbols])

        member_metrics: list[dict[str, Any]] = []
        recent_8w_values: list[float] = []
        for symbol in symbols:
            prices = close_matrix[symbol].dropna()
            recent_4w = _latest_period_return(prices, 4)
            recent_8w = _latest_period_return(prices, 8)
            recent_13w = _latest_period_return(prices, 13)
            recent_26w = _latest_period_return(prices, 26)
            recent_52w = _latest_period_return(prices, 52)
            group_corr = group_returns[symbol].corr(group_mean_returns)
            lag_weeks, lag_corr = _best_lag_correlation(
                _rolling_return(prices, 4),
                _rolling_return(group_close.mean(axis=1, skipna=True), 4),
                lag_window,
            )
            recent_8w_values.append(recent_8w if pd.notna(recent_8w) else np.nan)
            member_metrics.append(
                {
                    "symbol": symbol,
                    "name": symbol_names.get(symbol, symbol),
                    "correlation_to_group": group_corr,
                    "best_lag_weeks": lag_weeks,
                    "best_lag_correlation": lag_corr,
                    "recent_return_4w_pct": recent_4w,
                    "recent_return_8w_pct": recent_8w,
                    "recent_return_13w_pct": recent_13w,
                    "recent_return_26w_pct": recent_26w,
                    "recent_return_52w_pct": recent_52w,
                    "latest_close": prices.iloc[-1] if not prices.empty else pd.NA,
                }
            )

        valid_recent_8w = pd.Series(recent_8w_values, dtype="float64").dropna()
        if valid_recent_8w.empty:
            top_quartile_8w = pd.NA
            group_median_8w = pd.NA
        else:
            top_quartile_8w = float(valid_recent_8w.quantile(0.75))
            group_median_8w = float(valid_recent_8w.median())

        leaders = 0
        catch_up_candidates = 0
        latest_weekly_buy_count = 0
        latest_weekly_sell_count = 0
        for metric in member_metrics:
            gap_8w = pd.NA
            if pd.notna(group_median_8w) and pd.notna(metric["recent_return_8w_pct"]):
                gap_8w = float(group_median_8w - metric["recent_return_8w_pct"])

            status = "In Sync"
            is_leader = False
            is_candidate = False
            if pd.notna(metric["recent_return_8w_pct"]) and pd.notna(top_quartile_8w) and metric["recent_return_8w_pct"] >= top_quartile_8w:
                status = "Leader"
                is_leader = True
            elif (
                pd.notna(group_r8)
                and group_r8 >= group_strength_min_8w
                and pd.notna(gap_8w)
                and gap_8w >= catch_up_gap
                and pd.notna(metric["best_lag_correlation"])
                and metric["best_lag_correlation"] >= min_lag_corr
                and int(metric["best_lag_weeks"]) >= 1
            ):
                status = "Catch-up Candidate"
                is_candidate = True
            elif pd.notna(group_median_8w) and pd.notna(metric["recent_return_8w_pct"]) and metric["recent_return_8w_pct"] < group_median_8w:
                status = "Lagging"

            leaders += int(is_leader)
            catch_up_candidates += int(is_candidate)
            signal_context = latest_signals.get(metric["symbol"], {})
            latest_signal = signal_context.get("latest_week_signal", "NONE")
            latest_signal_is_fresh = bool(signal_context.get("latest_week_signal_is_fresh", False))
            latest_weekly_buy_count += int(latest_signal == "BUY" and latest_signal_is_fresh)
            latest_weekly_sell_count += int(latest_signal == "SELL" and latest_signal_is_fresh)
            candidate_score = pd.NA
            if is_candidate and pd.notna(gap_8w):
                candidate_score = float(gap_8w + (metric["best_lag_weeks"] * 2.0) + (metric["best_lag_correlation"] * 10.0))

            member_rows.append(
                {
                    "group_id": group_id,
                    "exchange": exchange,
                    "symbol": metric["symbol"],
                    "name": metric["name"],
                    "group_size": len(symbols),
                    "correlation_to_group": metric["correlation_to_group"],
                    "best_lag_weeks": metric["best_lag_weeks"],
                    "best_lag_correlation": metric["best_lag_correlation"],
                    "latest_close": metric["latest_close"],
                    "recent_return_4w_pct": metric["recent_return_4w_pct"],
                    "recent_return_8w_pct": metric["recent_return_8w_pct"],
                    "recent_return_13w_pct": metric["recent_return_13w_pct"],
                    "recent_return_26w_pct": metric["recent_return_26w_pct"],
                    "recent_return_52w_pct": metric["recent_return_52w_pct"],
                    "group_median_return_8w_pct": group_median_8w,
                    "group_return_8w_pct": group_r8,
                    "catch_up_gap_8w_pct": gap_8w,
                    "movement_status": status,
                    "is_leader": is_leader,
                    "is_catch_up_candidate": is_candidate,
                    "candidate_score": candidate_score,
                    "latest_week_signal": latest_signal,
                    "latest_week_signal_date": signal_context.get("latest_week_signal_date", pd.NA),
                    "latest_week_signal_is_fresh": latest_signal_is_fresh,
                }
            )

        group_rows.append(
            {
                "group_id": group_id,
                "exchange": exchange,
                "group_size": len(symbols),
                "avg_pair_correlation": pair_corr,
                "group_return_4w_pct": group_r4,
                "group_return_8w_pct": group_r8,
                "group_return_13w_pct": group_r13,
                "group_return_26w_pct": group_r26,
                "leaders_count": leaders,
                "catch_up_candidates_count": catch_up_candidates,
                "latest_weekly_buy_count": latest_weekly_buy_count,
                "latest_weekly_sell_count": latest_weekly_sell_count,
                "symbols": ", ".join(symbols),
            }
        )

    groups_frame = pd.DataFrame(group_rows)
    members_frame = pd.DataFrame(member_rows)
    if not groups_frame.empty:
        groups_frame = groups_frame.sort_values(
            ["catch_up_candidates_count", "group_size", "avg_pair_correlation", "group_return_8w_pct"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
    if not members_frame.empty:
        members_frame = members_frame.sort_values(
            ["is_catch_up_candidate", "candidate_score", "is_leader", "group_id", "recent_return_8w_pct"],
            ascending=[False, False, False, True, False],
        ).reset_index(drop=True)
    return groups_frame, members_frame


def _build_candidates_frame(members_frame: pd.DataFrame) -> pd.DataFrame:
    if members_frame.empty:
        return pd.DataFrame()
    candidates = members_frame[members_frame["is_catch_up_candidate"] == True].copy()
    if candidates.empty:
        return candidates
    return candidates.sort_values(
        ["candidate_score", "best_lag_correlation", "catch_up_gap_8w_pct"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _build_groups_from_correlation(
    correlation: pd.DataFrame,
    min_correlation: float,
    min_group_size: int,
    target_group_size: int,
    max_group_size: int,
) -> list[list[str]]:
    symbols = [str(symbol) for symbol in correlation.columns]
    if not symbols:
        return []

    min_group_size = max(int(min_group_size), 2)
    target_group_size = max(int(target_group_size), min_group_size)
    max_group_size = max(int(max_group_size), target_group_size)

    corr = correlation.copy()
    corr.index = corr.index.map(str)
    corr.columns = corr.columns.map(str)
    for symbol in symbols:
        corr.loc[symbol, symbol] = 1.0

    avg_corr = corr.where(~np.eye(len(corr), dtype=bool)).mean(axis=1, skipna=True).fillna(-1.0)
    remaining: set[str] = set(symbols)
    rejected_seeds: set[str] = set()
    groups: list[list[str]] = []

    while remaining - rejected_seeds:
        seed_pool = remaining - rejected_seeds
        seed = max(seed_pool, key=lambda symbol: (float(avg_corr.get(symbol, -1.0)), symbol))
        remaining.remove(seed)
        group = [seed]

        candidate_pool = [
            candidate
            for candidate in remaining
            if pd.notna(corr.loc[seed, candidate]) and float(corr.loc[seed, candidate]) >= min_correlation
        ]
        candidate_pool.sort(
            key=lambda candidate: (
                float(corr.loc[seed, candidate]),
                float(avg_corr.get(candidate, -1.0)),
                candidate,
            ),
            reverse=True,
        )

        for candidate in candidate_pool:
            if candidate not in remaining:
                continue
            if len(group) >= max_group_size:
                break
            pair_scores = pd.to_numeric(pd.Series([corr.loc[candidate, member] for member in group]), errors="coerce").dropna()
            if pair_scores.empty:
                continue
            avg_to_group = float(pair_scores.mean())
            min_to_group = float(pair_scores.min())
            if avg_to_group < min_correlation or min_to_group < (min_correlation - 0.08):
                continue
            group.append(candidate)
            remaining.remove(candidate)
            if len(group) >= target_group_size:
                continue

        if len(group) >= min_group_size:
            groups.append(sorted(group))
        else:
            rejected_seeds.add(seed)
            for symbol in group[1:]:
                remaining.add(symbol)

    groups.sort(key=lambda group: (-len(group), group[0]))
    return groups


def _period_return_from_prices(frame: pd.DataFrame, periods: int) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="float64")
    return ((frame.iloc[-1] / frame.shift(periods).iloc[-1]) - 1.0) * 100.0


def _latest_period_return(series: pd.Series, periods: int) -> float | pd.NA:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) <= periods:
        return pd.NA
    earlier = float(series.iloc[-(periods + 1)])
    latest = float(series.iloc[-1])
    if earlier == 0:
        return pd.NA
    return ((latest / earlier) - 1.0) * 100.0


def _rolling_return(series: pd.Series, periods: int) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.pct_change(periods) * 100.0


def _best_lag_correlation(stock_signal: pd.Series, group_signal: pd.Series, max_lag_weeks: int) -> tuple[int, float | pd.NA]:
    best_lag = 0
    best_corr = pd.NA
    for lag in range(0, max_lag_weeks + 1):
        shifted_group = group_signal.shift(lag)
        corr = stock_signal.corr(shifted_group)
        if pd.isna(corr):
            continue
        if pd.isna(best_corr) or float(corr) > float(best_corr):
            best_corr = float(corr)
            best_lag = lag
    return best_lag, best_corr


def _upper_triangle_mean(frame: pd.DataFrame) -> float | pd.NA:
    if frame.empty or len(frame) < 2:
        return pd.NA
    values = frame.to_numpy(dtype="float64", copy=True)
    upper = values[np.triu_indices_from(values, k=1)]
    upper = upper[~np.isnan(upper)]
    if upper.size == 0:
        return pd.NA
    return float(upper.mean())


def _build_summary(
    exchange: str,
    universe_rows: int,
    symbols_processed: int,
    symbols_with_history: int,
    eligible_symbols: int,
    groups_frame: pd.DataFrame,
    members_frame: pd.DataFrame,
    candidates_frame: pd.DataFrame,
) -> dict[str, Any]:
    grouped_symbols = int(members_frame["symbol"].nunique()) if not members_frame.empty else 0
    largest_group = int(groups_frame["group_size"].max()) if not groups_frame.empty else 0
    avg_group_corr = pd.to_numeric(groups_frame.get("avg_pair_correlation", pd.Series(dtype="float64")), errors="coerce").mean()
    return {
        "exchange": exchange,
        "universe_rows": universe_rows,
        "symbols_processed": symbols_processed,
        "symbols_with_history": symbols_with_history,
        "eligible_symbols": eligible_symbols,
        "groups_found": len(groups_frame),
        "grouped_symbols": grouped_symbols,
        "ungrouped_symbols": max(int(eligible_symbols) - grouped_symbols, 0),
        "largest_group_size": largest_group,
        "avg_group_correlation": float(avg_group_corr) if pd.notna(avg_group_corr) else 0.0,
        "catch_up_candidates": len(candidates_frame),
    }


def _latest_week_signal_context(strategy_output: pd.DataFrame, weekly: pd.DataFrame) -> dict[str, Any]:
    if weekly.empty:
        return {"latest_week_signal": "NONE", "latest_week_signal_date": pd.NA, "latest_week_signal_is_fresh": False}
    latest_bar_date = pd.to_datetime(weekly["date"], errors="coerce").max()
    if strategy_output.empty or "signal" not in strategy_output.columns:
        return {"latest_week_signal": "NONE", "latest_week_signal_date": pd.NA, "latest_week_signal_is_fresh": False}
    signals = strategy_output[strategy_output["signal"].isin(["BUY", "SELL"])].copy()
    if signals.empty:
        return {"latest_week_signal": "NONE", "latest_week_signal_date": pd.NA, "latest_week_signal_is_fresh": False}
    signals["date"] = pd.to_datetime(signals["date"], errors="coerce")
    latest = signals.sort_values("date").iloc[-1]
    latest_signal_date = pd.to_datetime(latest["date"], errors="coerce")
    return {
        "latest_week_signal": str(latest["signal"]),
        "latest_week_signal_date": latest_signal_date,
        "latest_week_signal_is_fresh": bool(pd.notna(latest_signal_date) and pd.notna(latest_bar_date) and latest_signal_date == latest_bar_date),
    }


def _kite_instruments_universe(storage: Storage, config: dict[str, Any], exchange: str) -> list[dict[str, str]]:
    instruments = storage.load_instruments()
    if instruments.empty:
        return []
    universe = build_universe(instruments, config)
    if universe.empty or "tradingsymbol" not in universe.columns:
        return []
    if "exchange" in universe.columns:
        universe = universe[universe["exchange"].astype(str).str.upper() == exchange.upper()]
    else:
        universe["exchange"] = exchange.upper()
    universe = universe.copy()
    universe["symbol"] = universe["tradingsymbol"].astype(str).str.upper().str.strip()
    universe["name"] = universe.get("name", universe["symbol"]).fillna("").astype(str).str.strip()
    universe["name"] = universe["name"].mask(universe["name"] == "", universe["symbol"])
    universe = universe[universe["symbol"] != ""].drop_duplicates(subset=["exchange", "symbol"], keep="last")
    universe = universe.sort_values(["exchange", "symbol"]).reset_index(drop=True)
    return universe[["exchange", "symbol", "name"]].to_dict(orient="records")


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _empty_summary(exchange: str) -> dict[str, Any]:
    return {
        "exchange": exchange,
        "universe_rows": 0,
        "symbols_processed": 0,
        "symbols_with_history": 0,
        "eligible_symbols": 0,
        "groups_found": 0,
        "grouped_symbols": 0,
        "ungrouped_symbols": 0,
        "largest_group_size": 0,
        "avg_group_correlation": 0.0,
        "catch_up_candidates": 0,
    }


def _emit_progress(
    callback: Callable[[dict[str, Any]], None] | None,
    phase: str,
    completed: int,
    total: int,
    current_symbol: str,
    current_exchange: str,
) -> None:
    if callback is None:
        return
    callback(
        {
            "phase": phase,
            "completed": completed,
            "total": total,
            "current_symbol": current_symbol,
            "current_exchange": current_exchange,
        }
    )
