from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from stock_screener.data.storage import Storage
from stock_screener.equity_runner_research import calculate_runner_features
from stock_screener.knox_envelope_pair_backtest import _candidate_symbols


STRATEGY_VERSION = "leader_pullback_rvol_v1"
MIN_AVERAGE_TRADED_VALUE = 50_000_000.0
MAX_MARKET_BREADTH_PCT = 40.0
MAX_SIGNAL_VOLUME_RATIO = 0.90
MAX_RSI2 = 25.0
MAX_DISTANCE_FROM_52W_HIGH_PCT = 25.0
TARGET_ATR_MULTIPLE = 3.0
STOP_ATR_MULTIPLE = 1.75
MIN_TARGET_PCT = 5.0
MAX_TARGET_PCT = 25.0
MIN_STOP_PCT = 2.0
MAX_STOP_PCT = 10.0
MAX_HOLD_SESSIONS = 5
ROUND_TRIP_COST_PCT = 0.15

# Frozen after development/validation research. The ranks are calculated only
# among eligible candidates on the same date; lower RSI(2) ranks higher.
RANK_WEIGHTS: dict[str, float] = {
    "rsi2": 0.15017586712732622,
    "relative_strength20_pct": 0.020371756536157186,
    "sma50_slope10_pct": 0.10457554610828042,
    "sma200_slope20_pct": 0.10546011711783919,
    "distance_52w_high_pct": 0.06956489070144264,
    "close_location_pct": 0.043036465377158446,
    "average_traded_value20": 0.08932781517199725,
    "adx14": 0.05165007388518549,
    "relative_volume20": 0.013238675119871835,
    "return_63d_pct": 0.3525987928547414,
}


@dataclass(frozen=True)
class HighExpectancyResult:
    summary: dict[str, Any]
    candidates: pd.DataFrame
    shortlist: pd.DataFrame


def calculate_high_expectancy_features(candles: pd.DataFrame) -> pd.DataFrame:
    frame = calculate_runner_features(candles)
    if frame.empty:
        return frame

    close = frame["close"]
    high = frame["high"]
    delta = close.diff()
    average_gain2 = delta.clip(lower=0.0).ewm(
        alpha=0.5, adjust=False, min_periods=2
    ).mean()
    average_loss2 = (-delta.clip(upper=0.0)).ewm(
        alpha=0.5, adjust=False, min_periods=2
    ).mean()
    relative_strength2 = average_gain2 / average_loss2.replace(0.0, np.nan)
    rsi2 = 100.0 - (100.0 / (1.0 + relative_strength2))
    rsi2 = rsi2.mask(average_loss2 == 0.0, 100.0)
    rsi2 = rsi2.mask((average_gain2 == 0.0) & (average_loss2 == 0.0), 50.0)

    sma20 = close.rolling(20, min_periods=20).mean()
    sma50 = close.rolling(50, min_periods=50).mean()
    sma200 = close.rolling(200, min_periods=200).mean()
    prior_high252 = high.shift(1).rolling(252, min_periods=100).max()

    frame["rsi2"] = rsi2
    frame["return_63d_pct"] = (close / close.shift(63) - 1.0) * 100.0
    frame["distance_sma200_pct"] = (close / sma200 - 1.0) * 100.0
    frame["distance_52w_high_pct"] = (close / prior_high252 - 1.0) * 100.0
    frame["sma20_slope5_pct"] = (sma20 / sma20.shift(5) - 1.0) * 100.0
    frame["sma50_slope10_pct"] = (sma50 / sma50.shift(10) - 1.0) * 100.0
    frame["sma200_slope20_pct"] = (sma200 / sma200.shift(20) - 1.0) * 100.0
    return frame


def build_latest_snapshot(
    storage: Storage,
    *,
    exchange: str = "NSE",
    symbols: Iterable[str] | None = None,
    as_of_date: str | pd.Timestamp | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> pd.DataFrame:
    selected_symbols = list(symbols) if symbols is not None else _candidate_symbols(
        storage, exchange, None
    )
    cutoff = _normalize_date(as_of_date)
    rows: list[dict[str, Any]] = []

    for completed, symbol in enumerate(selected_symbols, start=1):
        if progress_callback:
            progress_callback(
                {
                    "phase": "Scoring high-expectancy candidates",
                    "completed": completed - 1,
                    "total": len(selected_symbols),
                    "current_symbol": symbol,
                    "current_exchange": exchange,
                }
            )
        candles = storage.load_candles(exchange, symbol, "1D")
        if candles.empty:
            continue
        if cutoff is not None:
            dates = pd.to_datetime(candles.get("date"), errors="coerce", format="mixed")
            candles = candles.loc[dates.dt.normalize() <= cutoff].copy()
        features = calculate_high_expectancy_features(candles)
        if len(features) < 220:
            continue
        latest = features.iloc[-1]
        row = latest.to_dict()
        row["symbol"] = symbol
        rows.append(row)

    if progress_callback:
        progress_callback(
            {
                "phase": "Scoring high-expectancy candidates",
                "completed": len(selected_symbols),
                "total": len(selected_symbols),
                "current_symbol": "",
                "current_exchange": exchange,
            }
        )
    snapshot = pd.DataFrame(rows)
    if snapshot.empty:
        return snapshot
    snapshot["date"] = pd.to_datetime(snapshot["date"], errors="coerce", format="mixed").dt.normalize()
    benchmark = calculate_runner_features(storage.load_candles("NSE_INDEX", "NIFTY 50", "1D"))
    if not benchmark.empty:
        if cutoff is not None:
            benchmark = benchmark.loc[benchmark["date"].dt.normalize() <= cutoff].copy()
        benchmark_returns = benchmark[["date", "return_20d_pct"]].copy()
        benchmark_returns["date"] = benchmark_returns["date"].dt.normalize()
        benchmark_returns.rename(columns={"return_20d_pct": "nifty_return20_pct"}, inplace=True)
        snapshot = snapshot.merge(benchmark_returns, on="date", how="left", validate="many_to_one")
        snapshot["relative_strength20_pct"] = (
            pd.to_numeric(snapshot["return_20d_pct"], errors="coerce")
            - pd.to_numeric(snapshot["nifty_return20_pct"], errors="coerce")
        )
    else:
        snapshot["nifty_return20_pct"] = np.nan
        snapshot["relative_strength20_pct"] = np.nan
    return snapshot


def score_snapshot(snapshot: pd.DataFrame) -> HighExpectancyResult:
    if snapshot.empty:
        return HighExpectancyResult(_empty_summary(), pd.DataFrame(), pd.DataFrame())

    frame = snapshot.copy()
    latest_date = pd.to_datetime(frame["date"], errors="coerce").max().normalize()
    frame = frame.loc[pd.to_datetime(frame["date"], errors="coerce").dt.normalize() == latest_date].copy()
    for column in _required_numeric_columns():
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")

    breadth_eligible = frame["distance_sma20_pct"].notna()
    market_breadth = (
        float((frame.loc[breadth_eligible, "distance_sma20_pct"] > 0.0).mean() * 100.0)
        if breadth_eligible.any()
        else np.nan
    )
    required = list(RANK_WEIGHTS) + [
        "distance_sma200_pct",
        "data_quality_pass",
        "atr14_pct",
    ]
    ready = frame[required].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    eligible = (
        ready
        & frame["data_quality_pass"].fillna(False).astype(bool)
        & (frame["average_traded_value20"] >= MIN_AVERAGE_TRADED_VALUE)
        & (frame["rsi2"] <= MAX_RSI2)
        & (frame["distance_sma200_pct"] > 0.0)
        & (frame["sma200_slope20_pct"] > 0.0)
        & (frame["distance_52w_high_pct"] >= -MAX_DISTANCE_FROM_52W_HIGH_PCT)
    )
    candidates = frame.loc[eligible].copy()
    if not candidates.empty:
        candidates["rank_score"] = 0.0
        for feature, weight in RANK_WEIGHTS.items():
            ascending = feature != "rsi2"
            rank = candidates[feature].rank(pct=True, ascending=ascending, method="average")
            candidates[f"rank_{feature}"] = rank
            candidates["rank_score"] += rank * weight
        candidates = candidates.sort_values(
            ["rank_score", "relative_strength20_pct", "symbol"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        candidates["daily_rank"] = np.arange(1, len(candidates) + 1)
        candidates["market_breadth_pct"] = market_breadth
        candidates["breadth_pass"] = market_breadth <= MAX_MARKET_BREADTH_PCT
        candidates["low_volume_pullback_pass"] = (
            candidates["relative_volume20"] <= MAX_SIGNAL_VOLUME_RATIO
        )
        candidates["shortlist_pass"] = (
            (candidates["daily_rank"] == 1)
            & candidates["breadth_pass"]
            & candidates["low_volume_pullback_pass"]
        )
        candidates["target_pct"] = np.clip(
            candidates["atr14_pct"] * TARGET_ATR_MULTIPLE,
            MIN_TARGET_PCT,
            MAX_TARGET_PCT,
        )
        candidates["stop_pct"] = np.clip(
            candidates["atr14_pct"] * STOP_ATR_MULTIPLE,
            MIN_STOP_PCT,
            MAX_STOP_PCT,
        )
        candidates["reference_target_from_close"] = (
            candidates["close"] * (1.0 + candidates["target_pct"] / 100.0)
        )
        candidates["reference_stop_from_close"] = (
            candidates["close"] * (1.0 - candidates["stop_pct"] / 100.0)
        )
        candidates["max_hold_sessions"] = MAX_HOLD_SESSIONS
        candidates["entry_timing"] = "NEXT_SESSION_OPEN"
        candidates["stop_timing"] = "EXIT_AT_CLOSE"

    shortlist = (
        candidates.loc[candidates["shortlist_pass"]].copy()
        if "shortlist_pass" in candidates.columns
        else candidates.iloc[0:0].copy()
    )
    summary = {
        "strategy_version": STRATEGY_VERSION,
        "latest_data_date": latest_date.date().isoformat(),
        "symbols_on_latest_date": int(len(frame)),
        "breadth_eligible_symbols": int(breadth_eligible.sum()),
        "market_breadth_pct": market_breadth,
        "market_breadth_limit_pct": MAX_MARKET_BREADTH_PCT,
        "market_regime_pass": bool(np.isfinite(market_breadth) and market_breadth <= MAX_MARKET_BREADTH_PCT),
        "eligible_leader_pullbacks": int(len(candidates)),
        "shortlist_count": int(len(shortlist)),
        "max_signal_volume_ratio": MAX_SIGNAL_VOLUME_RATIO,
        "target_atr_multiple": TARGET_ATR_MULTIPLE,
        "stop_atr_multiple": STOP_ATR_MULTIPLE,
        "max_hold_sessions": MAX_HOLD_SESSIONS,
        "round_trip_cost_pct": ROUND_TRIP_COST_PCT,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    return HighExpectancyResult(summary, candidates, shortlist)


def run_high_expectancy_study(
    storage: Storage,
    *,
    exchange: str = "NSE",
    symbols: Iterable[str] | None = None,
    as_of_date: str | pd.Timestamp | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> HighExpectancyResult:
    snapshot = build_latest_snapshot(
        storage,
        exchange=exchange,
        symbols=symbols,
        as_of_date=as_of_date,
        progress_callback=progress_callback,
    )
    return score_snapshot(snapshot)


def simulate_adaptive_close_stop_trade(
    candles: pd.DataFrame,
    signal_index: int,
    atr_pct: float,
    *,
    target_atr_multiple: float = TARGET_ATR_MULTIPLE,
    stop_atr_multiple: float = STOP_ATR_MULTIPLE,
    max_hold_sessions: int = MAX_HOLD_SESSIONS,
    round_trip_cost_pct: float = ROUND_TRIP_COST_PCT,
) -> dict[str, Any] | None:
    """Simulate a closed trade; incomplete recent positions return ``None``."""
    frame = candles.reset_index(drop=True)
    entry_index = int(signal_index) + 1
    full_exit_index = int(signal_index) + int(max_hold_sessions)
    if entry_index >= len(frame):
        return None

    entry_price = float(frame.at[entry_index, "open"])
    if not np.isfinite(entry_price) or entry_price <= 0.0 or not np.isfinite(atr_pct):
        return None
    target_pct = float(
        np.clip(atr_pct * target_atr_multiple, MIN_TARGET_PCT, MAX_TARGET_PCT)
    )
    stop_pct = float(np.clip(atr_pct * stop_atr_multiple, MIN_STOP_PCT, MAX_STOP_PCT))
    target_price = entry_price * (1.0 + target_pct / 100.0)
    stop_close = entry_price * (1.0 - stop_pct / 100.0)
    available_exit_index = min(full_exit_index, len(frame) - 1)

    for bar_index in range(entry_index, available_exit_index + 1):
        opening = float(frame.at[bar_index, "open"])
        high = float(frame.at[bar_index, "high"])
        close = float(frame.at[bar_index, "close"])
        if opening >= target_price:
            exit_price, reason = opening, "GAP_TARGET"
        elif high >= target_price:
            exit_price, reason = target_price, "TARGET"
        elif close <= stop_close:
            exit_price, reason = close, "EOD_STOP"
        elif bar_index == full_exit_index:
            exit_price, reason = close, "MAX_HOLD"
        else:
            continue
        return {
            "entry_index": entry_index,
            "exit_index": bar_index,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "return_pct": (exit_price / entry_price - 1.0) * 100.0 - round_trip_cost_pct,
            "exit_reason": reason,
            "target_pct": target_pct,
            "stop_pct": stop_pct,
        }

    # A recent trade that has not hit target/stop and lacks the full future window
    # is still open. Excluding it prevents a false time exit at the data boundary.
    return None


def save_high_expectancy_outputs(result: HighExpectancyResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result.candidates.to_csv(output_dir / "candidates.csv", index=False)
    result.shortlist.to_csv(output_dir / "shortlist.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(result.summary, indent=2, default=str), encoding="utf-8"
    )


def load_high_expectancy_outputs(output_dir: Path) -> HighExpectancyResult:
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else _empty_summary()
    candidates_path = output_dir / "candidates.csv"
    shortlist_path = output_dir / "shortlist.csv"
    candidates = pd.read_csv(candidates_path) if candidates_path.exists() else pd.DataFrame()
    shortlist = pd.read_csv(shortlist_path) if shortlist_path.exists() else pd.DataFrame()
    return HighExpectancyResult(summary, candidates, shortlist)


def _required_numeric_columns() -> list[str]:
    return sorted(
        set(RANK_WEIGHTS)
        | {
            "distance_sma20_pct",
            "distance_sma200_pct",
            "atr14_pct",
            "average_traded_value20",
            "rsi2",
        }
    )


def _normalize_date(value: str | pd.Timestamp | None) -> pd.Timestamp | None:
    if value is None or str(value).strip() == "":
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid as-of date: {value}")
    return pd.Timestamp(parsed).normalize()


def _empty_summary() -> dict[str, Any]:
    return {
        "strategy_version": STRATEGY_VERSION,
        "symbols_on_latest_date": 0,
        "breadth_eligible_symbols": 0,
        "eligible_leader_pullbacks": 0,
        "shortlist_count": 0,
    }
