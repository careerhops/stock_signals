from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from stock_screener.data.storage import Storage


DEFAULT_START_DATE = "2026-04-25"


@dataclass(frozen=True)
class StrategyLabResult:
    summary: dict[str, Any]
    strategy_stats: pd.DataFrame
    trade_details: pd.DataFrame
    signal_universe: pd.DataFrame
    next_week_candidates: pd.DataFrame


def run_strategy_lab_study(
    config: dict[str, Any],
    storage: Storage,
    exchange: str = "NSE",
    start_date: str = DEFAULT_START_DATE,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> StrategyLabResult:
    signal_universe = _load_signal_universe(storage, exchange, start_date)
    if signal_universe.empty:
        return StrategyLabResult(_empty_summary(exchange, start_date), _empty_strategy_stats(), _empty_trade_details(), signal_universe, pd.DataFrame())

    trade_rows: list[dict[str, Any]] = []
    signal_records = signal_universe.to_dict(orient="records")
    grouped = signal_universe.groupby(["exchange", "symbol"], dropna=False)

    processed = 0
    _emit_progress(progress_callback, phase="Preparing signal cohorts", completed=0, total=len(grouped), current_symbol="", current_exchange=exchange)

    for (row_exchange, symbol), symbol_signals in grouped:
        daily = storage.load_candles(str(row_exchange), str(symbol), "1D")
        if daily.empty:
            processed += 1
            _emit_progress(progress_callback, phase="Preparing signal cohorts", completed=processed, total=len(grouped), current_symbol=str(symbol), current_exchange=str(row_exchange))
            continue

        frame = _prepare_daily_with_indicators(daily)
        if frame.empty:
            processed += 1
            _emit_progress(progress_callback, phase="Preparing signal cohorts", completed=processed, total=len(grouped), current_symbol=str(symbol), current_exchange=str(row_exchange))
            continue

        events = _build_symbol_events(symbol_signals)
        for event in events:
            trade_rows.extend(_simulate_event_strategies(frame, event))

        processed += 1
        _emit_progress(progress_callback, phase="Testing strategy set", completed=processed, total=len(grouped), current_symbol=str(symbol), current_exchange=str(row_exchange))

    trade_details = pd.DataFrame(trade_rows, columns=_trade_columns())
    strategy_stats = _build_strategy_stats(trade_details, signal_universe)
    summary = _build_summary(strategy_stats, signal_universe, exchange, start_date)
    next_week_candidates = _build_next_week_candidates(storage, exchange)
    return StrategyLabResult(summary, strategy_stats, trade_details, signal_universe, next_week_candidates)


def save_strategy_lab_outputs(result: StrategyLabResult, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "latest_summary.csv"
    strategy_stats_path = output_dir / "latest_strategy_stats.csv"
    trade_details_path = output_dir / "latest_trade_details.csv"
    signal_universe_path = output_dir / "latest_signal_universe.csv"
    next_week_candidates_path = output_dir / "latest_next_week_candidates.csv"
    pd.DataFrame([result.summary]).to_csv(summary_path, index=False)
    result.strategy_stats.to_csv(strategy_stats_path, index=False)
    result.trade_details.to_csv(trade_details_path, index=False)
    result.signal_universe.to_csv(signal_universe_path, index=False)
    result.next_week_candidates.to_csv(next_week_candidates_path, index=False)
    return {
        "summary": summary_path,
        "strategy_stats": strategy_stats_path,
        "trade_details": trade_details_path,
        "signal_universe": signal_universe_path,
        "next_week_candidates": next_week_candidates_path,
    }


def load_strategy_lab_outputs(output_dir: Path) -> StrategyLabResult:
    def _read(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    summary_path = output_dir / "latest_summary.csv"
    summary: dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary_frame = pd.read_csv(summary_path)
            if not summary_frame.empty:
                summary = summary_frame.iloc[0].to_dict()
        except pd.errors.EmptyDataError:
            summary = {}

    return StrategyLabResult(
        summary=summary,
        strategy_stats=_read(output_dir / "latest_strategy_stats.csv"),
        trade_details=_read(output_dir / "latest_trade_details.csv"),
        signal_universe=_read(output_dir / "latest_signal_universe.csv"),
        next_week_candidates=_read(output_dir / "latest_next_week_candidates.csv"),
    )


def _load_signal_universe(storage: Storage, exchange: str, start_date: str) -> pd.DataFrame:
    raw = storage.load_signals("latest_raw_signals.csv")
    if raw.empty:
        return pd.DataFrame()
    frame = raw.copy()
    if "date" not in frame.columns or "signal" not in frame.columns:
        return pd.DataFrame()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame[frame["date"].notna()].copy()
    frame = frame[frame["date"] >= pd.Timestamp(start_date)]
    frame = frame[frame["signal"].astype(str).str.upper().isin({"BUY", "SELL"})].copy()
    frame["exchange"] = frame.get("exchange", exchange).astype(str).str.upper()
    frame = frame[frame["exchange"] == exchange.upper()]
    symbol_column = "symbol" if "symbol" in frame.columns else "tradingsymbol" if "tradingsymbol" in frame.columns else ""
    if not symbol_column:
        return pd.DataFrame()
    frame["symbol"] = frame[symbol_column].astype(str).str.upper().str.strip()
    frame["name"] = frame.get("name", frame["symbol"]).fillna("").astype(str).str.strip().mask(lambda s: s == "", frame["symbol"])
    frame = frame[frame["symbol"] != ""]
    return frame[["exchange", "symbol", "name", "date", "signal"]].drop_duplicates(subset=["exchange", "symbol", "date", "signal"], keep="last").sort_values(["exchange", "symbol", "date"]).reset_index(drop=True)


def _latest_signal_cohort(storage: Storage, exchange: str) -> pd.DataFrame:
    raw = storage.load_signals("latest_raw_signals.csv")
    if raw.empty or "date" not in raw.columns or "signal" not in raw.columns:
        return pd.DataFrame()
    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame[frame["date"].notna()].copy()
    latest_date = frame["date"].max()
    if pd.isna(latest_date):
        return pd.DataFrame()
    frame = frame[frame["date"] == latest_date].copy()
    frame = frame[frame["signal"].astype(str).str.upper().isin({"BUY", "SELL"})]
    frame["exchange"] = frame.get("exchange", exchange).astype(str).str.upper()
    frame = frame[frame["exchange"] == exchange.upper()]
    symbol_column = "symbol" if "symbol" in frame.columns else "tradingsymbol" if "tradingsymbol" in frame.columns else ""
    if not symbol_column:
        return pd.DataFrame()
    frame["symbol"] = frame[symbol_column].astype(str).str.upper().str.strip()
    frame["name"] = frame.get("name", frame["symbol"]).fillna("").astype(str).str.strip().mask(lambda s: s == "", frame["symbol"])
    frame = frame[frame["symbol"] != ""]
    return frame[["exchange", "symbol", "name", "date", "signal"]].drop_duplicates(subset=["exchange", "symbol"], keep="last").sort_values(["signal", "symbol"]).reset_index(drop=True)


def _prepare_daily_with_indicators(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame[frame["date"].notna()].copy()
    if frame.empty:
        return pd.DataFrame()
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "open", "high", "low", "close", "volume"]).sort_values("date").reset_index(drop=True)
    if len(frame) < 30:
        return pd.DataFrame()

    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    volume = frame["volume"]

    frame["ema10"] = close.ewm(span=10, adjust=False).mean()
    frame["ema20"] = close.ewm(span=20, adjust=False).mean()
    frame["ema50"] = close.ewm(span=50, adjust=False).mean()
    frame["avg_volume20"] = volume.rolling(20, min_periods=5).mean()

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    frame["rsi14"] = 100 - (100 / (1 + rs))
    frame["rsi14"] = frame["rsi14"].fillna(50.0)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    frame["macd_hist"] = macd - signal

    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["atr14"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    frame["atr14"] = frame["atr14"].fillna((high - low).rolling(5, min_periods=1).mean())
    return frame


def _build_symbol_events(symbol_signals: pd.DataFrame) -> list[dict[str, Any]]:
    frame = symbol_signals.copy().sort_values("date").reset_index(drop=True)
    events: list[dict[str, Any]] = []
    for idx, row in frame.iterrows():
        next_signal_date = frame.iloc[idx + 1]["date"] if idx + 1 < len(frame) else pd.NaT
        events.append(
            {
                "exchange": row["exchange"],
                "symbol": row["symbol"],
                "name": row["name"],
                "signal_date": pd.Timestamp(row["date"]),
                "signal": str(row["signal"]).upper(),
                "direction": 1 if str(row["signal"]).upper() == "BUY" else -1,
                "next_signal_date": pd.Timestamp(next_signal_date) if pd.notna(next_signal_date) else pd.NaT,
            }
        )
    return events


def _simulate_event_strategies(frame: pd.DataFrame, event: dict[str, Any]) -> list[dict[str, Any]]:
    entry_idx_candidates = frame.index[frame["date"] > event["signal_date"]].tolist()
    if not entry_idx_candidates:
        return []
    entry_idx = int(entry_idx_candidates[0])
    entry_bar = frame.iloc[entry_idx]
    if pd.isna(entry_bar.get("ema20")) or pd.isna(entry_bar.get("atr14")):
        return []

    cap_idx = len(frame) - 1
    if pd.notna(event["next_signal_date"]):
        next_idx_candidates = frame.index[frame["date"] > event["next_signal_date"]].tolist()
        if next_idx_candidates:
            cap_idx = min(cap_idx, int(next_idx_candidates[0]))
    if cap_idx <= entry_idx:
        return []

    results: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        trade = _simulate_strategy(frame, event, strategy, entry_idx, cap_idx)
        if trade is not None:
            results.append(trade)
    return results


def _simulate_strategy(
    frame: pd.DataFrame,
    event: dict[str, Any],
    strategy: dict[str, Any],
    entry_idx: int,
    cap_idx: int,
) -> dict[str, Any] | None:
    direction = int(event["direction"])
    entry_bar = frame.iloc[entry_idx]
    if not strategy["entry_filter"](entry_bar, direction):
        return None

    exit_idx, exit_price, exit_reason = strategy["exit_rule"](frame, entry_idx, cap_idx, direction)
    if exit_idx is None or exit_price is None:
        return None

    entry_price = float(entry_bar["close"])
    return_pct = ((float(exit_price) - entry_price) / entry_price) * 100.0
    if direction < 0:
        return_pct *= -1.0

    hold_bars = int(exit_idx - entry_idx)
    hold_days = int((pd.Timestamp(frame.iloc[exit_idx]["date"]) - pd.Timestamp(entry_bar["date"])).days)
    entry_score = _alignment_score(entry_bar, direction)

    return {
        "strategy_name": strategy["name"],
        "strategy_family": strategy["family"],
        "exchange": event["exchange"],
        "symbol": event["symbol"],
        "name": event["name"],
        "signal": event["signal"],
        "signal_date": event["signal_date"],
        "entry_date": pd.Timestamp(entry_bar["date"]),
        "entry_price": entry_price,
        "exit_date": pd.Timestamp(frame.iloc[exit_idx]["date"]),
        "exit_price": float(exit_price),
        "exit_reason": exit_reason,
        "return_pct": return_pct,
        "holding_bars": hold_bars,
        "holding_days": hold_days,
        "win": return_pct > 0,
        "loss": return_pct < 0,
        "entry_rsi14": float(entry_bar["rsi14"]),
        "entry_atr14": float(entry_bar["atr14"]),
        "entry_ema20_gap_pct": ((entry_price - float(entry_bar["ema20"])) / float(entry_bar["ema20"])) * 100.0 if float(entry_bar["ema20"]) else 0.0,
        "entry_macd_hist": float(entry_bar["macd_hist"]),
        "entry_alignment_score": entry_score,
    }


def _build_next_week_candidates(storage: Storage, exchange: str) -> pd.DataFrame:
    cohort = _latest_signal_cohort(storage, exchange)
    if cohort.empty:
        return pd.DataFrame(
            columns=[
                "exchange",
                "symbol",
                "name",
                "signal",
                "signal_date",
                "latest_date",
                "close",
                "ema20",
                "ema50",
                "rsi14",
                "macd_hist",
                "volume",
                "avg_volume20",
                "vol_ratio",
                "ema_trend",
                "rsi_regime",
                "momentum_capture",
                "align_score",
                "strategy_hits",
                "bias_score",
            ]
        )

    rows: list[dict[str, Any]] = []
    for row in cohort.to_dict(orient="records"):
        daily = storage.load_candles(str(row["exchange"]), str(row["symbol"]), "1D")
        if daily.empty:
            continue
        frame = _prepare_daily_with_indicators(daily)
        if frame.empty:
            continue
        latest = frame.iloc[-1]
        direction = 1 if str(row["signal"]).upper() == "BUY" else -1
        ema_trend = (
            float(latest["close"]) > float(latest["ema20"]) > float(latest["ema50"])
            if direction > 0
            else float(latest["close"]) < float(latest["ema20"]) < float(latest["ema50"])
        )
        rsi_regime = float(latest["rsi14"]) >= 55 if direction > 0 else float(latest["rsi14"]) <= 45
        momentum_capture = bool(ema_trend and rsi_regime)
        avg_volume20 = float(latest["avg_volume20"]) if pd.notna(latest["avg_volume20"]) else 0.0
        vol_ratio = (float(latest["volume"]) / avg_volume20) if avg_volume20 > 0 else 0.0
        macd_hist = float(latest["macd_hist"])
        strategy_hits = int(ema_trend) + int(rsi_regime) + int(momentum_capture)
        bias_score = _alignment_score(latest, direction) + strategy_hits + int(vol_ratio >= 1.0) + int(macd_hist > 0 if direction > 0 else macd_hist < 0)
        rows.append(
            {
                "exchange": row["exchange"],
                "symbol": row["symbol"],
                "name": row["name"],
                "signal": row["signal"],
                "signal_date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                "latest_date": pd.Timestamp(latest["date"]).strftime("%Y-%m-%d"),
                "close": float(latest["close"]),
                "ema20": float(latest["ema20"]),
                "ema50": float(latest["ema50"]),
                "rsi14": float(latest["rsi14"]),
                "macd_hist": macd_hist,
                "volume": float(latest["volume"]),
                "avg_volume20": avg_volume20,
                "vol_ratio": vol_ratio,
                "ema_trend": bool(ema_trend),
                "rsi_regime": bool(rsi_regime),
                "momentum_capture": momentum_capture,
                "align_score": int(_alignment_score(latest, direction)),
                "strategy_hits": strategy_hits,
                "bias_score": bias_score,
            }
        )

    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return candidates

    buy = candidates[candidates["signal"].astype(str).str.upper() == "BUY"].copy()
    sell = candidates[candidates["signal"].astype(str).str.upper() == "SELL"].copy()
    if not buy.empty:
        buy = buy.sort_values(
            ["momentum_capture", "ema_trend", "rsi_regime", "bias_score", "vol_ratio", "rsi14"],
            ascending=[False, False, False, False, False, False],
        )
    if not sell.empty:
        sell = sell.sort_values(
            ["momentum_capture", "ema_trend", "rsi_regime", "bias_score", "vol_ratio", "rsi14"],
            ascending=[False, False, False, False, False, True],
        )
    return pd.concat([buy, sell], ignore_index=True)


def _alignment_score(row: pd.Series, direction: int) -> int:
    close = float(row["close"])
    ema20 = float(row["ema20"])
    ema50 = float(row["ema50"])
    rsi14 = float(row["rsi14"])
    macd_hist = float(row["macd_hist"])
    avg_volume20 = float(row["avg_volume20"]) if pd.notna(row["avg_volume20"]) else 0.0
    volume = float(row["volume"])

    score = 0
    if direction > 0:
        score += int(close > ema20 > ema50)
        score += int(rsi14 >= 55 and macd_hist > 0)
    else:
        score += int(close < ema20 < ema50)
        score += int(rsi14 <= 45 and macd_hist < 0)
    score += int(avg_volume20 > 0 and volume >= avg_volume20)
    return score


def _fixed_horizon_exit(horizon: int, family: str) -> Callable[[pd.DataFrame, int, int, int], tuple[int | None, float | None, str]]:
    def rule(frame: pd.DataFrame, entry_idx: int, cap_idx: int, direction: int) -> tuple[int | None, float | None, str]:
        exit_idx = min(cap_idx, entry_idx + horizon)
        if exit_idx <= entry_idx:
            return None, None, ""
        return exit_idx, float(frame.iloc[exit_idx]["close"]), f"{family}:fixed_{horizon}"

    return rule


def _ema_exit_rule(frame: pd.DataFrame, entry_idx: int, cap_idx: int, direction: int) -> tuple[int | None, float | None, str]:
    limit_idx = min(cap_idx, entry_idx + 10)
    for idx in range(entry_idx + 1, limit_idx + 1):
        row = frame.iloc[idx]
        if direction > 0 and float(row["close"]) < float(row["ema10"]):
            return idx, float(row["close"]), "ema10_break"
        if direction < 0 and float(row["close"]) > float(row["ema10"]):
            return idx, float(row["close"]), "ema10_break"
    if limit_idx <= entry_idx:
        return None, None, ""
    return limit_idx, float(frame.iloc[limit_idx]["close"]), "time_stop_10"


def _rsi_exit_rule(frame: pd.DataFrame, entry_idx: int, cap_idx: int, direction: int) -> tuple[int | None, float | None, str]:
    limit_idx = min(cap_idx, entry_idx + 10)
    for idx in range(entry_idx + 1, limit_idx + 1):
        row = frame.iloc[idx]
        if direction > 0 and float(row["rsi14"]) < 50:
            return idx, float(row["close"]), "rsi_lost_50"
        if direction < 0 and float(row["rsi14"]) > 50:
            return idx, float(row["close"]), "rsi_lost_50"
    if limit_idx <= entry_idx:
        return None, None, ""
    return limit_idx, float(frame.iloc[limit_idx]["close"]), "time_stop_10"


def _atr_bracket_exit_rule(frame: pd.DataFrame, entry_idx: int, cap_idx: int, direction: int) -> tuple[int | None, float | None, str]:
    entry_price = float(frame.iloc[entry_idx]["close"])
    atr = float(frame.iloc[entry_idx]["atr14"])
    if atr <= 0:
        return None, None, ""
    limit_idx = min(cap_idx, entry_idx + 15)
    if direction > 0:
        stop_price = entry_price - 1.5 * atr
        target_price = entry_price + 3.0 * atr
    else:
        stop_price = entry_price + 1.5 * atr
        target_price = entry_price - 3.0 * atr

    for idx in range(entry_idx + 1, limit_idx + 1):
        row = frame.iloc[idx]
        low = float(row["low"])
        high = float(row["high"])
        if direction > 0:
            if low <= stop_price:
                return idx, stop_price, "atr_stop"
            if high >= target_price:
                return idx, target_price, "atr_target"
        else:
            if high >= stop_price:
                return idx, stop_price, "atr_stop"
            if low <= target_price:
                return idx, target_price, "atr_target"
    if limit_idx <= entry_idx:
        return None, None, ""
    return limit_idx, float(frame.iloc[limit_idx]["close"]), "time_stop_15"


def _custom_exit_rule(frame: pd.DataFrame, entry_idx: int, cap_idx: int, direction: int) -> tuple[int | None, float | None, str]:
    entry_price = float(frame.iloc[entry_idx]["close"])
    atr = float(frame.iloc[entry_idx]["atr14"])
    if atr <= 0:
        return None, None, ""
    limit_idx = min(cap_idx, entry_idx + 15)
    if direction > 0:
        stop_price = entry_price - 1.2 * atr
    else:
        stop_price = entry_price + 1.2 * atr

    for idx in range(entry_idx + 1, limit_idx + 1):
        row = frame.iloc[idx]
        score = _alignment_score(row, direction)
        if direction > 0:
            if float(row["low"]) <= stop_price:
                return idx, stop_price, "atr_stop"
            if float(row["close"]) < float(row["ema10"]) or float(row["rsi14"]) < 50 or score < 2:
                return idx, float(row["close"]), "alignment_lost"
        else:
            if float(row["high"]) >= stop_price:
                return idx, stop_price, "atr_stop"
            if float(row["close"]) > float(row["ema10"]) or float(row["rsi14"]) > 50 or score < 2:
                return idx, float(row["close"]), "alignment_lost"
    if limit_idx <= entry_idx:
        return None, None, ""
    return limit_idx, float(frame.iloc[limit_idx]["close"]), "time_stop_15"


STRATEGIES: list[dict[str, Any]] = [
    {
        "name": "Control Hold 3D",
        "family": "control",
        "entry_filter": lambda row, direction: True,
        "exit_rule": _fixed_horizon_exit(3, "control"),
    },
    {
        "name": "Control Hold 5D",
        "family": "baseline",
        "entry_filter": lambda row, direction: True,
        "exit_rule": _fixed_horizon_exit(5, "baseline"),
    },
    {
        "name": "EMA Trend Hold 3D",
        "family": "trend",
        "entry_filter": lambda row, direction: (float(row["close"]) > float(row["ema20"]) > float(row["ema50"])) if direction > 0 else (float(row["close"]) < float(row["ema20"]) < float(row["ema50"])),
        "exit_rule": _fixed_horizon_exit(3, "trend"),
    },
    {
        "name": "RSI Regime Hold 3D",
        "family": "momentum",
        "entry_filter": lambda row, direction: float(row["rsi14"]) >= 55 if direction > 0 else float(row["rsi14"]) <= 45,
        "exit_rule": _fixed_horizon_exit(3, "momentum"),
    },
    {
        "name": "ATR Bracket",
        "family": "risk",
        "entry_filter": lambda row, direction: float(row["close"]) > float(row["ema20"]) if direction > 0 else float(row["close"]) < float(row["ema20"]),
        "exit_rule": _atr_bracket_exit_rule,
    },
    {
        "name": "Momentum Capture 3D",
        "family": "custom",
        "entry_filter": lambda row, direction: ((float(row["close"]) > float(row["ema20"]) > float(row["ema50"])) and float(row["rsi14"]) >= 55) if direction > 0 else ((float(row["close"]) < float(row["ema20"]) < float(row["ema50"])) and float(row["rsi14"]) <= 45),
        "exit_rule": _fixed_horizon_exit(3, "custom"),
    },
]


def _build_strategy_stats(trades: pd.DataFrame, signal_universe: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return _empty_strategy_stats()
    grouped = trades.groupby("strategy_name", dropna=False)
    stats = grouped["return_pct"].agg(
        trades="count",
        avg_return_pct="mean",
        median_return_pct="median",
        best_return_pct="max",
        worst_return_pct="min",
        total_return_pct="sum",
    ).reset_index()
    stats["wins"] = grouped["win"].sum().values
    stats["losses"] = grouped["loss"].sum().values
    stats["win_rate_pct"] = (stats["wins"] / stats["trades"]) * 100.0
    stats["avg_holding_days"] = grouped["holding_days"].mean().values
    stats["avg_alignment_score"] = grouped["entry_alignment_score"].mean().values
    stats["signal_mix_buy_pct"] = grouped.apply(lambda frame: float((frame["signal"] == "BUY").mean() * 100.0), include_groups=False).values
    stats["gross_profit_pct"] = grouped.apply(lambda frame: pd.to_numeric(frame.loc[frame["return_pct"] > 0, "return_pct"], errors="coerce").sum(), include_groups=False).values
    stats["gross_loss_pct_abs"] = grouped.apply(lambda frame: abs(pd.to_numeric(frame.loc[frame["return_pct"] < 0, "return_pct"], errors="coerce").sum()), include_groups=False).values
    stats["profit_factor"] = np.where(stats["gross_loss_pct_abs"] > 0, stats["gross_profit_pct"] / stats["gross_loss_pct_abs"], stats["gross_profit_pct"])
    stats["expectancy_pct"] = stats["avg_return_pct"]
    stats["strategy_family"] = stats["strategy_name"].map({strategy["name"]: strategy["family"] for strategy in STRATEGIES})
    stats["coverage_pct"] = (stats["trades"] / max(len(signal_universe), 1)) * 100.0
    stats["selection_score"] = (
        stats["avg_return_pct"].fillna(0.0) * stats["trades"].fillna(0.0)
        + stats["profit_factor"].fillna(0.0) * 10.0
        + stats["win_rate_pct"].fillna(0.0) * 0.2
    )
    return stats.sort_values(["selection_score", "profit_factor", "trades"], ascending=[False, False, False], na_position="last").reset_index(drop=True)


def _build_summary(strategy_stats: pd.DataFrame, signal_universe: pd.DataFrame, exchange: str, start_date: str) -> dict[str, Any]:
    best = strategy_stats.iloc[0].to_dict() if not strategy_stats.empty else {}
    technical_stats = strategy_stats[~strategy_stats["strategy_family"].isin({"control", "baseline"})] if not strategy_stats.empty else pd.DataFrame()
    best_technical = technical_stats.iloc[0].to_dict() if not technical_stats.empty else {}
    custom_stats = strategy_stats[strategy_stats["strategy_family"] == "custom"] if not strategy_stats.empty else pd.DataFrame()
    best_custom = custom_stats.iloc[0].to_dict() if not custom_stats.empty else {}
    signal_counts = signal_universe["signal"].astype(str).str.upper().value_counts().to_dict() if not signal_universe.empty else {}
    return {
        "exchange": exchange,
        "start_date": start_date,
        "signal_events": len(signal_universe),
        "buy_signals": signal_counts.get("BUY", 0),
        "sell_signals": signal_counts.get("SELL", 0),
        "strategies_tested": len(strategy_stats),
        "best_strategy_name": best.get("strategy_name", ""),
        "best_strategy_family": best.get("strategy_family", ""),
        "best_strategy_avg_return_pct": best.get("avg_return_pct", 0.0),
        "best_strategy_win_rate_pct": best.get("win_rate_pct", 0.0),
        "best_strategy_profit_factor": best.get("profit_factor", 0.0),
        "best_strategy_trades": best.get("trades", 0),
        "best_technical_strategy_name": best_technical.get("strategy_name", ""),
        "best_technical_strategy_family": best_technical.get("strategy_family", ""),
        "best_technical_strategy_avg_return_pct": best_technical.get("avg_return_pct", 0.0),
        "best_technical_strategy_profit_factor": best_technical.get("profit_factor", 0.0),
        "best_technical_strategy_trades": best_technical.get("trades", 0),
        "best_custom_strategy_name": best_custom.get("strategy_name", ""),
        "best_custom_strategy_avg_return_pct": best_custom.get("avg_return_pct", 0.0),
        "best_custom_strategy_profit_factor": best_custom.get("profit_factor", 0.0),
        "best_custom_strategy_trades": best_custom.get("trades", 0),
    }


def _empty_summary(exchange: str, start_date: str) -> dict[str, Any]:
    return {
        "exchange": exchange,
        "start_date": start_date,
        "signal_events": 0,
        "buy_signals": 0,
        "sell_signals": 0,
        "strategies_tested": 0,
        "best_strategy_name": "",
        "best_strategy_family": "",
        "best_strategy_avg_return_pct": 0.0,
        "best_strategy_win_rate_pct": 0.0,
        "best_strategy_profit_factor": 0.0,
        "best_strategy_trades": 0,
        "best_technical_strategy_name": "",
        "best_technical_strategy_family": "",
        "best_technical_strategy_avg_return_pct": 0.0,
        "best_technical_strategy_profit_factor": 0.0,
        "best_technical_strategy_trades": 0,
        "best_custom_strategy_name": "",
        "best_custom_strategy_avg_return_pct": 0.0,
        "best_custom_strategy_profit_factor": 0.0,
        "best_custom_strategy_trades": 0,
    }


def _empty_strategy_stats() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "strategy_name",
            "strategy_family",
            "trades",
            "wins",
            "losses",
            "win_rate_pct",
            "avg_return_pct",
            "median_return_pct",
            "best_return_pct",
            "worst_return_pct",
            "total_return_pct",
            "gross_profit_pct",
            "gross_loss_pct_abs",
            "profit_factor",
            "expectancy_pct",
            "avg_holding_days",
            "avg_alignment_score",
            "signal_mix_buy_pct",
            "coverage_pct",
            "selection_score",
        ]
    )


def _empty_trade_details() -> pd.DataFrame:
    return pd.DataFrame(columns=_trade_columns())


def _trade_columns() -> list[str]:
    return [
        "strategy_name",
        "strategy_family",
        "exchange",
        "symbol",
        "name",
        "signal",
        "signal_date",
        "entry_date",
        "entry_price",
        "exit_date",
        "exit_price",
        "exit_reason",
        "return_pct",
        "holding_bars",
        "holding_days",
        "win",
        "loss",
        "entry_rsi14",
        "entry_atr14",
        "entry_ema20_gap_pct",
        "entry_macd_hist",
        "entry_alignment_score",
    ]


def _emit_progress(
    progress_callback: Callable[[dict[str, Any]], None] | None,
    *,
    phase: str,
    completed: int,
    total: int,
    current_symbol: str,
    current_exchange: str,
) -> None:
    if not progress_callback:
        return
    progress_callback(
        {
            "phase": phase,
            "completed": completed,
            "total": total,
            "current_symbol": current_symbol,
            "current_exchange": current_exchange,
        }
    )
