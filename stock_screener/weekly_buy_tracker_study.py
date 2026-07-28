from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from stock_screener.data.storage import Storage
from stock_screener.resample import resample_daily_to_weekly
from stock_screener.strategy.weekly_buy_sell import run_weekly_buy_sell


DEFAULT_START_DATE = "2026-01-01"


@dataclass(frozen=True)
class WeeklyBuyTrackerResult:
    summary: dict[str, Any]
    stock_stats: pd.DataFrame
    s2_events: pd.DataFrame
    s3_events: pd.DataFrame


def run_weekly_buy_tracker_study(
    config: dict[str, Any],
    storage: Storage,
    exchange: str = "NSE",
    start_date: str = DEFAULT_START_DATE,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> WeeklyBuyTrackerResult:
    data_root = storage.data_root
    all_symbols = sorted(p.stem for p in (data_root / "candles" / exchange / "1D").glob("*.csv"))
    start_ts = pd.Timestamp(start_date)
    weekly_anchor = config.get("strategy", {}).get("weekly_anchor", "W-FRI")
    use_completed_weeks_only = bool(config.get("strategy", {}).get("use_completed_weeks_only", True))
    name_map = _load_name_map(storage, exchange)

    s2_rows: list[pd.DataFrame] = []
    s3_rows: list[pd.DataFrame] = []
    current_rows: list[dict[str, Any]] = []

    _emit_progress(
        progress_callback,
        phase="Analyzing weekly BUY history",
        completed=0,
        total=len(all_symbols),
        current_symbol="",
        current_exchange=exchange,
    )

    for index, symbol in enumerate(all_symbols, start=1):
        daily = storage.load_candles(exchange, symbol, "1D")
        _emit_progress(
            progress_callback,
            phase="Analyzing weekly BUY history",
            completed=index,
            total=len(all_symbols),
            current_symbol=symbol,
            current_exchange=exchange,
        )
        if daily.empty or len(daily) < 40:
            continue
        daily = daily.copy()
        daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
        daily = daily.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        if daily.empty:
            continue

        weekly = resample_daily_to_weekly(
            daily,
            weekly_anchor=weekly_anchor,
            use_completed_weeks_only=use_completed_weeks_only,
        )
        if weekly.empty:
            continue

        has_buy_history = False
        for sensitivity, bucket in ((2, s2_rows), (3, s3_rows)):
            strategy_config = _config_with_sensitivity(config, sensitivity)
            strategy_output = run_weekly_buy_sell(weekly, strategy_config)
            if strategy_output.empty:
                continue
            buys = strategy_output[
                (strategy_output["signal"].astype(str).str.upper() == "BUY")
                & (pd.to_datetime(strategy_output["date"], errors="coerce") >= start_ts)
            ].copy()
            if buys.empty:
                continue
            buys["date"] = pd.to_datetime(buys["date"], errors="coerce")
            buys = buys[buys["date"].notna()].copy()
            if buys.empty:
                continue
            buys["exchange"] = exchange
            buys["symbol"] = symbol
            buys["name"] = name_map.get(symbol, symbol)
            buys["sensitivity"] = sensitivity
            bucket.append(buys[["exchange", "symbol", "name", "date", "close", "sensitivity"]])
            has_buy_history = True

        if not has_buy_history:
            continue

        latest_daily = daily.iloc[-1]
        minervini = _evaluate_minervini_template(daily)
        obv_macd = _evaluate_obv_macd(daily)
        latest_volume_burst = _evaluate_latest_volume_burst(daily)
        current_rows.append(
            {
                "exchange": exchange,
                "symbol": symbol,
                "name": name_map.get(symbol, symbol),
                "latest_close": _to_float(latest_daily.get("close")),
                "latest_close_date": latest_daily.get("date"),
                **minervini,
                **obv_macd,
                **latest_volume_burst,
            }
        )

    s2_events = pd.concat(s2_rows, ignore_index=True) if s2_rows else _empty_event_frame()
    s3_events = pd.concat(s3_rows, ignore_index=True) if s3_rows else _empty_event_frame()
    current_frame = pd.DataFrame(current_rows)
    stock_stats = _build_stock_stats(exchange, start_ts, current_frame, s2_events, s3_events)
    summary = _build_summary(exchange, start_ts, len(all_symbols), stock_stats, s2_events, s3_events)
    return WeeklyBuyTrackerResult(summary, stock_stats, s2_events, s3_events)


def save_weekly_buy_tracker_outputs(result: WeeklyBuyTrackerResult, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "latest_summary.csv"
    stock_stats_path = output_dir / "latest_stock_stats.csv"
    s2_events_path = output_dir / "latest_s2_events.csv"
    s3_events_path = output_dir / "latest_s3_events.csv"
    pd.DataFrame([result.summary]).to_csv(summary_path, index=False)
    result.stock_stats.to_csv(stock_stats_path, index=False)
    result.s2_events.to_csv(s2_events_path, index=False)
    result.s3_events.to_csv(s3_events_path, index=False)
    return {
        "summary": summary_path,
        "stock_stats": stock_stats_path,
        "s2_events": s2_events_path,
        "s3_events": s3_events_path,
    }


def load_weekly_buy_tracker_outputs(output_dir: Path) -> WeeklyBuyTrackerResult:
    def _read(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    summary = {}
    summary_path = output_dir / "latest_summary.csv"
    if summary_path.exists():
        try:
            frame = pd.read_csv(summary_path)
            if not frame.empty:
                summary = frame.iloc[0].to_dict()
        except pd.errors.EmptyDataError:
            summary = {}

    return WeeklyBuyTrackerResult(
        summary=summary,
        stock_stats=_read(output_dir / "latest_stock_stats.csv"),
        s2_events=_read(output_dir / "latest_s2_events.csv"),
        s3_events=_read(output_dir / "latest_s3_events.csv"),
    )


def _build_stock_stats(
    exchange: str,
    start_ts: pd.Timestamp,
    current_frame: pd.DataFrame,
    s2_events: pd.DataFrame,
    s3_events: pd.DataFrame,
) -> pd.DataFrame:
    all_events = pd.concat([s2_events, s3_events], ignore_index=True) if (not s2_events.empty or not s3_events.empty) else _empty_event_frame()
    if all_events.empty:
        return pd.DataFrame()

    all_events = all_events.copy()
    all_events["date"] = pd.to_datetime(all_events["date"], errors="coerce")
    all_events = all_events.dropna(subset=["date"]).sort_values(["symbol", "date", "sensitivity"]).reset_index(drop=True)

    summary_rows: list[dict[str, Any]] = []
    for symbol, group in all_events.groupby("symbol", dropna=False):
        row: dict[str, Any] = {
            "exchange": exchange,
            "symbol": symbol,
            "name": str(group["name"].iloc[-1]) if "name" in group.columns else str(symbol),
            "total_buy_count": int(len(group)),
        }
        any_first = group.sort_values("date").iloc[0]
        any_last = group.sort_values("date").iloc[-1]
        row["first_buy_date"] = any_first["date"]
        row["first_buy_price"] = _to_float(any_first["close"])
        row["latest_buy_date"] = any_last["date"]
        row["latest_buy_price"] = _to_float(any_last["close"])
        for sensitivity in (2, 3):
            subset = group[group["sensitivity"] == sensitivity].sort_values("date")
            prefix = f"s{int(sensitivity)}"
            row[f"{prefix}_buy_count"] = int(len(subset))
            if subset.empty:
                row[f"first_{prefix}_buy_date"] = pd.NaT
                row[f"first_{prefix}_buy_price"] = pd.NA
                row[f"latest_{prefix}_buy_date"] = pd.NaT
                row[f"latest_{prefix}_buy_price"] = pd.NA
                continue
            row[f"first_{prefix}_buy_date"] = subset.iloc[0]["date"]
            row[f"first_{prefix}_buy_price"] = _to_float(subset.iloc[0]["close"])
            row[f"latest_{prefix}_buy_date"] = subset.iloc[-1]["date"]
            row[f"latest_{prefix}_buy_price"] = _to_float(subset.iloc[-1]["close"])
        summary_rows.append(row)

    stats = pd.DataFrame(summary_rows)
    if not current_frame.empty:
        merge_columns = ["exchange", "symbol"] + [column for column in current_frame.columns if column not in {"exchange", "symbol"}]
        stats = stats.merge(current_frame[merge_columns], on=["exchange", "symbol"], how="left", suffixes=("", "_current"))
    for source_col, target_col in (
        ("first_buy_price", "gain_vs_first_buy_pct"),
        ("latest_buy_price", "gain_vs_latest_buy_pct"),
        ("latest_s2_buy_price", "gain_vs_latest_s2_buy_pct"),
        ("latest_s3_buy_price", "gain_vs_latest_s3_buy_pct"),
    ):
        base = pd.to_numeric(stats.get(source_col), errors="coerce")
        current = pd.to_numeric(stats.get("latest_close"), errors="coerce")
        stats[target_col] = ((current - base) / base) * 100.0
        stats.loc[base.isna() | (base == 0) | current.isna(), target_col] = pd.NA
    stats["start_date"] = start_ts.strftime("%Y-%m-%d")
    stats = stats.sort_values(
        ["latest_buy_date", "total_buy_count", "symbol"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    return stats


def _build_summary(
    exchange: str,
    start_ts: pd.Timestamp,
    symbols_processed: int,
    stock_stats: pd.DataFrame,
    s2_events: pd.DataFrame,
    s3_events: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "exchange": exchange,
        "start_date": start_ts.strftime("%Y-%m-%d"),
        "symbols_processed": symbols_processed,
        "stocks_with_buy_history": int(len(stock_stats)),
        "s2_buy_events": int(len(s2_events)),
        "s3_buy_events": int(len(s3_events)),
        "unique_s2_symbols": int(s2_events["symbol"].nunique()) if not s2_events.empty else 0,
        "unique_s3_symbols": int(s3_events["symbol"].nunique()) if not s3_events.empty else 0,
        "latest_close_date": str(pd.to_datetime(stock_stats["latest_close_date"], errors="coerce").max().date()) if (not stock_stats.empty and "latest_close_date" in stock_stats.columns and pd.to_datetime(stock_stats["latest_close_date"], errors="coerce").notna().any()) else "",
    }


def _config_with_sensitivity(config: dict[str, Any], sensitivity: int) -> dict[str, Any]:
    updated = dict(config)
    strategy = dict(updated.get("strategy", {}))
    strategy["sensitivity"] = int(sensitivity)
    updated["strategy"] = strategy
    return updated


def _evaluate_minervini_template(daily: pd.DataFrame) -> dict[str, Any]:
    if daily.empty or len(daily) < 200:
        return {
            "minervini_rule_count": 0,
            "minervini_pass": False,
            "minervini_close_above_sma50": False,
            "minervini_close_above_sma150": False,
            "minervini_close_above_sma200": False,
            "minervini_sma50_above_sma150": False,
            "minervini_sma150_above_sma200": False,
            "minervini_sma200_above_sma200_20d_ago": False,
            "minervini_close_above_52w_low_30pct": False,
            "minervini_close_within_25pct_of_52w_high": False,
        }

    frame = daily.copy()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["high"] = pd.to_numeric(frame["high"], errors="coerce")
    frame["low"] = pd.to_numeric(frame["low"], errors="coerce")
    frame = frame.dropna(subset=["date", "close", "high", "low"]).sort_values("date").reset_index(drop=True)
    if frame.empty or len(frame) < 200:
        return {
            "minervini_rule_count": 0,
            "minervini_pass": False,
            "minervini_close_above_sma50": False,
            "minervini_close_above_sma150": False,
            "minervini_close_above_sma200": False,
            "minervini_sma50_above_sma150": False,
            "minervini_sma150_above_sma200": False,
            "minervini_sma200_above_sma200_20d_ago": False,
            "minervini_close_above_52w_low_30pct": False,
            "minervini_close_within_25pct_of_52w_high": False,
        }

    close_series = frame["close"]
    latest = frame.iloc[-1]
    close = float(latest["close"])

    sma_50 = close_series.rolling(50, min_periods=50).mean().iloc[-1]
    sma_150 = close_series.rolling(150, min_periods=150).mean().iloc[-1]
    sma_200 = close_series.rolling(200, min_periods=200).mean().iloc[-1]
    sma_200_20d_ago = close_series.rolling(200, min_periods=200).mean().shift(20).iloc[-1]
    high_52w = frame["high"].rolling(252, min_periods=180).max().iloc[-1]
    low_52w = frame["low"].rolling(252, min_periods=180).min().iloc[-1]

    rules = {
        "minervini_close_above_sma50": bool(pd.notna(sma_50) and close > float(sma_50)),
        "minervini_close_above_sma150": bool(pd.notna(sma_150) and close > float(sma_150)),
        "minervini_close_above_sma200": bool(pd.notna(sma_200) and close > float(sma_200)),
        "minervini_sma50_above_sma150": bool(pd.notna(sma_50) and pd.notna(sma_150) and float(sma_50) > float(sma_150)),
        "minervini_sma150_above_sma200": bool(pd.notna(sma_150) and pd.notna(sma_200) and float(sma_150) > float(sma_200)),
        "minervini_sma200_above_sma200_20d_ago": bool(pd.notna(sma_200) and pd.notna(sma_200_20d_ago) and float(sma_200) > float(sma_200_20d_ago)),
        "minervini_close_above_52w_low_30pct": bool(pd.notna(low_52w) and float(low_52w) > 0 and close >= float(low_52w) * 1.30),
        "minervini_close_within_25pct_of_52w_high": bool(pd.notna(high_52w) and float(high_52w) > 0 and close >= float(high_52w) * 0.75),
    }
    rule_count = int(sum(1 for passed in rules.values() if passed))
    return {
        "minervini_rule_count": rule_count,
        "minervini_pass": rule_count == 8,
        **rules,
    }


def _evaluate_obv_macd(daily: pd.DataFrame) -> dict[str, Any]:
    if daily.empty or len(daily) < 35:
        return {
            "obv_macd_line": pd.NA,
            "obv_macd_signal": pd.NA,
            "obv_macd_hist": pd.NA,
            "obv_macd_above_zero": False,
            "obv_macd_cross_up": False,
            "obv_macd_pass": False,
        }

    frame = daily.copy()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    frame = frame.dropna(subset=["date", "close", "volume"]).sort_values("date").reset_index(drop=True)
    if frame.empty or len(frame) < 35:
        return {
            "obv_macd_line": pd.NA,
            "obv_macd_signal": pd.NA,
            "obv_macd_hist": pd.NA,
            "obv_macd_above_zero": False,
            "obv_macd_cross_up": False,
            "obv_macd_pass": False,
        }

    direction = frame["close"].diff().fillna(0.0)
    obv_delta = pd.Series(0.0, index=frame.index, dtype="float64")
    obv_delta.loc[direction > 0] = frame.loc[direction > 0, "volume"]
    obv_delta.loc[direction < 0] = -frame.loc[direction < 0, "volume"]
    obv = obv_delta.cumsum()

    macd_line = obv.ewm(span=12, adjust=False).mean() - obv.ewm(span=26, adjust=False).mean()
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal_line

    latest_macd = macd_line.iloc[-1]
    prev_macd = macd_line.iloc[-2] if len(macd_line) > 1 else pd.NA
    above_zero = pd.notna(latest_macd) and float(latest_macd) > 0
    cross_up = (
        pd.notna(latest_macd)
        and pd.notna(prev_macd)
        and float(prev_macd) <= 0
        and float(latest_macd) > 0
    )

    return {
        "obv_macd_line": float(latest_macd) if pd.notna(latest_macd) else pd.NA,
        "obv_macd_signal": float(signal_line.iloc[-1]) if pd.notna(signal_line.iloc[-1]) else pd.NA,
        "obv_macd_hist": float(hist.iloc[-1]) if pd.notna(hist.iloc[-1]) else pd.NA,
        "obv_macd_above_zero": bool(above_zero),
        "obv_macd_cross_up": bool(cross_up),
        "obv_macd_pass": bool(cross_up),
    }


def _evaluate_latest_volume_burst(daily: pd.DataFrame) -> dict[str, Any]:
    if daily.empty or len(daily) < 10:
        return {
            "latest_volume": pd.NA,
            "prev_9d_avg_volume": pd.NA,
            "latest_volume_ratio_prev_9d": pd.NA,
            "latest_volume_3x_prev_9d": False,
        }

    frame = daily.copy()
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    frame = frame.dropna(subset=["date", "volume"]).sort_values("date").reset_index(drop=True)
    if frame.empty or len(frame) < 10:
        return {
            "latest_volume": pd.NA,
            "prev_9d_avg_volume": pd.NA,
            "latest_volume_ratio_prev_9d": pd.NA,
            "latest_volume_3x_prev_9d": False,
        }

    latest_volume = frame["volume"].iloc[-1]
    prev_9d_avg_volume = frame["volume"].iloc[-10:-1].mean()
    ratio = pd.NA
    passes = False
    if pd.notna(latest_volume) and pd.notna(prev_9d_avg_volume) and float(prev_9d_avg_volume) > 0:
        ratio = float(latest_volume) / float(prev_9d_avg_volume)
        passes = ratio >= 3.0

    return {
        "latest_volume": float(latest_volume) if pd.notna(latest_volume) else pd.NA,
        "prev_9d_avg_volume": float(prev_9d_avg_volume) if pd.notna(prev_9d_avg_volume) else pd.NA,
        "latest_volume_ratio_prev_9d": ratio,
        "latest_volume_3x_prev_9d": bool(passes),
    }


def _load_name_map(storage: Storage, exchange: str) -> dict[str, str]:
    instruments = storage.load_instruments()
    if instruments.empty:
        return {}
    frame = instruments.copy()
    if "exchange" in frame.columns:
        frame = frame[frame["exchange"].astype(str).str.upper() == exchange.upper()]
    symbol_column = "tradingsymbol" if "tradingsymbol" in frame.columns else "symbol"
    if symbol_column not in frame.columns or "name" not in frame.columns:
        return {}
    frame[symbol_column] = frame[symbol_column].astype(str).str.upper().str.strip()
    frame["name"] = frame["name"].fillna("").astype(str).str.strip()
    frame = frame[frame[symbol_column] != ""].drop_duplicates(subset=[symbol_column], keep="last")
    return dict(zip(frame[symbol_column], frame["name"]))


def _empty_event_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["exchange", "symbol", "name", "date", "close", "sensitivity"])


def _to_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _emit_progress(
    callback: Callable[[dict[str, Any]], None] | None,
    *,
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
