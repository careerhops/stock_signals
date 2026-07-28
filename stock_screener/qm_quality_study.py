from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from stock_screener.data.storage import Storage
from stock_screener.resample import resample_daily_to_weekly
from stock_screener.strategy.weekly_buy_sell import run_weekly_buy_sell
from stock_screener.weekly_buy_tracker_study import _config_with_sensitivity, _emit_progress, _load_name_map


DEFAULT_BUY_START_DATE = "2026-04-01"
DEFAULT_BUY_END_DATE = "2026-04-30"


@dataclass(frozen=True)
class QMQualityStudyResult:
    summary: dict[str, Any]
    stock_stats: pd.DataFrame
    buy_events: pd.DataFrame


def run_qm_quality_study(
    config: dict[str, Any],
    storage: Storage,
    exchange: str = "NSE",
    buy_start_date: str = DEFAULT_BUY_START_DATE,
    buy_end_date: str = DEFAULT_BUY_END_DATE,
    price_as_of_date: str | None = None,
    signal_frame: pd.DataFrame | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> QMQualityStudyResult:
    data_root = storage.data_root
    all_symbols = sorted(p.stem for p in (data_root / "candles" / exchange / "1D").glob("*.csv"))
    start_ts = pd.Timestamp(buy_start_date)
    end_ts = pd.Timestamp(buy_end_date)
    price_as_of_ts = pd.Timestamp(price_as_of_date) if str(price_as_of_date or "").strip() else None
    weekly_anchor = config.get("strategy", {}).get("weekly_anchor", "W-FRI")
    use_completed_weeks_only = bool(config.get("strategy", {}).get("use_completed_weeks_only", True))
    name_map = _load_name_map(storage, exchange)
    benchmark = _prepare_daily_frame(storage.load_candles("NSE_INDEX", "NIFTY 50", "1D"))

    buy_rows: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    signal_events_lookup = _normalize_signal_frame(signal_frame, exchange)
    if signal_events_lookup is not None:
        all_symbols = sorted(signal_events_lookup.keys())

    _emit_progress(
        progress_callback,
        phase="Analyzing Quantitative Momentum Quality",
        completed=0,
        total=len(all_symbols),
        current_symbol="",
        current_exchange=exchange,
    )

    for index, symbol in enumerate(all_symbols, start=1):
        daily = _prepare_daily_frame(storage.load_candles(exchange, symbol, "1D"))
        _emit_progress(
            progress_callback,
            phase="Analyzing Quantitative Momentum Quality",
            completed=index,
            total=len(all_symbols),
            current_symbol=symbol,
            current_exchange=exchange,
        )
        if daily.empty or len(daily) < 260:
            continue

        weekly = resample_daily_to_weekly(
            daily,
            weekly_anchor=weekly_anchor,
            use_completed_weeks_only=use_completed_weeks_only,
        )
        if weekly.empty:
            continue

        symbol_events: list[dict[str, Any]] = []
        if signal_events_lookup is not None:
            symbol_events = signal_events_lookup.get(symbol, [])
            if symbol_events:
                buy_rows.append(
                    pd.DataFrame(
                        [
                            {
                                "exchange": exchange,
                                "symbol": symbol,
                                "name": name_map.get(symbol, symbol),
                                "date": event["date"],
                                "close": event["close"],
                                "sensitivity": event["sensitivity"],
                            }
                            for event in symbol_events
                        ]
                    )
                )
        else:
            for sensitivity in (2, 3):
                strategy_output = run_weekly_buy_sell(weekly, _config_with_sensitivity(config, sensitivity))
                if strategy_output.empty:
                    continue
                buys = strategy_output[
                    (strategy_output["signal"].astype(str).str.upper() == "BUY")
                    & (pd.to_datetime(strategy_output["date"], errors="coerce") >= start_ts)
                    & (pd.to_datetime(strategy_output["date"], errors="coerce") <= end_ts)
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
                buy_rows.append(buys[["exchange", "symbol", "name", "date", "close", "sensitivity"]])
                for _, row in buys.iterrows():
                    symbol_events.append(
                        {
                            "date": pd.Timestamp(row["date"]),
                            "close": _to_float(row["close"]),
                            "sensitivity": int(sensitivity),
                        }
                    )

        if not symbol_events:
            continue

        symbol_events.sort(key=lambda item: item["date"])
        latest_event = symbol_events[-1]
        as_of_date = latest_event["date"]
        signal_snapshot = _snapshot_as_of(daily, benchmark, as_of_date)
        if not signal_snapshot:
            continue
        latest_close, latest_close_date = _price_on_or_before(daily, price_as_of_ts)
        signal_snapshot.update(
            {
                "exchange": exchange,
                "symbol": symbol,
                "name": name_map.get(symbol, symbol),
                "april_buy_count": len(symbol_events),
                "s2_april_buy_count": sum(1 for event in symbol_events if int(event["sensitivity"]) == 2),
                "s3_april_buy_count": sum(1 for event in symbol_events if int(event["sensitivity"]) == 3),
                "latest_april_buy_date": as_of_date,
                "latest_april_buy_price": latest_event["close"],
                "first_april_buy_date": symbol_events[0]["date"],
                "first_april_buy_price": symbol_events[0]["close"],
                "latest_close": latest_close,
                "latest_close_date": latest_close_date,
            }
        )
        latest_close = signal_snapshot.get("latest_close")
        latest_april_buy_price = signal_snapshot.get("latest_april_buy_price")
        if latest_close is not None and latest_april_buy_price not in (None, 0):
            signal_snapshot["current_gain_pct"] = ((float(latest_close) - float(latest_april_buy_price)) / float(latest_close)) * 100.0
        else:
            signal_snapshot["current_gain_pct"] = pd.NA
        diagnostics.append(signal_snapshot)

    buy_events = pd.concat(buy_rows, ignore_index=True) if buy_rows else _empty_event_frame()
    stock_stats = pd.DataFrame(diagnostics)
    if not stock_stats.empty:
        stock_stats = _score_qm_quality(stock_stats)
        stock_stats = stock_stats.sort_values(
            ["qm_composite_score", "current_gain_pct", "symbol"],
            ascending=[False, False, True],
            na_position="last",
        ).reset_index(drop=True)
    summary = _build_summary(
        exchange,
        buy_start_date,
        buy_end_date,
        all_symbols,
        stock_stats,
        buy_events,
        mode="latest_weekly_buy_bucket" if signal_events_lookup is not None else "date_range",
        price_as_of_date=price_as_of_ts.strftime("%Y-%m-%d") if price_as_of_ts is not None else "",
    )
    return QMQualityStudyResult(summary=summary, stock_stats=stock_stats, buy_events=buy_events)


def save_qm_quality_outputs(result: QMQualityStudyResult, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "latest_summary.csv"
    stock_stats_path = output_dir / "latest_stock_stats.csv"
    buy_events_path = output_dir / "latest_buy_events.csv"
    pd.DataFrame([result.summary]).to_csv(summary_path, index=False)
    result.stock_stats.to_csv(stock_stats_path, index=False)
    result.buy_events.to_csv(buy_events_path, index=False)
    return {"summary": summary_path, "stock_stats": stock_stats_path, "buy_events": buy_events_path}


def load_qm_quality_outputs(output_dir: Path) -> QMQualityStudyResult:
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
    return QMQualityStudyResult(summary=summary, stock_stats=_read(output_dir / "latest_stock_stats.csv"), buy_events=_read(output_dir / "latest_buy_events.csv"))


def _prepare_daily_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    daily = frame.copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily["close"] = pd.to_numeric(daily["close"], errors="coerce")
    daily = daily.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return daily


def _price_on_or_before(daily: pd.DataFrame, price_as_of_ts: pd.Timestamp | None) -> tuple[float | None, pd.Timestamp | pd.NaT]:
    if daily.empty:
        return None, pd.NaT
    if price_as_of_ts is None:
        row = daily.iloc[-1]
        return _to_float(row["close"]), pd.Timestamp(row["date"])
    subset = daily[daily["date"] <= price_as_of_ts]
    if subset.empty:
        return None, pd.NaT
    row = subset.iloc[-1]
    return _to_float(row["close"]), pd.Timestamp(row["date"])


def _normalize_signal_frame(signal_frame: pd.DataFrame | None, exchange: str) -> dict[str, list[dict[str, Any]]] | None:
    if signal_frame is None:
        return None
    if signal_frame.empty:
        return {}
    frame = signal_frame.copy()
    if "signal" in frame.columns:
        frame = frame[frame["signal"].astype(str).str.upper() == "BUY"].copy()
    if "exchange" in frame.columns:
        frame = frame[frame["exchange"].astype(str).str.upper() == exchange.upper()].copy()
    if frame.empty or "symbol" not in frame.columns or "date" not in frame.columns:
        return {}
    frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame.get("close"), errors="coerce")
    if "sensitivity" in frame.columns:
        frame["sensitivity"] = pd.to_numeric(frame["sensitivity"], errors="coerce").fillna(3).astype(int)
    else:
        frame["sensitivity"] = 3
    frame = frame.dropna(subset=["symbol", "date"]).copy()
    lookup: dict[str, list[dict[str, Any]]] = {}
    for symbol, group in frame.groupby("symbol", dropna=False):
        events = []
        for _, row in group.sort_values("date").iterrows():
            events.append({"date": pd.Timestamp(row["date"]), "close": _to_float(row.get("close")), "sensitivity": int(row["sensitivity"])})
        lookup[str(symbol)] = events
    return lookup


def _snapshot_as_of(daily: pd.DataFrame, benchmark: pd.DataFrame, as_of_date: pd.Timestamp) -> dict[str, Any]:
    as_of_daily = daily[daily["date"] <= as_of_date].copy()
    if len(as_of_daily) < 260:
        return {}
    bench_as_of = benchmark[benchmark["date"] <= as_of_date].copy() if not benchmark.empty else pd.DataFrame()

    close = as_of_daily["close"].astype(float).reset_index(drop=True)
    current_price = float(close.iloc[-1])

    mom_6m = _return_between(close, 126, 0)
    mom_9m = _return_between(close, 189, 0)
    mom_12_1 = _return_between(close, 252, 21)
    recent_1m = _return_between(close, 21, 0)
    beta_252 = _beta_252(as_of_daily, bench_as_of)
    pos_day_pct, neg_day_pct, top_gap_share_pct = _frog_in_pan_metrics(close)

    return {
        "as_of_date": as_of_date,
        "as_of_price": current_price,
        "momentum_6m_pct": mom_6m,
        "momentum_9m_pct": mom_9m,
        "momentum_12_1_pct": mom_12_1,
        "recent_1m_pct": recent_1m,
        "beta_252": beta_252,
        "positive_day_pct": pos_day_pct,
        "negative_day_pct": neg_day_pct,
        "top_gap_share_pct": top_gap_share_pct,
    }


def _return_between(close: pd.Series, lookback_days: int, skip_recent_days: int) -> float | None:
    end_index = len(close) - 1 - int(skip_recent_days)
    start_index = end_index - int(lookback_days)
    if start_index < 0 or end_index < 0 or end_index >= len(close):
        return None
    start_value = float(close.iloc[start_index])
    end_value = float(close.iloc[end_index])
    if start_value <= 0:
        return None
    return ((end_value / start_value) - 1.0) * 100.0


def _beta_252(stock_daily: pd.DataFrame, benchmark_daily: pd.DataFrame) -> float | None:
    if stock_daily.empty or benchmark_daily.empty:
        return None
    stock = stock_daily[["date", "close"]].copy()
    bench = benchmark_daily[["date", "close"]].copy()
    stock["ret"] = stock["close"].pct_change()
    bench["bench_ret"] = bench["close"].pct_change()
    merged = stock.merge(bench[["date", "bench_ret"]], on="date", how="inner").dropna(subset=["ret", "bench_ret"])
    if len(merged) < 126:
        return None
    merged = merged.tail(252)
    if merged["bench_ret"].std(ddof=0) == 0:
        return None
    covariance = float(np.cov(merged["ret"], merged["bench_ret"], ddof=0)[0, 1])
    variance = float(np.var(merged["bench_ret"], ddof=0))
    if variance == 0:
        return None
    return covariance / variance


def _frog_in_pan_metrics(close: pd.Series) -> tuple[float | None, float | None, float | None]:
    if len(close) < 260:
        return None, None, None
    window = close.iloc[-253:-21].pct_change().dropna()
    if window.empty:
        return None, None, None
    positive_days = float((window > 0).sum())
    negative_days = float((window < 0).sum())
    total_days = float(len(window))
    positive_pct = (positive_days / total_days) * 100.0 if total_days else None
    negative_pct = (negative_days / total_days) * 100.0 if total_days else None

    positive_returns = window[window > 0].sort_values(ascending=False)
    total_positive = float(positive_returns.sum()) if not positive_returns.empty else 0.0
    if total_positive > 0:
        top_gap_share = float(positive_returns.head(5).sum() / total_positive) * 100.0
    else:
        top_gap_share = None
    return positive_pct, negative_pct, top_gap_share


def _score_qm_quality(stock_stats: pd.DataFrame) -> pd.DataFrame:
    scored = stock_stats.copy()
    for column in (
        "momentum_6m_pct",
        "momentum_9m_pct",
        "momentum_12_1_pct",
        "recent_1m_pct",
        "beta_252",
        "positive_day_pct",
        "negative_day_pct",
        "top_gap_share_pct",
        "current_gain_pct",
    ):
        if column in scored.columns:
            scored[column] = pd.to_numeric(scored[column], errors="coerce")

    scored["beta_rank_pct"] = scored["beta_252"].rank(pct=True, ascending=True, method="average") * 100.0
    scored["mom6_rank_pct"] = scored["momentum_6m_pct"].rank(pct=True, ascending=True, method="average") * 100.0
    scored["mom9_rank_pct"] = scored["momentum_9m_pct"].rank(pct=True, ascending=True, method="average") * 100.0
    scored["mom12_1_rank_pct"] = scored["momentum_12_1_pct"].rank(pct=True, ascending=True, method="average") * 100.0
    scored["fip_rank_pct"] = scored["positive_day_pct"].rank(pct=True, ascending=True, method="average") * 100.0
    scored["gap_rank_pct"] = scored["top_gap_share_pct"].rank(pct=True, ascending=False, method="average") * 100.0
    scored["quality_rank_pct"] = ((scored["fip_rank_pct"].fillna(0) * 0.7) + (scored["gap_rank_pct"].fillna(0) * 0.3))
    scored["qm_composite_score"] = ((scored["mom12_1_rank_pct"].fillna(0) * 0.6) + (scored["quality_rank_pct"].fillna(0) * 0.4))

    scored["qm_beta_outlier"] = scored["beta_rank_pct"] >= 90.0
    scored["qm_mom6_outlier"] = scored["mom6_rank_pct"] <= 5.0
    scored["qm_mom9_outlier"] = scored["mom9_rank_pct"] <= 5.0
    scored["qm_outlier_pass"] = ~(scored["qm_beta_outlier"].fillna(False) | scored["qm_mom6_outlier"].fillna(False) | scored["qm_mom9_outlier"].fillna(False))
    scored["qm_quality_bucket"] = pd.cut(
        scored["qm_composite_score"],
        bins=[-np.inf, 25, 50, 75, np.inf],
        labels=["Low", "Medium", "High", "Elite"],
    ).astype("object")
    return scored


def _build_summary(
    exchange: str,
    buy_start_date: str,
    buy_end_date: str,
    all_symbols: list[str],
    stock_stats: pd.DataFrame,
    buy_events: pd.DataFrame,
    mode: str,
    price_as_of_date: str,
) -> dict[str, Any]:
    profitable = pd.to_numeric(stock_stats.get("current_gain_pct"), errors="coerce")
    profitable_count = int((profitable > 0).sum()) if len(profitable) else 0
    elite = stock_stats[stock_stats.get("qm_quality_bucket").astype(str) == "Elite"] if not stock_stats.empty and "qm_quality_bucket" in stock_stats.columns else pd.DataFrame()
    return {
        "exchange": exchange,
        "mode": mode,
        "buy_start_date": buy_start_date,
        "buy_end_date": buy_end_date,
        "price_as_of_date": price_as_of_date,
        "symbols_processed": len(all_symbols),
        "april_buy_symbols": int(len(stock_stats)),
        "april_buy_events": int(len(buy_events)),
        "profitable_today": profitable_count,
        "elite_qm_count": int(len(elite)),
        "outlier_pass_count": int(stock_stats["qm_outlier_pass"].fillna(False).sum()) if not stock_stats.empty and "qm_outlier_pass" in stock_stats.columns else 0,
        "latest_close_date": str(pd.to_datetime(stock_stats["latest_close_date"], errors="coerce").max().date()) if (not stock_stats.empty and "latest_close_date" in stock_stats.columns and pd.to_datetime(stock_stats["latest_close_date"], errors="coerce").notna().any()) else "",
        "avg_current_gain_pct": round(float(profitable.dropna().mean()), 2) if hasattr(profitable, "dropna") and not profitable.dropna().empty else None,
    }


def _empty_event_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["exchange", "symbol", "name", "date", "close", "sensitivity"])


def _to_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
