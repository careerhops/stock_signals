from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from stock_screener.data.storage import Storage
from stock_screener.resample import resample_daily_to_weekly
from stock_screener.strategy.weekly_buy_sell import run_weekly_buy_sell

DEFAULT_IN_SAMPLE_YEARS = 2
DEFAULT_OUT_OF_SAMPLE_MONTHS = 6
DEFAULT_MIN_SHARPE_RATIO = 3.0
MIN_TRADES_FOR_SHARPE = 3
EXCLUDED_EQUITY_SUFFIXES = {"BE", "BZ", "SM", "ST", "SG", "TB", "GB"}


@dataclass(frozen=True)
class BacktestResult:
    summary: dict[str, Any]
    stock_stats: pd.DataFrame
    trades: pd.DataFrame
    open_positions: pd.DataFrame


def run_buy_sell_backtest(
    config: dict[str, Any],
    storage: Storage,
    exchange: str = "NSE",
    as_of_date: Any | None = None,
) -> BacktestResult:
    return run_buy_sell_backtest_for_symbols(config, storage, exchange=exchange, symbols=None, as_of_date=as_of_date)


def run_buy_sell_backtest_for_symbols(
    config: dict[str, Any],
    storage: Storage,
    exchange: str = "NSE",
    symbols: set[str] | None = None,
    in_sample_years: int = DEFAULT_IN_SAMPLE_YEARS,
    out_of_sample_months: int = DEFAULT_OUT_OF_SAMPLE_MONTHS,
    min_sharpe_ratio: float = DEFAULT_MIN_SHARPE_RATIO,
    equity_only: bool = True,
    as_of_date: Any | None = None,
) -> BacktestResult:
    as_of_ts = _parse_as_of_date(as_of_date)
    candle_dir = storage.candles_dir / exchange / "1D"
    if not candle_dir.exists():
        return BacktestResult(
            _empty_summary(
                exchange,
                in_sample_years,
                out_of_sample_months,
                min_sharpe_ratio,
                requested_as_of_date=as_of_ts,
            ),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    strategy_cfg = config.get("strategy", {})
    scan_timeframe = config.get("data", {}).get("scan_timeframe", "1W")
    weekly_anchor = strategy_cfg.get("weekly_anchor", "W-FRI")
    use_completed_weeks_only = bool(strategy_cfg.get("use_completed_weeks_only", True))

    instruments = storage.load_instruments()
    name_map = _instrument_name_map(instruments, exchange)
    selected_symbols = {str(symbol).upper() for symbol in symbols} if symbols else None
    explicit_symbol_scope = selected_symbols is not None
    requested_symbol_count = len(selected_symbols) if selected_symbols is not None else 0
    eligible_symbols = _eligible_equity_symbols(instruments, exchange) if equity_only else set()
    if eligible_symbols:
        selected_symbols = selected_symbols & eligible_symbols if selected_symbols is not None else eligible_symbols
    trade_frames: list[pd.DataFrame] = []
    open_position_rows: list[dict[str, Any]] = []
    latest_candle_dates: list[pd.Timestamp] = []
    symbols_processed = 0
    symbols_with_closed_trades = 0

    for candle_path in sorted(candle_dir.glob("*.csv")):
        symbol = candle_path.stem
        if selected_symbols is not None and symbol.upper() not in selected_symbols:
            continue
        daily = storage.load_candles(exchange, symbol, "1D")
        if daily.empty:
            continue
        daily = _truncate_daily_to_as_of(daily, as_of_ts)
        if daily.empty:
            continue

        if "date" in daily.columns:
            latest_date = pd.to_datetime(daily["date"], errors="coerce").dropna().max()
            if pd.notna(latest_date):
                latest_candle_dates.append(pd.Timestamp(latest_date))

        symbols_processed += 1
        strategy_input = daily
        if scan_timeframe == "1W":
            strategy_input = resample_daily_to_weekly(daily, weekly_anchor, use_completed_weeks_only)

        strategy_output = run_weekly_buy_sell(strategy_input, config)
        trades, open_position = closed_trades_from_strategy(
            strategy_output,
            exchange=exchange,
            symbol=symbol,
            name=name_map.get(symbol, symbol),
        )
        if not trades.empty:
            trade_frames.append(trades)
            symbols_with_closed_trades += 1
        if open_position:
            open_position_rows.append(open_position)

    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else _empty_trades_frame()
    backtest_end_date = max(latest_candle_dates) if latest_candle_dates else _latest_trade_date(trades)
    trades, in_sample_start, out_of_sample_start = _apply_backtest_windows(
        trades,
        backtest_end_date,
        in_sample_years=in_sample_years,
        out_of_sample_months=out_of_sample_months,
    )
    open_positions = pd.DataFrame(open_position_rows)
    stock_stats = stock_level_stats(trades, min_sharpe_ratio=min_sharpe_ratio)
    summary = overall_summary(
        trades,
        open_positions,
        exchange=exchange,
        symbols_processed=symbols_processed,
        symbols_with_closed_trades=symbols_with_closed_trades,
        in_sample_years=in_sample_years,
        out_of_sample_months=out_of_sample_months,
        min_sharpe_ratio=min_sharpe_ratio,
        backtest_end_date=backtest_end_date,
        in_sample_start_date=in_sample_start,
        out_of_sample_start_date=out_of_sample_start,
        requested_as_of_date=as_of_ts,
    )
    if explicit_symbol_scope:
        summary["symbol_scope"] = "fresh_weekly_signals"
        summary["symbols_requested"] = requested_symbol_count
    else:
        summary["symbol_scope"] = "full_universe"
        summary["symbols_requested"] = 0
    summary["equity_only"] = equity_only
    summary["requested_as_of_date"] = _date_text(as_of_ts)

    return BacktestResult(summary, stock_stats, trades, open_positions)


def closed_trades_from_strategy(
    strategy_output: pd.DataFrame,
    exchange: str,
    symbol: str,
    name: str = "",
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    if strategy_output.empty:
        return _empty_trades_frame(), None

    frame = strategy_output.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.sort_values("date")

    active_buy: dict[str, Any] | None = None
    trade_rows: list[dict[str, Any]] = []

    for _, row in frame.iterrows():
        if bool(row.get("final_buy", False)):
            active_buy = {
                "exchange": exchange,
                "symbol": symbol,
                "name": name,
                "buy_date": row["date"],
                "buy_close": float(row["close"]),
            }
        elif bool(row.get("final_sell", False)) and active_buy is not None:
            sell_close = float(row["close"])
            buy_close = float(active_buy["buy_close"])
            return_pct = ((sell_close - buy_close) / buy_close) * 100
            buy_date = pd.to_datetime(active_buy["buy_date"])
            sell_date = pd.to_datetime(row["date"])
            max_gain = _max_gain_before_sell(frame, buy_date, sell_date, buy_close)
            trade_rows.append(
                {
                    **active_buy,
                    "sell_date": sell_date,
                    "sell_close": sell_close,
                    "return_pct": return_pct,
                    "outcome": _outcome(return_pct),
                    **max_gain,
                    "hit_5pct_before_sell": max_gain["max_gain_before_sell_pct"] >= 5,
                    "hit_10pct_before_sell": max_gain["max_gain_before_sell_pct"] >= 10,
                    "hit_15pct_before_sell": max_gain["max_gain_before_sell_pct"] >= 15,
                    "hit_20pct_before_sell": max_gain["max_gain_before_sell_pct"] >= 20,
                    "holding_days": int((sell_date - buy_date).days),
                    "holding_weeks": round((sell_date - buy_date).days / 7, 2),
                }
            )
            active_buy = None

    open_position = None
    if active_buy is not None:
        latest = frame.iloc[-1]
        latest_close = float(latest["close"])
        buy_close = float(active_buy["buy_close"])
        open_position = {
            **active_buy,
            "latest_date": latest["date"],
            "latest_close": latest_close,
            "open_return_pct": ((latest_close - buy_close) / buy_close) * 100,
        }

    return pd.DataFrame(trade_rows), open_position


def stock_level_stats(trades: pd.DataFrame, min_sharpe_ratio: float = DEFAULT_MIN_SHARPE_RATIO) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "exchange",
                "symbol",
                "name",
                "closed_trades",
                "wins",
                "losses",
                "breakeven",
                "win_rate_pct",
                "loss_rate_pct",
                "avg_return_pct",
                "median_return_pct",
                "best_return_pct",
                "worst_return_pct",
                "trades_went_up_before_sell",
                "went_up_before_sell_rate_pct",
                "avg_max_gain_before_sell_pct",
                "median_max_gain_before_sell_pct",
                "best_max_gain_before_sell_pct",
                "hit_5pct_before_sell_rate_pct",
                "hit_10pct_before_sell_rate_pct",
                "hit_15pct_before_sell_rate_pct",
                "hit_20pct_before_sell_rate_pct",
                "total_return_pct",
                "sharpe_ratio",
                "max_drawdown_pct",
                "max_drawdown_duration_trades",
                "max_drawdown_duration_days",
                "in_sample_trades",
                "in_sample_sharpe_ratio",
                "in_sample_max_drawdown_pct",
                "in_sample_max_drawdown_duration_days",
                "out_of_sample_trades",
                "out_of_sample_sharpe_ratio",
                "out_of_sample_max_drawdown_pct",
                "out_of_sample_max_drawdown_duration_days",
                "meets_sharpe_gate",
            ]
        )

    trades = trades.copy()
    if "max_gain_before_sell_pct" not in trades.columns:
        trades["max_gain_before_sell_pct"] = 0.0

    grouped = trades.groupby(["exchange", "symbol", "name"], dropna=False)
    stats = grouped["return_pct"].agg(
        closed_trades="count",
        avg_return_pct="mean",
        median_return_pct="median",
        best_return_pct="max",
        worst_return_pct="min",
    ).reset_index()
    max_gain_stats = grouped["max_gain_before_sell_pct"].agg(
        avg_max_gain_before_sell_pct="mean",
        median_max_gain_before_sell_pct="median",
        best_max_gain_before_sell_pct="max",
    ).reset_index()
    wins = grouped.apply(lambda frame: int((frame["return_pct"] > 0).sum()), include_groups=False).rename("wins")
    losses = grouped.apply(lambda frame: int((frame["return_pct"] < 0).sum()), include_groups=False).rename("losses")
    breakeven = grouped.apply(lambda frame: int((frame["return_pct"] == 0).sum()), include_groups=False).rename("breakeven")
    went_up = grouped.apply(lambda frame: int((frame["max_gain_before_sell_pct"] > 0).sum()), include_groups=False).rename(
        "trades_went_up_before_sell"
    )
    hit_5 = grouped.apply(lambda frame: int((frame["max_gain_before_sell_pct"] >= 5).sum()), include_groups=False).rename(
        "hit_5pct_before_sell"
    )
    hit_10 = grouped.apply(lambda frame: int((frame["max_gain_before_sell_pct"] >= 10).sum()), include_groups=False).rename(
        "hit_10pct_before_sell"
    )
    hit_15 = grouped.apply(lambda frame: int((frame["max_gain_before_sell_pct"] >= 15).sum()), include_groups=False).rename(
        "hit_15pct_before_sell"
    )
    hit_20 = grouped.apply(lambda frame: int((frame["max_gain_before_sell_pct"] >= 20).sum()), include_groups=False).rename(
        "hit_20pct_before_sell"
    )
    stats = stats.merge(wins, on=["exchange", "symbol", "name"])
    stats = stats.merge(losses, on=["exchange", "symbol", "name"])
    stats = stats.merge(breakeven, on=["exchange", "symbol", "name"])
    stats = stats.merge(max_gain_stats, on=["exchange", "symbol", "name"])
    stats = stats.merge(went_up, on=["exchange", "symbol", "name"])
    stats = stats.merge(hit_5, on=["exchange", "symbol", "name"])
    stats = stats.merge(hit_10, on=["exchange", "symbol", "name"])
    stats = stats.merge(hit_15, on=["exchange", "symbol", "name"])
    stats = stats.merge(hit_20, on=["exchange", "symbol", "name"])
    stats["win_rate_pct"] = (stats["wins"] / stats["closed_trades"]) * 100
    stats["loss_rate_pct"] = (stats["losses"] / stats["closed_trades"]) * 100
    stats["went_up_before_sell_rate_pct"] = (stats["trades_went_up_before_sell"] / stats["closed_trades"]) * 100
    stats["hit_5pct_before_sell_rate_pct"] = (stats["hit_5pct_before_sell"] / stats["closed_trades"]) * 100
    stats["hit_10pct_before_sell_rate_pct"] = (stats["hit_10pct_before_sell"] / stats["closed_trades"]) * 100
    stats["hit_15pct_before_sell_rate_pct"] = (stats["hit_15pct_before_sell"] / stats["closed_trades"]) * 100
    stats["hit_20pct_before_sell_rate_pct"] = (stats["hit_20pct_before_sell"] / stats["closed_trades"]) * 100
    quality_rows: list[dict[str, Any]] = []
    for (exchange, symbol, name), frame in grouped:
        all_metrics = _trade_quality_metrics(frame)
        in_sample_metrics = _trade_quality_metrics(_sample_trades(frame, "in_sample"))
        out_sample_metrics = _trade_quality_metrics(_sample_trades(frame, "out_of_sample"))
        quality_rows.append(
            {
                "exchange": exchange,
                "symbol": symbol,
                "name": name,
                "total_return_pct": all_metrics["total_return_pct"],
                "sharpe_ratio": all_metrics["sharpe_ratio"],
                "max_drawdown_pct": all_metrics["max_drawdown_pct"],
                "max_drawdown_duration_trades": all_metrics["max_drawdown_duration_trades"],
                "max_drawdown_duration_days": all_metrics["max_drawdown_duration_days"],
                "in_sample_trades": in_sample_metrics["closed_trades"],
                "in_sample_sharpe_ratio": in_sample_metrics["sharpe_ratio"],
                "in_sample_max_drawdown_pct": in_sample_metrics["max_drawdown_pct"],
                "in_sample_max_drawdown_duration_days": in_sample_metrics["max_drawdown_duration_days"],
                "out_of_sample_trades": out_sample_metrics["closed_trades"],
                "out_of_sample_sharpe_ratio": out_sample_metrics["sharpe_ratio"],
                "out_of_sample_max_drawdown_pct": out_sample_metrics["max_drawdown_pct"],
                "out_of_sample_max_drawdown_duration_days": out_sample_metrics["max_drawdown_duration_days"],
                "meets_sharpe_gate": (
                    in_sample_metrics["closed_trades"] >= MIN_TRADES_FOR_SHARPE
                    and out_sample_metrics["closed_trades"] >= MIN_TRADES_FOR_SHARPE
                    and in_sample_metrics["sharpe_ratio"] > min_sharpe_ratio
                    and out_sample_metrics["sharpe_ratio"] > min_sharpe_ratio
                ),
            }
        )
    if quality_rows:
        stats = stats.merge(pd.DataFrame(quality_rows), on=["exchange", "symbol", "name"], how="left")
    return stats[
        [
            "exchange",
            "symbol",
            "name",
            "closed_trades",
            "wins",
            "losses",
            "breakeven",
            "win_rate_pct",
            "loss_rate_pct",
            "avg_return_pct",
            "median_return_pct",
            "best_return_pct",
            "worst_return_pct",
            "trades_went_up_before_sell",
            "went_up_before_sell_rate_pct",
            "avg_max_gain_before_sell_pct",
            "median_max_gain_before_sell_pct",
            "best_max_gain_before_sell_pct",
            "hit_5pct_before_sell_rate_pct",
            "hit_10pct_before_sell_rate_pct",
            "hit_15pct_before_sell_rate_pct",
            "hit_20pct_before_sell_rate_pct",
            "total_return_pct",
            "sharpe_ratio",
            "max_drawdown_pct",
            "max_drawdown_duration_trades",
            "max_drawdown_duration_days",
            "in_sample_trades",
            "in_sample_sharpe_ratio",
            "in_sample_max_drawdown_pct",
            "in_sample_max_drawdown_duration_days",
            "out_of_sample_trades",
            "out_of_sample_sharpe_ratio",
            "out_of_sample_max_drawdown_pct",
            "out_of_sample_max_drawdown_duration_days",
            "meets_sharpe_gate",
        ]
    ].sort_values(
        [
            "meets_sharpe_gate",
            "out_of_sample_trades",
            "out_of_sample_sharpe_ratio",
            "in_sample_sharpe_ratio",
            "max_drawdown_pct",
            "closed_trades",
        ],
        ascending=[False, False, False, False, False, False],
    )


def overall_summary(
    trades: pd.DataFrame,
    open_positions: pd.DataFrame,
    exchange: str,
    symbols_processed: int,
    symbols_with_closed_trades: int,
    in_sample_years: int = DEFAULT_IN_SAMPLE_YEARS,
    out_of_sample_months: int = DEFAULT_OUT_OF_SAMPLE_MONTHS,
    min_sharpe_ratio: float = DEFAULT_MIN_SHARPE_RATIO,
    backtest_end_date: pd.Timestamp | None = None,
    in_sample_start_date: pd.Timestamp | None = None,
    out_of_sample_start_date: pd.Timestamp | None = None,
    requested_as_of_date: pd.Timestamp | None = None,
) -> dict[str, Any]:
    closed_trades = len(trades)
    wins = int((trades["return_pct"] > 0).sum()) if not trades.empty else 0
    losses = int((trades["return_pct"] < 0).sum()) if not trades.empty else 0
    breakeven = int((trades["return_pct"] == 0).sum()) if not trades.empty else 0
    max_gain = _max_gain_series(trades)
    went_up = int((max_gain > 0).sum()) if not trades.empty else 0

    all_metrics = _trade_quality_metrics(trades, aggregate_by_sell_date=True)
    in_sample_metrics = _trade_quality_metrics(_sample_trades(trades, "in_sample"), aggregate_by_sell_date=True)
    out_sample_metrics = _trade_quality_metrics(_sample_trades(trades, "out_of_sample"), aggregate_by_sell_date=True)
    sharpe_pass = all_metrics["sharpe_ratio"] > min_sharpe_ratio
    in_sample_pass = in_sample_metrics["sharpe_ratio"] > min_sharpe_ratio
    out_sample_pass = out_sample_metrics["sharpe_ratio"] > min_sharpe_ratio

    return {
        "exchange": exchange,
        "symbols_processed": symbols_processed,
        "symbols_with_closed_trades": symbols_with_closed_trades,
        "in_sample_years": in_sample_years,
        "out_of_sample_months": out_of_sample_months,
        "min_sharpe_ratio": min_sharpe_ratio,
        "requested_as_of_date": _date_text(requested_as_of_date),
        "backtest_start_date": _date_text(in_sample_start_date),
        "backtest_end_date": _date_text(backtest_end_date),
        "in_sample_start_date": _date_text(in_sample_start_date),
        "out_of_sample_start_date": _date_text(out_of_sample_start_date),
        "closed_trades": closed_trades,
        "winning_trades": wins,
        "losing_trades": losses,
        "breakeven_trades": breakeven,
        "open_positions": len(open_positions),
        "win_rate_pct": (wins / closed_trades * 100) if closed_trades else 0,
        "loss_rate_pct": (losses / closed_trades * 100) if closed_trades else 0,
        "avg_return_pct": float(trades["return_pct"].mean()) if closed_trades else 0,
        "median_return_pct": float(trades["return_pct"].median()) if closed_trades else 0,
        "best_return_pct": float(trades["return_pct"].max()) if closed_trades else 0,
        "worst_return_pct": float(trades["return_pct"].min()) if closed_trades else 0,
        "trades_went_up_before_sell": went_up,
        "went_up_before_sell_rate_pct": (went_up / closed_trades * 100) if closed_trades else 0,
        "avg_max_gain_before_sell_pct": float(max_gain.mean()) if closed_trades else 0,
        "median_max_gain_before_sell_pct": float(max_gain.median()) if closed_trades else 0,
        "hit_5pct_before_sell_rate_pct": _threshold_rate(trades, 5),
        "hit_10pct_before_sell_rate_pct": _threshold_rate(trades, 10),
        "hit_15pct_before_sell_rate_pct": _threshold_rate(trades, 15),
        "hit_20pct_before_sell_rate_pct": _threshold_rate(trades, 20),
        "total_return_pct": all_metrics["total_return_pct"],
        "sharpe_ratio": all_metrics["sharpe_ratio"],
        "sharpe_pass": sharpe_pass,
        "max_drawdown_pct": all_metrics["max_drawdown_pct"],
        "max_drawdown_duration_trades": all_metrics["max_drawdown_duration_trades"],
        "max_drawdown_duration_days": all_metrics["max_drawdown_duration_days"],
        "max_drawdown_peak_date": all_metrics["max_drawdown_peak_date"],
        "max_drawdown_trough_date": all_metrics["max_drawdown_trough_date"],
        "max_drawdown_recovery_date": all_metrics["max_drawdown_recovery_date"],
        "in_sample_closed_trades": in_sample_metrics["closed_trades"],
        "in_sample_total_return_pct": in_sample_metrics["total_return_pct"],
        "in_sample_sharpe_ratio": in_sample_metrics["sharpe_ratio"],
        "in_sample_sharpe_pass": in_sample_pass,
        "in_sample_max_drawdown_pct": in_sample_metrics["max_drawdown_pct"],
        "in_sample_max_drawdown_duration_days": in_sample_metrics["max_drawdown_duration_days"],
        "out_of_sample_closed_trades": out_sample_metrics["closed_trades"],
        "out_of_sample_total_return_pct": out_sample_metrics["total_return_pct"],
        "out_of_sample_sharpe_ratio": out_sample_metrics["sharpe_ratio"],
        "out_of_sample_sharpe_pass": out_sample_pass,
        "out_of_sample_max_drawdown_pct": out_sample_metrics["max_drawdown_pct"],
        "out_of_sample_max_drawdown_duration_days": out_sample_metrics["max_drawdown_duration_days"],
    }


def _apply_backtest_windows(
    trades: pd.DataFrame,
    backtest_end_date: pd.Timestamp | None,
    in_sample_years: int,
    out_of_sample_months: int,
) -> tuple[pd.DataFrame, pd.Timestamp | None, pd.Timestamp | None]:
    if backtest_end_date is None or pd.isna(backtest_end_date):
        return _with_empty_sample(trades), None, None

    end_date = pd.Timestamp(backtest_end_date).normalize()
    out_of_sample_months = max(0, int(out_of_sample_months))
    in_sample_years = max(1, int(in_sample_years))
    out_of_sample_start = end_date - pd.DateOffset(months=out_of_sample_months)
    in_sample_start = out_of_sample_start - pd.DateOffset(years=in_sample_years)

    if trades.empty or "sell_date" not in trades.columns:
        return _with_empty_sample(trades), in_sample_start, out_of_sample_start

    filtered = trades.copy()
    filtered["buy_date"] = pd.to_datetime(filtered["buy_date"], errors="coerce")
    filtered["sell_date"] = pd.to_datetime(filtered["sell_date"], errors="coerce")
    filtered = filtered[filtered["sell_date"].notna()]
    filtered = filtered[filtered["sell_date"] >= in_sample_start]
    if filtered.empty:
        return _with_empty_sample(filtered), in_sample_start, out_of_sample_start

    filtered["sample"] = "in_sample"
    filtered.loc[filtered["sell_date"] >= out_of_sample_start, "sample"] = "out_of_sample"
    filtered = filtered.sort_values(["sell_date", "symbol", "buy_date"]).reset_index(drop=True)
    return filtered, in_sample_start, out_of_sample_start


def _parse_as_of_date(value: Any | None) -> pd.Timestamp | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid as-of date: {value}")
    return pd.Timestamp(parsed).normalize()


def _truncate_daily_to_as_of(daily: pd.DataFrame, as_of_date: pd.Timestamp | None) -> pd.DataFrame:
    if as_of_date is None or daily.empty or "date" not in daily.columns:
        return daily
    truncated = daily.copy()
    truncated["date"] = pd.to_datetime(truncated["date"], errors="coerce")
    truncated = truncated[truncated["date"].notna()]
    truncated = truncated[truncated["date"] <= as_of_date]
    return truncated.sort_values("date").reset_index(drop=True)


def _with_empty_sample(trades: pd.DataFrame) -> pd.DataFrame:
    if "sample" in trades.columns:
        return trades
    enriched = trades.copy()
    enriched["sample"] = pd.Series(dtype="object")
    return enriched


def _sample_trades(trades: pd.DataFrame, sample: str) -> pd.DataFrame:
    if trades.empty or "sample" not in trades.columns:
        return trades.iloc[0:0].copy()
    return trades[trades["sample"].astype(str) == sample].copy()


def _trade_quality_metrics(trades: pd.DataFrame, aggregate_by_sell_date: bool = False) -> dict[str, Any]:
    if trades.empty or "return_pct" not in trades.columns:
        return _empty_quality_metrics()

    working = trades.copy()
    working["return_pct"] = pd.to_numeric(working["return_pct"], errors="coerce")
    working = working.dropna(subset=["return_pct"])
    if working.empty:
        return _empty_quality_metrics()
    closed_trade_count = int(len(working))

    if "sell_date" in working.columns:
        working["sell_date"] = pd.to_datetime(working["sell_date"], errors="coerce")
        working = working.sort_values(["sell_date", "symbol", "buy_date"], na_position="last")

    metric_frame = working
    if aggregate_by_sell_date and "sell_date" in working.columns:
        dated = working.dropna(subset=["sell_date"]).copy()
        if not dated.empty:
            dated["period_date"] = dated["sell_date"].dt.normalize()
            metric_frame = (
                dated.groupby("period_date", as_index=False)["return_pct"]
                .mean()
            )
            metric_frame["sell_date"] = metric_frame["period_date"]
            metric_frame["buy_date"] = metric_frame["period_date"]
            metric_frame["symbol"] = "PORTFOLIO"

    returns = metric_frame["return_pct"].astype(float)
    equity = (1.0 + returns / 100.0).cumprod()
    total_return_pct = float((equity.iloc[-1] - 1.0) * 100.0) if not equity.empty else 0.0
    drawdown = _trade_drawdown_metrics(metric_frame, equity)
    return {
        "closed_trades": closed_trade_count,
        "total_return_pct": total_return_pct,
        "sharpe_ratio": _trade_sharpe_ratio(returns),
        **drawdown,
    }


def _empty_quality_metrics() -> dict[str, Any]:
    return {
        "closed_trades": 0,
        "total_return_pct": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown_pct": 0.0,
        "max_drawdown_duration_trades": 0,
        "max_drawdown_duration_days": 0,
        "max_drawdown_peak_date": "",
        "max_drawdown_trough_date": "",
        "max_drawdown_recovery_date": "",
    }


def _trade_sharpe_ratio(returns_pct: pd.Series) -> float:
    returns = pd.to_numeric(returns_pct, errors="coerce").dropna() / 100.0
    if len(returns) < MIN_TRADES_FOR_SHARPE:
        return 0.0
    std = float(returns.std(ddof=1))
    if std <= 0:
        return 0.0
    return float((returns.mean() / std) * (len(returns) ** 0.5))


def _trade_drawdown_metrics(trades: pd.DataFrame, equity: pd.Series) -> dict[str, Any]:
    if trades.empty or equity.empty:
        return {
            "max_drawdown_pct": 0.0,
            "max_drawdown_duration_trades": 0,
            "max_drawdown_duration_days": 0,
            "max_drawdown_peak_date": "",
            "max_drawdown_trough_date": "",
            "max_drawdown_recovery_date": "",
        }

    equity = equity.reset_index(drop=True)
    running_peak = equity.cummax()
    drawdown_pct = ((equity / running_peak) - 1.0) * 100.0
    trough_index = int(drawdown_pct.idxmin())
    max_drawdown_pct = float(drawdown_pct.iloc[trough_index])
    peak_value = float(running_peak.iloc[trough_index])
    prior_peak_indexes = equity.iloc[: trough_index + 1][equity.iloc[: trough_index + 1] >= peak_value - 1e-12].index
    peak_index = int(prior_peak_indexes[-1]) if len(prior_peak_indexes) else trough_index

    recovery_index: int | None = None
    for index in range(trough_index + 1, len(equity)):
        if float(equity.iloc[index]) >= peak_value - 1e-12:
            recovery_index = index
            break

    duration = _longest_underwater_duration(trades, equity)
    return {
        "max_drawdown_pct": max_drawdown_pct,
        "max_drawdown_duration_trades": duration["trades"],
        "max_drawdown_duration_days": duration["days"],
        "max_drawdown_peak_date": _date_text(_trade_date_at(trades, peak_index)),
        "max_drawdown_trough_date": _date_text(_trade_date_at(trades, trough_index)),
        "max_drawdown_recovery_date": _date_text(_trade_date_at(trades, recovery_index)) if recovery_index is not None else "",
    }


def _longest_underwater_duration(trades: pd.DataFrame, equity: pd.Series) -> dict[str, int]:
    running_peak = float("-inf")
    peak_index = 0
    peak_date = _trade_date_at(trades, 0)
    underwater_start_index: int | None = None
    underwater_start_date: pd.Timestamp | None = None
    longest_trades = 0
    longest_days = 0

    for index, value in enumerate(equity.tolist()):
        current_date = _trade_date_at(trades, index)
        if value >= running_peak - 1e-12:
            if underwater_start_index is not None:
                longest_trades, longest_days = _max_duration(
                    longest_trades,
                    longest_days,
                    index - underwater_start_index,
                    _days_between(underwater_start_date, current_date),
                )
                underwater_start_index = None
                underwater_start_date = None
            running_peak = float(value)
            peak_index = index
            peak_date = current_date
        elif underwater_start_index is None:
            underwater_start_index = peak_index
            underwater_start_date = peak_date

    if underwater_start_index is not None:
        last_index = len(equity) - 1
        longest_trades, longest_days = _max_duration(
            longest_trades,
            longest_days,
            last_index - underwater_start_index,
            _days_between(underwater_start_date, _trade_date_at(trades, last_index)),
        )

    return {"trades": int(longest_trades), "days": int(longest_days)}


def _max_duration(
    current_trades: int,
    current_days: int,
    candidate_trades: int,
    candidate_days: int,
) -> tuple[int, int]:
    if candidate_days > current_days:
        return int(candidate_trades), int(candidate_days)
    if candidate_days == current_days and candidate_trades > current_trades:
        return int(candidate_trades), int(candidate_days)
    return int(current_trades), int(current_days)


def _trade_date_at(trades: pd.DataFrame, index: int | None) -> pd.Timestamp | None:
    if index is None or trades.empty:
        return None
    if "sell_date" not in trades.columns:
        return None
    try:
        value = trades.reset_index(drop=True).loc[index, "sell_date"]
    except (KeyError, IndexError, ValueError):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed)


def _days_between(start: pd.Timestamp | None, end: pd.Timestamp | None) -> int:
    if start is None or end is None or pd.isna(start) or pd.isna(end):
        return 0
    return max(0, int((pd.Timestamp(end) - pd.Timestamp(start)).days))


def _latest_trade_date(trades: pd.DataFrame) -> pd.Timestamp | None:
    if trades.empty or "sell_date" not in trades.columns:
        return None
    dates = pd.to_datetime(trades["sell_date"], errors="coerce").dropna()
    if dates.empty:
        return None
    return pd.Timestamp(dates.max())


def _date_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def save_backtest_outputs(
    result: BacktestResult,
    output_dir: Path,
    run_id: str = "latest",
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{run_id}_summary.csv"
    stock_stats_path = output_dir / f"{run_id}_stock_stats.csv"
    trades_path = output_dir / f"{run_id}_trades.csv"
    open_positions_path = output_dir / f"{run_id}_open_positions.csv"

    pd.DataFrame([result.summary]).to_csv(summary_path, index=False)
    result.stock_stats.to_csv(stock_stats_path, index=False)
    result.trades.to_csv(trades_path, index=False)
    result.open_positions.to_csv(open_positions_path, index=False)

    return {
        "summary": summary_path,
        "stock_stats": stock_stats_path,
        "trades": trades_path,
        "open_positions": open_positions_path,
    }


def _instrument_name_map(instruments: pd.DataFrame, exchange: str) -> dict[str, str]:
    if instruments.empty or not {"exchange", "tradingsymbol", "name"}.issubset(instruments.columns):
        return {}
    frame = instruments[instruments["exchange"].astype(str).str.upper() == exchange.upper()].copy()
    return dict(zip(frame["tradingsymbol"].astype(str), frame["name"].fillna("").astype(str)))


def _eligible_equity_symbols(instruments: pd.DataFrame, exchange: str) -> set[str]:
    required = {"exchange", "segment", "tradingsymbol", "instrument_type", "lot_size", "name"}
    if instruments.empty or not required.issubset(instruments.columns):
        return set()
    frame = instruments[
        (instruments["exchange"].astype(str).str.upper() == exchange.upper())
        & (instruments["segment"].astype(str).str.upper() == exchange.upper())
        & (instruments["instrument_type"].astype(str).str.upper() == "EQ")
    ].copy()
    if frame.empty:
        return set()
    frame["lot_size"] = pd.to_numeric(frame["lot_size"], errors="coerce")
    frame = frame[frame["lot_size"].fillna(1) == 1]
    symbols: set[str] = set()
    for _, row in frame.iterrows():
        symbol = str(row.get("tradingsymbol", "")).strip().upper()
        name = str(row.get("name", "")).strip().upper()
        if _is_ordinary_equity_symbol(symbol, name):
            symbols.add(symbol)
    return symbols


def _is_ordinary_equity_symbol(symbol: str, name: str) -> bool:
    if not symbol or " " in symbol:
        return False
    if symbol[0].isdigit() and "-" in symbol:
        return False
    suffix = symbol.rsplit("-", 1)[-1] if "-" in symbol else ""
    if suffix in EXCLUDED_EQUITY_SUFFIXES:
        return False
    if symbol.endswith("ETF") or symbol.endswith("BEES"):
        return False
    excluded_name_markers = (" ETF", "ETF ", "EXCHANGE TRADED", "MUTUAL FUND", "TREASURY BILL")
    return not any(marker in f" {name} " for marker in excluded_name_markers)


def _outcome(return_pct: float) -> str:
    if return_pct > 0:
        return "GAIN"
    if return_pct < 0:
        return "LOSS"
    return "BREAKEVEN"


def _max_gain_before_sell(
    frame: pd.DataFrame,
    buy_date: pd.Timestamp,
    sell_date: pd.Timestamp,
    buy_close: float,
) -> dict[str, Any]:
    if "high" not in frame.columns:
        return {
            "max_high_before_sell": buy_close,
            "max_gain_before_sell_pct": 0.0,
            "max_gain_date": pd.NA,
        }

    active_window = frame[(frame["date"] > buy_date) & (frame["date"] <= sell_date)].copy()
    if active_window.empty:
        return {
            "max_high_before_sell": buy_close,
            "max_gain_before_sell_pct": 0.0,
            "max_gain_date": pd.NA,
        }

    high_values = pd.to_numeric(active_window["high"], errors="coerce")
    if high_values.dropna().empty:
        return {
            "max_high_before_sell": buy_close,
            "max_gain_before_sell_pct": 0.0,
            "max_gain_date": pd.NA,
        }

    max_index = high_values.idxmax()
    max_high = float(high_values.loc[max_index])
    return {
        "max_high_before_sell": max_high,
        "max_gain_before_sell_pct": ((max_high - buy_close) / buy_close) * 100,
        "max_gain_date": active_window.loc[max_index, "date"],
    }


def _threshold_rate(trades: pd.DataFrame, threshold: float) -> float:
    if trades.empty:
        return 0
    return float((_max_gain_series(trades) >= threshold).mean() * 100)


def _max_gain_series(trades: pd.DataFrame) -> pd.Series:
    if "max_gain_before_sell_pct" not in trades.columns:
        return pd.Series([0.0] * len(trades), index=trades.index)
    return pd.to_numeric(trades["max_gain_before_sell_pct"], errors="coerce").fillna(0.0)


def _empty_summary(
    exchange: str,
    in_sample_years: int = DEFAULT_IN_SAMPLE_YEARS,
    out_of_sample_months: int = DEFAULT_OUT_OF_SAMPLE_MONTHS,
    min_sharpe_ratio: float = DEFAULT_MIN_SHARPE_RATIO,
    requested_as_of_date: pd.Timestamp | None = None,
) -> dict[str, Any]:
    return overall_summary(
        pd.DataFrame(columns=["return_pct", "max_gain_before_sell_pct"]),
        pd.DataFrame(),
        exchange,
        0,
        0,
        in_sample_years=in_sample_years,
        out_of_sample_months=out_of_sample_months,
        min_sharpe_ratio=min_sharpe_ratio,
        requested_as_of_date=requested_as_of_date,
    )


def _empty_trades_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "exchange",
            "symbol",
            "name",
            "buy_date",
            "buy_close",
            "sell_date",
            "sell_close",
            "return_pct",
            "outcome",
            "max_high_before_sell",
            "max_gain_before_sell_pct",
            "max_gain_date",
            "hit_5pct_before_sell",
            "hit_10pct_before_sell",
            "hit_15pct_before_sell",
            "hit_20pct_before_sell",
            "holding_days",
            "holding_weeks",
            "sample",
        ]
    )
