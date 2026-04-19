from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from stock_screener.data.storage import Storage
from stock_screener.resample import resample_daily_to_weekly
from stock_screener.strategy.weekly_buy_sell import run_weekly_buy_sell


@dataclass(frozen=True)
class BacktestResult:
    summary: dict[str, Any]
    stock_stats: pd.DataFrame
    trades: pd.DataFrame
    open_positions: pd.DataFrame


def run_buy_sell_backtest(config: dict[str, Any], storage: Storage, exchange: str = "NSE") -> BacktestResult:
    candle_dir = storage.candles_dir / exchange / "1D"
    if not candle_dir.exists():
        return BacktestResult(_empty_summary(exchange), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    strategy_cfg = config.get("strategy", {})
    scan_timeframe = config.get("data", {}).get("scan_timeframe", "1W")
    weekly_anchor = strategy_cfg.get("weekly_anchor", "W-FRI")
    use_completed_weeks_only = bool(strategy_cfg.get("use_completed_weeks_only", True))

    instruments = storage.load_instruments()
    name_map = _instrument_name_map(instruments, exchange)
    trade_frames: list[pd.DataFrame] = []
    open_position_rows: list[dict[str, Any]] = []
    symbols_processed = 0
    symbols_with_closed_trades = 0

    for candle_path in sorted(candle_dir.glob("*.csv")):
        symbol = candle_path.stem
        daily = storage.load_candles(exchange, symbol, "1D")
        if daily.empty:
            continue

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
    open_positions = pd.DataFrame(open_position_rows)
    stock_stats = stock_level_stats(trades)
    summary = overall_summary(
        trades,
        open_positions,
        exchange=exchange,
        symbols_processed=symbols_processed,
        symbols_with_closed_trades=symbols_with_closed_trades,
    )

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


def stock_level_stats(trades: pd.DataFrame) -> pd.DataFrame:
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
        ]
    ].sort_values(
        ["closed_trades", "hit_10pct_before_sell_rate_pct", "avg_max_gain_before_sell_pct"],
        ascending=[False, False, False],
    )


def overall_summary(
    trades: pd.DataFrame,
    open_positions: pd.DataFrame,
    exchange: str,
    symbols_processed: int,
    symbols_with_closed_trades: int,
) -> dict[str, Any]:
    closed_trades = len(trades)
    wins = int((trades["return_pct"] > 0).sum()) if not trades.empty else 0
    losses = int((trades["return_pct"] < 0).sum()) if not trades.empty else 0
    breakeven = int((trades["return_pct"] == 0).sum()) if not trades.empty else 0
    max_gain = _max_gain_series(trades)
    went_up = int((max_gain > 0).sum()) if not trades.empty else 0

    return {
        "exchange": exchange,
        "symbols_processed": symbols_processed,
        "symbols_with_closed_trades": symbols_with_closed_trades,
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
    }


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


def _empty_summary(exchange: str) -> dict[str, Any]:
    return overall_summary(
        pd.DataFrame(columns=["return_pct", "max_gain_before_sell_pct"]),
        pd.DataFrame(),
        exchange,
        0,
        0,
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
        ]
    )
