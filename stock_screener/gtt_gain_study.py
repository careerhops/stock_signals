from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from stock_screener.data.storage import Storage
from stock_screener.resample import resample_daily_to_weekly
from stock_screener.strategy.weekly_buy_sell import run_weekly_buy_sell
from stock_screener.universe import build_universe


GTT_THRESHOLDS = (5, 10, 15, 20, 25, 30)


@dataclass(frozen=True)
class GttGainStudyResult:
    summary: dict[str, Any]
    stock_stats: pd.DataFrame
    pair_details: pd.DataFrame
    open_positions: pd.DataFrame


def run_gtt_gain_study(
    config: dict[str, Any],
    storage: Storage,
    exchange: str = "NSE",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> GttGainStudyResult:
    universe_rows = _kite_instruments_universe(storage, config, exchange)
    if not universe_rows:
        return GttGainStudyResult(_empty_summary(exchange), pd.DataFrame(), _empty_pair_details(), pd.DataFrame())

    strategy_cfg = config.get("strategy", {})
    weekly_anchor = strategy_cfg.get("weekly_anchor", "W-FRI")
    use_completed_weeks_only = bool(strategy_cfg.get("use_completed_weeks_only", True))

    instruments = storage.load_instruments()
    name_map = _instrument_name_map(instruments, exchange)
    pair_rows: list[dict[str, Any]] = []
    open_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    symbols_processed = 0

    _emit_progress(
        progress_callback,
        phase="Preparing latest Kite NSE universe",
        completed=0,
        total=len(universe_rows),
        current_symbol="",
        current_exchange=exchange,
    )

    for index, universe_row in enumerate(universe_rows, start=1):
        row_exchange = str(universe_row.get("exchange") or exchange).upper()
        symbol = str(universe_row.get("symbol") or "").upper()
        name = str(universe_row.get("name") or name_map.get(symbol, symbol))
        if not symbol:
            continue

        symbols_processed += 1
        _emit_progress(
            progress_callback,
            phase="Analyzing BUY-to-SELL daily highs",
            completed=index - 1,
            total=len(universe_rows),
            current_symbol=symbol,
            current_exchange=row_exchange,
        )
        daily = storage.load_candles(row_exchange, symbol, "1D")
        if daily.empty:
            context_rows.append(_latest_signal_context(pd.DataFrame(), row_exchange, symbol, name))
            _emit_progress(
                progress_callback,
                phase="Analyzing BUY-to-SELL daily highs",
                completed=index,
                total=len(universe_rows),
                current_symbol=symbol,
                current_exchange=row_exchange,
            )
            continue

        daily = _prepare_daily(daily)
        weekly = resample_daily_to_weekly(daily, weekly_anchor, use_completed_weeks_only)
        if weekly.empty:
            context_rows.append(_latest_signal_context(pd.DataFrame(), row_exchange, symbol, name))
            _emit_progress(
                progress_callback,
                phase="Analyzing BUY-to-SELL daily highs",
                completed=index,
                total=len(universe_rows),
                current_symbol=symbol,
                current_exchange=row_exchange,
            )
            continue

        strategy_output = run_weekly_buy_sell(weekly, config)
        context_rows.append(_latest_signal_context(strategy_output, row_exchange, symbol, name))
        pairs, open_position = build_symbol_gtt_pairs(
            daily=daily,
            strategy_output=strategy_output,
            exchange=row_exchange,
            symbol=symbol,
            name=name,
        )
        if not pairs.empty:
            pair_rows.extend(pairs.to_dict(orient="records"))
        if open_position:
            open_rows.append(open_position)
        _emit_progress(
            progress_callback,
            phase="Analyzing BUY-to-SELL daily highs",
            completed=index,
            total=len(universe_rows),
            current_symbol=symbol,
            current_exchange=row_exchange,
        )

    _emit_progress(
        progress_callback,
        phase="Aggregating GTT statistics",
        completed=len(universe_rows),
        total=len(universe_rows),
        current_symbol="",
        current_exchange=exchange,
    )
    pair_details = pd.DataFrame(pair_rows)
    open_positions = pd.DataFrame(open_rows)
    stock_stats = build_stock_gtt_stats(pair_details, pd.DataFrame(context_rows))
    summary = build_gtt_summary(pair_details, open_positions, exchange, symbols_processed)
    return GttGainStudyResult(summary, stock_stats, _sort_pair_details(pair_details), _sort_open_positions(open_positions))


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
    if universe.empty:
        return []

    universe = universe.copy()
    universe["symbol"] = universe["tradingsymbol"].astype(str).str.upper().str.strip()
    universe["exchange"] = universe["exchange"].astype(str).str.upper().str.strip()
    if "name" not in universe.columns:
        universe["name"] = universe["symbol"]
    universe["name"] = universe["name"].fillna("").astype(str).str.strip()
    universe["name"] = universe["name"].mask(universe["name"] == "", universe["symbol"])
    universe = universe[universe["symbol"] != ""]
    universe = universe.drop_duplicates(subset=["exchange", "symbol"], keep="last").sort_values(["exchange", "symbol"])
    return universe[["exchange", "symbol", "name"]].to_dict(orient="records")


def build_symbol_gtt_pairs(
    daily: pd.DataFrame,
    strategy_output: pd.DataFrame,
    exchange: str,
    symbol: str,
    name: str = "",
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    if daily.empty or strategy_output.empty:
        return _empty_pair_details(), None

    daily_frame = _prepare_daily(daily)
    strategy = strategy_output.copy()
    strategy["date"] = pd.to_datetime(strategy["date"], errors="coerce")
    strategy = strategy.sort_values("date").reset_index(drop=True)

    active_buy: dict[str, Any] | None = None
    pair_rows: list[dict[str, Any]] = []

    for _, row in strategy.iterrows():
        if bool(row.get("final_buy", False)):
            active_buy = {
                "exchange": exchange,
                "symbol": symbol,
                "name": name,
                "buy_date": row["date"],
                "buy_close": float(row["close"]),
            }
        elif bool(row.get("final_sell", False)) and active_buy is not None:
            buy_close = float(active_buy["buy_close"])
            sell_close = float(row["close"])
            sell_date = pd.to_datetime(row["date"])
            buy_date = pd.to_datetime(active_buy["buy_date"])
            max_gain = max_gain_between_dates(daily_frame, buy_date, sell_date, buy_close)
            pair_rows.append(
                {
                    **active_buy,
                    "sell_date": sell_date,
                    "sell_close": sell_close,
                    "buy_to_sell_return_pct": ((sell_close - buy_close) / buy_close) * 100,
                    **max_gain,
                    **_threshold_flags(max_gain["max_gain_pct"]),
                }
            )
            active_buy = None

    open_position = None
    if active_buy is not None:
        open_position = _open_position_row(daily_frame, active_buy)

    return pd.DataFrame(pair_rows), open_position


def max_gain_between_dates(
    daily: pd.DataFrame,
    buy_date: pd.Timestamp,
    sell_date: pd.Timestamp,
    buy_close: float,
) -> dict[str, Any]:
    frame = _prepare_daily(daily)
    window = frame[(frame["date"] > buy_date) & (frame["date"] < sell_date)].copy()
    if window.empty:
        return {
            "valid_daily_window": False,
            "highest_price_between_buy_sell": pd.NA,
            "highest_price_date": pd.NA,
            "max_gain_pct": pd.NA,
            "went_above_buy_price": pd.NA,
        }

    highs = pd.to_numeric(window["high"], errors="coerce")
    if highs.dropna().empty:
        return {
            "valid_daily_window": False,
            "highest_price_between_buy_sell": pd.NA,
            "highest_price_date": pd.NA,
            "max_gain_pct": pd.NA,
            "went_above_buy_price": pd.NA,
        }

    max_index = highs.idxmax()
    highest_price = float(highs.loc[max_index])
    max_gain_pct = ((highest_price - buy_close) / buy_close) * 100
    return {
        "valid_daily_window": True,
        "highest_price_between_buy_sell": highest_price,
        "highest_price_date": window.loc[max_index, "date"],
        "max_gain_pct": max_gain_pct,
        "went_above_buy_price": max_gain_pct > 0,
    }


def build_stock_gtt_stats(pair_details: pd.DataFrame, latest_context: pd.DataFrame | None = None) -> pd.DataFrame:
    if pair_details.empty:
        return _merge_latest_context(_empty_stock_stats(), latest_context)

    frame = pair_details.copy()
    frame["valid_daily_window"] = _truthy(frame.get("valid_daily_window", pd.Series(False, index=frame.index)))
    if "max_gain_pct" not in frame.columns:
        frame["max_gain_pct"] = pd.NA
    frame["max_gain_pct"] = pd.to_numeric(frame["max_gain_pct"], errors="coerce")

    rows: list[dict[str, Any]] = []
    for (exchange, symbol, name), group in frame.groupby(["exchange", "symbol", "name"], dropna=False):
        valid = group[group["valid_daily_window"] & group["max_gain_pct"].notna()]
        valid_pairs = len(valid)
        closed_pairs = len(group)
        max_gain = valid["max_gain_pct"]
        row = {
            "exchange": exchange,
            "symbol": symbol,
            "name": name,
            "closed_pairs": closed_pairs,
            "valid_pairs": valid_pairs,
            "pairs_without_daily_window": closed_pairs - valid_pairs,
            "times_went_above_buy_price": int((max_gain > 0).sum()) if valid_pairs else 0,
            "went_above_buy_price_rate_pct": _rate(max_gain > 0) if valid_pairs else 0,
            "median_max_gain_pct": float(max_gain.median()) if valid_pairs else pd.NA,
            "avg_max_gain_pct": float(max_gain.mean()) if valid_pairs else pd.NA,
            "best_max_gain_pct": float(max_gain.max()) if valid_pairs else pd.NA,
            "low_sample": valid_pairs < 3,
            "suggested_conservative_gtt_pct": _suggested_target(max_gain, "conservative"),
            "suggested_moderate_gtt_pct": _suggested_target(max_gain, "moderate"),
        }
        for threshold in GTT_THRESHOLDS:
            hits = int((max_gain >= threshold).sum()) if valid_pairs else 0
            row[f"hit_{threshold}pct_count"] = hits
            row[f"hit_{threshold}pct_rate_pct"] = (hits / valid_pairs * 100) if valid_pairs else 0
        rows.append(row)

    stats = pd.DataFrame(rows)
    stats = _merge_latest_context(stats, latest_context)
    return stats.sort_values(
        ["valid_pairs", "hit_10pct_rate_pct", "median_max_gain_pct"],
        ascending=[False, False, False],
    )


def build_gtt_summary(
    pair_details: pd.DataFrame,
    open_positions: pd.DataFrame,
    exchange: str,
    symbols_processed: int,
) -> dict[str, Any]:
    closed_pairs = len(pair_details)
    if pair_details.empty:
        valid = pd.DataFrame()
    else:
        frame = pair_details.copy()
        frame["valid_daily_window"] = _truthy(frame.get("valid_daily_window", pd.Series(False, index=frame.index)))
        if "max_gain_pct" not in frame.columns:
            frame["max_gain_pct"] = pd.NA
        frame["max_gain_pct"] = pd.to_numeric(frame["max_gain_pct"], errors="coerce")
        valid = frame[frame["valid_daily_window"] & frame["max_gain_pct"].notna()]

    max_gain = valid["max_gain_pct"] if not valid.empty else pd.Series(dtype=float)
    summary = {
        "exchange": exchange,
        "symbols_processed": symbols_processed,
        "closed_pairs": closed_pairs,
        "valid_pairs": len(valid),
        "pairs_without_daily_window": closed_pairs - len(valid),
        "open_buy_positions": len(open_positions),
        "overall_median_max_gain_pct": float(max_gain.median()) if len(valid) else 0,
        "overall_avg_max_gain_pct": float(max_gain.mean()) if len(valid) else 0,
        "pairs_went_above_buy_price": int((max_gain > 0).sum()) if len(valid) else 0,
        "went_above_buy_price_rate_pct": _rate(max_gain > 0) if len(valid) else 0,
    }
    for threshold in GTT_THRESHOLDS:
        summary[f"hit_{threshold}pct_count"] = int((max_gain >= threshold).sum()) if len(valid) else 0
        summary[f"hit_{threshold}pct_rate_pct"] = _rate(max_gain >= threshold) if len(valid) else 0
    return summary


def save_gtt_gain_outputs(result: GttGainStudyResult, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "latest_summary.csv",
        "stock_stats": output_dir / "latest_stock_gtt_stats.csv",
        "pair_details": output_dir / "latest_pair_details.csv",
        "open_positions": output_dir / "latest_open_positions.csv",
    }
    pd.DataFrame([result.summary]).to_csv(paths["summary"], index=False)
    result.stock_stats.to_csv(paths["stock_stats"], index=False)
    result.pair_details.to_csv(paths["pair_details"], index=False)
    result.open_positions.to_csv(paths["open_positions"], index=False)
    return paths


def load_gtt_gain_outputs(output_dir: Path) -> GttGainStudyResult:
    summary_frame = _read_csv(output_dir / "latest_summary.csv")
    return GttGainStudyResult(
        summary=summary_frame.iloc[0].to_dict() if not summary_frame.empty else {},
        stock_stats=_read_csv(output_dir / "latest_stock_gtt_stats.csv"),
        pair_details=_read_csv(output_dir / "latest_pair_details.csv"),
        open_positions=_read_csv(output_dir / "latest_open_positions.csv"),
    )


def _open_position_row(daily: pd.DataFrame, active_buy: dict[str, Any]) -> dict[str, Any]:
    buy_date = pd.to_datetime(active_buy["buy_date"])
    buy_close = float(active_buy["buy_close"])
    window = daily[daily["date"] > buy_date].copy()
    latest = daily.iloc[-1] if not daily.empty else pd.Series(dtype=object)
    row = {
        **active_buy,
        "latest_date": latest.get("date", pd.NA),
        "latest_close": float(latest.get("close", buy_close)) if pd.notna(latest.get("close", pd.NA)) else pd.NA,
    }
    if window.empty or pd.to_numeric(window["high"], errors="coerce").dropna().empty:
        return {
            **row,
            "highest_price_since_buy": pd.NA,
            "highest_price_date": pd.NA,
            "open_max_gain_pct": pd.NA,
        }

    highs = pd.to_numeric(window["high"], errors="coerce")
    max_index = highs.idxmax()
    highest_price = float(highs.loc[max_index])
    return {
        **row,
        "highest_price_since_buy": highest_price,
        "highest_price_date": window.loc[max_index, "date"],
        "open_max_gain_pct": ((highest_price - buy_close) / buy_close) * 100,
    }


def _latest_signal_context(strategy_output: pd.DataFrame, exchange: str, symbol: str, name: str) -> dict[str, Any]:
    if strategy_output.empty:
        return {
            "exchange": exchange,
            "symbol": symbol,
            "name": name,
            "latest_week_date": pd.NA,
            "latest_close": pd.NA,
            "ema_20": pd.NA,
            "ema_50": pd.NA,
            "close_above_ema20": False,
            "ema20_above_ema50": False,
            "trend_confirmation": False,
            "latest_week_signal": "NONE",
            "latest_signal": "NONE",
            "latest_signal_date": pd.NA,
            "is_latest_signal_buy": False,
        }

    frame = strategy_output.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.sort_values("date").reset_index(drop=True)
    latest = frame.iloc[-1]
    signal_rows = frame[frame.get("signal", "NONE").astype(str).isin(["BUY", "SELL"])]
    latest_signal_row = signal_rows.iloc[-1] if not signal_rows.empty else None

    latest_close = _float_or_na(latest.get("close"))
    ema_20 = _float_or_na(latest.get("ema_20"))
    ema_50 = _float_or_na(latest.get("ema_50"))
    close_above_ema20 = _is_number(latest_close) and _is_number(ema_20) and float(latest_close) > float(ema_20)
    ema20_above_ema50 = _is_number(ema_20) and _is_number(ema_50) and float(ema_20) > float(ema_50)
    latest_signal = str(latest_signal_row.get("signal", "NONE")) if latest_signal_row is not None else "NONE"

    return {
        "exchange": exchange,
        "symbol": symbol,
        "name": name,
        "latest_week_date": latest.get("date", pd.NA),
        "latest_close": latest_close,
        "ema_20": ema_20,
        "ema_50": ema_50,
        "close_above_ema20": close_above_ema20,
        "ema20_above_ema50": ema20_above_ema50,
        "trend_confirmation": close_above_ema20 and ema20_above_ema50,
        "latest_week_signal": str(latest.get("signal", "NONE")),
        "latest_signal": latest_signal,
        "latest_signal_date": latest_signal_row.get("date", pd.NA) if latest_signal_row is not None else pd.NA,
        "is_latest_signal_buy": latest_signal == "BUY",
    }


def _merge_latest_context(stats: pd.DataFrame, latest_context: pd.DataFrame | None) -> pd.DataFrame:
    stats = _ensure_stock_stats_columns(stats)
    if latest_context is None or latest_context.empty:
        return stats

    context = latest_context.copy()
    context = context.drop_duplicates(subset=["exchange", "symbol"], keep="last")
    stat_columns = ["exchange", "symbol"] + _metric_columns()
    merged = context.merge(
        stats[[column for column in stat_columns if column in stats.columns]],
        on=["exchange", "symbol"],
        how="left",
    )
    for column in _count_columns():
        merged[column] = pd.to_numeric(merged.get(column), errors="coerce").fillna(0).astype(int)
    for column in _rate_columns():
        merged[column] = pd.to_numeric(merged.get(column), errors="coerce").fillna(0.0)
    for column in ("median_max_gain_pct", "avg_max_gain_pct", "best_max_gain_pct"):
        if column not in merged.columns:
            merged[column] = pd.NA
    merged["low_sample"] = pd.to_numeric(merged.get("valid_pairs"), errors="coerce").fillna(0).astype(int) < 3
    for column in ("suggested_conservative_gtt_pct", "suggested_moderate_gtt_pct"):
        if column not in merged.columns:
            merged[column] = pd.NA
    return merged[_stock_stats_columns()]


def _ensure_stock_stats_columns(stats: pd.DataFrame) -> pd.DataFrame:
    frame = stats.copy()
    for column in _stock_stats_columns():
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[_stock_stats_columns()]


def _stock_stats_columns() -> list[str]:
    return [
        "exchange",
        "symbol",
        "name",
        "latest_week_date",
        "latest_close",
        "ema_20",
        "ema_50",
        "close_above_ema20",
        "ema20_above_ema50",
        "trend_confirmation",
        "latest_week_signal",
        "latest_signal",
        "latest_signal_date",
        "is_latest_signal_buy",
        "closed_pairs",
        "valid_pairs",
        "pairs_without_daily_window",
        "times_went_above_buy_price",
        "went_above_buy_price_rate_pct",
        "median_max_gain_pct",
        "avg_max_gain_pct",
        "best_max_gain_pct",
        "hit_5pct_count",
        "hit_5pct_rate_pct",
        "hit_10pct_count",
        "hit_10pct_rate_pct",
        "hit_15pct_count",
        "hit_15pct_rate_pct",
        "hit_20pct_count",
        "hit_20pct_rate_pct",
        "hit_25pct_count",
        "hit_25pct_rate_pct",
        "hit_30pct_count",
        "hit_30pct_rate_pct",
        "low_sample",
        "suggested_conservative_gtt_pct",
        "suggested_moderate_gtt_pct",
    ]


def _count_columns() -> list[str]:
    return [
        "closed_pairs",
        "valid_pairs",
        "pairs_without_daily_window",
        "times_went_above_buy_price",
        "hit_5pct_count",
        "hit_10pct_count",
        "hit_15pct_count",
        "hit_20pct_count",
        "hit_25pct_count",
        "hit_30pct_count",
    ]


def _rate_columns() -> list[str]:
    return [
        "went_above_buy_price_rate_pct",
        "hit_5pct_rate_pct",
        "hit_10pct_rate_pct",
        "hit_15pct_rate_pct",
        "hit_20pct_rate_pct",
        "hit_25pct_rate_pct",
        "hit_30pct_rate_pct",
    ]


def _metric_columns() -> list[str]:
    return [
        "closed_pairs",
        "valid_pairs",
        "pairs_without_daily_window",
        "times_went_above_buy_price",
        "went_above_buy_price_rate_pct",
        "median_max_gain_pct",
        "avg_max_gain_pct",
        "best_max_gain_pct",
        "hit_5pct_count",
        "hit_5pct_rate_pct",
        "hit_10pct_count",
        "hit_10pct_rate_pct",
        "hit_15pct_count",
        "hit_15pct_rate_pct",
        "hit_20pct_count",
        "hit_20pct_rate_pct",
        "hit_25pct_count",
        "hit_25pct_rate_pct",
        "hit_30pct_count",
        "hit_30pct_rate_pct",
        "low_sample",
        "suggested_conservative_gtt_pct",
        "suggested_moderate_gtt_pct",
    ]


def _emit_progress(progress_callback: Callable[[dict[str, Any]], None] | None, **payload: Any) -> None:
    if progress_callback:
        progress_callback(payload)


def _threshold_flags(value: Any) -> dict[str, Any]:
    if pd.isna(value):
        return {f"hit_{threshold}pct": pd.NA for threshold in GTT_THRESHOLDS}
    max_gain = float(value)
    return {f"hit_{threshold}pct": max_gain >= threshold for threshold in GTT_THRESHOLDS}


def _suggested_target(max_gain: pd.Series, style: str) -> float | pd.NA:
    if len(max_gain.dropna()) < 3:
        return pd.NA
    if style == "conservative":
        value = float(max_gain.median())
        return max(0.0, min(value, 10.0))
    value = float(max_gain.quantile(0.75))
    return max(0.0, min(value, 20.0))


def _prepare_daily(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame.sort_values("date").reset_index(drop=True)


def _instrument_name_map(instruments: pd.DataFrame, exchange: str) -> dict[str, str]:
    if instruments.empty or not {"exchange", "tradingsymbol", "name"}.issubset(instruments.columns):
        return {}
    frame = instruments[instruments["exchange"].astype(str).str.upper() == exchange.upper()].copy()
    return dict(zip(frame["tradingsymbol"].astype(str), frame["name"].fillna("").astype(str)))


def _sort_pair_details(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_pair_details()
    return frame.sort_values(["sell_date", "symbol"], ascending=[False, True])


def _sort_open_positions(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return frame.sort_values(["buy_date", "symbol"], ascending=[False, True])


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _rate(mask: pd.Series) -> float:
    if len(mask) == 0:
        return 0
    return float(mask.mean() * 100)


def _empty_summary(exchange: str) -> dict[str, Any]:
    return build_gtt_summary(_empty_pair_details(), pd.DataFrame(), exchange, 0)


def _empty_pair_details() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "exchange",
            "symbol",
            "name",
            "buy_date",
            "buy_close",
            "sell_date",
            "sell_close",
            "buy_to_sell_return_pct",
            "valid_daily_window",
            "highest_price_between_buy_sell",
            "highest_price_date",
            "max_gain_pct",
            "went_above_buy_price",
            "hit_5pct",
            "hit_10pct",
            "hit_15pct",
            "hit_20pct",
            "hit_25pct",
            "hit_30pct",
        ]
    )


def _empty_stock_stats() -> pd.DataFrame:
    return pd.DataFrame(columns=_stock_stats_columns())


def _float_or_na(value: Any) -> float | pd.NA:
    if pd.isna(value):
        return pd.NA
    return float(value)


def _is_number(value: Any) -> bool:
    return pd.notna(value)
