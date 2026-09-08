from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
from typing import Any, Callable

import pandas as pd

from stock_screener.auth.kite_token import load_access_token
from stock_screener.config import get_data_root, load_config
from stock_screener.data.kite import KiteDataProvider
from stock_screener.data.storage import Storage
from stock_screener.filters import apply_filters
from stock_screener.notifications.telegram import build_telegram_message, send_telegram_message
from stock_screener.resample import resample_daily_to_weekly
from stock_screener.strategy.daily_confirmation import add_latest_daily_confirmation_columns
from stock_screener.strategy.weekly_shortlist import shortlist_benchmark_symbols
from stock_screener.strategy.weekly_buy_sell import run_weekly_buy_sell
from stock_screener.symbols import is_excluded_weekly_screener_instrument
from stock_screener.universe import build_universe


def _fetch_start_date(existing: pd.DataFrame, history_years: int) -> date:
    if existing.empty:
        return date.today() - timedelta(days=365 * history_years)
    last_date = pd.to_datetime(existing["date"]).max().date()
    # Overlap the latest saved day so corrected EOD candles replace stale values.
    return last_date


def daily_signal_config(config: dict[str, Any]) -> dict[str, Any]:
    daily_config = deepcopy(config)
    daily_config.setdefault("data", {})
    daily_config["data"]["scan_timeframe"] = "1D"

    daily_cfg = daily_config.get("daily_signals", {}) or {}
    if "max_signal_age_bars" in daily_cfg:
        daily_config.setdefault("filters", {}).setdefault("signal", {})
        daily_config["filters"]["signal"]["max_signal_age_bars"] = daily_cfg["max_signal_age_bars"]

    return daily_config


def _drop_excluded_weekly_rows(frame: pd.DataFrame, symbol_column: str) -> pd.DataFrame:
    if frame.empty or symbol_column not in frame.columns:
        return frame
    names = frame.get("name", pd.Series("", index=frame.index))
    excluded = pd.Series(
        [
            is_excluded_weekly_screener_instrument(symbol, name)
            for symbol, name in zip(frame[symbol_column], names)
        ],
        index=frame.index,
        dtype=bool,
    )
    return frame[~excluded].copy()


def run_daily_scan(
    config: dict[str, Any] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    config = config or load_config()
    data_root = get_data_root(config)
    storage = Storage(data_root)

    def emit_progress(**payload: Any) -> None:
        if progress_callback:
            progress_callback(payload)

    skip_kite_fetch = bool(config.get("data", {}).get("skip_kite_fetch", False))
    provider: KiteDataProvider | None = None
    verified_market_date: date | None = None

    if skip_kite_fetch:
        emit_progress(phase="Loading cached Kite instruments", completed=0, total=0, current_symbol="")
        instruments = storage.load_instruments()
        if instruments.empty:
            raise RuntimeError("No cached Kite instruments found. Run a Kite data refresh once before cached-only scans.")
    else:
        emit_progress(phase="Validating Kite session", completed=0, total=0, current_symbol="")
        access_token = load_access_token(data_root)
        if not access_token:
            raise RuntimeError("Kite access token not found. Start the dashboard and open /auth/kite/login, or run scripts/generate_kite_access_token.py.")

        provider = KiteDataProvider(access_token=access_token)
        provider.validate_session()

        emit_progress(phase="Loading Kite instruments", completed=0, total=0, current_symbol="")
        instruments = provider.instruments()
        storage.save_instruments(instruments)
        verified_market_date = _refresh_shortlist_benchmark_candles(
            provider,
            storage,
            instruments,
            history_years=int(config.get("data", {}).get("history_years", 10)),
        )

    universe = build_universe(instruments, config)
    universe_count_before_stock_filter = len(universe)
    universe = _drop_excluded_weekly_rows(universe, "tradingsymbol").reset_index(drop=True)
    excluded_non_stock_count = universe_count_before_stock_filter - len(universe)
    emit_progress(
        phase="Universe ready",
        completed=0,
        total=len(universe),
        current_symbol="",
    )
    history_years = int(config.get("data", {}).get("history_years", 10))
    scan_timeframe = config.get("data", {}).get("scan_timeframe", "1W")
    strategy_cfg = config.get("strategy", {})
    weekly_anchor = strategy_cfg.get("weekly_anchor", "W-FRI")
    use_completed_weeks_only = bool(strategy_cfg.get("use_completed_weeks_only", True))

    all_signal_rows: list[pd.DataFrame] = []
    all_daily_signal_rows: list[pd.DataFrame] = []
    scan_detail_rows: list[dict[str, Any]] = []
    updated_symbols = 0
    reused_symbols = 0
    fetched_symbols = 0
    today = date.today()
    refresh_cutoff = verified_market_date or _latest_completed_nse_calendar_date()
    stale_symbols_excluded = 0
    as_of_value = str(config.get("data", {}).get("analysis_as_of_date", "") or "").strip()
    as_of_timestamp = pd.to_datetime(as_of_value, errors="coerce") if as_of_value else pd.NaT
    daily_enabled = bool(config.get("daily_signals", {}).get("enabled", True))
    daily_config = daily_signal_config(config)

    for completed, (_, instrument) in enumerate(universe.iterrows(), start=1):
        exchange = str(instrument["exchange"])
        symbol = str(instrument["tradingsymbol"])
        token = int(instrument["instrument_token"])
        fetch_status = "not_started"
        fetch_error = ""
        new_rows = 0

        emit_progress(
            phase="Reading saved candles",
            completed=completed - 1,
            total=len(universe),
            current_symbol=symbol,
            current_exchange=exchange,
        )

        existing_daily = storage.load_candles(exchange, symbol, "1D")
        if skip_kite_fetch:
            fetch_status = "cached"
            daily = existing_daily
        else:
            existing_latest = _latest_candle_date(existing_daily)
            if existing_latest is not None and existing_latest >= refresh_cutoff:
                fetch_status = "reused_fresh"
                reused_symbols += 1
                daily = existing_daily
            else:
                from_date = _fetch_start_date(existing_daily, history_years)
                if provider is None:
                    raise RuntimeError("Kite data provider is not available for data refresh.")
                emit_progress(
                    phase="Fetching missing candles from Kite",
                    completed=completed - 1,
                    total=len(universe),
                    current_symbol=symbol,
                    current_exchange=exchange,
                )
                try:
                    new_daily = provider.daily_candles(token, from_date, refresh_cutoff)
                    new_rows = len(new_daily)
                    fetched_symbols += 1
                    daily = storage.merge_and_save_candles(exchange, symbol, new_daily, "1D")
                    if not new_daily.empty:
                        updated_symbols += 1
                        fetch_status = "updated"
                    else:
                        fetch_status = "no_new_rows"
                except Exception as exc:
                    fetch_error = str(exc)
                    fetch_status = "failed"
                    print(f"Failed fetching {exchange}:{symbol}: {exc}")
                    daily = existing_daily

            latest_after_refresh = _latest_candle_date(daily)
            if (
                verified_market_date is not None
                and latest_after_refresh != refresh_cutoff
            ):
                stale_symbols_excluded += 1
                if fetch_status != "failed":
                    fetch_status = "stale_after_refresh"
                daily = daily.iloc[0:0].copy()

        if daily.empty:
            scan_detail_rows.append(
                {
                    "exchange": exchange,
                    "symbol": symbol,
                    "name": instrument.get("name", symbol),
                    "fetch_status": fetch_status,
                    "fetch_error": fetch_error,
                    "new_rows": new_rows,
                    "daily_rows": 0,
                    "strategy_rows": 0,
                    "latest_candle_date": "",
                    "latest_close": pd.NA,
                    "latest_close_date": "",
                    "raw_signal_count": 0,
                    "latest_signal": "NONE",
                    "latest_signal_date": "",
                    "daily_raw_signal_count": 0,
                    "latest_daily_signal": "NONE",
                    "latest_daily_signal_date": "",
                }
            )
            emit_progress(
                phase="Running weekly strategy",
                completed=completed,
                total=len(universe),
                current_symbol=symbol,
                current_exchange=exchange,
            )
            continue

        if pd.notna(as_of_timestamp):
            daily_dates = pd.to_datetime(daily.get("date"), errors="coerce")
            daily = daily[
                daily_dates.dt.normalize() <= pd.Timestamp(as_of_timestamp).normalize()
            ].copy()
            if daily.empty:
                scan_detail_rows.append(
                    {
                        "exchange": exchange,
                        "symbol": symbol,
                        "name": instrument.get("name", symbol),
                        "fetch_status": fetch_status,
                        "fetch_error": fetch_error,
                        "new_rows": new_rows,
                        "daily_rows": 0,
                        "strategy_rows": 0,
                        "latest_candle_date": "",
                        "latest_close": pd.NA,
                        "latest_close_date": "",
                        "raw_signal_count": 0,
                        "latest_signal": "NONE",
                        "latest_signal_date": "",
                        "daily_raw_signal_count": 0,
                        "latest_daily_signal": "NONE",
                        "latest_daily_signal_date": "",
                    }
                )
                continue

        strategy_input = daily
        if scan_timeframe == "1W":
            strategy_input = resample_daily_to_weekly(daily, weekly_anchor, use_completed_weeks_only)

        strategy_output = run_weekly_buy_sell(strategy_input, config)
        signal_rows = strategy_output[strategy_output["signal"].isin(["BUY", "SELL"])].copy()
        signal_rows = add_latest_daily_confirmation_columns(signal_rows, daily)
        if daily_enabled:
            daily_strategy_output = run_weekly_buy_sell(daily, daily_config)
            daily_signal_rows = daily_strategy_output[daily_strategy_output["signal"].isin(["BUY", "SELL"])].copy()
            daily_signal_rows = add_latest_daily_confirmation_columns(daily_signal_rows, daily)
        else:
            daily_signal_rows = pd.DataFrame()

        latest_signal = "NONE"
        latest_signal_date = ""
        if not signal_rows.empty:
            latest_row = signal_rows.sort_values("date").iloc[-1]
            latest_signal = str(latest_row["signal"])
            latest_signal_date = str(latest_row["date"])

        latest_daily_signal = "NONE"
        latest_daily_signal_date = ""
        if not daily_signal_rows.empty:
            latest_daily_row = daily_signal_rows.sort_values("date").iloc[-1]
            latest_daily_signal = str(latest_daily_row["signal"])
            latest_daily_signal_date = str(latest_daily_row["date"])

        scan_detail_rows.append(
            {
                "exchange": exchange,
                "symbol": symbol,
                "name": instrument.get("name", symbol),
                "fetch_status": fetch_status,
                "fetch_error": fetch_error,
                "new_rows": new_rows,
                "daily_rows": len(daily),
                "strategy_rows": len(strategy_input),
                "latest_candle_date": str(pd.to_datetime(daily["date"]).max()),
                "latest_close": pd.to_numeric(pd.Series([daily.iloc[-1].get("close")]), errors="coerce").iloc[0],
                "latest_close_date": str(pd.to_datetime(daily.iloc[-1].get("date"), errors="coerce")),
                "raw_signal_count": len(signal_rows),
                "latest_signal": latest_signal,
                "latest_signal_date": latest_signal_date,
                "daily_raw_signal_count": len(daily_signal_rows),
                "latest_daily_signal": latest_daily_signal,
                "latest_daily_signal_date": latest_daily_signal_date,
            }
        )
        if not daily_signal_rows.empty:
            daily_signal_rows["exchange"] = exchange
            daily_signal_rows["symbol"] = symbol
            daily_signal_rows["name"] = instrument.get("name", symbol)
            daily_signal_rows["timeframe"] = "1D"
            all_daily_signal_rows.append(daily_signal_rows)

        if signal_rows.empty:
            emit_progress(
                phase="Running weekly strategy",
                completed=completed,
                total=len(universe),
                current_symbol=symbol,
                current_exchange=exchange,
            )
            continue

        signal_rows["exchange"] = exchange
        signal_rows["symbol"] = symbol
        signal_rows["name"] = instrument.get("name", symbol)
        signal_rows["timeframe"] = scan_timeframe
        all_signal_rows.append(signal_rows)

        emit_progress(
            phase="Running weekly strategy",
            completed=completed,
            total=len(universe),
            current_symbol=symbol,
            current_exchange=exchange,
        )

    emit_progress(phase="Saving results", completed=len(universe), total=len(universe), current_symbol="")
    raw_signals = pd.concat(all_signal_rows, ignore_index=True) if all_signal_rows else pd.DataFrame()
    raw_daily_signals = (
        pd.concat(all_daily_signal_rows, ignore_index=True)
        if all_daily_signal_rows
        else pd.DataFrame()
    )
    raw_signals = _drop_excluded_weekly_rows(raw_signals, "symbol")
    raw_daily_signals = _drop_excluded_weekly_rows(raw_daily_signals, "symbol")
    filtered = apply_filters(raw_signals, config)
    filtered_daily = apply_filters(raw_daily_signals, daily_config)
    scan_details = _drop_excluded_weekly_rows(pd.DataFrame(scan_detail_rows), "symbol")

    storage.save_signals("latest_raw_signals.csv", raw_signals)
    storage.save_signals("latest_filtered.csv", filtered)
    storage.save_signals("latest_daily_raw_signals.csv", raw_daily_signals)
    storage.save_signals("latest_daily_filtered.csv", filtered_daily)
    storage.save_signals("latest_scan_details.csv", scan_details)

    summary = {
        "scan_date": (
            pd.Timestamp(as_of_timestamp).strftime("%Y-%m-%d")
            if pd.notna(as_of_timestamp)
            else str(today)
        ),
        "analysis_as_of_date": (
            pd.Timestamp(as_of_timestamp).strftime("%Y-%m-%d")
            if pd.notna(as_of_timestamp)
            else ""
        ),
        "symbols_scanned": len(universe),
        "symbols_excluded_non_stock": excluded_non_stock_count,
        "symbols_updated": updated_symbols,
        "symbols_reused_fresh": reused_symbols,
        "symbols_fetched_from_kite": fetched_symbols,
        "symbols_stale_excluded": stale_symbols_excluded,
        "verified_market_date": (
            verified_market_date.isoformat() if verified_market_date is not None else ""
        ),
        "raw_signals": len(raw_signals),
        "filtered_matches": len(filtered),
        "daily_raw_signals": len(raw_daily_signals),
        "daily_filtered_matches": len(filtered_daily),
        "refresh_mode": "cached_only" if skip_kite_fetch else "kite_refresh",
        "dashboard_url": config.get("notifications", {}).get("dashboard_url", ""),
    }

    notifications_cfg = config.get("notifications", {})
    if notifications_cfg.get("enabled", True):
        if len(filtered) > 0 or notifications_cfg.get("send_when_no_matches", True):
            message = build_telegram_message(filtered, summary)
            send_telegram_message(config, message)

    print(summary)
    emit_progress(
        phase="Complete",
        completed=len(universe),
        total=len(universe),
        current_symbol="",
        summary=summary,
    )
    return summary


def _latest_candle_date(frame: pd.DataFrame) -> date | None:
    if frame.empty or "date" not in frame.columns:
        return None
    latest = pd.to_datetime(frame["date"], errors="coerce").max()
    return None if pd.isna(latest) else pd.Timestamp(latest).date()


def _latest_completed_nse_calendar_date(now_ist: pd.Timestamp | None = None) -> date:
    current = pd.Timestamp.now(tz="Asia/Kolkata") if now_ist is None else pd.Timestamp(now_ist)
    if current.tzinfo is None:
        current = current.tz_localize("Asia/Kolkata")
    else:
        current = current.tz_convert("Asia/Kolkata")
    cutoff = current.date()
    if current.weekday() >= 5 or (current.hour, current.minute) < (15, 45):
        cutoff -= timedelta(days=1)
    while cutoff.weekday() >= 5:
        cutoff -= timedelta(days=1)
    return cutoff


def notify_failure(config: dict[str, Any], error: Exception) -> None:
    notifications_cfg = config.get("notifications", {})
    if not notifications_cfg.get("enabled", True):
        return

    message = "\n".join(
        [
            "NSE/BSE Investment Screener",
            "Daily scan failed.",
            "",
            f"Reason: {error}",
            "",
            "If this is a Kite token issue, start the dashboard and open /auth/kite/login, or run scripts/generate_kite_access_token.py.",
        ]
    )

    try:
        send_telegram_message(config, message)
    except Exception as notify_error:
        print(f"Failed sending failure notification: {notify_error}")


def _refresh_shortlist_benchmark_candles(
    provider: KiteDataProvider,
    storage: Storage,
    instruments: pd.DataFrame,
    history_years: int,
) -> date | None:
    if instruments.empty:
        return None

    wanted = set(shortlist_benchmark_symbols())
    benchmark_rows = instruments[
        (instruments["exchange"].astype(str).str.upper() == "NSE")
        & (instruments["tradingsymbol"].astype(str).isin(wanted))
    ].drop_duplicates(subset=["tradingsymbol"])
    if benchmark_rows.empty:
        return None

    refresh_cutoff = _latest_completed_nse_calendar_date()
    verified_dates: list[date] = []
    for _, instrument in benchmark_rows.iterrows():
        symbol = str(instrument["tradingsymbol"])
        token = int(instrument["instrument_token"])
        existing = storage.load_candles("NSE_INDEX", symbol, "1D")
        existing_latest = _latest_candle_date(existing)
        if existing_latest is not None and existing_latest >= refresh_cutoff:
            verified_dates.append(existing_latest)
            continue
        from_date = _fetch_start_date(existing, history_years)
        if from_date > refresh_cutoff:
            continue
        try:
            new_daily = provider.daily_candles(token, from_date, refresh_cutoff)
        except Exception:
            continue
        merged = storage.merge_and_save_candles("NSE_INDEX", symbol, new_daily, "1D")
        latest = _latest_candle_date(merged)
        if latest is not None:
            verified_dates.append(latest)
    return max(verified_dates) if verified_dates else None


if __name__ == "__main__":
    loaded_config = load_config()
    try:
        run_daily_scan(loaded_config)
    except Exception as exc:
        notify_failure(loaded_config, exc)
        raise
