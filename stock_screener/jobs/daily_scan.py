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
from stock_screener.strategy.weekly_buy_sell import run_weekly_buy_sell
from stock_screener.universe import build_universe


def _fetch_start_date(existing: pd.DataFrame, history_years: int) -> date:
    if existing.empty:
        return date.today() - timedelta(days=365 * history_years)
    last_date = pd.to_datetime(existing["date"]).max().date()
    return last_date + timedelta(days=1)


def daily_signal_config(config: dict[str, Any]) -> dict[str, Any]:
    daily_config = deepcopy(config)
    daily_config.setdefault("data", {})
    daily_config["data"]["scan_timeframe"] = "1D"

    daily_cfg = daily_config.get("daily_signals", {}) or {}
    if "max_signal_age_bars" in daily_cfg:
        daily_config.setdefault("filters", {}).setdefault("signal", {})
        daily_config["filters"]["signal"]["max_signal_age_bars"] = daily_cfg["max_signal_age_bars"]

    return daily_config


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

    universe = build_universe(instruments, config)
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
    today = date.today()
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
            phase="Using cached candles" if skip_kite_fetch else "Fetching candles",
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
            from_date = _fetch_start_date(existing_daily, history_years)
            if from_date <= today:
                if provider is None:
                    raise RuntimeError("Kite data provider is not available for data refresh.")
                try:
                    new_daily = provider.daily_candles(token, from_date, today)
                    new_rows = len(new_daily)
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
            else:
                fetch_status = "already_current"
                daily = existing_daily

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

        strategy_input = daily
        if scan_timeframe == "1W":
            strategy_input = resample_daily_to_weekly(daily, weekly_anchor, use_completed_weeks_only)

        strategy_output = run_weekly_buy_sell(strategy_input, config)
        signal_rows = strategy_output[strategy_output["signal"].isin(["BUY", "SELL"])].copy()
        if daily_enabled:
            daily_strategy_output = run_weekly_buy_sell(daily, daily_config)
            daily_signal_rows = daily_strategy_output[daily_strategy_output["signal"].isin(["BUY", "SELL"])].copy()
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
    filtered = apply_filters(raw_signals, config)
    filtered_daily = apply_filters(raw_daily_signals, daily_config)
    scan_details = pd.DataFrame(scan_detail_rows)

    storage.save_signals("latest_raw_signals.csv", raw_signals)
    storage.save_signals("latest_filtered.csv", filtered)
    storage.save_signals("latest_daily_raw_signals.csv", raw_daily_signals)
    storage.save_signals("latest_daily_filtered.csv", filtered_daily)
    storage.save_signals("latest_scan_details.csv", scan_details)

    summary = {
        "scan_date": str(today),
        "symbols_scanned": len(universe),
        "symbols_updated": updated_symbols,
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


if __name__ == "__main__":
    loaded_config = load_config()
    try:
        run_daily_scan(loaded_config)
    except Exception as exc:
        notify_failure(loaded_config, exc)
        raise
