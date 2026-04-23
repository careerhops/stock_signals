from __future__ import annotations

from io import StringIO
import os
from typing import Any

import httpx
import pandas as pd


def build_telegram_message(filtered: pd.DataFrame, summary: dict[str, Any]) -> str:
    lines = [
        "NSE/BSE Investment Screener",
        f"Scan date: {summary.get('scan_date', 'unknown')}",
        f"Symbols scanned: {summary.get('symbols_scanned', 0)}",
        f"Filtered matches: {len(filtered)}",
        "",
    ]

    if filtered.empty:
        lines.append("No matching stocks found today.")
        return "\n".join(lines)

    lines.append("Matches:")
    for idx, row in filtered.head(20).iterrows():
        lines.append(
            f"{idx + 1}. {row.get('exchange')}:{row.get('symbol')} "
            f"{row.get('signal')} close={row.get('close')}"
        )

    if len(filtered) > 20:
        lines.append(f"...and {len(filtered) - 20} more.")

    dashboard_url = summary.get("dashboard_url")
    if dashboard_url:
        lines.extend(["", f"Dashboard: {dashboard_url}"])

    return "\n".join(lines)


def build_buy_signal_list_message(
    filtered: pd.DataFrame,
    inline_limit: int | None = None,
    filters_text: str = "",
) -> str:
    lines = [
        "Weekly BUY Signals",
        f"Total stocks: {len(filtered)}",
    ]
    if filters_text:
        lines.append(f"Filters: {filters_text}")
    lines.append("")

    if filtered.empty:
        lines.append("No weekly BUY signals are available.")
        return "\n".join(lines)

    display_frame = filtered.reset_index(drop=True)
    if inline_limit is not None:
        display_frame = display_frame.head(inline_limit)

    for index, row in display_frame.iterrows():
        symbol = row.get("symbol") or row.get("tradingsymbol") or ""
        exchange = row.get("exchange") or ""
        stock_name = row.get("company_name") or row.get("name") or ""
        signal_date = row.get("date") or ""
        close = row.get("close") or ""
        large_deal = "Yes" if bool(row.get("has_large_deal", False)) else "No"
        large_deal_summary = row.get("large_deal_summary") or ""
        large_deal_text = f"Large Deal: {large_deal}"
        if large_deal_summary:
            large_deal_text = f"{large_deal_text} ({large_deal_summary})"
        lines.append(
            f"{index + 1}. {signal_date} | {exchange}:{symbol} | {stock_name} | Close: {close} | {large_deal_text}"
        )

    if inline_limit is not None and len(filtered) > inline_limit:
        lines.extend(["", f"Showing top {inline_limit}. Full list is attached as CSV."])

    return "\n".join(lines)


def _telegram_credentials(config: dict[str, Any], required: bool = False) -> tuple[str | None, str | None]:
    notifications_cfg = config.get("notifications", {})
    telegram_cfg = notifications_cfg.get("telegram", {})
    token_env = telegram_cfg.get("bot_token_env", "TELEGRAM_BOT_TOKEN")
    chat_env = telegram_cfg.get("chat_id_env", "TELEGRAM_CHAT_ID")

    bot_token = os.getenv(token_env)
    chat_id = os.getenv(chat_env)
    if required and (not bot_token or not chat_id):
        raise RuntimeError(f"Telegram is not configured. Set {token_env} and {chat_env} in .env.")
    return bot_token, chat_id


def send_telegram_message(config: dict[str, Any], message: str, required: bool = False) -> None:
    bot_token, chat_id = _telegram_credentials(config, required=required)
    if not bot_token or not chat_id:
        print("Telegram not configured; skipping notification.")
        return

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    response = httpx.post(url, json={"chat_id": chat_id, "text": message}, timeout=30)
    _raise_for_telegram_error(response)


def buy_signals_to_csv_bytes(filtered: pd.DataFrame) -> bytes:
    export = pd.DataFrame()
    export["date"] = filtered.get("date", "")
    export["exchange"] = filtered.get("exchange", "")
    export["symbol"] = filtered.get("symbol", filtered.get("tradingsymbol", ""))
    export["stock_name"] = filtered.get("company_name", filtered.get("name", ""))
    export["signal"] = filtered.get("signal", "")
    export["signal_close_price"] = filtered.get("close", "")
    if "prior_pair_return_last_1_pct" in filtered.columns:
        export["last_pair_return_pct"] = filtered["prior_pair_return_last_1_pct"]
    if "median_pair_return_last_3_pct" in filtered.columns:
        export["median_last_3_pair_return_pct"] = filtered["median_pair_return_last_3_pct"]
    if "market_cap_cr" in filtered.columns:
        export["market_cap_cr"] = filtered["market_cap_cr"]
    if "market_cap_bucket" in filtered.columns:
        export["market_cap_bucket"] = filtered["market_cap_bucket"]
    if "has_large_deal" in filtered.columns:
        export["recent_large_deal"] = filtered["has_large_deal"].map(lambda value: "Yes" if bool(value) else "No")
    if "large_deal_summary" in filtered.columns:
        export["large_deal_summary"] = filtered["large_deal_summary"]
    if "large_deal_latest_date" in filtered.columns:
        export["large_deal_latest_date"] = filtered["large_deal_latest_date"]

    buffer = StringIO()
    export = export.where(pd.notna(export), "NA")
    export.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def build_gtt_stock_list_message(
    filtered: pd.DataFrame,
    inline_limit: int | None = None,
    filters_text: str = "",
) -> str:
    lines = [
        "GTT Gain Study Filtered Stocks",
        f"Total stocks: {len(filtered)}",
    ]
    if filters_text:
        lines.append(f"Filters: {filters_text}")
    lines.append("")

    if filtered.empty:
        lines.append("No GTT stocks match the selected filters.")
        return "\n".join(lines)

    display_frame = filtered.reset_index(drop=True)
    if inline_limit is not None:
        display_frame = display_frame.head(inline_limit)

    for index, row in display_frame.iterrows():
        symbol = row.get("symbol") or row.get("tradingsymbol") or ""
        exchange = row.get("exchange") or "NSE"
        stock_name = row.get("company_name") or row.get("name") or ""
        valid_pairs = row.get("valid_pairs", "")
        median_gain = row.get("median_max_gain_pct", "")
        median_days_to_peak = row.get("median_days_to_peak", "")
        peak_speed_bucket = row.get("peak_speed_bucket", "")
        hit_10 = row.get("hit_10pct_rate_pct", "")
        conservative = row.get("suggested_conservative_gtt_pct", "")
        technical_rating = row.get("weekly_technical_rating_status", "")
        lines.append(
            f"{index + 1}. {exchange}:{symbol} | {stock_name} | "
            f"Valid pairs: {valid_pairs} | Median max gain: {median_gain}% | "
            f"Median days to peak: {median_days_to_peak} ({peak_speed_bucket}) | Weekly tech: {technical_rating} | Hit 10%: {hit_10}% | "
            f"Conservative GTT: {conservative}%"
        )

    if inline_limit is not None and len(filtered) > inline_limit:
        lines.extend(["", f"Showing top {inline_limit}. Full list is attached as CSV."])

    return "\n".join(lines)


def gtt_stock_list_to_csv_bytes(filtered: pd.DataFrame) -> bytes:
    export = pd.DataFrame()
    export["exchange"] = filtered.get("exchange", "")
    export["symbol"] = filtered.get("symbol", filtered.get("tradingsymbol", ""))
    export["stock_name"] = filtered.get("company_name", filtered.get("name", ""))
    export["market_cap_cr"] = filtered.get("market_cap_cr", "")
    export["market_cap_bucket"] = filtered.get("market_cap_bucket", "")
    export["latest_signal"] = filtered.get("latest_signal", "")
    export["latest_signal_date"] = filtered.get("latest_signal_date", "")
    export["latest_week_signal"] = filtered.get("latest_week_signal", "")
    export["weekly_technical_rating"] = filtered.get("weekly_technical_rating", "")
    export["weekly_technical_rating_status"] = filtered.get("weekly_technical_rating_status", "")
    export["volume_confirmation"] = filtered.get("volume_confirmation", "")
    export["volume_confirmation_ratio"] = filtered.get("volume_confirmation_ratio", "")
    export["valid_pairs"] = filtered.get("valid_pairs", "")
    export["median_max_gain_pct"] = filtered.get("median_max_gain_pct", "")
    export["avg_max_gain_pct"] = filtered.get("avg_max_gain_pct", "")
    export["best_max_gain_pct"] = filtered.get("best_max_gain_pct", "")
    export["median_days_to_peak"] = filtered.get("median_days_to_peak", "")
    export["peak_speed_bucket"] = filtered.get("peak_speed_bucket", "")
    export["avg_days_to_peak"] = filtered.get("avg_days_to_peak", "")
    export["hit_10pct_rate_pct"] = filtered.get("hit_10pct_rate_pct", "")
    export["hit_20pct_rate_pct"] = filtered.get("hit_20pct_rate_pct", "")
    export["suggested_conservative_gtt_pct"] = filtered.get("suggested_conservative_gtt_pct", "")
    export["suggested_moderate_gtt_pct"] = filtered.get("suggested_moderate_gtt_pct", "")

    buffer = StringIO()
    export = export.where(pd.notna(export), "NA")
    export.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def send_telegram_document(
    config: dict[str, Any],
    file_bytes: bytes,
    filename: str,
    caption: str,
    required: bool = False,
) -> None:
    bot_token, chat_id = _telegram_credentials(config, required=required)
    if not bot_token or not chat_id:
        print("Telegram not configured; skipping document notification.")
        return

    url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
    response = httpx.post(
        url,
        data={"chat_id": chat_id, "caption": caption},
        files={"document": (filename, file_bytes, "text/csv")},
        timeout=30,
    )
    _raise_for_telegram_error(response)


def _raise_for_telegram_error(response: httpx.Response) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        try:
            error_payload = response.json()
            description = error_payload.get("description") or response.text
        except ValueError:
            description = response.text
        raise RuntimeError(f"Telegram API error: {description}") from exc


def send_buy_signal_list_to_telegram(
    config: dict[str, Any],
    filtered: pd.DataFrame,
    inline_limit: int = 10,
    filters_text: str = "",
) -> None:
    message = build_buy_signal_list_message(
        filtered,
        inline_limit=inline_limit if len(filtered) > inline_limit else None,
        filters_text=filters_text,
    )
    send_telegram_message(config, message, required=True)

    if len(filtered) > inline_limit:
        csv_bytes = buy_signals_to_csv_bytes(filtered)
        caption = f"Full Weekly BUY Signals list: {len(filtered)} stocks"
        if filters_text:
            caption = f"{caption}\nFilters: {filters_text}"
        send_telegram_document(
            config,
            csv_bytes,
            "weekly_buy_signals.csv",
            caption,
            required=True,
        )


def send_gtt_stock_list_to_telegram(
    config: dict[str, Any],
    filtered: pd.DataFrame,
    inline_limit: int = 10,
    filters_text: str = "",
) -> None:
    message = build_gtt_stock_list_message(
        filtered,
        inline_limit=inline_limit if len(filtered) > inline_limit else None,
        filters_text=filters_text,
    )
    send_telegram_message(config, message, required=True)

    if len(filtered) > inline_limit:
        csv_bytes = gtt_stock_list_to_csv_bytes(filtered)
        caption = f"Full GTT filtered stock list: {len(filtered)} stocks"
        if filters_text:
            caption = f"{caption}\nFilters: {filters_text}"
        send_telegram_document(
            config,
            csv_bytes,
            "gtt_filtered_stocks.csv",
            caption,
            required=True,
        )
