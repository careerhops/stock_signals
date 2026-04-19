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
