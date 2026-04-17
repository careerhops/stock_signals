from __future__ import annotations

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


def send_telegram_message(config: dict[str, Any], message: str) -> None:
    notifications_cfg = config.get("notifications", {})
    telegram_cfg = notifications_cfg.get("telegram", {})
    token_env = telegram_cfg.get("bot_token_env", "TELEGRAM_BOT_TOKEN")
    chat_env = telegram_cfg.get("chat_id_env", "TELEGRAM_CHAT_ID")

    bot_token = os.getenv(token_env)
    chat_id = os.getenv(chat_env)
    if not bot_token or not chat_id:
        print("Telegram not configured; skipping notification.")
        return

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    response = httpx.post(url, json={"chat_id": chat_id, "text": message}, timeout=30)
    response.raise_for_status()

