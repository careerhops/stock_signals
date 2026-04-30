from __future__ import annotations

from datetime import datetime
from html import escape
from io import StringIO
import os
from typing import Any

import httpx
import pandas as pd


RATING_RANK = {
    "STRONG BUY": 5,
    "BUY": 4,
    "NEUTRAL": 3,
    "SELL": 2,
    "STRONG SELL": 1,
}
HTML_REPORT_LIMIT = 30


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


def buy_signals_to_html_bytes(
    filtered: pd.DataFrame,
    filters_text: str = "",
    limit: int = HTML_REPORT_LIMIT,
) -> bytes:
    frame = _sort_buy_signals_for_report(filtered).head(limit).copy()
    title = "Weekly BUY Signal Report"
    cards = []
    for index, row in frame.reset_index(drop=True).iterrows():
        symbol = row.get("symbol") or row.get("tradingsymbol") or ""
        exchange = row.get("exchange") or ""
        name = row.get("company_name") or row.get("name") or ""
        signal_date = _fmt(row.get("date"))
        close = _fmt(row.get("close"))
        market_cap = _fmt(row.get("market_cap_cr"))
        volume_confirmed = _yes_no(row.get("volume_confirmation"))
        ema_stack = _yes_no(row.get("daily_ema_stack_confirmation", row.get("trend_confirmation")))
        obv = _yes_no(row.get("daily_obv_confirmation", row.get("obv_confirmation")))
        last_return = _fmt(row.get("prior_pair_return_last_1_pct"), suffix="%")
        median_return = _fmt(row.get("median_pair_return_last_3_pct"), suffix="%")
        large_deal = "Large Deal" if bool(row.get("has_large_deal", False)) else "No Large Deal"
        large_deal_class = "badge hot" if bool(row.get("has_large_deal", False)) else "badge muted"
        cards.append(
            f"""
            <article class="card" data-symbol="{_attr(symbol)}" data-name="{_attr(name)}">
              <div class="rank">#{index + 1}</div>
              <div class="card-main">
                <div class="card-title">
                  <span>{_html(exchange)}:{_html(symbol)}</span>
                  <small>{_html(name)}</small>
                </div>
                <div class="badges">
                  <span class="badge buy">BUY</span>
                  <span class="{large_deal_class}">{_html(large_deal)}</span>
                  <span class="badge">EMA {ema_stack}</span>
                  <span class="badge">OBV {obv}</span>
                  <span class="badge">Volume {volume_confirmed}</span>
                </div>
                <div class="metrics">
                  <div><span>Date</span><strong>{_html(signal_date)}</strong></div>
                  <div><span>Close</span><strong>₹{_html(close)}</strong></div>
                  <div><span>Market Cap</span><strong>{_html(market_cap)} Cr</strong></div>
                  <div><span>Last Pair</span><strong>{_html(last_return)}</strong></div>
                  <div><span>Median 3 Pairs</span><strong>{_html(median_return)}</strong></div>
                </div>
              </div>
            </article>
            """
        )
    return _report_html(title, frame, filters_text, cards, _buy_report_table(frame)).encode("utf-8")


def build_gtt_stock_list_message(
    filtered: pd.DataFrame,
    inline_limit: int | None = None,
    filters_text: str = "",
) -> str:
    filtered = _sort_gtt_stocks_for_telegram(filtered)
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
    filtered = _sort_gtt_stocks_for_telegram(filtered)
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
    export["daily_ema_stack_confirmation"] = filtered.get("daily_ema_stack_confirmation", filtered.get("trend_confirmation", ""))
    export["daily_obv_confirmation"] = filtered.get("daily_obv_confirmation", filtered.get("obv_confirmation", ""))
    export["daily_obv_slope_20d"] = filtered.get("daily_obv_slope_20d", "")
    export["valid_pairs"] = filtered.get("valid_pairs", "")
    export["median_max_gain_pct"] = filtered.get("median_max_gain_pct", "")
    export["avg_max_gain_pct"] = filtered.get("avg_max_gain_pct", "")
    export["best_max_gain_pct"] = filtered.get("best_max_gain_pct", "")
    export["median_days_to_peak"] = filtered.get("median_days_to_peak", "")
    export["peak_speed_bucket"] = filtered.get("peak_speed_bucket", "")
    export["avg_days_to_peak"] = filtered.get("avg_days_to_peak", "")
    export["hit_10pct_rate_pct"] = filtered.get("hit_10pct_rate_pct", "")
    export["hit_20pct_rate_pct"] = filtered.get("hit_20pct_rate_pct", "")
    if "low_sample" in filtered.columns:
        export["high_samples"] = filtered["low_sample"].map(
            lambda value: "No" if str(value).strip().lower() in {"1", "true", "yes", "y"} else "Yes"
        )
    export["suggested_conservative_gtt_pct"] = filtered.get("suggested_conservative_gtt_pct", "")
    export["suggested_moderate_gtt_pct"] = filtered.get("suggested_moderate_gtt_pct", "")

    buffer = StringIO()
    export = export.where(pd.notna(export), "NA")
    export.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def gtt_stock_list_to_html_bytes(
    filtered: pd.DataFrame,
    filters_text: str = "",
    limit: int = HTML_REPORT_LIMIT,
) -> bytes:
    frame = _sort_gtt_stocks_for_telegram(filtered).head(limit).copy()
    title = "GTT Filtered Stock Report"
    cards = []
    for index, row in frame.reset_index(drop=True).iterrows():
        symbol = row.get("symbol") or row.get("tradingsymbol") or ""
        exchange = row.get("exchange") or "NSE"
        name = row.get("company_name") or row.get("name") or ""
        rating = _fmt(row.get("weekly_technical_rating_status"))
        rating_class = "badge buy" if rating.upper() in {"STRONG BUY", "BUY"} else "badge"
        ema_stack = _yes_no(row.get("daily_ema_stack_confirmation", row.get("trend_confirmation")))
        obv = _yes_no(row.get("daily_obv_confirmation", row.get("obv_confirmation")))
        high_samples = "No" if _yes_no(row.get("low_sample")) == "Yes" else "Yes"
        cards.append(
            f"""
            <article class="card" data-symbol="{_attr(symbol)}" data-name="{_attr(name)}">
              <div class="rank">#{index + 1}</div>
              <div class="card-main">
                <div class="card-title">
                  <span>{_html(exchange)}:{_html(symbol)}</span>
                  <small>{_html(name)}</small>
                </div>
                <div class="badges">
                  <span class="{rating_class}">{_html(rating)}</span>
                  <span class="badge">EMA {ema_stack}</span>
                  <span class="badge">OBV {obv}</span>
                  <span class="badge">High Samples {high_samples}</span>
                </div>
                <div class="metrics">
                  <div><span>CMP</span><strong>₹{_html(_fmt(row.get("latest_close")))}</strong></div>
                  <div><span>Valid Pairs</span><strong>{_html(_fmt(row.get("valid_pairs")))}</strong></div>
                  <div><span>Median Gain</span><strong>{_html(_fmt(row.get("median_max_gain_pct"), suffix="%"))}</strong></div>
                  <div><span>Hit 10%</span><strong>{_html(_fmt(row.get("hit_10pct_rate_pct"), suffix="%"))}</strong></div>
                  <div><span>Median Peak Days</span><strong>{_html(_fmt(row.get("median_days_to_peak")))}</strong></div>
                  <div><span>Conservative GTT</span><strong>{_html(_fmt(row.get("suggested_conservative_gtt_pct"), suffix="%"))}</strong></div>
                </div>
              </div>
            </article>
            """
        )
    return _report_html(title, frame, filters_text, cards, _gtt_report_table(frame)).encode("utf-8")


def send_telegram_document(
    config: dict[str, Any],
    file_bytes: bytes,
    filename: str,
    caption: str,
    media_type: str = "text/csv",
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
        files={"document": (filename, file_bytes, media_type)},
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

    html_count = min(len(filtered), HTML_REPORT_LIMIT)
    html_bytes = buy_signals_to_html_bytes(filtered, filters_text=filters_text, limit=HTML_REPORT_LIMIT)
    send_telegram_document(
        config,
        html_bytes,
        _dated_filename("weekly_buy_signal_report", "html"),
        f"Readable Weekly BUY report: top {html_count} of {len(filtered)} stocks",
        media_type="text/html",
        required=True,
    )

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
            media_type="text/csv",
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

    html_count = min(len(filtered), HTML_REPORT_LIMIT)
    html_bytes = gtt_stock_list_to_html_bytes(filtered, filters_text=filters_text, limit=HTML_REPORT_LIMIT)
    send_telegram_document(
        config,
        html_bytes,
        _dated_filename("gtt_filtered_stock_report", "html"),
        f"Readable GTT report: top {html_count} of {len(filtered)} stocks",
        media_type="text/html",
        required=True,
    )

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
            media_type="text/csv",
            required=True,
        )


def _report_html(title: str, frame: pd.DataFrame, filters_text: str, cards: list[str], table_html: str) -> str:
    symbols = ", ".join(_symbols(frame))
    generated_at = datetime.now().strftime("%d %b %Y, %I:%M %p")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_html(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f9fb;
      --panel: #ffffff;
      --ink: #172033;
      --muted: #667085;
      --line: #dfe5ec;
      --green: #0f9f6e;
      --blue: #2563eb;
      --red: #d92d20;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    header {{ padding: 22px 0 18px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .subtle {{ color: var(--muted); }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 18px 0;
    }}
    .summary-card, .card, .table-panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    }}
    .summary-card {{ padding: 16px; }}
    .summary-card span {{ color: var(--muted); font-size: 13px; }}
    .summary-card strong {{ display: block; font-size: 24px; margin-top: 4px; }}
    .toolbar {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 18px 0; }}
    input, button {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      font: inherit;
      background: white;
    }}
    button {{ cursor: pointer; color: white; background: var(--blue); border-color: var(--blue); }}
    .symbols-box {{
      width: 100%;
      min-height: 58px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }}
    .cards {{ display: grid; gap: 12px; margin: 16px 0 24px; }}
    .card {{ display: grid; grid-template-columns: 54px 1fr; gap: 12px; padding: 14px; }}
    .rank {{ color: var(--muted); font-weight: 700; padding-top: 4px; }}
    .card-title {{ display: flex; flex-direction: column; gap: 2px; font-weight: 800; }}
    .card-title small {{ color: var(--muted); font-weight: 500; }}
    .badges {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0; }}
    .badge {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 4px 8px;
      font-size: 12px;
      color: #344054;
      background: #f8fafc;
    }}
    .badge.buy {{ color: #027a48; background: #ecfdf3; border-color: #abefc6; }}
    .badge.hot {{ color: #b42318; background: #fef3f2; border-color: #fecdca; }}
    .badge.muted {{ color: var(--muted); }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 8px;
    }}
    .metrics div {{ padding: 8px; border: 1px solid #eef2f6; border-radius: 8px; }}
    .metrics span {{ display: block; color: var(--muted); font-size: 12px; }}
    .metrics strong {{ font-size: 14px; }}
    .table-panel {{ overflow: auto; margin-top: 18px; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 880px; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }}
    th {{ background: #f8fafc; font-size: 12px; color: #475467; }}
    @media (max-width: 640px) {{
      main {{ padding: 14px; }}
      .card {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 22px; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>{_html(title)}</h1>
    <div class="subtle">Generated: {_html(generated_at)}</div>
    <div class="subtle">Filters: {_html(filters_text or "None")}</div>
    <div class="subtle">Showing top {len(frame)} stocks in this HTML report.</div>
  </header>
  <section class="summary">
    <div class="summary-card"><span>Total stocks</span><strong>{len(frame)}</strong></div>
    <div class="summary-card"><span>Strong/Buy rating</span><strong>{_rating_buy_count(frame)}</strong></div>
    <div class="summary-card"><span>EMA stack</span><strong>{_truthy_count(frame, "daily_ema_stack_confirmation", "trend_confirmation")}</strong></div>
    <div class="summary-card"><span>OBV rising</span><strong>{_truthy_count(frame, "daily_obv_confirmation", "obv_confirmation")}</strong></div>
  </section>
  <section class="toolbar">
    <input id="search" placeholder="Search symbol or company" oninput="filterCards()">
    <button type="button" onclick="copySymbols()">Copy TradingView symbols</button>
  </section>
  <textarea class="symbols-box" id="symbols" readonly>{_html(symbols)}</textarea>
  <section class="cards" id="cards">
    {''.join(cards) if cards else '<p class="subtle">No stocks to show.</p>'}
  </section>
  <section class="table-panel">
    {table_html}
  </section>
</main>
<script>
function filterCards() {{
  const q = document.getElementById('search').value.toLowerCase();
  document.querySelectorAll('.card').forEach(card => {{
    const text = (card.dataset.symbol + ' ' + card.dataset.name).toLowerCase();
    card.style.display = text.includes(q) ? '' : 'none';
  }});
}}
function copySymbols() {{
  const box = document.getElementById('symbols');
  box.select();
  document.execCommand('copy');
}}
</script>
</body>
</html>"""


def _buy_report_table(frame: pd.DataFrame) -> str:
    columns = [
        ("date", "Date"),
        ("exchange", "Exchange"),
        ("symbol", "Symbol"),
        ("company_name", "Name"),
        ("close", "Close"),
        ("market_cap_cr", "Market Cap Cr"),
        ("daily_ema_stack_confirmation", "EMA Stack"),
        ("daily_obv_confirmation", "OBV 20D"),
        ("volume_confirmation", "Volume"),
        ("prior_pair_return_last_1_pct", "Last Pair %"),
        ("median_pair_return_last_3_pct", "Median 3 %"),
        ("large_deal_summary", "Large Deal"),
    ]
    return _table_html(frame, columns)


def _gtt_report_table(frame: pd.DataFrame) -> str:
    columns = [
        ("exchange", "Exchange"),
        ("symbol", "Symbol"),
        ("company_name", "Name"),
        ("latest_close", "CMP"),
        ("weekly_technical_rating_status", "Tech Rating"),
        ("daily_ema_stack_confirmation", "EMA Stack"),
        ("daily_obv_confirmation", "OBV 20D"),
        ("valid_pairs", "Valid Pairs"),
        ("median_max_gain_pct", "Median Gain %"),
        ("hit_10pct_rate_pct", "Hit 10% %"),
        ("median_days_to_peak", "Median Peak Days"),
        ("suggested_conservative_gtt_pct", "Conservative GTT %"),
    ]
    return _table_html(frame, columns)


def _table_html(frame: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
    header = "".join(f"<th>{_html(label)}</th>" for _, label in columns)
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for column, _ in columns:
            value = row.get(column)
            if column == "company_name" and (value is None or pd.isna(value) or str(value).strip() == ""):
                value = row.get("name", "")
            if column == "symbol" and (value is None or pd.isna(value) or str(value).strip() == ""):
                value = row.get("tradingsymbol", "")
            cells.append(f"<td>{_html(_fmt(value))}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _sort_buy_signals_for_report(filtered: pd.DataFrame) -> pd.DataFrame:
    if filtered.empty:
        return filtered
    frame = filtered.copy()
    frame["_ema_sort"] = _truthy_series(frame.get("daily_ema_stack_confirmation", frame.get("trend_confirmation", False)))
    frame["_obv_sort"] = _truthy_series(frame.get("daily_obv_confirmation", frame.get("obv_confirmation", False)))
    frame["_vol_sort"] = _truthy_series(frame.get("volume_confirmation", False))
    frame["_median_sort"] = pd.to_numeric(_column_or_default(frame, "median_pair_return_last_3_pct", 0), errors="coerce").fillna(-9999)
    frame["_symbol_sort"] = _column_or_default(frame, "symbol", _column_or_default(frame, "tradingsymbol", "")).astype(str)
    return frame.sort_values(
        ["_ema_sort", "_obv_sort", "_vol_sort", "_median_sort", "_symbol_sort"],
        ascending=[False, False, False, False, True],
    ).drop(columns=["_ema_sort", "_obv_sort", "_vol_sort", "_median_sort", "_symbol_sort"], errors="ignore")


def _sort_gtt_stocks_for_report(filtered: pd.DataFrame) -> pd.DataFrame:
    if filtered.empty:
        return filtered
    frame = filtered.copy()
    frame["_rating_sort"] = _column_or_default(frame, "weekly_technical_rating_status", "").astype(str).str.upper().map(RATING_RANK).fillna(0)
    frame["_ema_sort"] = _truthy_series(frame.get("daily_ema_stack_confirmation", frame.get("trend_confirmation", False)))
    frame["_obv_sort"] = _truthy_series(frame.get("daily_obv_confirmation", frame.get("obv_confirmation", False)))
    frame["_hit10_sort"] = pd.to_numeric(_column_or_default(frame, "hit_10pct_rate_pct", 0), errors="coerce").fillna(-9999)
    frame["_gain_sort"] = pd.to_numeric(_column_or_default(frame, "median_max_gain_pct", 0), errors="coerce").fillna(-9999)
    frame["_pairs_sort"] = pd.to_numeric(_column_or_default(frame, "valid_pairs", 0), errors="coerce").fillna(0)
    frame["_symbol_sort"] = _column_or_default(frame, "symbol", _column_or_default(frame, "tradingsymbol", "")).astype(str)
    return frame.sort_values(
        ["_rating_sort", "_ema_sort", "_obv_sort", "_hit10_sort", "_gain_sort", "_pairs_sort", "_symbol_sort"],
        ascending=[False, False, False, False, False, False, True],
    ).drop(
        columns=["_rating_sort", "_ema_sort", "_obv_sort", "_hit10_sort", "_gain_sort", "_pairs_sort", "_symbol_sort"],
        errors="ignore",
    )


def _sort_gtt_stocks_for_telegram(filtered: pd.DataFrame) -> pd.DataFrame:
    if filtered.empty:
        return filtered
    frame = filtered.copy()
    median_return = _column_or_default(
        frame,
        "median_pair_return_last_3_pct",
        _column_or_default(frame, "median_max_gain_pct", pd.NA),
    )
    frame["_median_return_sort"] = pd.to_numeric(median_return, errors="coerce")
    frame["_median_return_missing"] = frame["_median_return_sort"].isna()
    frame["_symbol_sort"] = _column_or_default(frame, "symbol", _column_or_default(frame, "tradingsymbol", "")).astype(str)
    return frame.sort_values(
        ["_median_return_missing", "_median_return_sort", "_symbol_sort"],
        ascending=[True, True, True],
    ).drop(columns=["_median_return_missing", "_median_return_sort", "_symbol_sort"], errors="ignore")


def _symbols(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return []
    symbols = _column_or_default(frame, "symbol", _column_or_default(frame, "tradingsymbol", "")).fillna("").astype(str)
    exchange = _column_or_default(frame, "exchange", "NSE").fillna("NSE").astype(str)
    return [f"{ex}:{symbol}" for ex, symbol in zip(exchange, symbols) if symbol]


def _column_or_default(frame: pd.DataFrame, column: str, default: Any) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    if isinstance(default, pd.Series):
        return default
    return pd.Series([default] * len(frame), index=frame.index)


def _truthy_series(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})
    return pd.Series([values]).astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _truthy_count(frame: pd.DataFrame, preferred: str, fallback: str) -> int:
    if preferred in frame.columns:
        return int(_truthy_series(frame[preferred]).sum())
    if fallback in frame.columns:
        return int(_truthy_series(frame[fallback]).sum())
    return 0


def _rating_buy_count(frame: pd.DataFrame) -> int:
    if "weekly_technical_rating_status" not in frame.columns:
        return 0
    ratings = frame["weekly_technical_rating_status"].fillna("").astype(str).str.upper()
    return int(ratings.isin({"STRONG BUY", "BUY"}).sum())


def _yes_no(value: Any) -> str:
    if isinstance(value, pd.Series):
        value = value.iloc[0] if not value.empty else False
    return "Yes" if str(value).strip().lower() in {"1", "true", "yes", "y"} else "No"


def _fmt(value: Any, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, float):
        text = f"{value:.2f}".rstrip("0").rstrip(".")
    else:
        text = str(value)
    if text == "":
        return "NA"
    return f"{text}{suffix}"


def _html(value: Any) -> str:
    return escape(str(value), quote=False)


def _attr(value: Any) -> str:
    return escape(str(value), quote=True)


def _dated_filename(prefix: str, extension: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    return f"{prefix}_{stamp}.{extension}"
