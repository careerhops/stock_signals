from __future__ import annotations

from datetime import date, datetime, timedelta

from stock_screener.data.nse_large_deals import (
    NseLargeDealsClient,
    normalize_historical_large_deals,
    normalize_large_deals,
)
from stock_screener.data.supabase_store import SupabaseStore


def _nse_date(value: date) -> str:
    return value.strftime("%d-%m-%Y")


def fetch_and_store_current_large_deals() -> dict[str, int]:
    payload = NseLargeDealsClient().fetch_snapshot()
    rows = normalize_large_deals(payload)
    deal_dates = sorted({str(row.get("deal_date")) for row in rows if row.get("deal_date")})

    store = SupabaseStore()
    already_stored_dates = [
        deal_date for deal_date in deal_dates if store.count_large_deals_for_date(deal_date) > 0
    ]

    rows_to_store = [
        row for row in rows if row.get("deal_date") and str(row.get("deal_date")) not in already_stored_dates
    ]
    count = store.upsert_large_deals(rows_to_store)
    return {
        "fetched": len(rows),
        "stored": count,
        "skipped_existing_dates": len(already_stored_dates),
    }


def fetch_and_store_large_deals_range(from_date: date, to_date: date, incremental: bool = True) -> dict[str, int]:
    store = SupabaseStore()
    latest_stored_date = store.latest_large_deal_date() if incremental else None

    effective_from = from_date
    if latest_stored_date:
        latest_date = datetime.fromisoformat(latest_stored_date).date()
        if latest_date > effective_from:
            effective_from = latest_date

    if effective_from > to_date:
        return {"fetched": 0, "stored": 0}

    client = NseLargeDealsClient()
    rows = []

    for deal_type in ("bulk", "block", "short"):
        payload = client.fetch_historical(_nse_date(effective_from), _nse_date(to_date), deal_type)
        rows.extend(normalize_historical_large_deals(payload, deal_type))

    count = store.upsert_large_deals(rows)
    return {"fetched": len(rows), "stored": count}


def default_last_7_days_range() -> tuple[date, date]:
    today = date.today()
    return today - timedelta(days=7), today
