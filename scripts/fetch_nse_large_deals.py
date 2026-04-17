from __future__ import annotations

from dotenv import load_dotenv

from stock_screener.jobs.large_deals import fetch_and_store_current_large_deals


def main() -> None:
    load_dotenv()

    result = fetch_and_store_current_large_deals()

    print(f"Fetched {result['fetched']} NSE large deal rows.")
    print(f"Stored {result['stored']} rows into Supabase.")


if __name__ == "__main__":
    main()
