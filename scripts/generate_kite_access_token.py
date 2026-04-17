from __future__ import annotations

import os

from dotenv import load_dotenv
from kiteconnect import KiteConnect

from stock_screener.auth.kite_token import save_access_token
from stock_screener.config import get_data_root, load_config


def main() -> None:
    load_dotenv()
    api_key = os.getenv("KITE_API_KEY")
    api_secret = os.getenv("KITE_API_SECRET")

    if not api_key or not api_secret:
        raise RuntimeError("Set KITE_API_KEY and KITE_API_SECRET in .env first.")

    kite = KiteConnect(api_key=api_key)

    print("")
    print("Open this URL, login to Zerodha, then copy request_token from the redirect URL:")
    print(kite.login_url())
    print("")

    request_token = input("Paste Kite request_token: ").strip()
    if not request_token:
        raise RuntimeError("request_token is required.")

    session = kite.generate_session(request_token, api_secret=api_secret)
    access_token = session["access_token"]
    kite.set_access_token(access_token)
    profile = kite.profile()

    config = load_config()
    data_root = get_data_root(config)
    path = save_access_token(data_root, access_token, profile)

    print("")
    print("KITE_ACCESS_TOKEN=" + access_token)
    print("")
    print(f"Saved token to {path}")
    print("The scanner will use this saved token automatically.")


if __name__ == "__main__":
    main()
