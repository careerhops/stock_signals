from __future__ import annotations

import argparse
import csv
from datetime import date
import json
import os
from pathlib import Path
import sys
from typing import Iterable

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stock_screener.morningstar_fair_value import (
    FairValueRequest,
    MorningstarFairValueAgent,
    MorningstarFairValueError,
)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch current-month and previous-month Morningstar fair value.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="Morningstar India stock price-vs-fair page URL.")
    source.add_argument(
        "--input-csv",
        type=Path,
        help="CSV containing symbol and page_url columns.",
    )
    parser.add_argument("--symbol", help="NSE symbol; required with --url.")
    parser.add_argument("--as-of", type=date.fromisoformat, help="Target month as YYYY-MM-DD.")
    parser.add_argument("--output-csv", type=Path, help="Optional destination CSV.")
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=_env_float("MORNINGSTAR_REQUEST_DELAY_SECONDS", 2.0),
        help="Seconds to wait between stock requests in batch mode.",
    )
    parser.add_argument(
        "--jitter-seconds",
        type=float,
        default=_env_float("MORNINGSTAR_REQUEST_JITTER_SECONDS", 1.0),
        help="Maximum extra random seconds to add between stock requests.",
    )
    return parser.parse_args()


def load_requests(args: argparse.Namespace) -> list[FairValueRequest]:
    if args.url:
        if not args.symbol:
            raise MorningstarFairValueError("--symbol is required with --url.")
        return [FairValueRequest(symbol=args.symbol, page_url=args.url)]

    assert args.input_csv is not None
    with args.input_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    requests = [
        FairValueRequest(
            symbol=str(row.get("symbol") or "").strip(),
            page_url=str(row.get("page_url") or row.get("url") or "").strip(),
        )
        for row in rows
    ]
    invalid_rows = [
        index + 2
        for index, request in enumerate(requests)
        if not request.symbol or not request.page_url
    ]
    if invalid_rows:
        raise MorningstarFairValueError(
            f"Missing symbol or page_url in CSV row(s): {', '.join(map(str, invalid_rows))}."
        )
    return requests


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    records = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0]) if records else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(records)


def main() -> None:
    load_dotenv()
    args = parse_args()
    try:
        requests = load_requests(args)
        with MorningstarFairValueAgent() as agent:
            results = agent.fetch_many(
                requests,
                as_of_date=args.as_of,
                delay_seconds=args.delay_seconds,
                jitter_seconds=args.jitter_seconds,
            )
        rows = [result.to_dict() for result in results]
        if args.output_csv:
            write_csv(args.output_csv, rows)
            print(f"Saved {len(rows)} Morningstar fair-value row(s) to {args.output_csv}.")
        else:
            print(json.dumps(rows[0] if len(rows) == 1 else rows, indent=2))
    except MorningstarFairValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
