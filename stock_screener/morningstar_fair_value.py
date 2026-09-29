from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from html.parser import HTMLParser
import json
import os
import random
import re
import time
from typing import Any, Iterable
from urllib.parse import urlparse
from uuid import uuid4
import xml.etree.ElementTree as ET

import httpx


DEFAULT_MORNINGSTAR_BASE_URL = "https://www.morningstar.in"
DEFAULT_SAL_BASE_URL = "https://www.us-api.morningstar.com/sal/sal-service"
DEFAULT_CLIENT_ID = "RSIN_SAL"
DEFAULT_COMPONENT_VERSION = "4.86.0"
DEFAULT_PREMIUM_SAL_CONTENT_TYPE = "nNsGdN3REOnPMlKDShOYjlk6VYiEVLSdpfpXAm7o2Tk="
FAIR_VALUE_COMPONENT = "sal-price-fairvalue"
INVESTMENT_ID_PATTERN = re.compile(r"/stocks/(?P<investment_id>0p[0-9a-z]+)/", re.IGNORECASE)
NSE_SERIES_SUFFIXES = {"EQ", "BE", "BZ", "BL", "BT", "SM", "ST", "SZ", "IL", "IT", "IQ", "E1", "GC", "RL"}


class MorningstarFairValueError(RuntimeError):
    pass


@dataclass(frozen=True)
class FairValueRequest:
    symbol: str
    page_url: str


@dataclass(frozen=True)
class MorningstarStockPage:
    symbol: str
    lookup_symbol: str
    investment_id: str
    company_name: str
    exchange: str
    page_url: str


@dataclass(frozen=True)
class FairValueResult:
    symbol: str
    investment_id: str
    current_month: str
    current_fair_value: float | None
    previous_month: str
    previous_fair_value: float | None
    current_market_price: float | None
    current_market_price_date: str
    fair_value_market_price_difference: float | None
    fair_value_market_price_difference_pct: float | None
    fair_value_change: float | None
    fair_value_change_pct: float | None
    currency: str
    as_of_date: str
    latest_available_month: str
    source_url: str
    fetched_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _MorningstarApiAuth:
    access_token: str
    sal_content_type: str
    realtime_token: str
    page_investment_id: str


class MorningstarFairValueAgent:
    """Fetch monthly fair value from Morningstar's SAL service."""

    def __init__(
        self,
        *,
        access_token: str | None = None,
        sal_content_type: str | None = None,
        realtime_token: str | None = None,
        base_url: str | None = None,
        client_id: str = DEFAULT_CLIENT_ID,
        component_version: str = DEFAULT_COMPONENT_VERSION,
        locale: str = "en",
        timeout_seconds: float = 30.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.access_token = str(access_token or os.getenv("MORNINGSTAR_ACCESS_TOKEN", "")).strip()
        self.sal_content_type = str(
            sal_content_type
            or os.getenv("MORNINGSTAR_SAL_CONTENT_TYPE", "")
            or DEFAULT_PREMIUM_SAL_CONTENT_TYPE
        ).strip()
        self.realtime_token = str(
            realtime_token or os.getenv("MORNINGSTAR_REALTIME_TOKEN", "")
        ).strip()
        self.base_url = str(
            base_url or os.getenv("MORNINGSTAR_SAL_BASE_URL", DEFAULT_SAL_BASE_URL)
        ).rstrip("/")
        self.client_id = str(client_id or DEFAULT_CLIENT_ID).strip()
        self.component_version = str(component_version or DEFAULT_COMPONENT_VERSION).strip()
        self.locale = str(locale or "en").strip()
        self._owns_client = http_client is None
        self.http_client = http_client or httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=True,
            headers={
                "Accept": "application/json, text/plain, */*",
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/123.0.0.0 Safari/537.36"
                ),
            },
        )

    def __enter__(self) -> MorningstarFairValueAgent:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self.http_client.close()

    def fetch(
        self,
        request: FairValueRequest,
        *,
        as_of_date: date | None = None,
    ) -> FairValueResult:
        investment_id = investment_id_from_url(request.page_url)
        auth = self._resolve_api_auth(investment_id, request.page_url)
        payload = self._fetch_payload(investment_id, request.page_url, auth=auth)
        return extract_monthly_fair_values(
            payload,
            symbol=request.symbol,
            investment_id=investment_id,
            source_url=request.page_url,
            as_of_date=as_of_date,
        )

    def fetch_many(
        self,
        requests: Iterable[FairValueRequest],
        *,
        as_of_date: date | None = None,
        delay_seconds: float = 0.0,
        jitter_seconds: float = 0.0,
    ) -> list[FairValueResult]:
        request_list = list(requests)
        try:
            delay_seconds = max(float(delay_seconds), 0.0)
        except (TypeError, ValueError):
            delay_seconds = 0.0
        if delay_seconds != delay_seconds:
            delay_seconds = 0.0
        try:
            jitter_seconds = max(float(jitter_seconds), 0.0)
        except (TypeError, ValueError):
            jitter_seconds = 0.0
        if jitter_seconds != jitter_seconds:
            jitter_seconds = 0.0

        results: list[FairValueResult] = []
        for index, request in enumerate(request_list):
            results.append(self.fetch(request, as_of_date=as_of_date))
            if index < len(request_list) - 1 and (delay_seconds > 0 or jitter_seconds > 0):
                time.sleep(delay_seconds + random.uniform(0, jitter_seconds))
        return results

    def _resolve_api_auth(self, investment_id: str, page_url: str) -> _MorningstarApiAuth:
        access_token = self.access_token
        sal_content_type = self.sal_content_type
        realtime_token = self.realtime_token
        page_investment_id = ""

        if not access_token or not realtime_token:
            page_auth = self._fetch_page_auth(page_url)
            access_token = access_token or page_auth.access_token
            realtime_token = realtime_token or page_auth.realtime_token
            sal_content_type = sal_content_type or page_auth.sal_content_type
            page_investment_id = page_auth.page_investment_id

        if page_investment_id and page_investment_id.upper() != investment_id.upper():
            raise MorningstarFairValueError(
                f"Morningstar page resolved to {page_investment_id}, not {investment_id}."
            )
        if not access_token:
            raise MorningstarFairValueError(
                "Morningstar page did not expose an access token. Open the stock page in a "
                "browser to confirm it loads, or provide MORNINGSTAR_ACCESS_TOKEN."
            )
        if not sal_content_type:
            raise MorningstarFairValueError(
                "Morningstar SAL content type is missing. Provide MORNINGSTAR_SAL_CONTENT_TYPE."
            )
        return _MorningstarApiAuth(
            access_token=access_token,
            sal_content_type=sal_content_type,
            realtime_token=realtime_token,
            page_investment_id=page_investment_id,
        )

    def _fetch_page_auth(self, page_url: str) -> _MorningstarApiAuth:
        try:
            response = self.http_client.get(page_url)
            response.raise_for_status()
        except httpx.RequestError as exc:
            raise MorningstarFairValueError(
                f"Could not reach the Morningstar stock page: {exc}."
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise MorningstarFairValueError(
                f"Morningstar stock page request failed with HTTP {response.status_code}."
            ) from exc

        meta = _extract_meta(response.text)
        return _MorningstarApiAuth(
            access_token=str(meta.get("accessToken") or "").strip(),
            sal_content_type=self.sal_content_type or DEFAULT_PREMIUM_SAL_CONTENT_TYPE,
            realtime_token=str(meta.get("realTimeToken") or "").strip(),
            page_investment_id=str(meta.get("Ticker") or "").strip().upper(),
        )

    def _fetch_payload(
        self,
        investment_id: str,
        page_url: str,
        *,
        auth: _MorningstarApiAuth,
    ) -> dict[str, Any]:
        endpoint = f"{self.base_url}/stock/priceFairValueChart/{investment_id}/data"
        tracking = {
            "userId": "",
            "sessionId": "",
            "applicationArea": "",
            "requestId": "",
            "action": "",
            "component": "",
            "actionValue": "",
            "userType": "premium",
            "viewMode": "",
            "env": "prod",
            "subApplicationArea": "",
            "subComponent": FAIR_VALUE_COMPONENT,
            "securityType": "ST",
        }
        headers = {
            "Authorization": f"Bearer {auth.access_token}",
            "X-SAL-ContentType": auth.sal_content_type,
            "X-API-RequestId": str(uuid4()),
            "X-Requested-With": json.dumps(tracking, separators=(",", ":")),
            "Referer": page_url,
        }
        if auth.realtime_token:
            headers["X-API-REALTIME-E"] = auth.realtime_token
        try:
            response = self.http_client.get(
                endpoint,
                params={
                    "secExchangeList": "",
                    "locale": self.locale,
                    "clientId": self.client_id,
                    "component": FAIR_VALUE_COMPONENT,
                    "version": self.component_version,
                },
                headers=headers,
            )
        except httpx.RequestError as exc:
            raise MorningstarFairValueError(
                f"Could not reach Morningstar's fair-value service: {exc}."
            ) from exc
        if response.status_code in {401, 403}:
            raise MorningstarFairValueError(
                "Morningstar rejected the credentials or the account is not entitled "
                "to fair-value data."
            )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise MorningstarFairValueError(
                f"Morningstar fair-value request failed with HTTP {response.status_code}."
            ) from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise MorningstarFairValueError("Morningstar returned a non-JSON response.") from exc
        if not isinstance(payload, dict):
            raise MorningstarFairValueError(
                "Morningstar returned an unexpected fair-value payload."
            )
        return payload


def investment_id_from_url(page_url: str) -> str:
    parsed = urlparse(str(page_url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise MorningstarFairValueError("A valid Morningstar stock page URL is required.")
    if not parsed.netloc.lower().endswith("morningstar.in"):
        raise MorningstarFairValueError("The page URL must be on morningstar.in.")
    match = INVESTMENT_ID_PATTERN.search(parsed.path)
    if not match:
        raise MorningstarFairValueError(
            "Could not find the Morningstar investment ID in the page URL."
        )
    return match.group("investment_id").upper()


def resolve_morningstar_stock_page(
    symbol: str,
    *,
    exchange: str = "NSE",
    base_url: str = DEFAULT_MORNINGSTAR_BASE_URL,
    http_client: httpx.Client | None = None,
) -> MorningstarStockPage:
    """Resolve an exchange ticker to Morningstar India's stock price page."""

    requested_symbol = str(symbol or "").strip().upper()
    if not requested_symbol:
        raise MorningstarFairValueError("A stock symbol is required.")
    lookup_symbol = _lookup_symbol(requested_symbol)
    target_exchange = str(exchange or "NSE").strip().upper()
    base_url = str(base_url or DEFAULT_MORNINGSTAR_BASE_URL).rstrip("/")
    owns_client = http_client is None
    client = http_client or httpx.Client(timeout=30.0, follow_redirects=True)
    try:
        try:
            response = client.get(
                f"{base_url}/handlers/autocompletehandler.ashx",
                params={"criteria": lookup_symbol},
                headers={
                    "Accept": "application/xml,text/xml,*/*",
                    "Referer": f"{base_url}/",
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/123.0.0.0 Safari/537.36"
                    ),
                },
            )
            response.raise_for_status()
        except httpx.RequestError as exc:
            raise MorningstarFairValueError(
                f"Could not resolve {requested_symbol} on Morningstar: {exc}."
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise MorningstarFairValueError(
                f"Morningstar symbol lookup failed for {requested_symbol} with HTTP "
                f"{response.status_code}."
            ) from exc

        rows = _parse_autocomplete_rows(response.text)
        match = next(
            (
                row
                for row in rows
                if row.get("type", "").lower() == "stock"
                and row.get("ticker", "").upper() == lookup_symbol
                and row.get("exchange", "").upper() == target_exchange
                and INVESTMENT_ID_PATTERN.fullmatch(f"/stocks/{row.get('id', '').lower()}/")
                is not None
            ),
            None,
        )
        if match is None:
            choices = ", ".join(
                f"{row.get('ticker', '')}/{row.get('exchange', '')}/{row.get('type', '')}"
                for row in rows[:5]
            )
            suffix = f" Matches seen: {choices}." if choices else ""
            raise MorningstarFairValueError(
                f"Could not find an exact {target_exchange} stock match for {requested_symbol}."
                f"{suffix}"
            )

        investment_id = str(match["id"]).strip().upper()
        company_name = str(match.get("description") or "").strip()
        page_url = (
            f"{base_url}/stocks/{investment_id.lower()}/"
            f"{_stock_page_slug(target_exchange, company_name)}/price.aspx"
        )
        return MorningstarStockPage(
            symbol=requested_symbol,
            lookup_symbol=lookup_symbol,
            investment_id=investment_id,
            company_name=company_name,
            exchange=target_exchange,
            page_url=page_url,
        )
    finally:
        if owns_client:
            client.close()


class _MetaParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.values: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "meta":
            return
        values = {str(key).lower(): value or "" for key, value in attrs}
        name = values.get("name")
        if name:
            self.values[name] = values.get("content", "")


def _extract_meta(html: str) -> dict[str, str]:
    parser = _MetaParser()
    parser.feed(html or "")
    return parser.values


def _parse_autocomplete_rows(xml_text: str) -> list[dict[str, str]]:
    try:
        root = ET.fromstring(str(xml_text or "").strip())
    except ET.ParseError as exc:
        raise MorningstarFairValueError("Morningstar symbol lookup returned invalid XML.") from exc

    def local_name(tag: str) -> str:
        return str(tag).split("}")[-1]

    rows: list[dict[str, str]] = []
    for element in root.iter():
        if local_name(element.tag) != "Table":
            continue
        values = {local_name(child.tag): (child.text or "").strip() for child in element}
        rows.append(
            {
                "id": values.get("ID", ""),
                "type": values.get("Type", ""),
                "ticker": values.get("Ticker", ""),
                "description": values.get("Description", ""),
                "exchange": values.get("Exchange", ""),
            }
        )
    return rows


def _lookup_symbol(symbol: str) -> str:
    base, separator, suffix = str(symbol or "").strip().upper().rpartition("-")
    return base if separator and base and suffix in NSE_SERIES_SUFFIXES else str(symbol or "").strip().upper()


def _stock_page_slug(exchange: str, company_name: str) -> str:
    text = f"{exchange} {company_name}".lower()
    slug = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return slug or str(exchange or "nse").lower()


def extract_monthly_fair_values(
    payload: dict[str, Any],
    *,
    symbol: str,
    investment_id: str,
    source_url: str,
    as_of_date: date | None = None,
) -> FairValueResult:
    chart = payload.get("chart") if isinstance(payload.get("chart"), dict) else payload
    chart_datums = chart.get("chartDatums") if isinstance(chart, dict) else None
    if not isinstance(chart_datums, dict):
        raise MorningstarFairValueError(
            "Morningstar payload does not contain fair-value chart data."
        )

    monthly_values: dict[str, tuple[datetime, float]] = {}
    yearly = chart_datums.get("yearly")
    if isinstance(yearly, list):
        for year_record in yearly:
            if not isinstance(year_record, dict):
                continue
            monthly = year_record.get("monthly")
            if not isinstance(monthly, list):
                continue
            for record in monthly:
                parsed = _parse_monthly_record(record)
                if parsed is None:
                    continue
                record_date, fair_value = parsed
                month_key = record_date.strftime("%Y-%m")
                previous = monthly_values.get(month_key)
                if previous is None or record_date >= previous[0]:
                    monthly_values[month_key] = (record_date, fair_value)

    recent = chart_datums.get("recent") if isinstance(chart_datums.get("recent"), dict) else {}
    recent_value = _as_number(recent.get("latestFairValue"))
    recent_date = _first_date(
        recent.get("asOfClosePriceDate"),
        recent.get("asOfDate"),
        _list_item(recent.get("bf"), 0),
        _nested(payload, "footer", "asOfDate"),
    )
    if recent_value is not None and recent_date is not None:
        monthly_values[recent_date.strftime("%Y-%m")] = (recent_date, recent_value)

    if not monthly_values:
        raise MorningstarFairValueError("Morningstar returned no numeric monthly fair values.")

    target_date = as_of_date or date.today()
    current_month = target_date.strftime("%Y-%m")
    previous_month_date = _previous_month(target_date)
    previous_month = previous_month_date.strftime("%Y-%m")
    current_value = monthly_values.get(current_month, (None, None))[1]
    previous_value = monthly_values.get(previous_month, (None, None))[1]
    latest_available_month = max(monthly_values)
    current_market_price = _as_number(recent.get("latestClose"), recent.get("close"))
    current_market_price_date = ""
    market_price_date = _first_date(
        recent.get("asOfClosePriceDate"),
        recent.get("latestCloseDate"),
        _nested(payload, "footer", "asOfDate"),
    )
    if market_price_date is not None:
        current_market_price_date = market_price_date.date().isoformat()
    fair_value_market_price_difference = None
    fair_value_market_price_difference_pct = None
    if current_value is not None and current_market_price is not None:
        fair_value_market_price_difference = current_value - current_market_price
        if current_market_price != 0:
            fair_value_market_price_difference_pct = (
                fair_value_market_price_difference / current_market_price
            ) * 100.0
    change = None
    change_pct = None
    if current_value is not None and previous_value is not None:
        change = current_value - previous_value
        if previous_value != 0.0:
            change_pct = (current_value / previous_value - 1.0) * 100.0

    currency = ""
    if isinstance(chart, dict):
        currency = str(chart.get("fairValCurrency") or chart.get("currency") or "").strip()
    if not currency:
        currency = str(payload.get("currency") or "").strip()

    return FairValueResult(
        symbol=str(symbol or "").strip().upper(),
        investment_id=str(investment_id or "").strip().upper(),
        current_month=current_month,
        current_fair_value=current_value,
        previous_month=previous_month,
        previous_fair_value=previous_value,
        current_market_price=current_market_price,
        current_market_price_date=current_market_price_date,
        fair_value_market_price_difference=fair_value_market_price_difference,
        fair_value_market_price_difference_pct=fair_value_market_price_difference_pct,
        fair_value_change=change,
        fair_value_change_pct=change_pct,
        currency=currency,
        as_of_date=target_date.isoformat(),
        latest_available_month=latest_available_month,
        source_url=source_url,
        fetched_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    )


def _parse_monthly_record(record: Any) -> tuple[datetime, float] | None:
    if not isinstance(record, dict):
        return None
    bands = record.get("bf")
    record_date = _first_date(
        record.get("fairValueMonthlyDate"),
        _list_item(bands, 0),
        record.get("date"),
        record.get("asOfDate"),
    )
    # Morningstar's fair-value chart component reads the central fair value from bf[3].
    fair_value = _as_number(
        _list_item(bands, 3),
        record.get("fairValue"),
        record.get("fairVal"),
    )
    if record_date is None or fair_value is None:
        return None
    return record_date, fair_value


def _first_date(*values: Any) -> datetime | None:
    for value in values:
        parsed = _as_datetime(value)
        if parsed is not None:
            return parsed
    return None


def _as_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    text = str(value).strip()
    if not text:
        return None
    milliseconds = re.fullmatch(r"/Date\((\d+)(?:[+-]\d+)?\)/", text)
    if milliseconds:
        return datetime.fromtimestamp(
            int(milliseconds.group(1)) / 1000.0,
            tz=timezone.utc,
        ).replace(tzinfo=None)
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            return parsed
        return parsed.astimezone(timezone.utc).replace(tzinfo=None)
    except ValueError:
        pass
    for date_format in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text, date_format)
        except ValueError:
            continue
    return None


def _as_number(*values: Any) -> float | None:
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        try:
            parsed = float(str(value).replace(",", "").strip())
        except (TypeError, ValueError):
            continue
        if parsed == parsed and parsed not in {float("inf"), float("-inf")}:
            return parsed
    return None


def _list_item(value: Any, index: int) -> Any:
    return value[index] if isinstance(value, list) and len(value) > index else None


def _nested(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _previous_month(value: date) -> date:
    if value.month == 1:
        return date(value.year - 1, 12, 1)
    return date(value.year, value.month - 1, 1)
