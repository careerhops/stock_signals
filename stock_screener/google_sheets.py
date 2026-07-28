from __future__ import annotations

from base64 import urlsafe_b64encode
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode
from uuid import uuid4

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
import pandas as pd
import requests


GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_SHEETS_SCOPE = "https://www.googleapis.com/auth/spreadsheets"
GOOGLE_OAUTH_SCOPES = [
    GOOGLE_SHEETS_SCOPE,
    "openid",
    "email",
    "profile",
]
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
DEFAULT_WORKSHEET_TITLE = "Weekly Buy Tracker"


@dataclass(frozen=True)
class GoogleSheetsSettings:
    spreadsheet_id: str
    worksheet_title: str
    client_email: str
    configured: bool


@dataclass(frozen=True)
class GoogleOAuthClientSettings:
    client_id: str
    client_secret: str
    configured: bool


@dataclass(frozen=True)
class GoogleOAuthSession:
    refresh_token: str
    user_email: str
    configured: bool


def google_sheets_dir(data_root: Path) -> Path:
    path = data_root / "google_sheets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_google_sheet_target(
    data_root: Path,
    spreadsheet_id: str,
    worksheet_title: str,
) -> GoogleSheetsSettings:
    settings = load_google_sheets_settings(data_root)
    payload = {
        "spreadsheet_id": str(spreadsheet_id).strip(),
        "worksheet_title": str(worksheet_title).strip() or DEFAULT_WORKSHEET_TITLE,
        "client_email": settings.client_email,
        "configured": bool(str(spreadsheet_id).strip()),
    }
    (google_sheets_dir(data_root) / "settings.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return GoogleSheetsSettings(**payload)


def save_google_sheets_credentials(
    data_root: Path,
    service_account_json_text: str,
    spreadsheet_id: str,
    worksheet_title: str,
) -> GoogleSheetsSettings:
    info = json.loads(service_account_json_text)
    required = {"client_email", "private_key", "token_uri"}
    missing = [key for key in required if not str(info.get(key, "")).strip()]
    if missing:
        raise ValueError(f"Missing required service account fields: {', '.join(missing)}")

    directory = google_sheets_dir(data_root)
    (directory / "service_account.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    settings = {
        "spreadsheet_id": str(spreadsheet_id).strip(),
        "worksheet_title": str(worksheet_title).strip() or DEFAULT_WORKSHEET_TITLE,
        "client_email": str(info.get("client_email", "")).strip(),
        "configured": True,
    }
    (directory / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return GoogleSheetsSettings(**settings)


def load_google_sheets_settings(data_root: Path) -> GoogleSheetsSettings:
    directory = google_sheets_dir(data_root)
    settings_path = directory / "settings.json"
    if not settings_path.exists():
        return GoogleSheetsSettings("", DEFAULT_WORKSHEET_TITLE, "", False)
    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return GoogleSheetsSettings("", DEFAULT_WORKSHEET_TITLE, "", False)
    return GoogleSheetsSettings(
        spreadsheet_id=str(payload.get("spreadsheet_id", "")).strip(),
        worksheet_title=str(payload.get("worksheet_title", DEFAULT_WORKSHEET_TITLE)).strip() or DEFAULT_WORKSHEET_TITLE,
        client_email=str(payload.get("client_email", "")).strip(),
        configured=bool(payload.get("configured", False)),
    )


def has_google_sheets_credentials(data_root: Path) -> bool:
    return (google_sheets_dir(data_root) / "service_account.json").exists()


def oauth_client_path(data_root: Path) -> Path:
    return google_sheets_dir(data_root) / "oauth_client.json"


def oauth_session_path(data_root: Path) -> Path:
    return google_sheets_dir(data_root) / "oauth_session.json"


def oauth_state_path(data_root: Path) -> Path:
    return google_sheets_dir(data_root) / "oauth_state.json"


def save_google_oauth_client(data_root: Path, client_id: str, client_secret: str) -> GoogleOAuthClientSettings:
    payload = {
        "client_id": str(client_id).strip(),
        "client_secret": str(client_secret).strip(),
        "configured": bool(str(client_id).strip() and str(client_secret).strip()),
    }
    oauth_client_path(data_root).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return GoogleOAuthClientSettings(**payload)


def load_google_oauth_client(data_root: Path) -> GoogleOAuthClientSettings:
    env_client_id = str(os.getenv("GOOGLE_CLIENT_ID", "")).strip()
    env_client_secret = str(os.getenv("GOOGLE_CLIENT_SECRET", "")).strip()
    env_settings = GoogleOAuthClientSettings(
        client_id=env_client_id,
        client_secret=env_client_secret,
        configured=bool(env_client_id and env_client_secret),
    )
    path = oauth_client_path(data_root)
    if not path.exists():
        return env_settings
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return env_settings
    file_settings = GoogleOAuthClientSettings(
        client_id=str(payload.get("client_id", "")).strip(),
        client_secret=str(payload.get("client_secret", "")).strip(),
        configured=bool(payload.get("configured", False)),
    )
    return file_settings if file_settings.configured else env_settings


def save_google_oauth_session(data_root: Path, refresh_token: str, user_email: str) -> GoogleOAuthSession:
    payload = {
        "refresh_token": str(refresh_token).strip(),
        "user_email": str(user_email).strip(),
        "configured": bool(str(refresh_token).strip()),
    }
    oauth_session_path(data_root).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return GoogleOAuthSession(**payload)


def load_google_oauth_session(data_root: Path) -> GoogleOAuthSession:
    path = oauth_session_path(data_root)
    if not path.exists():
        return GoogleOAuthSession("", "", False)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return GoogleOAuthSession("", "", False)
    return GoogleOAuthSession(
        refresh_token=str(payload.get("refresh_token", "")).strip(),
        user_email=str(payload.get("user_email", "")).strip(),
        configured=bool(payload.get("configured", False)),
    )


def google_oauth_status(data_root: Path) -> dict[str, Any]:
    client = load_google_oauth_client(data_root)
    session = load_google_oauth_session(data_root)
    return {
        "client_configured": client.configured,
        "logged_in": session.configured,
        "user_email": session.user_email,
    }


def build_google_oauth_login_url(
    data_root: Path,
    redirect_uri: str,
    return_to: str,
) -> str:
    client = load_google_oauth_client(data_root)
    if not client.configured:
        raise RuntimeError("Save Google OAuth client ID and client secret first.")
    state = uuid4().hex
    oauth_state_path(data_root).write_text(
        json.dumps(
            {
                "state": state,
                "redirect_uri": redirect_uri,
                "return_to": return_to,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    query = urlencode(
        {
            "client_id": client.client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(GOOGLE_OAUTH_SCOPES),
            "access_type": "offline",
            "prompt": "consent",
            "include_granted_scopes": "true",
            "state": state,
        }
    )
    return f"{GOOGLE_AUTH_URL}?{query}"


def exchange_google_oauth_code(
    data_root: Path,
    code: str,
    state: str,
    redirect_uri: str,
) -> dict[str, Any]:
    client = load_google_oauth_client(data_root)
    if not client.configured:
        raise RuntimeError("Google OAuth client settings are missing.")
    saved_state = _load_google_oauth_state(data_root)
    if not saved_state or str(saved_state.get("state", "")).strip() != str(state).strip():
        raise RuntimeError("Google OAuth state mismatch.")
    expected_redirect_uri = str(saved_state.get("redirect_uri", "")).strip()
    if expected_redirect_uri and expected_redirect_uri != str(redirect_uri).strip():
        raise RuntimeError("Google OAuth redirect URI mismatch.")
    token_response = requests.post(
        GOOGLE_TOKEN_URL,
        data={
            "client_id": client.client_id,
            "client_secret": client.client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
        },
        timeout=30,
    )
    token_response.raise_for_status()
    token_payload = token_response.json()
    refresh_token = str(token_payload.get("refresh_token", "")).strip()
    access_token = str(token_payload.get("access_token", "")).strip()
    if not refresh_token:
        raise RuntimeError("Google did not return a refresh token. Revoke the app and try again with consent.")
    if not access_token:
        raise RuntimeError("Google did not return an access token.")
    user_email = _fetch_google_user_email(access_token)
    save_google_oauth_session(data_root, refresh_token, user_email)
    _delete_google_oauth_state(data_root)
    settings = load_google_sheets_settings(data_root)
    if settings.spreadsheet_id and user_email:
        payload = {
            "spreadsheet_id": settings.spreadsheet_id,
            "worksheet_title": settings.worksheet_title,
            "client_email": user_email,
            "configured": settings.configured,
        }
        (google_sheets_dir(data_root) / "settings.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "return_to": str(saved_state.get("return_to", "")).strip(),
        "user_email": user_email,
    }


def export_weekly_buy_tracker_to_google_sheet(
    data_root: Path,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    settings = load_google_sheets_settings(data_root)
    if not settings.configured or not settings.spreadsheet_id:
        raise RuntimeError("Google Sheets is not configured yet.")
    access_token = _google_sheets_access_token(data_root)
    worksheet_title = settings.worksheet_title or DEFAULT_WORKSHEET_TITLE
    sheet_id = _ensure_worksheet(settings.spreadsheet_id, worksheet_title, access_token)
    _clear_worksheet_values(settings.spreadsheet_id, worksheet_title, access_token)
    values = build_weekly_buy_tracker_sheet_values(frame)
    _write_worksheet_values(settings.spreadsheet_id, worksheet_title, values, access_token)
    _freeze_header_row(settings.spreadsheet_id, sheet_id, access_token)
    return {
        "spreadsheet_id": settings.spreadsheet_id,
        "worksheet_title": worksheet_title,
        "rows_written": max(len(values) - 1, 0),
        "spreadsheet_url": f"https://docs.google.com/spreadsheets/d/{settings.spreadsheet_id}/edit#gid={sheet_id}",
    }


def read_google_sheet_values(
    data_root: Path,
    spreadsheet_id: str,
    worksheet_title: str,
    cell_range: str = "A:ZZ",
) -> list[list[Any]]:
    access_token = _google_sheets_access_token(data_root)
    safe_range = quote(f"{worksheet_title}!{cell_range}", safe="")
    response = requests.get(
        f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/{safe_range}",
        headers=_google_headers(access_token),
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    values = payload.get("values", [])
    return values if isinstance(values, list) else []


def batch_update_google_sheet_values(
    data_root: Path,
    spreadsheet_id: str,
    updates: list[dict[str, Any]],
) -> dict[str, Any]:
    access_token = _google_sheets_access_token(data_root)
    if not updates:
        return {"updated_ranges": 0}
    response = requests.post(
        f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values:batchUpdate",
        headers=_google_headers(access_token),
        json={
            "valueInputOption": "USER_ENTERED",
            "data": updates,
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {"updated_ranges": 0}


def google_sheet_worksheet_id(data_root: Path, spreadsheet_id: str, worksheet_title: str) -> int:
    access_token = _google_sheets_access_token(data_root)
    metadata = _get_spreadsheet_metadata(spreadsheet_id, access_token)
    sheets = metadata.get("sheets", []) if isinstance(metadata, dict) else []
    for sheet in sheets:
        properties = sheet.get("properties", {})
        if str(properties.get("title", "")).strip() == str(worksheet_title).strip():
            return int(properties.get("sheetId"))
    raise RuntimeError(f'Worksheet "{worksheet_title}" was not found in the target spreadsheet.')


def build_weekly_buy_tracker_sheet_values(frame: pd.DataFrame) -> list[list[Any]]:
    headers = [
        "Exchange",
        "Symbol",
        "Name",
        "Latest Local Close",
        "Latest Local Close Date",
        "S2 Buy Count",
        "S3 Buy Count",
        "Total Buy Count",
        "First BUY Date",
        "First BUY Price",
        "Latest BUY Date",
        "Latest BUY Price",
        "Latest S2 BUY Date",
        "Latest S2 BUY Price",
        "Latest S3 BUY Date",
        "Latest S3 BUY Price",
        "GoogleFinance Current Price",
        "Effective Current Price",
        "Gain vs First BUY %",
        "Gain vs Latest BUY %",
        "Gain vs Latest S2 BUY %",
        "Gain vs Latest S3 BUY %",
    ]
    values: list[list[Any]] = [headers]
    if frame.empty:
        return values

    working = frame.copy().reset_index(drop=True)
    for index, row in working.iterrows():
        sheet_row = index + 2
        row_values = [
            row.get("exchange", ""),
            row.get("symbol", ""),
            row.get("name", ""),
            _sheet_scalar(row.get("latest_close")),
            _sheet_date(row.get("latest_close_date")),
            _sheet_int(row.get("s2_buy_count")),
            _sheet_int(row.get("s3_buy_count")),
            _sheet_int(row.get("total_buy_count")),
            _sheet_date(row.get("first_buy_date")),
            _sheet_scalar(row.get("first_buy_price")),
            _sheet_date(row.get("latest_buy_date")),
            _sheet_scalar(row.get("latest_buy_price")),
            _sheet_date(row.get("latest_s2_buy_date")),
            _sheet_scalar(row.get("latest_s2_buy_price")),
            _sheet_date(row.get("latest_s3_buy_date")),
            _sheet_scalar(row.get("latest_s3_buy_price")),
            f'=IFERROR(GOOGLEFINANCE(A{sheet_row}&":"&B{sheet_row},"price"),"")',
            f'=IF(Q{sheet_row}="",D{sheet_row},Q{sheet_row})',
            f'=IFERROR((R{sheet_row}-J{sheet_row})/J{sheet_row},"")',
            f'=IFERROR((R{sheet_row}-L{sheet_row})/L{sheet_row},"")',
            f'=IFERROR((R{sheet_row}-N{sheet_row})/N{sheet_row},"")',
            f'=IFERROR((R{sheet_row}-P{sheet_row})/P{sheet_row},"")',
        ]
        values.append(row_values)
    return values


def _load_service_account(data_root: Path) -> dict[str, Any]:
    credentials_path = google_sheets_dir(data_root) / "service_account.json"
    if not credentials_path.exists():
        raise RuntimeError("Google Sheets credentials are not saved yet.")
    try:
        return json.loads(credentials_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Saved Google Sheets credentials are unreadable.") from exc


def _google_sheets_access_token(data_root: Path) -> str:
    session = load_google_oauth_session(data_root)
    if session.configured:
        client = load_google_oauth_client(data_root)
        if not client.configured:
            raise RuntimeError("Google OAuth client settings are missing.")
        response = requests.post(
            GOOGLE_TOKEN_URL,
            data={
                "client_id": client.client_id,
                "client_secret": client.client_secret,
                "refresh_token": session.refresh_token,
                "grant_type": "refresh_token",
            },
            timeout=30,
        )
        response.raise_for_status()
        token_payload = response.json()
        access_token = str(token_payload.get("access_token", "")).strip()
        if not access_token:
            raise RuntimeError("Google refresh token exchange did not return an access token.")
        return access_token

    if has_google_sheets_credentials(data_root):
        service_account = _load_service_account(data_root)
        return _service_account_access_token(service_account)
    raise RuntimeError("Google Sheets authentication is not configured yet.")


def _service_account_access_token(info: dict[str, Any]) -> str:
    issued_at = datetime.now(timezone.utc)
    expires_at = issued_at + timedelta(hours=1)
    payload = {
        "iss": info["client_email"],
        "scope": GOOGLE_SHEETS_SCOPE,
        "aud": info.get("token_uri", GOOGLE_TOKEN_URL),
        "exp": int(expires_at.timestamp()),
        "iat": int(issued_at.timestamp()),
    }
    assertion = _jwt_assertion(payload, info["private_key"])
    response = requests.post(
        info.get("token_uri", GOOGLE_TOKEN_URL),
        data={
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "assertion": assertion,
        },
        timeout=30,
    )
    response.raise_for_status()
    token_payload = response.json()
    access_token = str(token_payload.get("access_token", "")).strip()
    if not access_token:
        raise RuntimeError("Google access token response did not include an access token.")
    return access_token


def _fetch_google_user_email(access_token: str) -> str:
    response = requests.get(
        GOOGLE_USERINFO_URL,
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    return str(payload.get("email", "")).strip()


def _load_google_oauth_state(data_root: Path) -> dict[str, Any]:
    path = oauth_state_path(data_root)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _delete_google_oauth_state(data_root: Path) -> None:
    path = oauth_state_path(data_root)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _jwt_assertion(payload: dict[str, Any], private_key_pem: str) -> str:
    header = {"alg": "RS256", "typ": "JWT"}
    encoded_header = _b64url(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    encoded_payload = _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{encoded_header}.{encoded_payload}".encode("utf-8")
    private_key = serialization.load_pem_private_key(private_key_pem.encode("utf-8"), password=None)
    signature = private_key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
    encoded_signature = _b64url(signature)
    return f"{encoded_header}.{encoded_payload}.{encoded_signature}"


def _ensure_worksheet(spreadsheet_id: str, worksheet_title: str, access_token: str) -> int:
    metadata = _get_spreadsheet_metadata(spreadsheet_id, access_token)
    sheets = metadata.get("sheets", []) if isinstance(metadata, dict) else []
    for sheet in sheets:
        properties = sheet.get("properties", {})
        if str(properties.get("title", "")).strip() == worksheet_title:
            return int(properties.get("sheetId"))
    response = requests.post(
        f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}:batchUpdate",
        headers=_google_headers(access_token),
        json={
            "requests": [
                {
                    "addSheet": {
                        "properties": {
                            "title": worksheet_title,
                        }
                    }
                }
            ]
        },
        timeout=30,
    )
    response.raise_for_status()
    replies = response.json().get("replies", [])
    if not replies:
        raise RuntimeError("Google Sheets did not return worksheet creation details.")
    return int(replies[0]["addSheet"]["properties"]["sheetId"])


def _get_spreadsheet_metadata(spreadsheet_id: str, access_token: str) -> dict[str, Any]:
    response = requests.get(
        f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}",
        headers=_google_headers(access_token),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _clear_worksheet_values(spreadsheet_id: str, worksheet_title: str, access_token: str) -> None:
    sheet_range = quote(f"{worksheet_title}!A:Z", safe="")
    response = requests.post(
        f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/{sheet_range}:clear",
        headers=_google_headers(access_token),
        timeout=30,
    )
    response.raise_for_status()


def _write_worksheet_values(spreadsheet_id: str, worksheet_title: str, values: list[list[Any]], access_token: str) -> None:
    safe_title = quote(f"{worksheet_title}!A1", safe="")
    response = requests.put(
        f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/{safe_title}?valueInputOption=USER_ENTERED",
        headers=_google_headers(access_token),
        json={"range": f"{worksheet_title}!A1", "majorDimension": "ROWS", "values": values},
        timeout=60,
    )
    response.raise_for_status()


def _freeze_header_row(spreadsheet_id: str, sheet_id: int, access_token: str) -> None:
    response = requests.post(
        f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}:batchUpdate",
        headers=_google_headers(access_token),
        json={
            "requests": [
                {
                    "updateSheetProperties": {
                        "properties": {
                            "sheetId": int(sheet_id),
                            "gridProperties": {
                                "frozenRowCount": 1,
                            },
                        },
                        "fields": "gridProperties.frozenRowCount",
                    }
                }
            ]
        },
        timeout=30,
    )
    response.raise_for_status()


def _google_headers(access_token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }


def _b64url(value: bytes) -> str:
    return urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _sheet_date(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    converted = pd.to_datetime(value, errors="coerce")
    if pd.isna(converted):
        return ""
    return converted.strftime("%Y-%m-%d")


def _sheet_scalar(value: Any) -> Any:
    if value is None or pd.isna(value):
        return ""
    try:
        numeric = float(value)
        return numeric
    except (TypeError, ValueError):
        return str(value)


def _sheet_int(value: Any) -> int:
    try:
        if pd.isna(value):
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0
