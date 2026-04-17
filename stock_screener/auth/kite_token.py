from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def token_path(data_root: Path) -> Path:
    directory = data_root / "secrets"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "kite_access_token.json"


def save_access_token(data_root: Path, access_token: str, profile: dict[str, Any] | None = None) -> Path:
    path = token_path(data_root)
    payload = {
        "access_token": access_token,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "profile": profile or {},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_access_token(data_root: Path) -> str | None:
    path = token_path(data_root)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        token = payload.get("access_token")
        if token:
            return str(token)

    return os.getenv("KITE_ACCESS_TOKEN") or None


def token_status(data_root: Path) -> dict[str, Any]:
    path = token_path(data_root)
    if not path.exists():
        return {"exists": False, "generated_at": None, "profile": {}}

    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "exists": True,
        "generated_at": payload.get("generated_at"),
        "profile": payload.get("profile", {}),
    }
