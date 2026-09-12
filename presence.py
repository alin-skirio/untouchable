"""Shared lock: a known face must be in the camera for controls to work."""

from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PATH = ROOT / ".presence.json"
STALE_SEC = 1.8


def publish(names: list[str]) -> None:
    payload = {"names": list(names), "at": time.time()}
    PATH.write_text(json.dumps(payload), encoding="utf-8")


def clear() -> None:
    PATH.unlink(missing_ok=True)


def recognized() -> list[str]:
    try:
        data = json.loads(PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    try:
        age = time.time() - float(data.get("at") or 0)
    except (TypeError, ValueError):
        return []
    if age > STALE_SEC:
        return []
    names = data.get("names") or []
    if not isinstance(names, list):
        return []
    return [str(name) for name in names if str(name).strip()]


def unlocked() -> bool:
    return bool(recognized())
