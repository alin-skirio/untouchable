"""Shared lock: a known face arms commands; only a lone stranger locks them.

A familiar face unlocks immediately. An empty frame does not lock. Commands
turn off if and only if every visible face is unidentified for 5 seconds.
Voice reads the same file the tracker writes each frame.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PATH = ROOT / ".presence.json"
STALE_SEC = 1.8
LOCK_AFTER_UNKNOWN_SEC = 5.0

_unlocked = False
_unknown_since: float | None = None


def publish(names: list[str], unlocked: bool | None = None) -> None:
    if unlocked is None:
        unlocked = bool(names)
    payload = {
        "names": list(names),
        "unlocked": bool(unlocked),
        "at": time.time(),
    }
    remaining = lock_in_sec()
    if remaining is not None:
        payload["lock_in"] = remaining
    PATH.write_text(json.dumps(payload), encoding="utf-8")


def clear() -> None:
    global _unlocked, _unknown_since
    _unlocked = False
    _unknown_since = None
    PATH.unlink(missing_ok=True)


def update(known: list[str], unknown: int) -> bool:
    """Refresh session lock from this camera frame. Returns whether commands are live."""
    global _unlocked, _unknown_since
    now = time.monotonic()
    names = [str(name) for name in known if str(name).strip()]
    if names:
        _unlocked = True
        _unknown_since = None
    elif unknown > 0:
        if _unknown_since is None:
            _unknown_since = now
        if now - _unknown_since >= LOCK_AFTER_UNKNOWN_SEC:
            _unlocked = False
    else:
        _unknown_since = None
    publish(names, _unlocked)
    return _unlocked


def lock_in_sec() -> float | None:
    """Seconds until a lone stranger locks commands, or None if that is not happening."""
    if _unknown_since is None or not _unlocked:
        return None
    remaining = LOCK_AFTER_UNKNOWN_SEC - (time.monotonic() - _unknown_since)
    return max(0.0, remaining)


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
    try:
        data = json.loads(PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    try:
        age = time.time() - float(data.get("at") or 0)
    except (TypeError, ValueError):
        return False
    if age > STALE_SEC:
        return False
    if "unlocked" in data:
        return bool(data.get("unlocked"))
    names = data.get("names") or []
    if not isinstance(names, list):
        return False
    return any(str(name).strip() for name in names)
