"""Welcome pill: one frosted label at the top of the screen.

Shown when the desk unlocks or a different authorized face appears.
The host process displays a single complete string, then fades out.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HOST = ROOT / "overlay_host.applescript"
MAX_NAME = 28

_proc: subprocess.Popen | None = None


def _display_name(name: str) -> str:
    cleaned = " ".join(name.split())
    if not cleaned:
        return ""
    if cleaned == cleaned.lower():
        return cleaned.title()
    return cleaned


def show_welcome(name: str) -> None:
    """Show 'Welcome, {Name}' in a pill that hugs the text."""
    if sys.platform != "darwin" or not HOST.is_file():
        return
    who = _display_name(name)[:MAX_NAME]
    if not who:
        return
    text = f"Welcome, {who}"
    close_welcome()
    try:
        _start_pill(text)
    except OSError:
        pass


def _start_pill(text: str) -> None:
    global _proc
    _proc = subprocess.Popen(
        ["osascript", str(HOST), text],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def close_welcome() -> None:
    global _proc
    if _proc is None:
        return
    if _proc.poll() is None:
        _proc.terminate()
        try:
            _proc.wait(timeout=0.4)
        except subprocess.TimeoutExpired:
            _proc.kill()
    _proc = None


def start() -> None:
    """Kept so older call sites do not launch a status HUD."""
    return


def set_status(_title: str, _detail: str = "", _locked: bool = False) -> None:
    """Status toasts are disabled. Welcome is the only overlay."""
    return


def stop() -> None:
    close_welcome()
