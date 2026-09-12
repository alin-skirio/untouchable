"""Open apps and links for gestures, via the macOS `open` command.

Site-specific launchers live here rather than in computer.py, which is the
generic primitive surface for Grok.
"""

from __future__ import annotations

import subprocess
import sys

TIKTOK_APP = "TikTok"
TIKTOK_URL = "https://www.tiktok.com"


def _open(args: list[str]) -> bool:
    try:
        result = subprocess.run(
            ["open", *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return result.returncode == 0


def open_tiktok() -> bool:
    """Open the TikTok app, falling back to the website when it is not installed."""
    if sys.platform != "darwin":
        print("Opening TikTok requires macOS.")
        return False
    if _open(["-a", TIKTOK_APP]):
        print("Opened TikTok.")
        return True
    if _open([TIKTOK_URL]):
        print(f"TikTok app not found; opened {TIKTOK_URL}")
        return True
    print("Could not open TikTok.")
    return False
