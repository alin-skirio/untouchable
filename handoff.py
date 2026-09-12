"""Package the current Mac context as a task for the Grok Bot named DESK.

Grok Bot has no public dispatch API. This writes a packet (screenshot + page +
ask), copies an @DESK message, opens the Grok Bot app, and pastes when possible.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import browser
import computer
import presence

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "handoffs"
OUT_FILE = OUT_DIR / "latest.md"
SHOT_FILE = OUT_DIR / "latest.png"
BOT_NAME = "DESK"
GROK_BOT_APPS = ("Grok Bot", "GrokBot")
PAGE_ARCHIVE = 8000
HINT_CHARS = 240
DEBOUNCE_SEC = 8.0

_AND_SPLITTERS = (
    r"send this off to desk\s+and\s+",
    r"send it off to desk\s+and\s+",
    r"send this to desk\s+and\s+",
    r"send it to desk\s+and\s+",
    r"hand this off to desk\s+and\s+",
    r"hand this off\s+and\s+",
    r"handoff to desk\s+and\s+",
    r"hand off to desk\s+and\s+",
    r"task desk\s+and\s+",
    r"ask desk to\s+",
    r"tell desk to\s+",
    r"desk to\s+",
)

_BARE_HANDOFF = re.compile(
    r"(?i)(?:hey\s+grok[,.!?]?\s*)?"
    r"(?:please\s+)?"
    r"(?:send this off to desk|send it off to desk|send this to desk|"
    r"send it to desk|hand this off(?: to desk)?|handoff to desk|"
    r"hand off to desk|task desk|tell desk|ask desk|step away)\s*"
)

_last_handoff_at = 0.0


def _ok(**extra) -> dict:
    return {"ok": True, **extra}


def _err(message: str) -> dict:
    return {"ok": False, "error": message}


def parse_desk_ask(utterance: str) -> str:
    """Pull the work clause after 'send this off to desk and …'."""
    raw = (utterance or "").strip()
    if not raw:
        return ""
    stripped = re.sub(r"(?i)^hey\s+grok[,.!?]?\s*", "", raw).strip()
    low = stripped.lower()
    for pat in _AND_SPLITTERS:
        match = re.search(pat, low)
        if match:
            ask = stripped[match.end() :].strip()
            ask = re.sub(r"^[,\s]+", "", ask)
            ask = re.sub(
                r"^(please\s+)?(continu(?:e|ed|ing)|keep(?:ing)?|go on)\s+",
                "",
                ask,
                flags=re.I,
            )
            return ask.strip()
    leftover = _BARE_HANDOFF.sub("", stripped, count=1).strip()
    leftover = re.sub(r"^and\s+", "", leftover, flags=re.I).strip()
    if leftover.lower() in {"", "this", "it", "please"}:
        return ""
    return leftover


def _copy(text: str) -> bool:
    proc = subprocess.run(
        ["pbcopy"],
        input=text,
        text=True,
        check=False,
        capture_output=True,
    )
    return proc.returncode == 0


def _capture_screen(path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    proc = subprocess.run(
        ["screencapture", "-x", "-t", "png", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0 and path.is_file() and path.stat().st_size > 1000


def _open_grok_bot() -> dict:
    last = _err("Could not open Grok Bot. Install the desktop app, then try again.")
    for name in GROK_BOT_APPS:
        opened = computer.open_app(name)
        if opened.get("ok"):
            opened["app"] = name
            return opened
        last = opened
    return last


def _page_text(page: dict) -> str:
    return str(page.get("text") or page.get("html") or "").strip()


def _clip(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def _facts(ask: str, ctx: dict, page: dict, shot_path: Path | None) -> dict:
    ask = (ask or "Continue the work visible on this screen.").strip()
    front = str(ctx.get("frontmost") or "unknown")
    window = str(ctx.get("window") or "")
    return {
        "ask": ask,
        "front": front,
        "window": window,
        "app": str(page.get("app") or front),
        "url": str(page.get("url") or "").strip(),
        "title": str(page.get("title") or "").strip(),
        "faces": ", ".join(presence.recognized()) or "none",
        "when": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "shot": str(shot_path) if shot_path else "",
        "hint": _clip(_page_text(page), HINT_CHARS),
        "archive": _clip(_page_text(page), PAGE_ARCHIVE),
    }


def _build_paste(facts: dict) -> str:
    """Short group message: DESK assigns; specialists open the URL themselves."""
    url = facts["url"] or "(no URL — use the screenshot on disk)"
    title = facts["title"] or "(no title)"
    shot = facts["shot"] or "(no screenshot)"
    hint = facts["hint"]
    lines = [
        f"@{BOT_NAME}",
        "",
        "Dispatch only. Do not summarize or rewrite the page in this thread.",
        "Your next message must @ one specialist with a short task (ask + URL). Do not paste page HTML or a long excerpt.",
        "",
        f"Ask: {facts['ask']}",
        f"URL: {url}",
        f"Title: {title}",
        f"App: {facts['app']}"
        + (f" — {facts['window']}" if facts["window"] else ""),
        "",
        "Routing:",
        "- Steps, how-to, recipe, flyer, or simplify this page → @Flyer with the ask and URL. Flyer opens the URL on the Bot computer.",
        "- Anything else that matches a specialist here → @ them the same way (ask + URL only).",
        "- No match → reply that a new specialist must be created (name + one-line job). Stop. Do not do the work.",
        "",
        "The person is leaving. The specialist continues without another prompt. Stop only for password, 2FA, CAPTCHA, or payment.",
        f"Screenshot on this Mac (attach or open if you can): {shot}",
    ]
    if hint:
        lines.extend(["", f"One-line context (not the article): {hint}"])
    return "\n".join(lines).strip() + "\n"


def _build_archive(facts: dict) -> str:
    """Full snapshot on disk. Not pasted — specialists should load the URL."""
    excerpt = facts["archive"] or "(No page text captured.)"
    return "\n".join(
        [
            f"Ask: {facts['ask']}",
            f"When: {facts['when']}",
            f"Face: {facts['faces']}",
            f"App: {facts['app']}",
            f"Window: {facts['window'] or '(none)'}",
            f"Title: {facts['title'] or '(none)'}",
            f"URL: {facts['url'] or '(none)'}",
            f"Screenshot: {facts['shot'] or '(none)'}",
            "",
            "Page text (archive only; do not paste this into Grok Bot):",
            excerpt,
            "",
        ]
    )


def handoff_to_desk(ask: str = "", paste: bool = True) -> dict:
    """Snapshot screen + page, copy an @DESK packet, open Grok Bot, optionally paste."""
    global _last_handoff_at

    work = parse_desk_ask(ask) or (ask or "").strip()
    if not work:
        work = "Continue the work visible on this screen."

    now = time.monotonic()
    if now - _last_handoff_at < DEBOUNCE_SEC and OUT_FILE.is_file():
        return _ok(
            bot=BOT_NAME,
            path=str(OUT_FILE),
            screenshot=str(SHOT_FILE) if SHOT_FILE.is_file() else "",
            ask=work,
            debounced=True,
            copied=True,
        )

    ctx = computer.get_context()
    page = browser.get_browser_page()
    if not page.get("ok"):
        page = {
            "ok": False,
            "app": ctx.get("frontmost") or "",
            "url": "",
            "title": ctx.get("window") or "",
            "text": "",
            "error": page.get("error"),
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shot_ok = _capture_screen(SHOT_FILE)
    facts = _facts(work, ctx, page, SHOT_FILE if shot_ok else None)
    brief = _build_paste(facts)
    OUT_FILE.write_text(_build_archive(facts) + "\n--- paste ---\n\n" + brief, encoding="utf-8")

    if not _copy(brief):
        return _err("Could not copy the DESK packet to the clipboard.")

    opened = _open_grok_bot()
    if not opened.get("ok"):
        return {
            "ok": False,
            "error": opened.get("error") or "Could not open Grok Bot.",
            "path": str(OUT_FILE),
            "screenshot": str(SHOT_FILE) if shot_ok else "",
            "copied": True,
            "bot": BOT_NAME,
            "ask": work,
        }

    pasted = False
    sent = False
    if paste:
        time.sleep(1.2)
        pasted = bool(computer.key_combo(["command", "v"]).get("ok"))
        if pasted:
            time.sleep(0.55)
            sent = bool(computer.key_combo(["return"]).get("ok"))

    _last_handoff_at = time.monotonic()
    return _ok(
        bot=BOT_NAME,
        path=str(OUT_FILE),
        screenshot=str(SHOT_FILE) if shot_ok else "",
        app=opened.get("app") or "Grok Bot",
        copied=True,
        pasted=pasted,
        sent=sent,
        url=str(page.get("url") or ""),
        title=str(page.get("title") or ""),
        ask=work,
        page_ok=bool(page.get("ok")),
        screenshot_ok=shot_ok,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Send the current Mac screen and page to Grok Bot DESK")
    parser.add_argument("ask", nargs="*", help="Work for DESK after 'and', or a full voice-style sentence")
    parser.add_argument(
        "--no-paste",
        action="store_true",
        help="Copy and open Grok Bot without pasting",
    )
    args = parser.parse_args()
    ask = " ".join(args.ask).strip()
    result = handoff_to_desk(ask, paste=not args.no_paste)
    if not result.get("ok"):
        print(result.get("error") or "Handoff failed.")
        if result.get("path"):
            print(f"Packet: {result['path']}")
        sys.exit(1)
    print(f"Ask: {result.get('ask')}")
    print(f"Sent to @{result['bot']}. Packet: {result['path']}")
    if result.get("screenshot"):
        print(f"Screenshot: {result['screenshot']}")
    elif not result.get("debounced"):
        print("No screenshot. Enable Screen Recording for Terminal or Cursor, then retry.")
    if result.get("debounced"):
        print("Same handoff was just sent; skipped a duplicate.")
    elif result.get("pasted") and result.get("sent"):
        print("Pasted and submitted in Grok Bot.")
    elif result.get("pasted"):
        print("Pasted but send may have missed. Click the composer and press Return.")
    else:
        print("Copied. Paste into the DESK group chat (⌘V).")


if __name__ == "__main__":
    main()
