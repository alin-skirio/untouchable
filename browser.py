"""Read the front Zen, Safari, or Chrome tab as URL, title, and HTML."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from urllib.request import Request, urlopen

ZEN_SUPPORT = Path.home() / "Library/Application Support/zen"
PAGE_TEXT_MAX = 16000


def _ok(**extra) -> dict:
    return {"ok": True, **extra}


def _err(message: str) -> dict:
    return {"ok": False, "error": message}


def compact_html(html: str, max_chars: int = PAGE_TEXT_MAX) -> str:
    if not html:
        return ""
    markup = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", html)
    markup = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", markup)
    markup = re.sub(r"(?is)<noscript[^>]*>.*?</noscript>", " ", markup)
    markup = re.sub(r"(?is)<!--.*?-->", " ", markup)
    markup = re.sub(r"\s+", " ", markup)
    return markup.strip()[: max(1000, min(int(max_chars), 16_000))]


def html_to_text(html: str, max_chars: int = PAGE_TEXT_MAX) -> str:
    import html as html_lib

    if not html:
        return ""
    markup = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", html)
    markup = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", markup)
    markup = re.sub(r"(?is)<br\s*/?>", "\n", markup)
    markup = re.sub(r"(?is)</p>", "\n", markup)
    markup = re.sub(r"(?is)<[^>]+>", " ", markup)
    text = html_lib.unescape(markup)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return text[:max_chars]


def _osascript(script: str, timeout: float = 8) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["osascript", "-e", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _parse_url_title(raw: str) -> tuple[str, str]:
    parts = (raw or "").split("\n", 1)
    url = parts[0].strip() if parts else ""
    title = parts[1].strip() if len(parts) > 1 else ""
    return url, title


def _zen_running() -> bool:
    proc = _osascript(
        'tell application "System Events" to get name of every process'
    )
    names = [part.strip().lower() for part in (proc.stdout or "").split(",")]
    return proc.returncode == 0 and any(name in {"zen", "zen browser"} for name in names)


def _zen_window_title() -> str:
    proc = _osascript(
        'tell application "System Events" to tell process "zen" to get name of front window'
    )
    title = (proc.stdout or "").strip()
    for suffix in (" — Zen Browser", " - Zen Browser", " — Zen", " - Zen"):
        if title.endswith(suffix):
            title = title[: -len(suffix)].strip()
    return title


def _moz_lz4_json(path: Path) -> dict | None:
    try:
        import lz4.block
    except ImportError:
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if not data.startswith(b"mozLz40\x00"):
        return None
    try:
        raw = lz4.block.decompress(data[8:])
        loaded = json.loads(raw)
    except Exception:
        return None
    return loaded if isinstance(loaded, dict) else None


def _zen_url_from_session(window_title: str) -> tuple[str, str]:
    profiles = ZEN_SUPPORT / "Profiles"
    if not profiles.is_dir():
        return "", ""
    files: list[Path] = []
    for profile in profiles.iterdir():
        if not profile.is_dir():
            continue
        for rel in ("sessionstore-backups/recovery.jsonlz4", "sessionstore.jsonlz4"):
            candidate = profile / rel
            if candidate.is_file():
                files.append(candidate)
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    tabs: list[tuple[str, str]] = []
    for path in files[:3]:
        data = _moz_lz4_json(path)
        if not data:
            continue
        windows = data.get("windows") or []
        if not isinstance(windows, list):
            continue
        for window in windows:
            if not isinstance(window, dict):
                continue
            for tab in window.get("tabs") or []:
                if not isinstance(tab, dict):
                    continue
                entries = tab.get("entries") or []
                if not isinstance(entries, list) or not entries:
                    continue
                idx = int(tab.get("index") or len(entries)) - 1
                idx = max(0, min(idx, len(entries) - 1))
                entry = entries[idx] if isinstance(entries[idx], dict) else {}
                url = str(entry.get("url") or "").strip()
                title = str(entry.get("title") or "").strip()
                if url.startswith(("http://", "https://")):
                    tabs.append((url, title))
        if tabs:
            break
    if not tabs:
        return "", ""
    if window_title:
        lowered = window_title.lower()
        for url, title in tabs:
            if title.lower() == lowered or lowered in title.lower():
                return url, title or window_title
    return tabs[0][0], tabs[0][1] or window_title


def _clipboard_restore(old: bytes) -> None:
    subprocess.run(["pbcopy"], input=old, check=False)


def _pasteboard_html() -> str:
    script = (
        'use framework "AppKit"\n'
        "set pb to current application's NSPasteboard's generalPasteboard()\n"
        'set html to pb\'s stringForType:"public.html"\n'
        "if html is missing value then return \"\"\n"
        "return html as string\n"
    )
    try:
        proc = _osascript(script, timeout=5)
    except subprocess.TimeoutExpired:
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout or ""


def _copy_from_zen(select_all: bool) -> str:
    old = subprocess.run(["pbpaste"], check=False, capture_output=True).stdout
    if select_all:
        keys = """
    keystroke "a" using command down
    delay 0.18
    keystroke "c" using command down
    delay 0.22
    key code 53
"""
    else:
        keys = """
    keystroke "l" using command down
    delay 0.12
    keystroke "c" using command down
    delay 0.15
    key code 53
"""
    script = f"""
tell application "Zen" to activate
delay 0.25
tell application "System Events"
  tell process "zen"
{keys}
  end tell
end tell
"""
    proc = _osascript(script, timeout=8)
    html = _pasteboard_html() if proc.returncode == 0 else ""
    text = subprocess.run(
        ["pbpaste"], check=False, capture_output=True, text=True
    ).stdout
    _clipboard_restore(old)
    if proc.returncode != 0:
        return ""
    if select_all:
        return (html or text or "").strip()
    return (text or "").strip()


def _fetch_public_html(url: str) -> str:
    if not url.lower().startswith(("http://", "https://")):
        return ""
    req = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:142.0) "
                "Gecko/20100101 Firefox/142.0"
            ),
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    try:
        with urlopen(req, timeout=12) as resp:
            raw = resp.read(900_000)
        return raw.decode("utf-8", "replace")
    except Exception:
        return ""


def _read_safari() -> dict:
    proc = _osascript(
        'tell application "Safari"\n'
        "  if (count of documents) is 0 then error \"Safari has no open page\"\n"
        "  set pageURL to URL of front document\n"
        "  set pageTitle to name of front document\n"
        '  set pageHTML to ""\n'
        "  try\n"
        '    set pageHTML to do JavaScript "document.documentElement.outerHTML" in front document\n'
        "  end try\n"
        "end tell\n"
        "return pageURL & linefeed & pageTitle & linefeed & pageHTML\n"
    )
    if proc.returncode != 0:
        return _err((proc.stderr or proc.stdout or "Safari failed").strip())
    parts = (proc.stdout or "").split("\n", 2)
    url = parts[0].strip() if parts else ""
    title = parts[1].strip() if len(parts) > 1 else ""
    html = parts[2] if len(parts) > 2 else ""
    return _ok(app="Safari", url=url, title=title, html=html)


def _read_chrome() -> dict:
    proc = _osascript(
        'tell application "Google Chrome"\n'
        "  if (count of windows) is 0 then error \"Chrome has no window\"\n"
        "  set theTab to active tab of front window\n"
        "  set pageURL to URL of theTab\n"
        "  set pageTitle to title of theTab\n"
        '  set pageHTML to ""\n'
        "  try\n"
        '    set pageHTML to execute theTab javascript "document.documentElement.outerHTML"\n'
        "  end try\n"
        "end tell\n"
        "return pageURL & linefeed & pageTitle & linefeed & pageHTML\n"
    )
    if proc.returncode != 0:
        return _err((proc.stderr or proc.stdout or "Chrome failed").strip())
    parts = (proc.stdout or "").split("\n", 2)
    url = parts[0].strip() if parts else ""
    title = parts[1].strip() if len(parts) > 1 else ""
    html = parts[2] if len(parts) > 2 else ""
    return _ok(app="Google Chrome", url=url, title=title, html=html)


def _read_zen() -> dict:
    if not _zen_running():
        return _err("Zen is not running")
    title = _zen_window_title()
    url, session_title = _zen_url_from_session(title)
    if session_title and not title:
        title = session_title
    if not url:
        copied = _copy_from_zen(select_all=False)
        if copied.lower().startswith(("http://", "https://")):
            url = copied.split()[0]
    html = _fetch_public_html(url) if url else ""
    if len(compact_html(html)) < 80:
        copied = _copy_from_zen(select_all=True)
        if copied:
            html = copied
    return _ok(app="Zen", url=url, title=title, html=html)


def get_browser_page(max_chars: int = PAGE_TEXT_MAX) -> dict:
    """URL, title, HTML, and text of the front Zen, Safari, or Chrome page."""
    limit = max(1000, min(int(max_chars), 12_000))
    readers = [_read_zen, _read_safari, _read_chrome]
    last_err = "No browser page found. Open a tab in Zen, Safari, or Google Chrome."
    for reader in readers:
        page = reader()
        if not page.get("ok"):
            last_err = str(page.get("error") or last_err)
            continue
        url = str(page.get("url") or "").strip()
        html = compact_html(str(page.get("html") or ""), limit)
        if not url and len(html) < 80:
            continue
        if len(html) < 80 and url:
            html = compact_html(_fetch_public_html(url), limit)
        if len(html) < 80:
            last_err = "Could not read the page. Keep the tab visible and try again."
            continue
        page["html"] = html
        page["text"] = html_to_text(html, limit)
        page["url"] = url
        page["title"] = str(page.get("title") or "").strip()
        return page
    return _err(last_err)


def open_file(path: str, app: str | None = None) -> dict:
    if not path.strip():
        return _err("path is empty")
    cmd = ["open"]
    if app and app.strip():
        cmd.extend(["-a", app.strip()])
    cmd.append(path.strip())
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return _err((result.stderr or result.stdout or "open failed").strip())
    return _ok(path=path.strip(), app=app or "")
