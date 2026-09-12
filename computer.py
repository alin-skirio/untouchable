"""Generic Mac control: keys, text, mouse, apps, and front-window context.

Grok should call these primitives. Do not add site-specific helpers here.
"""

from __future__ import annotations

import ctypes
import json
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

from mac_keys import (
    CGPoint,
    _button_codes,
    _clamp_to_display,
    _core_foundation,
    _core_graphics,
    _cursor_location,
    _display_bounds,
    _post_key,
    _post_mouse,
    kCGEventFlagMaskAlternate,
    kCGEventFlagMaskCommand,
    kCGEventFlagMaskControl,
    kCGEventFlagMaskShift,
    kCGEventLeftMouseDragged,
    kCGEventMouseMoved,
    kCGEventRightMouseDragged,
    kCGHIDEventTap,
    kCGMouseButtonLeft,
    smooth_scroll,
)

MAX_TEXT_LEN = 2000
PAGE_TEXT_MAX = 16000
MAX_PAGE_IMAGES = 16
ZEN_SUPPORT = Path.home() / "Library/Application Support/zen"
IMAGE_SKIP_RE = re.compile(
    r"logo|sprite|icon|favicon|pixel|emoji|avatar|badge|button|tracker|"
    r"spinner|placeholder|1x1|advert|adservice|/static/images/|"
    r"wikimedia-button|poweredby|footer|share-facebook|share-twitter",
    re.I,
)
DOM_IMAGES_JS = (
    "JSON.stringify([].slice.call(document.images,0,40).map(function(i){"
    "return {src:i.currentSrc||i.src||'',alt:i.alt||'',"
    "w:i.naturalWidth||i.width||0,h:i.naturalHeight||i.height||0}"
    "}))"
)
DOM_HTML_JS = (
    "(function(){var r=document.querySelector("
    "'main,article,[role=main],#mw-content-text,#content,.post-content,.entry-content'"
    ");if(!r)r=document.body;var c=r.cloneNode(true);"
    "var n=c.querySelectorAll('script,style,noscript,iframe,svg,canvas,template,form');"
    "for(var i=n.length-1;i>=0;i--){n[i].parentNode.removeChild(n[i]);}"
    "var h=c.innerHTML||'';return h.length>50000?h.slice(0,50000):h})()"
)
MODIFIERS = {
    "command": kCGEventFlagMaskCommand,
    "cmd": kCGEventFlagMaskCommand,
    "shift": kCGEventFlagMaskShift,
    "option": kCGEventFlagMaskAlternate,
    "alt": kCGEventFlagMaskAlternate,
    "control": kCGEventFlagMaskControl,
    "ctrl": kCGEventFlagMaskControl,
}

# macOS virtual key codes (ANSI US layout).
KEY_CODES = {
    "a": 0x00,
    "s": 0x01,
    "d": 0x02,
    "f": 0x03,
    "h": 0x04,
    "g": 0x05,
    "z": 0x06,
    "x": 0x07,
    "c": 0x08,
    "v": 0x09,
    "b": 0x0B,
    "q": 0x0C,
    "w": 0x0D,
    "e": 0x0E,
    "r": 0x0F,
    "y": 0x10,
    "t": 0x11,
    "1": 0x12,
    "2": 0x13,
    "3": 0x14,
    "4": 0x15,
    "6": 0x16,
    "5": 0x17,
    "equal": 0x18,
    "=": 0x18,
    "9": 0x19,
    "7": 0x1A,
    "minus": 0x1B,
    "-": 0x1B,
    "8": 0x1C,
    "0": 0x1D,
    "]": 0x1E,
    "o": 0x1F,
    "u": 0x20,
    "[": 0x21,
    "i": 0x22,
    "p": 0x23,
    "return": 0x24,
    "enter": 0x24,
    "l": 0x25,
    "j": 0x26,
    "'": 0x27,
    "k": 0x28,
    ";": 0x29,
    "\\": 0x2A,
    ",": 0x2B,
    "/": 0x2C,
    "n": 0x2D,
    "m": 0x2E,
    ".": 0x2F,
    "tab": 0x30,
    "space": 0x31,
    "`": 0x32,
    "delete": 0x33,
    "backspace": 0x33,
    "escape": 0x35,
    "esc": 0x35,
    "command": 0x37,
    "cmd": 0x37,
    "shift": 0x38,
    "option": 0x3A,
    "alt": 0x3A,
    "control": 0x3B,
    "ctrl": 0x3B,
    "f5": 0x60,
    "f6": 0x61,
    "f7": 0x62,
    "f3": 0x63,
    "f8": 0x64,
    "f9": 0x65,
    "f11": 0x67,
    "f10": 0x6D,
    "f12": 0x6F,
    "home": 0x73,
    "page_up": 0x74,
    "forward_delete": 0x75,
    "f4": 0x76,
    "end": 0x77,
    "f2": 0x78,
    "page_down": 0x79,
    "f1": 0x7A,
    "left": 0x7B,
    "right": 0x7C,
    "down": 0x7D,
    "up": 0x7E,
}


def _ok(**extra) -> dict:
    return {"ok": True, **extra}


def _err(message: str) -> dict:
    return {"ok": False, "error": message}


def _flags_and_keys(keys: list[str]) -> tuple[int, list[str]] | dict:
    flags = 0
    others: list[str] = []
    for raw in keys:
        name = str(raw).strip().lower()
        if not name:
            continue
        if name in MODIFIERS:
            flags |= MODIFIERS[name]
            continue
        if name not in KEY_CODES:
            return _err(f"Unknown key: {raw}")
        others.append(name)
    if not others and not flags:
        return _err("No keys provided")
    return flags, others


def _press_modifiers(cg, flags: int, down: bool) -> None:
    release_flags = flags if down else 0
    order = (
        ("command", kCGEventFlagMaskCommand),
        ("shift", kCGEventFlagMaskShift),
        ("option", kCGEventFlagMaskAlternate),
        ("control", kCGEventFlagMaskControl),
    )
    sequence = order if down else tuple(reversed(order))
    for name, mask in sequence:
        if flags & mask:
            _post_key(cg, KEY_CODES[name], down, release_flags if down else 0)


def key_combo(keys: list[str]) -> dict:
    """Press a shortcut, e.g. ['command', 't'] or ['escape']."""
    parsed = _flags_and_keys(keys)
    if isinstance(parsed, dict):
        return parsed
    flags, others = parsed
    try:
        cg = _core_graphics()
        _press_modifiers(cg, flags, True)
        time.sleep(0.03)
        for name in others:
            _post_key(cg, KEY_CODES[name], True, flags)
        time.sleep(0.03)
        for name in reversed(others):
            _post_key(cg, KEY_CODES[name], False, flags)
        _press_modifiers(cg, flags, False)
        return _ok(keys=[str(k).lower() for k in keys])
    except Exception as exc:
        return _err(str(exc))


def type_text(text: str) -> dict:
    """Type unicode text into the focused field."""
    if not isinstance(text, str):
        return _err("text must be a string")
    if len(text) > MAX_TEXT_LEN:
        return _err(f"text longer than {MAX_TEXT_LEN} characters")
    try:
        cg = _core_graphics()
        cg.CGEventKeyboardSetUnicodeString.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        for char in text:
            if char == "\n":
                key_combo(["return"])
                continue
            if char == "\t":
                key_combo(["tab"])
                continue
            code = ord(char)
            if code > 0xFFFF:
                return _err("emoji beyond BMP is not supported yet")
            buf = (ctypes.c_uint16 * 1)(code)
            for down in (True, False):
                event = cg.CGEventCreateKeyboardEvent(None, 0, down)
                if not event:
                    return _err("Could not create keyboard event")
                cg.CGEventKeyboardSetUnicodeString(event, 1, buf)
                cg.CGEventPost(kCGHIDEventTap, event)
        return _ok(length=len(text))
    except Exception as exc:
        return _err(str(exc))


def _point_at(x: float, y: float) -> CGPoint:
    cg = _core_graphics()
    bounds = _display_bounds(cg)
    point = CGPoint(float(x), float(y))
    return _clamp_to_display(point, bounds)


def mouse_move(x: float, y: float) -> dict:
    """Move the cursor to a top-left screen point."""
    try:
        cg = _core_graphics()
        cf = _core_foundation()
        point = _point_at(x, y)
        _post_mouse(cg, cf, kCGEventMouseMoved, point, kCGMouseButtonLeft)
        return _ok(x=point.x, y=point.y)
    except Exception as exc:
        return _err(str(exc))


def click(button: str = "left", count: int = 1, x: float | None = None, y: float | None = None) -> dict:
    """Click at the current cursor, or at x,y if given."""
    if button not in ("left", "right"):
        return _err("button must be left or right")
    if count < 1 or count > 3:
        return _err("count must be 1, 2, or 3")
    try:
        if x is not None and y is not None:
            moved = mouse_move(x, y)
            if not moved.get("ok"):
                return moved
        cg = _core_graphics()
        cf = _core_foundation()
        point = _cursor_location(cg, cf)
        code, down_type, up_type = _button_codes(button)
        for _ in range(count):
            _post_mouse(cg, cf, down_type, point, code, click=True)
            _post_mouse(cg, cf, up_type, point, code, click=True)
            time.sleep(0.04)
        return _ok(button=button, count=count, x=point.x, y=point.y)
    except Exception as exc:
        return _err(str(exc))


def drag(x: float, y: float, button: str = "left") -> dict:
    """Drag from the current cursor to x,y."""
    if button not in ("left", "right"):
        return _err("button must be left or right")
    try:
        cg = _core_graphics()
        cf = _core_foundation()
        start = _cursor_location(cg, cf)
        end = _point_at(x, y)
        code, down_type, up_type = _button_codes(button)
        _post_mouse(cg, cf, down_type, start, code, click=True)
        drag_type = kCGEventLeftMouseDragged if button == "left" else kCGEventRightMouseDragged
        _post_mouse(cg, cf, drag_type, end, code)
        _post_mouse(cg, cf, up_type, end, code, click=True)
        return _ok(from_x=start.x, from_y=start.y, x=end.x, y=end.y)
    except Exception as exc:
        return _err(str(exc))


def scroll(lines: int) -> dict:
    """Post a scroll-wheel event. Positive lines scroll up in Quartz units."""
    try:
        amount = int(lines)
        smooth_scroll(amount)
        return _ok(lines=amount)
    except Exception as exc:
        return _err(str(exc))


def open_app(name: str) -> dict:
    """Open or focus an application by name (Spotlight-style, not a URL)."""
    if not isinstance(name, str) or not name.strip():
        return _err("app name is empty")
    cleaned = name.strip()
    if any(ch in cleaned for ch in "\n\r;|&"):
        return _err("app name contains invalid characters")
    result = subprocess.run(
        ["open", "-a", cleaned],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "open failed").strip()
        return _err(err)
    subprocess.run(
        ["osascript", "-e", f'tell application "{cleaned}" to activate'],
        check=False,
        capture_output=True,
        text=True,
    )
    ctx = {"frontmost": ""}
    for _ in range(20):
        time.sleep(0.15)
        ctx = get_context()
        if (ctx.get("frontmost") or "").lower() == cleaned.lower():
            break
    return _ok(app=cleaned, context=ctx)


def make_new_note(body: str) -> dict:
    """Create a new Apple Note with this body. Used so demos do not depend on ⌘N focus."""
    if not isinstance(body, str) or not body.strip():
        return _err("note body is empty")
    text = body.strip()
    if len(text) > MAX_TEXT_LEN:
        return _err(f"text longer than {MAX_TEXT_LEN} characters")
    proc = subprocess.run(
        [
            "osascript",
            "-e",
            "on run argv",
            "-e",
            "set noteBody to item 1 of argv",
            "-e",
            'tell application "Notes"',
            "-e",
            "activate",
            "-e",
            "make new note with properties {body:noteBody}",
            "-e",
            "end tell",
            "-e",
            "end run",
            text,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "Notes AppleScript failed").strip()
        return _err(err)
    time.sleep(0.4)
    return _ok(body=text, context=get_context())


def get_context() -> dict:
    """Frontmost app, window title, cursor, and main display size."""
    script = (
        'tell application "System Events"\n'
        "  tell (first process whose frontmost is true)\n"
        "    set appName to name\n"
        '    set winName to ""\n'
        "    try\n"
        "      set winName to name of window 1\n"
        "    end try\n"
        "  end tell\n"
        "end tell\n"
        'return appName & linefeed & winName\n'
    )
    proc = subprocess.run(
        ["osascript", "-e", script],
        check=False,
        capture_output=True,
        text=True,
    )
    frontmost = ""
    window = ""
    if proc.returncode == 0:
        parts = (proc.stdout or "").rstrip("\n").split("\n", 1)
        frontmost = parts[0] if parts else ""
        window = parts[1] if len(parts) > 1 else ""
    cg = _core_graphics()
    cf = _core_foundation()
    point = _cursor_location(cg, cf)
    bounds = _display_bounds(cg)
    return {
        "ok": True,
        "frontmost": frontmost,
        "window": window,
        "cursor": {"x": point.x, "y": point.y},
        "display": {
            "x": bounds.origin.x,
            "y": bounds.origin.y,
            "width": bounds.size.width,
            "height": bounds.size.height,
        },
        "osascript_error": "" if proc.returncode == 0 else (proc.stderr or "").strip(),
    }


def dump_context() -> str:
    return json.dumps(get_context(), indent=2)


def _parse_url_title(raw: str) -> tuple[str, str]:
    parts = (raw or "").split("\n", 1)
    url = parts[0].strip() if parts else ""
    title = parts[1].strip() if len(parts) > 1 else ""
    return url, title


def _read_safari() -> dict:
    script = (
        'tell application "Safari"\n'
        "  if (count of documents) is 0 then error \"Safari has no open page\"\n"
        "  set pageURL to URL of front document\n"
        "  set pageTitle to name of front document\n"
        "end tell\n"
        "return pageURL & linefeed & pageTitle\n"
    )
    proc = subprocess.run(
        ["osascript", "-e", script],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return _err((proc.stderr or proc.stdout or "Safari AppleScript failed").strip())
    url, title = _parse_url_title(proc.stdout or "")
    return _ok(app="Safari", url=url, title=title, text="")


def _read_chrome() -> dict:
    script = (
        'tell application "Google Chrome"\n'
        "  if (count of windows) is 0 then error \"Chrome has no window\"\n"
        "  set theTab to active tab of front window\n"
        "  set pageURL to URL of theTab\n"
        "  set pageTitle to title of theTab\n"
        "end tell\n"
        "return pageURL & linefeed & pageTitle\n"
    )
    proc = subprocess.run(
        ["osascript", "-e", script],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return _err((proc.stderr or proc.stdout or "Chrome AppleScript failed").strip())
    url, title = _parse_url_title(proc.stdout or "")
    return _ok(app="Google Chrome", url=url, title=title, text="")


def _zen_process_running() -> bool:
    proc = subprocess.run(
        [
            "osascript",
            "-e",
            'tell application "System Events" to get name of every process',
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    names = [part.strip().lower() for part in (proc.stdout or "").split(",")]
    return proc.returncode == 0 and any(name in {"zen", "zen browser"} for name in names)


def _zen_window_title() -> str:
    for process in ("zen", "Zen", "Zen Browser"):
        proc = subprocess.run(
            [
                "osascript",
                "-e",
                f'tell application "System Events" to tell process "{process}" to get name of front window',
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        title = (proc.stdout or "").strip()
        if proc.returncode == 0 and title:
            for suffix in (" — Zen Browser", " - Zen Browser", " — Zen", " - Zen"):
                if title.endswith(suffix):
                    title = title[: -len(suffix)].strip()
            return title
    return ""


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
    except (OSError, ValueError, json.JSONDecodeError, Exception):
        return None
    return loaded if isinstance(loaded, dict) else None


def _zen_recovery_files() -> list[Path]:
    found: list[Path] = []
    profiles = ZEN_SUPPORT / "Profiles"
    if not profiles.is_dir():
        return found
    for profile in profiles.iterdir():
        if not profile.is_dir():
            continue
        for rel in (
            "sessionstore-backups/recovery.jsonlz4",
            "sessionstore.jsonlz4",
        ):
            candidate = profile / rel
            if candidate.is_file():
                found.append(candidate)
    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return found


def _tab_entry(tab: dict) -> dict:
    entries = tab.get("entries") or []
    if not isinstance(entries, list) or not entries:
        return {}
    idx = int(tab.get("index") or len(entries)) - 1
    if idx < 0 or idx >= len(entries):
        idx = len(entries) - 1
    entry = entries[idx]
    return entry if isinstance(entry, dict) else {}


def _zen_tabs_from_session(data: dict) -> list[tuple[str, str]]:
    tabs_out: list[tuple[str, str]] = []
    windows = data.get("windows") or []
    if not isinstance(windows, list):
        return tabs_out
    selected_window = int(data.get("selectedWindow") or 1) - 1
    ordered = list(enumerate(windows))
    if 0 <= selected_window < len(windows):
        ordered = [(selected_window, windows[selected_window])] + [
            item for item in ordered if item[0] != selected_window
        ]
    for _i, window in ordered:
        if not isinstance(window, dict):
            continue
        tabs = window.get("tabs") or []
        if not isinstance(tabs, list) or not tabs:
            continue
        selected = int(window.get("selected") or 1) - 1
        indices = list(range(len(tabs)))
        if 0 <= selected < len(tabs):
            indices = [selected] + [n for n in indices if n != selected]
        for n in indices:
            entry = _tab_entry(tabs[n] if isinstance(tabs[n], dict) else {})
            url = str(entry.get("url") or "").strip()
            title = str(entry.get("title") or "").strip()
            if url.startswith(("http://", "https://")):
                tabs_out.append((url, title))
    return tabs_out


def _read_zen() -> dict:
    if not _zen_process_running():
        return _err("Zen is not running")
    files = _zen_recovery_files()
    if not files:
        return _err("Could not find a Zen session file")
    data = None
    for path in files:
        data = _moz_lz4_json(path)
        if data:
            break
    if data is None:
        return _err("Install lz4 (pip install lz4) to read the current Zen tab.")
    tabs = _zen_tabs_from_session(data)
    if not tabs:
        return _err("Zen has no open http tab")
    window_title = _zen_window_title()
    url, title = tabs[0]
    if window_title:
        lowered = window_title.lower()
        for candidate_url, candidate_title in tabs:
            if candidate_title.lower() == lowered or lowered in candidate_title.lower():
                url, title = candidate_url, candidate_title
                break
        if not title:
            title = window_title
    return _ok(app="Zen", url=url, title=title or window_title, text="")


def _fetch_public_html(url: str) -> str:
    if not url.lower().startswith(("http://", "https://")):
        return ""
    try:
        from urllib.request import Request, urlopen

        req = Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15"
                ),
                "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
            },
        )
        with urlopen(req, timeout=3) as resp:
            raw = resp.read(700_000)
        return raw.decode("utf-8", "replace")
    except Exception:
        return ""


def compact_html(html: str, max_chars: int = PAGE_TEXT_MAX) -> str:
    """Drop scripts, chrome, and noisy attributes so the rewrite model sees structure."""
    if not html:
        return ""
    markup = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", html)
    markup = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", markup)
    markup = re.sub(r"(?is)<noscript[^>]*>.*?</noscript>", " ", markup)
    markup = re.sub(r"(?is)<!--.*?-->", " ", markup)
    markup = re.sub(r"(?is)</?(?:svg|path|iframe|canvas|template)[^>]*>", " ", markup)
    markup = re.sub(
        r"\s(?:class|id|style|onclick|onerror|loading|decoding|srcset|"
        r"sizes|data-[\w-]+|aria-[\w-]+)=(\"[^\"]*\"|'[^']*')",
        "",
        markup,
        flags=re.I,
    )
    markup = re.sub(r"\s+", " ", markup)
    return markup.strip()[: max(1000, min(int(max_chars), 20_000))]


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


def _largest_srcset(srcset: str) -> str:
    best = ""
    best_w = -1
    for part in srcset.split(","):
        bits = part.strip().split()
        if not bits:
            continue
        width = 0
        if len(bits) > 1 and bits[1].endswith("w"):
            try:
                width = int(bits[1][:-1])
            except ValueError:
                width = 0
        if width >= best_w:
            best_w = width
            best = bits[0]
    return best


def _upgrade_wiki_thumb(url: str) -> str:
    def bump(match: re.Match[str]) -> str:
        try:
            px = int(match.group(1))
        except ValueError:
            return match.group(0)
        if px < 640:
            return "/800px-"
        return match.group(0)

    return re.sub(r"/(\d{2,3})px-", bump, url, count=1)


def _normalize_image_url(src: str, base_url: str) -> str:
    src = (src or "").strip()
    if not src or src.startswith("data:"):
        return ""
    if src.startswith("//"):
        src = "https:" + src
    abs_url = urljoin(base_url, src)
    parsed = urlparse(abs_url)
    if parsed.scheme not in {"http", "https"}:
        return ""
    if IMAGE_SKIP_RE.search(abs_url):
        return ""
    if re.search(r"/(\d{1,2})px-", abs_url):
        return ""
    if "upload.wikimedia.org" in parsed.netloc:
        abs_url = _upgrade_wiki_thumb(abs_url)
    return abs_url


def extract_page_images(html: str, base_url: str, limit: int = MAX_PAGE_IMAGES) -> list[dict]:
    """Content photos from HTML: src, alt, optional caption."""
    if not html:
        return []
    found: list[dict] = []
    seen: set[str] = set()

    def add(src: str, alt: str = "", cap: str = "", width: int = 0) -> None:
        url = _normalize_image_url(src, base_url)
        if not url or url in seen:
            return
        if width and width < 80:
            return
        alt_text = re.sub(r"\s+", " ", alt or "").strip()
        if IMAGE_SKIP_RE.search(alt_text):
            return
        seen.add(url)
        item = {"src": url, "alt": alt_text}
        if cap:
            item["cap"] = re.sub(r"\s+", " ", cap).strip()
        found.append(item)

    for meta in re.finditer(
        r'<meta\b[^>]+property=["\']og:image["\'][^>]*>',
        html,
        re.I,
    ):
        tag = meta.group(0)
        match = re.search(r'''content=["']([^"']+)["']''', tag, re.I)
        if match:
            add(match.group(1), alt="Hero")

    for tag in re.finditer(r"<img\b[^>]*>", html, re.I):
        chunk = tag.group(0)
        srcset = re.search(r'''srcset=["']([^"']+)["']''', chunk, re.I)
        src = ""
        if srcset:
            src = _largest_srcset(srcset.group(1))
        if not src:
            src_match = re.search(
                r'''(?:src|data-src|data-lazy-src|data-original)=["']([^"']+)["']''',
                chunk,
                re.I,
            )
            src = src_match.group(1) if src_match else ""
        alt_match = re.search(r'''alt=["']([^"']*)["']''', chunk, re.I)
        width = 0
        width_match = re.search(r'''width=["'](\d+)["']''', chunk, re.I)
        if width_match:
            width = int(width_match.group(1))
        add(src, alt_match.group(1) if alt_match else "", width=width)
        if len(found) >= limit:
            break
    return found[:limit]


def _images_from_dom_json(raw: str, base_url: str) -> list[dict]:
    try:
        payload = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    html_bits = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        src = str(item.get("src") or "")
        alt = str(item.get("alt") or "")
        width = int(item.get("w") or 0)
        height = int(item.get("h") or 0)
        if (width and width < 80) or (height and height < 80):
            continue
        html_bits.append(f'<img src="{src}" alt="{alt}" width="{width}">')
    return extract_page_images("".join(html_bits), base_url)


def _run_browser_js(kind: str, js: str) -> str:
    if kind == "safari":
        script = (
            'tell application "Safari" to do JavaScript '
            f'"{js}" in front document'
        )
    elif kind == "chrome":
        script = (
            'tell application "Google Chrome"\n'
            f'  execute active tab of front window javascript "{js}"\n'
            "end tell"
        )
    else:
        return ""
    proc = subprocess.run(
        ["osascript", "-e", script],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return ""
    return (proc.stdout or "").strip()


def _dom_images_from_browser(kind: str, base_url: str) -> list[dict]:
    return _images_from_dom_json(_run_browser_js(kind, DOM_IMAGES_JS), base_url)


def _dom_html_from_browser(kind: str) -> str:
    return _run_browser_js(kind, DOM_HTML_JS)


def _merge_images(*groups: list[dict]) -> list[dict]:
    seen: set[str] = set()
    merged: list[dict] = []
    for group in groups:
        for item in group:
            src = str(item.get("src") or "")
            if not src or src in seen:
                continue
            seen.add(src)
            merged.append(item)
            if len(merged) >= MAX_PAGE_IMAGES:
                return merged
    return merged


def get_browser_page(max_chars: int = PAGE_TEXT_MAX) -> dict:
    """URL, title, HTML, and text of the front Zen, Safari, or Chrome page.

    Prefers the live DOM HTML from Safari/Chrome. Zen has no JS bridge, so
    the public HTML for that tab URL is fetched instead of copy-paste.
    """
    limit = max(1000, min(int(max_chars), 12_000))
    front = (get_context().get("frontmost") or "").lower()
    readers = [(_read_zen, "zen"), (_read_safari, "safari"), (_read_chrome, "chrome")]
    if "chrome" in front:
        readers = [(_read_chrome, "chrome"), (_read_zen, "zen"), (_read_safari, "safari")]
    elif "safari" in front:
        readers = [(_read_safari, "safari"), (_read_zen, "zen"), (_read_chrome, "chrome")]

    last_err = "No browser page found. Open a tab in Zen, Safari, or Google Chrome."
    for reader, kind in readers:
        page = reader()
        if not page.get("ok"):
            last_err = str(page.get("error") or last_err)
            continue
        url = str(page.get("url") or "").strip()
        if not url:
            continue
        html = ""
        source = ""
        if kind in {"safari", "chrome"}:
            html = _dom_html_from_browser(kind)
            if html:
                source = "dom"
        fetched_html = _fetch_public_html(url)
        if not html and fetched_html:
            html = fetched_html
            source = "fetched"
        html = compact_html(html, limit)
        text = html_to_text(html, limit)
        images = []
        if kind in {"safari", "chrome"}:
            images = _dom_images_from_browser(kind, url)
        images = _merge_images(images, extract_page_images(html, url))
        if fetched_html:
            images = _merge_images(images, extract_page_images(fetched_html, url))
        page["html"] = html
        page["text"] = text
        page["text_source"] = source
        page["images"] = images
        page["title"] = str(page.get("title") or "").strip()
        page["js_hint"] = (
            ""
            if len(html) >= 80
            else (
                "Could not load HTML for this tab. If it is behind a login, "
                "open it in Safari or Chrome with JavaScript from Apple Events enabled."
                if kind == "zen"
                else "Enable Develop > Allow JavaScript from Apple Events in Safari, "
                "or View > Developer > Allow JavaScript from Apple Events in Chrome."
            )
        )
        return page
    return _err(last_err)


def open_html_file(path: str, app: str | None = None) -> dict:
    """Open a local HTML file in the default browser, or a named app."""
    if not isinstance(path, str) or not path.strip():
        return _err("path is empty")
    target = path.strip()
    cmd = ["open"]
    if app and app.strip():
        cmd.extend(["-a", app.strip()])
    cmd.append(target)
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "open failed").strip()
        return _err(err)
    return _ok(path=target, app=app or "")
