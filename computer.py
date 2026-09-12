"""Generic Mac control: keys, text, mouse, apps, and front-window context.

Grok should call these primitives. Do not add site-specific helpers here.
"""

from __future__ import annotations

import ctypes
import json
import subprocess
import time

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
