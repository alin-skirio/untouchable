"""Global macOS keystrokes. These work even when the tracker window is not focused."""

from __future__ import annotations

import ctypes
import subprocess
import threading

kVK_Tab = 0x30
kVK_Command = 0x37
kVK_PageUp = 0x74
kCGHIDEventTap = 0
kCGEventFlagMaskCommand = 0x100000


def _core_graphics():
    cg = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
    cg.CGEventCreateKeyboardEvent.restype = ctypes.c_void_p
    cg.CGEventCreateKeyboardEvent.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint16,
        ctypes.c_bool,
    ]
    # Add scroll wheel event signature
    cg.CGEventCreateScrollWheelEvent.restype = ctypes.c_void_p
    cg.CGEventCreateScrollWheelEvent.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int32,
    ]
    cg.CGEventSetFlags.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    cg.CGEventPost.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
    return cg


def _post_key(cg, key_code: int, key_down: bool, flags: int = 0) -> None:
    event = cg.CGEventCreateKeyboardEvent(None, key_code, key_down)
    if not event:
        raise RuntimeError("Could not create keyboard event")
    if flags:
        cg.CGEventSetFlags(event, flags)
    cg.CGEventPost(kCGHIDEventTap, event)


def smooth_scroll(lines: int) -> None:
    """Generates a native scroll wheel event."""
    try:
        cg = _core_graphics()
        # 0 = lines unit, 1 = one scroll wheel, lines = amount to scroll
        event = cg.CGEventCreateScrollWheelEvent(None, 0, 1, lines)
        if event:
            cg.CGEventPost(kCGHIDEventTap, event)
    except Exception:
        pass


def _cmd_tab_quartz() -> None:
    cg = _core_graphics()
    _post_key(cg, kVK_Command, True, kCGEventFlagMaskCommand)
    _post_key(cg, kVK_Tab, True, kCGEventFlagMaskCommand)
    _post_key(cg, kVK_Tab, False, kCGEventFlagMaskCommand)
    _post_key(cg, kVK_Command, False, 0)


def _cmd_tab_osascript() -> None:
    subprocess.run(
        [
            "osascript",
            "-e",
            'tell application "System Events" to key code 48 using command down',
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def set_cmd_state(down: bool) -> None:
    flags = kCGEventFlagMaskCommand if down else 0
    try:
        cg = _core_graphics()
        _post_key(cg, kVK_Command, down, flags)
    except Exception:
        pass


def tap_tab() -> None:
    try:
        cg = _core_graphics()
        _post_key(cg, kVK_Tab, True, kCGEventFlagMaskCommand)
        _post_key(cg, kVK_Tab, False, kCGEventFlagMaskCommand)
    except Exception:
        pass


def async_cmd(down: bool) -> None:
    threading.Thread(target=set_cmd_state, args=(down,), daemon=True).start()


def async_tap_tab() -> None:
    threading.Thread(target=tap_tab, daemon=True).start()


def page_up() -> None:
    try:
        cg = _core_graphics()
        _post_key(cg, kVK_PageUp, True)
        _post_key(cg, kVK_PageUp, False)
    except Exception:
        pass


def async_page_up() -> None:
    threading.Thread(target=page_up, daemon=True).start()
