"""Global macOS keystrokes and mouse events via Quartz/CoreGraphics."""

from __future__ import annotations

import ctypes
import subprocess
import sys
import threading
from typing import Optional

kVK_Tab = 0x30
kVK_ANSI_T = 0x11
kVK_Command = 0x37
kVK_PageUp = 0x74
kCGHIDEventTap = 0
kCGSessionEventTap = 1
kCGHeadInsertEventTap = 0
kCGEventTapOptionDefault = 0
kCGEventFlagMaskShift = 0x20000
kCGEventFlagMaskControl = 0x40000
kCGEventFlagMaskAlternate = 0x80000
kCGEventFlagMaskCommand = 0x100000
kCGEventKeyDown = 10
kCGEventTapDisabledByTimeout = 0xFFFFFFFE
kCGEventTapDisabledByUserInput = 0xFFFFFFFF
kCGKeyboardEventKeycode = 9
kCGEventLeftMouseDown = 1
kCGEventLeftMouseUp = 2
kCGEventRightMouseDown = 3
kCGEventRightMouseUp = 4
kCGEventMouseMoved = 5
kCGEventLeftMouseDragged = 6
kCGEventRightMouseDragged = 7
kCGMouseButtonLeft = 0
kCGMouseButtonRight = 1
kCGMouseEventClickState = 1

_macos_warned = False
_held_button: Optional[str] = None
_cmd_t_monitor: Optional["CommandTMonitor"] = None


class CGPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


class CGSize(ctypes.Structure):
    _fields_ = [("width", ctypes.c_double), ("height", ctypes.c_double)]


class CGRect(ctypes.Structure):
    _fields_ = [("origin", CGPoint), ("size", CGSize)]


CGEventTapCallBack = ctypes.CFUNCTYPE(
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.c_void_p,
    ctypes.c_void_p,
)


def _is_macos() -> bool:
    global _macos_warned
    if sys.platform == "darwin":
        return True
    if not _macos_warned:
        _macos_warned = True
        print("Mouse and ⌘T reference hotkey require macOS (Quartz/CoreGraphics).")
    return False


def _core_graphics():
    cg = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
    cg.CGEventCreateKeyboardEvent.restype = ctypes.c_void_p
    cg.CGEventCreateKeyboardEvent.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint16,
        ctypes.c_bool,
    ]
    cg.CGEventCreateScrollWheelEvent.restype = ctypes.c_void_p
    cg.CGEventCreateScrollWheelEvent.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int32,
    ]
    cg.CGEventSetFlags.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    cg.CGEventPost.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
    cg.CGEventCreate.restype = ctypes.c_void_p
    cg.CGEventCreate.argtypes = [ctypes.c_void_p]
    cg.CGEventGetLocation.restype = CGPoint
    cg.CGEventGetLocation.argtypes = [ctypes.c_void_p]
    cg.CGEventCreateMouseEvent.restype = ctypes.c_void_p
    cg.CGEventCreateMouseEvent.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        CGPoint,
        ctypes.c_uint32,
    ]
    cg.CGEventSetIntegerValueField.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int64,
    ]
    cg.CGEventGetIntegerValueField.restype = ctypes.c_int64
    cg.CGEventGetIntegerValueField.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    cg.CGEventGetFlags.restype = ctypes.c_uint64
    cg.CGEventGetFlags.argtypes = [ctypes.c_void_p]
    cg.CGMainDisplayID.restype = ctypes.c_uint32
    cg.CGDisplayBounds.restype = CGRect
    cg.CGDisplayBounds.argtypes = [ctypes.c_uint32]
    cg.CGEventTapCreate.restype = ctypes.c_void_p
    cg.CGEventTapCreate.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint64,
        CGEventTapCallBack,
        ctypes.c_void_p,
    ]
    cg.CGEventTapEnable.argtypes = [ctypes.c_void_p, ctypes.c_bool]
    return cg


def _core_foundation():
    cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    cf.CFRelease.argtypes = [ctypes.c_void_p]
    cf.CFMachPortCreateRunLoopSource.restype = ctypes.c_void_p
    cf.CFMachPortCreateRunLoopSource.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_long,
    ]
    cf.CFRunLoopGetCurrent.restype = ctypes.c_void_p
    cf.CFRunLoopAddSource.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    cf.CFRunLoopRun.argtypes = []
    cf.CFRunLoopStop.argtypes = [ctypes.c_void_p]
    return cf


def _release(cf, ref) -> None:
    if ref:
        cf.CFRelease(ref)


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


def _display_bounds(cg) -> CGRect:
    return cg.CGDisplayBounds(cg.CGMainDisplayID())


def _cursor_location(cg, cf) -> CGPoint:
    event = cg.CGEventCreate(None)
    if not event:
        return CGPoint(0.0, 0.0)
    point = cg.CGEventGetLocation(event)
    _release(cf, event)
    return point


def _clamp_to_display(point: CGPoint, bounds: CGRect) -> CGPoint:
    x0 = bounds.origin.x
    y0 = bounds.origin.y
    x1 = x0 + max(bounds.size.width - 1.0, 0.0)
    y1 = y0 + max(bounds.size.height - 1.0, 0.0)
    point.x = min(max(point.x, x0), x1)
    point.y = min(max(point.y, y0), y1)
    return point


def _post_mouse(cg, cf, event_type: int, point: CGPoint, button: int, click: bool = False) -> None:
    event = cg.CGEventCreateMouseEvent(None, event_type, point, button)
    if not event:
        return
    if click:
        cg.CGEventSetIntegerValueField(event, kCGMouseEventClickState, 1)
    cg.CGEventPost(kCGHIDEventTap, event)
    _release(cf, event)


def _button_codes(button: str) -> tuple[int, int, int]:
    if button == "right":
        return kCGMouseButtonRight, kCGEventRightMouseDown, kCGEventRightMouseUp
    return kCGMouseButtonLeft, kCGEventLeftMouseDown, kCGEventLeftMouseUp


def move_mouse(dx: float, dy: float, dragging: Optional[str] = None) -> None:
    """Move the cursor by a pixel delta. Sequential — do not call from extra threads."""
    if not _is_macos() or (dx == 0 and dy == 0):
        return
    try:
        cg = _core_graphics()
        cf = _core_foundation()
        point = _cursor_location(cg, cf)
        point.x += dx
        point.y += dy
        point = _clamp_to_display(point, _display_bounds(cg))
        if dragging == "right":
            event_type, button = kCGEventRightMouseDragged, kCGMouseButtonRight
        elif dragging == "left":
            event_type, button = kCGEventLeftMouseDragged, kCGMouseButtonLeft
        else:
            event_type, button = kCGEventMouseMoved, kCGMouseButtonLeft
        _post_mouse(cg, cf, event_type, point, button)
    except Exception:
        pass


def page_up() -> None:
    try:
        cg = _core_graphics()
        _post_key(cg, kVK_PageUp, True)
        _post_key(cg, kVK_PageUp, False)
    except Exception:
        pass


def mouse_down(button: str) -> None:
    global _held_button
    if not _is_macos():
        return
    try:
        cg = _core_graphics()
        cf = _core_foundation()
        point = _cursor_location(cg, cf)
        code, down_type, _up_type = _button_codes(button)
        _post_mouse(cg, cf, down_type, point, code, click=True)
        _held_button = button
    except Exception:
        pass


def mouse_up(button: str) -> None:
    global _held_button
    if not _is_macos():
        return
    try:
        cg = _core_graphics()
        cf = _core_foundation()
        point = _cursor_location(cg, cf)
        code, _down_type, up_type = _button_codes(button)
        _post_mouse(cg, cf, up_type, point, code, click=True)
        if _held_button == button:
            _held_button = None
    except Exception:
        pass


def release_mouse() -> None:
    global _held_button
    if _held_button:
        mouse_up(_held_button)
        _held_button = None


class CommandTMonitor:
    """Global Quartz listen tap for ⌘T. Consumes the event while running."""

    def __init__(self):
        self._flag = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._runloop = None
        self._tap = None
        self._source = None
        self._callback = None
        self._cg = None

    def start(self) -> None:
        if not _is_macos() or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="cmd-t-tap", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._runloop is not None:
            try:
                cf = _core_foundation()
                if self._tap and self._cg:
                    self._cg.CGEventTapEnable(self._tap, False)
                cf.CFRunLoopStop(self._runloop)
            except Exception:
                pass
        self._runloop = None

    def consume(self) -> bool:
        if self._flag.is_set():
            self._flag.clear()
            return True
        return False

    def _on_event(self, _proxy, event_type, event, _refcon):
        cg = self._cg
        if event_type in (kCGEventTapDisabledByTimeout, kCGEventTapDisabledByUserInput):
            if self._tap and cg:
                cg.CGEventTapEnable(self._tap, True)
            return event
        if event_type == kCGEventKeyDown and cg:
            keycode = cg.CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode)
            flags = cg.CGEventGetFlags(event)
            extras = kCGEventFlagMaskShift | kCGEventFlagMaskControl | kCGEventFlagMaskAlternate
            if (
                keycode == kVK_ANSI_T
                and (flags & kCGEventFlagMaskCommand)
                and not (flags & extras)
            ):
                self._flag.set()
                return None
        return event

    def _run(self) -> None:
        try:
            cg = _core_graphics()
            cf = _core_foundation()
            self._cg = cg
            self._callback = CGEventTapCallBack(self._on_event)
            mask = ctypes.c_uint64(1 << kCGEventKeyDown)
            tap = cg.CGEventTapCreate(
                kCGSessionEventTap,
                kCGHeadInsertEventTap,
                kCGEventTapOptionDefault,
                mask,
                self._callback,
                None,
            )
            if not tap:
                print(
                    "Could not listen for ⌘T. Enable Accessibility for Cursor or Terminal "
                    "(System Settings → Privacy & Security → Accessibility)."
                )
                return
            self._tap = tap
            source = cf.CFMachPortCreateRunLoopSource(None, tap, 0)
            if not source:
                print("Could not attach the ⌘T event tap to a run loop.")
                return
            self._source = source
            loop = cf.CFRunLoopGetCurrent()
            common = ctypes.c_void_p.in_dll(cf, "kCFRunLoopCommonModes")
            cf.CFRunLoopAddSource(loop, source, common)
            cg.CGEventTapEnable(tap, True)
            self._runloop = loop
            cf.CFRunLoopRun()
        except Exception as exc:
            print(f"⌘T hotkey listener failed: {exc}")


def start_cmd_t_monitor() -> CommandTMonitor:
    global _cmd_t_monitor
    if _cmd_t_monitor is None:
        _cmd_t_monitor = CommandTMonitor()
        _cmd_t_monitor.start()
    return _cmd_t_monitor


def async_page_up() -> None:
    threading.Thread(target=page_up, daemon=True).start()
