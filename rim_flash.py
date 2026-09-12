"""Thin click-through bezel cue when pointer mode arms or releases."""

from __future__ import annotations

import sys
import threading
import time
from typing import Optional

DURATION = 0.58
FPS = 60
THICK = 4
CORNER = 26
SPARK = 110

_lock = threading.Lock()
_pending: Optional[bool] = None
_thread: Optional[threading.Thread] = None


def flash_pointer_rim(engage: bool) -> None:
    """Play a hairline screen-edge cue. Safe to call from the tracking loop."""
    global _pending, _thread
    with _lock:
        _pending = bool(engage)
        if _thread is None or not _thread.is_alive():
            _thread = threading.Thread(target=_rim_worker, daemon=True)
            _thread.start()


def _rim_worker() -> None:
    global _pending, _thread
    while True:
        with _lock:
            if _pending is None:
                _thread = None
                return
            engage = _pending
            _pending = None
        try:
            _play_rim(engage)
        except Exception:
            pass


def _play_rim(engage: bool) -> None:
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    ox, oy, sw, sh = _screen_box(root)
    if sw < 8 or sh < 8:
        root.destroy()
        return

    root.overrideredirect(True)
    root.geometry(f"{int(sw)}x{int(sh)}+{int(ox)}+{int(oy)}")
    root.attributes("-topmost", True)
    try:
        root.attributes("-toolwindow", True)
    except tk.TclError:
        pass

    bg = "black"
    if sys.platform == "darwin":
        try:
            root.wm_attributes("-transparent", True)
            bg = "systemTransparent"
        except tk.TclError:
            pass
        try:
            root.tk.call(
                "::tk::unsupported::MacWindowStyle",
                "style",
                root._w,
                "help",
                "noActivates",
            )
        except tk.TclError:
            pass
    else:
        try:
            root.wm_attributes("-transparentcolor", "black")
        except tk.TclError:
            pass

    root.config(bg=bg)
    canvas = tk.Canvas(
        root,
        width=int(sw),
        height=int(sh),
        bg=bg,
        highlightthickness=0,
        bd=0,
    )
    canvas.pack(fill="both", expand=True)
    root.deiconify()
    root.update_idletasks()
    root.update()
    _make_clickthrough(root)

    rgb = (92, 225, 255) if engage else (255, 138, 92)
    t0 = time.monotonic()
    frame_dt = 1.0 / FPS

    while True:
        with _lock:
            if _pending is not None:
                break
        t = (time.monotonic() - t0) / DURATION
        if t >= 1.0:
            break
        canvas.delete("all")
        _draw_frame(canvas, sw, sh, min(max(t, 0.0), 1.0), engage, rgb)
        root.update()
        time.sleep(frame_dt)

    try:
        root.destroy()
    except tk.TclError:
        pass


def _screen_box(root) -> tuple[float, float, float, float]:
    if sys.platform == "darwin":
        try:
            from mac_keys import display_bounds

            ox, oy, sw, sh = display_bounds()
            if sw > 0 and sh > 0:
                return ox, oy, sw, sh
        except Exception:
            pass
    return 0.0, 0.0, float(root.winfo_screenwidth()), float(root.winfo_screenheight())


def _make_clickthrough(root) -> None:
    try:
        if sys.platform == "win32":
            import ctypes

            hwnd = ctypes.windll.user32.GetParent(root.winfo_id()) or root.winfo_id()
            gwl_exstyle = -20
            style = ctypes.windll.user32.GetWindowLongW(hwnd, gwl_exstyle)
            ctypes.windll.user32.SetWindowLongW(
                hwnd,
                gwl_exstyle,
                style | 0x00080000 | 0x00000020 | 0x08000000,
            )
        elif sys.platform == "darwin":
            from AppKit import NSApp

            for win in NSApp.windows():
                win.setIgnoresMouseEvents_(True)
    except Exception:
        pass


def _hex(rgb: tuple[int, int, int], gain: float) -> str:
    gain = max(0.0, min(1.0, gain))
    return "#{:02x}{:02x}{:02x}".format(
        int(rgb[0] * gain),
        int(rgb[1] * gain),
        int(rgb[2] * gain),
    )


def _envelope(t: float, engage: bool) -> float:
    if engage:
        if t < 0.12:
            return t / 0.12
        if t < 0.62:
            return 1.0
        return max(0.0, 1.0 - (t - 0.62) / 0.38)
    if t < 0.18:
        return 1.0
    return max(0.0, 1.0 - (t - 0.18) / 0.82)


def _draw_frame(canvas, sw: float, sh: float, t: float, engage: bool, rgb) -> None:
    env = _envelope(t, engage)
    if env <= 0.01:
        return
    inset = THICK / 2 + 1.0
    x0, y0 = inset, inset
    x1, y1 = sw - inset, sh - inset
    hair = env * (0.55 if engage else max(0.0, 0.45 - t))
    if hair > 0.04:
        canvas.create_rectangle(
            x0, y0, x1, y1, outline=_hex(rgb, hair), width=THICK
        )

    arm = CORNER * (1.0 if engage else max(0.0, 1.0 - t * 1.15))
    corner_gain = min(1.0, env * 1.15)
    if arm > 2:
        _corners(canvas, x0, y0, x1, y1, arm, _hex(rgb, corner_gain))

    perim = 2.0 * ((x1 - x0) + (y1 - y0))
    if perim <= 0:
        return
    if engage:
        head = (t * 1.08) * perim
        spark_gain = env * (1.0 if t < 0.78 else max(0.0, 1.0 - (t - 0.78) / 0.22))
    else:
        head = (1.0 - t) * perim
        spark_gain = env * max(0.0, 1.0 - t)
    if spark_gain > 0.05:
        _spark(canvas, x0, y0, x1, y1, perim, head, _hex(rgb, spark_gain))


def _corners(canvas, x0, y0, x1, y1, arm: float, color: str) -> None:
    w = THICK + 1
    kw = {"fill": color, "width": w, "capstyle": "round", "joinstyle": "round"}
    canvas.create_line(x0, y0 + arm, x0, y0, x0 + arm, y0, **kw)
    canvas.create_line(x1 - arm, y0, x1, y0, x1, y0 + arm, **kw)
    canvas.create_line(x0, y1 - arm, x0, y1, x0 + arm, y1, **kw)
    canvas.create_line(x1 - arm, y1, x1, y1, x1, y1 - arm, **kw)


def _spark(canvas, x0, y0, x1, y1, perim: float, head: float, color: str) -> None:
    steps = 10
    pts = []
    for i in range(steps):
        pts.append(_perim_point(x0, y0, x1, y1, perim, head - (SPARK * i / (steps - 1))))
    for a, b in zip(pts, pts[1:]):
        canvas.create_line(
            a[0], a[1], b[0], b[1], fill=color, width=THICK + 1, capstyle="round"
        )


def _perim_point(x0, y0, x1, y1, perim: float, dist: float) -> tuple[float, float]:
    d = dist % perim
    top, right, bottom, left = x1 - x0, y1 - y0, x1 - x0, y1 - y0
    if d <= top:
        return x0 + d, y0
    d -= top
    if d <= right:
        return x1, y0 + d
    d -= right
    if d <= bottom:
        return x1 - d, y1
    d -= bottom
    if d <= left:
        return x0, y1 - d
    return x0, y0
