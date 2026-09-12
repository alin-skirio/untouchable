"""Small on-camera chip and two stacked circle buttons on the right."""

from __future__ import annotations

import cv2
import numpy as np

from mac_keys import display_bounds

CAM_WINDOW = "Hand Control — tracker"
RAIL_WINDOW = "•"
TRACKBAR_MIN = 25
TRACKBAR_MAX = 300

FEATURES = (
    ("pointer", "Pointer"),
    ("app_switcher", "App switcher"),
    ("scroll_up", "Scroll up"),
    ("scroll_down", "Scroll down"),
    ("flick", "Flick scroll"),
    ("t_pose", "T-pose TikTok"),
    ("face", "Face ID"),
    ("landmarks", "Landmarks"),
)

CREAM = (236, 238, 248)
PANEL = (36, 30, 28)
PEACH = (186, 196, 255)
ROSE = (112, 108, 232)
RING = (255, 252, 248)
SHADOW = (18, 14, 12)

BTN_R = 26
RAIL_W = 84
RAIL_H = 168


class CameraHud:
    def __init__(self) -> None:
        self.enabled = {key: True for key, _label in FEATURES}
        self.preview = True
        self._cam_hits: list[tuple[tuple[int, int], int, str]] = []
        self._rail_hits: list[tuple[tuple[int, int], int, str]] = []
        self._pending: str | None = None

    def on(self, key: str) -> bool:
        return bool(self.enabled.get(key, True))

    def take_click(self) -> str | None:
        action = self._pending
        self._pending = None
        return action

    def _queue(self, action: str) -> None:
        self._pending = action

    def on_cam_mouse(self, event, x, y, _flags, _userdata) -> None:
        _hit(event, x, y, self._cam_hits, self._queue)

    def on_rail_mouse(self, event, x, y, _flags, _userdata) -> None:
        _hit(event, x, y, self._rail_hits, self._queue)

    def apply(self, action: str) -> None:
        if action == "toggle_preview":
            self.preview = not self.preview

    def attach_cam(self, window: str) -> None:
        cv2.setMouseCallback(window, self.on_cam_mouse)

    def attach_rail(self) -> None:
        cv2.setMouseCallback(RAIL_WINDOW, self.on_rail_mouse)

    def draw_panel(self, frame, status_lines: list[str]) -> None:
        self._cam_hits = []
        headline = _headline(status_lines)
        (tw, _th), _ = cv2.getTextSize(headline, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        h = 44
        w = min(frame.shape[1] - 28, max(132, tw + 48))
        _soft_pill(frame, 16, 16, 16 + w, 16 + h)
        cv2.circle(frame, (34, 16 + h // 2), 6, PEACH, -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            headline,
            (48, 16 + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            CREAM,
            1,
            cv2.LINE_AA,
        )
        hgt, width = frame.shape[:2]
        _draw_stack(
            frame,
            width - 18 - BTN_R,
            hgt // 2 - BTN_R - 8,
            self.preview,
            self._cam_hits,
        )

    def draw_rail(self) -> None:
        self._rail_hits = []
        img = np.full((RAIL_H, RAIL_W, 3), PANEL, dtype=np.uint8)
        _draw_stack(img, RAIL_W // 2, 18 + BTN_R, self.preview, self._rail_hits)
        cv2.imshow(RAIL_WINDOW, img)


def _headline(lines: list[str]) -> str:
    skip = ("dist ", "⌘t", "no hands")
    for line in lines:
        low = line.strip().lower()
        if not low or any(low.startswith(s) for s in skip):
            continue
        return line.strip()[:28]
    return "Ready"


def _hit(event, x, y, hits, queue) -> None:
    if event != cv2.EVENT_LBUTTONUP:
        return
    for (cx, cy), radius, action in hits:
        if (x - cx) ** 2 + (y - cy) ** 2 <= radius * radius:
            queue(action)
            return


def _soft_pill(frame, x1: int, y1: int, x2: int, y2: int) -> None:
    overlay = frame.copy()
    h = y2 - y1
    r = h // 2
    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), PANEL, -1, cv2.LINE_AA)
    cv2.circle(overlay, (x1 + r, y1 + r), r, PANEL, -1, cv2.LINE_AA)
    cv2.circle(overlay, (x2 - r, y1 + r), r, PANEL, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)


def _draw_stack(
    img,
    cx: int,
    top_cy: int,
    preview: bool,
    hits: list,
) -> None:
    gap = 18
    hide_cy = top_cy
    quit_cy = top_cy + 2 * BTN_R + gap
    _circle_btn(img, cx, hide_cy, PEACH, "hide" if preview else "show")
    _circle_btn(img, cx, quit_cy, ROSE, "quit")
    hits.append(((cx, hide_cy), BTN_R + 4, "toggle_preview"))
    hits.append(((cx, quit_cy), BTN_R + 4, "quit"))


def _circle_btn(img, cx: int, cy: int, fill, kind: str) -> None:
    cv2.circle(img, (cx + 1, cy + 3), BTN_R + 1, SHADOW, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), BTN_R, fill, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), BTN_R, RING, 2, cv2.LINE_AA)
    cv2.circle(img, (cx - 7, cy - 8), 7, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (cx - 7, cy - 8), 7, fill, 1, cv2.LINE_AA)
    if kind == "quit":
        s = 8
        cv2.line(img, (cx - s, cy - s), (cx + s, cy + s), RING, 2, cv2.LINE_AA)
        cv2.line(img, (cx + s, cy - s), (cx - s, cy + s), RING, 2, cv2.LINE_AA)
    elif kind == "show":
        cv2.circle(img, (cx, cy + 2), 8, RING, 2, cv2.LINE_AA)
        cv2.circle(img, (cx, cy + 2), 2, RING, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (cx - 8, cy + 2), (cx + 8, cy + 2), RING, 2, cv2.LINE_AA)


def _dock_rail() -> None:
    ox, oy, sw, sh = display_bounds()
    if sw <= 0 or sh <= 0:
        return
    x = int(ox + sw - RAIL_W - 12)
    y = int(oy + max(72, (sh - RAIL_H) / 2))
    try:
        cv2.resizeWindow(RAIL_WINDOW, RAIL_W, RAIL_H)
        cv2.moveWindow(RAIL_WINDOW, x, y)
    except cv2.error:
        pass


def open_rail(hud: CameraHud) -> None:
    cv2.namedWindow(RAIL_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(RAIL_WINDOW, RAIL_W, RAIL_H)
    hud.attach_rail()
    _dock_rail()


def open_camera_window(hud: CameraHud) -> None:
    cv2.namedWindow(CAM_WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowTitle(CAM_WINDOW, "Hand Control")
    hud.attach_cam(CAM_WINDOW)


def window_open(name: str) -> bool:
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False
