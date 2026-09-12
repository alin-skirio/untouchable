"""Small on-camera chip and two stacked circle buttons on the right."""

from __future__ import annotations

import cv2

import chrome
from chrome import AMBER, DIM, GREEN, MUTE, ROSE, TEAL, TEXT, VIOLET
from mac_keys import display_bounds

CAM_WINDOW = "Hand Control — tracker"
RAIL_WINDOW = "Hand Control"

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
        self.faces_open = False
        self.faces: list[str] = []
        self.faces_sel = 0
        self.confirm_name: str | None = None
        self._cam_hits: list[tuple[tuple[int, int, int, int], str]] = []
        self._rail_hits: list[tuple[tuple[int, int, int, int], str]] = []
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

    def draw_panel(
        self,
        frame,
        status_lines: list[str],
        *,
        pointer_engaged: bool = False,
    ) -> None:
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


def _status_view(lines: list[str]) -> tuple[str, str, str, str]:
    locked = False
    face = ""
    mode = "Ready"
    meta = ""
    hint = ""
    for raw in lines:
        line = raw.strip()
        low = line.lower()
        if line == "Locked":
            locked = True
        elif line.startswith("Face: "):
            face = line[6:]
        elif low.startswith("dist "):
            meta = line.split(" ", 1)[-1]
        elif line.startswith("Pointer ·"):
            mode = "Click"
        elif line == "Pointer" or line.startswith("Pointer:"):
            mode = "Pointer"
            if line.startswith("Pointer:"):
                hint = line.removeprefix("Pointer:").strip()
        elif line == "App switcher":
            mode = "Switcher"
        elif low.startswith("tiktok"):
            mode = "TikTok"
        elif low.startswith("scroll"):
            mode = "Scroll"
        elif low.startswith("flick"):
            mode = "Flick"
        elif line == "T pose held":
            mode = "Hold"
        elif line == "Sent Tab":
            mode = "Tab"
        elif "→" in line or line.startswith(("Reference", "Cursor", "Could not")):
            hint = line
        elif line == "No hands in view":
            hint = hint or "No hands in view"
    if locked:
        return "Locked", "Idle", meta, hint or "Look at the camera to unlock"
    if face and face != "Unknown":
        return f"Hi, {face}"[:18], mode, meta, hint
    if face == "Unknown":
        return "New face", mode, meta, hint or "Open People to save this face"
    return "Live", mode, meta, hint


def _status_bar(frame, state: str, mode: str, meta: str) -> None:
    width = frame.shape[1]
    x1, y1 = PAD, PAD
    x2, y2 = width - PAD, PAD + 36
    _round_rect(frame, x1, y1, x2, y2, 18, GRAPHITE, fill=True)
    _round_rect(frame, x1, y1, x2, y2, 18, LINE, fill=False)
    locked = state == "Locked"
    cv2.circle(frame, (x1 + 16, y1 + 18), 5, ALERT if locked else ICE, -1, cv2.LINE_AA)
    label = f"{state}   {mode}"
    if meta:
        label += f"   {meta}"
    _text(frame, label, x1 + 30, y1 + 24, 0.48, WHITE)


def _hint_bar(frame, hint: str) -> None:
    hgt, width = frame.shape[:2]
    text = hint[:56]
    (tw, _th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    bar_w = min(width - 2 * PAD, tw + 28)
    x1, y1 = PAD, hgt - PAD - 30
    _round_rect(frame, x1, y1, x1 + bar_w, y1 + 30, 15, GRAPHITE, fill=True)
    _round_rect(frame, x1, y1, x1 + bar_w, y1 + 30, 15, LINE, fill=False)
    _text(frame, text, x1 + 14, y1 + 20, 0.4, MUTED)


def _draw_cam_controls(frame, preview: bool, faces_open: bool, hits: list) -> None:
    width = frame.shape[1]
    y = PAD + 44
    labels = [
        ("Hide camera" if preview else "Show camera", "toggle_preview", False, False),
        ("Done" if faces_open else "People", "close_faces" if faces_open else "toggle_faces", False, False),
        ("Quit", "quit", True, False),
    ]
    gap = 8
    total = len(labels) * PILL_CAM + (len(labels) - 1) * gap
    x = width - PAD - total
    if x < PAD:
        x = PAD
    for label, action, danger, primary in labels:
        _pill_btn(frame, x, y, PILL_CAM, label, hits, action, danger=danger, primary=primary)
        x += PILL_CAM + gap


def _draw_faces_panel(frame, hud: CameraHud) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    box_w = min(440, w - 40)
    row_h = 34
    names = hud.faces
    list_h = max(row_h, min(8, max(1, len(names))) * row_h)
    box_h = 108 + list_h + 58
    x1 = (w - box_w) // 2
    y1 = max(PAD + 86, (h - box_h) // 2)
    x2, y2 = x1 + box_w, y1 + box_h

    _round_rect(frame, x1, y1, x2, y2, 18, INK, fill=True)
    _round_rect(frame, x1, y1, x2, y2, 18, LINE, fill=False)
    _text(frame, "People", x1 + 22, y1 + 32, 0.62, WHITE)
    _text(
        frame,
        "Choose a saved face to delete, or add someone new.",
        x1 + 22,
        y1 + 54,
        0.38,
        MUTED,
    )

    list_y = y1 + 70
    if not names:
        _text(frame, "No saved faces yet — tap Add to start.", x1 + 22, list_y + 24, 0.42, MUTED)
    else:
        for i, name in enumerate(names):
            ry1 = list_y + i * row_h
            ry2 = ry1 + row_h - 6
            selected = i == hud.faces_sel
            _round_rect(frame, x1 + 18, ry1, x2 - 18, ry2, 10, SOFT if selected else (24, 22, 20), fill=True)
            if selected:
                _round_rect(frame, x1 + 18, ry1, x2 - 18, ry2, 10, ICE, fill=False)
            _text(frame, name, x1 + 32, ry1 + 20, 0.46, WHITE if selected else MUTED)
            hud._cam_hits.append(((x1 + 18, ry1, x2 - 18, ry2), f"pick_face:{i}"))

    btn_y = y2 - 46
    _pill_btn(frame, x1 + 18, btn_y, 108, "Add person", hud._cam_hits, "add_face", primary=True)
    if hud.confirm_name:
        _pill_btn(
            frame,
            x1 + 134,
            btn_y,
            168,
            f"Delete {hud.confirm_name}?",
            hud._cam_hits,
            "confirm_delete",
            danger=True,
        )
        _pill_btn(frame, x1 + 310, btn_y, 92, "Keep", hud._cam_hits, "cancel_delete")
    else:
        _pill_btn(frame, x1 + 134, btn_y, 92, "Delete", hud._cam_hits, "ask_delete", danger=True)
        _pill_btn(frame, x1 + 234, btn_y, 92, "Done", hud._cam_hits, "close_faces")


def _pill_btn(
    img,
    x: int,
    y: int,
    width: int,
    label: str,
    hits: list,
    action: str,
    *,
    danger: bool = False,
    primary: bool = False,
) -> None:
    x2, y2 = x + width, y + BTN_H
    if danger:
        fill, edge, color = (36, 28, 48), ALERT, (190, 190, 255)
    elif primary:
        fill, edge, color = (48, 42, 22), ICE, WHITE
    else:
        fill, edge, color = SOFT, LINE, WHITE
    _round_rect(img, x, y, x2, y2, BTN_H // 2, fill, fill=True)
    _round_rect(img, x, y, x2, y2, BTN_H // 2, edge, fill=False)
    shown = label[:18]
    (tw, th), _ = cv2.getTextSize(shown, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    tx = x + max(8, (width - tw) // 2)
    ty = y + (BTN_H + th) // 2
    _text(img, shown, tx, ty, 0.4, color)
    hits.append(((x, y, x2, y2), action))


def _round_rect(img, x1, y1, x2, y2, radius: int, color, *, fill: bool) -> None:
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    radius = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if fill:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        for cx, cy in (
            (x1 + radius, y1 + radius),
            (x2 - radius, y1 + radius),
            (x1 + radius, y2 - radius),
            (x2 - radius, y2 - radius),
        ):
            cv2.circle(img, (cx, cy), radius, color, -1, cv2.LINE_AA)
        return
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, 1, cv2.LINE_AA)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, 1, cv2.LINE_AA)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, 1, cv2.LINE_AA)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, 1, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, 1, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, 1, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, 1, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, 1, cv2.LINE_AA)


def _text(img, text: str, x: int, y: int, scale: float, color) -> None:
    cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _hit_rect(event, x, y, hits, queue) -> None:
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
