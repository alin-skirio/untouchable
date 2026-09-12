"""Camera chrome: readable controls, status, and a Faces manager."""

from __future__ import annotations

import cv2
import numpy as np

from mac_keys import display_bounds

CAM_WINDOW = "Hand Control — tracker"
RAIL_WINDOW = "Hand Control"
KEYS_WINDOW = " "

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

INK = (18, 16, 14)
GRAPHITE = (32, 28, 24)
LINE = (62, 56, 50)
MUTED = (168, 160, 152)
WHITE = (244, 242, 238)
ICE = (210, 190, 40)
ALERT = (72, 72, 214)
SOFT = (44, 40, 36)

BTN_H = 30
PILL_CAM = 118
PAD = 12
RAIL_W = 268
RAIL_H = 148


class CameraHud:
    def __init__(self) -> None:
        self.enabled = {key: True for key, _label in FEATURES}
        self.preview = False
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
        _hit_rect(event, x, y, self._cam_hits, self._queue)

    def on_rail_mouse(self, event, x, y, _flags, _userdata) -> None:
        _hit_rect(event, x, y, self._rail_hits, self._queue)

    def set_faces(self, names: list[str]) -> None:
        self.faces = list(names)
        if self.faces_sel >= len(self.faces):
            self.faces_sel = max(0, len(self.faces) - 1)
        self.confirm_name = None

    def close_faces(self) -> None:
        self.faces_open = False
        self.confirm_name = None

    def apply(self, action: str) -> None:
        if action == "toggle_preview":
            self.preview = not self.preview
            if not self.preview:
                self.close_faces()
        elif action == "toggle_faces":
            self.faces_open = not self.faces_open
            self.confirm_name = None
        elif action == "close_faces":
            self.close_faces()
        elif action.startswith("pick_face:"):
            try:
                self.faces_sel = int(action.split(":", 1)[1])
            except ValueError:
                return
            self.confirm_name = None
        elif action == "ask_delete":
            if 0 <= self.faces_sel < len(self.faces):
                self.confirm_name = self.faces[self.faces_sel]
        elif action == "cancel_delete":
            self.confirm_name = None

    def attach_cam(self, window: str) -> None:
        cv2.setMouseCallback(window, self.on_cam_mouse)

    def attach_rail(self) -> None:
        cv2.setMouseCallback(RAIL_WINDOW, self.on_rail_mouse)

    def draw_panel(self, frame, status_lines: list[str]) -> None:
        self._cam_hits = []
        state, mode, meta, hint = _status_view(status_lines)
        _status_bar(frame, state, mode, meta)
        if hint and not self.faces_open:
            _hint_bar(frame, hint)
        _draw_cam_controls(frame, self.preview, self.faces_open, self._cam_hits)
        if self.faces_open:
            _draw_faces_panel(frame, self)

    def draw_rail(self, status_lines: list[str] | None = None) -> None:
        self._rail_hits = []
        img = np.full((RAIL_H, RAIL_W, 3), INK, dtype=np.uint8)
        _round_rect(img, 1, 1, RAIL_W - 2, RAIL_H - 2, 16, GRAPHITE, fill=True)
        _round_rect(img, 1, 1, RAIL_W - 2, RAIL_H - 2, 16, LINE, fill=False)

        state, mode, meta, _hint = _status_view(status_lines or [])
        locked = state == "Locked"
        cv2.circle(img, (28, 28), 6, ALERT if locked else ICE, -1, cv2.LINE_AA)
        _text(img, "Camera hidden", 44, 24, 0.48, WHITE)
        who = state if state not in ("Live",) else "Waiting for a face"
        detail = f"{who}  ·  {mode}"
        if meta:
            detail += f"  ·  {meta}"
        _text(img, detail[:34], 44, 46, 0.38, MUTED)
        _text(img, "Gestures still run while this is tucked away.", 18, 74, 0.34, MUTED)

        _pill_btn(img, 16, 96, 148, "Show camera", self._rail_hits, "toggle_preview", primary=True)
        _pill_btn(img, 172, 96, 80, "Quit", self._rail_hits, "quit", danger=True)
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
        elif low.startswith("unknown face"):
            hint = line
    if locked:
        return "Locked", "Idle", meta, hint or "A known face unlocks"
    if face and face != "Unknown":
        return f"Hi, {face}"[:18], mode, meta, hint
    if face == "Unknown":
        return "New face", mode, meta, hint or "Open People to save this face"
    return "Live", mode, meta, hint


def overlay_label(lines: list[str]) -> str:
    """One-line fallback. Prefer overlay_copy for the desktop card."""
    title, detail, _locked = overlay_copy(lines)
    if detail:
        return f"{title} · {detail}"
    return title


def overlay_copy(lines: list[str]) -> tuple[str, str, bool]:
    """Title, subtitle, and lock flag for the centered overlay card."""
    state, mode, _meta, hint = _status_view(lines)
    if state == "Locked":
        return "Locked", "", True
    if hint.lower().startswith("unknown face"):
        countdown = hint.replace("Unknown face — ", "").replace("Unknown face - ", "")
        return countdown or "Locking", "", False
    who = state.removeprefix("Hi, ").strip()
    if who in ("", "Live"):
        who = "Hand Control"
    if who == "New face":
        return "New face", "", False
    if mode in ("Ready", "Idle", ""):
        return who, "", False
    return mode, "", False


status_view = _status_view


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
    for (x1, y1, x2, y2), action in hits:
        if x1 <= x <= x2 and y1 <= y <= y2:
            queue(action)
            return


def _dock_rail() -> None:
    ox, oy, sw, sh = display_bounds()
    if sw <= 0 or sh <= 0:
        return
    x = int(ox + sw - RAIL_W - 24)
    y = int(oy + sh - RAIL_H - 72)
    try:
        cv2.resizeWindow(RAIL_WINDOW, RAIL_W, RAIL_H)
        cv2.moveWindow(RAIL_WINDOW, x, y)
    except cv2.error:
        pass


def open_rail(hud: CameraHud) -> None:
    cv2.namedWindow(RAIL_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(RAIL_WINDOW, RAIL_W, RAIL_H)
    cv2.setWindowTitle(RAIL_WINDOW, "Hand Control")
    hud.attach_rail()
    _dock_rail()


def open_camera_window(hud: CameraHud) -> None:
    close_key_sink()
    cv2.namedWindow(CAM_WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowTitle(CAM_WINDOW, "Hand Control")
    hud.attach_cam(CAM_WINDOW)


def open_key_sink() -> None:
    """Tiny off-screen window so C and Q still work in overlay mode."""
    if window_open(KEYS_WINDOW):
        return
    cv2.namedWindow(KEYS_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(KEYS_WINDOW, 2, 2)
    try:
        cv2.moveWindow(KEYS_WINDOW, -120, -120)
    except cv2.error:
        pass


def close_key_sink() -> None:
    if not window_open(KEYS_WINDOW):
        return
    try:
        cv2.destroyWindow(KEYS_WINDOW)
    except cv2.error:
        pass


def window_open(name: str) -> bool:
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False
