"""Camera chrome: quiet-luxury HUD over the live feed."""

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

GESTURES = (
    (TEAL, "Pointer", "open → fist → gun"),
    (VIOLET, "Clicker", "thumb + middle"),
    (TEAL, "Switcher", "pinch + flap"),
    (GREEN, "Scroll", "index or pinky"),
    (VIOLET, "Flick", "open from a fist"),
    (TEAL, "TikTok", "two-handed T"),
    (VIOLET, "Voice", "hey Grok"),
)

RAIL_W = 248
RAIL_H = 128


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
        elif action == "toggle_chrome":
            self.chrome = not self.chrome
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

    def draw_panel(
        self,
        frame,
        status_lines: list[str],
        *,
        pointer_engaged: bool = False,
    ) -> None:
        self._cam_hits = []
        state, mode, meta, hint = _status_view(status_lines)
        locked = state == "Locked"
        who = _who(state, status_lines)
        if locked:
            chrome.dim_frame(frame, 0.46)
        glow = TEAL if pointer_engaged and not locked else MUTE
        if locked:
            glow = AMBER
        chrome.hairline_brackets(frame, inset=14, length=16, color=glow)
        _hud_toggle(frame, self.chrome, self._cam_hits)
        if locked:
            _lock_banner(frame, who)
        elif self.chrome:
            _brand_chip(frame)
            _unlock_pill(frame, who)
            if hint and not self.faces_open:
                _hint_bar(frame, hint)
            _gesture_legend(frame)
            _dock(frame, self._cam_hits)
        if self.faces_open:
            _draw_faces_panel(frame, self)

    def draw_rail(self, status_lines: list[str] | None = None) -> None:
        self._rail_hits = []
        img = _blank_rail()
        state, mode, _meta, _hint = _status_view(status_lines or [])
        locked = state == "Locked"
        chrome.glass_panel(img, 6, 6, RAIL_W - 6, RAIL_H - 6, radius=16)
        chrome.text(img, "Camera hidden", 18, 22, size=14, color=TEXT, anchor="lt")
        who = state if state not in ("Live",) else "Waiting for a face"
        detail = f"{who}  ·  {mode}"
        chrome.text(img, detail[:42], 18, 44, size=12, color=DIM, alpha=140, anchor="lt")
        chrome.text(
            img,
            "Gestures still run while tucked away.",
            18,
            62,
            size=11,
            color=MUTE,
            alpha=160,
            anchor="lt",
        )
        _pill_btn(img, 14, 86, 140, "Show camera", self._rail_hits, "toggle_preview", primary=True)
        _pill_btn(img, 160, 86, 72, "Quit", self._rail_hits, "quit", danger=True)
        cv2.circle(img, (RAIL_W - 20, 22), 4, ROSE if locked else GREEN, -1, cv2.LINE_AA)
        cv2.imshow(RAIL_WINDOW, img)


def _blank_rail():
    import numpy as np

    return np.full((RAIL_H, RAIL_W, 3), chrome.BASE, dtype=np.uint8)


def _who(state: str, lines: list[str]) -> str:
    if state.startswith("Hi, "):
        return chrome.display_name(state[4:])
    for raw in lines:
        if raw.startswith("Face: "):
            name = raw[6:].strip()
            if name and name != "Unknown":
                return chrome.display_name(name)
    return ""


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
    text = hint[:56]
    tw, _ = chrome.measure(text, 11)
    bar_w = min(frame.shape[1] - 32, tw + 24)
    x1, y1 = 14, frame.shape[0] - 72
    chrome.glass_panel(frame, x1, y1, x1 + bar_w, y1 + 24, radius=12)
    chrome.text(frame, text, x1 + 12, y1 + 12, size=11, color=DIM, alpha=180, anchor="lm")


def _gesture_legend(frame) -> None:
    h, w = frame.shape[:2]
    row_h = 18
    pad_x, pad_y = 12, 10
    box_w = 212
    box_h = pad_y * 2 + 12 + len(GESTURES) * row_h
    x = w - box_w - 12
    y = 48
    if y + box_h > h - 56:
        box_h = max(40, h - 56 - y)
    chrome.glass_panel(frame, x, y, x + box_w, y + box_h, radius=14)
    chrome.text(frame, "GESTURES", x + pad_x, y + pad_y, size=9, color=MUTE, alpha=160, weight="medium", anchor="lt")
    row_y = y + pad_y + 14
    for color, title, detail in GESTURES:
        if row_y + 12 > y + box_h - 6:
            break
        cv2.circle(frame, (x + pad_x + 3, row_y + 6), 2, color, -1, cv2.LINE_AA)
        chrome.text(
            frame,
            f"{title} · {detail}",
            x + pad_x + 12,
            row_y + 6,
            size=11,
            color=TEXT,
            alpha=210,
            anchor="lm",
        )
        row_y += row_h


def _dock(frame, hits: list) -> None:
    items = (
        ("Camera", "C", "toggle_preview"),
        ("Faces", "A / L", "toggle_faces"),
        ("Snap", "S", "snap_cursor"),
        ("Quit", "Q", "quit"),
    )
    h, w = frame.shape[:2]
    gap = 4
    pill_w = 78
    total = len(items) * pill_w + (len(items) - 1) * gap
    x = max(14, (w - total) // 2)
    y = h - 42
    chrome.glass_panel(frame, x - 8, y - 6, x + total + 8, y + 32, radius=16)
    for title, shortcut, action in items:
        _dock_item(frame, x, y, pill_w - 2, title, shortcut, hits, action)
        x += pill_w + gap


def _dock_item(frame, x, y, width, title, shortcut, hits, action) -> None:
    x2, y2 = x + width, y + 26
    chrome.text(frame, title, x + width // 2, y + 8, size=11, color=TEXT, alpha=220, anchor="mm")
    chrome.text(frame, shortcut, x + width // 2, y + 20, size=9, color=DIM, alpha=160, anchor="mm")
    hits.append(((x, y, x2, y2), action))


def _hud_toggle(frame, on: bool, hits: list) -> None:
    h, w = frame.shape[:2]
    label = "H · HUD  ON" if on else "H · HUD"
    tw, _ = chrome.measure(label, 11)
    width = tw + 28
    x1 = w - width - 12
    y1 = h - 36
    chrome.pill(
        frame,
        x1,
        y1,
        label,
        dot=TEAL if on else MUTE,
        height=24,
        pad_x=10,
        size=10,
        fill=(14, 16, 20, 180),
    )
    hits.append(((x1, y1, x1 + width, y1 + 24), "toggle_chrome"))


def _draw_faces_panel(frame, hud: CameraHud) -> None:
    h, w = frame.shape[:2]
    chrome.dim_frame(frame, 0.4)
    box_w = min(400, w - 32)
    row_h = 30
    names = hud.faces
    list_h = max(row_h, min(8, max(1, len(names))) * row_h)
    box_h = 88 + list_h + 50
    x1 = (w - box_w) // 2
    y1 = max(56, (h - box_h) // 2)
    x2, y2 = x1 + box_w, y1 + box_h
    chrome.glass_panel(frame, x1, y1, x2, y2, radius=16, fill=(10, 11, 15, 220))
    chrome.text(frame, "People", x1 + 18, y1 + 20, size=16, color=TEXT, weight="medium", anchor="lt")
    chrome.text(
        frame,
        "Pick a face to delete, or add someone new.",
        x1 + 18,
        y1 + 42,
        size=12,
        color=DIM,
        alpha=180,
        anchor="lt",
    )
    list_y = y1 + 58
    if not names:
        chrome.text(
            frame,
            "No saved faces yet — tap Add to start.",
            x1 + 22,
            list_y + 20,
            size=13,
            color=DIM,
            alpha=180,
            anchor="lt",
        )
    else:
        for i, name in enumerate(names):
            ry1 = list_y + i * row_h
            ry2 = ry1 + row_h - 6
            selected = i == hud.faces_sel
            chrome.glass_panel(
                frame,
                x1 + 16,
                ry1,
                x2 - 16,
                ry2,
                radius=10,
                fill=(20, 22, 26, 200) if selected else (16, 16, 18, 140),
                outline=(94, 234, 212, 80) if selected else (255, 255, 255, 20),
            )
            chrome.text(
                frame,
                name,
                x1 + 28,
                ry1 + 12,
                size=13,
                color=TEXT if selected else DIM,
                anchor="lt",
            )
            hud._cam_hits.append(((x1 + 16, ry1, x2 - 16, ry2), f"pick_face:{i}"))

    btn_y = y2 - 40
    _pill_btn(frame, x1 + 16, btn_y, 100, "Add person", hud._cam_hits, "add_face", primary=True)
    if hud.confirm_name:
        _pill_btn(
            frame,
            x1 + 122,
            btn_y,
            160,
            f"Delete {hud.confirm_name}?",
            hud._cam_hits,
            "confirm_delete",
            danger=True,
        )
        _pill_btn(frame, x1 + 288, btn_y, 80, "Keep", hud._cam_hits, "cancel_delete")
    else:
        _pill_btn(frame, x1 + 122, btn_y, 80, "Delete", hud._cam_hits, "ask_delete", danger=True)
        _pill_btn(frame, x1 + 208, btn_y, 80, "Done", hud._cam_hits, "close_faces")


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
    if danger:
        fill, edge, color = (36, 20, 28, 200), (251, 113, 133, 90), ROSE
    elif primary:
        fill, edge, color = (18, 28, 28, 200), (94, 234, 212, 70), TEXT
    else:
        fill, edge, color = (20, 22, 26, 180), (255, 255, 255, 28), TEXT
    chrome.glass_panel(img, x, y, x + width, y + 30, radius=15, fill=fill, outline=edge)
    chrome.text(img, label[:18], x + width // 2, y + 15, size=12, color=color, anchor="mm")
    hits.append(((x, y, x + width, y + 30), action))


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
    cv2.namedWindow(CAM_WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowTitle(CAM_WINDOW, "Hand Control")
    hud.attach_cam(CAM_WINDOW)


def window_open(name: str) -> bool:
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False
