"""Camera chrome: quiet-luxury HUD over the live feed."""

from __future__ import annotations

import cv2

import chrome
from chrome import AMBER, DIM, GREEN, MUTE, ROSE, TEAL, TEXT, VIOLET
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

GESTURES = (
    (TEAL, "Pointer", "Open → fist → gun"),
    (VIOLET, "Clicker", "Fold thumb + middle"),
    (TEAL, "App switcher", "Thumb+middle pinch, index flap"),
    (GREEN, "Scroll", "Index or pinky"),
    (VIOLET, "Flick scroll", "Open from a fist"),
    (TEAL, "TikTok", "Two-handed T"),
    (VIOLET, "Voice", 'After “hey Grok”'),
)

RAIL_W = 268
RAIL_H = 148


class CameraHud:
    def __init__(self) -> None:
        self.enabled = {key: True for key, _label in FEATURES}
        self.preview = False
        self.chrome = True
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
        chrome.hairline_brackets(frame, inset=16, length=20, color=glow)
        _live_mark(frame)
        _hud_toggle(frame, self.chrome, self._cam_hits)
        if locked:
            _lock_banner(frame, who)
        elif self.chrome:
            _wordmark(frame)
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
        chrome.glass_panel(img, 8, 8, RAIL_W - 8, RAIL_H - 8, radius=16)
        chrome.text(img, "Camera hidden", 22, 28, size=15, color=TEXT, anchor="lt")
        who = state if state not in ("Live",) else "Waiting for a face"
        detail = f"{who}  ·  {mode}"
        chrome.text(img, detail[:42], 22, 52, size=12, color=DIM, alpha=140, anchor="lt")
        chrome.text(
            img,
            "Gestures still run while this is tucked away.",
            22,
            74,
            size=11,
            color=MUTE,
            alpha=160,
            anchor="lt",
        )
        _pill_btn(img, 16, 100, 148, "Show camera", self._rail_hits, "toggle_preview", primary=True)
        _pill_btn(img, 172, 100, 80, "Quit", self._rail_hits, "quit", danger=True)
        cv2.circle(img, (RAIL_W - 22, 24), 4, ROSE if locked else GREEN, -1, cv2.LINE_AA)
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


def _wordmark(frame) -> None:
    tw, _ = chrome.measure("HAND", 13, "medium")
    chrome.text(frame, "HAND", 28, 28, size=13, color=TEAL, alpha=220, weight="medium", anchor="lt")
    chrome.text(
        frame,
        "CONTROL",
        28 + tw + 7,
        28,
        size=13,
        color=TEXT,
        alpha=210,
        weight="medium",
        anchor="lt",
    )


def _unlock_pill(frame, who: str) -> None:
    label = f"Unlocked · {who}" if who else "Unlocked"
    tw, _ = chrome.measure(label, 13)
    width = tw + 36
    x2 = frame.shape[1] - 22
    x1 = x2 - width
    chrome.pill(frame, x1, 18, label, dot=GREEN, height=28, size=12)


def _live_mark(frame) -> None:
    chrome.text(frame, "LIVE", 36, 48, size=10, color=MUTE, alpha=170, weight="medium", anchor="lt")


def _lock_banner(frame, who: str) -> None:
    h, w = frame.shape[:2]
    label = "Commands off · unfamiliar face"
    tw, _ = chrome.measure(label, 14)
    width = tw + 40
    x1 = (w - width) // 2
    y1 = 20
    chrome.glass_panel(
        frame,
        x1,
        y1,
        x1 + width,
        y1 + 34,
        radius=17,
        fill=(18, 12, 14, 190),
        outline=(251, 191, 36, 70),
    )
    cv2.circle(frame, (x1 + 16, y1 + 17), 3, AMBER, -1, cv2.LINE_AA)
    chrome.text(frame, label, x1 + 28, y1 + 17, size=13, color=AMBER, alpha=230, anchor="lm")
    if who:
        chrome.text(frame, who, w // 2, y1 + 52, size=12, color=ROSE, alpha=180, anchor="mt")


def _hint_bar(frame, hint: str) -> None:
    text = hint[:56]
    tw, _ = chrome.measure(text, 12)
    bar_w = min(frame.shape[1] - 40, tw + 28)
    x1, y1 = 22, frame.shape[0] - 86
    chrome.glass_panel(frame, x1, y1, x1 + bar_w, y1 + 28, radius=14)
    chrome.text(frame, text, x1 + 14, y1 + 14, size=12, color=DIM, alpha=180, anchor="lm")


def _gesture_legend(frame) -> None:
    h, w = frame.shape[:2]
    x = w - 220
    y = 78
    chrome.text(frame, "GESTURES", x, y, size=10, color=MUTE, alpha=160, weight="medium", anchor="lt")
    y += 16
    for color, title, detail in GESTURES:
        cv2.circle(frame, (x + 4, y + 8), 3, color, -1, cv2.LINE_AA)
        chrome.text(frame, title, x + 16, y + 8, size=12, color=TEXT, alpha=220, anchor="lm")
        chrome.text(frame, detail, x + 16, y + 22, size=10, color=DIM, alpha=155, anchor="lm")
        y += 36
        if y > h - 120:
            break


def _dock(frame, hits: list) -> None:
    items = (
        ("Camera", "C", "toggle_preview"),
        ("Faces", "A / L", "toggle_faces"),
        ("Snap cursor", "S", "snap_cursor"),
        ("Quit", "Q", "quit"),
    )
    h, w = frame.shape[:2]
    gap = 8
    pill_w = 118
    total = len(items) * pill_w + (len(items) - 1) * gap
    x = max(20, (w - total) // 2)
    y = h - 52
    chrome.glass_panel(frame, x - 10, y - 8, x + total + 10, y + 40, radius=20)
    for title, shortcut, action in items:
        _dock_item(frame, x, y, pill_w - 4, title, shortcut, hits, action)
        x += pill_w + gap


def _dock_item(frame, x, y, width, title, shortcut, hits, action) -> None:
    x2, y2 = x + width, y + 32
    chrome.text(frame, title, x + width // 2, y + 10, size=12, color=TEXT, alpha=220, anchor="mm")
    chrome.text(frame, shortcut, x + width // 2, y + 24, size=10, color=DIM, alpha=160, anchor="mm")
    hits.append(((x, y, x2, y2), action))


def _hud_toggle(frame, on: bool, hits: list) -> None:
    h, w = frame.shape[:2]
    label = "H · toggle HUD"
    tw, _ = chrome.measure(label, 11)
    x1 = w - tw - 78
    y1 = h - 34
    chrome.text(frame, label, x1, y1 + 12, size=11, color=MUTE, alpha=160, anchor="lm")
    btn = "ON" if on else "HUD"
    bx = w - 56
    chrome.pill(
        frame,
        bx,
        y1,
        btn,
        dot=TEAL if on else MUTE,
        height=24,
        pad_x=10,
        size=10,
        fill=(14, 16, 20, 180),
    )
    hits.append(((bx, y1, w - 16, y1 + 24), "toggle_chrome"))


def _draw_faces_panel(frame, hud: CameraHud) -> None:
    h, w = frame.shape[:2]
    chrome.dim_frame(frame, 0.4)
    box_w = min(440, w - 40)
    row_h = 34
    names = hud.faces
    list_h = max(row_h, min(8, max(1, len(names))) * row_h)
    box_h = 108 + list_h + 58
    x1 = (w - box_w) // 2
    y1 = max(86, (h - box_h) // 2)
    x2, y2 = x1 + box_w, y1 + box_h
    chrome.glass_panel(frame, x1, y1, x2, y2, radius=18, fill=(10, 11, 15, 220))
    chrome.text(frame, "People", x1 + 22, y1 + 28, size=18, color=TEXT, weight="medium", anchor="lt")
    chrome.text(
        frame,
        "Choose a saved face to delete, or add someone new.",
        x1 + 22,
        y1 + 52,
        size=12,
        color=DIM,
        alpha=180,
        anchor="lt",
    )
    list_y = y1 + 70
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
                x1 + 18,
                ry1,
                x2 - 18,
                ry2,
                radius=10,
                fill=(20, 22, 26, 200) if selected else (16, 16, 18, 140),
                outline=(94, 234, 212, 80) if selected else (255, 255, 255, 20),
            )
            chrome.text(
                frame,
                name,
                x1 + 32,
                ry1 + 14,
                size=13,
                color=TEXT if selected else DIM,
                anchor="lt",
            )
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
