"""Right-hand pointer: index-tip tracking, thumb-to-ring/index tap-or-hold, desk × distance scale."""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2

from .landmarks import index_tip_px, palm_size_px, thumb_to_index_side, thumb_to_ring_side

DEFAULT_BASE_SENSITIVITY = 1.0
SENSITIVITY_MIN = 0.25
SENSITIVITY_MAX = 3.0
SMOOTH_ALPHA = 0.35
DEADZONE_PX = 1.5
CONFIRM_FRAMES = 4
MIN_PALM_PX = 1.0
CLICK_ON = 0.36
CLICK_OFF = 0.52
HOLD_CLICK = 0.5
TOGGLE_HISTORY = 8
TOGGLE_COOLDOWN = 15
LEAVE_GRACE = 12
MAIN_FINGERS = frozenset({"index", "middle", "ring", "pinky"})

REFERENCE_DIR = Path(__file__).resolve().parent.parent
REFERENCE_PNG = REFERENCE_DIR / "hand_reference.png"
REFERENCE_JSON = REFERENCE_DIR / "hand_reference.json"


@dataclass
class CursorUpdate:
    engaged: bool
    dx: float
    dy: float
    button_down: Optional[str]
    button_up: Optional[str]
    click: Optional[str] = None


def _clamp_sensitivity(value: float) -> float:
    return min(SENSITIVITY_MAX, max(SENSITIVITY_MIN, value))


def load_reference() -> tuple[Optional[float], float]:
    """Return (palm_px or None, base_sensitivity)."""
    if not REFERENCE_JSON.is_file():
        return None, DEFAULT_BASE_SENSITIVITY
    try:
        data = json.loads(REFERENCE_JSON.read_text(encoding="utf-8"))
        palm = float(data.get("palm_px", 0.0) or 0.0)
        gain = float(data.get("base_sensitivity", DEFAULT_BASE_SENSITIVITY))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None, DEFAULT_BASE_SENSITIVITY
    return (palm if palm > 0 else None), _clamp_sensitivity(gain)


def save_reference(
    frame,
    palm_px: float,
    frame_w: int,
    frame_h: int,
    base_sensitivity: float,
) -> bool:
    payload = {
        "palm_px": palm_px,
        "frame_w": frame_w,
        "frame_h": frame_h,
        "base_sensitivity": _clamp_sensitivity(base_sensitivity),
    }
    try:
        REFERENCE_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if frame is not None:
            cv2.imwrite(str(REFERENCE_PNG), frame)
        return True
    except OSError:
        return False


def persist_sensitivity(base_sensitivity: float) -> bool:
    data: dict = {}
    if REFERENCE_JSON.is_file():
        try:
            loaded = json.loads(REFERENCE_JSON.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            data = {}
    data["base_sensitivity"] = _clamp_sensitivity(base_sensitivity)
    try:
        REFERENCE_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return True
    except OSError:
        return False


def _is_move(down: set[str]) -> bool:
    return (
        "index" not in down
        and "middle" not in down
        and "ring" in down
        and "pinky" in down
    )


def _pose_name(down: set[str]) -> str:
    if _is_move(down):
        return "point"
    curled = len(MAIN_FINGERS & down)
    if curled >= 4:
        return "fist"
    if curled == 0:
        return "open"
    return "other"


class PointerCursor:
    """Relative macOS pointer from the index tip, scaled by desk sensitivity × palm ratio."""

    def __init__(
        self,
        confirm_frames: int = CONFIRM_FRAMES,
        base_sensitivity: Optional[float] = None,
        smooth_alpha: float = SMOOTH_ALPHA,
        deadzone_px: float = DEADZONE_PX,
    ):
        palm, stored_gain = load_reference()
        self.confirm_frames = confirm_frames
        self.base_sensitivity = (
            stored_gain if base_sensitivity is None else _clamp_sensitivity(base_sensitivity)
        )
        self.smooth_alpha = smooth_alpha
        self.deadzone_px = deadzone_px
        self.idle_smooth_alpha = 0.18
        self.gesture_smooth_alpha = 0.35
        self.idle_deadzone_px = 0.8
        self.ref_palm_px: Optional[float] = palm
        self.engaged = False
        self.held_button: Optional[str] = None
        self._enter_count = 0
        self._clutch = True
        self._filtered: Optional[tuple[float, float]] = None
        self._prev: Optional[tuple[float, float]] = None
        self._poses: deque[str] = deque(maxlen=TOGGLE_HISTORY)
        self._toggle_cooldown = 0
        self._leave_grace = 0
        self._pending_button: Optional[str] = None
        self._pending_at = 0.0

    @property
    def has_reference(self) -> bool:
        return self.ref_palm_px is not None and self.ref_palm_px > 0

    def distance_ratio(self, current_palm_px: float) -> float:
        if not self.has_reference or current_palm_px <= 0:
            return 1.0
        return self.ref_palm_px / max(current_palm_px, MIN_PALM_PX)

    def set_base_sensitivity(self, value: float, persist: bool = True) -> None:
        self.base_sensitivity = _clamp_sensitivity(value)
        if persist:
            persist_sensitivity(self.base_sensitivity)

    def set_reference(self, palm_px: float, frame=None, frame_size=None) -> bool:
        if palm_px <= 0:
            return False
        self.ref_palm_px = palm_px
        self._clutch = True
        self._filtered = None
        self._prev = None
        if frame is not None and frame_size is not None:
            return save_reference(
                frame,
                palm_px,
                int(frame_size[0]),
                int(frame_size[1]),
                self.base_sensitivity,
            )
        return persist_sensitivity(self.base_sensitivity)

    def engage_from_s(self) -> None:
        """Engage immediately and clutch so the next relative frame does not jump."""
        self.engaged = True
        self._enter_count = self.confirm_frames
        self._clutch = True
        self._filtered = None
        self._prev = None
        self._leave_grace = 0
        self._toggle_cooldown = TOGGLE_COOLDOWN
        self._pending_button = None
        self._pending_at = 0.0

    def reset(self) -> Optional[str]:
        released = self.held_button
        self.engaged = False
        self.held_button = None
        self._enter_count = 0
        self._clutch = True
        self._filtered = None
        self._prev = None
        self._leave_grace = 0
        self._pending_button = None
        self._pending_at = 0.0
        return released

    def _scale(self, current_palm_px: float) -> float:
        return self.base_sensitivity * self.distance_ratio(current_palm_px)

    def _fast_fist_toggle(self) -> bool:
        """True on a very fast open + closed fist that lands on the pointing pose."""
        if self._toggle_cooldown > 0:
            return False
        poses = list(self._poses)
        if len(poses) < 3 or poses[-1] != "point":
            return False
        prior = poses[:-1]
        if "fist" not in prior or "open" not in prior:
            return False
        non_point = [pose for pose in prior if pose != "point"]
        return bool(non_point) and non_point[-1] == "fist"

    def _begin(self) -> None:
        self.engaged = True
        self._clutch = True
        self._filtered = None
        self._prev = None
        self._leave_grace = 0
        self._toggle_cooldown = TOGGLE_COOLDOWN
        self._pending_button = None
        self._pending_at = 0.0

    def _contact(self, distance: float, button: str) -> bool:
        sticky = self.held_button == button or self._pending_button == button
        return distance <= (CLICK_OFF if sticky else CLICK_ON)

    def _update_click(
        self, hand_landmarks, *, allow_press: bool = True
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        ring = thumb_to_ring_side(hand_landmarks)
        index = thumb_to_index_side(hand_landmarks)
        left_hit = self._contact(ring, "left")
        right_hit = self._contact(index, "right")
        active = self.held_button or self._pending_button
        button_down: Optional[str] = None
        button_up: Optional[str] = None
        click: Optional[str] = None

        def still_on(button: str) -> bool:
            if button == "left":
                return left_hit
            if button == "right":
                return right_hit
            return False

        if active:
            if still_on(active):
                if (
                    self.held_button is None
                    and time.monotonic() - self._pending_at >= HOLD_CLICK
                ):
                    self.held_button = active
                    self._pending_button = None
                    button_down = active
            else:
                if self.held_button:
                    button_up = self.held_button
                    self.held_button = None
                elif self._pending_button:
                    click = self._pending_button
                self._pending_button = None
        elif allow_press:
            if left_hit and right_hit:
                target = "left" if ring <= index else "right"
            elif left_hit:
                target = "left"
            elif right_hit:
                target = "right"
            else:
                target = None
            if target:
                self._pending_button = target
                self._pending_at = time.monotonic()
        return button_down, button_up, click

    def update(self, hand_landmarks, fingers_down: list[str], frame_size) -> CursorUpdate:
        width, height = frame_size
        down = set(fingers_down)
        move = _is_move(down)

        if self._toggle_cooldown > 0:
            self._toggle_cooldown -= 1

        self._poses.append(_pose_name(down))
        if self._fast_fist_toggle():
            if self.engaged:
                released = self.reset()
                self._toggle_cooldown = TOGGLE_COOLDOWN
                return CursorUpdate(False, 0.0, 0.0, None, released)
            self._begin()

        if not self.engaged:
            return CursorUpdate(False, 0.0, 0.0, None, None)

        if not move:
            self._leave_grace += 1
            if self._leave_grace >= LEAVE_GRACE:
                released = self.reset()
                return CursorUpdate(False, 0.0, 0.0, None, released)
            self._clutch = True
            _, button_up, click = self._update_click(hand_landmarks, allow_press=False)
            return CursorUpdate(True, 0.0, 0.0, None, button_up, click)

        self._leave_grace = 0

        raw = index_tip_px(hand_landmarks, width, height)
        palm = palm_size_px(hand_landmarks, width, height)
        if self._filtered is None:
            self._filtered = raw
        else:
            alpha = self.gesture_smooth_alpha if self.engaged else self.idle_smooth_alpha
            self._filtered = (
                alpha * raw[0] + (1.0 - alpha) * self._filtered[0],
                alpha * raw[1] + (1.0 - alpha) * self._filtered[1],
            )

        dx = dy = 0.0
        if self._clutch or self._prev is None:
            self._prev = self._filtered
            self._clutch = False
        else:
            dximg = self._filtered[0] - self._prev[0]
            dyimg = self._filtered[1] - self._prev[1]
            self._prev = self._filtered
            scale = self._scale(palm)
            dx = dximg * scale
            dy = dyimg * scale
            deadzone = self.deadzone_px if self.engaged else self.idle_deadzone_px
            if math.hypot(dx, dy) < deadzone:
                dx = dy = 0.0

        button_down, button_up, click = self._update_click(hand_landmarks)
        return CursorUpdate(True, dx, dy, button_down, button_up, click)
