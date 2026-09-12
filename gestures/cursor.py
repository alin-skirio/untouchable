"""Right-hand pointer cursor: two-finger point, hold-to-drag clicks, distance scale."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2

from .landmarks import INDEX_MCP, MIDDLE_MCP, WRIST, palm_size_px

# Screen pixels per image pixel at the ⌘T reference distance.
PIXEL_GAIN = 2.2
# Screen pixels per palm-length of travel when no reference is set.
PALM_GAIN = 750.0
SMOOTH_ALPHA = 0.35
DEADZONE_PX = 1.5
CONFIRM_FRAMES = 4
MIN_PALM_PX = 1.0

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


def load_reference_palm() -> Optional[float]:
    if not REFERENCE_JSON.is_file():
        return None
    try:
        data = json.loads(REFERENCE_JSON.read_text(encoding="utf-8"))
        palm = float(data.get("palm_px", 0.0))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return palm if palm > 0 else None


def save_reference(frame, palm_px: float, frame_w: int, frame_h: int) -> bool:
    payload = {"palm_px": palm_px, "frame_w": frame_w, "frame_h": frame_h}
    try:
        REFERENCE_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if frame is not None:
            cv2.imwrite(str(REFERENCE_PNG), frame)
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


def _is_click(down: set[str]) -> bool:
    return {"index", "middle", "ring", "pinky"}.issubset(down)


def _tracking_point(hand_landmarks, width: float, height: float) -> tuple[float, float]:
    lm = hand_landmarks.landmark
    x = (lm[WRIST].x + lm[INDEX_MCP].x + lm[MIDDLE_MCP].x) / 3.0
    y = (lm[WRIST].y + lm[INDEX_MCP].y + lm[MIDDLE_MCP].y) / 3.0
    return x * width, y * height


class PointerCursor:
    """Relative macOS pointer from a two-finger pose, scaled by a ⌘T reference."""

    def __init__(
        self,
        confirm_frames: int = CONFIRM_FRAMES,
        pixel_gain: float = PIXEL_GAIN,
        palm_gain: float = PALM_GAIN,
        smooth_alpha: float = SMOOTH_ALPHA,
        deadzone_px: float = DEADZONE_PX,
    ):
        self.confirm_frames = confirm_frames
        self.pixel_gain = pixel_gain
        self.palm_gain = palm_gain
        self.smooth_alpha = smooth_alpha
        self.deadzone_px = deadzone_px
        self.idle_smooth_alpha = 0.18
        self.gesture_smooth_alpha = 0.35
        self.idle_deadzone_px = 0.8
        self.ref_palm_px: Optional[float] = load_reference_palm()
        self.engaged = False
        self.held_button: Optional[str] = None
        self._enter_count = 0
        self._clicking = False
        self._clutch = True
        self._filtered: Optional[tuple[float, float]] = None
        self._prev: Optional[tuple[float, float]] = None

    @property
    def has_reference(self) -> bool:
        return self.ref_palm_px is not None and self.ref_palm_px > 0

    def set_reference(self, palm_px: float, frame=None, frame_size=None) -> bool:
        if palm_px <= 0:
            return False
        self.ref_palm_px = palm_px
        self._clutch = True
        self._filtered = None
        self._prev = None
        if frame is not None and frame_size is not None:
            return save_reference(frame, palm_px, int(frame_size[0]), int(frame_size[1]))
        return True

    def reset(self) -> Optional[str]:
        released = self.held_button
        self.engaged = False
        self.held_button = None
        self._enter_count = 0
        self._clicking = False
        self._clutch = True
        self._filtered = None
        self._prev = None
        return released

    def _scale(self, current_palm_px: float) -> float:
        palm = max(current_palm_px, MIN_PALM_PX)
        if self.has_reference:
            return self.pixel_gain * (self.ref_palm_px / palm)
        return self.palm_gain / palm

    def update(self, hand_landmarks, fingers_down: list[str], frame_size) -> CursorUpdate:
        width, height = frame_size
        down = set(fingers_down)
        move = _is_move(down)
        click = _is_click(down)

        if not self.engaged:
            if move:
                self._enter_count += 1
                if self._enter_count >= self.confirm_frames:
                    self.engaged = True
                    self._clutch = True
                    self._filtered = None
                    self._prev = None
            else:
                self._enter_count = 0
            if not self.engaged:
                return CursorUpdate(False, 0.0, 0.0, None, None)

        if not move and not click:
            released = self.reset()
            return CursorUpdate(False, 0.0, 0.0, None, released)

        raw = _tracking_point(hand_landmarks, width, height)
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

        button_down: Optional[str] = None
        button_up: Optional[str] = None
        if click and not self._clicking:
            self._clicking = True
            self.held_button = "right" if "thumb" not in down else "left"
            button_down = self.held_button
        elif move and self._clicking:
            self._clicking = False
            button_up = self.held_button
            self.held_button = None

        return CursorUpdate(True, dx, dy, button_down, button_up)
