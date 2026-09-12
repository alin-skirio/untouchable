"""Right-hand pointer: open → fist → index+middle+thumb out, then index-tip tracking.

While engaged, folding the thumb left-clicks and folding the middle finger right-clicks.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2

from .landmarks import index_tip_px, palm_size_px

DEFAULT_BASE_SENSITIVITY = 1.0
SENSITIVITY_MIN = 0.25
SENSITIVITY_MAX = 3.0
SMOOTH_ALPHA = 0.35
DEADZONE_PX = 1.5
CONFIRM_FRAMES = 4
MIN_PALM_PX = 1.0
LEAVE_GRACE = 12
CLICK_CONFIRM = 2
MAIN_FINGERS = frozenset({"index", "middle", "ring", "pinky"})

REFERENCE_DIR = Path(__file__).resolve().parent.parent
REFERENCE_PNG = REFERENCE_DIR / "hand_reference.png"
REFERENCE_JSON = REFERENCE_DIR / "hand_reference.json"


@dataclass
class CursorUpdate:
    engaged: bool
    dx: float
    dy: float
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


def _is_open(down: set[str]) -> bool:
    """All five digits extended."""
    return "thumb" not in down and not (MAIN_FINGERS & down)


def _is_fist(down: set[str]) -> bool:
    return len(MAIN_FINGERS & down) >= 4


def _is_move(down: set[str]) -> bool:
    """Index, middle, and thumb out; ring and pinky curled."""
    return (
        "thumb" not in down
        and "index" not in down
        and "middle" not in down
        and "ring" in down
        and "pinky" in down
    )


def _is_pointer_hold(down: set[str]) -> bool:
    """Stay in pointer while index tracks; thumb/middle may fold for clicks."""
    return "index" not in down and "ring" in down and "pinky" in down


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
        self.arm_hint = "Pointer: open hand to arm"
        self._seq = "idle"
        self._enter_count = 0
        self._clutch = True
        self._filtered: Optional[tuple[float, float]] = None
        self._prev: Optional[tuple[float, float]] = None
        self._leave_grace = 0
        self._thumb_fold = 0
        self._middle_fold = 0

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
        self._thumb_fold = 0
        self._middle_fold = 0
        self._seq = "idle"
        self.arm_hint = "Pointer"

    def reset(self) -> None:
        self.engaged = False
        self._enter_count = 0
        self._clutch = True
        self._filtered = None
        self._prev = None
        self._leave_grace = 0
        self._thumb_fold = 0
        self._middle_fold = 0
        self._seq = "idle"
        self.arm_hint = "Pointer: open hand to arm"

    def _scale(self, current_palm_px: float) -> float:
        return self.base_sensitivity * self.distance_ratio(current_palm_px)

    def _begin(self) -> None:
        self.engaged = True
        self._enter_count = self.confirm_frames
        self._clutch = True
        self._filtered = None
        self._prev = None
        self._leave_grace = 0
        self._thumb_fold = 0
        self._middle_fold = 0
        self._seq = "idle"
        self.arm_hint = "Pointer"

    def _click_from_folds(self, down: set[str]) -> Optional[str]:
        """Thumb fold → left click; middle fold → right click. Edge-triggered."""
        click = None
        if "thumb" in down:
            self._thumb_fold += 1
            if self._thumb_fold == CLICK_CONFIRM:
                click = "left"
        else:
            self._thumb_fold = 0

        if "middle" in down:
            self._middle_fold += 1
            if self._middle_fold == CLICK_CONFIRM and click is None:
                click = "right"
        else:
            self._middle_fold = 0
        return click

    def _arm_sequence(self, down: set[str]) -> bool:
        """Advance open → fist → pointer pose. Returns True when control should start."""
        if _is_open(down):
            self._seq = "open"
            self._enter_count = 0
            self.arm_hint = "Pointer: fist next"
            return False
        if _is_fist(down):
            if self._seq in ("open", "fist"):
                self._seq = "fist"
                self._enter_count = 0
                self.arm_hint = "Pointer: index, middle, thumb out"
            else:
                self.arm_hint = "Pointer: open hand first"
            return False
        if _is_move(down):
            if self._seq != "fist":
                self._enter_count = 0
                self.arm_hint = "Pointer: open → fist first"
                return False
            self._enter_count += 1
            self.arm_hint = "Pointer: hold pose"
            if self._enter_count >= self.confirm_frames:
                return True
            return False
        self._enter_count = 0
        if self._seq == "fist":
            self.arm_hint = "Pointer: index, middle, thumb out"
        elif self._seq == "open":
            self.arm_hint = "Pointer: fist next"
        else:
            self.arm_hint = "Pointer: open hand to arm"
        return False

    def update(self, hand_landmarks, fingers_down: list[str], frame_size) -> CursorUpdate:
        width, height = frame_size
        down = set(fingers_down)
        move = _is_move(down)
        hold = _is_pointer_hold(down)

        if not self.engaged:
            if self._arm_sequence(down):
                self._begin()
            if not self.engaged:
                return CursorUpdate(False, 0.0, 0.0)

        if not hold:
            self._leave_grace += 1
            if self._leave_grace >= LEAVE_GRACE:
                self.reset()
                return CursorUpdate(False, 0.0, 0.0)
            self._clutch = True
            return CursorUpdate(True, 0.0, 0.0)

        self._leave_grace = 0
        click = self._click_from_folds(down)

        if not move:
            self._clutch = True
            return CursorUpdate(True, 0.0, 0.0, click)

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

        return CursorUpdate(True, dx, dy, click)
