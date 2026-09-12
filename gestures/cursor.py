"""Right-hand pointer: open → fist → gun pose (index+middle+thumb out).

While engaged, folding the thumb left-clicks and folding the middle finger right-clicks.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2

from .landmarks import (
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_TIP,
    THUMB_TIP,
    WRIST,
    dist2,
    index_track_px,
    palm_size_px,
)

DEFAULT_BASE_SENSITIVITY = 1.6
SENSITIVITY_MIN = 0.25
SENSITIVITY_MAX = 4.0
DEADZONE_PX = 1.8
DEFAULT_PALM_PX = 220.0
# Palm-width of fingertip travel → screen pixels at 1.0 gain. Desk-range, not a twitch.
COVER_PX = 620.0
SCALE_MIN = 1.8
SCALE_MAX = 5.4
CONFIRM_FRAMES = 8
ARM_STEP_FRAMES = 4
SEQ_GRACE = 4
MIN_PALM_PX = 1.0
EXIT_FIST_FRAMES = 4
CLICK_CONFIRM = 3
CLICK_SETTLE = 3
PALM_SMOOTH = 0.18
GUN_THUMB_CLEAR = 0.55
SWITCHER_COOLDOWN = 20
MAIN_FINGERS = frozenset({"index", "middle", "ring", "pinky"})

# One Euro: low cutoff when still, rises with speed so flicks are not laggy.
EURO_MINCUTOFF = 1.2
EURO_BETA = 0.04
EURO_DCUTOFF = 1.0

REFERENCE_DIR = Path(__file__).resolve().parent.parent
REFERENCE_PNG = REFERENCE_DIR / "hand_reference.png"
REFERENCE_JSON = REFERENCE_DIR / "hand_reference.json"


@dataclass
class CursorUpdate:
    engaged: bool
    dx: float
    dy: float
    click: Optional[str] = None


class _OneEuro:
    """Speed-aware low-pass. Slow motion stays quiet; fast motion stays tight."""

    def __init__(
        self,
        mincutoff: float = EURO_MINCUTOFF,
        beta: float = EURO_BETA,
        dcutoff: float = EURO_DCUTOFF,
    ):
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self._x: Optional[float] = None
        self._dx = 0.0

    def reset(self) -> None:
        self._x = None
        self._dx = 0.0

    def filter(self, x: float, dt: float) -> float:
        if self._x is None or dt <= 0:
            self._x = x
            self._dx = 0.0
            return x
        dx = (x - self._x) / dt
        self._dx = _lowpass(dx, self._dx, _alpha(self.dcutoff, dt))
        cutoff = self.mincutoff + self.beta * abs(self._dx)
        self._x = _lowpass(x, self._x, _alpha(cutoff, dt))
        return self._x


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


def _lowpass(value: float, prev: float, alpha: float) -> float:
    return alpha * value + (1.0 - alpha) * prev


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
    """Four mains extended. Thumb is ignored — MediaPipe often marks it folded."""
    return not (MAIN_FINGERS & down)


def _is_fist(down: set[str]) -> bool:
    """All four mains curled — a real fist, not a pinch with a finger still out."""
    return len(MAIN_FINGERS & down) >= 4


def _is_exit_fist(down: set[str]) -> bool:
    """Index plus two other mains curled. Ends pointer without matching a click."""
    mains = MAIN_FINGERS & down
    return "index" in mains and len(mains) >= 3


def _is_gun(down: set[str]) -> bool:
    """Index, middle, and thumb out; ring and pinky curled."""
    return (
        "thumb" not in down
        and "index" not in down
        and "middle" not in down
        and "ring" in down
        and "pinky" in down
    )


def _gun_spread(hand_landmarks) -> bool:
    """Thumb stands off the index/middle cluster. A pinch is not a gun."""
    lm = hand_landmarks.landmark
    palm = max(dist2(lm[WRIST], lm[MIDDLE_MCP]), 0.04)
    thumb_index = dist2(lm[THUMB_TIP], lm[INDEX_TIP]) / palm
    thumb_middle = dist2(lm[THUMB_TIP], lm[MIDDLE_TIP]) / palm
    return thumb_index >= GUN_THUMB_CLEAR and thumb_middle >= GUN_THUMB_CLEAR


def _is_pointer_hold(down: set[str]) -> bool:
    """Stay in pointer while index tracks; thumb/middle may fold for clicks."""
    return "index" not in down and "ring" in down and "pinky" in down


def _ballistic(dx: float, dy: float) -> tuple[float, float]:
    """Keep desk-range motion linear; only a little extra on a real flick."""
    speed = math.hypot(dx, dy)
    gain = 0.92 + 0.16 * (1.0 - math.exp(-speed / 22.0))
    return dx * gain, dy * gain


class PointerCursor:
    """Relative macOS pointer from the index, scaled by desk sensitivity × palm ratio."""

    def __init__(
        self,
        confirm_frames: int = CONFIRM_FRAMES,
        base_sensitivity: Optional[float] = None,
        smooth_alpha: float = 0.35,
        deadzone_px: float = DEADZONE_PX,
    ):
        palm, stored_gain = load_reference()
        self.confirm_frames = confirm_frames
        self.base_sensitivity = (
            stored_gain if base_sensitivity is None else _clamp_sensitivity(base_sensitivity)
        )
        self.smooth_alpha = smooth_alpha
        self.deadzone_px = deadzone_px
        self.ref_palm_px: Optional[float] = palm
        self.engaged = False
        self.arm_hint = "Pointer: open hand to arm"
        self._seq = "idle"
        self._enter_count = 0
        self._open_count = 0
        self._fist_count = 0
        self._seq_miss = 0
        self._clutch = True
        self._prev: Optional[tuple[float, float]] = None
        self._palm_px: Optional[float] = None
        self._leave_grace = 0
        self._thumb_fold = 0
        self._middle_fold = 0
        self._settle = 0
        self._fist_exit = 0
        self._suppress = 0
        self._last_t: Optional[float] = None
        self._fx = _OneEuro()
        self._fy = _OneEuro()

    @property
    def has_reference(self) -> bool:
        return self.ref_palm_px is not None and self.ref_palm_px > 0

    def distance_ratio(self, current_palm_px: float) -> float:
        """Farther hand (smaller palm) → larger ratio. Always live, even without ⌘-."""
        if current_palm_px <= 0:
            return 1.0
        ref = self.ref_palm_px if self.has_reference else DEFAULT_PALM_PX
        ratio = ref / max(current_palm_px, MIN_PALM_PX)
        return min(2.1, max(0.55, ratio))

    def _scale(self, current_palm_px: float) -> float:
        """Image pixels → screen pixels. Live palm keeps desk-range motion at any depth."""
        palm = max(current_palm_px, MIN_PALM_PX)
        raw = self.base_sensitivity * COVER_PX / palm
        return min(SCALE_MAX, max(SCALE_MIN, raw))

    def set_base_sensitivity(self, value: float, persist: bool = True) -> None:
        self.base_sensitivity = _clamp_sensitivity(value)
        if persist:
            persist_sensitivity(self.base_sensitivity)

    def set_reference(self, palm_px: float, frame=None, frame_size=None) -> bool:
        if palm_px <= 0:
            return False
        self.ref_palm_px = palm_px
        self._clutch = True
        self._prev = None
        self._palm_px = None
        self._fx.reset()
        self._fy.reset()
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
        self._begin()

    def reset(self) -> None:
        self.engaged = False
        self._enter_count = 0
        self._open_count = 0
        self._fist_count = 0
        self._seq_miss = 0
        self._clutch = True
        self._prev = None
        self._palm_px = None
        self._leave_grace = 0
        self._thumb_fold = 0
        self._middle_fold = 0
        self._settle = 0
        self._fist_exit = 0
        self._suppress = 0
        self._last_t = None
        self._fx.reset()
        self._fy.reset()
        self._seq = "idle"
        self.arm_hint = "Pointer: open hand to arm"

    def _begin(self) -> None:
        self.engaged = True
        self._enter_count = self.confirm_frames
        self._open_count = 0
        self._fist_count = 0
        self._seq_miss = 0
        self._clutch = True
        self._prev = None
        self._palm_px = None
        self._leave_grace = 0
        self._thumb_fold = 0
        self._middle_fold = 0
        self._settle = 0
        self._fist_exit = 0
        self._last_t = None
        self._fx.reset()
        self._fy.reset()
        self._seq = "idle"
        self.arm_hint = "Pointer · fist to exit"

    def cancel_arm(self) -> None:
        """Drop a half-finished arm sequence without touching an active pointer."""
        if not self.engaged:
            self._drop_sequence()

    def suppress_arm(self, frames: int = SWITCHER_COOLDOWN) -> None:
        """Ignore arming after app switcher so a pinch-flap cannot become the gun."""
        self.cancel_arm()
        if not self.engaged:
            self._suppress = max(self._suppress, frames)

    def _dt(self) -> float:
        now = time.monotonic()
        if self._last_t is None:
            self._last_t = now
            return 1.0 / 30.0
        dt = now - self._last_t
        self._last_t = now
        return min(0.08, max(1.0 / 90.0, dt))

    def _smooth_palm(self, palm: float) -> float:
        if self._palm_px is None or palm <= 0:
            self._palm_px = palm
            return palm
        self._palm_px = PALM_SMOOTH * palm + (1.0 - PALM_SMOOTH) * self._palm_px
        return self._palm_px

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

    def _drop_sequence(self) -> None:
        self._seq = "idle"
        self._enter_count = 0
        self._open_count = 0
        self._fist_count = 0
        self._seq_miss = 0
        self.arm_hint = "Pointer: open hand to arm"

    def _miss(self, hint: str) -> None:
        self._seq_miss += 1
        if self._seq_miss >= SEQ_GRACE:
            self._drop_sequence()
            return
        self.arm_hint = hint

    def _arm_sequence(self, down: set[str], hand_landmarks) -> bool:
        """Advance open → fist → spread gun. One bad frame does not restart it."""
        if _is_open(down):
            self._seq_miss = 0
            self._fist_count = 0
            self._enter_count = 0
            self._open_count += 1
            if self._open_count >= ARM_STEP_FRAMES:
                self._seq = "open"
                self.arm_hint = "Pointer: fist next"
            else:
                self.arm_hint = "Pointer: open hand to arm"
            return False

        if _is_fist(down):
            if self._seq in ("open", "fist"):
                self._seq_miss = 0
                self._enter_count = 0
                self._fist_count += 1
                if self._fist_count >= ARM_STEP_FRAMES:
                    self._seq = "fist"
                    self.arm_hint = "Pointer: gun pose — index, middle, thumb"
                else:
                    self.arm_hint = "Pointer: fist next"
            else:
                self._miss("Pointer: open hand first")
            return False

        if _is_gun(down) and _gun_spread(hand_landmarks):
            if self._seq != "fist":
                self._miss("Pointer: open → fist first")
                return False
            self._seq_miss = 0
            self._enter_count += 1
            self.arm_hint = "Pointer: hold gun pose"
            return self._enter_count >= self.confirm_frames

        if self._seq == "idle":
            self.arm_hint = "Pointer: open hand to arm"
            return False
        if self._seq == "fist":
            self._miss("Pointer: gun pose — index, middle, thumb")
        elif self._seq == "open":
            self._miss("Pointer: fist next")
        else:
            self._miss("Pointer: open hand to arm")
        return False

    def update(self, hand_landmarks, fingers_down: list[str], frame_size) -> CursorUpdate:
        width, height = frame_size
        down = set(fingers_down)
        hold = _is_pointer_hold(down)

        if not self.engaged:
            if self._suppress > 0:
                self._suppress -= 1
                return CursorUpdate(False, 0.0, 0.0)
            if self._arm_sequence(down, hand_landmarks):
                self._begin()
            if not self.engaged:
                return CursorUpdate(False, 0.0, 0.0)

        if _is_exit_fist(down):
            self._fist_exit += 1
            if self._fist_exit >= EXIT_FIST_FRAMES:
                self.reset()
                return CursorUpdate(False, 0.0, 0.0)
            self._clutch = True
            return CursorUpdate(True, 0.0, 0.0)
        self._fist_exit = 0

        if not hold:
            self._clutch = True
            return CursorUpdate(True, 0.0, 0.0)

        click = self._click_from_folds(down)
        if click:
            self._settle = CLICK_SETTLE
            self._clutch = True

        raw = index_track_px(hand_landmarks, width, height)
        palm = self._smooth_palm(palm_size_px(hand_landmarks, width, height))
        dt = self._dt()
        filtered = (self._fx.filter(raw[0], dt), self._fy.filter(raw[1], dt))

        if self._clutch or self._prev is None or self._settle > 0:
            self._prev = filtered
            self._clutch = False
            if self._settle > 0:
                self._settle -= 1
            return CursorUpdate(True, 0.0, 0.0, click)

        dximg = filtered[0] - self._prev[0]
        dyimg = filtered[1] - self._prev[1]
        self._prev = filtered
        scale = self._scale(palm)
        dx, dy = _ballistic(dximg * scale, dyimg * scale)
        if math.hypot(dx, dy) < self.deadzone_px:
            dx = dy = 0.0

        return CursorUpdate(True, dx, dy, click)
