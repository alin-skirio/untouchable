"""T pose gesture: two flat hands held perpendicular, like a sports timeout signal."""

from __future__ import annotations

import math

from .landmarks import (
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_TIP,
    PINKY_MCP,
    PINKY_TIP,
    RING_TIP,
    THUMB_TIP,
    WRIST,
    palm_size_px,
)
from .pose import four_fingers_open

KEY_POINTS = (
    WRIST,
    INDEX_MCP,
    MIDDLE_MCP,
    PINKY_MCP,
    THUMB_TIP,
    INDEX_TIP,
    MIDDLE_TIP,
    RING_TIP,
    PINKY_TIP,
)

PERPENDICULAR_TOLERANCE_DEG = 30.0
MAX_GAP = 1.1  # palm lengths between the closest key points of the two hands
CONFIRM_FRAMES = 4
COOLDOWN_FRAMES = 90


def _px(landmark, width: float, height: float) -> tuple[float, float]:
    return landmark.x * width, landmark.y * height


def _hand_axis(lm, width: float, height: float) -> tuple[float, float]:
    """Wrist → middle fingertip in pixels, so the angle is not aspect-distorted."""
    wx, wy = _px(lm[WRIST], width, height)
    tx, ty = _px(lm[MIDDLE_TIP], width, height)
    return tx - wx, ty - wy


def _line_angle_between(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Undirected angle between two vectors, 0-90 degrees (90 = perpendicular)."""
    ax, ay = a
    bx, by = b
    na = math.hypot(ax, ay)
    nb = math.hypot(bx, by)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    cos = abs((ax * bx + ay * by) / (na * nb))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _closest_gap_px(lm_a, lm_b, width: float, height: float) -> float:
    best = float("inf")
    for i in KEY_POINTS:
        ax, ay = _px(lm_a[i], width, height)
        for j in KEY_POINTS:
            bx, by = _px(lm_b[j], width, height)
            gap = math.hypot(ax - bx, ay - by)
            if gap < best:
                best = gap
    return best


def _bar_meets_stem_tip(stem, bar, width: float, height: float) -> bool:
    """True when the bar hand sits against the stem's fingertips, so it reads as T not X."""
    bx, by = _px(bar[MIDDLE_MCP], width, height)
    tx, ty = _px(stem[MIDDLE_TIP], width, height)
    wx, wy = _px(stem[WRIST], width, height)
    return math.hypot(bx - tx, by - ty) < math.hypot(bx - wx, by - wy)


class TPose:
    """Fires once when both flat hands form a T, then waits for release."""

    def __init__(
        self,
        confirm_frames: int = CONFIRM_FRAMES,
        cooldown_frames: int = COOLDOWN_FRAMES,
        tolerance_deg: float = PERPENDICULAR_TOLERANCE_DEG,
        max_gap: float = MAX_GAP,
    ):
        self.confirm_frames = confirm_frames
        self.cooldown_frames = cooldown_frames
        self.tolerance_deg = tolerance_deg
        self.max_gap = max_gap
        self.count = 0
        self.cooldown = 0
        self.latched = False
        self.holding = False

    def reset(self) -> None:
        self.count = 0
        self.latched = False
        self.holding = False

    def update(self, hands, frame_size) -> bool:
        """hands: list of (hand_landmarks, fingers_down). True on the frame the T forms."""
        if self.cooldown > 0:
            self.cooldown -= 1

        matched = self._matches(hands, frame_size)
        self.holding = matched

        if not matched:
            self.count = 0
            self.latched = False
            return False

        # Latched until the hands leave the pose, so holding a T fires once.
        if self.latched or self.cooldown > 0:
            return False

        self.count += 1
        if self.count >= self.confirm_frames:
            self.count = 0
            self.latched = True
            self.cooldown = self.cooldown_frames
            return True
        return False

    def _matches(self, hands, frame_size) -> bool:
        if not hands or len(hands) < 2:
            return False
        width, height = frame_size
        flat = [landmarks for landmarks, down in hands if four_fingers_open(down)]
        if len(flat) < 2:
            return False
        for i in range(len(flat)):
            for j in range(i + 1, len(flat)):
                if self._pair_makes_t(flat[i], flat[j], width, height):
                    return True
        return False

    def _pair_makes_t(self, a, b, width: float, height: float) -> bool:
        lm_a, lm_b = a.landmark, b.landmark

        angle = _line_angle_between(
            _hand_axis(lm_a, width, height), _hand_axis(lm_b, width, height)
        )
        if abs(angle - 90.0) > self.tolerance_deg:
            return False

        palm = (
            palm_size_px(a, width, height) + palm_size_px(b, width, height)
        ) / 2.0
        if palm < 1e-6:
            return False
        if _closest_gap_px(lm_a, lm_b, width, height) / palm > self.max_gap:
            return False

        return _bar_meets_stem_tip(lm_a, lm_b, width, height) or _bar_meets_stem_tip(
            lm_b, lm_a, width, height
        )
