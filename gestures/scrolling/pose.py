"""Shared pose checks for the Scrolling mode."""

from __future__ import annotations

from ..landmarks import (
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_TIP,
    PINKY_TIP,
    RING_TIP,
    WRIST,
    dist2,
)

ALL_FINGERS = ("thumb", "index", "middle", "ring", "pinky")
FOUR_TIPS = (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)


def palm_size(lm) -> float:
    return max(dist2(lm[WRIST], lm[MIDDLE_MCP]), 0.04)


def all_fingers_down(fingers_down: list[str]) -> bool:
    down = set(fingers_down)
    return all(name in down for name in ALL_FINGERS)


def four_fingers_open(fingers_down: list[str]) -> bool:
    down = set(fingers_down)
    return not any(name in down for name in ("index", "middle", "ring", "pinky"))


def fingers_together(lm, palm: float, together: float) -> bool:
    """True when index–middle, middle–ring, and ring–pinky pinches are near zero."""
    index_middle = dist2(lm[INDEX_TIP], lm[MIDDLE_TIP]) / palm
    middle_ring = dist2(lm[MIDDLE_TIP], lm[RING_TIP]) / palm
    ring_pinky = dist2(lm[RING_TIP], lm[PINKY_TIP]) / palm
    return max(index_middle, middle_ring, ring_pinky) <= together


def four_finger_width(lm, palm: float) -> float:
    return dist2(lm[INDEX_TIP], lm[PINKY_TIP]) / palm


def four_finger_y(lm) -> float:
    """Average image-y of the four fingertips (increases downward)."""
    return sum(lm[i].y for i in FOUR_TIPS) / 4.0
