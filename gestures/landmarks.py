"""Shared MediaPipe landmark indices and distance helpers."""

from __future__ import annotations

import math

THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
WRIST = 0
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
MIDDLE_MCP = 9
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
PINKY_MCP = 17

PALM_MIN = 0.04


def dist2(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def palm_size(lm) -> float:
    return max(dist2(lm[WRIST], lm[MIDDLE_MCP]), PALM_MIN)


def palm_size_px(hand_landmarks, width: float, height: float) -> float:
    """Wrist-to-middle-MCP length in pixels (distance proxy)."""
    lm = hand_landmarks.landmark
    return math.hypot(
        (lm[WRIST].x - lm[MIDDLE_MCP].x) * width,
        (lm[WRIST].y - lm[MIDDLE_MCP].y) * height,
    )


def index_tip_px(hand_landmarks, width: float, height: float) -> tuple[float, float]:
    tip = hand_landmarks.landmark[INDEX_TIP]
    return tip.x * width, tip.y * height


def _point_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    if length2 < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def thumb_to_finger_side(hand_landmarks, mcp: int, pip: int, dip: int, tip: int) -> float:
    """Thumb-tip distance to a finger's MCP→TIP shaft, in palm units."""
    lm = hand_landmarks.landmark
    thumb = lm[THUMB_TIP]
    palm = palm_size(lm)
    px, py = thumb.x, thumb.y
    dist = min(
        _point_to_segment(px, py, lm[a].x, lm[a].y, lm[b].x, lm[b].y)
        for a, b in ((mcp, pip), (pip, dip), (dip, tip))
    )
    return dist / palm


def thumb_to_ring_side(hand_landmarks) -> float:
    """Thumb-tip distance to the curled ring-finger shaft, in palm units."""
    return thumb_to_finger_side(hand_landmarks, RING_MCP, RING_PIP, RING_DIP, RING_TIP)


def thumb_to_index_side(hand_landmarks) -> float:
    """Thumb-tip distance to the pointing index-finger shaft, in palm units."""
    return thumb_to_finger_side(hand_landmarks, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP)


def triple_pinch_span(hand_landmarks) -> float:
    """Largest pairwise distance among thumb, index, and middle tips, in palm units."""
    lm = hand_landmarks.landmark
    palm = dist2(lm[WRIST], lm[MIDDLE_MCP])
    if palm < 0.04:
        palm = 0.04
    thumb, index, middle = lm[THUMB_TIP], lm[INDEX_TIP], lm[MIDDLE_TIP]
    span = max(
        dist2(thumb, index),
        dist2(thumb, middle),
        dist2(index, middle),
    )
    return span / palm
