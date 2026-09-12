"""Shared MediaPipe landmark indices and distance helpers."""

from __future__ import annotations

import math

THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17


def dist2(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def palm_size_px(hand_landmarks, width: float, height: float) -> float:
    """Wrist-to-middle-MCP length in pixels (distance proxy)."""
    lm = hand_landmarks.landmark
    return math.hypot(
        (lm[WRIST].x - lm[MIDDLE_MCP].x) * width,
        (lm[WRIST].y - lm[MIDDLE_MCP].y) * height,
    )


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
