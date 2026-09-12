"""Shared MediaPipe landmark indices and distance helpers."""

from __future__ import annotations

import math

THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
WRIST = 0
MIDDLE_MCP = 9


def dist2(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


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
