"""Shared pose checks for scroll gestures."""

from __future__ import annotations

import math

from .landmarks import (
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_TIP,
    PINKY_TIP,
    RING_TIP,
    THUMB_TIP,
    WRIST,
    dist2,
)

TOGETHER = 0.42
LINE_DEV = 0.18
SCROLL_PX_SEC = 280.0
FAST_PX_SEC = 480.0
THUMB_TUCKED = 1.05

ALL_FINGERS = ("thumb", "index", "middle", "ring", "pinky")
ALL_TIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
ADJACENT_TIPS = (
    (THUMB_TIP, INDEX_TIP),
    (INDEX_TIP, MIDDLE_TIP),
    (MIDDLE_TIP, RING_TIP),
    (RING_TIP, PINKY_TIP),
)
FOUR_TIPS = (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
PINKY_MCP = 17


def palm_size(lm) -> float:
    return max(dist2(lm[WRIST], lm[MIDDLE_MCP]), 0.04)


def all_fingers_down(fingers_down: list[str]) -> bool:
    down = set(fingers_down)
    return all(name in down for name in ALL_FINGERS)


def four_fingers_open(fingers_down: list[str]) -> bool:
    down = set(fingers_down)
    return not any(name in down for name in ("index", "middle", "ring", "pinky"))


def adjacent_distances_zero(lm, palm: float, together: float) -> bool:
    """True when every neighboring fingertip pair is pinched (near-zero distance)."""
    return all(dist2(lm[a], lm[b]) / palm <= together for a, b in ADJACENT_TIPS)


def tips_in_line(lm, palm: float, max_dev: float) -> bool:
    """True when thumb → pinky tips lie on one straight line."""
    xs = [lm[i].x for i in ALL_TIPS]
    ys = [lm[i].y for i in ALL_TIPS]
    ax, ay = xs[0], ys[0]
    bx, by = xs[-1], ys[-1]
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length < 0.08:
        return True
    inv = 1.0 / (length * palm)
    return all(abs((x - ax) * dy - (y - ay) * dx) * inv <= max_dev for x, y in zip(xs, ys))


def fingers_in_line(lm, palm: float, together: float, max_dev: float) -> bool:
    """All five fingertips touch their neighbors and form a straight line."""
    return adjacent_distances_zero(lm, palm, together) and tips_in_line(lm, palm, max_dev)


def four_finger_width(lm, palm: float) -> float:
    return dist2(lm[INDEX_TIP], lm[PINKY_TIP]) / palm


def four_finger_y(lm) -> float:
    """Average image-y of the four fingertips (increases downward)."""
    return sum(lm[i].y for i in FOUR_TIPS) / 4.0


def palm_center(lm) -> tuple[float, float]:
    return (
        (lm[WRIST].x + lm[MIDDLE_MCP].x) / 2.0,
        (lm[WRIST].y + lm[MIDDLE_MCP].y) / 2.0,
    )


def palm_reach(lm, palm: float) -> float:
    """Mean fingertip distance from the palm center, in palm units."""
    cx, cy = palm_center(lm)
    total = 0.0
    for i in ALL_TIPS:
        total += math.hypot(lm[i].x - cx, lm[i].y - cy)
    return (total / len(ALL_TIPS)) / palm


def pinky_pointing_up(lm, fingers_down: list[str]) -> bool:
    """True when the pinky is extended and its tip sits above the palm and knuckle."""
    if "pinky" in fingers_down:
        return False
    _, cy = palm_center(lm)
    tip = lm[PINKY_TIP]
    knuckle = lm[PINKY_MCP]
    return tip.y < cy - 0.03 and tip.y < knuckle.y


def pinky_only_up(lm, fingers_down: list[str]) -> bool:
    """True when the pinky points up and index, middle, and ring are curled."""
    down = set(fingers_down)
    others_down = all(name in down for name in ("index", "middle", "ring"))
    return others_down and pinky_pointing_up(lm, fingers_down)


def index_pointing_up(lm, fingers_down: list[str]) -> bool:
    """True when the index is extended and its tip sits above the palm and knuckle."""
    if "index" in fingers_down:
        return False
    _, cy = palm_center(lm)
    tip = lm[INDEX_TIP]
    knuckle = lm[INDEX_MCP]
    return tip.y < cy - 0.03 and tip.y < knuckle.y


def index_only_up(lm, fingers_down: list[str]) -> bool:
    """True when the index points up and middle, ring, and pinky are curled."""
    down = set(fingers_down)
    others_down = all(name in down for name in ("middle", "ring", "pinky"))
    return others_down and index_pointing_up(lm, fingers_down)


def thumb_tucked(lm, fingers_down: list[str]) -> bool:
    """True when the thumb is folded into the fist, not standing off the palm."""
    if "thumb" not in fingers_down:
        return False
    palm = palm_size(lm)
    cx, cy = palm_center(lm)
    reach = math.hypot(lm[THUMB_TIP].x - cx, lm[THUMB_TIP].y - cy) / palm
    return reach <= THUMB_TUCKED


def pinky_scroll_rate(lm, fingers_down: list[str]) -> float:
    """Pixels per second for pinky scroll, or 0 if the pose is not active."""
    if not pinky_only_up(lm, fingers_down):
        return 0.0
    if thumb_tucked(lm, fingers_down):
        return FAST_PX_SEC
    return SCROLL_PX_SEC


def index_scroll_rate(lm, fingers_down: list[str]) -> float:
    """Pixels per second for index-pointer scroll, or 0 if the pose is not active."""
    if not index_only_up(lm, fingers_down):
        return 0.0
    if thumb_tucked(lm, fingers_down):
        return FAST_PX_SEC
    return SCROLL_PX_SEC


def pointing_up(lm) -> bool:
    """True when the fingertips sit above the palm (image y grows downward)."""
    _, cy = palm_center(lm)
    tip_y = sum(lm[i].y for i in ALL_TIPS) / len(ALL_TIPS)
    return tip_y < cy - 0.03


def all_fingers_pointing_up(lm, fingers_down: list[str]) -> bool:
    """True when every fingertip sits above the palm and none of the four mains are curled."""
    if any(name in fingers_down for name in ("index", "middle", "ring", "pinky")):
        return False
    _, cy = palm_center(lm)
    return all(lm[i].y < cy - 0.03 for i in ALL_TIPS)
