"""Hand gesture helpers."""

from __future__ import annotations

import math
from collections import deque

THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
WRIST = 0
MIDDLE_MCP = 9


def _dist2(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def triple_pinch_span(hand_landmarks) -> float:
    """Largest pairwise distance among thumb, index, and middle tips, in palm units."""
    lm = hand_landmarks.landmark
    palm = _dist2(lm[WRIST], lm[MIDDLE_MCP])
    if palm < 0.04:
        palm = 0.04
    thumb, index, middle = lm[THUMB_TIP], lm[INDEX_TIP], lm[MIDDLE_TIP]
    span = max(
        _dist2(thumb, index),
        _dist2(thumb, middle),
        _dist2(index, middle),
    )
    return span / palm


class AppSwitcher:
    """Tracks holding thumb+middle to keep ⌘ active while tapping index to press Tab.

    Ring and pinky fingers are completely ignored during gesture recognition.
    """

    def __init__(
        self,
        pinch: float = 0.34,
        release: float = 0.50,
        flap: float = 0.45,
        confirm_frames: int = 3,
        min_wrist_dist: float = 1.05,
        min_hand_width: float = 0.30,  # NEW: Minimum width to ensure hand isn't sideways
    ):
        self.pinch = pinch
        self.release = release
        self.flap = flap
        self.confirm_frames = confirm_frames
        self.min_wrist_dist = min_wrist_dist
        self.min_hand_width = min_hand_width
        self.pinch_count = 0
        self.active = False
        self.index_up = False

    def reset(self) -> None:
        self.active = False
        self.index_up = False
        self.pinch_count = 0

    def update(self, hand_landmarks) -> tuple[bool, bool, bool]:
        lm = hand_landmarks.landmark
        palm = max(_dist2(lm[WRIST], lm[MIDDLE_MCP]), 0.04)

        # --- NEW: Check if hand is facing the camera ---
        # Measure 2D distance between Index MCP (knuckle 5) and Pinky MCP (knuckle 17)
        INDEX_MCP, PINKY_MCP = 5, 17
        hand_width = _dist2(lm[INDEX_MCP], lm[PINKY_MCP]) / palm
        is_sideways = hand_width < self.min_hand_width

        # Distances between key active fingers (thumb, index, middle)
        tm_dist = _dist2(lm[THUMB_TIP], lm[MIDDLE_TIP]) / palm
        index_dist = min(
            _dist2(lm[THUMB_TIP], lm[INDEX_TIP]),
            _dist2(lm[MIDDLE_TIP], lm[INDEX_TIP]),
        ) / palm

        # Distance checks relative to wrist using ONLY thumb and middle finger
        thumb_wrist = _dist2(lm[THUMB_TIP], lm[WRIST]) / palm
        middle_wrist = _dist2(lm[MIDDLE_TIP], lm[WRIST]) / palm

        # In a closed fist, the active fingertips collapse tightly toward the wrist
        is_fist = (thumb_wrist < self.min_wrist_dist) or (middle_wrist < 0.90)

        started, tapped, ended = False, False, False

        if not self.active:
            # We now require the hand to NOT be sideways to initiate the pinch
            if not is_fist and not is_sideways and max(tm_dist, index_dist) <= self.pinch:
                self.pinch_count += 1
                if self.pinch_count >= self.confirm_frames:
                    self.active = True
                    self.index_up = False
                    started = True
            else:
                self.pinch_count = 0
        else:
            self.pinch_count = 0
            if tm_dist >= self.release or is_fist:
                self.active = False
                self.index_up = False
                ended = True
            else:
                if not self.index_up and index_dist > self.flap:
                    self.index_up = True
                elif self.index_up and index_dist <= self.pinch:
                    self.index_up = False
                    tapped = True

        return started, tapped, ended



class SwipeScroller:
    """Triggers a scroll when the 4 main fingers quickly pop from a curled fist to fully extended."""
    
    def __init__(self, history_size: int = 10, scroll_amount: int = 150):
        # 10 frames equals roughly a third of a second for the motion to happen
        self.history = deque(maxlen=history_size)
        self.scroll_amount = scroll_amount
        self.cooldown = 0

    def update(self, hand_landmarks, fingers_down: list[str]) -> int:
        if self.cooldown > 0:
            self.cooldown -= 1
            return 0

        # 1. Count how many of the 4 main fingers are currently curled (down)
        main_fingers = {"index", "middle", "ring", "pinky"}
        down_count = len(main_fingers.intersection(set(fingers_down)))
        self.history.append(down_count)

        # 2. Check if the hand is fully open right now (0 main fingers down)
        if down_count == 0:
            # 3. Look back in recent history to see if it was a fist (>= 3 fingers down)
            if len(self.history) >= 3 and max(self.history) >= 3:
                self.history.clear()
                self.cooldown = 20  # Pause to prevent multiple scrolls from one action
                return self.scroll_amount

        return 0



