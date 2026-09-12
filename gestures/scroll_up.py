"""ScrollUp: pinky-up hold, then fist, then page follows palm-reach."""

from __future__ import annotations

import time

from .pose import (
    TOGETHER,
    adjacent_distances_zero,
    all_fingers_down,
    all_fingers_pointing_up,
    palm_reach,
    palm_size,
    pinky_pointing_up,
)

IDLE = "idle"
WAIT_FIST = "wait_fist"
RAISING = "raising"


class ScrollUp:
    """1. Pinky pointing up for 3s (fingers touching) initiates.
    2. Make a fist.
    3. Raise: page scrolls up as palm-reach grows.
    4. Stop when reach is max and all fingers point up.
    5. Fist again to repeat from 2.

    If the fingers separate at any time, the whole gesture terminates
    and must start again from the 3-second pinky hold.
    """

    def __init__(
        self,
        fist_reach: float = 1.20,
        expanded_reach: float = 1.90,
        page_lines: float = 48.0,
        together: float = TOGETHER,
        initiate_seconds: float = 3.0,
    ):
        self.fist_reach = fist_reach
        self.expanded_reach = expanded_reach
        self.page_lines = page_lines
        self.together = together
        self.initiate_seconds = initiate_seconds
        self.phase = IDLE
        self.pinky_started_at: float | None = None
        self.hold_seconds = 0.0
        self.last_reach = 0.0
        self.accrued = 0.0

    @property
    def active(self) -> bool:
        return self.phase == RAISING

    def reset(self) -> None:
        self.phase = IDLE
        self.pinky_started_at = None
        self.hold_seconds = 0.0
        self.last_reach = 0.0
        self.accrued = 0.0

    def update(self, hand_landmarks, fingers_down: list[str]) -> int:
        """Return how many lines to scroll up this frame (0 if none)."""
        lm = hand_landmarks.landmark
        palm = palm_size(lm)
        touching = adjacent_distances_zero(lm, palm, self.together)
        reach = palm_reach(lm, palm)
        in_fist = all_fingers_down(fingers_down) and touching and reach <= self.fist_reach
        pinky_up = pinky_pointing_up(lm, fingers_down)
        pointing_up = all_fingers_pointing_up(lm, fingers_down)

        if self.phase == IDLE:
            if touching and pinky_up:
                now = time.monotonic()
                if self.pinky_started_at is None:
                    self.pinky_started_at = now
                self.hold_seconds = now - self.pinky_started_at
                if self.hold_seconds >= self.initiate_seconds:
                    self.phase = WAIT_FIST
                    self.hold_seconds = self.initiate_seconds
                    self.pinky_started_at = None
            else:
                self.pinky_started_at = None
                self.hold_seconds = 0.0
            return 0

        if not touching:
            self.reset()
            return 0

        if self.phase == WAIT_FIST:
            if in_fist:
                self.phase = RAISING
                self.last_reach = min(reach, self.fist_reach)
                self.accrued = 0.0
            return 0

        # RAISING
        if in_fist:
            self.last_reach = min(reach, self.fist_reach)
            self.accrued = 0.0
            return 0

        span = max(self.expanded_reach - self.fist_reach, 1e-6)
        if reach > self.last_reach:
            usable = min(reach, self.expanded_reach) - self.last_reach
            if usable > 0:
                self.accrued += (usable / span) * self.page_lines
            self.last_reach = min(reach, self.expanded_reach)

        lines = int(self.accrued)
        self.accrued -= lines

        if reach >= self.expanded_reach and pointing_up:
            leftover = self.accrued
            self.accrued = 0.0
            self.phase = WAIT_FIST
            if leftover >= 0.5:
                lines += 1

        return max(0, lines)
