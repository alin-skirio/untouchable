"""Scrolling mode: tight right fist arms the mode, then ScrollUp or ScrollDown fires."""

from __future__ import annotations

import time

from .pose import (
    all_fingers_down,
    fingers_together,
    four_finger_width,
    four_finger_y,
    four_fingers_open,
    palm_size,
)
from .scroll_down import ScrollDown
from .scroll_up import ScrollUp

__all__ = ["Scrolling", "ScrollUp", "ScrollDown"]


class Scrolling:
    """Steps 1–3 enter this mode; ScrollUp and ScrollDown live inside it.

    Enter scrolling (right hand):
      1. thumb, index, middle, ring, and pinky all curled down
      2. pinch distances near zero for index–middle, middle–ring, ring–pinky
      3. hold that fist for 10 seconds

    Then:
      - pop the four fingers open (width grows) without dropping them → ScrollUp
      - pop the four fingers open (width grows) while dropping them → ScrollDown
    """

    def __init__(
        self,
        together: float = 0.42,
        width_increase: float = 0.28,
        confirm_seconds: float = 10.0,
        open_window: int = 12,
        cooldown_frames: int = 20,
        scroll_amount: int = 150,
    ):
        self.together = together
        self.confirm_seconds = confirm_seconds
        self.open_window = open_window
        self.cooldown_frames = cooldown_frames
        self.scroll_amount = scroll_amount
        self.scroll_up = ScrollUp(width_increase=width_increase)
        self.scroll_down = ScrollDown(width_increase=width_increase)

        self.active = False
        self.fist_started_at: float | None = None
        self.active_frames = 0
        self.fist_width = 0.0
        self.fist_y = 0.0
        self.cooldown = 0
        self.last_action: str | None = None
        self.hold_seconds = 0.0

    def reset(self) -> None:
        self.active = False
        self.fist_started_at = None
        self.active_frames = 0
        self.fist_width = 0.0
        self.fist_y = 0.0
        self.last_action = None
        self.hold_seconds = 0.0

    def update(self, hand_landmarks, fingers_down: list[str]) -> int:
        """Return +amount for ScrollUp, -amount for ScrollDown, else 0."""
        self.last_action = None
        if self.cooldown > 0:
            self.cooldown -= 1
            return 0

        lm = hand_landmarks.landmark
        palm = palm_size(lm)
        in_fist = all_fingers_down(fingers_down) and fingers_together(lm, palm, self.together)
        opened = four_fingers_open(fingers_down)
        width = four_finger_width(lm, palm)
        tip_y = four_finger_y(lm)

        if not self.active:
            if in_fist:
                now = time.monotonic()
                if self.fist_started_at is None:
                    self.fist_started_at = now
                    self.fist_width = width
                    self.fist_y = tip_y
                else:
                    self.fist_width = min(self.fist_width, width)
                    self.fist_y = tip_y
                self.hold_seconds = now - self.fist_started_at
                if self.hold_seconds >= self.confirm_seconds:
                    self.active = True
                    self.active_frames = 0
                    self.hold_seconds = self.confirm_seconds
            else:
                self.fist_started_at = None
                self.hold_seconds = 0.0
                self.fist_width = 0.0
                self.fist_y = 0.0
            return 0

        if in_fist:
            self.fist_width = min(self.fist_width, width)
            self.fist_y = tip_y
            self.active_frames = 0
            return 0

        self.active_frames += 1
        if self.active_frames > self.open_window:
            self.reset()
            return 0

        width_delta = width - self.fist_width
        travel_y = tip_y - self.fist_y

        if self.scroll_down.triggered(opened, width_delta, travel_y):
            self.reset()
            self.last_action = "down"
            self.cooldown = self.cooldown_frames
            return -self.scroll_amount

        if self.scroll_up.triggered(opened, width_delta, travel_y):
            self.reset()
            self.last_action = "up"
            self.cooldown = self.cooldown_frames
            return self.scroll_amount

        return 0
