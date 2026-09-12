"""Scrolling mode: tight right fist arms the mode, then ScrollUp or ScrollDown fires."""

from __future__ import annotations

import time

from .pose import (
    all_fingers_down,
    fingers_in_line,
    four_finger_width,
    four_finger_y,
    four_fingers_open,
    palm_reach,
    palm_size,
    pointing_up,
)
from .scroll_down import ScrollDown
from .scroll_up import ScrollUp

__all__ = ["Scrolling", "ScrollUp", "ScrollDown"]


class Scrolling:
    """Steps 1–3 enter this mode; ScrollUp and ScrollDown live inside it.

    Every scrolling action requires all five fingertips in a straight line
    with neighboring distances near zero (thumb–index–middle–ring–pinky).

    Enter scrolling (right hand):
      1. thumb, index, middle, ring, and pinky all curled down
      2. those tips stay in a line with near-zero neighbor distances
      3. hold that pose for 10 seconds

    Then, still with the fingers in that line:
      - fist → fingertips at max palm reach, pointing up → ScrollUp (one page)
      - repeat that fist → expanded cycle to page up again
      - pop the four fingers open while dropping them → ScrollDown
    """

    def __init__(
        self,
        together: float = 0.42,
        line_dev: float = 0.18,
        width_increase: float = 0.28,
        confirm_seconds: float = 10.0,
        cooldown_frames: int = 20,
        scroll_amount: int = 150,
    ):
        self.together = together
        self.line_dev = line_dev
        self.confirm_seconds = confirm_seconds
        self.cooldown_frames = cooldown_frames
        self.scroll_amount = scroll_amount
        self.scroll_up = ScrollUp()
        self.scroll_down = ScrollDown(width_increase=width_increase)

        self.active = False
        self.fist_started_at: float | None = None
        self.fist_width = 0.0
        self.fist_y = 0.0
        self.cooldown = 0
        self.last_action: str | None = None
        self.hold_seconds = 0.0

    def reset(self) -> None:
        self.active = False
        self.fist_started_at = None
        self.fist_width = 0.0
        self.fist_y = 0.0
        self.last_action = None
        self.hold_seconds = 0.0
        self.scroll_up.reset()

    def update(self, hand_landmarks, fingers_down: list[str]) -> str | None:
        """Return 'up', 'down', or None."""
        self.last_action = None
        if self.cooldown > 0:
            self.cooldown -= 1
            return None

        lm = hand_landmarks.landmark
        palm = palm_size(lm)
        in_line = fingers_in_line(lm, palm, self.together, self.line_dev)
        in_fist = all_fingers_down(fingers_down) and in_line
        opened = four_fingers_open(fingers_down)
        width = four_finger_width(lm, palm)
        tip_y = four_finger_y(lm)
        reach = palm_reach(lm, palm)
        raised = pointing_up(lm)

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
                    self.hold_seconds = self.confirm_seconds
                    self.scroll_up.from_fist = True
            else:
                self.fist_started_at = None
                self.hold_seconds = 0.0
                self.fist_width = 0.0
                self.fist_y = 0.0
            return None

        if in_fist:
            self.fist_width = min(self.fist_width, width)
            self.fist_y = tip_y

        if self.scroll_up.update(
            in_line=in_line,
            in_fist=in_fist,
            reach=reach,
            pointing_up=raised,
        ):
            self.last_action = "up"
            self.cooldown = self.cooldown_frames
            return "up"

        if not in_fist:
            width_delta = width - self.fist_width
            travel_y = tip_y - self.fist_y
            if self.scroll_down.triggered(opened, width_delta, travel_y, in_line):
                self.last_action = "down"
                self.cooldown = self.cooldown_frames
                return "down"

        return None
