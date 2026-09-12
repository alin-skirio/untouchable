"""ScrollDown gesture: open and drop a lined-up hand to scroll down."""

from __future__ import annotations

from .pose import (
    LINE_DEV,
    TOGETHER,
    all_fingers_down,
    fingers_in_line,
    four_finger_width,
    four_finger_y,
    four_fingers_open,
    palm_size,
)


class ScrollDown:
    def __init__(
        self,
        width_increase: float = 0.28,
        min_down: float = 0.035,
        together: float = TOGETHER,
        line_dev: float = LINE_DEV,
        cooldown_frames: int = 20,
        scroll_amount: int = 150,
    ):
        self.width_increase = width_increase
        self.min_down = min_down
        self.together = together
        self.line_dev = line_dev
        self.cooldown_frames = cooldown_frames
        self.scroll_amount = scroll_amount
        self.fist_width: float | None = None
        self.fist_y: float | None = None
        self.cooldown = 0

    def reset(self) -> None:
        self.fist_width = None
        self.fist_y = None
        self.cooldown = 0

    def update(self, hand_landmarks, fingers_down: list[str]) -> int:
        """Return how many lines to scroll down this frame (0 if none)."""
        if self.cooldown > 0:
            self.cooldown -= 1
            return 0

        lm = hand_landmarks.landmark
        palm = palm_size(lm)
        in_line = fingers_in_line(lm, palm, self.together, self.line_dev)
        in_fist = all_fingers_down(fingers_down) and in_line
        opened = four_fingers_open(fingers_down)
        width = four_finger_width(lm, palm)
        tip_y = four_finger_y(lm)

        if in_fist:
            self.fist_width = width if self.fist_width is None else min(self.fist_width, width)
            self.fist_y = tip_y
            return 0

        if self.fist_width is None or self.fist_y is None:
            return 0

        width_delta = width - self.fist_width
        travel_y = tip_y - self.fist_y
        if (
            in_line
            and opened
            and width_delta >= self.width_increase
            and travel_y >= self.min_down
        ):
            self.cooldown = self.cooldown_frames
            self.fist_width = None
            self.fist_y = None
            return self.scroll_amount

        return 0
