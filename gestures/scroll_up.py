"""ScrollUp gesture: smooth continuous page-scroll while the left pinky is up."""

from __future__ import annotations

from .pose import pinky_scroll_rate


class ScrollUp:
    def __init__(self):
        self.accrued = 0.0

    def reset(self) -> None:
        self.accrued = 0.0

    def update(self, hand_landmarks, fingers_down: list[str]) -> int:
        """Return a small line count each frame the pinky-only pose is held."""
        rate = pinky_scroll_rate(hand_landmarks.landmark, fingers_down)
        if rate <= 0:
            self.accrued = 0.0
            return 0

        self.accrued += rate
        lines = int(self.accrued)
        self.accrued -= lines
        return max(0, lines)
