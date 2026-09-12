"""ScrollDown gesture: smooth continuous page-scroll while the pinky is up."""

from __future__ import annotations

from .pose import pinky_pointing_up


class ScrollDown:
    def __init__(self, lines_per_frame: float = 1.6):
        self.lines_per_frame = lines_per_frame
        self.accrued = 0.0

    def reset(self) -> None:
        self.accrued = 0.0

    def update(self, hand_landmarks, fingers_down: list[str]) -> int:
        """Return a small line count each frame the pinky stays pointing up."""
        if not pinky_pointing_up(hand_landmarks.landmark, fingers_down):
            self.accrued = 0.0
            return 0

        self.accrued += self.lines_per_frame
        lines = int(self.accrued)
        self.accrued -= lines
        return max(0, lines)
