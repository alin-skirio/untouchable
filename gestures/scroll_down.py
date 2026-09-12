"""ScrollDown gesture: page-scroll down while the pinky is pointing up."""

from __future__ import annotations

from .pose import pinky_pointing_up


class ScrollDown:
    def __init__(self, scroll_amount: int = 150, cooldown_frames: int = 18):
        self.scroll_amount = scroll_amount
        self.cooldown_frames = cooldown_frames
        self.cooldown = 0

    def reset(self) -> None:
        self.cooldown = 0

    def update(self, hand_landmarks, fingers_down: list[str]) -> int:
        """Return lines to scroll down this frame while the pinky stays up."""
        if self.cooldown > 0:
            self.cooldown -= 1

        if not pinky_pointing_up(hand_landmarks.landmark, fingers_down):
            return 0

        if self.cooldown > 0:
            return 0

        self.cooldown = self.cooldown_frames
        return self.scroll_amount
