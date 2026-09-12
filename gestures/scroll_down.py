"""ScrollDown gesture: hold the right pinky up to stream a smooth pixel scroll."""

from __future__ import annotations

from .pose import pinky_scroll_rate


class ScrollDown:
    def reset(self) -> None:
        return

    def update(self, hand_landmarks, fingers_down: list[str]) -> float:
        """Pixels per second while the pinky-only pose is held, else 0."""
        return pinky_scroll_rate(hand_landmarks.landmark, fingers_down)
