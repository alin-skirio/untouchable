"""ScrollUp gesture: hold the right index up to stream a smooth pixel scroll."""

from __future__ import annotations

from .pose import index_scroll_rate


class ScrollUp:
    def reset(self) -> None:
        return

    def update(self, hand_landmarks, fingers_down: list[str]) -> float:
        """Pixels per second while the index-only pose is held, else 0."""
        return index_scroll_rate(hand_landmarks.landmark, fingers_down)
