"""Swipe scroll gesture: uncurl a fist into an open hand to scroll."""

from __future__ import annotations

from collections import deque


class SwipeScroller:
    """Triggers a scroll when the 4 main fingers quickly pop from a curled fist to fully extended."""

    def __init__(self, history_size: int = 10, scroll_amount: int = 150):
        # 10 frames equals roughly a third of a second for the motion to happen
        self.history = deque(maxlen=history_size)
        self.scroll_amount = scroll_amount
        self.cooldown = 0

    def reset(self) -> None:
        self.history.clear()
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
