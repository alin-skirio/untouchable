"""Flick-up gesture: every fingertip rises together. Drives next-video in TikTok mode."""

from __future__ import annotations

from collections import deque

from .pose import FOUR_TIPS

DEFAULT_RISE_PX = 20.0
HISTORY_FRAMES = 8
MIN_HISTORY = 3
COOLDOWN_FRAMES = 12


def rise_threshold_px(ref_palm_px: float | None, default_px: float = DEFAULT_RISE_PX) -> float:
    """Half a calibrated palm, or the default until ⌘T sets a reference."""
    if ref_palm_px and ref_palm_px > 0:
        return ref_palm_px / 2.0
    return default_px


class FlickUp:
    """Fires when every tracked fingertip has risen past the threshold in a short window.

    History is kept per hand label, so the left and right hands flick independently.
    """

    def __init__(
        self,
        history_frames: int = HISTORY_FRAMES,
        cooldown_frames: int = COOLDOWN_FRAMES,
        tips=FOUR_TIPS,
    ):
        self.history_frames = history_frames
        self.cooldown_frames = cooldown_frames
        self.tips = tuple(tips)
        self.last_rise = 0.0
        self._history: dict[str, deque[tuple[float, ...]]] = {}
        self._cooldown: dict[str, int] = {}

    def reset(self) -> None:
        self._history.clear()
        self._cooldown.clear()
        self.last_rise = 0.0

    def forget(self, label: str) -> None:
        """Drop a hand's history when it leaves the frame."""
        self._history.pop(label, None)
        self._cooldown.pop(label, None)

    def update(
        self,
        label: str,
        hand_landmarks,
        height: float,
        threshold_px: float,
    ) -> bool:
        lm = hand_landmarks.landmark
        current = tuple(lm[i].y * height for i in self.tips)
        history = self._history.setdefault(label, deque(maxlen=self.history_frames))

        cooldown = self._cooldown.get(label, 0)
        if cooldown > 0:
            self._cooldown[label] = cooldown - 1
            history.append(current)
            return False

        fired = False
        if len(history) >= MIN_HISTORY:
            # Image y grows downward, so a rise is the climb from each tip's recent low.
            rise = min(
                max(frame[i] for frame in history) - current[i]
                for i in range(len(self.tips))
            )
            self.last_rise = rise
            if rise >= threshold_px:
                fired = True
                self._cooldown[label] = self.cooldown_frames
                history.clear()

        history.append(current)
        return fired
