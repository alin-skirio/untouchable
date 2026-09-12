"""Flick-up gesture: every fingertip rises together. Drives next-video in TikTok mode."""

from __future__ import annotations

from collections import deque

from .pose import FOUR_TIPS

DEFAULT_RISE_PX = 12.0
PALM_FRACTION = 0.3  # of a calibrated palm
HISTORY_FRAMES = 12  # ~0.4s at 30fps, so a lazy flick still accumulates
# 8px is about as low as the threshold can go: at 6px, ordinary landmark
# jitter from a motionless hand starts firing the gesture on its own.
DEFAULT_RISE_PX = 8.0
PALM_FRACTION = 0.12  # of a calibrated palm, ~8-13px at normal desk distance
HISTORY_FRAMES = 16  # ~0.5s at 30fps, so a slow drift upward still counts
MIN_HISTORY = 2
COOLDOWN_FRAMES = 8
MIN_TIP_RATIO = 0.5  # slack for the worst-tracked tip, usually the pinky
COOLDOWN_FRAMES = 6
MIN_TIP_RATIO = 0.3  # slack for the worst-tracked tip, usually the pinky


def rise_threshold_px(ref_palm_px: float | None, default_px: float = DEFAULT_RISE_PX) -> float:
    """A fraction of a calibrated palm, or the default until ⌘T sets a reference."""
    if ref_palm_px and ref_palm_px > 0:
        return ref_palm_px * PALM_FRACTION
    return default_px


class FlickUp:
    """Fires when the fingertips as a group have risen past the threshold in a short window.

    The average tip has to clear the threshold and every tip has to clear
    ``min_tip_ratio`` of it, so the whole hand still has to travel but one
    badly tracked finger cannot veto the flick.

    History is kept per hand label, so the left and right hands flick independently.
    """

    def __init__(
        self,
        history_frames: int = HISTORY_FRAMES,
        cooldown_frames: int = COOLDOWN_FRAMES,
        tips=FOUR_TIPS,
        min_tip_ratio: float = MIN_TIP_RATIO,
    ):
        self.history_frames = history_frames
        self.cooldown_frames = cooldown_frames
        self.tips = tuple(tips)
        self.min_tip_ratio = min_tip_ratio
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
            rises = [
                max(frame[i] for frame in history) - current[i]
                for i in range(len(self.tips))
            ]
            self.last_rise = sum(rises) / len(rises)
            if self.last_rise >= threshold_px and min(rises) >= threshold_px * self.min_tip_ratio:
                fired = True
                self._cooldown[label] = self.cooldown_frames
                history.clear()

        history.append(current)
        return fired
