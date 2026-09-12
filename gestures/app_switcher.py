"""App switcher gesture: hold thumb+middle (⌘) and tap index (Tab)."""

from __future__ import annotations

from .landmarks import INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP, THUMB_TIP, WRIST, dist2


class AppSwitcher:
    """Tracks holding thumb+middle to keep ⌘ active while tapping index to press Tab.

    Ring and pinky fingers are completely ignored during gesture recognition.
    """

    def __init__(
        self,
        pinch: float = 0.34,
        release: float = 0.50,
        flap: float = 0.45,
        confirm_frames: int = 3,
        min_wrist_dist: float = 1.05,
        min_hand_width: float = 0.30,  # NEW: Minimum width to ensure hand isn't sideways
    ):
        self.pinch = pinch
        self.release = release
        self.flap = flap
        self.confirm_frames = confirm_frames
        self.min_wrist_dist = min_wrist_dist
        self.min_hand_width = min_hand_width
        self.pinch_count = 0
        self.active = False
        self.index_up = False

    def reset(self) -> None:
        self.active = False
        self.index_up = False
        self.pinch_count = 0

    @property
    def busy(self) -> bool:
        return self.active or self.pinch_count > 0

    def update(self, hand_landmarks) -> tuple[bool, bool, bool]:
        lm = hand_landmarks.landmark
        palm = max(dist2(lm[WRIST], lm[MIDDLE_MCP]), 0.04)

        # --- NEW: Check if hand is facing the camera ---
        # Measure 2D distance between Index MCP (knuckle 5) and Pinky MCP (knuckle 17)
        INDEX_MCP, PINKY_MCP = 5, 17
        hand_width = dist2(lm[INDEX_MCP], lm[PINKY_MCP]) / palm
        is_sideways = hand_width < self.min_hand_width

        # Distances between key active fingers (thumb, index, middle)
        tm_dist = dist2(lm[THUMB_TIP], lm[MIDDLE_TIP]) / palm
        index_dist = min(
            dist2(lm[THUMB_TIP], lm[INDEX_TIP]),
            dist2(lm[MIDDLE_TIP], lm[INDEX_TIP]),
        ) / palm

        # Distance checks relative to wrist using ONLY thumb and middle finger
        thumb_wrist = dist2(lm[THUMB_TIP], lm[WRIST]) / palm
        middle_wrist = dist2(lm[MIDDLE_TIP], lm[WRIST]) / palm

        # In a closed fist, the active fingertips collapse tightly toward the wrist
        is_fist = (thumb_wrist < self.min_wrist_dist) or (middle_wrist < 0.90)

        started, tapped, ended = False, False, False

        if not self.active:
            # We now require the hand to NOT be sideways to initiate the pinch
            if not is_fist and not is_sideways and max(tm_dist, index_dist) <= self.pinch:
                self.pinch_count += 1
                if self.pinch_count >= self.confirm_frames:
                    self.active = True
                    self.index_up = False
                    started = True
            else:
                self.pinch_count = 0
        else:
            self.pinch_count = 0
            if tm_dist >= self.release or is_fist:
                self.active = False
                self.index_up = False
                ended = True
            else:
                if not self.index_up and index_dist > self.flap:
                    self.index_up = True
                elif self.index_up and index_dist <= self.pinch:
                    self.index_up = False
                    tapped = True

        return started, tapped, ended
