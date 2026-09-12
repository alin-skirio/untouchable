"""On-frame bezel cue when pointer mode is armed. Avoids Tk, which crashes OpenCV on macOS."""

from __future__ import annotations

import cv2

ENGAGE_BGR = (255, 220, 80)
RELEASE_BGR = (80, 138, 255)


def flash_pointer_rim(_engage: bool) -> None:
    """Kept for call sites. Screen-wide Tk overlay is disabled (it abort()s with OpenCV)."""
    return


def draw_pointer_bezel(frame, engaged: bool) -> None:
    if frame is None or not engaged:
        return
    height, width = frame.shape[:2]
    inset = 8
    cv2.rectangle(
        frame,
        (inset, inset),
        (width - inset - 1, height - inset - 1),
        ENGAGE_BGR,
        3,
        cv2.LINE_AA,
    )
