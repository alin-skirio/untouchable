"""Headless runner for the hand tracker. Use app.py for the menu bar UI.

Gestures still drive the system: pinch right thumb+middle and tap index to
cycle apps with ⌘Tab, pop a right fist open to flick-scroll, point the left or
right pinky to scroll, and run open hand → fist → index+middle+thumb out to
take over the cursor. ⌘T captures the desk-distance reference.

Face recognition runs on the same feed and reports in the console.

Ctrl+C to quit.
"""

from __future__ import annotations

import argparse
import signal
import time

from tracker import HandTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Webcam hand tracker for Mac controls (no window)"
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Accepted for compatibility; this runner never opens a window",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        metavar="N",
        help="Force camera index N (skips auto-prefer of the Mac built-in camera)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    running = True

    def stop(_signum=None, _frame=None) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    tracker = HandTracker(camera=args.camera)
    tracker.start()
    print("Tracking in the background. ⌘T sets the desk reference. Ctrl+C to quit.")

    try:
        while running:
            if tracker.camera_lost:
                if tracker.error:
                    print(f"Tracker stopped: {tracker.error}")
                break
            time.sleep(0.1)
    finally:
        tracker.stop()


if __name__ == "__main__":
    main()
