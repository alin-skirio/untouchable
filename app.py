"""Menu bar entry point.

Puts a small icon in the macOS status bar. Click it to open a panel with the
live camera view and controls; tracking keeps running when the panel is closed.
"""

from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hand Control menu bar app")
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        metavar="N",
        help="Force camera index N (skips auto-prefer of the Mac built-in camera)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the camera panel immediately instead of starting collapsed",
    )
    return parser.parse_args()


def main() -> None:
    if sys.platform != "darwin":
        raise SystemExit("The menu bar app requires macOS. Use track.py for headless runs.")

    args = parse_args()

    try:
        from PyObjCTools import AppHelper
    except ImportError:
        raise SystemExit(
            "PyObjC is missing. Install it with: pip install -r requirements.txt"
        )

    from tracker import HandTracker
    from ui.menubar_app import MenuBarApp

    tracker = HandTracker(camera=args.camera)
    tracker.start()

    controller = MenuBarApp.alloc().init()
    controller.setup(tracker)
    if args.open:
        controller.showPanel_(None)

    print("Hand Control is in the menu bar. Click the icon for the camera panel.")
    try:
        AppHelper.runEventLoop(installInterrupt=True)
    finally:
        tracker.stop()


if __name__ == "__main__":
    main()
