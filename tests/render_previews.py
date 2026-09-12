"""Render overlay + camera HUD previews for visual checks."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

import overlay
from hud import CameraHud

OUT = Path("/tmp/hand-control-ui")
OUT.mkdir(parents=True, exist_ok=True)


def _cam(name: str, lines: list[str], *, chrome: bool = True, locked: bool = False) -> None:
    hud = CameraHud()
    hud.chrome = chrome
    frame = np.full((540, 960, 3), (15, 11, 10), dtype=np.uint8)
    if locked:
        frame[:] = (36, 30, 28)
    cv2.circle(frame, (320, 270), 78, (30, 28, 26), -1)
    hud.draw_panel(frame, lines, pointer_engaged="Pointer" in lines)
    cv2.imwrite(str(OUT / name), frame)


def _shot(url: str, name: str) -> None:
    dest = OUT / name
    subprocess.run(
        [
            "google-chrome",
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--no-sandbox",
            "--window-size=1440,900",
            f"--screenshot={dest}",
            url,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> None:
    overlay.stop()
    overlay.show_welcome("Alex")
    overlay.sync(name="Alex", unlocked=True, pointer=True, voice=True)
    base = overlay.server_url() + "?preview=1"
    time.sleep(0.15)
    _shot(base, "overlay-intro.png")
    overlay.dismiss_intro()
    overlay.set_hud(False)
    overlay.sync(name="Alex", unlocked=True, pointer=True, voice=True)
    time.sleep(0.15)
    _shot(base, "overlay-hud-off.png")
    overlay.set_hud(True)
    time.sleep(0.15)
    _shot(base, "overlay-hud-on.png")
    overlay.sync(name="Alex", unlocked=False, lock_reason="unfamiliar face")
    time.sleep(0.15)
    _shot(base, "overlay-locked.png")
    overlay.stop()

    _cam("camera-hud-on.png", ["Face: Alex", "Pointer"])
    _cam("camera-hud-off.png", ["Face: Alex"], chrome=False)
    _cam("camera-hud-locked.png", ["Locked", "Face: Unknown"], locked=True)
    print(f"Wrote previews to {OUT}")


if __name__ == "__main__":
    main()
