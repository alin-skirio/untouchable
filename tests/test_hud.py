"""Camera chrome: HUD on/off, lock restyle, no Hershey text."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from hud import CameraHud, overlay_copy


def _frame(w: int = 960, h: int = 540) -> np.ndarray:
    img = np.full((h, w, 3), (15, 11, 10), dtype=np.uint8)
    cv2.circle(img, (w // 3, h // 2), 70, (28, 26, 24), -1)
    return img


class CameraHudTests(unittest.TestCase):
    def test_defaults_chrome_on(self) -> None:
        hud = CameraHud()
        self.assertTrue(hud.chrome)

    def test_toggle_chrome_action(self) -> None:
        hud = CameraHud()
        hud.apply("toggle_chrome")
        self.assertFalse(hud.chrome)
        hud.apply("toggle_chrome")
        self.assertTrue(hud.chrome)

    def test_hud_on_draws_wordmark_and_dock(self) -> None:
        hud = CameraHud()
        frame = _frame()
        hud.draw_panel(frame, ["Face: Alex", "Pointer"], pointer_engaged=True)
        path = "/tmp/camera-hud-on.png"
        cv2.imwrite(path, frame)
        # Wordmark / legend live in the upper band; dock near the bottom.
        top = frame[0:70]
        self.assertGreater(int(top.mean()), 8)
        self.assertTrue(hud._cam_hits)
        actions = {action for _box, action in hud._cam_hits}
        self.assertIn("toggle_preview", actions)
        self.assertIn("toggle_faces", actions)
        self.assertIn("quit", actions)
        self.assertIn("toggle_chrome", actions)

    def test_hud_off_keeps_toggle_and_hides_dock(self) -> None:
        hud = CameraHud()
        hud.chrome = False
        frame = _frame()
        hud.draw_panel(frame, ["Face: Alex"])
        actions = {action for _box, action in hud._cam_hits}
        self.assertEqual(actions, {"toggle_chrome"})
        cv2.imwrite("/tmp/camera-hud-off.png", frame)

    def test_locked_banner(self) -> None:
        hud = CameraHud()
        frame = np.full((540, 960, 3), (80, 72, 64), dtype=np.uint8)
        before = frame.mean()
        hud.draw_panel(frame, ["Locked", "Face: Unknown"])
        self.assertLess(frame.mean(), before)
        cv2.imwrite("/tmp/camera-hud-locked.png", frame)
        title, _detail, locked = overlay_copy(["Locked"])
        self.assertTrue(locked)
        self.assertEqual(title, "Locked")

    def test_shortcuts_still_mapped(self) -> None:
        hud = CameraHud()
        frame = _frame()
        hud.draw_panel(frame, ["Face: Alex"])
        actions = {action for _box, action in hud._cam_hits}
        self.assertIn("toggle_preview", actions)  # C
        self.assertIn("toggle_faces", actions)  # A / L
        self.assertIn("snap_cursor", actions)  # S
        self.assertIn("quit", actions)  # Q


if __name__ == "__main__":
    unittest.main()
