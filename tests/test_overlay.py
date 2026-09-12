"""Desk overlay state machine: intro frost, dismiss, HUD toggle."""

from __future__ import annotations

import json
import time
import unittest
import urllib.error
import urllib.request

import overlay


class OverlayStateTests(unittest.TestCase):
    def setUp(self) -> None:
        overlay.stop()

    def tearDown(self) -> None:
        overlay.stop()

    def test_welcome_starts_intro_and_defaults_hud_off(self) -> None:
        overlay.show_welcome("alex")
        snap = overlay.snapshot()
        self.assertTrue(snap["intro"])
        self.assertFalse(snap["hud"])
        self.assertTrue(snap["capture"])
        self.assertEqual(snap["display_name"], "Alex")
        self.assertEqual(snap["mode"], "intro")
        self.assertTrue(overlay.STATE_PATH.is_file())
        self.assertTrue(overlay.STATE_TXT.is_file())
        self.assertIn("intro=1", overlay.STATE_TXT.read_text(encoding="utf-8"))

    def test_click_or_esc_dismisses_intro_hud_stays_off(self) -> None:
        overlay.show_welcome("Alex")
        overlay.apply_event("dismiss_intro")
        snap = overlay.snapshot()
        self.assertFalse(snap["intro"])
        self.assertFalse(snap["hud"])
        self.assertFalse(snap["capture"])
        self.assertEqual(snap["mode"], "desk")

    def test_timeout_dismisses_intro(self) -> None:
        overlay.INTRO_SEC = 0.05
        overlay.show_welcome("Alex")
        time.sleep(0.08)
        overlay.tick()
        self.assertFalse(overlay.intro_active())
        self.assertFalse(overlay.desk_hud())
        overlay.INTRO_SEC = 2.5

    def test_h_toggles_desk_hud_after_intro(self) -> None:
        overlay.show_welcome("Alex")
        overlay.dismiss_intro()
        overlay.toggle_hud()
        self.assertTrue(overlay.desk_hud())
        overlay.toggle_hud()
        self.assertFalse(overlay.desk_hud())

    def test_h_during_intro_dismisses_and_turns_hud_on(self) -> None:
        overlay.show_welcome("Alex")
        overlay.toggle_hud()
        self.assertFalse(overlay.intro_active())
        self.assertTrue(overlay.desk_hud())

    def test_lock_restyle(self) -> None:
        overlay.show_welcome("Alex")
        overlay.dismiss_intro()
        overlay.sync(name="Alex", unlocked=False, lock_reason="unfamiliar face")
        snap = overlay.snapshot()
        self.assertEqual(snap["mode"], "locked")
        self.assertEqual(snap["lock_reason"], "unfamiliar face")
        self.assertFalse(snap["unlocked"])

    def test_http_state_and_events(self) -> None:
        overlay.show_welcome("Alex")
        url = overlay.server_url()
        self.assertTrue(url.startswith("http://127.0.0.1:"))
        with urllib.request.urlopen(url + "state.json", timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        self.assertTrue(data["intro"])
        req = urllib.request.Request(
            url + "event",
            data=json.dumps({"type": "dismiss_intro"}).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        self.assertFalse(data["intro"])
        with urllib.request.urlopen(url, timeout=2) as resp:
            html = resp.read().decode("utf-8")
        self.assertIn("Welcome", html)
        self.assertIn("Desk ready", html)
        self.assertIn("toggle HUD", html)
        self.assertNotIn("Hershey", html)


if __name__ == "__main__":
    unittest.main()
