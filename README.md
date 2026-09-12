# Hand Control

Personal Mac experiment: use a webcam to read hand motion, then later map that motion to system controls. This first step is a live camera view with hand tracking.

## Run

```bash
cd ~/code/personal/untouchable
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python track.py
```

Hold one or both hands in front of the camera. Each hand gets a skeleton and a motion trail (teal = left, amber = right). Curl **exactly one** finger to move that hand's trail onto that fingertip; if more than one finger is down, the trail stays put.

Pinch your **right** thumb, index, and middle fingertips together to send **⌘Tab**. Open those fingers before pinching again. The shortcut is sent system-wide, so it still works if the preview is in the background or closed.

Hold both hands flat and **perpendicular** so they form a **T** — one hand's palm resting against the other hand's fingertips, like a timeout signal — to open TikTok and enter **TikTok mode**. If the TikTok app isn't installed, it opens tiktok.com instead.

In TikTok mode, **flick all your fingertips up** to jump to the next video. The flick has to cover **30% of a calibrated palm**, which is `12px` until **⌘T** sets a desk reference, and it can build up over about a third of a second so a lazy flick still counts. Either hand works, and each is tracked separately. To retune, edit `DEFAULT_RISE_PX`, `PALM_FRACTION`, and `HISTORY_FRAMES` in `gestures/flick_up.py`. Make the **T** again to close the tab and leave the mode. The T fires once per pose, so drop your hands before making another.
In TikTok mode, **flick all your fingertips up** to jump to the next video. The flick only has to cover `8px` (or **12% of a calibrated palm** once **⌘T** sets a desk reference, roughly the same distance at normal desk range), and it can build up over about half a second, so a slow drift upward counts too. Either hand works, and each is tracked separately. To retune, edit `DEFAULT_RISE_PX`, `PALM_FRACTION`, and `HISTORY_FRAMES` in `gestures/flick_up.py`. Make the **T** again to close the tab and leave the mode. The T fires once per pose, so drop your hands before making another.

Face recognition (OpenCV SFace embeddings): press **A** to add a named profile (saved under `profiles/`), **L** to list profiles. First run downloads YuNet + SFace models into `models/`. Re-enroll after encoder upgrades. Press **Q** or **Esc** in the preview to quit, or **Ctrl+C** in the terminal.

Background-only (no window):

```bash
python track.py --no-preview
```

macOS will ask for **Camera** permission the first time. To send ⌘Tab you also need **Accessibility** enabled for **Cursor** (if you run from here) or **Terminal**: System Settings → Privacy & Security → Accessibility.
