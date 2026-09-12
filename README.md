# Hand Control

Personal Mac experiment: use a webcam to read hand motion and map it to system controls. It lives in the menu bar — click the icon for a panel with the live camera view.

## Run

```bash
cd ~/code/personal/untouchable
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

A hand icon appears in the menu bar. Click it to show or hide the camera panel; right-click (or control-click) for Add Face, List Profiles, and Quit. Tracking keeps running while the panel is closed. Pass `--open` to start with the panel showing, or `--camera N` to pick a camera.

The panel has the live feed, a **Desk** sensitivity slider, and buttons for **Set Reference**, **Place Cursor**, and **Add Face**.

Hold one or both hands in front of the camera. Each hand gets a skeleton and a motion trail (teal = left, amber = right). Curl **exactly one** finger to move that hand's trail onto that fingertip; if more than one finger is down, the trail stays put.

Pinch your **right** thumb, index, and middle fingertips together to send **⌘Tab**. Open those fingers before pinching again. The shortcut is sent system-wide, so it still works if the preview is in the background or closed.

Face recognition (OpenCV SFace embeddings): use **Add Face** to enroll a named profile (saved under `profiles/`) and **List Profiles** to print what's saved. First run downloads YuNet + SFace models into `models/`. Re-enroll after encoder upgrades. Quit from the icon's right-click menu, or **Ctrl+C** in the terminal.

Headless, no menu bar or window:

```bash
python track.py
```

macOS will ask for **Camera** permission the first time. To send ⌘Tab you also need **Accessibility** enabled for **Cursor** (if you run from here) or **Terminal**: System Settings → Privacy & Security → Accessibility.
