# Hand Control

Personal Mac experiment: use a webcam to read hand motion, then later map that motion to system controls. This first step is a live camera view with hand tracking.

## Run

```bash
cd ~/hand-control
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python track.py
```

Hold one or both hands in front of the camera. Each hand gets a skeleton and a motion trail (teal = left, amber = right). Curl **exactly one** finger to move that hand's trail onto that fingertip; if more than one finger is down, the trail stays put.

Pinch your **right** thumb, index, and middle fingertips together to send **⌘Tab**. Open those fingers before pinching again. The shortcut is sent system-wide, so it still works if the preview is in the background or closed. Press **Q** or **Esc** in the preview to quit, or **Ctrl+C** in the terminal.

Background-only (no window):

```bash
python track.py --no-preview
```

macOS will ask for **Camera** permission the first time. To send ⌘Tab you also need **Accessibility** enabled for **Cursor** (if you run from here) or **Terminal**: System Settings → Privacy & Security → Accessibility.
