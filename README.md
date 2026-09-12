# Untouchable

Control your Mac with your hands over a webcam. MediaPipe reads 21 landmarks per hand, and gestures become real system events — pointer moves, clicks, scrolls, keyboard shortcuts — posted through Quartz/CoreGraphics. That means they land in whatever app is in front; the preview window never needs focus, and tracking keeps running with the window closed.

**macOS only.** The event layer talks to CoreGraphics directly and has no Windows or Linux equivalent.

## Run

```bash
cd ~/code/personal/untouchable
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python track.py
```

Useful flags: `--no-preview` runs tracking with no window at all, and `--camera N` picks a different capture device.

macOS will ask for **Camera** permission the first time. Sending keys and moving the pointer also needs **Accessibility** enabled for whichever app launches the script — **Terminal**, or **Cursor** if you run it from the editor: System Settings → Privacy & Security → Accessibility.

## Pointer

Steer the pointer with your **right hand**, but it has to be armed first so it can't grab the cursor by accident. Show an **open hand**, close it into a **fist**, then extend **index, middle, and thumb** with ring and pinky curled. The HUD walks you through the sequence and tells you which step it wants next. Once engaged, your index tip drives the pointer.

While engaged, **fold your thumb** to left-click and **fold your middle finger** to right-click. Both are hold-to-drag: the button stays down as long as the finger stays folded, so you can drag windows and select text rather than only tapping.

Movement is relative and scaled by how far you are from the camera, so the same physical hand motion covers the same screen distance whether you're leaning in or sitting back. See [Calibration](#calibration) for how that's measured. The **Desk** slider on the preview window sets base sensitivity from `0.25x` to `3.00x` and is saved between runs.

## Calibration

Press **⌘T** to capture a desk-distance reference. It saves a snapshot plus your palm size in pixels, and every distance-sensitive gesture is measured against it.

Pointer scaling becomes `base_sensitivity × (reference_palm ÷ current_palm)`, which is what keeps cursor travel consistent as you move toward or away from the camera. The TikTok flick threshold is derived from the same reference. ⌘T is captured by a global event tap, so it works even when the preview isn't focused.

## Gestures

| Gesture | Action |
| --- | --- |
| Pinch right thumb + index + middle | Hold ⌘ to start app switching |
| Flap index up and down while holding thumb + middle | Send Tab to step through apps |
| Release thumb + middle | Select the highlighted app |
| Pop a right fist open (index through pinky) | Flick-scroll down |
| Left pinky up, other fingers curled | Scroll up continuously |
| Right pinky up, other fingers curled | Scroll down continuously |
| Thumb pressed into that fist | Scroll faster; release to return to normal speed |
| Right thumb + ring + pinky down, index + middle up and together, swipe up | Scroll down, scaled by how hard you swipe |
| Both hands flat and perpendicular, forming a **T** | Toggle TikTok mode |
| Flick all fingertips up (TikTok mode only) | Next video |

Curling **exactly one** finger moves that hand's motion trail onto that fingertip. With more than one finger down the trail stays where it is. Trails are colored teal for the left hand and amber for the right.

## TikTok mode

Hold both hands flat and **perpendicular** so they form a **T** — one palm resting against the other hand's fingertips, like a timeout signal. That opens TikTok, or tiktok.com if the app isn't installed, and enters **TikTok mode**.

In TikTok mode, **flick all your fingertips up** to jump to the next video. The flick only has to cover `8px`, or 12% of a calibrated palm once ⌘T has set a reference, which works out to about the same distance at normal desk range. It can accumulate over roughly half a second, so a slow drift upward counts too. Either hand works and each is tracked independently. The average fingertip has to clear the threshold and every fingertip has to clear a third of it, so one badly tracked finger won't block a real flick, but a single twitching finger won't trigger one either.

Make the **T** again to send ⌘W and leave the mode. Note that ⌘W goes to whatever is frontmost, so click back to TikTok before exiting. The T fires once per pose, so drop your hands before forming another.

To retune the flick, edit `DEFAULT_RISE_PX`, `PALM_FRACTION`, and `HISTORY_FRAMES` at the top of `gestures/flick_up.py`. `8px` is close to the practical floor — below about `6px`, ordinary landmark jitter from a motionless hand starts firing the gesture by itself.

## Face profiles

Face recognition runs on the same camera feed using OpenCV YuNet detection and SFace embeddings. Press **A** to add a named profile and **L** to list the ones you have. Profiles are saved under `profiles/`, and the first run downloads both models into `models/`. Re-enroll after an encoder upgrade, since old embeddings won't match.

## Keys

| Key | Action |
| --- | --- |
| **⌘T** | Capture the desk-distance reference (works globally) |
| **S** | Warp the pointer onto your index tip and engage pointer mode |
| **A** | Add a named face profile |
| **L** | List saved face profiles |
| **Q** or **Esc** | Quit |
| **Ctrl+C** | Quit from the terminal, including with `--no-preview` |

Preview keys need the window focused. Esc backs out of the name prompt first if you're mid-enrollment.

## Layout

| Path | Purpose |
| --- | --- |
| `track.py` | Entry point: camera loop, preview HUD, and gesture wiring |
| `gestures/` | One module per gesture, plus shared landmark and pose helpers |
| `mac_keys.py` | Global keystrokes, mouse events, and the ⌘T event tap via Quartz |
| `computer.py` | Generic Mac control primitives for the agent — keys, text, mouse, apps |
| `launcher.py` | Site- and app-specific launchers, kept out of `computer.py` |
| `faceid.py` | Face detection, recognition, and profile storage |
| `rim_flash.py` | On-frame bezel cue when pointer mode is armed |
| `grok_agent.py`, `voice.py` | Grok function-calling loop and the "hey grok" voice front end |

Each gesture module owns its own thresholds, confirm-frame counts, and cooldowns, so tuning one gesture doesn't disturb the others.
