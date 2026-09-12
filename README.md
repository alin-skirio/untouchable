# Hand Control

A personal Mac desk: webcam hand tracking plus a “hey Grok” voice listener. A recognized face unlocks the desk; gestures and voice then drive the pointer, apps, scroll, and a few page tools.

Gestures run from `track.py`. Voice runs from `voice.py`. `run.py` starts both.

## Run

```bash
cd ~/hand-control
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

Put an `XAI_API_KEY` in `.env` if you want voice, page simplify, or Desk handoff.

Tracker only (no mic):

```bash
python track.py
```

Show the camera window from the start:

```bash
python track.py --preview
```

`--no-preview` is the default. Press **C** anytime to show or hide the camera. **Q** or **Ctrl+C** quits.

macOS will ask for **Camera** on first launch. Gestures that type, click, or scroll also need **Accessibility** for Cursor (if you run from here) or Terminal: System Settings → Privacy & Security → Accessibility. Voice needs **Microphone**.

## Face lock

OpenCV YuNet + SFace identify faces on the same camera feed. Commands stay off until a saved profile is in view.

- A known face unlocks immediately and shows a **Welcome {Name}** frost intro (click, Esc, or ~2.5s).
- An empty frame does not lock. Walk away; the desk stays armed.
- If every visible face is a stranger for **5 seconds**, gestures and voice lock again.

Press **A** in the camera window to enroll (guided head turns, saved under `profiles/`). Press **L** to open People — add or delete profiles. First run downloads YuNet and SFace into `models/`. Re-enroll after encoder upgrades.

## Gestures

Right hand does the work unless noted. Tiny background hands are ignored. Hold a pose a few frames before it fires.

### Pointer

Arm in three beats: **open hand → fist → gun** (index, middle, and thumb out; ring and pinky curled). The index then moves the Mac cursor.

- Fold the **thumb** to left-click.
- Fold the **middle** finger to right-click.
- Make a **fist** to exit.
- Press **S** to warp the cursor onto the index tip and engage immediately.
- **⌘-** captures a desk-distance palm reference so sensitivity stays even as you lean in or back.

### App switcher

Pinch **right thumb + middle**. That holds ⌘ and sends the first Tab. Flap the **index** up and down to walk the app switcher. Release thumb and middle to land on the highlighted app.

### Scroll

- **Index only** up (middle, ring, pinky curled) — scroll up.
- **Pinky only** up (index, middle, ring curled) — scroll down.
- Tuck the **thumb** into the fist to go faster; let it stand off to return to normal speed.
- **Pop a fist open** (index through pinky uncurl together) — one flick-scroll down.

### TikTok mode

Hold both hands flat and **perpendicular** so they form a **T** — one palm against the other hand’s fingertips, like a timeout. That opens the TikTok app, or tiktok.com if the app is missing.

In that mode, **flick all fingertips up** to jump to the next video. The rise has to cover about **30% of a calibrated palm** (`12px` until ⌘- sets a reference). It can build up over about a third of a second, so a lazy flick still counts. Either hand works. Make the T again to close the tab and leave the mode. Drop your hands before making another T; it fires once per pose.

## Voice

Say **“hey Grok”** then a request. The listener prefers Grok Voice realtime and falls back to speech-to-text plus the same tool loop.

Grok can open apps, press shortcuts, type, move and click the mouse, scroll, and read what is focused. A few spoken shortcuts are wired directly:

| You say | What happens |
|---|---|
| *hey Grok, simplify this* | Reads the front Zen, Safari, or Chrome page and builds a step-by-step flyer (`simplified/latest.png`) |
| *hey Grok, send this off to desk and …* | Screenshots the Mac, captures the current site, and pastes an `@DESK` packet into Grok Bot so a specialist can continue |
| *hey Grok, shut down* | Quits the listener (not the Mac) |

Finish the Desk ask after **and**. The listener waits a beat so you can keep talking before it sends.

## Camera window

Hidden by default. **C** toggles it. While it is open:

| Key | Action |
|---|---|
| **C** | Hide or show the camera |
| **A** | Enroll a face |
| **L** | People list |
| **S** | Place the cursor on the index tip |
| **H** | Toggle HUD chrome |
| **Q** / **Esc** | Quit (Esc dismisses the intro first; also cancels enroll) |

After the intro the desk stays clear: HUD chrome defaults **off**. Press **H** (or the small HUD control) for status chips and the gesture hint bar. The camera window defaults HUD **on**; **H** hides it down to corner brackets.

**H** is a global desk shortcut (same Accessibility permission as ⌘-). The Terminal/overlay window does not need focus. Watch the console for `HUD on` / `HUD off`. If the frost cards never appear, check `.overlay.host.log`.

A locked / unfamiliar-face session restyles to a quiet amber “Commands off · unfamiliar face” state. Gestures, voice, and Face ID are unchanged.

## Layout

| File | Role |
|---|---|
| `run.py` | Starts tracker + voice together |
| `track.py` | Camera loop, gestures, face lock |
| `voice.py` | Wake-phrase listener |
| `gestures/` | Pointer, switcher, scroll, flick, T-pose |
| `faceid.py` | Enrollment and recognition |
| `presence.py` | Shared lock file the tracker writes and voice reads |
| `computer.py` | Generic Mac keys, mouse, apps, front-window context |
| `browser.py` | Front tab URL / HTML from Zen, Safari, or Chrome |
| `simplify.py` | Page → Grok Imagine flyer |
| `handoff.py` | Screen + page packet to the DESK Grok Bot |
| `overlay.py` | Desk intro frost + HUD overlay |
| `overlay_hud.html` | Overlay chrome (WebView) |
| `chrome.py` | Shared tokens + PIL camera type |
| `hud.py` | Camera window chrome |
