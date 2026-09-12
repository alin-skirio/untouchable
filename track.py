"""Live webcam preview with MediaPipe hand landmarks.

Hold one or both hands in view. The trail follows a fingertip on each hand.
Curl exactly one finger to move that hand's trail onto that fingertip.
If more than one finger is down, the trail stays on the current finger.

Pinch right thumb+index+middle to start app switching (holds ⌘).
Flap your index finger up and down while holding thumb+middle to send Tab.
Release thumb+middle to select the active application.

ScrollUp and ScrollDown are always watched on the right hand. Hold the
pinky up for 3 seconds (fingers touching) to start ScrollUp, make a
fist, then raise: the page follows and stops when all fingers point up
at max palm-reach. Fist again to repeat. If the fingers separate, it
cancels. Open and drop a lined-up hand for ScrollDown.

Close the preview or use --no-preview; Ctrl+C to quit.
"""

from __future__ import annotations

import argparse
import math
import signal
import time
import threading
from collections import deque

import cv2
import mediapipe as mp

from gestures.app_switcher import AppSwitcher
from gestures.scroll_down import ScrollDown
from gestures.scroll_up import ScrollUp
from mac_keys import async_cmd, async_tap_tab, set_cmd_state, smooth_scroll

TRAIL_LENGTH = 24
MAX_HANDS = 2
DEFAULT_FINGER = "index"
CONFIRM_FRAMES = 4
MIN_PALM_SIZE_PX = 75.0  # Ignore background hands with small pixel dimensions
FOLD_ANGLE_DEG = 145.0
THUMB_FOLD_ANGLE_DEG = 150.0

FINGERS = (
    ("thumb", 2, 3, 4),
    ("index", 5, 6, 8),
    ("middle", 9, 10, 12),
    ("ring", 13, 14, 16),
    ("pinky", 17, 18, 20),
)
TIP_INDEX = {name: tip for name, _mcp, _pip, tip in FINGERS}

HAND_COLORS = {
    "Left": (0, 220, 160),
    "Right": (40, 180, 255),
}
DEFAULT_COLOR = (0, 200, 255)


def open_camera(preferred: int | None = None) -> cv2.VideoCapture:
    """Tries indices 0, 1, and 2 to handle Continuity Camera or secondary webcams."""
    indices = (preferred,) if preferred is not None else (0, 1, 2)
    for idx in indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if cap.isOpened():
            # Test frame grab
            ok, _ = cap.read()
            if ok:
                print(f"Connected to camera at index {idx}.")
                return cap
            cap.release()

    raise SystemExit(
        "Could not open any active camera feed.\n"
        "1. Ensure no other application (Zoom, FaceTime, Browser) is using the camera.\n"
        "2. On macOS: System Settings → Privacy & Security → Camera, then allow "
        "Cursor (or Terminal) and run this script again."
    )


def _pt(landmark) -> tuple[float, float, float]:
    return (landmark.x, landmark.y, landmark.z)


def _angle_at(landmarks, a: int, b: int, c: int) -> float:
    ax, ay, az = _pt(landmarks[a])
    bx, by, bz = _pt(landmarks[b])
    cx, cy, cz = _pt(landmarks[c])
    bax, bay, baz = ax - bx, ay - by, az - bz
    bcx, bcy, bcz = cx - bx, cy - by, cz - bz
    na = math.sqrt(bax * bax + bay * bay + baz * baz)
    nc = math.sqrt(bcx * bcx + bcy * bcy + bcz * bcz)
    if na < 1e-6 or nc < 1e-6:
        return 180.0
    cos = (bax * bcx + bay * bcy + baz * bcz) / (na * nc)
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


def fingers_down(hand_landmarks) -> list[str]:
    lm = hand_landmarks.landmark
    down: list[str] = []
    for name, mcp, pip, tip in FINGERS:
        limit = THUMB_FOLD_ANGLE_DEG if name == "thumb" else FOLD_ANGLE_DEG
        if _angle_at(lm, mcp, pip, tip) < limit:
            down.append(name)
    return down


def draw_hud(frame, lines: list[str]) -> None:
    height = 58 + 26 * max(len(lines), 1)
    cv2.rectangle(frame, (16, 16), (780, 16 + height), (18, 18, 18), -1)
    y = 46
    for line in lines:
        cv2.putText(
            frame,
            line,
            (28, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        y += 26
    cv2.putText(
        frame,
        "Q or Esc to quit",
        (28, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (170, 170, 170),
        1,
        cv2.LINE_AA,
    )


def draw_trail(frame, trail: deque[tuple[int, int]], color) -> None:
    if len(trail) < 2:
        return
    points = list(trail)
    for i in range(1, len(points)):
        thickness = max(1, int(6 * i / len(points)))
        cv2.line(frame, points[i - 1], points[i], color, thickness, cv2.LINE_AA)


def draw_tip(frame, point: tuple[int, int], color, label: str) -> None:
    cv2.circle(frame, point, 16, color, 2)
    cv2.circle(frame, point, 5, color, -1)
    cv2.putText(
        frame,
        label,
        (point[0] + 14, point[1] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Webcam hand tracker for Mac controls")
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Run without a window so tracking continues fully in the background",
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
    preview = not args.no_preview
    running = True

    def stop(_signum=None, _frame=None) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        model_complexity=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75,
    )
    cap = open_camera(preferred=args.camera)
    trails: dict[str, deque[tuple[int, int]]] = {
        "Left": deque(maxlen=TRAIL_LENGTH),
        "Right": deque(maxlen=TRAIL_LENGTH),
    }
    active_finger: dict[str, str] = {}
    pending: dict[str, tuple[str, int]] = {}
    
    switcher = AppSwitcher()
    scroll_up = ScrollUp()
    scroll_down = ScrollDown()
    
    last_cmd_tab_at = 0.0
    last_scroll_at = 0.0
    last_scroll_amount = 0
    last_scroll_action = ""
    failed_frame_count = 0
    window = "Hand Control — tracker"

    if preview:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        print("Tracking in the background too. Close this window or click away; Ctrl+C or Q to quit.")
    else:
        print("Running without a preview. Hold thumb+middle & tap index to cycle apps. Ctrl+C to quit.")

    try:
        while running:
            ok, frame = cap.read()
            if not ok:
                failed_frame_count += 1
                if failed_frame_count > 15:
                    print("Lost the camera feed.")
                    break
                time.sleep(0.03)
                continue

            failed_frame_count = 0  # Reset counter on successful frame read

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = hands.process(rgb)
            rgb.flags.writeable = True

            height, width = frame.shape[:2]
            seen: set[str] = set()
            status_lines: list[str] = []
            saw_right = False

            if result.multi_hand_landmarks:
                handedness_list = result.multi_handedness or []
                for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
                    wrist = hand_landmarks.landmark[0]
                    middle_mcp = hand_landmarks.landmark[9]
                    palm_size_px = math.hypot((wrist.x - middle_mcp.x) * width, (wrist.y - middle_mcp.y) * height)

                    if palm_size_px < MIN_PALM_SIZE_PX:
                        continue

                    label = "Hand"
                    if i < len(handedness_list):
                        label = handedness_list[i].classification[0].label
                    color = HAND_COLORS.get(label, DEFAULT_COLOR)
                    seen.add(label)

                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

                    current = active_finger.get(label, DEFAULT_FINGER)
                    down = fingers_down(hand_landmarks)
                    if len(down) == 1:
                        candidate = down[0]
                        prev, count = pending.get(label, (candidate, 0))
                        count = count + 1 if prev == candidate else 1
                        pending[label] = (candidate, count)
                        if count >= CONFIRM_FRAMES and candidate != current:
                            current = candidate
                            active_finger[label] = current
                            trails[label].clear()
                    else:
                        pending.pop(label, None)

                    active_finger.setdefault(label, current)
                    tip = hand_landmarks.landmark[TIP_INDEX[current]]
                    ix, iy = int(tip.x * width), int(tip.y * height)
                    trail = trails.setdefault(label, deque(maxlen=TRAIL_LENGTH))
                    trail.append((ix, iy))
                    draw_trail(frame, trail, color)
                    draw_tip(frame, (ix, iy), color, f"{label} {current}")
                    extra = ""
                    if len(down) == 1:
                        extra = "  (1 down)"
                    elif len(down) > 1:
                        extra = f"  ({len(down)} down, locked)"
                    status_lines.append(
                        f"{label}  {current}  x={tip.x:.2f}  y={tip.y:.2f}{extra}"
                    )

                    if label == "Right":
                        saw_right = True
                        
                        # --- App Switcher Logic ---
                        started, tapped, ended = switcher.update(hand_landmarks)

                        if started:
                            async_cmd(True)
                            async_tap_tab()
                            last_cmd_tab_at = time.monotonic()
                        elif tapped:
                            async_tap_tab()
                            last_cmd_tab_at = time.monotonic()
                        elif ended:
                            async_cmd(False)

                        if switcher.active:
                            status_lines.append("Right pinch: App Switcher Active (⌘ Held)")

                        # Always check ScrollUp and ScrollDown
                        up_lines = scroll_up.update(hand_landmarks, down)
                        down_lines = 0 if up_lines > 0 else scroll_down.update(hand_landmarks, down)
                        if up_lines > 0:
                            threading.Thread(
                                target=smooth_scroll, args=(up_lines,), daemon=True
                            ).start()
                            last_scroll_at = time.monotonic()
                            last_scroll_amount = up_lines
                            last_scroll_action = "up"
                        elif down_lines > 0:
                            threading.Thread(
                                target=smooth_scroll, args=(-down_lines,), daemon=True
                            ).start()
                            last_scroll_at = time.monotonic()
                            last_scroll_amount = down_lines
                            last_scroll_action = "down"
                        elif scroll_up.phase == "raising":
                            status_lines.append(
                                "ScrollUp: raise touching fingers — page follows"
                            )
                        elif scroll_up.phase == "wait_fist":
                            status_lines.append("ScrollUp: make a fist")
                        elif scroll_up.hold_seconds > 0:
                            status_lines.append(
                                f"ScrollUp: hold pinky up {scroll_up.hold_seconds:.1f}/{scroll_up.initiate_seconds:.0f}s"
                            )

            if not saw_right:
                if switcher.active:
                    switcher.reset()
                    async_cmd(False)
                scroll_up.reset()
                scroll_down.reset()

            for label, trail in trails.items():
                if label not in seen:
                    trail.clear()
                    active_finger.pop(label, None)
                    pending.pop(label, None)

            if time.monotonic() - last_cmd_tab_at < 0.8:
                status_lines.append("Sent Tab")
            if time.monotonic() - last_scroll_at < 1.0:
                status_lines.append(
                    f"Scroll {last_scroll_action} ({abs(last_scroll_amount)} lines)"
                )
            
            if not status_lines:
                status_lines = ["No hands in view"]
            status_lines.append("Hold Right thumb+middle & tap index = ⌘Tab cycle")

            if preview:
                draw_hud(frame, status_lines)
                cv2.imshow(window, frame)
                visible = cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE)
                if visible < 1:
                    preview = False
                    cv2.destroyAllWindows()
                    print("Preview closed. Tracking still running in the background. Ctrl+C to quit.")
                else:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
    finally:
        set_cmd_state(False)
        hands.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
