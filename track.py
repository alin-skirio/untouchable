"""Live webcam preview with MediaPipe hand landmarks and face profiles.

Hold one or both hands in view. The trail follows a fingertip on each hand.
Curl exactly one finger to move that hand's trail onto that fingertip.
If more than one finger is down, the trail stays on the current finger.

Pinch right thumb+index+middle to start app switching (holds ⌘).
Flap your index finger up and down while holding thumb+middle to send Tab.
Release thumb+middle to select the active application.

Pop a right fist open (index through pinky) to flick-scroll down.

Point only the left pinky up (index, middle, and ring curled) to
ScrollUp, or only the right pinky up to ScrollDown. Press the thumb
into the fist to go faster; release it to return to the normal speed.

Point index+middle+thumb with ring+pinky curled to move the cursor — but only after
open hand → fist → that pose. Press S to warp the cursor onto the index tip.
⌘- captures a desk-distance reference; the Desk slider is base sensitivity.
Hold right thumb+ring+pinky down, keep index+middle up and together,
and swipe up rapidly to scroll down dynamically based on swipe severity.

Hold both hands flat and perpendicular so they form a T (one hand's palm
against the other's fingertips) to open TikTok and enter TikTok mode. In
that mode, flicking all fingertips up by a fraction of a calibrated palm
(12px until ⌘- sets a reference) goes to the next video. Make the T again
to close the tab and leave the mode.

Face recognition runs on the same camera feed. Press A to add a named
profile (saved in profiles.db), L to list profiles.

Close the preview or use --no-preview; Ctrl+C to quit.
"""

from __future__ import annotations

import argparse
import math
import signal
import threading
import time
from collections import deque

import cv2
import mediapipe as mp

from faceid import FaceID, NameEntryUI, open_camera
from gestures.app_switcher import AppSwitcher
from gestures.cursor import PointerCursor
from gestures.landmarks import INDEX_TIP, palm_size_px
from gestures.scroll_down import ScrollDown
from gestures.scroll_up import ScrollUp
from gestures.flick_up import FlickUp, rise_threshold_px
from gestures.swipe_scroller import SwipeScroller
from gestures.t_pose import TPose
from hud import (
    CAM_WINDOW,
    RAIL_WINDOW,
    CameraHud,
    open_camera_window,
    open_rail,
    window_open,
)
from launcher import close_tiktok_tab, next_tiktok_video, open_tiktok
import presence
from mac_keys import (
    async_cmd,
    async_tap_tab,
    display_bounds,
    mouse_down,
    mouse_up,
    move_mouse,
    release_mouse,
    set_cmd_state,
    set_mouse_position,
    smooth_scroll,
    start_cmd_t_monitor,
    start_hud_keys,
)
from rim_flash import flash_pointer_rim

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
    hud = CameraHud()
    hud.preview = not args.no_preview
    running = True

    def stop(_signum=None, _frame=None) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        model_complexity=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75,
    )
    face_id = FaceID()
    name_ui = NameEntryUI()
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
    scroller = SwipeScroller()
    pointer = PointerCursor()
    t_pose = TPose()
    flick = FlickUp()
    tiktok_mode = False
    cmd_t = start_cmd_t_monitor()

    last_cmd_tab_at = 0.0
    last_scroll_at = 0.0
    last_scroll_amount = 0
    last_scroll_action = ""
    last_ref_at = 0.0
    last_ref_msg = ""
    last_action_at = 0.0
    last_action_msg = ""
    last_right_palm = 0.0
    last_right_index: tuple[float, float] | None = None
    failed_frame_count = 0

    names = face_id.list_names()
    print(
        f"Loaded {len(names)} face profile(s) from {face_id.profiles_dir}"
        + (f": {', '.join(names)}" if names else "")
    )
    if hud.preview:
        open_camera_window(hud)
        print("Peach circle hides the camera. Rose circle quits.")
    else:
        print("Peach circle shows the camera. Rose circle quits.")
        open_rail(hud)
    preview_opened_at = time.monotonic() if hud.preview else 0.0

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
            saw_left = False
            pose_hands: list[tuple[object, list[str]]] = []

            # Face recognition on the shared feed — also the lock for gestures.
            status_lines.extend(face_id.process(frame, rgb))
            known = face_id.recognized_names()
            presence.publish(known)
            armed = bool(known)
            if not armed:
                status_lines.insert(0, "Locked")

            if result.multi_hand_landmarks:
                handedness_list = result.multi_handedness or []
                for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
                    label = "Hand"
                    if i < len(handedness_list):
                        label = handedness_list[i].classification[0].label

                    palm_px = palm_size_px(hand_landmarks, width, height)
                    if palm_px < MIN_PALM_SIZE_PX and not (
                        pointer.engaged and label == "Right"
                    ):
                        continue
                    color = HAND_COLORS.get(label, DEFAULT_COLOR)
                    seen.add(label)

                    if hud.on("landmarks"):
                        chrome.draw_hand_landmarks(frame, hand_landmarks, width, height)

                    current = active_finger.get(label, DEFAULT_FINGER)
                    down = fingers_down(hand_landmarks)
                    pose_hands.append((hand_landmarks, down))

                    # t_pose.holding is last frame's value, which suppresses the
                    # flick while hands are rising into a T to leave the mode.
                    if tiktok_mode and not t_pose.holding:
                        if flick.update(
                            label,
                            hand_landmarks,
                            height,
                            rise_threshold_px(pointer.ref_palm_px),
                        ):
                            threading.Thread(target=next_tiktok_video, daemon=True).start()
                            last_action_msg = "Flick up → next video"
                            last_action_at = time.monotonic()

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
                    if hud.on("landmarks") and hud.chrome and hud.preview:
                        draw_trail(frame, trail, color)
                    extra = ""
                    if len(down) == 1:
                        extra = "  (1 down)"
                    elif len(down) > 1:
                        extra = f"  ({len(down)} down, locked)"
                    status_lines.append(
                        f"{label}  {current}  x={tip.x:.2f}  y={tip.y:.2f}{extra}"
                    )

                    if label == "Left":
                        saw_left = True
                        if armed and hud.on("scroll_up"):
                            up_lines = scroll_up.update(hand_landmarks, down)
                            if up_lines > 0:
                                threading.Thread(
                                    target=smooth_scroll, args=(up_lines,), daemon=True
                                ).start()
                                last_scroll_at = time.monotonic()
                                last_scroll_amount = up_lines
                                last_scroll_action = "up"

                    if label == "Right":
                        saw_right = True
                        last_right_palm = palm_px
                        index = hand_landmarks.landmark[INDEX_TIP]
                        last_right_index = (index.x, index.y)

                        was_pointing = pointer.engaged
                        if armed and hud.on("pointer"):
                            cursor = pointer.update(hand_landmarks, down, (width, height))
                            if cursor.dx or cursor.dy:
                                move_mouse(cursor.dx, cursor.dy)
                            if cursor.click:
                                mouse_down(cursor.click)
                                mouse_up(cursor.click)
                            if cursor.engaged and not was_pointing:
                                flash_pointer_rim(True)
                            elif was_pointing and not cursor.engaged:
                                flash_pointer_rim(False)

                            if cursor.engaged:
                                scroll_down.reset()
                                scroller.reset()
                                if cursor.click == "left":
                                    status_lines.append("Pointer · left click")
                                elif cursor.click == "right":
                                    status_lines.append("Pointer · right click")
                                else:
                                    status_lines.append("Pointer")
                            elif was_pointing:
                                scroll_down.reset()
                                scroller.reset()
                            else:
                                status_lines.append(pointer.arm_hint)

                        if armed and not pointer.engaged:
                            if hud.on("app_switcher"):
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
                                    status_lines.append("App switcher")

                            if hud.on("scroll_down"):
                                down_lines = scroll_down.update(hand_landmarks, down)
                                if down_lines > 0:
                                    threading.Thread(
                                        target=smooth_scroll, args=(-down_lines,), daemon=True
                                    ).start()
                                    last_scroll_at = time.monotonic()
                                    last_scroll_amount = down_lines
                                    last_scroll_action = "down"
                            else:
                                down_lines = 0
                            if down_lines <= 0 and hud.on("flick"):
                                scroll_amount = scroller.update(hand_landmarks, down)
                                if scroll_amount > 0:
                                    threading.Thread(
                                        target=smooth_scroll, args=(-scroll_amount,), daemon=True
                                    ).start()
                                    last_scroll_at = time.monotonic()
                                    last_scroll_amount = scroll_amount
                                    last_scroll_action = "flick"

            if not armed or not saw_left:
                scroll_up.reset()

            if not armed or not saw_right:
                if pointer.engaged:
                    pointer.reset()
                    flash_pointer_rim(False)
                if switcher.active:
                    switcher.reset()
                    async_cmd(False)
                scroll_down.reset()
                scroller.reset()

            # Two flat hands held perpendicular (a "T") toggle TikTok mode.
            if not armed or pointer.engaged or not hud.on("t_pose"):
                t_pose.reset()
            elif t_pose.update(pose_hands, (width, height)):
                flick.reset()
                if tiktok_mode:
                    tiktok_mode = False
                    threading.Thread(target=close_tiktok_tab, daemon=True).start()
                    last_action_msg = "T pose → closed TikTok"
                else:
                    tiktok_mode = True
                    threading.Thread(target=open_tiktok, daemon=True).start()
                    last_action_msg = "T pose → TikTok mode on"
                scroller.reset()
                scroll_down.reset()
                last_action_at = time.monotonic()
                print(last_action_msg)

            for label, trail in trails.items():
                if label not in seen:
                    trail.clear()
                    active_finger.pop(label, None)
                    pending.pop(label, None)
                    flick.forget(label)

            if cmd_t.consume():
                if not armed or switcher.active:
                    pass
                elif saw_right and last_right_palm >= MIN_PALM_SIZE_PX:
                    saved = pointer.set_reference(last_right_palm, frame, (width, height))
                    last_ref_msg = "Reference captured" if saved else "Could not save reference"
                    last_ref_at = time.monotonic()
                    print(last_ref_msg)
                else:
                    last_ref_msg = "No hand for reference"
                    last_ref_at = time.monotonic()
                    print(last_ref_msg)

            if time.monotonic() - last_cmd_tab_at < 0.8:
                status_lines.append("Sent Tab")
            if time.monotonic() - last_scroll_at < 1.0:
                if last_scroll_action == "up":
                    status_lines.append(f"Scroll up · {abs(last_scroll_amount)}")
                elif last_scroll_action == "flick":
                    status_lines.append(f"Flick · {abs(last_scroll_amount)}")
                else:
                    status_lines.append(f"Scroll down · {abs(last_scroll_amount)}")
            if not seen:
                status_lines.append("No hands in view")
            if last_ref_msg and time.monotonic() - last_ref_at < 1.5:
                status_lines.append(last_ref_msg)
            if t_pose.holding:
                status_lines.append("T pose held")
            if tiktok_mode:
                status_lines.append(
                    f"TikTok mode · flick tips up {rise_threshold_px(pointer.ref_palm_px):.0f}px"
                    " = next video"
                )
            if last_action_msg and time.monotonic() - last_action_at < 1.5:
                status_lines.append(last_action_msg)
            status_lines.append(f"Dist {pointer.distance_ratio(last_right_palm):.1f}x")

            action = hud.take_click()
            key_snap = False
            if action == "quit":
                break
            if action:
                was_preview = hud.preview
                hud.apply(action)
                if hud.preview and not was_preview:
                    open_camera_window(hud)
                    preview_opened_at = time.monotonic()
                elif was_preview and not hud.preview:
                    name_ui.close()
                    try:
                        cv2.destroyWindow(CAM_WINDOW)
                    except cv2.error:
                        pass
            if not seen and not face_id.enrolling and "No hands in view" not in status_lines:
                status_lines.append("No hands in view")
            status_lines.append("Hold Right thumb+middle & tap index = ⌘Tab cycle")
            status_lines.append("S = place cursor on index tip")
            status_lines.append("Open → fist → index+middle+thumb out = pointer")
            status_lines.append("Pointer: thumb fold = left click, middle fold = right click")
            status_lines.append("Two flat hands in a T = TikTok mode on/off")

            if not hud.preview:
                if not window_open(RAIL_WINDOW):
                    open_rail(hud)
                hud.draw_rail()
            elif window_open(RAIL_WINDOW):
                try:
                    cv2.destroyWindow(RAIL_WINDOW)
                except cv2.error:
                    pass

            if hud.preview:
                hud.draw_panel(frame, status_lines, pointer_engaged=pointer.engaged)
                name_ui.draw(frame)
                cv2.imshow(CAM_WINDOW, frame)
                if (
                    time.monotonic() - preview_opened_at > 0.5
                    and not window_open(CAM_WINDOW)
                ):
                    hud.preview = False
                    name_ui.close()
                    try:
                        cv2.destroyWindow(CAM_WINDOW)
                    except cv2.error:
                        pass
                    print("Camera hidden. Peach circle brings it back.")

            key = cv2.waitKey(1) & 0xFF
            if not hud.preview:
                if key in (ord("q"), 27):
                    break
                continue
            if name_ui.active:
                result = name_ui.handle_key(key)
                if isinstance(result, str):
                    face_id.begin_enroll(result)
                # False = cancelled; None = still typing
                continue
            if key in (ord("q"),):
                break
            if key in (ord("s"), ord("S")) or key_snap:
                if not armed or switcher.active or not hud.on("pointer"):
                    pass
                elif saw_right and last_right_index is not None:
                    ox, oy, sw, sh = display_bounds()
                    if sw > 0 and sh > 0:
                        nx, ny = last_right_index
                        set_mouse_position(
                            ox + nx * sw,
                            oy + ny * sh,
                        )
                        was_engaged = pointer.engaged
                        pointer.engage_from_s()
                        scroll_up.reset()
                        scroll_down.reset()
                        scroller.reset()
                        if not was_engaged:
                            flash_pointer_rim(True)
                        last_ref_msg = "Cursor on index tip"
                        last_ref_at = time.monotonic()
                    else:
                        last_ref_msg = "Could not place cursor"
                        last_ref_at = time.monotonic()
                else:
                    last_ref_msg = "No hand to place cursor"
                    last_ref_at = time.monotonic()
            if key == 27:  # Esc
                if face_id.enrolling:
                    face_id.cancel_enroll()
                else:
                    break
            elif key in (ord("a"), ord("A")):
                if hud.on("face") and not face_id.enrolling:
                    name_ui.open()
            elif key in (ord("l"), ord("L")):
                names = face_id.list_names()
                if names:
                    print("Saved face profiles: " + ", ".join(names))
                else:
                    print("No face profiles yet. Press A to add one.")
    finally:
        presence.clear()
        release_mouse()
        set_cmd_state(False)
        cmd_t.stop()
        hands.close()
        face_id.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
