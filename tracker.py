"""Headless hand/face tracking loop.

Owns the camera, MediaPipe, the gesture objects, and the macOS event output.
Runs on a worker thread and publishes the latest annotated frame so a UI can
render it. Nothing here touches OpenCV HighGUI or AppKit.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from collections import deque
from typing import Optional

import cv2
import mediapipe as mp

from faceid import FaceID, open_camera
from gestures.app_switcher import AppSwitcher
from gestures.cursor import SENSITIVITY_MAX, SENSITIVITY_MIN, PointerCursor
from gestures.landmarks import INDEX_TIP, palm_size_px
from gestures.scroll_down import ScrollDown
from gestures.scroll_up import ScrollUp
from gestures.swipe_scroller import SwipeScroller
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
)
from rim_flash import draw_pointer_bezel, flash_pointer_rim

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


def draw_hud(frame, lines: list[str]) -> None:
    height = 58 + 26 * max(len(lines), 1)
    cv2.rectangle(frame, (16, 16), (860, 16 + height), (18, 18, 18), -1)
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


class HandTracker:
    """Runs the vision loop on a worker thread and publishes annotated frames."""

    def __init__(self, camera: Optional[int] = None, draw_hud_overlay: bool = True):
        self.camera_index = camera
        self.draw_hud_overlay = draw_hud_overlay

        self.pointer = PointerCursor()
        self.switcher = AppSwitcher()
        self.scroll_up = ScrollUp()
        self.scroll_down = ScrollDown()
        self.scroller = SwipeScroller()

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._commands: "queue.Queue[tuple[str, dict]]" = queue.Queue()
        self._lock = threading.Lock()
        self._frame = None
        self._status: list[str] = []
        self._preview_enabled = False
        self._camera_lost = False
        self._error: Optional[str] = None
        self._face_names: list[str] = []

    # --- lifecycle -----------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name="hand-tracker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._thread = None

    # --- UI-facing state ------------------------------------------------

    def set_preview_enabled(self, enabled: bool) -> None:
        """Only annotate and publish frames while something is watching."""
        self._preview_enabled = enabled
        if not enabled:
            with self._lock:
                self._frame = None

    def latest_frame(self):
        with self._lock:
            if self._frame is None:
                return None
            return self._frame, list(self._status)

    def status_lines(self) -> list[str]:
        with self._lock:
            return list(self._status)

    @property
    def engaged(self) -> bool:
        return self.pointer.engaged

    @property
    def camera_lost(self) -> bool:
        return self._camera_lost

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def sensitivity(self) -> float:
        return self.pointer.base_sensitivity

    @property
    def has_reference(self) -> bool:
        return self.pointer.has_reference

    def face_names(self) -> list[str]:
        return list(self._face_names)

    def set_sensitivity(self, value: float, persist: bool = True) -> None:
        value = min(SENSITIVITY_MAX, max(SENSITIVITY_MIN, value))
        self.pointer.set_base_sensitivity(value, persist=persist)

    def persist_sensitivity(self) -> None:
        self.pointer.set_base_sensitivity(self.pointer.base_sensitivity, persist=True)

    def request(self, name: str, **kwargs) -> None:
        self._commands.put((name, kwargs))

    # --- worker ---------------------------------------------------------

    def _run(self) -> None:
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_styles = mp.solutions.drawing_styles

        hands = None
        face_id = None
        cap = None
        cmd_t = None
        try:
            hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=MAX_HANDS,
                model_complexity=1,
                min_detection_confidence=0.75,
                min_tracking_confidence=0.75,
            )
            face_id = FaceID()
            self._face_names = face_id.list_names()
            print(
                f"Loaded {len(self._face_names)} face profile(s) from {face_id.profiles_dir}"
                + (f": {', '.join(self._face_names)}" if self._face_names else "")
            )
            cap = open_camera(preferred=self.camera_index)
            cmd_t = start_cmd_t_monitor()
        except (Exception, SystemExit) as exc:
            # open_camera raises SystemExit, which would otherwise kill this
            # thread silently and leave the UI waiting forever.
            self._error = str(exc)
            self._camera_lost = True
            print(f"Tracker could not start: {exc}")
            if cmd_t is not None:
                cmd_t.stop()
            if hands is not None:
                hands.close()
            if face_id is not None:
                face_id.close()
            if cap is not None:
                cap.release()
            return

        trails: dict[str, deque[tuple[int, int]]] = {
            "Left": deque(maxlen=TRAIL_LENGTH),
            "Right": deque(maxlen=TRAIL_LENGTH),
        }
        active_finger: dict[str, str] = {}
        pending: dict[str, tuple[str, int]] = {}

        last_cmd_tab_at = 0.0
        last_scroll_at = 0.0
        last_scroll_amount = 0
        last_scroll_action = ""
        last_ref_at = 0.0
        last_ref_msg = ""
        last_right_palm = 0.0
        last_right_index: tuple[float, float] | None = None
        failed_frame_count = 0

        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    failed_frame_count += 1
                    if failed_frame_count > 15:
                        print("Lost the camera feed.")
                        self._camera_lost = True
                        break
                    time.sleep(0.03)
                    continue

                failed_frame_count = 0

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

                # Face recognition / enrollment on the shared feed (draws on frame).
                status_lines.extend(face_id.process(frame, rgb))

                if result.multi_hand_landmarks:
                    handedness_list = result.multi_handedness or []
                    for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
                        label = "Hand"
                        if i < len(handedness_list):
                            label = handedness_list[i].classification[0].label

                        palm_px = palm_size_px(hand_landmarks, width, height)
                        if palm_px < MIN_PALM_SIZE_PX and not (
                            self.pointer.engaged and label == "Right"
                        ):
                            continue
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

                        if label == "Left":
                            saw_left = True
                            up_lines = self.scroll_up.update(hand_landmarks, down)
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

                            was_pointing = self.pointer.engaged
                            cursor = self.pointer.update(hand_landmarks, down, (width, height))
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
                                self.scroll_down.reset()
                                self.scroller.reset()
                                if cursor.click == "left":
                                    status_lines.append("Pointer · left click")
                                elif cursor.click == "right":
                                    status_lines.append("Pointer · right click")
                                else:
                                    status_lines.append("Pointer")
                            elif was_pointing:
                                self.scroll_down.reset()
                                self.scroller.reset()
                            else:
                                status_lines.append(self.pointer.arm_hint)
                                started, tapped, ended = self.switcher.update(hand_landmarks)

                                if started:
                                    async_cmd(True)
                                    async_tap_tab()
                                    last_cmd_tab_at = time.monotonic()
                                elif tapped:
                                    async_tap_tab()
                                    last_cmd_tab_at = time.monotonic()
                                elif ended:
                                    async_cmd(False)

                                if self.switcher.active:
                                    status_lines.append(
                                        "Right pinch: App Switcher Active (⌘ Held)"
                                    )

                                down_lines = self.scroll_down.update(hand_landmarks, down)
                                if down_lines > 0:
                                    threading.Thread(
                                        target=smooth_scroll, args=(-down_lines,), daemon=True
                                    ).start()
                                    last_scroll_at = time.monotonic()
                                    last_scroll_amount = down_lines
                                    last_scroll_action = "down"
                                else:
                                    scroll_amount = self.scroller.update(hand_landmarks, down)
                                    if scroll_amount > 0:
                                        threading.Thread(
                                            target=smooth_scroll,
                                            args=(-scroll_amount,),
                                            daemon=True,
                                        ).start()
                                        last_scroll_at = time.monotonic()
                                        last_scroll_amount = scroll_amount
                                        last_scroll_action = "flick"

                if not saw_left:
                    self.scroll_up.reset()

                if not saw_right:
                    if self.pointer.engaged:
                        self.pointer.reset()
                        flash_pointer_rim(False)
                    if self.switcher.active:
                        self.switcher.reset()
                        async_cmd(False)
                    self.scroll_down.reset()
                    self.scroller.reset()

                for label, trail in trails.items():
                    if label not in seen:
                        trail.clear()
                        active_finger.pop(label, None)
                        pending.pop(label, None)

                if cmd_t.consume():
                    if self.switcher.active:
                        pass
                    elif saw_right and last_right_palm >= MIN_PALM_SIZE_PX:
                        saved = self.pointer.set_reference(
                            last_right_palm, frame, (width, height)
                        )
                        last_ref_msg = (
                            "Reference captured" if saved else "Could not save reference"
                        )
                        last_ref_at = time.monotonic()
                        print(last_ref_msg)
                    else:
                        last_ref_msg = "No hand for reference"
                        last_ref_at = time.monotonic()
                        print(last_ref_msg)

                message = self._drain_commands(
                    face_id, frame, (width, height), saw_right, last_right_palm, last_right_index
                )
                if message:
                    last_ref_msg = message
                    last_ref_at = time.monotonic()

                if time.monotonic() - last_cmd_tab_at < 0.8:
                    status_lines.append("Sent Tab")
                if time.monotonic() - last_scroll_at < 1.0:
                    if last_scroll_action == "up":
                        status_lines.append(f"Scroll up ({abs(last_scroll_amount)} lines)")
                    elif last_scroll_action == "flick":
                        status_lines.append(f"Flick scroll ({abs(last_scroll_amount)} lines)")
                    else:
                        status_lines.append(f"Scroll down ({abs(last_scroll_amount)} lines)")
                if not seen:
                    status_lines.append("No hands in view")
                if self.pointer.has_reference:
                    status_lines.append("⌘T reference set")
                else:
                    status_lines.append("⌘T to set reference")
                status_lines.append(
                    f"Dist {self.pointer.distance_ratio(last_right_palm):.1f}x"
                )
                if last_ref_msg and time.monotonic() - last_ref_at < 1.5:
                    status_lines.append(last_ref_msg)

                if self._preview_enabled:
                    if self.draw_hud_overlay:
                        draw_hud(frame, status_lines)
                        draw_pointer_bezel(frame, self.pointer.engaged)
                    with self._lock:
                        self._frame = frame
                        self._status = status_lines
                else:
                    with self._lock:
                        self._status = status_lines
        finally:
            release_mouse()
            set_cmd_state(False)
            if cmd_t is not None:
                cmd_t.stop()
            hands.close()
            face_id.close()
            cap.release()
            self._running = False

    def _drain_commands(
        self,
        face_id: FaceID,
        frame,
        frame_size: tuple[int, int],
        saw_right: bool,
        last_right_palm: float,
        last_right_index: tuple[float, float] | None,
    ) -> Optional[str]:
        """Apply queued UI commands on the worker thread. Returns a status message."""
        message: Optional[str] = None
        while True:
            try:
                name, kwargs = self._commands.get_nowait()
            except queue.Empty:
                break

            if name == "set_reference":
                if self.switcher.active:
                    continue
                if saw_right and last_right_palm >= MIN_PALM_SIZE_PX:
                    saved = self.pointer.set_reference(last_right_palm, frame, frame_size)
                    message = "Reference captured" if saved else "Could not save reference"
                else:
                    message = "No hand for reference"
                print(message)

            elif name == "place_cursor":
                if self.switcher.active:
                    continue
                if saw_right and last_right_index is not None:
                    ox, oy, sw, sh = display_bounds()
                    if sw > 0 and sh > 0:
                        nx, ny = last_right_index
                        set_mouse_position(ox + nx * sw, oy + ny * sh)
                        was_engaged = self.pointer.engaged
                        self.pointer.engage_from_s()
                        self.scroll_up.reset()
                        self.scroll_down.reset()
                        self.scroller.reset()
                        if not was_engaged:
                            flash_pointer_rim(True)
                        message = "Cursor on index tip"
                    else:
                        message = "Could not place cursor"
                else:
                    message = "No hand to place cursor"

            elif name == "add_face":
                profile = (kwargs.get("name") or "").strip()
                if not profile:
                    message = "Face name was empty"
                elif face_id.enrolling:
                    message = "Already enrolling a face"
                else:
                    face_id.begin_enroll(profile)
                    message = f"Enrolling {profile}"

            elif name == "cancel_face":
                if face_id.enrolling:
                    face_id.cancel_enroll()
                    message = "Enrollment cancelled"

            elif name == "list_faces":
                names = face_id.list_names()
                self._face_names = names
                if names:
                    print("Saved face profiles: " + ", ".join(names))
                else:
                    print("No face profiles yet. Add one from the panel.")

        return message
