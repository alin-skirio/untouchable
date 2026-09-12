"""Local face recognition profiles for the hand-control tracker.

Stores named face encodings in profiles.db and matches live Face Mesh landmarks.
"""

from __future__ import annotations

import re
import sqlite3
import subprocess
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

DB_PATH = Path(__file__).resolve().parent / "profiles.db"
ENROLL_SAMPLES = 20
MATCH_THRESHOLD = 0.085
LEFT_EYE = 33
RIGHT_EYE = 263
NOSE_TIP = 1

_CAMERA_AVOID = ("iphone", "ipad", "continuity", "apple vision")
_CAMERA_PREFER = ("facetime", "built-in", "macbook", "imac", "studio display")


def _camera_score(name: str) -> int:
    lower = name.lower()
    if any(token in lower for token in _CAMERA_AVOID):
        return -100
    if any(token in lower for token in _CAMERA_PREFER):
        return 100
    return 0


def _list_macos_cameras() -> list[tuple[int, str]]:
    script = (
        "import AVFoundation\n"
        "let devices = AVCaptureDevice.devices(for: .video)\n"
        "for (i, d) in devices.enumerated() {\n"
        '  print("\\(i)\\t\\(d.localizedName)")\n'
        "}\n"
    )
    try:
        proc = subprocess.run(
            ["swift", "-e", script],
            capture_output=True,
            text=True,
            timeout=45,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    cameras: list[tuple[int, str]] = []
    for line in proc.stdout.splitlines():
        match = re.match(r"^(\d+)\t(.*)$", line.strip())
        if match:
            cameras.append((int(match.group(1)), match.group(2).strip()))
    return cameras


def _camera_try_order(preferred: int | None = None) -> list[tuple[int, str]]:
    if preferred is not None:
        return [(preferred, f"index {preferred}")]

    named = _list_macos_cameras()
    if named:
        ranked = sorted(named, key=lambda item: _camera_score(item[1]), reverse=True)
        preferred_only = [item for item in ranked if _camera_score(item[1]) >= 0]
        return preferred_only or ranked

    # Continuity Camera often steals index 0; try built-in candidates first.
    return [(idx, f"index {idx}") for idx in (1, 0, 2, 3)]


def open_camera(preferred: int | None = None) -> cv2.VideoCapture:
    """Open the Mac built-in camera when possible; avoid Continuity Camera / iPhone."""
    tried: list[str] = []
    for idx, name in _camera_try_order(preferred):
        if _camera_score(name) < 0 and preferred is None:
            tried.append(f"{idx}:{name} (skipped)")
            continue

        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                print(f"Connected to camera {idx}: {name}")
                return cap
            cap.release()
        tried.append(f"{idx}:{name}")

    detail = ", ".join(tried) if tried else "none found"
    raise SystemExit(
        "Could not open the Mac camera.\n"
        f"Tried: {detail}\n"
        "1. Close apps using the camera (Zoom, FaceTime, browsers).\n"
        "2. On macOS: System Settings → Privacy & Security → Camera, then allow "
        "Terminal (or Cursor) and run again.\n"
        "3. Or pass --camera N to force an index."
    )


def face_encoding(landmarks) -> np.ndarray | None:
    """Build a pose-normalized encoding from Face Mesh landmarks."""
    pts = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark], dtype=np.float32)
    if pts.shape[0] < 468:
        return None

    left = pts[LEFT_EYE]
    right = pts[RIGHT_EYE]
    nose = pts[NOSE_TIP]
    eye_dist = float(np.linalg.norm(left - right))
    if eye_dist < 1e-5:
        return None

    centered = pts - nose
    angle = np.arctan2(right[1] - left[1], right[0] - left[0])
    cos_a, sin_a = float(np.cos(-angle)), float(np.sin(-angle))
    rot = np.array(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    aligned = centered @ rot.T
    aligned /= eye_dist
    return aligned.reshape(-1)


def landmark_bbox(landmarks, width: int, height: int, pad: float = 0.08):
    xs = [lm.x for lm in landmarks.landmark]
    ys = [lm.y for lm in landmarks.landmark]
    x1 = max(0, int((min(xs) - pad) * width))
    y1 = max(0, int((min(ys) - pad) * height))
    x2 = min(width - 1, int((max(xs) + pad) * width))
    y2 = min(height - 1, int((max(ys) + pad) * height))
    return x1, y1, x2, y2


def match_profile(
    encoding: np.ndarray,
    profiles: list[tuple[str, np.ndarray]],
    threshold: float = MATCH_THRESHOLD,
) -> tuple[str | None, float]:
    if not profiles:
        return None, float("inf")

    best_name = None
    best_dist = float("inf")
    for name, known in profiles:
        if known.shape != encoding.shape:
            continue
        dist = float(np.linalg.norm(encoding - known))
        if dist < best_dist:
            best_dist = dist
            best_name = name

    if best_name is None or best_dist > threshold:
        return None, best_dist
    return best_name, best_dist


class FaceID:
    """Local named face profiles + live Face Mesh recognition."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                encoding BLOB NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self.conn.commit()
        self.profiles: list[tuple[str, np.ndarray]] = []
        self.reload()

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        self.enrolling = False
        self._enroll_name: str | None = None
        self._enroll_samples: list[np.ndarray] = []
        self._enroll_deadline = 0.0

    def reload(self) -> None:
        rows = self.conn.execute(
            "SELECT name, encoding FROM profiles ORDER BY name"
        ).fetchall()
        self.profiles = [
            (name, np.frombuffer(blob, dtype=np.float32).copy()) for name, blob in rows
        ]

    def list_names(self) -> list[str]:
        return [name for name, _ in self.profiles]

    def close(self) -> None:
        self._face_mesh.close()
        self.conn.close()

    def begin_enroll(self, name: str) -> None:
        name = name.strip()
        if not name:
            print("Enrollment cancelled (empty name).")
            return
        self.enrolling = True
        self._enroll_name = name
        self._enroll_samples = []
        self._enroll_deadline = time.monotonic() + 12.0
        print(f"Enrolling '{name}' — look at the camera ({ENROLL_SAMPLES} samples)...")

    def cancel_enroll(self) -> None:
        if self.enrolling:
            print("Enrollment cancelled.")
        self.enrolling = False
        self._enroll_name = None
        self._enroll_samples = []

    def _finish_enroll(self) -> None:
        name = self._enroll_name or "unknown"
        samples = self._enroll_samples
        self.enrolling = False
        self._enroll_name = None
        self._enroll_samples = []

        if len(samples) < max(5, ENROLL_SAMPLES // 3):
            print(f"Not enough face samples ({len(samples)}). Try again.")
            return

        mean_encoding = np.mean(np.stack(samples, axis=0), axis=0).astype(np.float32)
        self.conn.execute(
            """
            INSERT INTO profiles (name, encoding, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                encoding = excluded.encoding,
                created_at = excluded.created_at
            """,
            (name, mean_encoding.tobytes(), time.time()),
        )
        self.conn.commit()
        self.reload()
        print(f"Saved profile '{name}' ({len(samples)} samples). {len(self.profiles)} total.")

    def process(self, frame, rgb) -> list[str]:
        """Run face recognition on the shared camera frame. Mutates frame for overlays.

        Returns short status lines for the tracker HUD.
        """
        height, width = frame.shape[:2]
        results = self._face_mesh.process(rgb)
        status: list[str] = []

        if self.enrolling:
            if time.monotonic() > self._enroll_deadline:
                self._finish_enroll()
            elif results.multi_face_landmarks:
                face = results.multi_face_landmarks[0]
                encoding = face_encoding(face)
                if encoding is not None:
                    self._enroll_samples.append(encoding)
                x1, y1, x2, y2 = landmark_bbox(face, width, height)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 200, 120), 2)
                label = (
                    f"Enrolling {self._enroll_name}: "
                    f"{len(self._enroll_samples)}/{ENROLL_SAMPLES}"
                )
                cv2.putText(
                    frame,
                    label,
                    (x1, max(24, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (40, 200, 120),
                    2,
                    cv2.LINE_AA,
                )
                status.append(label)
                if len(self._enroll_samples) >= ENROLL_SAMPLES:
                    self._finish_enroll()
            else:
                status.append(
                    f"Enrolling {self._enroll_name}: face not found "
                    f"({len(self._enroll_samples)}/{ENROLL_SAMPLES})"
                )
            return status

        if not results.multi_face_landmarks:
            return status

        for face in results.multi_face_landmarks:
            encoding = face_encoding(face)
            x1, y1, x2, y2 = landmark_bbox(face, width, height)
            name, dist = (
                match_profile(encoding, self.profiles)
                if encoding is not None
                else (None, float("inf"))
            )
            if name:
                color = (40, 200, 120)
                label = name
                status.append(f"Face: {name}")
            else:
                color = (40, 180, 255)
                label = "Unknown"
                status.append("Face: Unknown")

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, max(24, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2,
                cv2.LINE_AA,
            )

        return status


def prompt_name() -> str | None:
    print("\nAdd face profile")
    try:
        name = input("Enter a name for this face (empty to cancel): ").strip()
    except EOFError:
        return None
    return name or None
