"""Live webcam face recognition with a local profile database.

Uses MediaPipe Face Mesh landmark geometry for recognition.
Press A to enroll a named profile (saved in profiles.db).
Press L to list profiles. Press Q or Esc to quit.
"""

from __future__ import annotations

import argparse
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


def connect_db(path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE COLLATE NOCASE,
            encoding BLOB NOT NULL,
            created_at REAL NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def encoding_to_blob(encoding: np.ndarray) -> bytes:
    return np.asarray(encoding, dtype=np.float32).tobytes()


def blob_to_encoding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


def load_profiles(conn: sqlite3.Connection) -> list[tuple[str, np.ndarray]]:
    rows = conn.execute("SELECT name, encoding FROM profiles ORDER BY name").fetchall()
    return [(name, blob_to_encoding(blob)) for name, blob in rows]


def save_profile(conn: sqlite3.Connection, name: str, encoding: np.ndarray) -> None:
    conn.execute(
        """
        INSERT INTO profiles (name, encoding, created_at)
        VALUES (?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            encoding = excluded.encoding,
            created_at = excluded.created_at
        """,
        (name.strip(), encoding_to_blob(encoding), time.time()),
    )
    conn.commit()


def list_profiles(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT name FROM profiles ORDER BY name").fetchall()
    return [row[0] for row in rows]


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


def landmark_bbox(landmarks, width: int, height: int, pad: float = 0.08):
    xs = [lm.x for lm in landmarks.landmark]
    ys = [lm.y for lm in landmarks.landmark]
    x1 = max(0, int((min(xs) - pad) * width))
    y1 = max(0, int((min(ys) - pad) * height))
    x2 = min(width - 1, int((max(xs) + pad) * width))
    y2 = min(height - 1, int((max(ys) + pad) * height))
    return x1, y1, x2, y2


def draw_hud(frame, lines: list[str]) -> None:
    y = 28
    for line in lines:
        cv2.putText(
            frame,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        y += 26


def prompt_name() -> str | None:
    print("\nAdd profile")
    try:
        name = input("Enter a name for this face (empty to cancel): ").strip()
    except EOFError:
        return None
    return name or None


def enroll_face(
    face_mesh,
    cap: cv2.VideoCapture,
    conn: sqlite3.Connection,
    window: str,
) -> list[tuple[str, np.ndarray]]:
    name = prompt_name()
    if not name:
        print("Enrollment cancelled.")
        return load_profiles(conn)

    print(f"Hold still and look at the camera. Capturing {ENROLL_SAMPLES} samples for '{name}'...")
    samples: list[np.ndarray] = []
    deadline = time.monotonic() + 12.0

    while len(samples) < ENROLL_SAMPLES and time.monotonic() < deadline:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        height, width = frame.shape[:2]

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            encoding = face_encoding(face)
            if encoding is not None:
                samples.append(encoding)
            x1, y1, x2, y2 = landmark_bbox(face, width, height)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 200, 120), 2)
            label = f"Enrolling {name}: {len(samples)}/{ENROLL_SAMPLES}"
        else:
            label = f"Enrolling {name}: face not found ({len(samples)}/{ENROLL_SAMPLES})"

        draw_hud(frame, [label, "Look at the camera"])
        cv2.imshow(window, frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            print("Enrollment interrupted.")
            return load_profiles(conn)

    if len(samples) < max(5, ENROLL_SAMPLES // 3):
        print(f"Not enough face samples ({len(samples)}). Try again with better lighting.")
        return load_profiles(conn)

    mean_encoding = np.mean(np.stack(samples, axis=0), axis=0).astype(np.float32)
    save_profile(conn, name, mean_encoding)
    profiles = load_profiles(conn)
    print(f"Saved profile '{name}' ({len(samples)} samples). {len(profiles)} profile(s) total.")
    return profiles


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Face ID with named profiles.")
    parser.add_argument(
        "--db",
        type=Path,
        default=DB_PATH,
        help=f"SQLite profile database path (default: {DB_PATH.name})",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        metavar="N",
        help="Force camera index N",
    )
    args = parser.parse_args()

    conn = connect_db(args.db)
    profiles = load_profiles(conn)
    print(f"Loaded {len(profiles)} profile(s) from {args.db}")
    if profiles:
        print("Profiles: " + ", ".join(name for name, _ in profiles))
    print("Controls: A = add profile, L = list profiles, Q/Esc = quit")

    cap = open_camera(preferred=args.camera)
    window = "Face ID"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=3,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame grab failed.")
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            status = [
                f"Profiles: {len(profiles)}",
                "A add  |  L list  |  Q quit",
            ]

            if results.multi_face_landmarks:
                for face in results.multi_face_landmarks:
                    encoding = face_encoding(face)
                    x1, y1, x2, y2 = landmark_bbox(face, width, height)
                    name, dist = (
                        match_profile(encoding, profiles)
                        if encoding is not None
                        else (None, float("inf"))
                    )
                    if name:
                        color = (40, 200, 120)
                        label = name
                        status.append(f"Match: {name} ({dist:.3f})")
                    else:
                        color = (40, 180, 255)
                        label = "Unknown"
                        status.append("Unknown face")

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(24, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
            else:
                status.append("No face in view")

            draw_hud(frame, status)
            cv2.imshow(window, frame)

            visible = cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE)
            if visible < 1:
                break

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("a"), ord("A")):
                profiles = enroll_face(face_mesh, cap, conn, window)
            elif key in (ord("l"), ord("L")):
                names = list_profiles(conn)
                if names:
                    print("Saved profiles: " + ", ".join(names))
                else:
                    print("No profiles saved yet. Press A to add one.")
    finally:
        face_mesh.close()
        conn.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
