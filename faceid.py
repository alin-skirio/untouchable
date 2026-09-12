"""Local face recognition profiles for the hand-control tracker.

Named encodings are saved under profiles/ so they persist between runs.
"""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

PROFILES_DIR = Path(__file__).resolve().parent / "profiles"
INDEX_PATH = PROFILES_DIR / "index.json"
LEGACY_DB_PATH = Path(__file__).resolve().parent / "profiles.db"
ENROLL_SAMPLES = 20
# L2 distance on pose-normalized mesh encodings. Higher = looser match.
MATCH_THRESHOLD = 0.55
MATCH_CONFIRM_FRAMES = 3
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


def draw_face_label(frame, box, text: str, color) -> None:
    """Draw a filled name plate on top of the face box."""
    x1, y1, x2, y2 = box
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.85
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad_x, pad_y = 10, 8
    label_h = th + pad_y * 2
    label_w = tw + pad_x * 2

    # Prefer above the box; fall back inside the top edge if needed.
    top = y1 - label_h - 4
    if top < 4:
        top = y1 + 4
    left = max(4, min(x1, frame.shape[1] - label_w - 4))

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(
        frame,
        (left, top),
        (left + label_w, top + label_h),
        color,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (left + pad_x, top + pad_y + th),
        font,
        scale,
        (20, 20, 20),
        thickness,
        cv2.LINE_AA,
    )


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


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^\w\-]+", "_", name.strip(), flags=re.UNICODE).strip("_")
    return (cleaned or "profile").lower()


class FaceID:
    """Local named face profiles + live Face Mesh recognition."""

    def __init__(self, profiles_dir: Path = PROFILES_DIR):
        self.profiles_dir = Path(profiles_dir)
        self.index_path = self.profiles_dir / "index.json"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.profiles: list[tuple[str, np.ndarray]] = []
        self._migrate_legacy_db()
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
        self._pending_name: str | None = None
        self._pending_count = 0
        self._stable_name: str | None = None

    def _read_index(self) -> dict:
        if not self.index_path.exists():
            return {"profiles": []}
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"profiles": []}
        if not isinstance(data, dict) or "profiles" not in data:
            return {"profiles": []}
        return data

    def _write_index(self, entries: list[dict]) -> None:
        payload = {"profiles": entries}
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.index_path)

    def _migrate_legacy_db(self) -> None:
        """One-time import from the old profiles.db if file profiles are empty."""
        if self.index_path.exists() or not LEGACY_DB_PATH.exists():
            return
        try:
            conn = sqlite3.connect(str(LEGACY_DB_PATH))
            rows = conn.execute("SELECT name, encoding, created_at FROM profiles").fetchall()
            conn.close()
        except sqlite3.Error:
            return
        if not rows:
            return
        entries = []
        for name, blob, created_at in rows:
            encoding = np.frombuffer(blob, dtype=np.float32).copy()
            file_name = f"{_safe_filename(name)}.npy"
            np.save(self.profiles_dir / file_name, encoding)
            entries.append(
                {
                    "name": name,
                    "file": file_name,
                    "created_at": created_at or time.time(),
                }
            )
        self._write_index(entries)
        print(f"Migrated {len(entries)} profile(s) from {LEGACY_DB_PATH.name} → {self.profiles_dir}")

    def reload(self) -> None:
        index = self._read_index()
        loaded: list[tuple[str, np.ndarray]] = []
        valid_entries: list[dict] = []
        for entry in index.get("profiles", []):
            name = entry.get("name")
            file_name = entry.get("file")
            if not name or not file_name:
                continue
            path = self.profiles_dir / file_name
            if not path.exists():
                print(f"Missing encoding file for '{name}': {path}")
                continue
            try:
                encoding = np.load(path)
            except OSError as exc:
                print(f"Could not load '{name}': {exc}")
                continue
            loaded.append((name, np.asarray(encoding, dtype=np.float32).reshape(-1)))
            valid_entries.append(entry)

        if len(valid_entries) != len(index.get("profiles", [])):
            self._write_index(valid_entries)

        self.profiles = sorted(loaded, key=lambda item: item[0].lower())

    def list_names(self) -> list[str]:
        return [name for name, _ in self.profiles]

    def close(self) -> None:
        self._face_mesh.close()

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
        print(f"Profiles are saved to {self.profiles_dir}")

    def cancel_enroll(self) -> None:
        if self.enrolling:
            print("Enrollment cancelled.")
        self.enrolling = False
        self._enroll_name = None
        self._enroll_samples = []

    def save_profile(self, name: str, encoding: np.ndarray) -> None:
        """Persist a named encoding to disk (survives restarts)."""
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        encoding = np.asarray(encoding, dtype=np.float32).reshape(-1)
        file_name = f"{_safe_filename(name)}.npy"
        path = self.profiles_dir / file_name
        tmp = self.profiles_dir / f"{_safe_filename(name)}.tmp.npy"
        np.save(tmp, encoding)
        tmp.replace(path)

        index = self._read_index()
        entries = [
            e for e in index.get("profiles", []) if e.get("name", "").lower() != name.lower()
        ]
        entries.append(
            {
                "name": name,
                "file": file_name,
                "created_at": time.time(),
            }
        )
        self._write_index(entries)
        self.reload()

        # Verify round-trip so silent disk failures are obvious.
        if not any(n.lower() == name.lower() for n, _ in self.profiles):
            raise RuntimeError(f"Failed to persist profile '{name}' to {path}")

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
        self.save_profile(name, mean_encoding)
        print(
            f"Saved profile '{name}' ({len(samples)} samples) → {self.profiles_dir}. "
            f"{len(self.profiles)} total."
        )

    def process(self, frame, rgb) -> list[str]:
        """Run face recognition on the shared camera frame. Mutates frame for overlays."""
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
                label = (
                    f"Enrolling {self._enroll_name}: "
                    f"{len(self._enroll_samples)}/{ENROLL_SAMPLES}"
                )
                draw_face_label(frame, (x1, y1, x2, y2), label, (40, 200, 120))
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
            self._pending_name = None
            self._pending_count = 0
            self._stable_name = None
            return status

        # Label the primary face with the matched profile name.
        face = results.multi_face_landmarks[0]
        encoding = face_encoding(face)
        x1, y1, x2, y2 = landmark_bbox(face, width, height)
        raw_name, _dist = (
            match_profile(encoding, self.profiles)
            if encoding is not None
            else (None, float("inf"))
        )

        if raw_name == self._pending_name and raw_name is not None:
            self._pending_count += 1
        else:
            self._pending_name = raw_name
            self._pending_count = 1 if raw_name else 0

        if self._pending_count >= MATCH_CONFIRM_FRAMES and raw_name:
            self._stable_name = raw_name
        elif raw_name is None:
            self._stable_name = None

        if self._stable_name:
            draw_face_label(frame, (x1, y1, x2, y2), self._stable_name, (40, 200, 120))
            status.append(f"Face: {self._stable_name}")
        else:
            draw_face_label(frame, (x1, y1, x2, y2), "Unknown", (40, 180, 255))
            status.append("Face: Unknown")

        return status


def prompt_name() -> str | None:
    print("\nAdd face profile")
    try:
        name = input("Enter a name for this face (empty to cancel): ").strip()
    except EOFError:
        return None
    return name or None
