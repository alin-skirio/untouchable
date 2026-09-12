"""Local face recognition profiles for the hand-control tracker.

Named encodings are saved under profiles/ so they persist between runs.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import subprocess
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

PROFILES_DIR = Path(__file__).resolve().parent / "profiles"
INDEX_PATH = PROFILES_DIR / "index.json"
LEGACY_DB_PATH = Path(__file__).resolve().parent / "profiles.db"
ENCODING_VERSION = 3
ENROLL_SAMPLES = 60
ENROLL_SECONDS = 20.0
MAX_TEMPLATES = 16
# Cosine distance (0 = identical). Lower = stricter.
MATCH_THRESHOLD = 0.12
MATCH_MARGIN = 0.025  # Best person must beat 2nd-best by this much
MATCH_CONFIRM_FRAMES = 3
MATCH_HOLD_FRAMES = 20
SMOOTH_FRAMES = 5
MIN_EYE_DIST_PX = 42.0
LEFT_EYE = 33
RIGHT_EYE = 263
NOSE_TIP = 1
CHIN = 152

# Identity-heavy landmarks (skip highly expressive mouth interior).
_ENCODING_LANDMARKS = (
    # brows
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
    # eyes
    33, 133, 160, 159, 158, 144, 153, 154, 155, 145,
    362, 263, 387, 386, 385, 373, 380, 381, 382, 374,
    # nose
    1, 2, 98, 327, 168, 6, 197, 195, 5, 4,
    # cheeks / mid-face
    50, 101, 205, 280, 330, 425,
    # jaw / face oval (coarse identity + yaw cue)
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
)

# Higher weight = more influence on identity match (eyes/nose > jaw).
_LANDMARK_WEIGHTS = np.array(
    (
        # brows (10)
        1.3, 1.3, 1.4, 1.3, 1.3, 1.3, 1.3, 1.4, 1.3, 1.3,
        # eyes (20)
        1.8, 1.6, 1.7, 1.7, 1.7, 1.6, 1.6, 1.6, 1.6, 1.6,
        1.8, 1.6, 1.7, 1.7, 1.7, 1.6, 1.6, 1.6, 1.6, 1.6,
        # nose (10)
        1.7, 1.5, 1.5, 1.5, 1.6, 1.6, 1.5, 1.5, 1.5, 1.6,
        # cheeks (6)
        1.2, 1.2, 1.1, 1.2, 1.2, 1.1,
        # jaw / oval (36)
        *([0.75] * 36),
    ),
    dtype=np.float32,
)

# Polylines used only for on-screen structure visualization.
_STRUCTURE_PATHS: tuple[tuple[tuple[int, ...], tuple[int, int, int]], ...] = (
    # left brow, right brow
    ((70, 63, 105, 66, 107), (80, 200, 255)),
    ((336, 296, 334, 293, 300), (80, 200, 255)),
    # left eye, right eye
    ((33, 160, 158, 133, 153, 144, 33), (60, 230, 160)),
    ((362, 385, 387, 263, 373, 380, 362), (60, 230, 160)),
    # nose bridge + tip / wings
    ((168, 6, 197, 195, 5, 4, 1), (255, 180, 80)),
    ((98, 2, 327), (255, 180, 80)),
    # cheeks
    ((50, 101, 205), (200, 140, 255)),
    ((280, 330, 425), (200, 140, 255)),
    # face oval / jaw
    (
        (
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10,
        ),
        (90, 180, 255),
    ),
)

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


def face_quality(landmarks, width: int, height: int) -> float | None:
    """Return a quality score, or None if the face is too small / unreliable."""
    lm = landmarks.landmark
    left, right = lm[LEFT_EYE], lm[RIGHT_EYE]
    eye_dist_px = math.hypot((left.x - right.x) * width, (left.y - right.y) * height)
    if eye_dist_px < MIN_EYE_DIST_PX:
        return None

    # Reject extreme profile views where one eye is far behind the other.
    yaw = abs(left.z - right.z) / max(abs(left.x - right.x), 1e-4)
    if yaw > 2.8:
        return None

    xs = [p.x for p in lm]
    ys = [p.y for p in lm]
    face_h = (max(ys) - min(ys)) * height
    if face_h < 90:
        return None

    # Prefer larger, more frontal faces.
    frontal = 1.0 / (1.0 + yaw)
    return float(eye_dist_px / width * 4.0 + frontal)


def face_encoding(landmarks) -> np.ndarray | None:
    """Build a pose-normalized, weighted, L2-normalized identity vector."""
    all_pts = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark], dtype=np.float32)
    if all_pts.shape[0] < 468:
        return None

    left = all_pts[LEFT_EYE]
    right = all_pts[RIGHT_EYE]
    nose = all_pts[NOSE_TIP]
    chin = all_pts[CHIN] if all_pts.shape[0] > CHIN else nose
    eye_mid = 0.5 * (left + right)
    eye_dist = float(np.linalg.norm(left[:2] - right[:2]))
    if eye_dist < 1e-5:
        return None

    idx = np.array(_ENCODING_LANDMARKS, dtype=np.int32)
    idx = idx[idx < all_pts.shape[0]]
    pts = all_pts[idx]

    # Center on eye midpoint (more stable than nose alone), level the eyes.
    centered = pts - eye_mid
    angle = np.arctan2(right[1] - left[1], right[0] - left[0])
    cos_a, sin_a = float(np.cos(-angle)), float(np.sin(-angle))
    rot = np.array(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    aligned = centered @ rot.T

    # Scale by inter-ocular distance and mild face-height cue.
    face_h = float(np.linalg.norm((chin - eye_mid)[:2])) or eye_dist
    scale = 0.65 * eye_dist + 0.35 * face_h
    aligned /= max(scale, 1e-5)

    weights = _LANDMARK_WEIGHTS[: aligned.shape[0], None]
    weighted = aligned * weights
    vec = weighted.reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return None
    return (vec / norm).astype(np.float32)


def select_diverse_templates(
    samples: list[np.ndarray],
    max_templates: int = MAX_TEMPLATES,
) -> np.ndarray:
    """Pick a diverse subset of encodings so off-angle poses are represented."""
    stacked = np.stack(samples, axis=0).astype(np.float32)
    # Always keep the mean as a strong frontal-ish prior.
    mean_vec = stacked.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean_vec)) or 1.0
    mean_vec = (mean_vec / mean_norm).astype(np.float32)

    if stacked.shape[0] <= max_templates - 1:
        return np.vstack([mean_vec[None, :], stacked])

    start = int(np.argmin(np.linalg.norm(stacked - mean_vec[None, :], axis=1)))
    chosen = [start]
    while len(chosen) < max_templates - 1:
        dists = np.min(
            np.linalg.norm(stacked[:, None, :] - stacked[chosen][None, :, :], axis=2),
            axis=1,
        )
        dists[chosen] = -1.0
        chosen.append(int(np.argmax(dists)))
    return np.vstack([mean_vec[None, :], stacked[chosen]])


def _cosine_distances(encoding: np.ndarray, templates: np.ndarray) -> np.ndarray:
    mats = np.asarray(templates, dtype=np.float32)
    if mats.ndim == 1:
        mats = mats.reshape(1, -1)
    # Encodings are L2-normalized; still re-normalize for safety.
    enc = encoding / max(float(np.linalg.norm(encoding)), 1e-8)
    norms = np.linalg.norm(mats, axis=1, keepdims=True)
    mats = mats / np.maximum(norms, 1e-8)
    sims = mats @ enc
    return 1.0 - sims


def match_profile(
    encoding: np.ndarray,
    profiles: list[tuple[str, np.ndarray]],
    threshold: float = MATCH_THRESHOLD,
) -> tuple[str | None, float]:
    """Match using mean of best template hits + a margin against the runner-up."""
    if not profiles:
        return None, float("inf")

    scored: list[tuple[str, float]] = []
    for name, templates in profiles:
        mats = np.asarray(templates, dtype=np.float32)
        if mats.ndim == 1:
            mats = mats.reshape(1, -1)
        if mats.shape[1] != encoding.shape[0]:
            continue
        dists = _cosine_distances(encoding, mats)
        k = min(3, dists.shape[0])
        score = float(np.mean(np.partition(dists, k - 1)[:k]))
        scored.append((name, score))

    if not scored:
        return None, float("inf")

    scored.sort(key=lambda item: item[1])
    best_name, best_dist = scored[0]
    second = scored[1][1] if len(scored) > 1 else best_dist + 1.0

    if best_dist > threshold:
        return None, best_dist
    if len(scored) > 1 and (second - best_dist) < MATCH_MARGIN and best_dist > threshold * 0.55:
        # Ambiguous between two people — don't guess.
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


def draw_face_label(frame, box, text: str, color) -> None:
    """Draw a filled name plate on top of the face box."""
    x1, y1, x2, y2 = box
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.85
    thickness = 2
    (tw, th), _baseline = cv2.getTextSize(text, font, scale, thickness)
    pad_x, pad_y = 10, 8
    label_h = th + pad_y * 2
    label_w = tw + pad_x * 2

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


def draw_face_structure(frame, landmarks, color=(90, 180, 255)) -> None:
    """Plot the landmark structures used for recognition encodings."""
    height, width = frame.shape[:2]
    lm = landmarks.landmark
    n = len(lm)

    def pt(idx: int) -> tuple[int, int] | None:
        if idx >= n:
            return None
        return int(lm[idx].x * width), int(lm[idx].y * height)

    for path, path_color in _STRUCTURE_PATHS:
        points = [pt(i) for i in path]
        for a, b in zip(points, points[1:]):
            if a is None or b is None:
                continue
            cv2.line(frame, a, b, path_color, 1, cv2.LINE_AA)

    # Anchor points used for pose normalization.
    for idx, anchor_color, radius in (
        (LEFT_EYE, (40, 255, 40), 4),
        (RIGHT_EYE, (40, 255, 40), 4),
        (NOSE_TIP, (0, 200, 255), 5),
    ):
        p = pt(idx)
        if p is not None:
            cv2.circle(frame, p, radius, anchor_color, -1, cv2.LINE_AA)

    for idx in _ENCODING_LANDMARKS:
        p = pt(idx)
        if p is None:
            continue
        cv2.circle(frame, p, 2, color, -1, cv2.LINE_AA)


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
        self._hold_frames = 0
        self._encoding_buf: deque[np.ndarray] = deque(maxlen=SMOOTH_FRAMES)

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
            mats = np.asarray(encoding, dtype=np.float32)
            if mats.ndim == 1:
                mats = mats.reshape(1, -1)
            expected = len(_ENCODING_LANDMARKS) * 3
            if mats.shape[1] != expected:
                print(
                    f"Profile '{name}' is an old format ({mats.shape[1]} dims) — "
                    "press A to re-enroll for better accuracy."
                )
                continue
            # Re-normalize templates in case they were saved pre-normalization.
            norms = np.linalg.norm(mats, axis=1, keepdims=True)
            mats = mats / np.maximum(norms, 1e-8)
            loaded.append((name, mats.astype(np.float32)))
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
        self._enroll_deadline = time.monotonic() + ENROLL_SECONDS
        print(f"Enrolling '{name}' — slowly turn your head left/right while facing the camera.")
        print(f"Capturing up to {ENROLL_SAMPLES} pose samples ({int(ENROLL_SECONDS)}s)...")
        print(f"Profiles are saved to {self.profiles_dir}")

    def cancel_enroll(self) -> None:
        if self.enrolling:
            print("Enrollment cancelled.")
        self.enrolling = False
        self._enroll_name = None
        self._enroll_samples = []

    def save_profile(self, name: str, templates: np.ndarray) -> None:
        """Persist named pose templates to disk (survives restarts)."""
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        templates = np.asarray(templates, dtype=np.float32)
        if templates.ndim == 1:
            templates = templates.reshape(1, -1)
        file_name = f"{_safe_filename(name)}.npy"
        path = self.profiles_dir / file_name
        tmp = self.profiles_dir / f"{_safe_filename(name)}.tmp.npy"
        np.save(tmp, templates)
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
                "templates": int(templates.shape[0]),
                "encoding_version": ENCODING_VERSION,
            }
        )
        self._write_index(entries)
        self.reload()

        if not any(n.lower() == name.lower() for n, _ in self.profiles):
            raise RuntimeError(f"Failed to persist profile '{name}' to {path}")

    def _finish_enroll(self) -> None:
        name = self._enroll_name or "unknown"
        samples = self._enroll_samples
        self.enrolling = False
        self._enroll_name = None
        self._enroll_samples = []

        if len(samples) < max(8, ENROLL_SAMPLES // 4):
            print(f"Not enough face samples ({len(samples)}). Try again.")
            return

        templates = select_diverse_templates(samples, MAX_TEMPLATES)
        self.save_profile(name, templates)
        print(
            f"Saved profile '{name}' ({len(samples)} frames → {templates.shape[0]} pose templates) "
            f"→ {self.profiles_dir}. {len(self.profiles)} total."
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
                quality = face_quality(face, width, height)
                encoding = face_encoding(face) if quality is not None else None
                if encoding is not None:
                    self._enroll_samples.append(encoding)
                x1, y1, x2, y2 = landmark_bbox(face, width, height)
                draw_face_structure(frame, face)
                label = (
                    f"Enrolling {self._enroll_name}: "
                    f"{len(self._enroll_samples)}/{ENROLL_SAMPLES}"
                )
                if quality is None:
                    label += " (move closer)"
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
            self._hold_frames = 0
            self._encoding_buf.clear()
            return status

        # Label the primary face with the matched profile name.
        face = results.multi_face_landmarks[0]
        x1, y1, x2, y2 = landmark_bbox(face, width, height)
        draw_face_structure(frame, face)

        quality = face_quality(face, width, height)
        encoding = face_encoding(face) if quality is not None else None
        raw_name = None
        dist = float("inf")
        if encoding is not None:
            self._encoding_buf.append(encoding)
            if len(self._encoding_buf) >= 2:
                smoothed = np.mean(np.stack(self._encoding_buf, axis=0), axis=0)
                sn = float(np.linalg.norm(smoothed)) or 1.0
                smoothed = (smoothed / sn).astype(np.float32)
            else:
                smoothed = encoding
            raw_name, dist = match_profile(smoothed, self.profiles)

        if raw_name == self._pending_name and raw_name is not None:
            self._pending_count += 1
        else:
            self._pending_name = raw_name
            self._pending_count = 1 if raw_name else 0

        if self._pending_count >= MATCH_CONFIRM_FRAMES and raw_name:
            self._stable_name = raw_name
            self._hold_frames = MATCH_HOLD_FRAMES
        elif raw_name is None and self._hold_frames > 0:
            self._hold_frames -= 1
        else:
            if raw_name is None:
                self._stable_name = None
                self._hold_frames = 0

        if self._stable_name:
            draw_face_label(frame, (x1, y1, x2, y2), self._stable_name, (40, 200, 120))
            status.append(f"Face: {self._stable_name} ({dist:.3f})" if dist < 1e9 else f"Face: {self._stable_name}")
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
