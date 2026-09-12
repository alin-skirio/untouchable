"""Local face recognition profiles for the hand-control tracker.

Uses OpenCV YuNet + SFace (appearance embeddings) for identity, with MediaPipe
Face Mesh kept for pose coaching and on-screen structure. Named encodings are
saved under profiles/ so they persist between runs.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import subprocess
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

PROFILES_DIR = Path(__file__).resolve().parent / "profiles"
MODELS_DIR = Path(__file__).resolve().parent / "models"
INDEX_PATH = PROFILES_DIR / "index.json"
LEGACY_DB_PATH = Path(__file__).resolve().parent / "profiles.db"
ENCODING_VERSION = 4
ENCODING_DIM = 128  # SFace feature length
ENROLL_SAMPLES_PER_STEP = 12
ENROLL_STEP_TIMEOUT = 16.0
MAX_TEMPLATES = 24
# Cosine distance on L2-normalized SFace embeddings (0 = identical).
# Strangers vs Audrey/Alex are typically ~0.8+; keep the accept ceiling well below that.
MATCH_SCORE_GLOBAL_MAX = 0.36
MATCH_MARGIN_RATIO = 0.85  # With 2+ profiles, best must be clearly better than 2nd
MATCH_CONFIRM_FRAMES = 3
MATCH_HOLD_FRAMES = 0
FACE_GONE_FRAMES = 10  # Missed-face frames before clearing a locked identity
UNLOCK_MISMATCH_FRAMES = 5  # Failed identity re-checks before dropping a locked name
PENDING_MISS_FRAMES = 8  # Allow brief encode gaps without resetting confirm streak
SMOOTH_FRAMES = 5
MAX_FACES = 4
TRACK_IOU_MIN = 0.12
MIN_EYE_DIST_PX = 40.0
LEFT_EYE = 33
RIGHT_EYE = 263
NOSE_TIP = 1
CHIN = 152
FOREHEAD = 10
MAX_MATCH_YAW = 1.6
MAX_MATCH_POSE = 0.90  # Skip ID on hard angles
SINGLE_PROFILE_CAP = 0.32  # Extra-strict when only one enrolled face exists

YUNET_NAME = "face_detection_yunet_2023mar.onnx"
SFACE_NAME = "face_recognition_sface_2021dec.onnx"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    + YUNET_NAME
)
SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
    + SFACE_NAME
)

# Guided enrollment poses.
# yaw/pitch are normalized: 0 ≈ looking straight at the camera.
# yaw < 0 ≈ turned toward user's left on this camera; pitch > 0 ≈ looking down.
ENROLL_STEPS: tuple[dict, ...] = (
    {
        "id": "center",
        "title": "Look straight at the camera",
        "hint": "Keep your face centered and still",
        "yaw": (-0.28, 0.28),
        "pitch": (-0.28, 0.28),
    },
    {
        "id": "left",
        "title": "Turn your head LEFT",
        "hint": "Slowly rotate left, then hold",
        "yaw": (-1.20, -0.18),
        "pitch": (-0.45, 0.45),
    },
    {
        "id": "right",
        "title": "Turn your head RIGHT",
        "hint": "Slowly rotate right, then hold",
        "yaw": (0.18, 1.20),
        "pitch": (-0.45, 0.45),
    },
    {
        "id": "up",
        "title": "Look UP",
        "hint": "Tilt your chin up a little, then hold",
        "yaw": (-0.45, 0.45),
        "pitch": (-1.10, -0.16),
    },
    {
        "id": "down",
        "title": "Look DOWN",
        "hint": "Tilt your chin down a little, then hold",
        "yaw": (-0.45, 0.45),
        "pitch": (0.16, 1.10),
    },
    {
        "id": "center_final",
        "title": "Look straight again",
        "hint": "Return to center and hold still",
        "yaw": (-0.28, 0.28),
        "pitch": (-0.28, 0.28),
    },
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


def face_quality(
    landmarks,
    width: int,
    height: int,
    *,
    allow_angle: bool = False,
) -> float | None:
    """Return a quality score, or None if the face is too small / unreliable."""
    lm = landmarks.landmark
    left, right = lm[LEFT_EYE], lm[RIGHT_EYE]
    eye_dist_px = math.hypot((left.x - right.x) * width, (left.y - right.y) * height)
    if eye_dist_px < MIN_EYE_DIST_PX:
        return None

    yaw = abs(left.z - right.z) / max(abs(left.x - right.x), 1e-4)
    if not allow_angle and yaw > MAX_MATCH_YAW:
        return None

    xs = [p.x for p in lm]
    ys = [p.y for p in lm]
    face_h = (max(ys) - min(ys)) * height
    if face_h < 100:
        return None

    frontal = 1.0 / (1.0 + yaw)
    return float(eye_dist_px / width * 4.0 + frontal)


def head_pose(landmarks) -> tuple[float, float]:
    """Estimate yaw/pitch where (0, 0) is roughly looking straight at the camera."""
    lm = landmarks.landmark
    left, right, nose = lm[LEFT_EYE], lm[RIGHT_EYE], lm[NOSE_TIP]
    chin = lm[CHIN]
    forehead = lm[FOREHEAD]
    eye_dist = math.hypot(left.x - right.x, left.y - right.y)
    if eye_dist < 1e-5:
        return 0.0, 0.0

    eye_mid_x = 0.5 * (left.x + right.x)
    eye_mid_y = 0.5 * (left.y + right.y)

    # This camera: negative yaw ≈ user turned toward their left.
    yaw = (nose.x - eye_mid_x) / eye_dist

    # Nose naturally sits below the eyes; measure pitch against the forehead→chin line
    # so looking straight is near 0 instead of a large positive offset.
    face_span = chin.y - forehead.y
    if abs(face_span) < 1e-5:
        pitch = 0.0
    else:
        nose_frac = (nose.y - forehead.y) / face_span
        # Empirically ~0.45–0.55 when facing the camera.
        pitch = (nose_frac - 0.50) * 3.0

    return float(yaw), float(pitch)


def pose_matches_step(yaw: float, pitch: float, step: dict) -> bool:
    y0, y1 = step["yaw"]
    p0, p1 = step["pitch"]
    return y0 <= yaw <= y1 and p0 <= pitch <= p1


def pose_guidance(yaw: float, pitch: float, step: dict) -> str:
    """Tell the user how to move into the target pose."""
    y0, y1 = step["yaw"]
    p0, p1 = step["pitch"]
    if pose_matches_step(yaw, pitch, step):
        return "Perfect — hold still…"

    tips: list[str] = []
    if yaw < y0:
        tips.append("turn more RIGHT")
    elif yaw > y1:
        tips.append("turn more LEFT")
    if pitch < p0:
        tips.append("look more DOWN")
    elif pitch > p1:
        tips.append("look more UP")

    if not tips:
        return step["hint"]
    return "Almost — " + " and ".join(tips)


def draw_enroll_coach(
    frame,
    *,
    name: str,
    step_index: int,
    step: dict,
    step_count: int,
    step_progress: float,
    overall_progress: float,
    in_pose: bool,
    message: str,
) -> None:
    """Big on-screen instructions for guided enrollment."""
    import chrome

    h, w = frame.shape[:2]
    panel_h = 92
    y1 = h - panel_h - 8
    edge = (94, 234, 212, 90) if in_pose else (255, 255, 255, 28)
    chrome.glass_panel(frame, 8, y1, w - 8, h - 8, radius=14, fill=(10, 11, 15, 210), outline=edge)
    step_label = f"{step_index + 1}/{len(ENROLL_STEPS)}  {name}".upper()
    chrome.text(frame, step_label, 18, y1 + 16, size=11, color=chrome.DIM, alpha=180, anchor="lt")
    chrome.text(frame, step["title"], 18, y1 + 34, size=16, color=chrome.TEXT, weight="medium", anchor="lt")
    chrome.text(
        frame,
        message or step["hint"],
        18,
        y1 + 56,
        size=12,
        color=chrome.TEAL if in_pose else chrome.DIM,
        alpha=220 if in_pose else 170,
        anchor="lt",
    )

    bar_x1, bar_x2 = 18, w - 18
    bar_y1, bar_y2 = y1 + 74, y1 + 78
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (48, 48, 48), -1)
    fill = int(bar_x1 + (bar_x2 - bar_x1) * max(0.0, min(1.0, step_progress)))
    if fill > bar_x1:
        cv2.rectangle(frame, (bar_x1, bar_y1), (fill, bar_y2), (94, 234, 212), -1)

    total = len(ENROLL_STEPS)
    for i in range(total):
        cx = 22 + i * 14
        cy = y1 + 8
        done = i < step_index or (i == step_index and step_progress >= 1.0)
        current = i == step_index
        color = (94, 234, 212) if done or current else (80, 80, 80)
        cv2.rectangle(frame, (cx, cy), (cx + 8, cy + 3), color, -1)

    # Direction cue chevrons
    cue = ""
    sid = step["id"]
    if sid == "left":
        cue = "<<<"
    elif sid == "right":
        cue = ">>>"
    elif sid == "up":
        cue = "^ ^ ^"
    elif sid == "down":
        cue = "v v v"
    if cue:
        import chrome

        chrome.text(
            frame,
            cue,
            w - 36,
            y1 + 48,
            size=18,
            color=chrome.TEAL,
            alpha=210,
            anchor="rt",
        )


def _ensure_model(path: Path, url: str) -> Path:
    """Download an ONNX model once if missing."""
    if path.exists() and path.stat().st_size > 10_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    print(f"Downloading {path.name} (one-time)…")
    try:
        urllib.request.urlretrieve(url, tmp)
        if not tmp.exists() or tmp.stat().st_size < 10_000:
            raise RuntimeError(f"Download failed or file too small: {url}")
        tmp.replace(path)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise
    print(f"Saved {path}")
    return path


class SFaceEncoder:
    """YuNet face detector + SFace appearance embeddings."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        yunet = _ensure_model(models_dir / YUNET_NAME, YUNET_URL)
        sface = _ensure_model(models_dir / SFACE_NAME, SFACE_URL)
        self._detector = cv2.FaceDetectorYN.create(
            str(yunet),
            "",
            (320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000,
        )
        self._recognizer = cv2.FaceRecognizerSF.create(str(sface), "")
        self._input_size = (320, 320)

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        height, width = frame_bgr.shape[:2]
        size = (width, height)
        if size != self._input_size:
            self._detector.setInputSize(size)
            self._input_size = size
        _retval, faces = self._detector.detect(frame_bgr)
        if faces is None or len(faces) == 0:
            return None
        return faces

    def encode_face(self, frame_bgr: np.ndarray, face_row: np.ndarray) -> np.ndarray | None:
        try:
            aligned = self._recognizer.alignCrop(frame_bgr, face_row)
            feat = self._recognizer.feature(aligned)
        except cv2.error:
            return None
        vec = np.asarray(feat, dtype=np.float32).reshape(-1)
        if vec.shape[0] != ENCODING_DIM:
            return None
        norm = float(np.linalg.norm(vec))
        if norm < 1e-8:
            return None
        return (vec / norm).astype(np.float32)

    def encode_near(
        self,
        frame_bgr: np.ndarray,
        anchor_xy: tuple[float, float] | None = None,
    ) -> np.ndarray | None:
        """Encode the face closest to an anchor (mesh center), or the largest face."""
        faces = self.detect(frame_bgr)
        if faces is None:
            return None
        if anchor_xy is None:
            areas = faces[:, 2] * faces[:, 3]
            idx = int(np.argmax(areas))
        else:
            centers = faces[:, 0:2] + 0.5 * faces[:, 2:4]
            anchor = np.array(anchor_xy, dtype=np.float32)
            dists = np.linalg.norm(centers - anchor[None, :], axis=1)
            idx = int(np.argmin(dists))
            # Only reject when multiple faces are present and none is near the mesh.
            if faces.shape[0] > 1:
                limit = max(float(faces[idx, 2]), float(faces[idx, 3])) * 1.75
                if float(dists[idx]) > limit:
                    return None
        return self.encode_face(frame_bgr, faces[idx])

    def list_detections(
        self,
        frame_bgr: np.ndarray,
        max_faces: int = MAX_FACES,
    ) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
        """Return [(box, face_row), ...] largest / highest-score first."""
        faces = self.detect(frame_bgr)
        if faces is None:
            return []
        # Prefer higher detector score, then larger area.
        order = np.lexsort((-faces[:, 2] * faces[:, 3], -faces[:, -1]))
        picked: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
        for idx in order[:max_faces]:
            row = faces[int(idx)]
            x, y, w, h = [float(v) for v in row[:4]]
            box = (int(x), int(y), int(x + w), int(y + h))
            picked.append((box, row))
        return picked


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def clamp_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    pad: float = 0.04,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 = int(x1 - bw * pad)
    y1 = int(y1 - bh * pad)
    x2 = int(x2 + bw * pad)
    y2 = int(y2 + bh * pad)
    return (
        max(0, x1),
        max(0, y1),
        min(width - 1, x2),
        min(height - 1, y2),
    )


@dataclass
class FaceTrack:
    track_id: int
    box: tuple[int, int, int, int]
    stable_name: str | None = None
    pending_name: str | None = None
    pending_count: int = 0
    pending_miss_frames: int = 0
    lock_mismatch_frames: int = 0
    locked_dist: float = field(default_factory=lambda: float("inf"))
    encoding_buf: deque = field(default_factory=lambda: deque(maxlen=SMOOTH_FRAMES))
    gone_frames: int = 0


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
    enc = encoding / max(float(np.linalg.norm(encoding)), 1e-8)
    norms = np.linalg.norm(mats, axis=1, keepdims=True)
    mats = mats / np.maximum(norms, 1e-8)
    sims = mats @ enc
    return 1.0 - sims


def profile_score(encoding: np.ndarray, templates: np.ndarray) -> float:
    """Distance of a live encoding to a profile (lower = closer)."""
    dists = _cosine_distances(encoding, templates)
    best = float(np.min(dists))
    k = min(3, int(dists.shape[0]))
    mean_best = float(np.mean(np.partition(dists, k - 1)[:k]))
    # Prefer the closest templates, but don't let a single lucky hit dominate.
    return 0.65 * best + 0.35 * mean_best


def compute_accept_threshold(templates: np.ndarray) -> float:
    """Learn how tight a profile should be from its own enrollment templates.

    Strangers must score worse than this person's typical self-variation.
    """
    mats = np.asarray(templates, dtype=np.float32)
    if mats.ndim == 1:
        mats = mats.reshape(1, -1)
    n = mats.shape[0]
    if n < 2:
        return 0.28

    norms = np.linalg.norm(mats, axis=1, keepdims=True)
    mats = mats / np.maximum(norms, 1e-8)
    # Pairwise cosine distances (upper triangle).
    sims = mats @ mats.T
    dists = 1.0 - sims
    iu = np.triu_indices(n, k=1)
    pairwise = dists[iu]
    if pairwise.size == 0:
        return 0.28

    p50 = float(np.percentile(pairwise, 50))
    p85 = float(np.percentile(pairwise, 85))
    thresh = 0.55 * p50 + 0.45 * p85 + 0.02
    return float(np.clip(thresh, 0.18, 0.34))


def match_profile(
    encoding: np.ndarray,
    profiles: list[tuple[str, np.ndarray, float]],
    *,
    exclude_names: set[str] | None = None,
) -> tuple[str | None, float]:
    """Open-set match using per-profile adaptive thresholds + relative margin."""
    if not profiles:
        return None, float("inf")

    excluded = {n.lower() for n in exclude_names} if exclude_names else set()

    scored: list[tuple[str, float, float]] = []
    for name, templates, accept_thresh in profiles:
        if name.lower() in excluded:
            continue
        mats = np.asarray(templates, dtype=np.float32)
        if mats.ndim == 1:
            mats = mats.reshape(1, -1)
        if mats.shape[1] != encoding.shape[0]:
            continue
        score = profile_score(encoding, mats)
        scored.append((name, score, float(accept_thresh)))

    if not scored:
        return None, float("inf")

    scored.sort(key=lambda item: item[1])
    best_name, best_score, best_thresh = scored[0]

    # Threshold relative to how many profiles are still in play.
    active_count = len(scored)
    effective_thresh = min(best_thresh, MATCH_SCORE_GLOBAL_MAX)
    if active_count == 1:
        # One enrolled person: don't treat every face as that person.
        effective_thresh = min(effective_thresh, SINGLE_PROFILE_CAP)

    if best_score > effective_thresh:
        return None, best_score

    if len(scored) >= 2:
        second_score = scored[1][1]
        if second_score <= 1e-8:
            return None, best_score
        # Best must be clearly closer than the runner-up.
        if best_score > second_score * MATCH_MARGIN_RATIO:
            return None, best_score
        if (second_score - best_score) < 0.03:
            return None, best_score

    return best_name, best_score


def profile_templates(
    profiles: list[tuple[str, np.ndarray, float]],
    name: str,
) -> tuple[np.ndarray, float] | None:
    for profile_name, templates, thresh in profiles:
        if profile_name.lower() == name.lower():
            return templates, float(thresh)
    return None


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
    import chrome

    x1, y1, x2, y2 = box
    tw, th = chrome.measure(text, 12)
    pad_x, pad_y = 8, 6
    label_h = th + pad_y * 2
    label_w = tw + pad_x * 2 + 10
    top = y1 - label_h - 4
    if top < 4:
        top = y1 + 3
    left = max(4, min(x1, frame.shape[1] - label_w - 4))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    chrome.pill(frame, left, top, text, height=label_h, pad_x=pad_x, size=12, fill=(10, 11, 15, 210))


def draw_face_structure(frame, landmarks, color=(90, 180, 255)) -> None:
    """Plot Face Mesh structure for coaching / preview overlays."""
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

    for idx, anchor_color, radius in (
        (LEFT_EYE, (40, 255, 40), 4),
        (RIGHT_EYE, (40, 255, 40), 4),
        (NOSE_TIP, (0, 200, 255), 5),
    ):
        p = pt(idx)
        if p is not None:
            cv2.circle(frame, p, radius, anchor_color, -1, cv2.LINE_AA)


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^\w\-]+", "_", name.strip(), flags=re.UNICODE).strip("_")
    return (cleaned or "profile").lower()


class FaceID:
    """Local named face profiles + live SFace recognition."""

    def __init__(self, profiles_dir: Path = PROFILES_DIR):
        self.profiles_dir = Path(profiles_dir)
        self.index_path = self.profiles_dir / "index.json"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        # (name, templates[N,D], accept_threshold)
        self.profiles: list[tuple[str, np.ndarray, float]] = []
        self._migrate_legacy_db()
        self.reload()

        self._encoder = SFaceEncoder()
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=MAX_FACES,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        self.enrolling = False
        self._enroll_name: str | None = None
        self._enroll_samples: list[np.ndarray] = []
        self._enroll_step = 0
        self._enroll_step_samples = 0
        self._enroll_step_deadline = 0.0
        self._tracks: dict[int, FaceTrack] = {}
        self._next_track_id = 1

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
        loaded: list[tuple[str, np.ndarray, float]] = []
        valid_entries: list[dict] = []
        index_dirty = False
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
            if mats.shape[1] != ENCODING_DIM:
                print(
                    f"Profile '{name}' is an old landmark format ({mats.shape[1]} dims) — "
                    "press A to re-enroll with the new SFace encoder."
                )
                continue
            norms = np.linalg.norm(mats, axis=1, keepdims=True)
            mats = mats / np.maximum(norms, 1e-8)
            mats = mats.astype(np.float32)

            thresh = compute_accept_threshold(mats)
            stored = entry.get("accept_threshold")
            if not isinstance(stored, (int, float)) or abs(float(stored) - thresh) > 1e-6:
                entry = dict(entry)
                entry["accept_threshold"] = float(thresh)
                index_dirty = True

            loaded.append((name, mats, thresh))
            valid_entries.append(entry)

        if index_dirty or len(valid_entries) != len(index.get("profiles", [])):
            self._write_index(valid_entries)

        self.profiles = sorted(loaded, key=lambda item: item[0].lower())

    def list_names(self) -> list[str]:
        return [name for name, _templates, _thresh in self.profiles]

    def delete_profile(self, name: str) -> bool:
        target = name.strip()
        if not target:
            return False
        index = self._read_index()
        kept: list[dict] = []
        removed: dict | None = None
        for entry in index.get("profiles", []):
            if str(entry.get("name") or "").lower() == target.lower():
                removed = entry
            else:
                kept.append(entry)
        if removed is None:
            return False
        file_name = str(removed.get("file") or f"{_safe_filename(target)}.npy")
        path = self.profiles_dir / file_name
        path.unlink(missing_ok=True)
        self._write_index(kept)
        self.reload()
        for track in self._tracks.values():
            if (track.stable_name or "").lower() == target.lower():
                track.stable_name = None
        print(f"Removed profile '{target}'. {len(self.profiles)} left.")
        return True

    def recognized_names(self) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        for track in self._tracks.values():
            name = track.stable_name
            if not name:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            names.append(name)
        return names

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
        self._enroll_step = 0
        self._enroll_step_samples = 0
        self._enroll_step_deadline = time.monotonic() + ENROLL_STEP_TIMEOUT
        print(f"Enrolling '{name}' with guided poses → {self.profiles_dir}")
        print("Follow the on-screen instructions (look straight, left, right, up, down).")

    def cancel_enroll(self) -> None:
        if self.enrolling:
            print("Enrollment cancelled.")
        self.enrolling = False
        self._enroll_name = None
        self._enroll_samples = []
        self._enroll_step = 0
        self._enroll_step_samples = 0

    def save_profile(self, name: str, templates: np.ndarray) -> None:
        """Persist named pose templates to disk (survives restarts)."""
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        templates = np.asarray(templates, dtype=np.float32)
        if templates.ndim == 1:
            templates = templates.reshape(1, -1)
        norms = np.linalg.norm(templates, axis=1, keepdims=True)
        templates = templates / np.maximum(norms, 1e-8)
        accept_threshold = compute_accept_threshold(templates)

        stem = _safe_filename(name)
        file_name = f"{stem}.npy"
        path = self.profiles_dir / file_name
        tmp = self.profiles_dir / f"{stem}.tmp.npy"
        np.save(tmp, templates.astype(np.float32))
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
                "accept_threshold": float(accept_threshold),
            }
        )
        self._write_index(entries)
        self.reload()

        if not any(n.lower() == name.lower() for n, _t, _th in self.profiles):
            raise RuntimeError(f"Failed to persist profile '{name}' to {path}")
        print(f"Match threshold for '{name}': {accept_threshold:.3f} (lower = stricter)")

    def _advance_enroll_step(self) -> None:
        self._enroll_step += 1
        self._enroll_step_samples = 0
        self._enroll_step_deadline = time.monotonic() + ENROLL_STEP_TIMEOUT
        if self._enroll_step >= len(ENROLL_STEPS):
            self._finish_enroll()

    def _finish_enroll(self) -> None:
        name = self._enroll_name or "unknown"
        samples = self._enroll_samples
        steps_done = self._enroll_step
        self.enrolling = False
        self._enroll_name = None
        self._enroll_samples = []
        self._enroll_step = 0
        self._enroll_step_samples = 0

        min_needed = ENROLL_SAMPLES_PER_STEP * max(3, len(ENROLL_STEPS) // 2)
        if len(samples) < min_needed:
            print(
                f"Not enough face samples ({len(samples)}). "
                "Try again and hold each pose until the bar fills."
            )
            return

        templates = select_diverse_templates(samples, MAX_TEMPLATES)
        self.save_profile(name, templates)
        print(
            f"Saved profile '{name}' "
            f"({len(samples)} frames across {min(steps_done + 1, len(ENROLL_STEPS))} poses "
            f"→ {templates.shape[0]} templates) → {self.profiles_dir}. "
            f"{len(self.profiles)} total."
        )

    def process(self, frame, rgb) -> list[str]:
        """Run face recognition on the shared camera frame. Mutates frame for overlays."""
        height, width = frame.shape[:2]
        results = self._face_mesh.process(rgb)
        status: list[str] = []

        if self.enrolling:
            step = ENROLL_STEPS[min(self._enroll_step, len(ENROLL_STEPS) - 1)]
            step_progress = self._enroll_step_samples / float(ENROLL_SAMPLES_PER_STEP)
            overall = (self._enroll_step + min(step_progress, 1.0)) / len(ENROLL_STEPS)
            in_pose = False
            message = step["hint"]

            if time.monotonic() > self._enroll_step_deadline:
                if self._enroll_step_samples >= max(5, ENROLL_SAMPLES_PER_STEP // 3):
                    self._advance_enroll_step()
                else:
                    # Give another window on the same step.
                    self._enroll_step_deadline = time.monotonic() + ENROLL_STEP_TIMEOUT
                    message = "Hold the pose longer — try again"
                if not self.enrolling:
                    return ["Enrollment complete"]
                step = ENROLL_STEPS[min(self._enroll_step, len(ENROLL_STEPS) - 1)]
                step_progress = self._enroll_step_samples / float(ENROLL_SAMPLES_PER_STEP)
                overall = (self._enroll_step + min(step_progress, 1.0)) / len(ENROLL_STEPS)

            if results.multi_face_landmarks:
                face = results.multi_face_landmarks[0]
                quality = face_quality(face, width, height, allow_angle=True)
                yaw, pitch = head_pose(face)
                in_pose = pose_matches_step(yaw, pitch, step)
                x1, y1, x2, y2 = landmark_bbox(face, width, height)
                encoding = None
                if quality is not None:
                    encoding = self._encoder.encode_near(
                        frame, ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
                    )

                draw_face_structure(frame, face)
                draw_face_label(
                    frame,
                    (x1, y1, x2, y2),
                    self._enroll_name or "Enrolling",
                    (40, 200, 120) if in_pose else (40, 180, 255),
                )

                if quality is None:
                    message = "Move closer to the camera"
                elif encoding is None:
                    message = "Hold steady…"
                elif in_pose:
                    self._enroll_samples.append(encoding)
                    self._enroll_step_samples += 1
                    message = "Perfect — hold still…"
                    step_progress = self._enroll_step_samples / float(ENROLL_SAMPLES_PER_STEP)
                    if self._enroll_step_samples >= ENROLL_SAMPLES_PER_STEP:
                        finished_name = self._enroll_name or ""
                        self._advance_enroll_step()
                        if not self.enrolling:
                            draw_enroll_coach(
                                frame,
                                name=finished_name,
                                step_index=len(ENROLL_STEPS) - 1,
                                step=ENROLL_STEPS[-1],
                                step_count=len(ENROLL_STEPS),
                                step_progress=1.0,
                                overall_progress=1.0,
                                in_pose=True,
                                message="All poses captured — profile saved!",
                            )
                            return [f"Saved profile '{finished_name}'"]
                        step = ENROLL_STEPS[self._enroll_step]
                        step_progress = 0.0
                        overall = self._enroll_step / len(ENROLL_STEPS)
                        message = step["hint"]
                else:
                    message = pose_guidance(yaw, pitch, step)
            else:
                message = "Face not found — center yourself in the frame"

            if self.enrolling:
                draw_enroll_coach(
                    frame,
                    name=self._enroll_name or "",
                    step_index=self._enroll_step,
                    step=step,
                    step_count=len(ENROLL_STEPS),
                    step_progress=step_progress,
                    overall_progress=overall,
                    in_pose=in_pose,
                    message=message,
                )
                status.append(
                    f"Enroll {self._enroll_step + 1}/{len(ENROLL_STEPS)}: {step['title']}"
                )
            return status

        return self._process_recognition(frame, results, width, height, status)

    def _associate_detections(
        self,
        detections: list[tuple[tuple[int, int, int, int], np.ndarray]],
    ) -> list[tuple[FaceTrack, tuple[int, int, int, int], np.ndarray]]:
        """Greedy IoU match of live detections to existing tracks; spawn new tracks."""
        track_ids = list(self._tracks.keys())
        unused_tracks = set(track_ids)
        unused_dets = set(range(len(detections)))
        pairs: list[tuple[float, int, int]] = []
        for tid in track_ids:
            for di, (box, _row) in enumerate(detections):
                iou = box_iou(self._tracks[tid].box, box)
                if iou >= TRACK_IOU_MIN:
                    pairs.append((iou, tid, di))
        pairs.sort(reverse=True)

        matched: list[tuple[FaceTrack, tuple[int, int, int, int], np.ndarray]] = []
        for iou, tid, di in pairs:
            if tid not in unused_tracks or di not in unused_dets:
                continue
            unused_tracks.remove(tid)
            unused_dets.remove(di)
            box, row = detections[di]
            track = self._tracks[tid]
            track.box = box
            track.gone_frames = 0
            matched.append((track, box, row))

        for tid in list(unused_tracks):
            track = self._tracks[tid]
            track.gone_frames += 1
            if track.gone_frames >= FACE_GONE_FRAMES:
                del self._tracks[tid]

        for di in unused_dets:
            box, row = detections[di]
            track = FaceTrack(track_id=self._next_track_id, box=box)
            self._next_track_id += 1
            self._tracks[track.track_id] = track
            matched.append((track, box, row))

        return matched

    def _mesh_for_box(self, results, box, width: int, height: int):
        if not results.multi_face_landmarks:
            return None
        cx, cy = box_center(box)
        best = None
        best_dist = float("inf")
        for face in results.multi_face_landmarks:
            mx1, my1, mx2, my2 = landmark_bbox(face, width, height)
            mcx, mcy = box_center((mx1, my1, mx2, my2))
            dist = math.hypot(cx - mcx, cy - mcy)
            if dist < best_dist:
                best_dist = dist
                best = face
        # Require mesh to be reasonably near the YuNet box.
        bw = max(1, box[2] - box[0])
        bh = max(1, box[3] - box[1])
        if best is None or best_dist > 0.75 * max(bw, bh):
            return None
        return best

    def _update_track(
        self,
        track: FaceTrack,
        encoding: np.ndarray | None,
        claimed_names: set[str],
    ) -> str:
        """Update one track's identity state. Returns display label."""
        if track.stable_name:
            if encoding is None:
                claimed_names.add(track.stable_name.lower())
                return track.stable_name

            locked = profile_templates(self.profiles, track.stable_name)
            still_ok = False
            if locked is not None:
                templates, thresh = locked
                unlock_thresh = min(thresh, MATCH_SCORE_GLOBAL_MAX) + 0.08
                if len(self.profiles) == 1:
                    unlock_thresh = min(unlock_thresh, SINGLE_PROFILE_CAP + 0.08)
                dist = profile_score(encoding, templates)
                still_ok = dist <= unlock_thresh
                track.locked_dist = dist

            if still_ok:
                track.lock_mismatch_frames = 0
                claimed_names.add(track.stable_name.lower())
                return track.stable_name

            track.lock_mismatch_frames += 1
            if track.lock_mismatch_frames < UNLOCK_MISMATCH_FRAMES:
                claimed_names.add(track.stable_name.lower())
                return track.stable_name

            track.stable_name = None
            track.locked_dist = float("inf")
            track.lock_mismatch_frames = 0
            track.pending_name = None
            track.pending_count = 0
            track.pending_miss_frames = 0
            track.encoding_buf.clear()

        raw_name = None
        dist = float("inf")
        if encoding is not None:
            track.encoding_buf.append(encoding)
            if len(track.encoding_buf) >= 2:
                smoothed = np.mean(np.stack(track.encoding_buf, axis=0), axis=0)
                sn = float(np.linalg.norm(smoothed)) or 1.0
                smoothed = (smoothed / sn).astype(np.float32)
            else:
                smoothed = encoding
            raw_name, dist = match_profile(
                smoothed,
                self.profiles,
                exclude_names=claimed_names,
            )

        if encoding is None:
            track.pending_miss_frames += 1
            if track.pending_miss_frames >= PENDING_MISS_FRAMES:
                track.pending_name = None
                track.pending_count = 0
        elif raw_name == track.pending_name and raw_name is not None:
            track.pending_count += 1
            track.pending_miss_frames = 0
        else:
            track.pending_name = raw_name
            track.pending_count = 1 if raw_name else 0
            track.pending_miss_frames = 0

        if track.pending_count >= MATCH_CONFIRM_FRAMES and raw_name:
            track.stable_name = raw_name
            track.locked_dist = dist
            track.lock_mismatch_frames = 0
            track.pending_miss_frames = 0
            track.encoding_buf.clear()
            claimed_names.add(raw_name.lower())
            return raw_name

        if track.stable_name:
            claimed_names.add(track.stable_name.lower())
            return track.stable_name
        return "Unknown"

    def _process_recognition(self, frame, results, width: int, height: int, status: list[str]) -> list[str]:
        detections = self._encoder.list_detections(frame, max_faces=MAX_FACES)
        if not detections:
            # Age out every track when nothing is detected.
            dead = []
            for tid, track in self._tracks.items():
                track.gone_frames += 1
                if track.gone_frames >= FACE_GONE_FRAMES:
                    dead.append(tid)
            for tid in dead:
                del self._tracks[tid]
            return status

        matched = self._associate_detections(detections)

        # Resolve locked faces first so their names stay reserved.
        matched_sorted = sorted(
            matched,
            key=lambda item: (0 if item[0].stable_name else 1, -((item[1][2] - item[1][0]) * (item[1][3] - item[1][1]))),
        )
        claimed_names: set[str] = set()
        labels: list[tuple[FaceTrack, tuple[int, int, int, int], str]] = []

        for track, box, row in matched_sorted:
            encoding = self._encoder.encode_face(frame, row)
            # Pose gate via mesh when available (skip extreme angles).
            mesh = self._mesh_for_box(results, box, width, height)
            if mesh is not None:
                yaw, pitch = head_pose(mesh)
                if abs(yaw) > MAX_MATCH_POSE or abs(pitch) > MAX_MATCH_POSE:
                    encoding = None
                if face_quality(mesh, width, height) is None:
                    encoding = None
            label = self._update_track(track, encoding, claimed_names)
            labels.append((track, box, label))

        # Draw left-to-right for stable overlays.
        labels.sort(key=lambda item: item[1][0])
        for track, box, label in labels:
            draw_box = clamp_box(box, width, height)
            mesh = self._mesh_for_box(results, box, width, height)
            if mesh is not None:
                draw_face_structure(frame, mesh)
            color = (110, 231, 183) if label != "Unknown" else (36, 191, 251)
            draw_face_label(frame, draw_box, label, color)
            if label == "Unknown":
                status.append("Face: Unknown")
            else:
                status.append(f"Face: {label}")

        return status


def prompt_name() -> str | None:
    """Terminal fallback for name entry (used only without a preview window)."""
    print("\nAdd face profile")
    try:
        name = input("Enter a name for this face (empty to cancel): ").strip()
    except EOFError:
        return None
    return name or None


class NameEntryUI:
    """On-screen text field for typing a profile name in the OpenCV window."""

    def __init__(self, max_len: int = 24):
        self.active = False
        self.text = ""
        self.max_len = max_len
        self._blink_on = True
        self._last_blink = time.monotonic()

    def open(self) -> None:
        self.active = True
        self.text = ""
        self._blink_on = True
        self._last_blink = time.monotonic()

    def close(self) -> None:
        self.active = False
        self.text = ""

    def handle_key(self, key: int) -> str | None | bool:
        """Handle a key while the name field is open.

        Returns:
            str: confirmed name
            False: cancelled
            None: still editing
        """
        if not self.active or key == 255 or key == -1:
            return None

        if key in (27,):  # Esc
            self.close()
            return False
        if key in (13, 10):  # Enter
            name = self.text.strip()
            self.close()
            return name if name else False
        if key in (8, 127):  # Backspace / Delete
            self.text = self.text[:-1]
            return None
        if 32 <= key <= 126 and len(self.text) < self.max_len:
            ch = chr(key)
            # Allow letters, numbers, space, and simple name punctuation.
            if ch.isalnum() or ch in (" ", "-", "'", "."):
                self.text += ch
            return None
        return None

    def draw(self, frame) -> None:
        if not self.active:
            return

        now = time.monotonic()
        if now - self._last_blink > 0.45:
            self._blink_on = not self._blink_on
            self._last_blink = now

        h, w = frame.shape[:2]
        box_w, box_h = min(360, w - 28), 92
        x1 = (w - box_w) // 2
        y1 = (h - box_h) // 2
        x2, y2 = x1 + box_w, y1 + box_h

        import chrome

        chrome.dim_frame(frame, 0.4)
        chrome.glass_panel(frame, x1, y1, x2, y2, radius=16, fill=(10, 11, 15, 230))
        chrome.text(frame, "Your name", x1 + 14, y1 + 16, size=12, color=chrome.TEAL, alpha=220, anchor="lt")
        chrome.text(
            frame,
            "Type a name, then press Enter",
            x1 + 14,
            y1 + 34,
            size=12,
            color=chrome.DIM,
            alpha=180,
            anchor="lt",
        )

        field_y1, field_y2 = y1 + 52, y1 + 78
        chrome.glass_panel(frame, x1 + 14, field_y1, x2 - 14, field_y2, radius=10, fill=(16, 18, 22, 220))

        caret = "|" if self._blink_on else " "
        if self.text:
            display = self.text + caret
            color = chrome.TEXT
            alpha = 230
        else:
            display = "name" + caret
            color = chrome.MUTE
            alpha = 180
        chrome.text(frame, display, x1 + 24, field_y1 + 13, size=14, color=color, alpha=alpha, anchor="lt")
