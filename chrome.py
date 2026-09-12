"""Quiet-luxury drawing kit for camera chrome and shared overlay tokens.

Camera frames stay in OpenCV; text and glass are PIL + alpha, never Hershey.
"""

from __future__ import annotations

import math
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont

    HAS_PIL = True
except ImportError:  # pragma: no cover
    Image = ImageDraw = ImageFont = None  # type: ignore
    HAS_PIL = False
    print("Pillow is missing. Run:  pip install pillow")

ROOT = Path(__file__).resolve().parent

# Hex tokens from the redesign. OpenCV is BGR.
BASE = (15, 11, 10)  # #0a0b0f
TEAL = (212, 234, 94)  # #5eead4
VIOLET = (250, 139, 167)  # #a78bfa
GREEN = (183, 231, 110)  # #6ee7b7
AMBER = (36, 191, 251)  # #fbbf24
ROSE = (133, 113, 251)  # #fb7185
WHITE = (255, 255, 255)
TEXT = (230, 230, 230)  # ~rgba(255,255,255,0.90)
DIM = (122, 122, 122)  # ~0.48
MUTE = (71, 71, 71)  # ~0.28

HEX = {
    "base": "#0a0b0f",
    "text": "rgba(255,255,255,0.90)",
    "dim": "rgba(255,255,255,0.48)",
    "mute": "rgba(255,255,255,0.28)",
    "teal": "#5eead4",
    "violet": "#a78bfa",
    "green": "#6ee7b7",
    "amber": "#fbbf24",
    "rose": "#fb7185",
}

HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
)
HAND_TIPS = (4, 8, 12, 16, 20)

_FONT_CANDIDATES = (
    ROOT / "fonts" / "Inter-Regular.ttf",
    Path("/usr/share/fonts/truetype/macos/Inter-Regular.ttf"),
    Path("/usr/share/fonts/truetype/macos/Inter-Medium.ttf"),
    Path("/System/Library/Fonts/SFNS.ttf"),
    Path("/System/Library/Fonts/SFNSText.ttf"),
    Path("/Library/Fonts/SF-Pro-Text-Regular.otf"),
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("/usr/share/fonts/truetype/inter/Inter-Regular.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
)

_FONT_MEDIUM_CANDIDATES = (
    Path("/usr/share/fonts/truetype/macos/Inter-Medium.ttf"),
    Path("/usr/share/fonts/truetype/macos/Inter-SemiBold.ttf"),
    Path("/Library/Fonts/SF-Pro-Text-Medium.otf"),
)


def display_name(name: str, limit: int = 28) -> str:
    cleaned = " ".join(str(name or "").split())
    if not cleaned:
        return ""
    if cleaned == cleaned.lower():
        cleaned = cleaned.title()
    return cleaned[:limit]


def hex_rgba(hex_color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    raw = hex_color.lstrip("#")
    r = int(raw[0:2], 16)
    g = int(raw[2:4], 16)
    b = int(raw[4:6], 16)
    return r, g, b, alpha


def bgr(hex_color: str) -> tuple[int, int, int]:
    r, g, b, _ = hex_rgba(hex_color)
    return b, g, r


@lru_cache(maxsize=16)
def _font_file(weight: str = "regular") -> str | None:
    pool = _FONT_MEDIUM_CANDIDATES if weight == "medium" else _FONT_CANDIDATES
    for path in pool + _FONT_CANDIDATES:
        if path.is_file():
            return str(path)
    return None


@lru_cache(maxsize=64)
def font(size: int, weight: str = "regular"):
    if not HAS_PIL:
        return None
    path = _font_file(weight)
    if path:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def measure(text: str, size: int, weight: str = "regular") -> tuple[int, int]:
    if not HAS_PIL:
        return max(1, int(len(text) * size * 0.55)), max(1, size)
    face = font(size, weight)
    box = face.getbbox(text or " ")
    return max(1, box[2] - box[0]), max(1, box[3] - box[1])


def _as_rgba(frame: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb).convert("RGBA")


def _blit(frame: np.ndarray, overlay: Image.Image) -> None:
    rgb = overlay.convert("RGB")
    bgr_img = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    if overlay.mode == "RGBA":
        alpha = np.array(overlay.split()[-1], dtype=np.float32) / 255.0
        alpha = alpha[..., None]
        blended = (bgr_img.astype(np.float32) * alpha) + (frame.astype(np.float32) * (1.0 - alpha))
        frame[:] = blended.astype(np.uint8)
        return
    frame[:] = bgr_img


def dim_frame(frame: np.ndarray, amount: float = 0.42) -> None:
    """Quiet locked look: darken the live feed without a neon wash."""
    amount = max(0.0, min(1.0, amount))
    wash = np.full_like(frame, BASE)
    cv2.addWeighted(wash, amount, frame, 1.0 - amount, 0, frame)


def rounded_rect(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    radius: int,
    fill: tuple[int, int, int, int] | None = None,
    outline: tuple[int, int, int, int] | None = None,
    width: int = 1,
) -> None:
    x1, y1, x2, y2 = box
    radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    draw.rounded_rectangle((x1, y1, x2, y2), radius=radius, fill=fill, outline=outline, width=width)


def glass_panel(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    radius: int = 16,
    fill: tuple[int, int, int, int] = (10, 11, 15, 150),
    outline: tuple[int, int, int, int] = (255, 255, 255, 28),
) -> None:
    if not HAS_PIL:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), MUTE, 1, cv2.LINE_AA)
        return
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return
    crop = frame[y1:y2, x1:x2]
    layer = Image.new("RGBA", (x2 - x1, y2 - y1), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    rounded_rect(draw, (0, 0, x2 - x1 - 1, y2 - y1 - 1), radius, fill=fill, outline=outline, width=1)
    _blit(crop, layer)


def text(
    frame: np.ndarray,
    content: str,
    x: int,
    y: int,
    *,
    size: int = 14,
    color: tuple[int, int, int] = TEXT,
    alpha: int = 230,
    weight: str = "regular",
    anchor: str = "lt",
) -> tuple[int, int]:
    if not content:
        return 0, 0
    if not HAS_PIL:
        return measure(content, size, weight)
    face = font(size, weight)
    tw, th = measure(content, size, weight)
    pad = max(6, size)
    ax, ay = int(x), int(y)
    horiz, vert = (anchor + "lt")[:2]
    if horiz == "m":
        left = ax - tw // 2 - pad
    elif horiz == "r":
        left = ax - tw - pad
    else:
        left = ax - pad
    if vert == "m":
        top = ay - th // 2 - pad
    elif vert == "b":
        top = ay - th - pad
    else:
        top = ay - pad
    width = tw + pad * 2
    height = th + pad * 2
    h, w = frame.shape[:2]
    left = max(0, left)
    top = max(0, top)
    right = min(w, left + width)
    bottom = min(h, top + height)
    if right <= left or bottom <= top:
        return tw, th
    crop = frame[top:bottom, left:right]
    layer = Image.new("RGBA", (right - left, bottom - top), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    rgba = (int(color[2]), int(color[1]), int(color[0]), int(alpha))
    draw.text((ax - left, ay - top), content, font=face, fill=rgba, anchor=anchor)
    _blit(crop, layer)
    return tw, th


def text_center(
    frame: np.ndarray,
    content: str,
    cx: int,
    cy: int,
    *,
    size: int = 14,
    color: tuple[int, int, int] = TEXT,
    alpha: int = 230,
    weight: str = "regular",
) -> tuple[int, int]:
    return text(frame, content, cx, cy, size=size, color=color, alpha=alpha, weight=weight, anchor="mm")


def pill(
    frame: np.ndarray,
    x: int,
    y: int,
    label: str,
    *,
    dot: tuple[int, int, int] | None = None,
    height: int = 32,
    pad_x: int = 14,
    size: int = 13,
    fill: tuple[int, int, int, int] = (12, 14, 18, 168),
    outline: tuple[int, int, int, int] = (255, 255, 255, 32),
    text_color: tuple[int, int, int] = TEXT,
    text_alpha: int = 230,
) -> tuple[int, int, int, int]:
    tw, th = measure(label, size)
    extra = 18 if dot is not None else 0
    width = tw + pad_x * 2 + extra
    x2, y2 = x + width, y + height
    glass_panel(frame, x, y, x2, y2, radius=height // 2, fill=fill, outline=outline)
    tx = x + pad_x + extra
    if dot is not None:
        cv2.circle(frame, (x + pad_x + 4, y + height // 2), 3, dot, -1, cv2.LINE_AA)
    text(frame, label, tx, y + height // 2, size=size, color=text_color, alpha=text_alpha, anchor="lm")
    return x, y, x2, y2


def hairline_brackets(
    frame: np.ndarray,
    inset: int = 18,
    length: int = 22,
    color: tuple[int, int, int] = MUTE,
    thickness: int = 1,
) -> tuple[int, int, int, int]:
    h, w = frame.shape[:2]
    x1, y1 = inset, inset
    x2, y2 = w - inset - 1, h - inset - 1
    segs = (
        ((x1, y1), (x1 + length, y1)),
        ((x1, y1), (x1, y1 + length)),
        ((x2, y1), (x2 - length, y1)),
        ((x2, y1), (x2, y1 + length)),
        ((x1, y2), (x1 + length, y2)),
        ((x1, y2), (x1, y2 - length)),
        ((x2, y2), (x2 - length, y2)),
        ((x2, y2), (x2, y2 - length)),
    )
    for a, b in segs:
        cv2.line(frame, a, b, color, thickness, cv2.LINE_AA)
    return x1, y1, x2, y2


def breathe(period: float = 2.6, lo: float = 0.55, hi: float = 1.0) -> float:
    phase = (time.monotonic() % period) / period
    return lo + (hi - lo) * (0.5 - 0.5 * math.cos(phase * math.tau))


def draw_hand_landmarks(frame: np.ndarray, hand_landmarks, width: int, height: int) -> None:
    """Sparse teal / violet dots — quiet luxury, not the default MediaPipe rainbow."""
    lm = hand_landmarks.landmark
    pts = [(int(p.x * width), int(p.y * height)) for p in lm]
    line = (80, 80, 80)
    for a, b in HAND_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], line, 1, cv2.LINE_AA)
    for i, pt in enumerate(pts):
        color = TEAL if i in HAND_TIPS else VIOLET
        cv2.circle(frame, pt, 3 if i in HAND_TIPS else 2, color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 3 if i in HAND_TIPS else 2, (40, 40, 40), 1, cv2.LINE_AA)
