"""Convert OpenCV BGR frames into NSImage for display in an NSImageView."""

from __future__ import annotations

import cv2
import numpy as np
from AppKit import NSBitmapImageRep, NSDeviceRGBColorSpace, NSImage
from Foundation import NSData, NSMakeSize

# Render at 2x the point size so the preview stays crisp on Retina displays.
RETINA_SCALE = 2


def _rep_from_rgba(rgba: np.ndarray):
    """NSBitmapImageRep backed by AppKit's own buffer, so nothing can dangle."""
    height, width = rgba.shape[:2]
    rep = NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
        None,
        width,
        height,
        8,
        4,
        True,
        False,
        NSDeviceRGBColorSpace,
        width * 4,
        32,
    )
    if rep is None:
        return None
    buffer = rep.bitmapData()
    buffer[:] = rgba.tobytes()
    return rep


def _rep_from_jpeg(frame: np.ndarray):
    """Fallback path: let AppKit decode a JPEG instead of poking at raw planes."""
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return None
    data = NSData.dataWithBytes_length_(encoded.tobytes(), int(encoded.size))
    return NSBitmapImageRep.imageRepWithData_(data)


def nsimage_from_bgr(frame: np.ndarray, point_width: float = 0.0):
    """Build an NSImage from a BGR frame, sized in points for a Retina-crisp draw."""
    if frame is None or frame.size == 0:
        return None

    height, width = frame.shape[:2]
    if point_width and point_width > 0:
        target_w = max(1, int(round(point_width * RETINA_SCALE)))
        if target_w < width:
            target_h = max(1, int(round(height * target_w / width)))
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            height, width = frame.shape[:2]
        point_size = NSMakeSize(point_width, height * point_width / width)
    else:
        point_size = NSMakeSize(width, height)

    rep = None
    try:
        rgba = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
        rep = _rep_from_rgba(rgba)
    except Exception:
        rep = None
    if rep is None:
        rep = _rep_from_jpeg(frame)
    if rep is None:
        return None

    image = NSImage.alloc().initWithSize_(point_size)
    image.addRepresentation_(rep)
    return image
