"""Video-level "blur vs. bitrate" plausibility check.

Every other check in this package looks at a single frame or a short
rolling window. This one is different: it looks at the video as a whole,
using two numbers that only make sense at that scale:

  - bits-per-pixel (bpp): how much data was actually spent encoding each
    pixel of each frame, derived from `file_size / (width * height *
    frame_count)`. This is a standard, resolution/duration-independent
    proxy for "how generous was the encode" - a well-known video-quality
    metric (low bpp = heavily compressed/blocky, high bpp = plenty of
    room to preserve fine detail).
  - the video's own overall face sharpness (variance-of-Laplacian on the
    face crop, resized to a canonical width so different resolutions /
    zoom levels are comparable), taken as a median over every frame in
    the whole clip.

A real camera-sourced clip that was encoded at a generous bitrate should
retain real fine detail (skin texture, hair, sensor noise) - if a video
was given plenty of bits yet the subject still looks soft/waxy across the
*entire* clip (not just a momentary blur onset, which `blur_onset_spike`
in `blur_checks.py` already covers), that mismatch is a red flag: either
the source content itself was inherently low-detail before encoding
(consistent with many generative pipelines, which don't reproduce real
sensor-level detail even when exported at a high quality setting), or
something else is unusual about how this file was produced.

This check is deliberately gated on *high* bpp: a blurry video encoded at
a *low* bitrate is fully and mundanely explained by ordinary compression
and is not evaluated here (that's a completely different, extremely
common, non-suspicious situation - e.g. a heavily-compressed real phone
video). It only fires when there was no bitrate excuse available.

This only applies to file input - a live webcam feed has no fixed "file
size" and no fixed "length" to form a ratio from, so the caller should
skip this check for webcam sources.

Calibration note: thresholds below were set from only three known
clips at time of writing (one real, two confirmed-fake), so treat this
as a coarse, low-confidence signal - not a certainty. See README for the
concrete numbers this was calibrated against.
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from .history import Signal

# --- Tunable thresholds -------------------------------------------------

FACE_CANONICAL_WIDTH = 200   # resize face crops to this width before measuring sharpness so
                              # measurements are comparable across different resolutions/zoom levels
MIN_SAMPLES = 20              # need at least this many whole-video face-sharpness samples to trust the metric

# Real-world reference points measured directly from this project's test clips:
#   real_baseline.mp4 (genuine webcam clip): bpp=0.124, median normalized face sharpness=319
#   example1 (face-swap deepfake, heavily compressed): bpp=0.028, sharpness=33   -> bpp too low to judge, correctly skipped
#   example2 (confirmed AI-generated clip):  bpp=0.260, sharpness=78            -> 2x the bitrate of the real clip, 4x softer
HIGH_BPP_GATE = 0.05          # below this, blur is plausibly just ordinary compression - don't judge it
LOW_SHARPNESS_THRESHOLD = 180.0   # normalized face sharpness below this (while bpp is high) is suspiciously soft
LOW_SHARPNESS_SATURATE = 50.0


def _clip01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def normalized_face_sharpness(gray_face_crop: np.ndarray) -> Optional[float]:
    """Variance-of-Laplacian on a face crop resized to a canonical width,
    so the result is comparable across videos of different resolution or
    subject-to-camera distance.
    """
    h, w = gray_face_crop.shape[:2]
    if h < 10 or w < 10:
        return None
    scale = FACE_CANONICAL_WIDTH / float(w)
    resized = cv2.resize(gray_face_crop, (FACE_CANONICAL_WIDTH, max(1, int(round(h * scale)))))
    return float(cv2.Laplacian(resized, cv2.CV_64F).var())


def bits_per_pixel(file_size_bytes: int, width: int, height: int, frame_count: int) -> Optional[float]:
    if width <= 0 or height <= 0 or frame_count <= 0:
        return None
    return (file_size_bytes * 8.0) / (width * height * frame_count)


def assess_blur_vs_bitrate(face_sharpness_samples: List[float], file_size_bytes: Optional[int],
                            width: int, height: int, frame_count: int) -> Optional[Signal]:
    """Returns a `blur_vs_bitrate_mismatch` Signal if this video was given
    a generous bitrate but the subject's face is still soft throughout,
    or None if the check doesn't apply / doesn't fire.
    """
    if file_size_bytes is None or len(face_sharpness_samples) < MIN_SAMPLES:
        return None

    bpp = bits_per_pixel(file_size_bytes, width, height, frame_count)
    if bpp is None or bpp < HIGH_BPP_GATE:
        return None  # not enough data allocated to expect real sharpness - not a fair comparison

    median_sharpness = float(np.median(face_sharpness_samples))
    if median_sharpness >= LOW_SHARPNESS_THRESHOLD:
        return None

    score = _clip01((LOW_SHARPNESS_THRESHOLD - median_sharpness) / (LOW_SHARPNESS_THRESHOLD - LOW_SHARPNESS_SATURATE))
    return Signal(
        "blur_vs_bitrate_mismatch",
        score,
        f"Video was encoded with a generous bitrate ({bpp:.3f} bits/pixel) yet the face stays soft "
        f"throughout the clip (median sharpness={median_sharpness:.0f}) - real camera footage encoded "
        "this generously would normally retain fine detail",
    )
