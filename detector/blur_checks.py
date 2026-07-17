"""Localized-blur anomaly checks.

Generative/deepfake pipelines frequently blur or smear regions that are
hard to synthesize correctly (hands, fingers, the seam around a swapped
face) specifically to visually hide artifacts that would otherwise be
obvious. A region that is suspiciously blurrier than the rest of the same
frame - or that suddenly gets blurrier than its own recent baseline - is
therefore a useful red flag on top of (not instead of) the geometric
glitch/physics checks.

Sharpness is measured with the classic variance-of-Laplacian metric:
higher variance means more high-frequency detail (sharper), lower
variance means smoother/blurrier.

The hand-blur check specifically gates on the hand's own recent motion
(`_hand_velocity_ratio`): a fast-moving hand legitimately looks blurrier
from ordinary camera-shutter motion blur, which is not evidence of
anything - only a hand that's blurry *while relatively still* is
suspicious. Skipping this made a real difference in testing: a genuine
video of someone gesturing expressively looked highly suspicious to a
naive version of this check.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

from .history import LandmarkHistory, Signal
from .landmarks import FrameLandmarks, HAND_WRIST, HAND_MIDDLE_MCP
from .quality_checks import normalized_face_sharpness

# --- Tunable thresholds -------------------------------------------------

MIN_ROI_SIZE = 24              # minimum crop side (px) to trust a sharpness reading
SHARPNESS_HISTORY = 30

FACE_PAD_RATIO = 0.25          # padding around the face bbox, as a fraction of its size
HAND_PAD_RATIO = 0.35           # padding around the tight hand-landmark bbox, as a fraction of its size

MIN_BG_SHARPNESS = 15.0        # background must be at least this sharp for a mismatch to be meaningful
BLUR_MISMATCH_RATIO = 0.35     # face-vs-background sharpness ratio below this is suspicious
BLUR_MISMATCH_SATURATE_RATIO = 0.08

HAND_MISMATCH_RATIO = 0.45     # hand-vs-face sharpness ratio below this is suspicious
HAND_MISMATCH_SATURATE_RATIO = 0.10
HAND_MAX_VELOCITY_RATIO = 0.12  # skip hands moving faster than this (palm-widths/frame) - real motion blur

BLUR_SPIKE_Z_THRESHOLD = 3.0   # how far below its own recent baseline sharpness must drop
BLUR_SPIKE_Z_SATURATE = 7.0
BLUR_SPIKE_MIN_HISTORY_SEC = 0.20


def _clip01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def _dist(a, b) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))


def _laplacian_sharpness(gray_roi: np.ndarray) -> Optional[float]:
    if gray_roi.size == 0 or gray_roi.shape[0] < MIN_ROI_SIZE or gray_roi.shape[1] < MIN_ROI_SIZE:
        return None
    return float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())


def _bbox_from_points(points, pad_ratio: float, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    pad_x = (x1 - x0) * pad_ratio
    pad_y = (y1 - y0) * pad_ratio
    return (
        max(int(x0 - pad_x), 0),
        max(int(y0 - pad_y), 0),
        min(int(x1 + pad_x), frame_w),
        min(int(y1 + pad_y), frame_h),
    )


def _boxes_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)


class BlurChecker:
    """Stateful localized-blur checks; call `analyze()` once per frame."""

    def __init__(self, history_window: int = SHARPNESS_HISTORY):
        self._face_sharpness_history: Deque[Tuple[float, float]] = deque(maxlen=history_window)
        # Unbounded, whole-clip record of normalized (resolution-independent)
        # face sharpness, used only by the video-level blur-vs-bitrate check
        # (see quality_checks.py) after the whole video has been processed.
        self.face_sharpness_normalized_all: List[float] = []

    def analyze(self, history: LandmarkHistory, frame_bgr: np.ndarray) -> List[Signal]:
        fl = history.current
        if fl is None:
            return []

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        frame_h, frame_w = gray.shape
        signals: List[Signal] = []

        if not (fl.face_present and fl.face is not None):
            return signals

        face_bbox = _bbox_from_points([(p.x, p.y) for p in fl.face], FACE_PAD_RATIO, frame_w, frame_h)
        face_sharpness = _laplacian_sharpness(gray[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]])
        if face_sharpness is None:
            return signals

        self._face_sharpness_history.append((fl.timestamp_sec, face_sharpness))

        norm_sharpness = normalized_face_sharpness(gray[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]])
        if norm_sharpness is not None:
            self.face_sharpness_normalized_all.append(norm_sharpness)

        bg_sharpness = self._background_sharpness(gray, face_bbox)
        mismatch = self._check_blur_mismatch(face_sharpness, bg_sharpness)
        if mismatch is not None:
            signals.append(mismatch)

        spike = self._check_blur_spike(list(self._face_sharpness_history))
        if spike is not None:
            signals.append(spike)

        hand_signal = self._check_hand_blur(gray, fl, history.previous, frame_w, frame_h, face_sharpness)
        if hand_signal is not None:
            signals.append(hand_signal)

        return signals

    def _background_sharpness(self, gray: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Optional[float]:
        """Median sharpness of the four frame corners, as a stand-in for
        "the rest of the scene" - cheap and avoids needing segmentation.
        """
        frame_h, frame_w = gray.shape
        cw, ch = frame_w // 6, frame_h // 6
        corners = [
            (0, 0, cw, ch),
            (frame_w - cw, 0, frame_w, ch),
            (0, frame_h - ch, cw, frame_h),
            (frame_w - cw, frame_h - ch, frame_w, frame_h),
        ]
        values = []
        for box in corners:
            if _boxes_overlap(box, face_bbox):
                continue
            s = _laplacian_sharpness(gray[box[1]:box[3], box[0]:box[2]])
            if s is not None:
                values.append(s)
        return float(np.median(values)) if values else None

    def _check_blur_mismatch(self, face_sharpness: float, bg_sharpness: Optional[float]) -> Optional[Signal]:
        if bg_sharpness is None or bg_sharpness < MIN_BG_SHARPNESS:
            return None
        ratio = face_sharpness / bg_sharpness
        if ratio >= BLUR_MISMATCH_RATIO:
            return None
        score = _clip01((BLUR_MISMATCH_RATIO - ratio) / (BLUR_MISMATCH_RATIO - BLUR_MISMATCH_SATURATE_RATIO))
        return Signal("blur_mismatch", score,
                      f"Face is selectively blurrier than the rest of the scene (ratio={ratio:.2f}) - "
                      "possible attempt to hide artifacts")

    def _check_blur_spike(self, samples: List[Tuple[float, float]]) -> Optional[Signal]:
        if len(samples) < 6 or samples[-1][0] - samples[0][0] < BLUR_SPIKE_MIN_HISTORY_SEC:
            return None
        values = [value for _, value in samples]
        z = LandmarkHistory.robust_zscore_of_last(values)
        if z > -BLUR_SPIKE_Z_THRESHOLD:
            return None
        magnitude = -z
        score = _clip01((magnitude - BLUR_SPIKE_Z_THRESHOLD) / (BLUR_SPIKE_Z_SATURATE - BLUR_SPIKE_Z_THRESHOLD))
        return Signal("blur_onset_spike", score,
                      f"Face sharpness suddenly dropped vs. its own recent baseline (z={z:.1f})")

    def _check_hand_blur(self, gray: np.ndarray, fl: FrameLandmarks, prev_fl: Optional[FrameLandmarks],
                          frame_w: int, frame_h: int, face_sharpness: float) -> Optional[Signal]:
        if not fl.hands_present or not fl.hands:
            return None

        worst_ratio = None
        for hand_idx, hand in enumerate(fl.hands):
            if len(hand) < 21:
                continue
            velocity_ratio = self._hand_velocity_ratio(fl, prev_fl, hand_idx)
            if velocity_ratio is not None and velocity_ratio > HAND_MAX_VELOCITY_RATIO:
                # Fast-moving hand: real motion blur is expected here and
                # isn't evidence of deliberate concealment, so skip it.
                continue
            bbox = _bbox_from_points([(p.x, p.y) for p in hand], HAND_PAD_RATIO, frame_w, frame_h)
            sharpness = _laplacian_sharpness(gray[bbox[1]:bbox[3], bbox[0]:bbox[2]])
            if sharpness is None:
                continue
            ratio = sharpness / max(face_sharpness, 1e-6)
            if worst_ratio is None or ratio < worst_ratio:
                worst_ratio = ratio

        if worst_ratio is None or worst_ratio >= HAND_MISMATCH_RATIO:
            return None
        score = _clip01((HAND_MISMATCH_RATIO - worst_ratio) / (HAND_MISMATCH_RATIO - HAND_MISMATCH_SATURATE_RATIO))
        return Signal("hand_blur_anomaly", score,
                      f"Hand region much blurrier than the face (ratio={worst_ratio:.2f}) - "
                      "generative models often smear hands to hide malformed fingers")

    @staticmethod
    def _hand_velocity_ratio(fl: FrameLandmarks, prev_fl: Optional[FrameLandmarks], hand_idx: int) -> Optional[float]:
        """Wrist displacement since the previous frame, normalized by palm
        size - used to distinguish real motion blur (fast hand) from a
        stationary hand that's suspiciously blurry anyway.
        """
        if prev_fl is None or not prev_fl.hands_present or not prev_fl.hands:
            return None
        if hand_idx >= len(fl.handedness):
            return None
        label = fl.handedness[hand_idx]
        if label not in prev_fl.handedness:
            return None
        prev_hand = prev_fl.hands[prev_fl.handedness.index(label)]
        hand = fl.hands[hand_idx]
        if len(prev_hand) < 21 or len(hand) < 21:
            return None

        palm_scale = _dist(hand[HAND_WRIST], hand[HAND_MIDDLE_MCP])
        if palm_scale < 1e-3:
            return None
        displacement = _dist(hand[HAND_WRIST], prev_hand[HAND_WRIST])
        return displacement / palm_scale
