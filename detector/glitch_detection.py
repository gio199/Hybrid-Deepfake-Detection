"""Temporal 'glitch' anomaly checks.

These checks don't know anything about human anatomy - they only look at
how landmarks move (and how the raw pixels move) from frame to frame, and
flag patterns that look like tracking/rendering glitches rather than
natural human motion: high-frequency jitter, sudden teleports, detection
flicker, and pixel-level flicker around the face (a common face-swap
blending-seam artifact).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import cv2
import numpy as np

from .history import LandmarkHistory, PointSample, Signal
from .face_crop import aligned_face_crop
from .landmarks import (
    FrameLandmarks,
    FACE_NOSE_TIP, FACE_FOREHEAD, FACE_LEFT_CHEEK_EDGE, FACE_RIGHT_CHEEK_EDGE,
    FACE_LEFT_EYE_OUTER_CORNER, FACE_RIGHT_EYE_OUTER_CORNER,
    POSE_NOSE, POSE_LEFT_WRIST, POSE_RIGHT_WRIST, POSE_LEFT_ANKLE, POSE_RIGHT_ANKLE,
)

# --- Tunable thresholds -------------------------------------------------

# Points chosen here are relatively rigid on the skull/skeleton during
# normal human behavior (talking, blinking) so that natural mouth/jaw
# articulation during speech doesn't itself look like "jitter".
REPRESENTATIVE_FACE_POINTS = [FACE_NOSE_TIP, FACE_FOREHEAD, FACE_LEFT_CHEEK_EDGE, FACE_RIGHT_CHEEK_EDGE,
                              FACE_LEFT_EYE_OUTER_CORNER, FACE_RIGHT_EYE_OUTER_CORNER]
REPRESENTATIVE_POSE_POINTS = [POSE_NOSE, POSE_LEFT_WRIST, POSE_RIGHT_WRIST, POSE_LEFT_ANKLE, POSE_RIGHT_ANKLE]

MIN_POSE_VISIBILITY = 0.5        # ignore pose points that are occluded / out of frame

JITTER_MIN_WINDOW = 6            # min samples of uninterrupted history needed
JITTER_WINDOW_SECONDS = 1.0      # same physical window regardless of source FPS
JITTER_MIN_DURATION_SECONDS = 0.20
JITTER_NORMALIZER = 2.2          # divides normalized path/second into ~[0,1]

JUMP_Z_THRESHOLD = 4.5           # z-score above which a displacement is a "jump"
JUMP_Z_SATURATE = 10.0           # z-score at which the jump score saturates to 1.0

FLICKER_WINDOW = 10              # frames looked at for detection on/off flicker
FLICKER_MIN_TRANSITIONS = 3      # transitions within window considered anomalous
FLICKER_MAX_TRANSITIONS = 7      # transitions within window -> score saturates to 1.0

PIXEL_FLICKER_CROP_SIZE = 64
PIXEL_FLICKER_Z_THRESHOLD = 3.0
PIXEL_FLICKER_Z_SATURATE = 7.0


def _clip01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def _visible_pose_series(history: LandmarkHistory, idx: int) -> List[PointSample]:
    """Pose landmark series with low-visibility (occluded/out-of-frame)
    samples dropped, since their positions are unreliable and would
    otherwise look like meaningless "jitter" or "jumps".
    """
    return [s for s in history.pose_series(idx) if s.visibility >= MIN_POSE_VISIBILITY]


def _path_efficiency_jitter(samples: List[PointSample], scale: Optional[float]) -> float:
    """Ratio-based jitter metric: high when a point wiggles back and forth a
    lot (long path) without much net displacement (low path efficiency).
    """
    if len(samples) < JITTER_MIN_WINDOW or not scale or scale < 1e-3:
        return 0.0

    pts = np.array([[s.x, s.y] for s in samples], dtype=np.float64)
    segment_lengths = np.hypot(*(pts[1:] - pts[:-1]).T)
    path_length = float(np.sum(segment_lengths))
    net_displacement = float(np.hypot(*(pts[-1] - pts[0])))

    if path_length < 1e-6:
        return 0.0

    duration_sec = samples[-1].timestamp_sec - samples[0].timestamp_sec
    if duration_sec < JITTER_MIN_DURATION_SECONDS:
        return 0.0

    norm_path_length = path_length / scale / duration_sec
    efficiency = net_displacement / path_length  # 1.0 = perfectly straight motion
    jitter_raw = norm_path_length * (1.0 - efficiency)
    return _clip01(jitter_raw / JITTER_NORMALIZER)


class GlitchDetector:
    """Stateful glitch checks; call `analyze()` once per frame, in order."""

    def __init__(self, flicker_window: int = FLICKER_WINDOW, pixel_history_window: int = 30):
        self._face_presence: Deque[bool] = deque(maxlen=flicker_window)
        self._pose_presence: Deque[bool] = deque(maxlen=flicker_window)
        self._prev_face_crop: Optional[np.ndarray] = None
        self._pixel_diff_history: Deque[float] = deque(maxlen=pixel_history_window)

    def observe_missing(self) -> None:
        """Record a complete detector dropout for an otherwise-live track."""
        self._face_presence.append(False)
        self._pose_presence.append(False)
        self._prev_face_crop = None

    def analyze(self, history: LandmarkHistory, frame_bgr: np.ndarray) -> List[Signal]:
        signals: List[Signal] = []

        jitter = self._check_jitter(history)
        if jitter is not None:
            signals.append(jitter)

        jump = self._check_sudden_jump(history)
        if jump is not None:
            signals.append(jump)

        flicker = self._check_detection_flicker(history)
        if flicker is not None:
            signals.append(flicker)

        pixel_flicker = self._check_pixel_flicker(history, frame_bgr)
        if pixel_flicker is not None:
            signals.append(pixel_flicker)

        return signals

    # --- individual checks -------------------------------------------------

    def _check_jitter(self, history: LandmarkHistory) -> Optional[Signal]:
        face_scale = history.current_face_scale()
        pose_scale = history.current_pose_scale()

        scores = []
        for idx in REPRESENTATIVE_FACE_POINTS:
            series = LandmarkHistory.consecutive_tail(
                history.face_series(idx), duration_sec=JITTER_WINDOW_SECONDS
            )
            scores.append(_path_efficiency_jitter(series, face_scale))
        for idx in REPRESENTATIVE_POSE_POINTS:
            series = LandmarkHistory.consecutive_tail(
                _visible_pose_series(history, idx), duration_sec=JITTER_WINDOW_SECONDS
            )
            scores.append(_path_efficiency_jitter(series, pose_scale))

        scores = [s for s in scores if s > 0]
        if not scores:
            return None
        score = float(np.mean(sorted(scores, reverse=True)[: max(1, len(scores) // 2)]))
        if score <= 0.05:
            return None
        return Signal("landmark_jitter", score, f"High-frequency landmark jitter (score={score:.2f})")

    def _check_sudden_jump(self, history: LandmarkHistory) -> Optional[Signal]:
        face_scale = history.current_face_scale()
        pose_scale = history.current_pose_scale()

        max_z = 0.0
        for idx in REPRESENTATIVE_FACE_POINTS:
            series = history.face_series(idx)
            if len(series) < 6 or not face_scale or not LandmarkHistory.latest_pair_is_consecutive(series):
                continue
            disps = [d / face_scale for d in LandmarkHistory.displacement_series(series, as_rate=True)]
            z = LandmarkHistory.zscore_of_last(disps)
            max_z = max(max_z, z)

        for idx in REPRESENTATIVE_POSE_POINTS:
            series = _visible_pose_series(history, idx)
            if len(series) < 6 or not pose_scale or not LandmarkHistory.latest_pair_is_consecutive(series):
                continue
            disps = [d / pose_scale for d in LandmarkHistory.displacement_series(series, as_rate=True)]
            z = LandmarkHistory.zscore_of_last(disps)
            max_z = max(max_z, z)

        if max_z < JUMP_Z_THRESHOLD:
            return None
        score = _clip01((max_z - JUMP_Z_THRESHOLD) / (JUMP_Z_SATURATE - JUMP_Z_THRESHOLD))
        return Signal("landmark_jump", score, f"Sudden landmark teleport (z={max_z:.1f})")

    def _check_detection_flicker(self, history: LandmarkHistory) -> Optional[Signal]:
        fl = history.current
        if fl is None:
            return None
        self._face_presence.append(fl.face_present)
        self._pose_presence.append(fl.pose_present)

        def transitions(seq: Deque[bool]) -> int:
            if len(seq) < 4:
                return 0
            seq_list = list(seq)
            return sum(1 for a, b in zip(seq_list, seq_list[1:]) if a != b)

        t_face = transitions(self._face_presence)
        t_pose = transitions(self._pose_presence)
        t = max(t_face, t_pose)

        if t < FLICKER_MIN_TRANSITIONS:
            return None
        score = _clip01((t - FLICKER_MIN_TRANSITIONS) / (FLICKER_MAX_TRANSITIONS - FLICKER_MIN_TRANSITIONS))
        return Signal("detection_flicker", score, f"Face/pose detection flickering on/off ({t} transitions)")

    def _check_pixel_flicker(self, history: LandmarkHistory, frame_bgr: np.ndarray) -> Optional[Signal]:
        fl = history.current
        if fl is None or not fl.face_present or fl.face is None:
            self._prev_face_crop = None
            return None

        crop_bgr = aligned_face_crop(frame_bgr, fl, PIXEL_FLICKER_CROP_SIZE)
        crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) if crop_bgr is not None else None
        if crop is None:
            return None

        # Remove global brightness/contrast changes before comparing crops.
        crop = (crop - float(np.mean(crop))) / max(float(np.std(crop)), 8.0)

        if self._prev_face_crop is None:
            self._prev_face_crop = crop
            return None

        mad = float(np.mean(np.abs(crop - self._prev_face_crop)))
        self._prev_face_crop = crop
        self._pixel_diff_history.append(mad)

        z = LandmarkHistory.zscore_of_last(list(self._pixel_diff_history))
        if z < PIXEL_FLICKER_Z_THRESHOLD:
            return None
        score = _clip01((z - PIXEL_FLICKER_Z_THRESHOLD) / (PIXEL_FLICKER_Z_SATURATE - PIXEL_FLICKER_Z_THRESHOLD))
        return Signal("pixel_flicker", score, f"Face-region pixel flicker / blending seam (z={z:.1f})")
