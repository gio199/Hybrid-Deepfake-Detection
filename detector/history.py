"""Rolling history buffer of per-frame landmarks, with helpers for
displacement / velocity / z-score style outlier statistics used by the
glitch and physics checks.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np

from .landmarks import FrameLandmarks, POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER


@dataclass
class PointSample:
    frame_idx: int
    x: float
    y: float
    z: float
    visibility: float = 1.0


@dataclass
class Signal:
    """A single named anomaly signal produced by a glitch/physics check.

    `score` is normalized to [0, 1] (0 = no anomaly, 1 = maximally anomalous)
    so that different checks can be combined/weighted consistently.
    """

    name: str
    score: float
    reason: str = ""


class LandmarkHistory:
    """Keeps the last `window` frames of landmarks and exposes helpers to
    compute normalized displacement / velocity / rolling statistics for
    any given landmark index, used by the anomaly-detection modules.
    """

    def __init__(self, window: int = 30):
        self.window = window
        self._buffer: Deque[FrameLandmarks] = deque(maxlen=window)

    def push(self, frame_landmarks: FrameLandmarks) -> None:
        self._buffer.append(frame_landmarks)

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def current(self) -> Optional[FrameLandmarks]:
        return self._buffer[-1] if self._buffer else None

    @property
    def previous(self) -> Optional[FrameLandmarks]:
        return self._buffer[-2] if len(self._buffer) >= 2 else None

    def frames(self) -> List[FrameLandmarks]:
        return list(self._buffer)

    # --- scale normalization helpers -------------------------------------------------

    def current_face_scale(self) -> Optional[float]:
        """Approx face size (pixels) used to normalize facial landmark motion."""
        fl = self.current
        if fl is None:
            return None
        return fl.face_width()

    def current_pose_scale(self) -> Optional[float]:
        """Approx torso size (pixels) used to normalize body landmark motion."""
        fl = self.current
        if fl is None or not fl.pose_present:
            return None
        l_sh = fl.pose[POSE_LEFT_SHOULDER]
        r_sh = fl.pose[POSE_RIGHT_SHOULDER]
        return float(np.hypot(l_sh.x - r_sh.x, l_sh.y - r_sh.y))

    # --- series extraction -------------------------------------------------

    def face_series(self, idx: int) -> List[PointSample]:
        samples = []
        for fl in self._buffer:
            if fl.face_present and fl.face is not None:
                p = fl.face[idx]
                samples.append(PointSample(fl.frame_idx, p.x, p.y, p.z))
        return samples

    def pose_series(self, idx: int) -> List[PointSample]:
        samples = []
        for fl in self._buffer:
            if fl.pose_present and fl.pose is not None:
                p = fl.pose[idx]
                samples.append(PointSample(fl.frame_idx, p.x, p.y, p.z, p.visibility))
        return samples

    # --- generic stats -------------------------------------------------

    @staticmethod
    def displacement(a: PointSample, b: PointSample) -> float:
        return float(np.hypot(a.x - b.x, a.y - b.y))

    @staticmethod
    def zscore_of_last(values: List[float]) -> float:
        """Z-score of the last value vs. the mean/std of the *preceding* values.

        Returns 0.0 if there isn't enough history to compute a meaningful
        baseline (avoids flagging anomalies during warm-up).
        """
        if len(values) < 5:
            return 0.0
        baseline = np.asarray(values[:-1], dtype=np.float64)
        last = values[-1]
        mean = float(np.mean(baseline))
        std = float(np.std(baseline))
        if std < 1e-6:
            return 0.0
        return float((last - mean) / std)

    @staticmethod
    def robust_zscore_of_last(values: List[float]) -> float:
        """Median/MAD-based ("modified") z-score of the last value vs. the
        preceding values. More robust to outliers than mean/std, which
        matters for short windows where one bad frame can skew mean/std.
        Returns 0.0 if there isn't enough history or no variance to compare.
        """
        if len(values) < 6:
            return 0.0
        baseline = np.asarray(values[:-1], dtype=np.float64)
        last = values[-1]
        median = float(np.median(baseline))
        mad = float(np.median(np.abs(baseline - median)))
        if mad < 1e-9:
            return 0.0
        return float(0.6745 * (last - median) / mad)

    @staticmethod
    def displacement_series(samples: List[PointSample]) -> List[float]:
        """Frame-to-frame pixel displacement for a series of consecutive samples."""
        disps = []
        for prev, cur in zip(samples, samples[1:]):
            disps.append(LandmarkHistory.displacement(prev, cur))
        return disps
