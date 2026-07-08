"""Hand/finger anatomical-plausibility checks. NOT CURRENTLY WIRED IN.

Hands are one of the hardest things for generative models (GANs,
diffusion, face-swap/full-body synthesis pipelines) to get consistently
right, so the idea here was reasonable: mirror `physics_checks.py`'s
approach (track a derived quantity's own recent baseline, flag sudden
deviations via a robust z-score) but applied to MediaPipe's 21-point hand
landmarks instead of the 33-point body pose, to catch finger length/bend
implausibility.

In practice this was tried, measured, and rejected (kept here for
reference so it isn't re-attempted the same way): MediaPipe's per-frame
hand landmark estimates are too noisy under natural, fast, expressive
hand motion (2D projected finger length changes a lot from simple
foreshortening as a finger rotates toward/away from the camera; fast
gestures produce large frame-to-frame angle deltas that look identical
to a "discontinuous glitch"). A real test video of someone gesturing
expressively triggered `finger_bone_length` on 57.8% of frames and
`finger_joint_motion` on 30% of frames - both *far* higher than an
actual confirmed deepfake clip (38% / 5.2%) in side-by-side testing. That
makes these checks net-harmful as evidence rather than net-helpful, so
`main.py` does not call `HandChecker`, and `finger_bone_length` /
`finger_joint_motion` are intentionally absent from
`scoring.DEFAULT_WEIGHTS`. `detector/blur_checks.py`'s `hand_blur_anomaly`
check still uses the hand landmarks (for a tight crop, not shape
consistency) and remains enabled.

Note: the Hand Landmarker model doesn't expose a per-landmark visibility
score (unlike Pose), so hand quality is instead gated on a minimum palm
size in pixels (too-small/far hands are noisy) and MediaPipe's own
handedness detection confidence.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .history import LandmarkHistory, Signal
from .landmarks import FrameLandmarks, HAND_BONES, HAND_JOINT_ANGLES, HAND_WRIST, HAND_MIDDLE_MCP

# --- Tunable thresholds -------------------------------------------------

MIN_HISTORY_FRAMES = 8
MIN_HAND_SCALE_PX = 18.0    # ignore hands smaller than this (palm wrist->middle_mcp distance)

FINGER_BONE_Z_THRESHOLD = 3.5
FINGER_BONE_Z_SATURATE = 8.0
FINGER_BONE_MIN_REL_CHANGE = 0.16   # fingers are thin/noisy; a bit more tolerant than body bones

FINGER_JOINT_Z_THRESHOLD = 3.5
FINGER_JOINT_Z_SATURATE = 8.0
FINGER_JOINT_MIN_DELTA_DEG = 22.0

_HAND_LABELS = ("Left", "Right")


def _dist(a, b) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))


def _angle_deg(a, vertex, b) -> Optional[float]:
    v1 = np.array([a.x - vertex.x, a.y - vertex.y], dtype=np.float64)
    v2 = np.array([b.x - vertex.x, b.y - vertex.y], dtype=np.float64)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def _clip01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def _compute_hand_metrics(fl: FrameLandmarks) -> Dict[str, Dict[str, float]]:
    """Returns {"Left": {...}, "Right": {...}} metrics for whichever hands
    MediaPipe found and are large enough to trust in this frame.
    """
    result: Dict[str, Dict[str, float]] = {}
    if not fl.hands_present or not fl.hands:
        return result

    for hand_idx, hand in enumerate(fl.hands):
        if hand_idx >= len(fl.handedness):
            continue
        label = fl.handedness[hand_idx]
        if label not in _HAND_LABELS or len(hand) < 21:
            continue

        palm_scale = _dist(hand[HAND_WRIST], hand[HAND_MIDDLE_MCP])
        if palm_scale < MIN_HAND_SCALE_PX:
            continue

        metrics: Dict[str, float] = {"scale:palm": palm_scale}
        for name, (a, b) in HAND_BONES.items():
            metrics[f"bone:{name}"] = _dist(hand[a], hand[b])
        for name, (a, vertex, b) in HAND_JOINT_ANGLES.items():
            angle = _angle_deg(hand[a], hand[vertex], hand[b])
            if angle is not None:
                metrics[f"angle:{name}"] = angle

        # If both hands are labeled the same (a MediaPipe handedness slip),
        # keep whichever appears first this frame rather than overwriting.
        result.setdefault(label, metrics)

    return result


class HandChecker:
    """Rigid-hand plausibility checks over a landmark history window."""

    def analyze(self, history: LandmarkHistory) -> List[Signal]:
        frames = history.frames()
        if len(frames) < MIN_HISTORY_FRAMES:
            return []

        per_frame_metrics = [_compute_hand_metrics(fl) for fl in frames]

        signals: List[Signal] = []
        bone_signal = self._check_finger_bones(per_frame_metrics)
        if bone_signal is not None:
            signals.append(bone_signal)

        joint_signal = self._check_finger_joints(per_frame_metrics)
        if joint_signal is not None:
            signals.append(joint_signal)

        return signals

    def _check_finger_bones(self, per_frame_metrics: List[Dict[str, Dict[str, float]]]) -> Optional[Signal]:
        worst_z, worst_name = 0.0, None

        for label in _HAND_LABELS:
            for bone_name in HAND_BONES:
                key = f"bone:{bone_name}"
                ratios = [
                    m[label][key] / m[label]["scale:palm"]
                    for m in per_frame_metrics
                    if label in m and key in m[label] and m[label].get("scale:palm", 0) > 1e-3
                ]
                z = self._evaluate_ratio_series(ratios, FINGER_BONE_MIN_REL_CHANGE)
                if abs(z) > abs(worst_z):
                    worst_z, worst_name = z, f"{label.lower()}_{bone_name}"

        if worst_name is None or abs(worst_z) < FINGER_BONE_Z_THRESHOLD:
            return None
        score = _clip01((abs(worst_z) - FINGER_BONE_Z_THRESHOLD) / (FINGER_BONE_Z_SATURATE - FINGER_BONE_Z_THRESHOLD))
        return Signal("finger_bone_length", score,
                      f"Finger segment length changed implausibly ({worst_name}, z={worst_z:.1f})")

    def _check_finger_joints(self, per_frame_metrics: List[Dict[str, Dict[str, float]]]) -> Optional[Signal]:
        worst_z, worst_name, worst_delta = 0.0, None, 0.0

        for label in _HAND_LABELS:
            for joint_name in HAND_JOINT_ANGLES:
                key = f"angle:{joint_name}"
                values = [m[label][key] for m in per_frame_metrics if label in m and key in m[label]]
                if len(values) < MIN_HISTORY_FRAMES:
                    continue
                deltas = [b - a for a, b in zip(values, values[1:])]
                if len(deltas) < 6 or abs(deltas[-1]) < FINGER_JOINT_MIN_DELTA_DEG:
                    continue
                z = LandmarkHistory.robust_zscore_of_last(deltas)
                if abs(z) > abs(worst_z):
                    worst_z, worst_name, worst_delta = z, f"{label.lower()}_{joint_name}", deltas[-1]

        if worst_name is None or abs(worst_z) < FINGER_JOINT_Z_THRESHOLD:
            return None
        score = _clip01((abs(worst_z) - FINGER_JOINT_Z_THRESHOLD) / (FINGER_JOINT_Z_SATURATE - FINGER_JOINT_Z_THRESHOLD))
        return Signal("finger_joint_motion", score,
                      f"Implausibly abrupt finger bend ({worst_name}, {worst_delta:+.0f} deg, z={worst_z:.1f})")

    @staticmethod
    def _evaluate_ratio_series(values: List[float], min_rel_change: float) -> float:
        if len(values) < MIN_HISTORY_FRAMES:
            return 0.0
        baseline_median = float(np.median(values[:-1]))
        if baseline_median < 1e-6:
            return 0.0
        rel_change = abs(values[-1] - baseline_median) / baseline_median
        if rel_change < min_rel_change:
            return 0.0
        return LandmarkHistory.robust_zscore_of_last(values)
