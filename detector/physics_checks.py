"""Human-anatomy / rigid-body-physics plausibility checks.

Unlike glitch_detection.py (which only looks at raw point motion), these
checks derive physically-meaningful quantities from the landmarks - bone
lengths, joint angles, head pose, left/right symmetry - and flag frames
where those quantities suddenly deviate from what a rigid human body
should look like.

Every check tracks its own derived quantity across the rolling history
window and flags *sudden deviations from that quantity's own recent
baseline* (via a robust median/MAD z-score) rather than a fixed absolute
threshold. This automatically adapts to a given person's proportions and
camera framing, and only fires on genuine discontinuities - which is
exactly the kind of artifact face-swap / warping deepfakes tend to
introduce.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np

from .history import LandmarkHistory, Signal
from .landmarks import (
    FrameLandmarks,
    FACE_LEFT_IRIS_CENTER, FACE_RIGHT_IRIS_CENTER,
    FACE_LEFT_MOUTH_CORNER, FACE_RIGHT_MOUTH_CORNER,
    FACE_FOREHEAD, FACE_CHIN,
    FACE_LEFT_EYE, FACE_RIGHT_EYE,
    FACE_POSE_LANDMARK_IDS, FACE_POSE_MODEL_3D,
    POSE_BONES, POSE_JOINT_ANGLES,
    POSE_LEFT_SHOULDER, POSE_LEFT_ELBOW, POSE_LEFT_WRIST,
    POSE_RIGHT_SHOULDER, POSE_RIGHT_ELBOW, POSE_RIGHT_WRIST,
    POSE_LEFT_HIP, POSE_LEFT_KNEE, POSE_LEFT_ANKLE,
    POSE_RIGHT_HIP, POSE_RIGHT_KNEE, POSE_RIGHT_ANKLE,
)

# --- Tunable thresholds -------------------------------------------------

MIN_HISTORY_FRAMES = 6
MIN_HISTORY_SECONDS = 0.25
MIN_POSE_VISIBILITY = 0.5   # ignore pose points that are occluded / out of frame - their positions are unreliable

BONE_LENGTH_Z_THRESHOLD = 3.5
BONE_LENGTH_Z_SATURATE = 8.0
BONE_LENGTH_MIN_REL_CHANGE = 0.12   # ignore sub-12% wobble even if "significant" statistically

JOINT_ANGLE_Z_THRESHOLD = 3.5
JOINT_ANGLE_Z_SATURATE = 8.0
JOINT_ANGLE_MIN_DELTA_DEG = 18.0    # ignore small natural angle changes

HEAD_POSE_Z_THRESHOLD = 3.0
HEAD_POSE_Z_SATURATE = 7.0
HEAD_POSE_MIN_RESIDUAL_RATIO = 0.035  # residual must be >=3.5% of face width to matter

SYMMETRY_Z_THRESHOLD = 4.0
SYMMETRY_Z_SATURATE = 9.0
SYMMETRY_MIN_REL_CHANGE = 0.15

_NON_SCALE_POSE_BONES = {k: v for k, v in POSE_BONES.items() if k not in ("shoulder_span", "hip_span")}


def _dist(a, b) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))


def _dist3(a, b) -> float:
    return float(np.linalg.norm([a.x - b.x, a.y - b.y, a.z - b.z]))


def _pose_visible(fl: FrameLandmarks, *idxs: int) -> bool:
    """True only if every given pose landmark is confidently visible.

    MediaPipe still returns a position for occluded/out-of-frame body
    points, but that position is essentially noise, so any check built on
    top of it (bone length, joint angle, symmetry) needs to skip those
    frames rather than treat the noise as a physical anomaly.
    """
    if fl.pose is None:
        return False
    return all(fl.pose[i].visibility >= MIN_POSE_VISIBILITY for i in idxs)


def _angle_deg(a, vertex, b) -> Optional[float]:
    v1 = np.array([a.x - vertex.x, a.y - vertex.y, a.z - vertex.z], dtype=np.float64)
    v2 = np.array([b.x - vertex.x, b.y - vertex.y, b.z - vertex.z], dtype=np.float64)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def _clip01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def _head_pose_residual_ratio(fl: FrameLandmarks) -> Optional[float]:
    """Fits a generic rigid 3D face model to 6 stable 2D landmarks via
    solvePnP, then measures how well those same 6 points re-project under
    the best-fit rigid rotation/translation. A rigid, undistorted face
    should fit consistently frame to frame; a warped/blended fake face
    tends to produce extra, inconsistent residual on top of the person's
    own baseline model-mismatch (which the caller z-scores away).
    """
    if not fl.face_present or fl.face is None:
        return None

    try:
        image_points = np.array(
            [[fl.face[idx].x, fl.face[idx].y] for idx in FACE_POSE_LANDMARK_IDS], dtype=np.float64
        )
        model_points = np.array(
            [FACE_POSE_MODEL_3D[idx] for idx in FACE_POSE_LANDMARK_IDS], dtype=np.float64
        )
    except (IndexError, KeyError):
        return None

    focal_length = float(fl.image_w)
    center = (fl.image_w / 2.0, fl.image_h / 2.0)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype=np.float64
    )
    dist_coeffs = np.zeros((4, 1))

    ok, rvec, tvec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None

    reprojected, _ = cv2.projectPoints(model_points, rvec, tvec, camera_matrix, dist_coeffs)
    reprojected = reprojected.reshape(-1, 2)
    residual_px = float(np.mean(np.linalg.norm(reprojected - image_points, axis=1)))

    face_scale = fl.face_width()
    if not face_scale or face_scale < 1e-3:
        return None
    return residual_px / face_scale


def _compute_frame_metrics(fl: FrameLandmarks) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    if fl.pose_present and fl.pose is not None:
        # World coordinates greatly reduce perspective/foreshortening
        # artifacts in bone lengths and joint angles. Fall back to image
        # coordinates for compatibility with synthetic/unit-test inputs.
        pose_points = fl.pose_world if fl.pose_world and len(fl.pose_world) == len(fl.pose) else fl.pose
        for name, (a, b) in POSE_BONES.items():
            if _pose_visible(fl, a, b):
                metrics[f"bone:{name}"] = _dist3(pose_points[a], pose_points[b])
        for name, (a, vertex, b) in POSE_JOINT_ANGLES.items():
            if _pose_visible(fl, a, vertex, b):
                angle = _angle_deg(pose_points[a], pose_points[vertex], pose_points[b])
                if angle is not None:
                    metrics[f"angle:{name}"] = angle

        shoulder_span = metrics.get("bone:shoulder_span")
        if shoulder_span and shoulder_span > 1e-3:
            metrics["scale:torso"] = shoulder_span

        if _pose_visible(fl, POSE_LEFT_SHOULDER, POSE_LEFT_ELBOW, POSE_LEFT_WRIST,
                          POSE_RIGHT_SHOULDER, POSE_RIGHT_ELBOW, POSE_RIGHT_WRIST):
            left_arm = metrics.get("bone:left_upper_arm", 0) + metrics.get("bone:left_forearm", 0)
            right_arm = metrics.get("bone:right_upper_arm", 0) + metrics.get("bone:right_forearm", 0)
            if right_arm > 1e-3:
                metrics["symmetry:arm_ratio"] = left_arm / right_arm

        if _pose_visible(fl, POSE_LEFT_HIP, POSE_LEFT_KNEE, POSE_LEFT_ANKLE,
                          POSE_RIGHT_HIP, POSE_RIGHT_KNEE, POSE_RIGHT_ANKLE):
            left_leg = metrics.get("bone:left_thigh", 0) + metrics.get("bone:left_shin", 0)
            right_leg = metrics.get("bone:right_thigh", 0) + metrics.get("bone:right_shin", 0)
            if right_leg > 1e-3:
                metrics["symmetry:leg_ratio"] = left_leg / right_leg

    if fl.face_present and fl.face is not None:
        face_scale = fl.face_width()
        if face_scale and face_scale > 1e-3:
            metrics["scale:face"] = face_scale
            metrics["bone:interocular"] = _dist(fl.face[FACE_LEFT_IRIS_CENTER], fl.face[FACE_RIGHT_IRIS_CENTER])
            metrics["bone:mouth_width"] = _dist(fl.face[FACE_LEFT_MOUTH_CORNER], fl.face[FACE_RIGHT_MOUTH_CORNER])
            metrics["bone:face_height"] = _dist(fl.face[FACE_FOREHEAD], fl.face[FACE_CHIN])

        left_eye_w = _dist(fl.face[FACE_LEFT_EYE[0]], fl.face[FACE_LEFT_EYE[3]])
        right_eye_w = _dist(fl.face[FACE_RIGHT_EYE[0]], fl.face[FACE_RIGHT_EYE[3]])
        if right_eye_w > 1e-3:
            metrics["symmetry:eye_ratio"] = left_eye_w / right_eye_w

        residual_ratio = _head_pose_residual_ratio(fl)
        if residual_ratio is not None:
            metrics["head_pose_residual"] = residual_ratio

    return metrics


class PhysicsChecker:
    """Stateless-ish rigid-body plausibility checks over a landmark history window."""

    def analyze(self, history: LandmarkHistory) -> List[Signal]:
        frames = history.frames()
        if (
            len(frames) < MIN_HISTORY_FRAMES
            or frames[-1].timestamp_sec - frames[0].timestamp_sec < MIN_HISTORY_SECONDS
        ):
            return []

        series = [_compute_frame_metrics(fl) for fl in frames]

        signals: List[Signal] = []

        bone_signal = self._check_bone_lengths(series)
        if bone_signal is not None:
            signals.append(bone_signal)

        angle_signal = self._check_joint_angles(series)
        if angle_signal is not None:
            signals.append(angle_signal)

        head_pose_signal = self._check_head_pose(series)
        if head_pose_signal is not None:
            signals.append(head_pose_signal)

        symmetry_signal = self._check_symmetry(series)
        if symmetry_signal is not None:
            signals.append(symmetry_signal)

        return signals

    # --- individual checks -------------------------------------------------

    def _check_bone_lengths(self, series: List[Dict[str, float]]) -> Optional[Signal]:
        worst_z, worst_name = 0.0, None

        for name in _NON_SCALE_POSE_BONES:
            ratios = [m[f"bone:{name}"] / m["scale:torso"] for m in series
                      if f"bone:{name}" in m and m.get("scale:torso", 0) > 1e-3]
            z = self._evaluate_ratio_series(ratios, BONE_LENGTH_MIN_REL_CHANGE)
            if abs(z) > abs(worst_z):
                worst_z, worst_name = z, f"pose:{name}"

        for key, scale_key in (("bone:interocular", "scale:face"), ("bone:mouth_width", "scale:face"),
                                ("bone:face_height", "scale:face")):
            ratios = [m[key] / m[scale_key] for m in series if key in m and m.get(scale_key, 0) > 1e-3]
            z = self._evaluate_ratio_series(ratios, BONE_LENGTH_MIN_REL_CHANGE)
            if abs(z) > abs(worst_z):
                worst_z, worst_name = z, key

        if worst_name is None or abs(worst_z) < BONE_LENGTH_Z_THRESHOLD:
            return None
        score = _clip01((abs(worst_z) - BONE_LENGTH_Z_THRESHOLD) / (BONE_LENGTH_Z_SATURATE - BONE_LENGTH_Z_THRESHOLD))
        return Signal("bone_length", score, f"Rigid segment length changed implausibly ({worst_name}, z={worst_z:.1f})")

    def _check_joint_angles(self, series: List[Dict[str, float]]) -> Optional[Signal]:
        worst_z, worst_name, worst_delta = 0.0, None, 0.0

        for name in POSE_JOINT_ANGLES:
            key = f"angle:{name}"
            values = [m[key] for m in series if key in m]
            if len(values) < MIN_HISTORY_FRAMES:
                continue
            deltas = [b - a for a, b in zip(values, values[1:])]
            if len(deltas) < 6:
                continue
            z = LandmarkHistory.robust_zscore_of_last(deltas)
            if abs(deltas[-1]) < JOINT_ANGLE_MIN_DELTA_DEG:
                continue
            if abs(z) > abs(worst_z):
                worst_z, worst_name, worst_delta = z, name, deltas[-1]

        if worst_name is None or abs(worst_z) < JOINT_ANGLE_Z_THRESHOLD:
            return None
        score = _clip01((abs(worst_z) - JOINT_ANGLE_Z_THRESHOLD) / (JOINT_ANGLE_Z_SATURATE - JOINT_ANGLE_Z_THRESHOLD))
        return Signal("joint_angle_motion", score,
                      f"Implausibly abrupt joint bend ({worst_name}, {worst_delta:+.0f} deg, z={worst_z:.1f})")

    def _check_head_pose(self, series: List[Dict[str, float]]) -> Optional[Signal]:
        values = [m["head_pose_residual"] for m in series if "head_pose_residual" in m]
        if len(values) < MIN_HISTORY_FRAMES or values[-1] < HEAD_POSE_MIN_RESIDUAL_RATIO:
            return None
        z = LandmarkHistory.robust_zscore_of_last(values)
        if z < HEAD_POSE_Z_THRESHOLD:
            return None
        score = _clip01((z - HEAD_POSE_Z_THRESHOLD) / (HEAD_POSE_Z_SATURATE - HEAD_POSE_Z_THRESHOLD))
        return Signal("head_pose_reprojection", score,
                      f"Face landmarks don't fit a rigid head model (residual={values[-1]:.3f}, z={z:.1f})")

    def _check_symmetry(self, series: List[Dict[str, float]]) -> Optional[Signal]:
        worst_z, worst_name = 0.0, None
        for key in ("symmetry:arm_ratio", "symmetry:leg_ratio", "symmetry:eye_ratio"):
            values = [m[key] for m in series if key in m]
            z = self._evaluate_ratio_series(values, SYMMETRY_MIN_REL_CHANGE)
            if abs(z) > abs(worst_z):
                worst_z, worst_name = z, key

        if worst_name is None or abs(worst_z) < SYMMETRY_Z_THRESHOLD:
            return None
        score = _clip01((abs(worst_z) - SYMMETRY_Z_THRESHOLD) / (SYMMETRY_Z_SATURATE - SYMMETRY_Z_THRESHOLD))
        return Signal("symmetry_break", score, f"Left/right symmetry broke unexpectedly ({worst_name}, z={worst_z:.1f})")

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
