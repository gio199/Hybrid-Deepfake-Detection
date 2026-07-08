"""Draws face mesh / pose skeleton overlays plus anomaly highlights and a
running fake-likelihood score panel, and writes the annotated video.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from mediapipe.tasks.python import vision

from .landmarks import (
    FrameLandmarks,
    FACE_POSE_LANDMARK_IDS,
    FACE_LEFT_IRIS_CENTER, FACE_RIGHT_IRIS_CENTER, FACE_LEFT_MOUTH_CORNER, FACE_RIGHT_MOUTH_CORNER,
    FACE_FOREHEAD, FACE_CHIN, FACE_LEFT_EYE, FACE_RIGHT_EYE,
    POSE_BONES, POSE_JOINT_ANGLES,
)
from .glitch_detection import REPRESENTATIVE_FACE_POINTS, REPRESENTATIVE_POSE_POINTS

_FaceConn = vision.FaceLandmarksConnections
_face_connection_groups = (
    _FaceConn.FACE_LANDMARKS_CONTOURS + _FaceConn.FACE_LANDMARKS_LEFT_IRIS + _FaceConn.FACE_LANDMARKS_RIGHT_IRIS
)
FACE_MESH_CONNECTIONS = [(c.start, c.end) for c in _face_connection_groups]
POSE_CONNECTIONS = [(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS]
HAND_CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]

MESH_COLOR = (200, 180, 60)       # subtle teal/cyan (BGR)
SKELETON_COLOR = (80, 200, 80)    # green (BGR)
HAND_COLOR = (180, 120, 255)      # pink (BGR)
HIGHLIGHT_COLOR = (0, 0, 255)     # red (BGR)
FRAME_ALERT_COLOR = (0, 0, 255)

# Which landmark indices to highlight per fired anomaly category.
_BONE_FACE_POINTS = [FACE_LEFT_IRIS_CENTER, FACE_RIGHT_IRIS_CENTER, FACE_LEFT_MOUTH_CORNER,
                      FACE_RIGHT_MOUTH_CORNER, FACE_FOREHEAD, FACE_CHIN]
_BONE_POSE_POINTS = sorted({idx for pair in POSE_BONES.values() for idx in pair})
_JOINT_VERTEX_POINTS = sorted({vertex for (_, vertex, _) in POSE_JOINT_ANGLES.values()})
_SYMMETRY_POINTS = [FACE_LEFT_EYE[0], FACE_LEFT_EYE[3], FACE_RIGHT_EYE[0], FACE_RIGHT_EYE[3]]
_ALL_HAND_POINTS = list(range(21))

_CATEGORY_FACE_HIGHLIGHTS: Dict[str, List[int]] = {
    "landmark_jitter": REPRESENTATIVE_FACE_POINTS,
    "landmark_jump": REPRESENTATIVE_FACE_POINTS,
    "bone_length": _BONE_FACE_POINTS,
    "head_pose_reprojection": FACE_POSE_LANDMARK_IDS,
    "symmetry_break": _SYMMETRY_POINTS,
}
_CATEGORY_POSE_HIGHLIGHTS: Dict[str, List[int]] = {
    "landmark_jitter": REPRESENTATIVE_POSE_POINTS,
    "landmark_jump": REPRESENTATIVE_POSE_POINTS,
    "bone_length": _BONE_POSE_POINTS,
    "joint_angle_motion": _JOINT_VERTEX_POINTS,
}
_CATEGORY_HAND_HIGHLIGHTS: Dict[str, List[int]] = {
    "hand_blur_anomaly": _ALL_HAND_POINTS,
    "finger_bone_length": _ALL_HAND_POINTS,
    "finger_joint_motion": _ALL_HAND_POINTS,
}
_FRAME_LEVEL_CATEGORIES = {"detection_flicker", "pixel_flicker", "blur_mismatch", "blur_onset_spike"}


def _score_color(score: float) -> Tuple[int, int, int]:
    """Green -> yellow -> red gradient (BGR) for a 0..1 score."""
    score = max(0.0, min(1.0, score))
    if score < 0.5:
        t = score / 0.5
        b, g, r = 0, 200, int(200 * t)
    else:
        t = (score - 0.5) / 0.5
        b, g, r = 0, int(200 * (1 - t)), 200
    return (b, g, r)


def draw_connections(frame: np.ndarray, points: Sequence[Optional[Tuple[float, float]]],
                      connections: List[Tuple[int, int]], color: Tuple[int, int, int]) -> None:
    for a, b in connections:
        if a >= len(points) or b >= len(points):
            continue
        pa, pb = points[a], points[b]
        if pa is None or pb is None:
            continue
        cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), color, 1, cv2.LINE_AA)


def draw_overlay(frame_bgr: np.ndarray, fl: FrameLandmarks, category_scores: Dict[str, float],
                  combined_score: float, rolling_score: float) -> np.ndarray:
    frame = frame_bgr  # draw in place; caller owns the frame buffer for this iteration

    fired = {name for name, score in category_scores.items() if score > 0.05}

    if fl.face_present and fl.face is not None:
        face_points = [(p.x, p.y) for p in fl.face]
        draw_connections(frame, face_points, FACE_MESH_CONNECTIONS, MESH_COLOR)

        highlight_idxs: set = set()
        for cat in fired:
            highlight_idxs.update(_CATEGORY_FACE_HIGHLIGHTS.get(cat, []))
        for idx in highlight_idxs:
            if idx < len(face_points):
                x, y = face_points[idx]
                cv2.circle(frame, (int(x), int(y)), 4, HIGHLIGHT_COLOR, -1, cv2.LINE_AA)

    if fl.pose_present and fl.pose is not None:
        pose_points = [(p.x, p.y) if p.visibility > 0.3 else None for p in fl.pose]
        draw_connections(frame, pose_points, POSE_CONNECTIONS, SKELETON_COLOR)

        highlight_idxs = set()
        for cat in fired:
            highlight_idxs.update(_CATEGORY_POSE_HIGHLIGHTS.get(cat, []))
        for idx in highlight_idxs:
            if idx < len(pose_points) and pose_points[idx] is not None:
                x, y = pose_points[idx]
                cv2.circle(frame, (int(x), int(y)), 5, HIGHLIGHT_COLOR, -1, cv2.LINE_AA)

    if fl.hands_present and fl.hands:
        hand_highlight_idxs: set = set()
        for cat in fired:
            hand_highlight_idxs.update(_CATEGORY_HAND_HIGHLIGHTS.get(cat, []))
        for hand in fl.hands:
            hand_points = [(p.x, p.y) for p in hand]
            draw_connections(frame, hand_points, HAND_CONNECTIONS, HAND_COLOR)
            for idx in hand_highlight_idxs:
                if idx < len(hand_points):
                    x, y = hand_points[idx]
                    cv2.circle(frame, (int(x), int(y)), 3, HIGHLIGHT_COLOR, -1, cv2.LINE_AA)

    if fired & _FRAME_LEVEL_CATEGORIES:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), FRAME_ALERT_COLOR, 6, cv2.LINE_AA)

    _draw_score_panel(frame, category_scores, combined_score, rolling_score)
    return frame


def _draw_score_panel(frame: np.ndarray, category_scores: Dict[str, float], combined_score: float,
                       rolling_score: float) -> None:
    panel_w = 300
    top_categories = sorted(category_scores.items(), key=lambda kv: kv[1], reverse=True)[:4]
    panel_h = 46 + 20 * max(1, len(top_categories))

    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)

    header_color = _score_color(rolling_score)
    cv2.putText(frame, f"Fake likelihood (rolling): {rolling_score * 100:4.1f}%", (18, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, header_color, 2, cv2.LINE_AA)

    y = 50
    for name, score in top_categories:
        bar_max = 120
        bar_len = int(bar_max * max(0.0, min(1.0, score)))
        cv2.putText(frame, name, (18, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (170, y + 2), (170 + bar_max, y + 14), (70, 70, 70), 1)
        if bar_len > 0:
            cv2.rectangle(frame, (170, y + 2), (170 + bar_len, y + 14), _score_color(score), -1)
        y += 20


class AnnotatedVideoWriter:
    """Thin wrapper around cv2.VideoWriter that opens lazily on first frame."""

    def __init__(self, output_path: str, fps: float, frame_size: Tuple[int, int]):
        self.output_path = output_path
        self.fps = fps if fps > 1e-3 else 30.0
        self.frame_size = frame_size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(output_path, fourcc, self.fps, frame_size)
        if not self._writer.isOpened():
            raise IOError(f"Could not open video writer for: {output_path}")

    def write(self, frame_bgr: np.ndarray) -> None:
        self._writer.write(frame_bgr)

    def release(self) -> None:
        self._writer.release()

    def __enter__(self) -> "AnnotatedVideoWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
