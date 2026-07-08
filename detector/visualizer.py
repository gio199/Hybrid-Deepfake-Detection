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
from .object_detection import TrackedObject

_FaceConn = vision.FaceLandmarksConnections
_face_connection_groups = (
    _FaceConn.FACE_LANDMARKS_CONTOURS + _FaceConn.FACE_LANDMARKS_LEFT_IRIS + _FaceConn.FACE_LANDMARKS_RIGHT_IRIS
)
FACE_MESH_CONNECTIONS = [(c.start, c.end) for c in _face_connection_groups]
POSE_CONNECTIONS = [(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS]
HAND_CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]

MESH_COLOR = (200, 180, 60)       # subtle teal/cyan (BGR) - person 0's default color
SKELETON_COLOR = (80, 200, 80)    # green (BGR)
HAND_COLOR = (180, 120, 255)      # pink (BGR)
HIGHLIGHT_COLOR = (0, 0, 255)     # red (BGR)
FRAME_ALERT_COLOR = (0, 0, 255)
OBJECT_COLOR = (0, 165, 255)      # orange (BGR) - normal (non-flagged) tracked object box

# Each tracked person is drawn (mesh + skeleton + hands) in one color from
# this palette, rotating by `person_id % len(_PERSON_PALETTE)`, so multiple
# people are easy to visually tell apart. Pure red is deliberately excluded
# here since it's reserved for anomaly highlights (`HIGHLIGHT_COLOR`).
_PERSON_PALETTE = [
    (200, 180, 60),   # teal
    (255, 150, 50),   # blue
    (60, 200, 200),   # yellow
    (200, 100, 220),  # purple/magenta
    (100, 220, 100),  # light green
    (180, 180, 180),  # light gray
]


def _person_color(person_id: int) -> Tuple[int, int, int]:
    return _PERSON_PALETTE[person_id % len(_PERSON_PALETTE)]

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


def _draw_person(frame: np.ndarray, person_id: int, fl: FrameLandmarks, category_scores: Dict[str, float]) -> None:
    color = _person_color(person_id)
    fired = {name for name, score in category_scores.items() if score > 0.05}

    label_point = None

    if fl.face_present and fl.face is not None:
        face_points = [(p.x, p.y) for p in fl.face]
        draw_connections(frame, face_points, FACE_MESH_CONNECTIONS, color)
        label_point = face_points[FACE_FOREHEAD] if FACE_FOREHEAD < len(face_points) else face_points[0]

        highlight_idxs: set = set()
        for cat in fired:
            highlight_idxs.update(_CATEGORY_FACE_HIGHLIGHTS.get(cat, []))
        for idx in highlight_idxs:
            if idx < len(face_points):
                x, y = face_points[idx]
                cv2.circle(frame, (int(x), int(y)), 4, HIGHLIGHT_COLOR, -1, cv2.LINE_AA)

    if fl.pose_present and fl.pose is not None:
        pose_points = [(p.x, p.y) if p.visibility > 0.3 else None for p in fl.pose]
        draw_connections(frame, pose_points, POSE_CONNECTIONS, color)
        if label_point is None:
            nose = pose_points[0] if pose_points else None
            label_point = nose

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
            draw_connections(frame, hand_points, HAND_CONNECTIONS, color)
            for idx in hand_highlight_idxs:
                if idx < len(hand_points):
                    x, y = hand_points[idx]
                    cv2.circle(frame, (int(x), int(y)), 3, HIGHLIGHT_COLOR, -1, cv2.LINE_AA)

    if label_point is not None:
        x, y = label_point
        cv2.putText(frame, f"P{person_id}", (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def _draw_objects(frame: np.ndarray, objects: List[TrackedObject], flagged_object_ids: set) -> None:
    for obj in objects:
        box = obj.box
        flagged = obj.object_id in flagged_object_ids
        color = HIGHLIGHT_COLOR if flagged else OBJECT_COLOR
        thickness = 3 if flagged else 1
        p0, p1 = (int(box.x0), int(box.y0)), (int(box.x1), int(box.y1))
        cv2.rectangle(frame, p0, p1, color, thickness, cv2.LINE_AA)
        label = f"{obj.category} {box.score:.2f}"
        cv2.putText(frame, label, (p0[0], max(0, p0[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_overlay(frame_bgr: np.ndarray, people: List[Tuple[int, FrameLandmarks, Dict[str, float]]],
                  objects: List[TrackedObject], object_flagged_ids: set,
                  frame_category_scores: Dict[str, float], combined_score: float, rolling_score: float) -> np.ndarray:
    """Draws every currently-tracked person (each in its own color, labeled
    `P<id>`, with that person's own anomaly highlights) and every
    currently-visible tracked object (boxed and labeled, highlighted red if
    an object-level anomaly fired on it this frame), plus the running score
    panel (driven by `frame_category_scores` - the worst tracked person's
    signals plus any object-level signals, i.e. exactly what was scored for
    this frame).
    """
    frame = frame_bgr  # draw in place; caller owns the frame buffer for this iteration

    for person_id, fl, category_scores in people:
        _draw_person(frame, person_id, fl, category_scores)

    _draw_objects(frame, objects, object_flagged_ids)

    fired = {name for name, score in frame_category_scores.items() if score > 0.05}
    if fired & _FRAME_LEVEL_CATEGORIES:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), FRAME_ALERT_COLOR, 6, cv2.LINE_AA)

    _draw_score_panel(frame, frame_category_scores, combined_score, rolling_score, len(people))
    return frame


def _draw_score_panel(frame: np.ndarray, category_scores: Dict[str, float], combined_score: float,
                       rolling_score: float, people_count: int = 0) -> None:
    panel_w = 300
    top_categories = sorted(category_scores.items(), key=lambda kv: kv[1], reverse=True)[:4]
    panel_h = 66 + 20 * max(1, len(top_categories))

    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)

    header_color = _score_color(rolling_score)
    cv2.putText(frame, f"Fake likelihood (rolling): {rolling_score * 100:4.1f}%", (18, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, header_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"People tracked: {people_count}", (18, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    y = 70
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
