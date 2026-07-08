"""MediaPipe-based facial + body landmark extraction.

Wraps mediapipe's FaceMesh (refined, with iris points) and Pose solutions
and converts their normalized outputs into a simple, framework-agnostic
data structure (`FrameLandmarks`) used by the rest of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from .model_utils import ensure_model


# --- Named landmark indices (kept here so other modules share one source of truth) ---

# FaceMesh (refine_landmarks=True -> 478 points: 468 mesh + 10 iris)
FACE_LEFT_EYE = [362, 385, 387, 263, 373, 380]      # 6-point EAR ring, subject's left eye
FACE_RIGHT_EYE = [33, 160, 158, 133, 153, 144]      # 6-point EAR ring, subject's right eye
FACE_LEFT_IRIS_CENTER = 468
FACE_RIGHT_IRIS_CENTER = 473
FACE_NOSE_TIP = 1
FACE_CHIN = 152
FACE_FOREHEAD = 10
FACE_LEFT_CHEEK_EDGE = 234
FACE_RIGHT_CHEEK_EDGE = 454
FACE_LEFT_MOUTH_CORNER = 61
FACE_RIGHT_MOUTH_CORNER = 291
FACE_LEFT_EYE_OUTER_CORNER = 33
FACE_RIGHT_EYE_OUTER_CORNER = 263

# Classic 6-point correspondence used for rigid head-pose estimation via
# cv2.solvePnP (nose tip, chin, eye outer corners, mouth corners), paired
# with a generic anthropometric 3D face model (millimeters, arbitrary
# origin at the nose tip). This is the same well-known point set used in
# many dlib/OpenCV head-pose tutorials, remapped to MediaPipe indices.
FACE_POSE_LANDMARK_IDS = [
    FACE_NOSE_TIP, FACE_CHIN, FACE_LEFT_EYE_OUTER_CORNER,
    FACE_RIGHT_EYE_OUTER_CORNER, FACE_LEFT_MOUTH_CORNER, FACE_RIGHT_MOUTH_CORNER,
]

FACE_POSE_MODEL_3D = {
    FACE_NOSE_TIP: (0.0, 0.0, 0.0),
    FACE_CHIN: (0.0, -330.0, -65.0),
    FACE_LEFT_EYE_OUTER_CORNER: (225.0, 170.0, -135.0),
    FACE_RIGHT_EYE_OUTER_CORNER: (-225.0, 170.0, -135.0),
    FACE_LEFT_MOUTH_CORNER: (150.0, -150.0, -125.0),
    FACE_RIGHT_MOUTH_CORNER: (-150.0, -150.0, -125.0),
}

# Pose (33 body landmarks)
POSE_NOSE = 0
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_ELBOW = 13
POSE_RIGHT_ELBOW = 14
POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16
POSE_LEFT_HIP = 23
POSE_RIGHT_HIP = 24
POSE_LEFT_KNEE = 25
POSE_RIGHT_KNEE = 26
POSE_LEFT_ANKLE = 27
POSE_RIGHT_ANKLE = 28

# Rigid body segments assumed to have roughly constant length across a clip.
POSE_BONES = {
    "left_upper_arm": (POSE_LEFT_SHOULDER, POSE_LEFT_ELBOW),
    "left_forearm": (POSE_LEFT_ELBOW, POSE_LEFT_WRIST),
    "right_upper_arm": (POSE_RIGHT_SHOULDER, POSE_RIGHT_ELBOW),
    "right_forearm": (POSE_RIGHT_ELBOW, POSE_RIGHT_WRIST),
    "left_thigh": (POSE_LEFT_HIP, POSE_LEFT_KNEE),
    "left_shin": (POSE_LEFT_KNEE, POSE_LEFT_ANKLE),
    "right_thigh": (POSE_RIGHT_HIP, POSE_RIGHT_KNEE),
    "right_shin": (POSE_RIGHT_KNEE, POSE_RIGHT_ANKLE),
    "shoulder_span": (POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER),
    "hip_span": (POSE_LEFT_HIP, POSE_RIGHT_HIP),
}

# Joints for angle-limit checks: (name, point_a, vertex, point_b, max_valid_extension_deg)
POSE_JOINT_ANGLES = {
    "left_elbow": (POSE_LEFT_SHOULDER, POSE_LEFT_ELBOW, POSE_LEFT_WRIST),
    "right_elbow": (POSE_RIGHT_SHOULDER, POSE_RIGHT_ELBOW, POSE_RIGHT_WRIST),
    "left_knee": (POSE_LEFT_HIP, POSE_LEFT_KNEE, POSE_LEFT_ANKLE),
    "right_knee": (POSE_RIGHT_HIP, POSE_RIGHT_KNEE, POSE_RIGHT_ANKLE),
}

# Hand landmarker (21 points per hand): wrist + 4 points per finger (base -> tip).
HAND_WRIST = 0
HAND_THUMB_CMC, HAND_THUMB_MCP, HAND_THUMB_IP, HAND_THUMB_TIP = 1, 2, 3, 4
HAND_INDEX_MCP, HAND_INDEX_PIP, HAND_INDEX_DIP, HAND_INDEX_TIP = 5, 6, 7, 8
HAND_MIDDLE_MCP, HAND_MIDDLE_PIP, HAND_MIDDLE_DIP, HAND_MIDDLE_TIP = 9, 10, 11, 12
HAND_RING_MCP, HAND_RING_PIP, HAND_RING_DIP, HAND_RING_TIP = 13, 14, 15, 16
HAND_PINKY_MCP, HAND_PINKY_PIP, HAND_PINKY_DIP, HAND_PINKY_TIP = 17, 18, 19, 20

# Finger segments ("bones"), each expected to keep a roughly constant length
# relative to the palm across a clip - just like POSE_BONES, but for fingers,
# which is where deepfakes/generative pipelines most often go wrong.
HAND_BONES = {
    "thumb_1": (HAND_WRIST, HAND_THUMB_CMC),
    "thumb_2": (HAND_THUMB_CMC, HAND_THUMB_MCP),
    "thumb_3": (HAND_THUMB_MCP, HAND_THUMB_IP),
    "thumb_4": (HAND_THUMB_IP, HAND_THUMB_TIP),
    "index_1": (HAND_WRIST, HAND_INDEX_MCP),
    "index_2": (HAND_INDEX_MCP, HAND_INDEX_PIP),
    "index_3": (HAND_INDEX_PIP, HAND_INDEX_DIP),
    "index_4": (HAND_INDEX_DIP, HAND_INDEX_TIP),
    "middle_1": (HAND_WRIST, HAND_MIDDLE_MCP),
    "middle_2": (HAND_MIDDLE_MCP, HAND_MIDDLE_PIP),
    "middle_3": (HAND_MIDDLE_PIP, HAND_MIDDLE_DIP),
    "middle_4": (HAND_MIDDLE_DIP, HAND_MIDDLE_TIP),
    "ring_1": (HAND_WRIST, HAND_RING_MCP),
    "ring_2": (HAND_RING_MCP, HAND_RING_PIP),
    "ring_3": (HAND_RING_PIP, HAND_RING_DIP),
    "ring_4": (HAND_RING_DIP, HAND_RING_TIP),
    "pinky_1": (HAND_WRIST, HAND_PINKY_MCP),
    "pinky_2": (HAND_PINKY_MCP, HAND_PINKY_PIP),
    "pinky_3": (HAND_PINKY_PIP, HAND_PINKY_DIP),
    "pinky_4": (HAND_PINKY_DIP, HAND_PINKY_TIP),
}

# Finger knuckle joints for abrupt-bend checks (same "own recent baseline"
# approach as POSE_JOINT_ANGLES): (point_a, vertex, point_b).
HAND_JOINT_ANGLES = {
    "thumb_ip": (HAND_THUMB_MCP, HAND_THUMB_IP, HAND_THUMB_TIP),
    "index_pip": (HAND_INDEX_MCP, HAND_INDEX_PIP, HAND_INDEX_DIP),
    "index_dip": (HAND_INDEX_PIP, HAND_INDEX_DIP, HAND_INDEX_TIP),
    "middle_pip": (HAND_MIDDLE_MCP, HAND_MIDDLE_PIP, HAND_MIDDLE_DIP),
    "middle_dip": (HAND_MIDDLE_PIP, HAND_MIDDLE_DIP, HAND_MIDDLE_TIP),
    "ring_pip": (HAND_RING_MCP, HAND_RING_PIP, HAND_RING_DIP),
    "ring_dip": (HAND_RING_PIP, HAND_RING_DIP, HAND_RING_TIP),
    "pinky_pip": (HAND_PINKY_MCP, HAND_PINKY_PIP, HAND_PINKY_DIP),
    "pinky_dip": (HAND_PINKY_PIP, HAND_PINKY_DIP, HAND_PINKY_TIP),
}


@dataclass
class Landmark:
    x: float  # pixel coords
    y: float
    z: float = 0.0
    visibility: float = 1.0


@dataclass
class FrameLandmarks:
    """A single tracked *person's* landmarks in a single frame. When there
    are multiple people in frame, one of these exists per tracked person
    (see `person_tracker.py`, which builds these from the raw per-model
    detection lists `MediaPipeExtractor.process_multi()` returns).
    """

    frame_idx: int
    timestamp_sec: float
    image_w: int
    image_h: int
    face: Optional[List[Landmark]] = None
    pose: Optional[List[Landmark]] = None
    face_present: bool = False
    pose_present: bool = False
    face_detection_score: float = 0.0
    pose_detection_score: float = 0.0
    hands: List[List[Landmark]] = field(default_factory=list)
    handedness: List[str] = field(default_factory=list)
    hands_present: bool = False

    def face_point(self, idx: int) -> Optional[Landmark]:
        if not self.face_present or self.face is None:
            return None
        return self.face[idx]

    def pose_point(self, idx: int) -> Optional[Landmark]:
        if not self.pose_present or self.pose is None:
            return None
        return self.pose[idx]

    def face_width(self) -> Optional[float]:
        left = self.face_point(FACE_LEFT_CHEEK_EDGE)
        right = self.face_point(FACE_RIGHT_CHEEK_EDGE)
        if left is None or right is None:
            return None
        return float(np.hypot(left.x - right.x, left.y - right.y))

    def anchor_point(self) -> Optional[Tuple[float, float]]:
        """A single representative (x, y) location for this person, used
        by the cross-model association and cross-frame tracking in
        `person_tracker.py`. Prefers the pose nose (most stable, present
        even when the face is turned away) then falls back to the face
        nose tip.
        """
        if self.pose_present and self.pose is not None:
            p = self.pose[POSE_NOSE]
            return (p.x, p.y)
        if self.face_present and self.face is not None:
            p = self.face[FACE_NOSE_TIP]
            return (p.x, p.y)
        return None

    def scale_estimate(self) -> Optional[float]:
        """A rough person-size estimate (pixels) used to normalize
        tracking-match distances so the same absolute pixel gap counts as
        "close" for a large near-camera person and "far" for a small
        distant one.
        """
        if self.pose_present and self.pose is not None:
            l_sh = self.pose[POSE_LEFT_SHOULDER]
            r_sh = self.pose[POSE_RIGHT_SHOULDER]
            span = float(np.hypot(l_sh.x - r_sh.x, l_sh.y - r_sh.y))
            if span > 1e-3:
                return span
        width = self.face_width()
        if width:
            return width * 2.0  # shoulders are roughly ~2x face width
        return None


DEFAULT_MAX_PEOPLE = 1


@dataclass
class RawFrameDetections:
    """The raw, per-model, per-frame output of `MediaPipeExtractor.process_multi()` -
    i.e. before any cross-model association or cross-frame identity tracking has
    happened. `person_tracker.py` consumes this to build one `FrameLandmarks` per
    physical person.
    """

    frame_idx: int
    timestamp_sec: float
    image_w: int
    image_h: int
    faces: List[List[Landmark]] = field(default_factory=list)
    poses: List[List[Landmark]] = field(default_factory=list)
    pose_scores: List[float] = field(default_factory=list)
    hands: List[List[Landmark]] = field(default_factory=list)
    handedness: List[str] = field(default_factory=list)


class MediaPipeExtractor:
    """Runs MediaPipe's FaceLandmarker + PoseLandmarker + HandLandmarker
    (Tasks API) on frames, detecting up to `max_people` people at once.

    Note: the legacy `mediapipe.solutions.face_mesh` / `.pose` API was
    removed from the mediapipe package in version 0.10.30+, so this uses
    the modern Tasks API with locally-cached model bundles instead.
    """

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5,
                 max_people: int = DEFAULT_MAX_PEOPLE):
        self.max_people = max_people
        face_model_path = ensure_model("face_landmarker.task")
        pose_model_path = ensure_model("pose_landmarker_full.task")
        hand_model_path = ensure_model("hand_landmarker.task")

        face_options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=face_model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=max_people,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

        pose_options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=pose_model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=max_people,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

        hand_options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=hand_model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_people * 2,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

    def process_multi(self, frame_bgr: np.ndarray, frame_idx: int, timestamp_sec: float) -> RawFrameDetections:
        """Runs all three models on one frame and returns their raw,
        unassociated per-model detection lists (one entry per detected
        face/pose/hand, in whatever order MediaPipe returns them - there is
        no cross-model correspondence yet).
        """
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO running mode only requires a monotonically increasing
        # integer timestamp per stream; the frame index satisfies that
        # regardless of the source's real playback fps.
        timestamp_ms = frame_idx

        face_result = self._face_landmarker.detect_for_video(mp_image, timestamp_ms)
        pose_result = self._pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        hand_result = self._hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        result = RawFrameDetections(frame_idx=frame_idx, timestamp_sec=timestamp_sec, image_w=w, image_h=h)

        for mesh in face_result.face_landmarks:
            result.faces.append([Landmark(x=lm.x * w, y=lm.y * h, z=lm.z * w) for lm in mesh])

        for pts in pose_result.pose_landmarks:
            result.poses.append([
                Landmark(x=lm.x * w, y=lm.y * h, z=lm.z * w, visibility=lm.visibility or 1.0) for lm in pts
            ])
            visibilities = [lm.visibility or 1.0 for lm in pts]
            result.pose_scores.append(float(np.mean(visibilities)) if visibilities else 0.0)

        for hand in hand_result.hand_landmarks:
            result.hands.append([Landmark(x=lm.x * w, y=lm.y * h, z=lm.z * w) for lm in hand])
        result.handedness = [
            categories[0].category_name if categories else "Unknown"
            for categories in hand_result.handedness
        ]

        return result

    def close(self) -> None:
        self._face_landmarker.close()
        self._pose_landmarker.close()
        self._hand_landmarker.close()

    def __enter__(self) -> "MediaPipeExtractor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
