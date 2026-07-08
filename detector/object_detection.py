"""Generic object detection + lightweight cross-frame tracking.

Wraps MediaPipe's `ObjectDetector` Task (EfficientDet-Lite0, trained on
COCO's 80 everyday-object classes) and tracks each detected object across
frames with simple greedy IoU matching - the same "good enough, no
training required" philosophy as `person_tracker.py`, just simpler since
there's no cross-model association step (one model, one set of boxes).

Object detectors are noticeably noisier than the dedicated face/pose
models this project otherwise relies on: boxes jitter more, a detection
can drop for a frame or two even for a perfectly static real object, and
similar classes (e.g. "cup" vs "bowl") can flip. The tracker's thresholds
are deliberately more forgiving than `person_tracker.py`'s to absorb this,
and `object_checks.py`'s anomaly thresholds should be treated as a
hypothesis to validate against real footage, not an assumed win.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from .model_utils import ensure_model

DEFAULT_MAX_RESULTS = 15
DEFAULT_SCORE_THRESHOLD = 0.35

IOU_MATCH_THRESHOLD = 0.25     # lower than a typical tracker's ~0.3-0.5 - object boxes are noisier frame-to-frame
MAX_MISSED_FRAMES = 8          # shorter grace period than people (15) - object detections flicker more readily
HISTORY_WINDOW = 30


@dataclass
class ObjectBox:
    x0: float
    y0: float
    x1: float
    y1: float
    category: str
    score: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)


def _iou(a: ObjectBox, b: ObjectBox) -> float:
    ix0, iy0 = max(a.x0, b.x0), max(a.y0, b.y0)
    ix1, iy1 = min(a.x1, b.x1), min(a.y1, b.y1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = a.area + b.area - inter
    if union <= 1e-6:
        return 0.0
    return inter / union


class ObjectExtractor:
    """Runs MediaPipe's EfficientDet-Lite0 ObjectDetector (Tasks API) on frames."""

    def __init__(self, max_results: int = DEFAULT_MAX_RESULTS, score_threshold: float = DEFAULT_SCORE_THRESHOLD):
        model_path = ensure_model("efficientdet_lite0.tflite")
        options = vision.ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            max_results=max_results,
            score_threshold=score_threshold,
            # People are already tracked far more precisely by the dedicated
            # face/pose pipeline (`person_tracker.py`) - a generic "person"
            # box here would just be redundant clutter, not new information.
            category_denylist=["person"],
        )
        self._detector = vision.ObjectDetector.create_from_options(options)

    def process(self, frame_bgr: np.ndarray, frame_idx: int) -> List[ObjectBox]:
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect_for_video(mp_image, frame_idx)

        boxes: List[ObjectBox] = []
        for detection in result.detections:
            if not detection.categories:
                continue
            top = detection.categories[0]
            bb = detection.bounding_box
            boxes.append(ObjectBox(
                x0=float(bb.origin_x), y0=float(bb.origin_y),
                x1=float(bb.origin_x + bb.width), y1=float(bb.origin_y + bb.height),
                category=top.category_name or "object",
                score=float(top.score),
            ))
        return boxes

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> "ObjectExtractor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


@dataclass
class TrackedObject:
    object_id: int
    category: str
    box: ObjectBox
    boxes_history: Deque[ObjectBox] = field(default_factory=lambda: deque(maxlen=HISTORY_WINDOW))
    presence_history: Deque[bool] = field(default_factory=lambda: deque(maxlen=HISTORY_WINDOW))
    missed_frames: int = 0
    is_present: bool = True


class ObjectTracker:
    """Greedy same-category IoU matching to keep a stable `object_id` for
    each tracked object across frames.
    """

    def __init__(self, iou_threshold: float = IOU_MATCH_THRESHOLD, max_missed_frames: int = MAX_MISSED_FRAMES):
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self._tracks: Dict[int, TrackedObject] = {}
        self._next_id = 0

    def update(self, boxes: List[ObjectBox]) -> List[TrackedObject]:
        candidates = []
        for track_id, track in self._tracks.items():
            for box_idx, box in enumerate(boxes):
                if box.category != track.category:
                    continue
                iou = _iou(track.box, box)
                if iou >= self.iou_threshold:
                    candidates.append((iou, track_id, box_idx))
        candidates.sort(key=lambda c: -c[0])

        assignment: Dict[int, int] = {}
        used_tracks: set = set()
        used_boxes: set = set()
        for iou, track_id, box_idx in candidates:
            if track_id in used_tracks or box_idx in used_boxes:
                continue
            assignment[track_id] = box_idx
            used_tracks.add(track_id)
            used_boxes.add(box_idx)

        for track_id, track in self._tracks.items():
            if track_id in assignment:
                box = boxes[assignment[track_id]]
                track.box = box
                track.boxes_history.append(box)
                track.missed_frames = 0
                track.is_present = True
            else:
                track.missed_frames += 1
                track.is_present = False
            track.presence_history.append(track.is_present)

        for box_idx, box in enumerate(boxes):
            if box_idx in used_boxes:
                continue
            track_id = self._next_id
            self._next_id += 1
            new_track = TrackedObject(object_id=track_id, category=box.category, box=box)
            new_track.boxes_history.append(box)
            new_track.presence_history.append(True)
            self._tracks[track_id] = new_track

        stale_ids = [tid for tid, t in self._tracks.items() if t.missed_frames > self.max_missed_frames]
        for track_id in stale_ids:
            del self._tracks[track_id]

        return list(self._tracks.values())
