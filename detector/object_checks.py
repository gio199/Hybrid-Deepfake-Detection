"""Object-level anomaly checks: does a tracked object flicker in and out
of detection, suddenly change size/shape, or "teleport" - the same
temporal-implausibility idea as `glitch_detection.py`'s landmark checks,
applied to generic tracked objects instead of body/face points.

Rationale: generative video pipelines don't just get hands and faces
wrong - background/foreground objects can morph, flicker, or drift in
physically implausible ways too (a common visible tell in AI video, e.g.
a cup that subtly changes shape between frames, or an object that
flickers in and out as the model's temporal consistency breaks down).

CALIBRATION RESULT: tried, measured, and excluded from the score (kept
computed/visible for reference, same treatment as `hand_checks.py`'s
finger-geometry checks). EfficientDet-Lite0's own detection noise turned
out to be at least as large on real footage as on confirmed-fake footage,
so these checks have no practical discriminative power:
  - on `real_baseline.mp4` (genuine webcam clip), a misclassified
    background object flickered on/off as "tv" 4-8x in a 15-frame window
    (`object_flicker` up to 1.0) and its box area jumped hard enough to
    saturate the z-score check (z > 30) - purely from detector instability
    on an ordinary, static background object.
  - on a confirmed-fake clip, the same failure mode showed up just as
    readily (a person's head flickering between "umbrella" and no
    detection 7-10x in a window) - i.e. exactly the same noise signature,
    not a fake-specific tell.
Because of this, `object_flicker`/`object_size_jump`/`object_teleport` are
all weighted to 0.0 in `scoring.DEFAULT_WEIGHTS`. They still run every
frame and are still drawn in the annotated video/JSON report, in case a
user wants to manually inspect them for a specific clip - they're just not
trusted as automatic scoring evidence given the above.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .history import LandmarkHistory, Signal
from .object_detection import TrackedObject

# --- Tunable thresholds -------------------------------------------------

MIN_HISTORY_FRAMES = 8
MIN_OBJECT_SCALE_PX = 20.0     # ignore tiny/far objects - too noisy to trust

FLICKER_MIN_TRANSITIONS = 3    # presence on/off flips within the tracked window
FLICKER_SATURATE_TRANSITIONS = 7.0

SIZE_JUMP_Z_THRESHOLD = 3.5
SIZE_JUMP_Z_SATURATE = 8.0
SIZE_JUMP_MIN_REL_CHANGE = 0.25   # ignore small area changes - ordinary detector jitter

TELEPORT_Z_THRESHOLD = 3.5
TELEPORT_Z_SATURATE = 8.0
TELEPORT_MIN_REL_DISPLACEMENT = 0.5   # as a fraction of the object's own size


def _clip01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


class ObjectChecker:
    """Stateless: derives everything from the `TrackedObject` histories
    the caller already maintains via `object_detection.ObjectTracker`.
    """

    def __init__(self) -> None:
        # Populated by the most recent `analyze()` call: {signal_name: object_id}
        # for whichever fired this call - lets the caller (main.py/visualizer)
        # highlight the specific offending box without parsing signal text.
        self.last_flagged_ids: Dict[str, int] = {}

    def analyze(self, tracks: List[TrackedObject]) -> List[Signal]:
        self.last_flagged_ids = {}
        signals: List[Signal] = []

        flicker = self._check_flicker(tracks)
        if flicker is not None:
            signals.append(flicker)

        size_jump = self._check_size_jump(tracks)
        if size_jump is not None:
            signals.append(size_jump)

        teleport = self._check_teleport(tracks)
        if teleport is not None:
            signals.append(teleport)

        return signals

    def _check_flicker(self, tracks: List[TrackedObject]) -> Optional[Signal]:
        worst_score, worst_label, worst_id, worst_transitions = 0.0, None, None, 0
        for t in tracks:
            if not t.is_present or len(t.presence_history) < MIN_HISTORY_FRAMES:
                continue
            history = list(t.presence_history)
            transitions = sum(1 for a, b in zip(history, history[1:]) if a != b)
            if transitions < FLICKER_MIN_TRANSITIONS:
                continue
            score = _clip01((transitions - FLICKER_MIN_TRANSITIONS) /
                             (FLICKER_SATURATE_TRANSITIONS - FLICKER_MIN_TRANSITIONS))
            if score > worst_score:
                worst_score, worst_label, worst_id = score, f"{t.category}#{t.object_id}", t.object_id
                worst_transitions = transitions

        if worst_label is None:
            return None
        self.last_flagged_ids["object_flicker"] = worst_id
        return Signal("object_flicker", worst_score,
                      f"Object '{worst_label}' detection flickered on/off {worst_transitions}x recently")

    def _check_size_jump(self, tracks: List[TrackedObject]) -> Optional[Signal]:
        worst_z, worst_label, worst_id = 0.0, None, None
        for t in tracks:
            if not t.is_present:
                continue
            boxes = [b for b in t.boxes_history if b.width >= MIN_OBJECT_SCALE_PX and b.height >= MIN_OBJECT_SCALE_PX]
            if len(boxes) < MIN_HISTORY_FRAMES:
                continue
            areas = [b.area for b in boxes]
            ratios = [b / a for a, b in zip(areas, areas[1:]) if a > 1e-3]
            if len(ratios) < MIN_HISTORY_FRAMES - 1:
                continue
            if abs(ratios[-1] - 1.0) < SIZE_JUMP_MIN_REL_CHANGE:
                continue
            z = LandmarkHistory.robust_zscore_of_last(ratios)
            if abs(z) > abs(worst_z):
                worst_z, worst_label, worst_id = z, f"{t.category}#{t.object_id}", t.object_id

        if worst_label is None or abs(worst_z) < SIZE_JUMP_Z_THRESHOLD:
            return None
        score = _clip01((abs(worst_z) - SIZE_JUMP_Z_THRESHOLD) / (SIZE_JUMP_Z_SATURATE - SIZE_JUMP_Z_THRESHOLD))
        self.last_flagged_ids["object_size_jump"] = worst_id
        return Signal("object_size_jump", score,
                      f"Object '{worst_label}' bounding-box area changed implausibly fast (z={worst_z:.1f})")

    def _check_teleport(self, tracks: List[TrackedObject]) -> Optional[Signal]:
        worst_z, worst_label, worst_id = 0.0, None, None
        for t in tracks:
            if not t.is_present:
                continue
            boxes = [b for b in t.boxes_history if b.width >= MIN_OBJECT_SCALE_PX and b.height >= MIN_OBJECT_SCALE_PX]
            if len(boxes) < MIN_HISTORY_FRAMES:
                continue
            ratios = []
            for a, b in zip(boxes, boxes[1:]):
                size_proxy = max(float(np.sqrt(max(a.area, 1e-3))), 1e-3)
                disp = float(np.hypot(*(np.subtract(b.center, a.center))))
                ratios.append(disp / size_proxy)
            if len(ratios) < MIN_HISTORY_FRAMES - 1:
                continue
            if ratios[-1] < TELEPORT_MIN_REL_DISPLACEMENT:
                continue
            z = LandmarkHistory.robust_zscore_of_last(ratios)
            if abs(z) > abs(worst_z):
                worst_z, worst_label, worst_id = z, f"{t.category}#{t.object_id}", t.object_id

        if worst_label is None or abs(worst_z) < TELEPORT_Z_THRESHOLD:
            return None
        score = _clip01((abs(worst_z) - TELEPORT_Z_THRESHOLD) / (TELEPORT_Z_SATURATE - TELEPORT_Z_THRESHOLD))
        self.last_flagged_ids["object_teleport"] = worst_id
        return Signal("object_teleport", score,
                      f"Object '{worst_label}' position jumped implausibly (z={worst_z:.1f})")
