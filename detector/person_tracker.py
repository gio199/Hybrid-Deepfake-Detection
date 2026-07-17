"""Turns MediaPipe's raw, unassociated per-model detections (N faces, N
poses, 2N hands - no correspondence between them) into a stable set of
per-person `FrameLandmarks`, tracked across frames.

This is deliberately a lightweight tracker (velocity prediction plus
global Hungarian assignment, without a learned re-identification embedding).
It keeps stable identities through ordinary motion and brief occlusion
without pretending to solve full multi-object tracking. Two failure modes
remain:

  - Two people crossing paths very closely can swap IDs. This does not
    create a false "fake" signal by itself - the checks that depend on a
    continuous history (jitter/bone-length/etc.) simply see what looks
    like a brand new person and return 0 anomaly score until they
    accumulate enough fresh history again (see `MIN_HISTORY_FRAMES`-style
    guards throughout `glitch_detection.py`/`physics_checks.py`).
  - A face and body that belong together might occasionally fail to
    associate (e.g. a face partly out of frame while the body is fully
    visible) and get tracked as two separate "people" for a moment. This
    slightly inflates the people count but does not corrupt any one
    check's own-history comparison.

Measured/fixed failure mode: raising `num_poses` above 1 makes MediaPipe's
`PoseLandmarker` prone to emitting a spurious second (or third) "ghost"
pose for the *same* physical person - same rough position, a noticeably
lower confidence, and importantly a nose-to-nose distance from the real
pose that's tiny relative to body scale (measured on a real single-person
clip: consistently <0.22x shoulder-span, vs. a real second person who'd be
at least their own body-width away). `_deduplicate_poses`/`_deduplicate_faces`
below run a same-frame NMS pass (suppress a lower-confidence detection
whose anchor is within `POSE_NMS_RATIO`/`FACE_NMS_RATIO` of a kept one)
before any cross-model association happens, specifically to remove these
ghosts rather than let them become phantom extra tracked "people".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .landmarks import (
    FrameLandmarks, Landmark, RawFrameDetections,
    FACE_NOSE_TIP, HAND_WRIST, POSE_NOSE, POSE_LEFT_WRIST, POSE_RIGHT_WRIST,
)

# --- Tunable thresholds -------------------------------------------------

POSE_NMS_RATIO = 0.35         # suppress a lower-confidence pose whose nose is this close (x its scale) to a kept one
FACE_NMS_RATIO = 0.35         # same idea for faces, keyed off face width instead of a detection score (unavailable)

FACE_POSE_ASSOC_RATIO = 0.6   # max face<->pose nose distance, as a fraction of face width, to call them the same person
HAND_ASSOC_RATIO = 1.6        # max hand-wrist<->pose-wrist distance, as a fraction of person scale, to assign a hand
HAND_FALLBACK_ASSOC_RATIO = 1.2  # looser fallback: hand vs. any person's anchor point, for face-only/pose-only people

MATCH_DISTANCE_RATIO = 1.2    # max frame-to-frame anchor movement, as a fraction of the person's own scale, to keep the same id
MAX_MISSED_FRAMES = 15        # ~0.5s at 30fps - matches the flicker-tolerance window used elsewhere in this project
DEFAULT_SCALE_PX = 80.0       # fallback scale (px) if neither pose nor face size is available
VELOCITY_SMOOTHING = 0.65
INVALID_MATCH_COST = 1e6


def _dist_xy(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _dist_lm(a: Landmark, b: Landmark) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))


@dataclass
class _UnlabeledPerson:
    landmarks: FrameLandmarks
    anchor: Tuple[float, float]
    scale: float


def _nms_keep_indices(anchors: List[Tuple[float, float]], scales: List[float], scores: List[float],
                       ratio: float) -> List[int]:
    """Greedy same-frame NMS: processes detections best-score-first, keeps
    a detection unless its anchor is within `ratio * scale` of an
    already-kept (necessarily higher- or equal-scoring) detection.
    """
    order = sorted(range(len(anchors)), key=lambda i: -scores[i])
    kept: List[int] = []
    for i in order:
        is_duplicate = False
        for j in kept:
            threshold = ratio * max(scales[i], scales[j])
            if _dist_xy(anchors[i], anchors[j]) <= threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(i)
    return sorted(kept)


def _deduplicate_poses(
    raw: RawFrameDetections,
) -> Tuple[List[List[Landmark]], List[List[Landmark]], List[float]]:
    if len(raw.poses) <= 1:
        return raw.poses, raw.pose_worlds, raw.pose_scores
    anchors, scales, scores = [], [], []
    for i, p in enumerate(raw.poses):
        if len(p) <= POSE_NOSE:
            anchors.append((0.0, 0.0))
            scales.append(DEFAULT_SCALE_PX)
        else:
            anchors.append((p[POSE_NOSE].x, p[POSE_NOSE].y))
            l_sh, r_sh = p[11], p[12]
            span = _dist_lm(l_sh, r_sh)
            scales.append(span if span > 1e-3 else DEFAULT_SCALE_PX)
        scores.append(raw.pose_scores[i] if i < len(raw.pose_scores) else 0.0)
    keep = _nms_keep_indices(anchors, scales, scores, POSE_NMS_RATIO)
    worlds = [raw.pose_worlds[i] if i < len(raw.pose_worlds) else [] for i in keep]
    return [raw.poses[i] for i in keep], worlds, [raw.pose_scores[i] for i in keep]


def _deduplicate_faces(raw: RawFrameDetections) -> List[List[Landmark]]:
    if len(raw.faces) <= 1:
        return raw.faces
    anchors, scales = [], []
    for f in raw.faces:
        if len(f) <= 454:
            anchors.append((0.0, 0.0))
            scales.append(DEFAULT_SCALE_PX)
        else:
            anchors.append((f[FACE_NOSE_TIP].x, f[FACE_NOSE_TIP].y))
            scales.append(_dist_lm(f[234], f[454]))
    # No real per-face confidence is available from this API surface, so
    # larger (closer/more-visible) faces are preferred as a proxy - a
    # smaller/partial ghost duplicate is unlikely to be the more genuine one.
    keep = _nms_keep_indices(anchors, scales, scales, FACE_NMS_RATIO)
    return [raw.faces[i] for i in keep]


def _associate_frame(raw: RawFrameDetections) -> List[_UnlabeledPerson]:
    """Cross-model association within a single frame: pairs up faces,
    poses, and hands that likely belong to the same physical person.
    Returns one entry per detected person (face-only, pose-only, or both),
    with no cross-frame identity yet.
    """
    faces = _deduplicate_faces(raw)
    poses, pose_worlds, pose_scores = _deduplicate_poses(raw)
    n_faces = len(faces)
    n_poses = len(poses)

    face_scales = [_dist_lm(f[234], f[454]) if len(f) > 454 else DEFAULT_SCALE_PX for f in faces]

    # Greedy nearest-neighbor face<->pose association, cheapest pairs first.
    candidates = []
    for fi in range(n_faces):
        if len(faces[fi]) <= FACE_NOSE_TIP:
            continue
        face_anchor = (faces[fi][FACE_NOSE_TIP].x, faces[fi][FACE_NOSE_TIP].y)
        for pi in range(n_poses):
            if len(poses[pi]) <= POSE_NOSE:
                continue
            pose_anchor = (poses[pi][POSE_NOSE].x, poses[pi][POSE_NOSE].y)
            d = _dist_xy(face_anchor, pose_anchor)
            threshold = FACE_POSE_ASSOC_RATIO * face_scales[fi]
            if d <= threshold:
                candidates.append((d, fi, pi))
    candidates.sort(key=lambda c: c[0])

    matched_face_to_pose: Dict[int, int] = {}
    used_poses: set = set()
    used_faces: set = set()
    for d, fi, pi in candidates:
        if fi in used_faces or pi in used_poses:
            continue
        matched_face_to_pose[fi] = pi
        used_faces.add(fi)
        used_poses.add(pi)

    people: List[_UnlabeledPerson] = []

    def _make_person(face_idx: Optional[int], pose_idx: Optional[int]) -> _UnlabeledPerson:
        fl = FrameLandmarks(
            frame_idx=raw.frame_idx, timestamp_sec=raw.timestamp_sec,
            image_w=raw.image_w, image_h=raw.image_h,
        )
        if face_idx is not None:
            fl.face = faces[face_idx]
            fl.face_present = True
            fl.face_detection_score = 1.0
        if pose_idx is not None:
            fl.pose = poses[pose_idx]
            if pose_idx < len(pose_worlds) and pose_worlds[pose_idx]:
                fl.pose_world = pose_worlds[pose_idx]
            fl.pose_present = True
            fl.pose_detection_score = pose_scores[pose_idx] if pose_idx < len(pose_scores) else 0.0

        anchor = fl.anchor_point() or (raw.image_w / 2.0, raw.image_h / 2.0)
        scale = fl.scale_estimate() or DEFAULT_SCALE_PX
        return _UnlabeledPerson(landmarks=fl, anchor=anchor, scale=scale)

    for fi, pi in matched_face_to_pose.items():
        people.append(_make_person(fi, pi))
    for fi in range(n_faces):
        if fi not in used_faces:
            people.append(_make_person(fi, None))
    for pi in range(n_poses):
        if pi not in used_poses:
            people.append(_make_person(None, pi))

    _assign_hands(raw, people)
    return people


def _assign_hands(raw: RawFrameDetections, people: List[_UnlabeledPerson]) -> None:
    """Assigns each detected hand to the nearest person, preferring a
    person's pose wrist landmarks (precise) and falling back to a looser
    match against any person's general anchor point (for face-only people,
    or when the pose model missed the wrist specifically).
    """
    for hand_idx, hand in enumerate(raw.hands):
        if len(hand) <= HAND_WRIST:
            continue
        wrist = hand[HAND_WRIST]

        best_person: Optional[_UnlabeledPerson] = None
        best_dist = None
        for person in people:
            fl = person.landmarks
            if not fl.pose_present or fl.pose is None:
                continue
            for wrist_idx in (POSE_LEFT_WRIST, POSE_RIGHT_WRIST):
                if wrist_idx >= len(fl.pose):
                    continue
                d = _dist_lm(wrist, fl.pose[wrist_idx])
                threshold = HAND_ASSOC_RATIO * person.scale
                if d <= threshold and (best_dist is None or d < best_dist):
                    best_dist, best_person = d, person

        if best_person is None:
            # Fallback: nearest person's general anchor, looser threshold -
            # covers face-only people or a pose that's missing wrist points.
            for person in people:
                d = _dist_xy((wrist.x, wrist.y), person.anchor)
                threshold = HAND_FALLBACK_ASSOC_RATIO * person.scale
                if d <= threshold and (best_dist is None or d < best_dist):
                    best_dist, best_person = d, person

        if best_person is not None:
            best_person.landmarks.hands.append(hand)
            label = raw.handedness[hand_idx] if hand_idx < len(raw.handedness) else "Unknown"
            best_person.landmarks.handedness.append(label)
            best_person.landmarks.hands_present = True


@dataclass
class _Track:
    anchor: Tuple[float, float]
    scale: float
    missed_frames: int = 0
    velocity: Tuple[float, float] = (0.0, 0.0)
    last_timestamp_sec: float = 0.0


class MultiPersonTracker:
    """Assigns stable integer `person_id`s to the unlabeled per-frame
    people produced by `_associate_frame`, using velocity-predicted global
    assignment against the existing tracks.
    """

    def __init__(self, max_missed_frames: int = MAX_MISSED_FRAMES):
        self.max_missed_frames = max_missed_frames
        self._tracks: Dict[int, _Track] = {}
        self._next_id = 0
        self.total_people_seen = 0

    def update(self, raw: RawFrameDetections) -> List[Tuple[int, FrameLandmarks]]:
        people = _associate_frame(raw)

        matched_track_to_person: Dict[int, int] = {}
        used_people: set = set()
        track_ids = list(self._tracks)
        if track_ids and people:
            cost_matrix = np.full((len(track_ids), len(people)), INVALID_MATCH_COST, dtype=np.float64)
            for row, track_id in enumerate(track_ids):
                track = self._tracks[track_id]
                elapsed = max(0.0, raw.timestamp_sec - track.last_timestamp_sec)
                predicted_anchor = (
                    track.anchor[0] + track.velocity[0] * elapsed,
                    track.anchor[1] + track.velocity[1] * elapsed,
                )
                for column, person in enumerate(people):
                    distance = _dist_xy(predicted_anchor, person.anchor)
                    threshold = MATCH_DISTANCE_RATIO * max(track.scale, person.scale)
                    if distance <= threshold:
                        scale_penalty = abs(float(np.log(max(person.scale, 1e-3) / max(track.scale, 1e-3))))
                        cost_matrix[row, column] = distance / max(threshold, 1e-6) + 0.10 * scale_penalty

            rows, columns = linear_sum_assignment(cost_matrix)
            for row, column in zip(rows, columns):
                if cost_matrix[row, column] >= INVALID_MATCH_COST:
                    continue
                track_id = track_ids[row]
                person_idx = int(column)
                matched_track_to_person[track_id] = person_idx
                used_people.add(person_idx)

        results: List[Tuple[int, FrameLandmarks]] = []

        for track_id, person_idx in matched_track_to_person.items():
            person = people[person_idx]
            track = self._tracks[track_id]
            elapsed = raw.timestamp_sec - track.last_timestamp_sec
            if elapsed > 1e-6:
                measured_velocity = (
                    (person.anchor[0] - track.anchor[0]) / elapsed,
                    (person.anchor[1] - track.anchor[1]) / elapsed,
                )
                track.velocity = (
                    (1.0 - VELOCITY_SMOOTHING) * track.velocity[0]
                    + VELOCITY_SMOOTHING * measured_velocity[0],
                    (1.0 - VELOCITY_SMOOTHING) * track.velocity[1]
                    + VELOCITY_SMOOTHING * measured_velocity[1],
                )
            track.anchor = person.anchor
            track.scale = person.scale
            track.missed_frames = 0
            track.last_timestamp_sec = raw.timestamp_sec
            results.append((track_id, person.landmarks))

        for person_idx, person in enumerate(people):
            if person_idx in used_people:
                continue
            new_id = self._next_id
            self._next_id += 1
            self.total_people_seen += 1
            self._tracks[new_id] = _Track(
                anchor=person.anchor,
                scale=person.scale,
                missed_frames=0,
                last_timestamp_sec=raw.timestamp_sec,
            )
            results.append((new_id, person.landmarks))

        stale_ids = []
        for track_id in self._tracks:
            if track_id not in matched_track_to_person and all(rid != track_id for rid, _ in results):
                self._tracks[track_id].missed_frames += 1
                if self._tracks[track_id].missed_frames > self.max_missed_frames:
                    stale_ids.append(track_id)
        for track_id in stale_ids:
            del self._tracks[track_id]

        return results

    @property
    def active_person_count(self) -> int:
        return len(self._tracks)
