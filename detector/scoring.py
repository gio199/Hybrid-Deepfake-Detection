"""Aggregates per-frame glitch/physics evidence into a video anomaly score.

Per frame, correlated signals are first collapsed by evidence group, then
the group maxima are combined with a weighted noisy-OR. This prevents one
bad landmark estimate from multiplying into several supposedly independent
pieces of evidence. The final video-level score blends how *anomalous*
frames are on average with how *often* frames cross the flag threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .history import Signal

# Physics-derived checks are weighted higher than pure motion/pixel checks
# since they're more specific to rigid-body/anatomical implausibility and
# less prone to false positives from camera shake or fast real motion.
# Blur checks are weighted similarly high: selectively blurring a region
# (face seam, hands) is a common, fairly deliberate way generative/editing
# pipelines hide artifacts, so it's a fairly specific signal too.
#
# Note on finger bone-length/joint-angle consistency checks (tried and
# rejected, see detector/hand_checks.py): MediaPipe's hand landmarks are
# noisy enough under natural, fast hand articulation (finger foreshortening
# as they rotate towards/away from the camera, self-occlusion when
# fingers cross) that a genuine person gesturing expressively triggered
# these checks *more* than an actual deepfake did in testing - the
# opposite of what's needed to be useful evidence. They're implemented
# but intentionally left out of DEFAULT_WEIGHTS/main.py's pipeline.
#
# Note on `hand_blur_anomaly` (weighted to 0.0, i.e. computed but excluded
# from the score): the idea was sound (hands blurrier than the face might
# mean concealment) but testing showed the dominant real-world cause of a
# blurry hand is mundane depth-of-field - hands move toward/away from the
# camera during gesturing and drift out of the focal plane while the face
# (the autofocus target) stays sharp. A real gesturing video fired this on
# ~21% of frames even after gating out fast-motion-blur frames, while it
# never fired at all on either confirmed-real or confirmed-fake video in
# the test set - a 0% true-positive rate. Left computed (visible in the
# JSON `category_breakdown`) for informational/manual-review purposes
# only.
#
# Note on `object_flicker` / `object_size_jump` / `object_teleport`
# (weighted to 0.0, i.e. computed but excluded from the score - see
# object_checks.py): the idea was sound (generative video can make
# background/foreground objects morph, flicker, or drift implausibly,
# same as it does to faces/bodies), but validating against this project's
# test clips showed EfficientDet-Lite0 is simply too unreliable at this
# task on ordinary talking-head footage - it flickered a misclassified
# "tv" on/off in the *real* baseline clip's background 6x in a 15-frame
# window (score 0.75) and its bounding box jumped so hard between frames
# it saturated the z-score check (z>30), while on a *confirmed-fake* clip
# it just as readily flickered a person's head between "umbrella" and
# nothing 10x in a window (score 1.0) - i.e. the noise floor from the
# detector's own class-confusion and box instability is at least as large
# on real footage as on fake footage, so it has no discriminative power in
# practice. Left computed (visible in the JSON `category_breakdown` and
# drawn in the annotated video) for informational/manual-review purposes
# only, exactly like `hand_blur_anomaly`.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "landmark_jitter": 0.6,
    "landmark_jump": 1.0,
    "detection_flicker": 0.8,
    "pixel_flicker": 1.0,
    "bone_length": 1.3,
    "joint_angle_motion": 1.1,
    "head_pose_reprojection": 1.4,
    "symmetry_break": 1.0,
    "blur_mismatch": 1.1,
    "hand_blur_anomaly": 0.0,
    "blur_onset_spike": 0.9,
    "object_flicker": 0.0,
    "object_size_jump": 0.0,
    "object_teleport": 0.0,
    "learned_face": 1.0,
}

# Signals derived from the same underlying landmarks or pixels are
# correlated. Combining them independently lets one tracking failure
# multiply into several pieces of apparent evidence, so take the strongest
# contribution within a group and only noisy-OR across distinct groups.
SIGNAL_GROUPS: Dict[str, str] = {
    "landmark_jitter": "landmark_motion",
    "landmark_jump": "landmark_motion",
    "detection_flicker": "detection_reliability",
    "pixel_flicker": "pixel_temporal",
    "bone_length": "body_geometry",
    "joint_angle_motion": "body_geometry",
    "symmetry_break": "body_geometry",
    "head_pose_reprojection": "face_geometry",
    "blur_mismatch": "appearance_quality",
    "blur_onset_spike": "appearance_quality",
    "hand_blur_anomaly": "appearance_quality",
    "object_flicker": "object_reliability",
    "object_size_jump": "object_reliability",
    "object_teleport": "object_reliability",
    "learned_face": "learned_forensics",
}
FUSION_METHOD = "grouped-max-noisy-or-v2"


def combine_signals(signals: List[Signal], weights: Dict[str, float]) -> float:
    """Weighted noisy-OR combination of independent anomaly signals into a
    single [0, 1] score: individually-weak evidence compounds towards a
    high combined score, while one strong, highly-weighted signal (e.g. a
    severe head-pose reprojection failure) can push the score high on its
    own. Shared by `AnomalyAggregator` (per-frame scoring) and `main.py`
    (ranking which tracked person is "worst" in a multi-person frame) so
    both always agree on how signals combine.
    """
    grouped_contributions: Dict[str, float] = {}
    for s in signals:
        weight = weights.get(s.name, 1.0)
        contribution = _clip01(weight * s.score)
        group = SIGNAL_GROUPS.get(s.name, s.name)
        grouped_contributions[group] = max(grouped_contributions.get(group, 0.0), contribution)

    survival = 1.0
    for contribution in grouped_contributions.values():
        survival *= (1.0 - contribution)
    return _clip01(1.0 - survival)

DEFAULT_FRAME_FLAG_THRESHOLD = 0.4

# The final 0-100 score blends how anomalous frames are *on average* with
# how *often* frames cross the "flagged" threshold.
#
# Note on an alternative that was tried and rejected: weighting a "top-k
# burst" statistic into this formula (to better catch short, localized
# anomaly clusters) sounds appealing but empirically backfired - genuine
# footage routinely has a handful of single/few-frame MediaPipe tracking
# hiccups (fast head turns, brief occlusion) whose noisy-OR combined score
# saturates to ~1.0 just like a real deepfake artifact would, so a
# burst-weighted term pushed real calibration videos into "LIKELY FAKE"
# territory. Localized-burst intensity is instead surfaced separately as
# `peak_window_score` (see `_peak_window_score`) - an informational
# diagnostic that does NOT affect the score/verdict, so users can still
# notice "this clip has a short but severe anomaly cluster" without that
# fact silently corrupting the overall anomaly score.
MEAN_SCORE_WEIGHT = 0.5
FRAC_FLAGGED_WEIGHT = 0.5
PEAK_WINDOW_SECONDS = 0.75

# `blur_vs_bitrate_mismatch` (see quality_checks.py) is a whole-video statistic,
# not a per-frame one, so it can't sit in the per-frame noisy-OR combination
# above - it's added on top of the per-frame-derived final01 instead. It's
# additive rather than part of a weighted average because it should be able to
# meaningfully move a borderline score (e.g. a confirmed-fake compilation clip
# whose per-frame geometry looks mostly clean but is suspiciously soft despite
# a generous bitrate) without ever being able to manufacture a high score by
# itself from a single coarse whole-clip measurement, and without diluting the
# per-frame checks' influence on videos where it doesn't apply (webcam input,
# low-bitrate clips, or clips with too few tracked face frames all return no
# global signal at all, leaving the score identical to before this check
# existed). Calibrated on only 3 clips at time of writing - see
# quality_checks.py's module docstring for the numbers.
GLOBAL_QUALITY_WEIGHT = 0.15

VERDICT_LIKELY_FAKE_THRESHOLD = 55.0
VERDICT_SUSPICIOUS_THRESHOLD = 30.0
MIN_EVIDENCE_DURATION_SEC = 1.0
MIN_EVIDENCE_COVERAGE = 0.25


def _clip01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


@dataclass
class FrameScore:
    frame_idx: int
    timestamp_sec: float
    combined_score: float
    category_scores: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    evaluated: bool = True
    worst_person_id: Optional[int] = None
    people_present: int = 0


@dataclass
class FlaggedRange:
    start_frame: int
    end_frame: int
    peak_score: float
    categories: List[str]


@dataclass
class VideoResult:
    final_score: float
    verdict: str
    category_breakdown: Dict[str, Dict[str, float]]
    flagged_ranges: List[FlaggedRange]
    frame_scores: List[FrameScore]
    frames_evaluated: int
    frames_total: int
    peak_window_score: float = 0.0
    peak_window_timestamp_sec: float = 0.0
    global_quality_score: float = 0.0
    global_quality_reason: str = ""
    max_people_detected: int = 0
    people_track_count: int = 0
    evidence_coverage: float = 0.0
    evidence_duration_sec: float = 0.0
    evidence_warning: str = ""
    frame_flag_threshold: float = DEFAULT_FRAME_FLAG_THRESHOLD
    fusion_method: str = FUSION_METHOD


class AnomalyAggregator:
    """Accumulates per-frame signals over a whole video and produces a
    final VideoResult when `finalize()` is called.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 frame_flag_threshold: float = DEFAULT_FRAME_FLAG_THRESHOLD):
        self.weights = weights or DEFAULT_WEIGHTS
        self.frame_flag_threshold = frame_flag_threshold
        self._frame_scores: List[FrameScore] = []

    def add_frame(self, frame_idx: int, timestamp_sec: float, signals: List[Signal],
                  evaluated: bool, worst_person_id: Optional[int] = None,
                  people_present: int = 0) -> FrameScore:
        category_scores = {s.name: s.score for s in signals}
        reasons = [s.reason for s in signals if s.score > 0.05]
        combined = combine_signals(signals, self.weights) if evaluated else 0.0

        fs = FrameScore(
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            combined_score=combined,
            category_scores=category_scores,
            reasons=reasons,
            evaluated=evaluated,
            worst_person_id=worst_person_id,
            people_present=people_present,
        )
        self._frame_scores.append(fs)
        return fs

    def finalize(self, global_signals: Optional[List[Signal]] = None,
                 max_people_detected: int = 0, people_track_count: int = 0) -> VideoResult:
        evaluated_scores = [fs for fs in self._frame_scores if fs.evaluated]
        global_score, global_reason = self._worst_global_signal(global_signals)

        if not evaluated_scores:
            return VideoResult(
                final_score=0.0,
                verdict="NO_FACE_OR_BODY_DETECTED",
                category_breakdown={},
                flagged_ranges=[],
                frame_scores=self._frame_scores,
                frames_evaluated=0,
                frames_total=len(self._frame_scores),
                max_people_detected=max_people_detected,
                people_track_count=people_track_count,
                frame_flag_threshold=self.frame_flag_threshold,
            )

        scores = [fs.combined_score for fs in evaluated_scores]
        mean_score = float(np.mean(scores))
        flagged = [fs for fs in evaluated_scores if fs.combined_score >= self.frame_flag_threshold]
        frac_flagged = len(flagged) / len(evaluated_scores)

        final01 = (MEAN_SCORE_WEIGHT * mean_score + FRAC_FLAGGED_WEIGHT * frac_flagged
                   + GLOBAL_QUALITY_WEIGHT * global_score)
        final_score = round(100.0 * _clip01(final01), 1)

        peak_score, peak_ts = self._peak_window(evaluated_scores)
        coverage = len(evaluated_scores) / len(self._frame_scores) if self._frame_scores else 0.0
        frame_deltas = [
            b.timestamp_sec - a.timestamp_sec
            for a, b in zip(self._frame_scores, self._frame_scores[1:])
            if b.timestamp_sec > a.timestamp_sec
        ]
        nominal_delta = float(np.median(frame_deltas)) if frame_deltas else 0.0
        evidence_duration = len(evaluated_scores) * nominal_delta
        evidence_warnings = []
        if coverage < MIN_EVIDENCE_COVERAGE:
            evidence_warnings.append(
                f"only {coverage * 100:.1f}% of frames contained usable face/body evidence"
            )
        if evidence_duration < MIN_EVIDENCE_DURATION_SEC:
            evidence_warnings.append(
                f"only {evidence_duration:.2f}s of usable evidence was available"
            )
        evidence_warning = "; ".join(evidence_warnings)
        verdict = "INSUFFICIENT EVIDENCE" if evidence_warning else self._verdict(final_score)

        return VideoResult(
            final_score=final_score,
            verdict=verdict,
            category_breakdown=self._category_breakdown(evaluated_scores),
            flagged_ranges=self._flagged_ranges(),
            frame_scores=self._frame_scores,
            frames_evaluated=len(evaluated_scores),
            frames_total=len(self._frame_scores),
            peak_window_score=peak_score,
            peak_window_timestamp_sec=peak_ts,
            global_quality_score=global_score,
            global_quality_reason=global_reason,
            max_people_detected=max_people_detected,
            people_track_count=people_track_count,
            evidence_coverage=coverage,
            evidence_duration_sec=evidence_duration,
            evidence_warning=evidence_warning,
            frame_flag_threshold=self.frame_flag_threshold,
        )

    @staticmethod
    def _worst_global_signal(global_signals: Optional[List[Signal]]) -> "tuple[float, str]":
        if not global_signals:
            return 0.0, ""
        worst = max(global_signals, key=lambda s: s.score)
        return worst.score, worst.reason

    @staticmethod
    def _peak_window(evaluated_scores: List[FrameScore]) -> "tuple[float, float]":
        """Rolling-average combined score over a short (~0.5-1s) window,
        reported as an informational "worst moment" diagnostic - useful for
        spotting a short, severe anomaly cluster in an otherwise-clean
        clip (e.g. a compilation video with one bad cut). Deliberately
        excluded from the main score; see the note above `PEAK_WINDOW_FRAMES`.
        """
        if not evaluated_scores:
            return 0.0, 0.0

        left = 0
        running_sum = 0.0
        best_score = -1.0
        best_left = 0
        best_right = 0
        for right, frame_score in enumerate(evaluated_scores):
            running_sum += frame_score.combined_score
            while (
                left < right
                and frame_score.timestamp_sec - evaluated_scores[left].timestamp_sec > PEAK_WINDOW_SECONDS
            ):
                running_sum -= evaluated_scores[left].combined_score
                left += 1
            window_score = running_sum / (right - left + 1)
            if window_score > best_score:
                best_score = window_score
                best_left, best_right = left, right
        center_timestamp = (
            evaluated_scores[best_left].timestamp_sec + evaluated_scores[best_right].timestamp_sec
        ) / 2.0
        return float(max(best_score, 0.0)), float(center_timestamp)

    def _category_breakdown(self, evaluated_scores: List[FrameScore]) -> Dict[str, Dict[str, float]]:
        breakdown: Dict[str, Dict[str, float]] = {}
        for category in sorted(self.weights.keys()):
            values = [fs.category_scores.get(category, 0.0) for fs in evaluated_scores]
            fired = [v for v in values if v > 0.0]
            breakdown[category] = {
                "mean": float(np.mean(values)) if values else 0.0,
                "peak": float(max(values)) if values else 0.0,
                "frame_frac": float(len(fired) / len(values)) if values else 0.0,
            }
        return breakdown

    def _flagged_ranges(self) -> List[FlaggedRange]:
        ranges: List[FlaggedRange] = []
        current: Optional[FlaggedRange] = None
        current_categories: set = set()

        for fs in self._frame_scores:
            is_flagged = fs.evaluated and fs.combined_score >= self.frame_flag_threshold
            if is_flagged:
                if current is None:
                    current = FlaggedRange(start_frame=fs.frame_idx, end_frame=fs.frame_idx,
                                            peak_score=fs.combined_score, categories=[])
                    current_categories = set()
                current.end_frame = fs.frame_idx
                current.peak_score = max(current.peak_score, fs.combined_score)
                current_categories.update(k for k, v in fs.category_scores.items() if v > 0.05)
            else:
                if current is not None:
                    current.categories = sorted(current_categories)
                    ranges.append(current)
                    current = None

        if current is not None:
            current.categories = sorted(current_categories)
            ranges.append(current)

        return ranges

    @staticmethod
    def _verdict(score: float) -> str:
        if score >= VERDICT_LIKELY_FAKE_THRESHOLD:
            return "LIKELY FAKE"
        if score >= VERDICT_SUSPICIOUS_THRESHOLD:
            return "SUSPICIOUS / INCONCLUSIVE"
        return "LIKELY REAL"
