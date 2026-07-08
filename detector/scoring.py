"""Aggregates per-frame glitch/physics anomaly signals into a single
video-level fake-likelihood score.

Per frame, the individual category scores (each in [0, 1]) are combined
with a weighted noisy-OR: independent pieces of weak evidence compound
towards a high combined score, while a single strong signal (e.g. a
severe head-pose reprojection failure) can push a frame's score high on
its own. The final video-level score blends how *anomalous* frames are on
average with how *often* frames cross the "flagged" threshold.
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
}

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
# fact silently corrupting the calibrated overall likelihood.
MEAN_SCORE_WEIGHT = 0.5
FRAC_FLAGGED_WEIGHT = 0.5
PEAK_WINDOW_FRAMES = 15   # ~0.5-1s at typical fps; used only for the informational peak-window stat

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
GLOBAL_QUALITY_WEIGHT = 0.35

VERDICT_LIKELY_FAKE_THRESHOLD = 55.0
VERDICT_SUSPICIOUS_THRESHOLD = 30.0


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
                  evaluated: bool) -> FrameScore:
        category_scores = {s.name: s.score for s in signals}
        reasons = [s.reason for s in signals if s.score > 0.05]
        combined = self._noisy_or(signals) if evaluated else 0.0

        fs = FrameScore(
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            combined_score=combined,
            category_scores=category_scores,
            reasons=reasons,
            evaluated=evaluated,
        )
        self._frame_scores.append(fs)
        return fs

    def _noisy_or(self, signals: List[Signal]) -> float:
        survival = 1.0
        for s in signals:
            weight = self.weights.get(s.name, 1.0)
            contribution = _clip01(weight * s.score)
            survival *= (1.0 - contribution)
        return _clip01(1.0 - survival)

    def finalize(self, global_signals: Optional[List[Signal]] = None) -> VideoResult:
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
            )

        scores = [fs.combined_score for fs in evaluated_scores]
        mean_score = float(np.mean(scores))
        flagged = [fs for fs in evaluated_scores if fs.combined_score >= self.frame_flag_threshold]
        frac_flagged = len(flagged) / len(evaluated_scores)

        final01 = (MEAN_SCORE_WEIGHT * mean_score + FRAC_FLAGGED_WEIGHT * frac_flagged
                   + GLOBAL_QUALITY_WEIGHT * global_score)
        final_score = round(100.0 * _clip01(final01), 1)

        peak_score, peak_ts = self._peak_window(evaluated_scores)

        return VideoResult(
            final_score=final_score,
            verdict=self._verdict(final_score),
            category_breakdown=self._category_breakdown(evaluated_scores),
            flagged_ranges=self._flagged_ranges(),
            frame_scores=self._frame_scores,
            frames_evaluated=len(evaluated_scores),
            frames_total=len(self._frame_scores),
            peak_window_score=peak_score,
            peak_window_timestamp_sec=peak_ts,
            global_quality_score=global_score,
            global_quality_reason=global_reason,
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
        n = len(evaluated_scores)
        if n == 0:
            return 0.0, 0.0
        window = min(PEAK_WINDOW_FRAMES, n)
        scores = np.array([fs.combined_score for fs in evaluated_scores])
        kernel = np.ones(window) / window
        rolling = np.convolve(scores, kernel, mode="valid")
        best_idx = int(np.argmax(rolling))
        center_idx = min(best_idx + window // 2, n - 1)
        return float(rolling[best_idx]), float(evaluated_scores[center_idx].timestamp_sec)

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
