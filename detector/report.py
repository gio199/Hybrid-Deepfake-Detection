"""Builds the JSON report and an anomaly-timeline PNG chart for a finished
analysis run.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

# Keep Matplotlib's font/config cache inside the designated project folder.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(_PROJECT_ROOT, ".runtime_cache", "matplotlib"),
)

import matplotlib

matplotlib.use("Agg")  # headless-safe backend, no GUI/display required
import matplotlib.pyplot as plt
import numpy as np

from .scoring import VideoResult

MAX_SERIES_POINTS = 2000


def _downsample(values: List[float], max_points: int) -> List[float]:
    if len(values) <= max_points:
        return values
    step = len(values) / max_points
    idxs = [int(i * step) for i in range(max_points)]
    return [values[i] for i in idxs]


def build_report_dict(result: VideoResult, video_meta: Dict) -> Dict:
    frame_idxs = [fs.frame_idx for fs in result.frame_scores]
    combined = [fs.combined_score for fs in result.frame_scores]
    timestamps = [fs.timestamp_sec for fs in result.frame_scores]

    return {
        "video": video_meta,
        "result": {
            "final_score": result.final_score,
            "verdict": result.verdict,
            "frames_evaluated": result.frames_evaluated,
            "frames_total": result.frames_total,
            "peak_window_score": round(result.peak_window_score * 100.0, 1),
            "peak_window_timestamp_sec": round(result.peak_window_timestamp_sec, 2),
            "global_quality_score": round(result.global_quality_score * 100.0, 1),
            "global_quality_reason": result.global_quality_reason,
            "max_people_detected": result.max_people_detected,
            "people_track_count": result.people_track_count,
            "evidence_coverage": round(result.evidence_coverage, 4),
            "evidence_duration_sec": round(result.evidence_duration_sec, 2),
            "evidence_warning": result.evidence_warning,
            "score_kind": "anomaly_score",
            "fusion_method": result.fusion_method,
        },
        "category_breakdown": result.category_breakdown,
        "flagged_ranges": [
            {
                "start_frame": r.start_frame,
                "end_frame": r.end_frame,
                "peak_score": round(r.peak_score, 3),
                "categories": r.categories,
            }
            for r in result.flagged_ranges
        ],
        "timeline": {
            "frame_idx": _downsample(frame_idxs, MAX_SERIES_POINTS),
            "timestamp_sec": _downsample(timestamps, MAX_SERIES_POINTS),
            "combined_score": [round(v, 4) for v in _downsample(combined, MAX_SERIES_POINTS)],
        },
    }


def save_report(result: VideoResult, video_meta: Dict, report_path: str) -> str:
    """Writes the JSON report to `report_path` and a companion
    `<name>_timeline.png` chart next to it. Returns the chart path.
    """
    report_dict = build_report_dict(result, video_meta)

    os.makedirs(os.path.dirname(os.path.abspath(report_path)) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    chart_path = _chart_path_for(report_path)
    _save_timeline_chart(result, chart_path)
    return chart_path


def _chart_path_for(report_path: str) -> str:
    base, _ext = os.path.splitext(report_path)
    return f"{base}_timeline.png"


def _save_timeline_chart(result: VideoResult, chart_path: str) -> None:
    evaluated = [fs for fs in result.frame_scores if fs.evaluated]
    if not evaluated:
        return

    timestamps = np.array([fs.timestamp_sec for fs in evaluated])
    scores = np.array([fs.combined_score * 100.0 for fs in evaluated])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timestamps, scores, color="#d62728", linewidth=1.0, label="Per-frame anomaly score")

    if len(scores) >= 5:
        window = max(3, len(scores) // 50)
        kernel = np.ones(window) / window
        smoothed = np.convolve(scores, kernel, mode="same")
        ax.plot(timestamps, smoothed, color="#1f77b4", linewidth=1.8, label="Smoothed")

    ax.axhline(result.frame_flag_threshold * 100, color="gray", linestyle="--", linewidth=0.8,
               label="Flag threshold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Anomaly score (0-100)")
    ax.set_title(f"Anomaly timeline - final score {result.final_score:.1f}/100 ({result.verdict})")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=120)
    plt.close(fig)
