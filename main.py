#!/usr/bin/env python
"""CLI entry point for the heuristic deepfake video detector.

Examples:
    python main.py --input clip.mp4 --output out_annotated.mp4 --report report.json
    python main.py --webcam 0 --output out_annotated.mp4 --report report.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2

from detector.blur_checks import BlurChecker
from detector.capture import FrameSource
from detector.glitch_detection import GlitchDetector
from detector.history import LandmarkHistory, Signal
from detector.landmarks import FrameLandmarks, MediaPipeExtractor
from detector.object_checks import ObjectChecker
from detector.object_detection import ObjectExtractor, ObjectTracker, TrackedObject
from detector.person_tracker import MultiPersonTracker
from detector.physics_checks import PhysicsChecker
from detector.quality_checks import assess_blur_vs_bitrate
from detector.report import save_report
from detector.scoring import AnomalyAggregator, VERDICT_SUSPICIOUS_THRESHOLD, combine_signals
from detector.visualizer import AnnotatedVideoWriter, draw_overlay

ROLLING_DISPLAY_WINDOW = 15
PERSON_IDLE_DROP_FRAMES = 300   # drop a person's whole pipeline (own history/checkers) after this many missed frames


@dataclass
class PersonPipeline:
    """Bundles the per-person stateful checkers. `GlitchDetector` and
    `BlurChecker` hold their own internal rolling state (presence deques,
    sharpness history, etc.), so each tracked person needs an independent
    instance - sharing one across people would let one person's motion/
    blur history contaminate another's baseline.
    """

    history: LandmarkHistory
    glitch_detector: GlitchDetector = field(default_factory=GlitchDetector)
    physics_checker: PhysicsChecker = field(default_factory=PhysicsChecker)
    blur_checker: BlurChecker = field(default_factory=BlurChecker)
    idle_frames: int = 0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Heuristic deepfake detector: tracks facial/body landmarks and flags "
                    "glitches and physically implausible motion to produce a fake-likelihood score."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--input", type=str, help="Path to an input video file.")
    source_group.add_argument("--webcam", type=int, nargs="?", const=0, metavar="INDEX",
                               help="Webcam device index to read a live feed from (default 0).")

    parser.add_argument("--output", type=str, default="output_annotated.mp4",
                         help="Path to write the annotated output video (default: output_annotated.mp4).")
    parser.add_argument("--report", type=str, default="report.json",
                         help="Path to write the JSON report (default: report.json).")
    parser.add_argument("--no-display", action="store_true",
                         help="Don't open a live preview window while processing.")
    parser.add_argument("--no-output-video", action="store_true",
                         help="Skip writing the annotated output video (report only).")
    parser.add_argument("--history-window", type=int, default=30,
                         help="Number of past frames kept for jitter/z-score baselines (default: 30).")
    parser.add_argument("--frame-flag-threshold", type=float, default=None,
                         help="Override the per-frame anomaly threshold (0-1) used for flagging (default: 0.4).")
    parser.add_argument("--max-frames", type=int, default=None,
                         help="Stop after processing N frames (useful for quick tests).")
    parser.add_argument("--max-people", type=int, default=1,
                         help="Maximum number of people to detect/track at once (default: 1). "
                              "Note: MediaPipe's PoseLandmarker is measurably less temporally stable "
                              "per-instance in multi-pose mode than in its single-pose mode (~7x higher "
                              "frame-to-frame jitter measured on real footage - see README), so raising "
                              "this above 1 trades some landmark-jitter-check precision for multi-person "
                              "coverage. Only raise it if your footage actually has multiple people.")
    parser.add_argument("--no-object-detection", action="store_true",
                         help="Disable generic object detection/tracking (faster, person-only analysis).")

    return parser.parse_args(argv)


def _people_with_scores(tracked_people: List[Tuple[int, FrameLandmarks]],
                         person_pipelines: Dict[int, PersonPipeline],
                         weights: Dict[str, float], history_window: int,
                         frame_bgr) -> Tuple[List[Tuple[int, FrameLandmarks, Dict[str, float]]],
                                              Optional[int], List[Signal]]:
    """Updates every currently-tracked person's own pipeline and returns
    (per-person display info, id of the worst-scoring person, that
    person's signals) - the "worst person this frame" drives the video's
    single overall score, per the project's chosen design.
    """
    people_display: List[Tuple[int, FrameLandmarks, Dict[str, float]]] = []
    worst_person_id: Optional[int] = None
    worst_score = -1.0
    worst_signals: List[Signal] = []

    for person_id, fl in tracked_people:
        pipeline = person_pipelines.get(person_id)
        if pipeline is None:
            pipeline = PersonPipeline(history=LandmarkHistory(window=history_window))
            person_pipelines[person_id] = pipeline
        pipeline.idle_frames = 0
        pipeline.history.push(fl)

        signals: List[Signal] = []
        signals.extend(pipeline.glitch_detector.analyze(pipeline.history, frame_bgr))
        signals.extend(pipeline.physics_checker.analyze(pipeline.history))
        signals.extend(pipeline.blur_checker.analyze(pipeline.history, frame_bgr))

        category_scores = {s.name: s.score for s in signals}
        people_display.append((person_id, fl, category_scores))

        person_score = combine_signals(signals, weights)
        if person_score > worst_score:
            worst_score = person_score
            worst_person_id = person_id
            worst_signals = signals

    return people_display, worst_person_id, worst_signals


def main(argv=None) -> int:
    args = parse_args(argv)

    is_webcam = args.webcam is not None
    source_arg = args.webcam if is_webcam else args.input

    try:
        source = FrameSource(source_arg, is_webcam=is_webcam, max_frames=args.max_frames)
    except IOError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    show_display = not args.no_display
    write_output_video = not args.no_output_video

    if is_webcam and not show_display and args.max_frames is None:
        print("Warning: running webcam mode with --no-display and no --max-frames. "
              "Press Ctrl+C to stop and generate the report.")

    extractor = MediaPipeExtractor(max_people=args.max_people)
    person_tracker = MultiPersonTracker()
    person_pipelines: Dict[int, PersonPipeline] = {}

    object_extractor: Optional[ObjectExtractor] = None
    object_tracker: Optional[ObjectTracker] = None
    object_checker: Optional[ObjectChecker] = None
    if not args.no_object_detection:
        object_extractor = ObjectExtractor()
        object_tracker = ObjectTracker()
        object_checker = ObjectChecker()

    aggregator_kwargs = {}
    if args.frame_flag_threshold is not None:
        aggregator_kwargs["frame_flag_threshold"] = args.frame_flag_threshold
    aggregator = AnomalyAggregator(**aggregator_kwargs)

    writer = None
    if write_output_video:
        writer = AnnotatedVideoWriter(args.output, fps=source.fps, frame_size=(source.width, source.height))

    rolling_scores = deque(maxlen=ROLLING_DISPLAY_WINDOW)

    progress = None
    if not is_webcam and source.total_frames:
        try:
            from tqdm import tqdm
            progress = tqdm(total=min(source.total_frames, args.max_frames or source.total_frames), unit="frame")
        except ImportError:
            progress = None

    frames_processed = 0
    max_people_seen_at_once = 0
    start_time = time.time()
    window_name = "Deepfake Detector - press 'q' to stop"

    try:
        for frame_info in source:
            raw = extractor.process_multi(frame_info.frame, frame_info.index, frame_info.timestamp_sec)
            tracked_people = person_tracker.update(raw)
            max_people_seen_at_once = max(max_people_seen_at_once, len(tracked_people))

            people_display, worst_person_id, worst_signals = _people_with_scores(
                tracked_people, person_pipelines, aggregator.weights, args.history_window, frame_info.frame,
            )

            seen_ids = {pid for pid, _ in tracked_people}
            for pid, pipeline in list(person_pipelines.items()):
                if pid in seen_ids:
                    continue
                pipeline.idle_frames += 1
                if pipeline.idle_frames > PERSON_IDLE_DROP_FRAMES:
                    del person_pipelines[pid]

            tracked_objects: List[TrackedObject] = []
            object_signals: List[Signal] = []
            flagged_object_ids: set = set()
            if object_extractor is not None:
                boxes = object_extractor.process(frame_info.frame, frame_info.index)
                tracked_objects = object_tracker.update(boxes)
                object_signals = object_checker.analyze(tracked_objects)
                fired_names = {s.name for s in object_signals if s.score > 0.05}
                flagged_object_ids = {oid for name, oid in object_checker.last_flagged_ids.items()
                                       if name in fired_names}

            all_signals = list(worst_signals) + list(object_signals)
            evaluated = len(tracked_people) > 0
            frame_score = aggregator.add_frame(
                frame_info.index, frame_info.timestamp_sec, all_signals, evaluated,
                worst_person_id=worst_person_id, people_present=len(tracked_people),
            )

            rolling_scores.append(frame_score.combined_score)
            rolling = sum(rolling_scores) / len(rolling_scores)

            if writer is not None or show_display:
                frame_category_scores = {s.name: s.score for s in all_signals}
                annotated = draw_overlay(frame_info.frame, people_display, tracked_objects, flagged_object_ids,
                                          frame_category_scores, frame_score.combined_score, rolling)
                if writer is not None:
                    writer.write(annotated)
                if show_display:
                    cv2.imshow(window_name, annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            frames_processed += 1
            if progress is not None:
                progress.update(1)

    except KeyboardInterrupt:
        print("\nInterrupted - finalizing report with frames processed so far...")

    finally:
        if progress is not None:
            progress.close()
        if writer is not None:
            writer.release()
        if show_display:
            cv2.destroyAllWindows()
        extractor.close()
        if object_extractor is not None:
            object_extractor.close()
        source.release()

    elapsed = time.time() - start_time

    global_signals: List[Signal] = []
    file_size_bytes = None
    if not is_webcam and isinstance(source_arg, str) and os.path.isfile(source_arg):
        file_size_bytes = os.path.getsize(source_arg)
        # Blur-vs-bitrate is evaluated per tracked person (each person's own
        # face-sharpness history vs. the shared file-level bitrate), and the
        # worst result across people is what's reported/scored - consistent
        # with "worst tracked person drives the video-level signal".
        for pipeline in person_pipelines.values():
            quality_signal = assess_blur_vs_bitrate(
                pipeline.blur_checker.face_sharpness_normalized_all, file_size_bytes,
                source.width, source.height, frames_processed,
            )
            if quality_signal is not None:
                global_signals.append(quality_signal)

    result = aggregator.finalize(
        global_signals=global_signals,
        max_people_detected=max_people_seen_at_once,
        people_track_count=person_tracker.total_people_seen,
    )

    video_meta = {
        "source": str(source_arg),
        "is_webcam": is_webcam,
        "fps": source.fps,
        "width": source.width,
        "height": source.height,
        "frames_processed": frames_processed,
        "duration_sec": round(frames_processed / source.fps, 2) if source.fps else None,
        "processing_time_sec": round(elapsed, 2),
        "file_size_bytes": file_size_bytes,
        "max_people": args.max_people,
        "object_detection_enabled": object_extractor is not None,
    }

    chart_path = save_report(result, video_meta, args.report)

    _print_summary(result, args.report, chart_path, writer_path=args.output if write_output_video else None)
    return 0


def _print_summary(result, report_path: str, chart_path: str, writer_path) -> None:
    print("\n" + "=" * 50)
    if result.verdict == "NO_FACE_OR_BODY_DETECTED":
        print(" No face or body was detected in this video - no score computed.")
        print("=" * 50)
        return

    print(f" Fake-likelihood score: {result.final_score:.1f} / 100")
    print(f" Verdict: {result.verdict}")
    print(f" Frames evaluated: {result.frames_evaluated} / {result.frames_total}")
    print(f" People tracked: max {result.max_people_detected} at once, {result.people_track_count} distinct total")
    peak_pct = result.peak_window_score * 100.0
    print(f" Peak local anomaly burst: {peak_pct:.1f} / 100 (around t={result.peak_window_timestamp_sec:.1f}s)")
    if peak_pct >= 55.0 and result.final_score < VERDICT_SUSPICIOUS_THRESHOLD:
        print(f"   Note: overall average is low, but a short severe anomaly cluster was found around "
              f"t={result.peak_window_timestamp_sec:.1f}s - worth a manual look (common in compilations "
              f"or clips with a single bad cut).")

    if result.global_quality_score > 0.05:
        print(f" Whole-video blur-vs-bitrate check: {result.global_quality_score * 100:.1f} / 100")
        print(f"   {result.global_quality_reason}")

    top_categories = sorted(result.category_breakdown.items(), key=lambda kv: kv[1]["mean"], reverse=True)
    top_categories = [(name, stats) for name, stats in top_categories if stats["mean"] > 0.001][:5]
    if top_categories:
        print(" Top contributing signals:")
        for name, stats in top_categories:
            print(f"   - {name:<24}: mean={stats['mean']:.2f}  peak={stats['peak']:.2f}  "
                  f"frames={stats['frame_frac'] * 100:.1f}%")

    if result.flagged_ranges:
        print(f" Flagged frame ranges: {len(result.flagged_ranges)}")
        for r in result.flagged_ranges[:5]:
            print(f"   - frames {r.start_frame}-{r.end_frame} (peak={r.peak_score:.2f}): "
                  f"{', '.join(r.categories)}")
        if len(result.flagged_ranges) > 5:
            print(f"   ... and {len(result.flagged_ranges) - 5} more (see {report_path})")

    if writer_path:
        print(f" Annotated video written to: {writer_path}")
    print(f" JSON report written to: {report_path}")
    print(f" Anomaly timeline chart written to: {chart_path}")
    print("=" * 50)


if __name__ == "__main__":
    sys.exit(main())
