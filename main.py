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

import cv2

from detector.blur_checks import BlurChecker
from detector.capture import FrameSource
from detector.glitch_detection import GlitchDetector
from detector.history import LandmarkHistory
from detector.landmarks import MediaPipeExtractor
from detector.physics_checks import PhysicsChecker
from detector.quality_checks import assess_blur_vs_bitrate
from detector.report import save_report
from detector.scoring import AnomalyAggregator, VERDICT_SUSPICIOUS_THRESHOLD
from detector.visualizer import AnnotatedVideoWriter, draw_overlay

ROLLING_DISPLAY_WINDOW = 15


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

    return parser.parse_args(argv)


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

    extractor = MediaPipeExtractor()
    history = LandmarkHistory(window=args.history_window)
    glitch_detector = GlitchDetector()
    physics_checker = PhysicsChecker()
    blur_checker = BlurChecker(history_window=args.history_window)
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
    start_time = time.time()
    window_name = "Deepfake Detector - press 'q' to stop"

    try:
        for frame_info in source:
            frame_landmarks = extractor.process(frame_info.frame, frame_info.index, frame_info.timestamp_sec)
            history.push(frame_landmarks)

            signals = []
            signals.extend(glitch_detector.analyze(history, frame_info.frame))
            signals.extend(physics_checker.analyze(history))
            signals.extend(blur_checker.analyze(history, frame_info.frame))

            evaluated = frame_landmarks.face_present or frame_landmarks.pose_present
            frame_score = aggregator.add_frame(frame_info.index, frame_info.timestamp_sec, signals, evaluated)

            rolling_scores.append(frame_score.combined_score)
            rolling = sum(rolling_scores) / len(rolling_scores)

            if writer is not None or show_display:
                annotated = draw_overlay(frame_info.frame, frame_landmarks, frame_score.category_scores,
                                          frame_score.combined_score, rolling)
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
        source.release()

    elapsed = time.time() - start_time

    global_signals = []
    file_size_bytes = None
    if not is_webcam and isinstance(source_arg, str) and os.path.isfile(source_arg):
        file_size_bytes = os.path.getsize(source_arg)
        quality_signal = assess_blur_vs_bitrate(
            blur_checker.face_sharpness_normalized_all, file_size_bytes,
            source.width, source.height, frames_processed,
        )
        if quality_signal is not None:
            global_signals.append(quality_signal)

    result = aggregator.finalize(global_signals=global_signals)

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
