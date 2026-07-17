#!/usr/bin/env python
"""Prepare, analyze, annotate, and chart AIGVDBench examples 4-6."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from prepare_aigvd_examples import EXAMPLES, prepare_examples


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--no-object-detection", action="store_true")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--dashboard",
        type=Path,
        default=project_root / "benchmarks" / "results_dashboard.html",
    )
    args = parser.parse_args()

    if not args.skip_prepare:
        list(prepare_examples(project_root, offline=args.offline))

    for example in EXAMPLES:
        example_dir = project_root / example["id"]
        stem = Path(example["filename"]).stem
        input_path = example_dir / example["filename"]
        output_path = example_dir / f"{stem}_annotated.mp4"
        report_path = example_dir / f"{stem}_report.json"
        if args.skip_existing and report_path.exists() and output_path.exists():
            print(f"Skipping existing {example['id']}")
            continue

        command = [
            sys.executable,
            str(project_root / "main.py"),
            "--input",
            str(input_path.relative_to(project_root)),
            "--output",
            str(output_path.relative_to(project_root)),
            "--report",
            str(report_path.relative_to(project_root)),
            "--no-display",
        ]
        if args.no_object_detection:
            command.append("--no-object-detection")
        if args.max_frames is not None:
            command.extend(["--max-frames", str(args.max_frames)])
        subprocess.run(command, cwd=project_root, check=True)

    subprocess.run(
        [
            sys.executable,
            str(project_root / "benchmarks" / "render_dashboard.py"),
            str(project_root / "benchmarks" / "bundled_manifest.csv"),
            "--output",
            str(args.dashboard),
            "--png",
            str(project_root / "benchmarks" / "score_comparison.png"),
        ],
        cwd=project_root,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
