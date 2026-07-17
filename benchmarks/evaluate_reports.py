#!/usr/bin/env python
"""Evaluate detector JSON reports listed in a labeled CSV manifest.

Manifest columns:
    report_path,label[,split,group,notes]

Positive labels accept generated, altered, synthetic, fake, or 1. Negative
labels accept real, authentic, or 0. Report paths are resolved relative to
the manifest file, which makes manifests portable.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


def _parse_label(raw: str) -> int:
    normalized = raw.strip().lower()
    if normalized in {
        "1", "ai", "ai-altered", "altered", "fake", "generated",
        "synthetic", "true", "yes",
    }:
        return 1
    if normalized in {"0", "authentic", "real", "false", "no"}:
        return 0
    raise ValueError(
        f"Unsupported label {raw!r}; use real/authentic, generated/altered, or 0/1"
    )


def load_rows(manifest_path: Path) -> List[Dict]:
    rows = []
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        for manifest_row in csv.DictReader(handle):
            report_path = (manifest_path.parent / manifest_row["report_path"]).resolve()
            with report_path.open(encoding="utf-8") as report_handle:
                report = json.load(report_handle)
            result = report["result"]
            rows.append({
                "report_path": str(report_path),
                "label": _parse_label(manifest_row["label"]),
                "score": float(result["final_score"]) / 100.0,
                "verdict": str(result.get("verdict", "")),
                "split": manifest_row.get("split", ""),
                "group": manifest_row.get("group", ""),
            })
    if not rows:
        raise ValueError("Manifest contains no reports")
    return rows


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _roc_auc(rows: List[Dict]) -> Optional[float]:
    positives = [row["score"] for row in rows if row["label"] == 1]
    negatives = [row["score"] for row in rows if row["label"] == 0]
    if not positives or not negatives:
        return None
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def _average_precision(rows: List[Dict]) -> Optional[float]:
    ranked = sorted(rows, key=lambda row: row["score"], reverse=True)
    positives = sum(row["label"] for row in ranked)
    if not positives:
        return None
    hits = 0
    precision_sum = 0.0
    for rank, row in enumerate(ranked, start=1):
        if row["label"] == 1:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / positives


def _expected_calibration_error(rows: List[Dict], bins: int = 10) -> float:
    total = len(rows)
    error = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        members = [
            row for row in rows
            if lower <= row["score"] < upper or (bin_index == bins - 1 and row["score"] == 1.0)
        ]
        if not members:
            continue
        confidence = sum(row["score"] for row in members) / len(members)
        accuracy = sum(row["label"] for row in members) / len(members)
        error += len(members) / total * abs(confidence - accuracy)
    return error


def evaluate(rows: List[Dict], threshold: float) -> Dict:
    classified = [row for row in rows if row["verdict"] != "INSUFFICIENT EVIDENCE"]
    tp = sum(row["score"] >= threshold and row["label"] == 1 for row in classified)
    fp = sum(row["score"] >= threshold and row["label"] == 0 for row in classified)
    tn = sum(row["score"] < threshold and row["label"] == 0 for row in classified)
    fn = sum(row["score"] < threshold and row["label"] == 1 for row in classified)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "samples": len(rows),
        "classified_samples": len(classified),
        "selective_coverage": _safe_div(len(classified), len(rows)),
        "threshold": threshold,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "accuracy": _safe_div(tp + tn, len(classified)),
        "precision": precision,
        "recall": recall,
        "specificity": _safe_div(tn, tn + fp),
        "f1": _safe_div(2 * precision * recall, precision + recall),
        "roc_auc": _roc_auc(rows),
        "average_precision": _average_precision(rows),
        "brier_score": sum((row["score"] - row["label"]) ** 2 for row in rows) / len(rows),
        "expected_calibration_error": _expected_calibration_error(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")

    metrics = evaluate(load_rows(args.manifest.resolve()), args.threshold)
    rendered = json.dumps(metrics, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
