#!/usr/bin/env python
"""Render a self-contained visual dashboard from a labeled report manifest."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path
from typing import Dict, List


def _parse_label(raw: str) -> int:
    normalized = raw.strip().lower()
    if normalized in {
        "1", "ai", "ai-altered", "altered", "fake", "generated",
        "synthetic", "true", "yes",
    }:
        return 1
    if normalized in {"0", "authentic", "real", "false", "no"}:
        return 0
    raise ValueError(f"Unsupported label {raw!r}")


def _relative_href(path: Path, output_dir: Path) -> str:
    return Path(os.path.relpath(path, output_dir)).as_posix()


def load_samples(manifest_path: Path, output_path: Path, threshold: float) -> List[Dict]:
    samples: List[Dict] = []
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            report_path = (manifest_path.parent / row["report_path"]).resolve()
            with report_path.open(encoding="utf-8") as report_handle:
                report = json.load(report_handle)

            result = report["result"]
            score = float(result["final_score"])
            label = _parse_label(row["label"])
            annotated_raw = row.get("annotated_path", "").strip()
            annotated_path = (
                (manifest_path.parent / annotated_raw).resolve() if annotated_raw else None
            )
            timeline_path = report_path.with_name(report_path.stem + "_timeline.png")
            samples.append({
                "id": f"sample-{index}",
                "name": row.get("name", "").strip() or report_path.stem,
                "label": "generated" if label else "real",
                "label_value": label,
                "score": round(score, 1),
                "verdict": str(result.get("verdict", "")),
                "correct_at_threshold": (score >= threshold) == bool(label),
                "threshold": threshold,
                "group": row.get("group", ""),
                "notes": row.get("notes", ""),
                "source_url": row.get("source_url", ""),
                "report_href": _relative_href(report_path, output_path.parent),
                "timeline_image_href": (
                    _relative_href(timeline_path, output_path.parent)
                    if timeline_path.exists() else ""
                ),
                "annotated_href": (
                    _relative_href(annotated_path, output_path.parent)
                    if annotated_path is not None else ""
                ),
                "annotated_expected": annotated_path is not None,
                "annotated_available": bool(annotated_path and annotated_path.exists()),
                "duration_sec": float(report.get("video", {}).get("duration_sec", 0.0)),
                "evidence_coverage": float(result.get("evidence_coverage", 0.0)),
                "timeline": {
                    "timestamps": report.get("timeline", {}).get("timestamp_sec", []),
                    "scores": [
                        round(float(value) * 100.0, 2)
                        for value in report.get("timeline", {}).get("combined_score", [])
                    ],
                },
                "categories": [
                    {
                        "name": name,
                        "mean": round(float(values.get("mean", 0.0)) * 100.0, 1),
                        "peak": round(float(values.get("peak", 0.0)) * 100.0, 1),
                        "frame_frac": round(float(values.get("frame_frac", 0.0)) * 100.0, 1),
                    }
                    for name, values in sorted(
                        report.get("category_breakdown", {}).items(),
                        key=lambda item: float(item[1].get("mean", 0.0)),
                        reverse=True,
                    )[:6]
                ],
            })
    if not samples:
        raise ValueError("Manifest contains no samples")
    return samples


def _build_score_svg(samples: List[Dict], threshold: float) -> str:
    width = 1000
    left = 250
    right = 80
    top = 35
    row_height = 42
    plot_width = width - left - right
    height = top + row_height * len(samples) + 45
    threshold_x = left + plot_width * threshold / 100.0
    parts = [
        (
            f'<svg class="score-chart" viewBox="0 0 {width} {height}" role="img" '
            'aria-labelledby="score-title score-desc">'
        ),
        '<title id="score-title">Detector anomaly score by example</title>',
        (
            '<desc id="score-desc">Horizontal bars compare anomaly scores from zero to one '
            f'hundred against the {threshold:g} threshold.</desc>'
        ),
    ]
    for tick in range(0, 101, 20):
        x = left + plot_width * tick / 100.0
        parts.append(
            f'<line class="grid" x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" '
            f'y2="{height - 30}"></line>'
        )
        parts.append(
            f'<text class="axis-label" x="{x:.1f}" y="{height - 8}" '
            f'text-anchor="middle">{tick}</text>'
        )
    parts.append(
        f'<line class="threshold" x1="{threshold_x:.1f}" y1="{top - 16}" '
        f'x2="{threshold_x:.1f}" y2="{height - 30}"></line>'
    )
    parts.append(
        f'<text class="threshold-label" x="{threshold_x + 6:.1f}" y="{top - 18}">'
        f'threshold {threshold:g}</text>'
    )
    for index, sample in enumerate(samples):
        y = top + index * row_height
        bar_width = plot_width * sample["score"] / 100.0
        status = "correct" if sample["correct_at_threshold"] else "miss"
        parts.append(
            f'<text class="sample-label" x="{left - 12}" y="{y + 21}" '
            f'text-anchor="end">{html.escape(sample["name"])}</text>'
        )
        parts.append(
            f'<rect class="score-bar {sample["label"]} {status}" '
            f'x="{left}" y="{y + 6}" width="{bar_width:.1f}" height="22" rx="4"></rect>'
        )
        score_x = min(left + bar_width + 8, width - 40)
        parts.append(
            f'<text class="score-label" x="{score_x:.1f}" y="{y + 22}">'
            f'{sample["score"]:.1f}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def build_html(samples: List[Dict], threshold: float) -> str:
    score_svg = _build_score_svg(samples, threshold)
    data_json = json.dumps(samples, ensure_ascii=False).replace("</", "<\\/")
    buttons = "".join(
        (
            f'<button type="button" class="sample-button" data-index="{index}" '
            f'aria-pressed="{"true" if index == 0 else "false"}">'
            f'<span>{html.escape(sample["name"])}</span>'
            f'<strong>{sample["score"]:.1f}</strong></button>'
        )
        for index, sample in enumerate(samples)
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI-generated video detector benchmark dashboard</title>
<style>
:root {{
  color-scheme: light dark;
  --bg: #f7f8fb;
  --surface: #ffffff;
  --text: #18202a;
  --muted: #5d6875;
  --border: #d9dee7;
  --grid: #dfe4ec;
  --real: #247f6d;
  --generated: #b85642;
  --miss: #a57416;
  --focus: #2459a9;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #11151b;
    --surface: #1a2028;
    --text: #edf1f7;
    --muted: #aab3c0;
    --border: #38414d;
    --grid: #2b333e;
    --real: #5bc2aa;
    --generated: #ef8d76;
    --miss: #e5b654;
    --focus: #84adf1;
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 15px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
main {{ width: min(1180px, calc(100% - 32px)); margin: 28px auto 48px; }}
h1 {{ margin: 0 0 4px; font-size: clamp(1.5rem, 3vw, 2.25rem); font-weight: 650; }}
h2 {{ margin: 0 0 14px; font-size: 1.12rem; }}
p {{ margin: 0; }}
.subtitle {{ color: var(--muted); margin-bottom: 24px; }}
.panel {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: clamp(14px, 2.5vw, 24px);
  margin-bottom: 18px;
}}
.legend {{ display: flex; flex-wrap: wrap; gap: 16px; color: var(--muted); margin-bottom: 8px; }}
.legend span::before {{
  content: ""; display: inline-block; width: 12px; height: 12px;
  margin-right: 6px; border-radius: 3px; vertical-align: -1px;
}}
.legend .real::before {{ background: var(--real); }}
.legend .generated::before {{ background: var(--generated); }}
.legend .miss::before {{ border: 2px solid var(--miss); }}
.score-chart {{ display: block; width: 100%; height: auto; overflow: visible; }}
.score-chart .grid {{ stroke: var(--grid); stroke-width: 1; }}
.score-chart .threshold {{ stroke: var(--muted); stroke-width: 2; stroke-dasharray: 6 5; }}
.score-chart text {{ fill: var(--text); font-size: 14px; }}
.score-chart .axis-label, .score-chart .threshold-label {{ fill: var(--muted); font-size: 12px; }}
.score-chart .score-bar.real {{ fill: var(--real); }}
.score-chart .score-bar.generated {{ fill: var(--generated); }}
.score-chart .score-bar.miss {{ stroke: var(--miss); stroke-width: 4; }}
.chooser {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 8px; }}
.sample-button {{
  appearance: none; display: flex; justify-content: space-between; gap: 12px;
  padding: 10px 12px; border: 1px solid var(--border); border-radius: 8px;
  background: transparent; color: var(--text); font: inherit; text-align: left; cursor: pointer;
}}
.sample-button:hover {{ background: color-mix(in srgb, var(--surface), var(--text) 6%); }}
.sample-button[aria-pressed="true"] {{ border-color: var(--focus); outline: 2px solid var(--focus); }}
.detail-grid {{ display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(260px, .6fr); gap: 22px; }}
.sample-heading {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 10px; margin-bottom: 8px; }}
.badge {{
  display: inline-block; padding: 2px 8px; border-radius: 999px;
  border: 1px solid var(--border); color: var(--muted); font-size: .82rem;
}}
.metrics {{ color: var(--muted); margin-bottom: 16px; }}
.timeline {{ display: block; width: 100%; height: auto; }}
.timeline .grid {{ stroke: var(--grid); stroke-width: 1; }}
.timeline .line {{ fill: none; stroke: var(--generated); stroke-width: 2.5; vector-effect: non-scaling-stroke; }}
.timeline .threshold {{ stroke: var(--muted); stroke-width: 1.5; stroke-dasharray: 5 5; }}
.timeline text {{ fill: var(--muted); font-size: 12px; }}
video {{ width: 100%; border-radius: 8px; background: var(--bg); margin-bottom: 12px; }}
.media-placeholder {{
  min-height: 180px; display: grid; place-items: center; text-align: center;
  border: 1px dashed var(--border); border-radius: 8px; color: var(--muted); padding: 18px;
}}
table {{ width: 100%; border-collapse: collapse; margin-top: 14px; }}
th, td {{ padding: 7px 6px; border-bottom: 1px solid var(--border); text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ color: var(--muted); font-weight: 600; }}
.links {{ display: flex; flex-wrap: wrap; gap: 12px; margin-top: 12px; }}
a {{ color: var(--focus); }}
.notes {{ color: var(--muted); margin-top: 10px; }}
@media (max-width: 760px) {{
  main {{ width: min(100% - 20px, 1180px); margin-top: 14px; }}
  .detail-grid {{ grid-template-columns: 1fr; }}
  .score-chart .sample-label {{ font-size: 11px; }}
}}
</style>
</head>
<body>
<main>
  <h1>AI-generated video detector benchmark</h1>
  <p class="subtitle">Anomaly scores are review signals, not calibrated probabilities.</p>
  <section class="panel" aria-labelledby="comparison-title">
    <h2 id="comparison-title">Score comparison</h2>
    <div class="legend" aria-label="Chart legend">
      <span class="real">Known real</span>
      <span class="generated">AI-generated or altered</span>
      <span class="miss">Wrong side of threshold</span>
    </div>
    {score_svg}
  </section>
  <section class="panel" aria-labelledby="inspect-title">
    <h2 id="inspect-title">Inspect a run</h2>
    <div class="chooser">{buttons}</div>
  </section>
  <section class="panel" id="detail" aria-live="polite"></section>
</main>
<script>
const samples = {data_json};
const detail = document.getElementById("detail");
const buttons = Array.from(document.querySelectorAll(".sample-button"));

function escapeHtml(value) {{
  return String(value).replace(/[&<>"']/g, char => ({{
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
  }})[char]);
}}

function timelineSvg(sample) {{
  const times = sample.timeline.timestamps;
  const scores = sample.timeline.scores;
  if (!times.length || times.length !== scores.length) {{
    return '<p class="media-placeholder">No evaluated timeline data.</p>';
  }}
  const width = 760, height = 270, left = 44, right = 14, top = 14, bottom = 32;
  const plotWidth = width - left - right, plotHeight = height - top - bottom;
  const maxTime = Math.max(...times, 0.001);
  const points = scores.map((score, index) => {{
    const x = left + (Number(times[index]) / maxTime) * plotWidth;
    const y = top + (1 - Number(score) / 100) * plotHeight;
    return `${{x.toFixed(1)}},${{y.toFixed(1)}}`;
  }}).join(" ");
  const thresholdY = top + (1 - sample.threshold / 100) * plotHeight;
  return `<svg class="timeline" viewBox="0 0 ${{width}} ${{height}}" role="img"
    aria-label="Anomaly score over ${{maxTime.toFixed(1)}} seconds">
    <line class="grid" x1="${{left}}" y1="${{top}}" x2="${{left}}" y2="${{height-bottom}}"></line>
    <line class="grid" x1="${{left}}" y1="${{height-bottom}}" x2="${{width-right}}" y2="${{height-bottom}}"></line>
    <line class="threshold" x1="${{left}}" y1="${{thresholdY.toFixed(1)}}"
      x2="${{width-right}}" y2="${{thresholdY.toFixed(1)}}"></line>
    <polyline class="line" points="${{points}}"></polyline>
    <text x="8" y="${{top+5}}">100</text><text x="20" y="${{height-bottom+4}}">0</text>
    <text x="${{left}}" y="${{height-8}}">0s</text>
    <text x="${{width-right}}" y="${{height-8}}" text-anchor="end">${{maxTime.toFixed(1)}}s</text>
  </svg>`;
}}

function categoryTable(sample) {{
  if (!sample.categories.length) return "";
  const rows = sample.categories.map(category => `<tr>
    <td>${{escapeHtml(category.name.replaceAll("_", " "))}}</td>
    <td>${{category.mean.toFixed(1)}}</td><td>${{category.peak.toFixed(1)}}</td>
    <td>${{category.frame_frac.toFixed(1)}}%</td></tr>`).join("");
  return `<table><thead><tr><th>Top signal</th><th>Mean</th><th>Peak</th><th>Frames</th></tr></thead>
    <tbody>${{rows}}</tbody></table>`;
}}

function render(index) {{
  const sample = samples[index];
  buttons.forEach((button, buttonIndex) => {{
    button.setAttribute("aria-pressed", buttonIndex === index ? "true" : "false");
  }});
  let media;
  if (sample.annotated_available) {{
    media = `<video controls preload="metadata" src="${{escapeHtml(sample.annotated_href)}}"></video>`;
  }} else if (sample.annotated_expected) {{
    media = `<div class="media-placeholder">Run <code>benchmarks/run_aigvd_examples.py</code>
      to materialize the annotated video.</div>`;
  }} else {{
    media = `<div class="media-placeholder">No current-code annotated video is linked for
      this example. Its report and timeline are shown.</div>`;
  }}
  const sourceLink = sample.source_url
    ? `<a href="${{escapeHtml(sample.source_url)}}" target="_blank" rel="noreferrer">Source</a>` : "";
  const timelineImageLink = sample.timeline_image_href
    ? `<a href="${{escapeHtml(sample.timeline_image_href)}}">Timeline PNG</a>` : "";
  detail.innerHTML = `<div class="sample-heading"><h2>${{escapeHtml(sample.name)}}</h2>
      <span class="badge">${{escapeHtml(sample.label)}}</span>
      <span class="badge">${{escapeHtml(sample.verdict)}}</span></div>
    <p class="metrics"><strong>${{sample.score.toFixed(1)}}/100</strong> anomaly score ·
      ${{sample.duration_sec.toFixed(1)}}s · ${{(sample.evidence_coverage * 100).toFixed(0)}}% evidence coverage</p>
    <div class="detail-grid"><div>
      ${{timelineSvg(sample)}}${{categoryTable(sample)}}
    </div><div>${{media}}<div class="links">
      <a href="${{escapeHtml(sample.report_href)}}">JSON report</a>
      ${{timelineImageLink}}${{sourceLink}}</div>
      <p class="notes">${{escapeHtml(sample.notes)}}</p>
    </div></div>`;
}}

buttons.forEach((button, index) => button.addEventListener("click", () => render(index)));
render(0);
</script>
</body>
</html>
"""


def save_score_chart(samples: List[Dict], threshold: float, destination: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(project_root / ".runtime_cache" / "matplotlib"),
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [sample["name"] for sample in samples]
    scores = [sample["score"] for sample in samples]
    colors = ["#247f6d" if sample["label"] == "real" else "#b85642" for sample in samples]
    figure_height = max(4.0, 0.55 * len(samples) + 1.8)
    fig, ax = plt.subplots(figsize=(10, figure_height))
    bars = ax.barh(range(len(samples)), scores, color=colors, alpha=0.9)
    for bar, sample in zip(bars, samples):
        if not sample["correct_at_threshold"]:
            bar.set_edgecolor("#a57416")
            bar.set_linewidth(3)
        ax.text(
            min(sample["score"] + 1.3, 98.0),
            bar.get_y() + bar.get_height() / 2,
            f'{sample["score"]:.1f}',
            va="center",
            fontsize=9,
        )
    ax.axvline(threshold, color="#666666", linestyle="--", linewidth=1.2, label=f"Threshold {threshold:g}")
    ax.set_yticks(range(len(samples)), labels=names)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Anomaly score (0-100; not a calibrated probability)")
    ax.set_title("AI-generated video detector score comparison")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="lower right")
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results_dashboard.html"))
    parser.add_argument("--png", type=Path, help="Optional static score-comparison PNG.")
    parser.add_argument("--threshold", type=float, default=55.0)
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 100.0:
        parser.error("--threshold must be between 0 and 100")

    manifest_path = args.manifest.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples = load_samples(manifest_path, output_path, args.threshold)
    output_path.write_text(build_html(samples, args.threshold), encoding="utf-8")
    if args.png:
        save_score_chart(samples, args.threshold, args.png.resolve())
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
