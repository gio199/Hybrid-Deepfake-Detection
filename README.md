# Heuristic Deepfake Video Detector

A **rule-based** (no training, no dataset required) command-line tool that:

1. Tracks facial, body, and hand landmarks in a video file or webcam feed
   using [MediaPipe](https://developers.google.com/mediapipe) Face Mesh +
   Pose + Hands.
2. Watches the landmarks over time for **glitches** (jitter, sudden
   "teleporting" points, flickering detections, blending seams).
3. Checks the landmark configuration against basic **human physics/anatomy**
   (bone-length consistency, joint angle limits, head-pose reprojection
   error, left/right symmetry).
4. Checks for **localized/selective blur** (face region blurrier than the
   rest of the scene, sudden sharpness drops) - a common trick used to
   visually hide generation artifacts.
5. Checks whether the **whole video's face detail is implausibly soft
   given how much data was spent encoding it** (bits-per-pixel vs. face
   sharpness) - real camera footage encoded at a generous bitrate should
   retain real fine detail; if it doesn't, that's a red flag that isn't
   explained away by ordinary compression.
6. Combines all of these signals into a single **fake-likelihood score
   (0-100%)**, an annotated output video, a JSON report, and a PNG anomaly
   timeline chart.

## How it works (pipeline)

```
Video/Webcam -> Frame capture -> MediaPipe FaceMesh+Pose
             -> Landmark history buffer
             -> Glitch checks (jitter / jumps / flicker)
             -> Physics checks (bone length / joint angles / head-pose / symmetry)
             -> Blur checks (region blur mismatch / hand blur / blur-onset spike)
             -> Whole-video blur-vs-bitrate check (file size/length vs. face sharpness)
             -> Weighted aggregation -> final score + peak-burst diagnostic
             -> Annotated video + JSON report + timeline chart
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

This uses MediaPipe's modern **Tasks API** (`FaceLandmarker` /
`PoseLandmarker` / `HandLandmarker`), not the older `mediapipe.solutions`
API (which was removed from the `mediapipe` package in version 0.10.30+).
The first time you run the tool it will automatically download the three
model bundles it needs (`face_landmarker.task`, `pose_landmarker_full.task`,
`hand_landmarker.task`, a few MB each) into a local `models/` folder -
this requires an internet connection once; after that everything runs
fully offline.

## Usage

Analyze a video file:

```bash
python main.py --input path/to/video.mp4 --output out_annotated.mp4 --report report.json
```

Analyze a live webcam feed (press `q` in the preview window to stop):

```bash
python main.py --webcam 0 --output out_annotated.mp4 --report report.json
```

Useful flags:

| Flag | Description |
|---|---|
| `--input PATH` | Path to an input video file. Mutually exclusive with `--webcam`. |
| `--webcam INDEX` | Webcam device index (usually `0`). Mutually exclusive with `--input`. |
| `--output PATH` | Where to write the annotated output video (default `output_annotated.mp4`). |
| `--report PATH` | Where to write the JSON report (default `report.json`). |
| `--no-display` | Don't open a live preview window (video-file mode only writes to disk). |
| `--history-window N` | Number of past frames kept for jitter/z-score baselines (default `30`). |
| `--max-frames N` | Stop after N frames (useful for quick tests). |

At the end of a run, the tool prints a summary like:

```
==============================================
 Fake-likelihood score: 71.4 / 100
 Verdict: LIKELY FAKE
 Frames evaluated: 812 / 820
 Peak local anomaly burst: 96.0 / 100 (around t=11.9s)
 Whole-video blur-vs-bitrate check: 79.1 / 100
   Video was encoded with a generous bitrate (0.261 bits/pixel) yet the face stays soft...
 Top contributing signals:
   - head_pose_reprojection : 0.82
   - landmark_jump          : 0.64
   - bone_length             : 0.51
==============================================
```

`Peak local anomaly burst` is a separate, informational-only diagnostic (see
[Scoring: average vs. burst](#scoring-average-vs-burst) below) - it does not
affect the score or verdict, but flags when a short segment (~0.5-1s) is
much worse than the rest of the clip, which is common in compilation
videos or clips with a single bad cut.

## Output files

- **Annotated video** (`--output`): original frames with the face mesh /
  pose skeleton drawn, landmarks that triggered a rule highlighted in red,
  and a running score overlay.
- **JSON report** (`--report`): video metadata, final score, per-category
  breakdown, list of flagged frame ranges with reasons, and (downsampled)
  per-frame raw scores.
- **`<report>_timeline.png`**: a chart of the anomaly score over time,
  rendered next to the JSON report.

## Localized blur checks

Generative/editing pipelines often blur or smear the parts of a frame
that are hardest to synthesize correctly (a face-swap seam, malformed
fingers) specifically to make the artifact less noticeable. `detector/blur_checks.py`
measures sharpness with the classic variance-of-Laplacian metric and
looks for three patterns:

- **`blur_mismatch`**: the face region is much blurrier than the rest of
  the same frame (sampled from the four corners), *while the background
  itself is genuinely sharp*. A blurry face in front of a blurry/dark
  background is not flagged - only a face that's suspiciously singled out.
- **`hand_blur_anomaly`**: a hand region is much blurrier than the face
  in the same frame. This one is computed and shown in the JSON report
  for informational purposes only, but weighted to 0 in the score itself
  - see [Hand landmarks](#hand-landmarks-what-was-tried-and-why-most-of-it-is-off-by-default)
  below for why.
- **`blur_onset_spike`**: the face region's sharpness suddenly drops well
  below its own recent rolling baseline (robust z-score), i.e. the video
  briefly gets blurry in that specific spot when it wasn't before.

These are combined with the geometric checks via the same weighted
noisy-OR as everything else, so a blurred region compounds with, rather
than replaces, evidence like an implausible bone length or a landmark
jump. They deliberately do **not** fire on generic whole-frame blur
(motion blur, low light, heavy compression, portrait-mode background
bokeh) since those affect the face and background (or face and hands)
roughly equally - only a *lopsided* blur pattern is suspicious.

## Whole-video blur-vs-bitrate check

Every check above looks at a single frame or a short rolling window. This
one is different - it looks at the **entire clip as one data point**,
using `detector/quality_checks.py`:

- **bits-per-pixel (bpp)**: `file_size_bytes * 8 / (width * height *
  frame_count)` - a standard, resolution/duration-independent measure of
  how generously the video was encoded. Low bpp means heavy compression
  (blur is expected and not suspicious); high bpp means plenty of room
  was available to preserve real detail.
- **whole-clip face sharpness**: the median variance-of-Laplacian of the
  face crop (resized to a canonical width so different resolutions/zoom
  levels are comparable) across *every* frame of the video, not just a
  rolling window.

If a video was encoded generously (bpp above a threshold) yet the face
stays soft across the **whole** clip, that's a mismatch a real camera
rarely produces - real sensors and lenses put real detail into the file
when they're given the bits to store it. This is deliberately gated on
*high* bpp so it never fires on the extremely common, totally mundane
case of a low-bitrate/heavily-compressed real video (which explains its
own blur without needing this check at all).

This only runs on file input (a live webcam has no fixed "file size" or
"length"), and only once enough frames with a tracked face have been
seen. Unlike the per-frame checks, it can't sit in the per-frame
noisy-OR combination (see `scoring.py`), so it's blended in as an
additive term on top of the per-frame-derived score, capped so it alone
can move the final score by at most `GLOBAL_QUALITY_WEIGHT * 100` points
(currently 35) - enough to be decisive when it fires, but never able to
manufacture a high score purely by itself from one coarse measurement.

**Calibration (as of writing, only 3 labeled clips available - treat
this as a coarse, low-confidence heuristic, not a certainty):**

| Clip | Verdict | bpp | Median face sharpness | This check |
|---|---|---|---|---|
| `real_baseline.mp4` | Real | 0.124 | 319 | Not flagged (sharp, as expected) |
| `example1` (face-swap deepfake) | Fake | 0.028 | 33 | Not flagged (bpp too low to judge - fully explained by compression) |
| `example2` (confirmed AI-generated) | Fake | **0.260** (2x the real clip's bitrate) | **77** (4x softer than the real clip) | **Flagged strongly (79/100)** - moved the final score from 30.9 to 58.6, correctly crossing into LIKELY FAKE |

## Hand landmarks: what was tried, and why most of it is off by default

Hands are notoriously hard for generative models to get right, so this
tool also runs MediaPipe's `HandLandmarker` (21 points/hand) and draws it
in the annotated video. Three additional anomaly signals were built on
top of it and rigorously A/B tested against both confirmed-real and
confirmed-fake footage:

| Signal | Idea | Verdict |
|---|---|---|
| `finger_bone_length` | Finger segment length should stay constant relative to the palm, like `bone_length` for body limbs | **Rejected** - fired on 57.8% of frames in a real video of someone gesturing expressively, vs. 38% on a confirmed deepfake. MediaPipe's per-frame finger landmarks are too noisy under natural fast articulation (2D foreshortening as fingers rotate toward/away from the camera) to tell real motion from a warped hand. |
| `finger_joint_motion` | Abrupt frame-to-frame knuckle-bend should be rare, like `joint_angle_motion` for elbows/knees | **Rejected** - same root cause; a finger can go from straight to fully curled in 2-3 frames during normal expressive gesturing, which looks identical to a "discontinuous glitch" to this check. Fired on 30% of frames in the real gesturing video vs. 5.2% on the confirmed deepfake - backwards. |
| `hand_blur_anomaly` | A hand blurrier than the face might mean deliberate concealment | **Kept, but weighted to 0 (informational only)** - even after gating out frames where the hand is moving fast (real motion blur), it still fired on ~21% of frames in the real gesturing video. The actual cause: hands move toward/away from the camera during gesturing and drift out of the focus plane while the face (the autofocus target) stays sharp - ordinary depth-of-field, not concealment. It also never fired at all on either the confirmed-real or confirmed-fake test videos, i.e. a 0% true-positive rate in testing. |

Both `detector/hand_checks.py` (the finger-shape checks) and the disabled
weight in `detector/scoring.py` document this in detail. The lesson,
consistent with the burst-weighting finding below: a plausible-sounding
heuristic still needs to be validated against genuine footage with the
*specific* behavior it might be confused by (fast gesturing, depth-of-field)
before it's trusted to move the score.

## Scoring: average vs. burst

The final 0-100 score is `50% mean anomaly score + 50% fraction of
frames flagged` across the whole clip - deliberately an *average*, not a
peak. An earlier iteration tried weighting a "top-k burst" statistic
directly into this formula to better catch short, localized anomaly
clusters (e.g. a compilation video where only a few seconds are actually
fake). That was reverted after testing: genuine footage routinely has a
handful of single/few-frame MediaPipe tracking hiccups (a fast head turn,
a hand briefly crossing the face) whose combined score saturates to
~1.0 for a moment, just like a real artifact would - so weighting bursts
into the *score* pushed real calibration clips into "LIKELY FAKE".

Instead, burst intensity is surfaced separately as **`Peak local anomaly
burst`** (a rolling ~0.5-1s window average, reported in the CLI summary
and JSON as `peak_window_score` / `peak_window_timestamp_sec`). It's
informational only - it tells you *where* to look manually when the
overall average is low but something briefly spiked, without silently
corrupting the calibrated score for every video.

## Important caveats

This tool is **heuristic and explainable**, not a trained deep-learning
classifier. That means:

- It can be **fooled** by high-quality deepfakes that don't trip the
  physics/glitch thresholds.
- It can produce **false positives** on real footage with fast motion,
  low light, low resolution, heavy compression, or unusual poses, since
  all of these can look like "physically implausible" motion to simple
  heuristics.
- It only reasons about **landmark geometry, temporal consistency, and
  coarse regional sharpness**, not deep pixel-level generative artifacts
  (texture statistics, frequency-domain fingerprints, GAN/diffusion
  "fingerprints") that more advanced deep-learning detectors use.
- The blur checks can be fooled by legitimate **shallow depth of field**
  where the subject is deliberately soft (rare, since background bokeh
  is far more common than foreground blur), or miss artifacts hidden by
  *uniform* whole-frame blur rather than *localized* blur.
- The whole-video blur-vs-bitrate check was calibrated on only 3 labeled
  clips (see above). It could plausibly false-positive on genuine but
  poor-quality captures - e.g. a real video shot slightly out of focus,
  in heavy low-light noise-reduction, or re-encoded at a generous bitrate
  from an already-soft source - so treat a flag from it as a strong hint
  to look closer, not a standalone verdict.

Treat the score as a rough, explainable signal to prioritize manual review,
not a definitive verdict.

## Tuning thresholds

All thresholds and category weights live at the top of
`detector/glitch_detection.py`, `detector/physics_checks.py`,
`detector/blur_checks.py`, `detector/quality_checks.py`, and
`detector/scoring.py`. If you find too many false positives/negatives on
your footage, adjust the constants there.

## Project layout

```
main.py                     CLI entry point
detector/
  capture.py                Video file / webcam frame source
  landmarks.py               MediaPipe Face Mesh + Pose extraction
  history.py                 Rolling per-landmark history buffer
  glitch_detection.py        Jitter / jump / flicker anomaly checks
  physics_checks.py          Bone length / joint angle / head-pose / symmetry checks
  blur_checks.py              Localized blur checks (region mismatch / hand blur / blur spike)
  quality_checks.py           Whole-video blur-vs-bitrate check (file size/length vs. face sharpness)
  hand_checks.py              Finger bone-length/joint checks (implemented, NOT wired in - see README)
  scoring.py                 Aggregation into a final likelihood score + peak-burst diagnostic + global quality blend
  visualizer.py               Drawing + annotated video writer
  report.py                  JSON report + timeline chart
```
