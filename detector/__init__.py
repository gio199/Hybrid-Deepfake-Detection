"""Heuristic deepfake video detector package.

Rule-based pipeline: MediaPipe landmark extraction -> temporal glitch
checks + physical plausibility checks -> aggregated likelihood score.
"""

__version__ = "0.1.0"
