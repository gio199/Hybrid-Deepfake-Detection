"""Optional ONNX face-forensics detector.

The repository does not bundle third-party weights. This adapter lets a
validated spatial detector participate in the same explainable signal
pipeline using OpenCV DNN, without requiring a heavyweight training stack.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import cv2
import numpy as np

from .face_crop import aligned_face_crop
from .history import Signal
from .landmarks import FrameLandmarks


@dataclass(frozen=True)
class LearnedModelConfig:
    input_size: int = 224
    scale: float = 1.0 / 255.0
    mean: Sequence[float] = (0.0, 0.0, 0.0)
    std: Sequence[float] = (1.0, 1.0, 1.0)
    swap_rb: bool = True
    output_type: str = "softmax"
    fake_index: int = 1
    sample_interval_sec: float = 0.25

    @classmethod
    def load(cls, path: Optional[str]) -> "LearnedModelConfig":
        if path is None:
            return cls()
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        if self.input_size < 16:
            raise ValueError("learned model input_size must be at least 16")
        if len(self.mean) != 3 or len(self.std) != 3 or any(value <= 0 for value in self.std):
            raise ValueError("learned model mean/std must contain three values and std must be positive")
        if self.output_type not in {"softmax", "sigmoid", "probability"}:
            raise ValueError("learned model output_type must be softmax, sigmoid, or probability")
        if self.fake_index < 0:
            raise ValueError("learned model fake_index must be non-negative")
        if self.sample_interval_sec <= 0:
            raise ValueError("learned model sample_interval_sec must be positive")


@dataclass
class _TrackPrediction:
    timestamp_sec: float
    score: float


class LearnedFaceDetector:
    """Runs a configured ONNX classifier on aligned face crops."""

    def __init__(self, model_path: str, config_path: Optional[str] = None):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Learned detector model not found: {model_path}")
        if config_path is None:
            sidecar = os.path.splitext(model_path)[0] + ".json"
            config_path = sidecar if os.path.isfile(sidecar) else None
        self.config = LearnedModelConfig.load(config_path)
        self.config.validate()
        try:
            self._net = cv2.dnn.readNetFromONNX(model_path)
        except cv2.error as exc:
            raise ValueError(f"Could not load ONNX learned detector: {exc}") from exc
        self.model_path = model_path
        self.config_path = config_path
        self._track_predictions: Dict[int, _TrackPrediction] = {}

    def analyze(
        self,
        person_id: int,
        landmarks: FrameLandmarks,
        frame_bgr: np.ndarray,
    ) -> Optional[Signal]:
        if not landmarks.face_present:
            return None
        previous = self._track_predictions.get(person_id)
        should_run = (
            previous is None
            or landmarks.timestamp_sec - previous.timestamp_sec >= self.config.sample_interval_sec
        )
        if should_run:
            crop = aligned_face_crop(frame_bgr, landmarks, self.config.input_size)
            if crop is None:
                return None
            score = self._predict(crop)
            previous = _TrackPrediction(landmarks.timestamp_sec, score)
            self._track_predictions[person_id] = previous
        if previous is None:
            return None
        return Signal(
            "learned_face",
            previous.score,
            f"Learned aligned-face detector score={previous.score:.3f}",
        )

    def forget_track(self, person_id: int) -> None:
        self._track_predictions.pop(person_id, None)

    def _predict(self, crop_bgr: np.ndarray) -> float:
        image = crop_bgr[:, :, ::-1] if self.config.swap_rb else crop_bgr
        image = image.astype(np.float32) * float(self.config.scale)
        mean = np.asarray(self.config.mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(self.config.std, dtype=np.float32).reshape(1, 1, 3)
        image = (image - mean) / std
        blob = np.transpose(image, (2, 0, 1))[None, ...]
        self._net.setInput(np.ascontiguousarray(blob))
        output = np.asarray(self._net.forward(), dtype=np.float64).reshape(-1)
        return self._probability_from_output(
            output,
            output_type=self.config.output_type,
            fake_index=self.config.fake_index,
        )

    @staticmethod
    def _probability_from_output(output: np.ndarray, output_type: str, fake_index: int) -> float:
        if output.size == 0:
            raise ValueError("Learned detector returned an empty output")
        if output_type == "sigmoid":
            value = float(output[fake_index] if output.size > 1 else output[0])
            probability = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, value))))
        elif output_type == "softmax":
            if fake_index >= output.size:
                raise ValueError(
                    f"fake_index {fake_index} is outside model output of size {output.size}"
                )
            shifted = output - np.max(output)
            probabilities = np.exp(shifted) / np.sum(np.exp(shifted))
            probability = float(probabilities[fake_index])
        else:
            if fake_index >= output.size:
                raise ValueError(
                    f"fake_index {fake_index} is outside model output of size {output.size}"
                )
            probability = float(output[fake_index])
        return float(max(0.0, min(1.0, probability)))
