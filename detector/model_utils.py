"""Downloads and caches the MediaPipe Tasks model bundles used by this
project (face_landmarker.task, pose_landmarker.task).

The legacy `mediapipe.solutions` API (FaceMesh/Pose classes) was removed
from the mediapipe package starting with version 0.10.30; this project
uses the modern MediaPipe Tasks API instead, which loads pre-trained
model bundles from disk. These are downloaded once and cached under
`models/` at the project root.
"""

from __future__ import annotations

import os
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

MODEL_URLS = {
    "face_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "pose_landmarker_full.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "efficientdet_lite0.tflite":
        "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite",
}


def ensure_model(model_name: str) -> str:
    """Returns a local path to `model_name`, downloading it into `models/`
    the first time it's needed.
    """
    if model_name not in MODEL_URLS:
        raise ValueError(f"Unknown model: {model_name}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    local_path = os.path.join(MODELS_DIR, model_name)

    if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
        url = MODEL_URLS[model_name]
        print(f"Downloading {model_name} model (one-time download)...")
        tmp_path = local_path + ".part"
        try:
            urllib.request.urlretrieve(url, tmp_path)
            os.replace(tmp_path, local_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    return local_path
