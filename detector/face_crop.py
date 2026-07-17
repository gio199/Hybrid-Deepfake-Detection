"""Shared face alignment utilities for temporal and learned detectors."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .landmarks import (
    FACE_CHIN,
    FACE_LEFT_EYE_OUTER_CORNER,
    FACE_RIGHT_EYE_OUTER_CORNER,
    FrameLandmarks,
)


def aligned_face_crop(
    frame_bgr: np.ndarray,
    landmarks: FrameLandmarks,
    size: int = 224,
) -> Optional[np.ndarray]:
    """Warp eyes and chin to a stable square BGR crop."""
    if not landmarks.face_present or landmarks.face is None or size < 16:
        return None
    try:
        src = np.float32([
            [
                landmarks.face[FACE_LEFT_EYE_OUTER_CORNER].x,
                landmarks.face[FACE_LEFT_EYE_OUTER_CORNER].y,
            ],
            [
                landmarks.face[FACE_RIGHT_EYE_OUTER_CORNER].x,
                landmarks.face[FACE_RIGHT_EYE_OUTER_CORNER].y,
            ],
            [landmarks.face[FACE_CHIN].x, landmarks.face[FACE_CHIN].y],
        ])
    except IndexError:
        return None

    dst = np.float32([
        [size * 0.30, size * 0.34],
        [size * 0.70, size * 0.34],
        [size * 0.50, size * 0.86],
    ])
    if abs(float(cv2.contourArea(src))) < 1.0:
        return None
    transform = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(
        frame_bgr,
        transform,
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
