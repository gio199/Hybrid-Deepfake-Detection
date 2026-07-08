"""Frame source abstraction: opens a video file or a webcam and yields frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Union

import cv2


@dataclass
class FrameInfo:
    """A single captured frame plus its position in the stream."""

    index: int
    timestamp_sec: float
    frame: "cv2.Mat"


class FrameSource:
    """Wraps cv2.VideoCapture for both video files and webcams.

    Usage:
        with FrameSource("clip.mp4") as source:
            for frame_info in source:
                ...
    """

    def __init__(self, source: Union[str, int], is_webcam: bool = False, max_frames: Optional[int] = None):
        self.source = source
        self.is_webcam = is_webcam
        self.max_frames = max_frames

        capture_target = source
        self._cap = cv2.VideoCapture(capture_target)
        if not self._cap.isOpened():
            kind = "webcam" if is_webcam else "video file"
            raise IOError(f"Could not open {kind}: {source}")

        self.fps: float = self._cap.get(cv2.CAP_PROP_FPS) or 0.0
        if self.fps <= 1e-3:
            # Some webcams / containers fail to report fps; assume a sane default.
            self.fps = 30.0

        self.width: int = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        raw_total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_frames: Optional[int] = raw_total if (not is_webcam and raw_total > 0) else None

        self._frame_idx = 0

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def __iter__(self) -> Iterator[FrameInfo]:
        return self

    def __next__(self) -> FrameInfo:
        if self.max_frames is not None and self._frame_idx >= self.max_frames:
            raise StopIteration

        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise StopIteration

        timestamp_sec = self._frame_idx / self.fps
        info = FrameInfo(index=self._frame_idx, timestamp_sec=timestamp_sec, frame=frame)
        self._frame_idx += 1
        return info

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
