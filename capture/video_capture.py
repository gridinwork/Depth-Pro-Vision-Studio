"""Video file playback with seek support."""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import cv2

from utils.images import fit_cover
from utils.timers import Stopwatch


class VideoCaptureSource:
    def __init__(self) -> None:
        self._cap: cv2.VideoCapture | None = None
        self._temp: Path | None = None
        self.fps = 30.0
        self.frame_count = 0
        self.path = ""

    def open(self, path: str) -> None:
        self.close()
        self.path = path
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            suffix = Path(path).suffix or ".mp4"
            temporary = Path(tempfile.gettempdir()) / f"dpvs_video{suffix}"
            shutil.copy(path, temporary)
            cap.release()
            cap = cv2.VideoCapture(str(temporary))
            self._temp = temporary
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        self.fps = fps if fps > 1 else 30.0
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._cap = cap

    def read(self, width: int, height: int):
        if self._cap is None:
            return None
        clock = Stopwatch()
        ok, frame = self._cap.read()
        capture_ms = clock.ms()
        if not ok or frame is None:
            return None
        frame = fit_cover(frame, width, height)
        index = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
        fraction = 0.0
        if self.frame_count > 0:
            fraction = min(1.0, index / self.frame_count)
        return frame, self.fps, capture_ms, fraction

    def seek_fraction(self, fraction: float) -> None:
        if self._cap is None or self.frame_count <= 0:
            return
        fraction = min(1.0, max(0.0, float(fraction)))
        frame_index = int(fraction * max(0, self.frame_count - 1))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    def restart(self) -> None:
        self.seek_fraction(0.0)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._temp is not None:
            try:
                self._temp.unlink(missing_ok=True)
            except OSError:
                pass
            self._temp = None
            time.sleep(0.02)
