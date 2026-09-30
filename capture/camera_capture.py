"""OpenCV webcam capture."""

from __future__ import annotations

import time

import cv2

from utils.images import fit_cover
from utils.logger import get_logger
from utils.timers import FpsMeter, Stopwatch

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass


def list_cameras(limit: int = 6) -> list[int]:
    found: list[int] = []
    for index in range(limit):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        opened = cap.isOpened()
        cap.release()
        if opened:
            found.append(index)
    return found


class CameraCapture:
    def __init__(self) -> None:
        self._cap: cv2.VideoCapture | None = None
        self._fps = FpsMeter()
        self._logger = get_logger("camera")

    def open(self, index: int, width: int, height: int) -> None:
        self.close()
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {index}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        self._cap = cap
        self._fps.reset()
        self._logger.info("Camera selected index=%s resolution=%sx%s", index, width, height)

    def read(self, width: int, height: int):
        if self._cap is None:
            return None
        clock = Stopwatch()
        ok, frame = self._cap.read()
        capture_ms = clock.ms()
        if not ok or frame is None:
            return None
        frame = fit_cover(frame, width, height)
        fps = self._fps.tick()
        return frame, fps, capture_ms

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            time.sleep(0.05)
