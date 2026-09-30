"""Thread-safe latest-frame mailbox.

Capture publishes RGB continuously. The depth worker reads only the newest
frame and drops everything in between. The GUI and recorder share the latest
composite.
"""

from __future__ import annotations

import threading
import time

import numpy as np


class FrameHub:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.rgb: np.ndarray | None = None
        self.rgb_id = 0
        self.capture_fps = 0.0
        self.capture_ms = 0.0
        self.depth: np.ndarray | None = None
        self.depth_meta: dict = {}
        self.depth_id = 0
        self.composite: np.ndarray | None = None
        self.layout = None
        self.updated_at = 0.0

    def publish_rgb(self, frame: np.ndarray, fps: float, capture_ms: float) -> int:
        with self._lock:
            self.rgb = frame
            self.rgb_id += 1
            self.capture_fps = float(fps)
            self.capture_ms = float(capture_ms)
            self.updated_at = time.perf_counter()
            return self.rgb_id

    def latest_rgb(self):
        with self._lock:
            if self.rgb is None:
                return None
            return self.rgb_id, self.rgb.copy(), self.capture_fps, self.capture_ms

    def latest_rgb_since(self, last_id: int):
        with self._lock:
            if self.rgb is None or self.rgb_id == last_id:
                return None
            return self.rgb_id, self.rgb.copy(), self.capture_fps, self.capture_ms

    def clear_rgb(self) -> None:
        with self._lock:
            self.rgb = None

    def publish_depth(self, depth: np.ndarray, meta: dict) -> int:
        with self._lock:
            self.depth = depth
            self.depth_meta = dict(meta)
            self.depth_id += 1
            return self.depth_id

    def latest_depth_since(self, last_id: int):
        with self._lock:
            if self.depth is None or self.depth_id == last_id:
                return None
            return self.depth_id, self.depth.copy(), dict(self.depth_meta)

    def clear_depth(self) -> None:
        with self._lock:
            self.depth = None
            self.depth_meta = {}
            self.depth_id += 1

    def set_composite(self, image: np.ndarray, layout) -> None:
        with self._lock:
            self.composite = image
            self.layout = layout

    def latest_composite(self):
        with self._lock:
            if self.composite is None:
                return None
            return self.composite.copy()
