"""15-second demo recorder driven by time.perf_counter, not by inference frames."""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from recording.video_encoder import VideoEncoder
from utils.logger import get_logger
from utils.paths import DEMO_DIR, ensure_dirs


class DemoRecorder(QThread):
    started = Signal(str)
    progress = Signal(float)
    finished_ok = Signal(object)
    failed = Signal(str)
    cancelled = Signal(object)

    def __init__(self, hub) -> None:
        super().__init__()
        self.hub = hub
        self._quit = threading.Event()
        self._start = threading.Event()
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._width = 1920
        self._height = 1080
        self._fps = 30
        self._duration = 15.0
        self._path: Path | None = None
        self.write_ms = 0.0
        self.active = False
        self._logger = get_logger("record")

    def request_start(self, width: int, height: int, fps: int, duration: float = 15.0) -> Path:
        ensure_dirs()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = DEMO_DIR / f"depth_pro_demo_{stamp}.mp4"
        with self._lock:
            self._width = int(width)
            self._height = int(height)
            self._fps = int(fps)
            self._duration = float(duration)
            self._path = path
        self._cancel.clear()
        self._start.set()
        return path

    def request_cancel(self) -> None:
        self._cancel.set()

    def shutdown(self) -> None:
        self._cancel.set()
        self._quit.set()
        self._start.set()

    def run(self) -> None:
        while not self._quit.is_set():
            if not self._start.wait(0.1):
                continue
            self._start.clear()
            if self._quit.is_set():
                break
            self._record_once()

    def _record_once(self) -> None:
        with self._lock:
            width = self._width
            height = self._height
            fps = self._fps
            duration = self._duration
            path = self._path
        if path is None:
            self.failed.emit("Demo path is empty.")
            return
        encoder = VideoEncoder()
        frames = 0
        wall = 0.0
        self.active = True
        self._logger.info("Recording started %s %sx%s @ %s fps", path, width, height, fps)
        self.started.emit(str(path))
        t0 = time.perf_counter()
        try:
            codec = encoder.open(path, width, height, fps)
            frame_count = int(round(duration * fps))
            for index in range(frame_count):
                if self._cancel.is_set() or self._quit.is_set():
                    break
                deadline = t0 + (index + 1) / fps
                composite = self.hub.latest_composite()
                if composite is None:
                    import numpy as np

                    composite = np.zeros((height, width, 3), dtype=np.uint8)
                write_clock = time.perf_counter()
                encoder.write(composite)
                self.write_ms = (time.perf_counter() - write_clock) * 1000.0
                frames += 1
                remaining = max(0.0, duration - (time.perf_counter() - t0))
                self.progress.emit(remaining)
                delay = deadline - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            encoder.close()
            wall = time.perf_counter() - t0
        except Exception as exc:
            self.active = False
            self._logger.exception("Recording failed")
            try:
                encoder.close()
            except Exception:
                self._logger.exception("Encoder close failed")
            self.failed.emit(str(exc))
            return

        self.active = False
        info = {
            "path": str(path),
            "duration": frames / fps if fps else 0.0,
            "wall_duration": wall,
            "resolution": f"{width}x{height}",
            "fps": fps,
            "frames": frames,
            "size": path.stat().st_size if path.exists() else 0,
            "codec": codec,
            "cancelled": self._cancel.is_set(),
        }
        if self._cancel.is_set():
            self._logger.info("Recording cancelled %s", path)
            self.cancelled.emit(info)
            return
        self._logger.info("Recording completed %s duration=%.2fs", path, info["duration"])
        self.finished_ok.emit(info)
