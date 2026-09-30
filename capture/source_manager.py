"""Capture thread. The GUI thread only flips flags; reads stay off the UI thread."""

from __future__ import annotations

import threading
import time

from PySide6.QtCore import QThread, Signal

from capture.camera_capture import CameraCapture, list_cameras
from capture.image_source import ImageSource
from capture.video_capture import VideoCaptureSource
from utils.logger import get_logger


class SourceManager(QThread):
    cameras_found = Signal(object)
    source_opened = Signal(str)
    source_failed = Signal(str)
    video_position = Signal(float)
    video_finished = Signal()

    def __init__(self, hub) -> None:
        super().__init__()
        self.hub = hub
        self._stop = threading.Event()
        self._active = threading.Event()
        self._paused = threading.Event()
        self._lock = threading.Lock()
        self._generation = 0
        self.mode = "webcam"
        self.camera_index = 0
        self.width = 1280
        self.height = 720
        self.video_path = ""
        self.image_path = ""
        self._seek = None
        self._restart = False
        self._scan = False
        self._logger = get_logger("capture")
        self._camera = CameraCapture()
        self._video = VideoCaptureSource()
        self._image = ImageSource()

    def configure(self, **kwargs) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self, key, value)
            self._generation += 1

    def set_active(self, active: bool) -> None:
        if active:
            self._paused.clear()
            self._active.set()
        else:
            self._active.clear()
            with self._lock:
                self._generation += 1

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._paused.set()
        else:
            self._paused.clear()

    def request_seek(self, fraction: float) -> None:
        with self._lock:
            self._seek = float(fraction)

    def request_restart(self) -> None:
        with self._lock:
            self._restart = True
            self._seek = 0.0

    def stop(self) -> None:
        self._stop.set()
        self._active.set()

    def request_scan(self) -> None:
        self._scan = True

    def scan_cameras(self) -> None:
        found = list_cameras()
        self._logger.info("Cameras found: %s", found)
        self.cameras_found.emit(found)

    def run(self) -> None:
        try:
            self.scan_cameras()
        except Exception as exc:
            self.source_failed.emit(str(exc))
        while not self._stop.is_set():
            if self._stop.is_set():
                break
            if self._scan and not self._active.is_set():
                self._scan = False
                try:
                    self.scan_cameras()
                except Exception as exc:
                    self.source_failed.emit(str(exc))
            if not self._active.is_set():
                self._close_sources()
                time.sleep(0.05)
                continue
            with self._lock:
                mode = self.mode
                generation = self._generation
                width = int(self.width)
                height = int(self.height)
                camera_index = int(self.camera_index)
                video_path = self.video_path
                image_path = self.image_path
            try:
                if mode == "webcam":
                    self._run_camera(generation, camera_index, width, height)
                elif mode == "video":
                    self._run_video(generation, video_path, width, height)
                elif mode == "image":
                    self._run_image(generation, image_path, width, height)
                else:
                    time.sleep(0.05)
            except Exception as exc:
                self._logger.exception("Capture failed")
                self.source_failed.emit(str(exc))
                self._active.clear()
                time.sleep(0.2)
        self._close_sources()

    def _alive(self, generation: int) -> bool:
        if self._stop.is_set() or not self._active.is_set():
            return False
        with self._lock:
            return generation == self._generation

    def _run_camera(self, generation: int, index: int, width: int, height: int) -> None:
        self._camera.open(index, width, height)
        self.source_opened.emit(f"Camera {index} {width}x{height}")
        self._logger.info("Resolution %sx%s", width, height)
        while self._alive(generation):
            if self._paused.is_set():
                time.sleep(0.03)
                continue
            sample = self._camera.read(width, height)
            if sample is None:
                time.sleep(0.01)
                continue
            frame, fps, capture_ms = sample
            self.hub.publish_rgb(frame, fps, capture_ms)
        self._camera.close()

    def _run_video(self, generation: int, path: str, width: int, height: int) -> None:
        if not path:
            self.source_failed.emit("Choose a video file first.")
            self._active.clear()
            return
        self._video.open(path)
        self.source_opened.emit(path)
        interval = 1.0 / self._video.fps
        next_tick = time.perf_counter()
        while self._alive(generation):
            with self._lock:
                seek = self._seek
                restart = self._restart
                self._seek = None
                self._restart = False
            if restart:
                self._video.restart()
            elif seek is not None:
                self._video.seek_fraction(seek)
            if self._paused.is_set() and seek is None and not restart:
                time.sleep(0.03)
                continue
            sample = self._video.read(width, height)
            if sample is None:
                self.video_finished.emit()
                self._paused.set()
                time.sleep(0.05)
                continue
            frame, fps, capture_ms, fraction = sample
            self.hub.publish_rgb(frame, fps, capture_ms)
            self.video_position.emit(fraction)
            next_tick += interval
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.perf_counter()
        self._video.close()

    def _run_image(self, generation: int, path: str, width: int, height: int) -> None:
        if not path:
            self.source_failed.emit("Choose an image first.")
            self._active.clear()
            return
        frame = self._image.open(path, width, height)
        self.hub.publish_rgb(frame, 0.0, 0.0)
        self.source_opened.emit(path)
        while self._alive(generation):
            time.sleep(0.05)
        self._image.close()

    def _close_sources(self) -> None:
        self._camera.close()
        self._video.close()
        self._image.close()
