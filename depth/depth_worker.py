"""Background Depth Pro inference. Always consumes the newest frame only."""

from __future__ import annotations

import threading
import time

from PySide6.QtCore import QThread, Signal

from depth.depth_pro_backend import DepthProBackend
from depth.depth_smoothing import DepthSmoother
from utils.logger import get_logger
from utils.timers import FpsMeter


class DepthWorker(QThread):
    devices_ready = Signal(object)
    model_ready = Signal(object)
    model_failed = Signal(str)
    stats_ready = Signal(object)
    inference_failed = Signal(str)

    def __init__(self, hub) -> None:
        super().__init__()
        self.hub = hub
        self.backend = DepthProBackend()
        self.smoother = DepthSmoother()
        self._stop = threading.Event()
        self._infer = threading.Event()
        self._load = threading.Event()
        self._lock = threading.Lock()
        self.device_pref = "auto"
        self.precision_pref = "auto"
        self.scale = 1.0
        self.smoothing_enabled = True
        self.smoothing_alpha = 0.35
        self._paused = False
        self._reprocess = False
        self._logger = get_logger("depth")
        self._fps = FpsMeter(0.8)
        self._dropped = 0

    def request_load(self, device_pref: str, precision_pref: str) -> None:
        with self._lock:
            self.device_pref = device_pref
            self.precision_pref = precision_pref
        self._load.set()

    def set_infer_enabled(self, enabled: bool) -> None:
        if enabled:
            self._paused = False
            self._infer.set()
        else:
            self._infer.clear()

    def set_paused(self, paused: bool) -> None:
        self._paused = paused

    def set_scale(self, scale: float) -> None:
        with self._lock:
            self.scale = float(scale)

    def set_smoothing(self, enabled: bool, alpha: float) -> None:
        with self._lock:
            self.smoothing_enabled = bool(enabled)
            self.smoothing_alpha = float(alpha)

    def reset_filter(self) -> None:
        self.smoother.reset()

    def reprocess(self) -> None:
        self._reprocess = True

    def stop(self) -> None:
        self._stop.set()
        self._infer.clear()
        self._load.set()

    def run(self) -> None:
        try:
            devices = self.backend.runtime_info()
            self.devices_ready.emit(devices)
            self._logger.info(
                "Torch %s CUDA %s available=%s",
                devices.get("torch"),
                devices.get("cuda"),
                devices.get("cuda_available"),
            )
        except Exception as exc:
            self.model_failed.emit(str(exc))
            self._logger.exception("Device probe failed")

        last_frame_id = -1
        while not self._stop.is_set():
            if self._load.is_set():
                self._load.clear()
                self._load_model()
                last_frame_id = -1
                continue
            if self._reprocess:
                last_frame_id = -1
                self._reprocess = False
            if not self._infer.is_set() or self._paused or not self.backend.ready:
                time.sleep(0.02)
                continue
            latest = self.hub.latest_rgb_since(last_frame_id)
            if latest is None:
                time.sleep(0.005)
                continue
            frame_id, frame, _fps, _capture_ms = latest
            dropped = max(0, frame_id - last_frame_id - 1) if last_frame_id >= 0 else 0
            last_frame_id = frame_id
            self._dropped += dropped
            with self._lock:
                scale = self.scale
                smooth_on = self.smoothing_enabled
                smooth_alpha = self.smoothing_alpha
            try:
                result = self.backend.infer(frame, scale=scale)
                depth = self.smoother.apply(result["depth"], smooth_on, smooth_alpha)
                depth_fps = self._fps.tick()
                meta = {
                    "focal_px": result["focal_px"],
                    "preprocess_ms": result["preprocess_ms"],
                    "model_ms": result["model_ms"],
                    "post_ms": result["post_ms"],
                    "inference_ms": result["inference_ms"],
                    "depth_fps": depth_fps,
                    "dropped": self._dropped,
                    "dropped_delta": dropped,
                    "input_hw": result["input_hw"],
                    "source_hw": result["source_hw"],
                    "model_img_size": result["model_img_size"],
                    "frame_id": frame_id,
                    "device": self.backend.device_label,
                    "precision": self.backend.precision_label,
                }
                gpu = self.backend.status()
                meta["gpu_name"] = gpu.get("gpu_name") or ""
                meta["vram_used"] = gpu.get("vram_used") or 0
                meta["vram_total"] = gpu.get("vram_total") or 0
                self.hub.publish_depth(depth, meta)
                self.stats_ready.emit(meta)
            except Exception as exc:
                message = str(exc)
                self._logger.exception("Inference failed")
                self.inference_failed.emit(message)
                if "out of memory" in message.lower():
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        self._logger.exception("Could not clear CUDA cache")
                time.sleep(0.3)

    def _load_model(self) -> None:
        if self._stop.is_set():
            return
        with self._lock:
            device_pref = self.device_pref
            precision_pref = self.precision_pref
        self._logger.info(
            "Loading Depth Pro device=%s precision=%s checkpoint=%s",
            device_pref,
            precision_pref,
            self.backend.checkpoint,
        )
        try:
            status = self.backend.load(device_pref, precision_pref)
            self.smoother.reset()
            self._fps.reset()
            self._logger.info(
                "Depth Pro loaded device=%s precision=%s",
                status.get("device"),
                status.get("precision"),
            )
            self.model_ready.emit(status)
        except Exception as exc:
            self._logger.exception("Model load failed")
            self.model_failed.emit(str(exc))
