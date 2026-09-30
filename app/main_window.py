"""Depth Pro Vision Studio main window."""

from __future__ import annotations

import time
from datetime import datetime

import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from app.control_panel import ControlPanel
from app.demo_saved_dialog import DemoSavedDialog
from app.depth_info_panel import DepthInfoPanel, vram_text
from app.model_manager_dialog import ModelManagerDialog
from app.settings_dialog import SettingsDialog
from app.video_widget import VideoWidget
from capture.frame_hub import FrameHub
from capture.source_manager import SourceManager
from depth.depth_analyzer import analyze
from depth.depth_worker import DepthWorker
from recording.demo_recorder import DemoRecorder
from utils.downloader import checkpoint_info
from utils.images import imwrite
from utils.logger import get_logger
from utils.paths import (
    DEPTH_MAP_DIR,
    RAW_DEPTH_DIR,
    SCREENSHOT_DIR,
    ensure_dirs,
)
from utils.settings_store import load_settings, save_settings
from visualization.depth_colormap import colorize
from visualization.renderer import RenderState, render


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Depth Pro Vision Studio")
        self.resize(1600, 940)
        self._log = get_logger("app")
        ensure_dirs()
        self.settings = load_settings()
        self.hub = FrameHub()
        self.capture = SourceManager(self.hub)
        self.depth_worker = DepthWorker(self.hub)
        self.recorder = DemoRecorder(self.hub)

        self.points: list[dict] = []
        self.roi = None
        self.cursor = None
        self.current_rgb = None
        self.current_depth = None
        self.depth_meta: dict = {}
        self.analysis: dict = {}
        self.model_status: dict = {"ready": False}
        self.camera_fps = 0.0
        self.capture_ms = 0.0
        self._rgb_id = -1
        self._depth_seen = -1
        self._running = False
        self._paused = False
        self._countdown_value = 0
        self._countdown_started = 0.0
        self._recording = False
        self._record_remaining = 0.0
        self._render_ms = 0.0
        self._cloud = None
        self._last_histogram = 0.0
        self._last_cloud = 0.0
        self._oom_reported = False
        self._arm_recording = False
        self._point_sig = None
        self._video_path = ""
        self._image_path = ""
        self._devices: list[dict] = []

        self.controls = ControlPanel()
        self.video = VideoWidget()
        self.info = DepthInfoPanel()
        center = QWidget()
        row = QHBoxLayout(center)
        row.setContentsMargins(8, 0, 8, 0)
        row.addWidget(self.video, 1)
        row.addWidget(self.info)
        shell = QWidget()
        layout = QVBoxLayout(shell)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.controls)
        layout.addWidget(center, 1)
        self.setCentralWidget(shell)
        self.statusBar().showMessage("Depth Pro Vision Studio")

        self._connect()
        self._apply_settings()
        geometry = self.settings.get("geometry")
        if isinstance(geometry, list) and len(geometry) == 4:
            self.setGeometry(*[int(v) for v in geometry])

        self.capture.start()
        self.depth_worker.start()
        self.recorder.start()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start(33)
        self._refresh_model_labels()

    def _connect(self) -> None:
        c = self.controls
        c.start_clicked.connect(self._start)
        c.stop_clicked.connect(self._stop)
        c.pause_clicked.connect(self._pause)
        c.record_clicked.connect(self._record)
        c.cancel_clicked.connect(self._cancel_record)
        c.screenshot_clicked.connect(self._save_screenshot)
        c.save_rgb_clicked.connect(self._save_rgb)
        c.save_depth_clicked.connect(self._save_depth)
        c.save_raw_clicked.connect(self._save_raw)
        c.model_clicked.connect(self._open_model_manager)
        c.settings_clicked.connect(self._open_settings)
        c.browse_clicked.connect(self._browse)
        c.rescan_clicked.connect(self._rescan_cameras)
        c.source_changed.connect(self._on_source_changed)
        c.camera_changed.connect(self._on_camera_changed)
        c.resolution_changed.connect(lambda _value: self._restart_if_running())
        c.device_changed.connect(lambda _value: self._reload_model())
        c.precision_changed.connect(lambda _value: self._reload_model())
        c.scale_changed.connect(self._on_scale_changed)
        c.play_clicked.connect(self._start)
        c.restart_clicked.connect(self._restart_video)
        c.seek_changed.connect(self._seek_video)
        c.view_changed.connect(lambda _value: None)
        c.colormap_changed.connect(lambda _value: None)

        self.video.cursor_at.connect(self._on_cursor)
        self.video.cursor_cleared.connect(self._on_cursor_cleared)
        self.video.point_clicked.connect(self._on_point)
        self.video.roi_finished.connect(self._on_roi)
        self.info.options_changed.connect(self._on_options)
        self.info.clear_points_clicked.connect(self._clear_points)
        self.info.clear_roi_clicked.connect(self._clear_roi)
        self.info.point_cloud_toggled.connect(self._toggle_cloud)

        self.capture.cameras_found.connect(self._on_cameras)
        self.capture.source_opened.connect(self._on_source_opened)
        self.capture.source_failed.connect(self._on_source_failed)
        self.capture.video_position.connect(self.controls.set_video_position)
        self.capture.video_finished.connect(lambda: self.statusBar().showMessage("Video finished"))
        self.depth_worker.devices_ready.connect(self._on_devices)
        self.depth_worker.model_ready.connect(self._on_model_ready)
        self.depth_worker.model_failed.connect(self._on_model_failed)
        self.depth_worker.inference_failed.connect(self._on_inference_failed)
        self.recorder.progress.connect(self._on_record_progress)
        self.recorder.finished_ok.connect(self._on_record_finished)
        self.recorder.cancelled.connect(self._on_record_cancelled)
        self.recorder.failed.connect(self._on_record_failed)

    def _apply_settings(self) -> None:
        s = self.settings
        self.controls.select_value(self.controls.source, s.get("source", "webcam"))
        self.controls.select_value(self.controls.resolution, s.get("resolution", "1280x720"))
        self.controls.select_value(self.controls.device, s.get("device", "auto"))
        self.controls.select_value(self.controls.precision, s.get("precision", "auto"))
        self.controls.select_value(self.controls.view, s.get("view_mode", "split"))
        self.controls.select_value(self.controls.colormap, s.get("colormap", "turbo"))
        self.controls.select_value(self.controls.scale, float(s.get("processing_scale", 1.0)))
        self.controls.select_value(self.controls.demo_resolution, s.get("demo_resolution", "1920x1080"))
        self.info.cursor_check.setChecked(bool(s.get("show_cursor", True)))
        self.info.center_check.setChecked(bool(s.get("show_center", True)))
        self.info.fps_check.setChecked(bool(s.get("show_fps", True)))
        self.info.focal_check.setChecked(bool(s.get("show_focal", True)))
        self.info.legend_check.setChecked(bool(s.get("legend_enabled", False)))
        self.info.bands_check.setChecked(bool(s.get("bands_enabled", True)))
        self.info.near_spin.setValue(float(s.get("near_m", 1.0)))
        self.info.mid_spin.setValue(float(s.get("mid_m", 3.0)))
        self.info.threshold_check.setChecked(bool(s.get("threshold_enabled", False)))
        self.info.threshold_slider.setValue(int(float(s.get("threshold_m", 1.5)) * 100))
        self.info.obstacle_check.setChecked(bool(s.get("obstacle_enabled", False)))
        self.info.obstacle_spin.setValue(float(s.get("obstacle_m", 1.0)))
        self.info.smooth_check.setChecked(bool(s.get("smoothing_enabled", True)))
        self.info.smooth_preset.setCurrentText(str(s.get("smoothing_preset", "MEDIUM")))
        self.info.alpha_slider.setValue(int(float(s.get("smoothing_alpha", 0.35)) * 100))
        self.info.overlay_slider.setValue(int(float(s.get("overlay_alpha", 0.55)) * 100))
        self.info.debug_check.setChecked(bool(s.get("debug", False)))
        self.info.histogram_check.setChecked(bool(s.get("histogram", False)))
        self.controls.set_transport_enabled(self.controls.source.currentData() == "video")
        self._push_smoothing()

    def _collect_settings(self) -> dict:
        geometry = self.geometry()
        self.settings.update(
            {
                "source": self.controls.source.currentData(),
                "camera_index": int(self.controls.camera.currentData() or 0),
                "resolution": self.controls.resolution.currentData(),
                "device": self.controls.device.currentData(),
                "precision": self.controls.precision.currentData(),
                "view_mode": self.controls.view.currentData(),
                "colormap": self.controls.colormap.currentData(),
                "processing_scale": float(self.controls.scale.currentData()),
                "overlay_alpha": self.info.overlay_alpha(),
                "smoothing_enabled": self.info.smooth_check.isChecked(),
                "smoothing_preset": self.info.smooth_preset.currentData(),
                "smoothing_alpha": self.info.smoothing_alpha(),
                "near_m": float(self.info.near_spin.value()),
                "mid_m": float(self.info.mid_spin.value()),
                "bands_enabled": self.info.bands_check.isChecked(),
                "threshold_enabled": self.info.threshold_check.isChecked(),
                "threshold_m": self.info.threshold_meters(),
                "obstacle_enabled": self.info.obstacle_check.isChecked(),
                "obstacle_m": float(self.info.obstacle_spin.value()),
                "show_cursor": self.info.cursor_check.isChecked(),
                "show_center": self.info.center_check.isChecked(),
                "show_fps": self.info.fps_check.isChecked(),
                "show_focal": self.info.focal_check.isChecked(),
                "legend_enabled": self.info.legend_check.isChecked(),
                "debug": self.info.debug_check.isChecked(),
                "histogram": self.info.histogram_check.isChecked(),
                "demo_resolution": self.controls.demo_resolution.currentData(),
                "demo_fps": int(self.settings.get("demo_fps", 30)),
                "max_points": int(self.settings.get("max_points", 10)),
                "geometry": [geometry.x(), geometry.y(), geometry.width(), geometry.height()],
            }
        )
        return self.settings

    def _on_devices(self, runtime: dict) -> None:
        self._devices = list(runtime.get("devices") or [])
        self.controls.set_devices(self._devices, self.settings.get("device", "auto"))
        self._log.info(
            "Torch %s CUDA %s GPUs %s",
            runtime.get("torch"),
            runtime.get("cuda"),
            [item.get("name") for item in self._devices],
        )
        if checkpoint_info()["status"] == "INSTALLED":
            self._reload_model()
        else:
            self.info.set_model("MISSING", "CHECKPOINT REQUIRED", "--", "", "")

    def _reload_model(self) -> None:
        if checkpoint_info()["status"] != "INSTALLED":
            self.info.set_model("MISSING", "CHECKPOINT REQUIRED", "--", "", "")
            return
        device = self.controls.device.currentData() or "auto"
        precision = self.controls.precision.currentData() or "auto"
        self.info.set_model(checkpoint_info()["status"], "LOADING", precision.upper(), "", "")
        self.statusBar().showMessage("Loading Depth Pro...")
        self.depth_worker.request_load(device, precision)

    def _on_model_ready(self, status: dict) -> None:
        self.model_status = status
        self._refresh_model_labels()
        self._log.info("Depth Pro loaded checkpoint=%s", status.get("checkpoint"))
        self.statusBar().showMessage(f"Model ready on {status.get('device')}")
        if self._running:
            self.depth_worker.set_infer_enabled(True)
            self._log.info("Inference started")

    def _on_model_failed(self, message: str) -> None:
        self.model_status = {"ready": False}
        self.info.set_model(checkpoint_info()["status"], "ERROR", "--", "", "")
        self.statusBar().showMessage(message)
        self._log.error("Model load error: %s", message)
        QMessageBox.warning(self, "Depth Pro", message)

    def _refresh_model_labels(self) -> None:
        info = checkpoint_info()
        status = self.model_status
        if status.get("ready"):
            state = "READY"
        elif info["status"] != "INSTALLED":
            state = "CHECKPOINT REQUIRED"
        else:
            state = self.info.model_state.text()
        gpu = status.get("gpu_name") or ""
        self.info.set_model(
            info["status"],
            state,
            str(status.get("precision") or "--"),
            gpu,
            vram_text(int(status.get("vram_used") or 0), int(status.get("vram_total") or 0)),
        )

    def _on_cameras(self, indices) -> None:
        self.controls.set_cameras(list(indices))
        saved = int(self.settings.get("camera_index", 0))
        self.controls.select_value(self.controls.camera, saved if saved in list(indices) else (indices[0] if indices else -1))

    def _start(self) -> None:
        if not self._configure_capture():
            return
        self._paused = False
        self._running = True
        self.capture.set_paused(False)
        self.depth_worker.set_paused(False)
        self.depth_worker.reset_filter()
        self.capture.set_active(True)
        if self.model_status.get("ready"):
            self.depth_worker.set_infer_enabled(True)
            self._log.info("Inference started")
            self.statusBar().showMessage("Running")
        else:
            self.statusBar().showMessage("Source started. Depth will begin when the model is ready.")

    def _stop(self) -> None:
        self._running = False
        self._paused = False
        self.capture.set_active(False)
        self.depth_worker.set_infer_enabled(False)
        self._log.info("Inference stopped")
        self.statusBar().showMessage("Stopped")

    def _pause(self) -> None:
        if not self._running:
            return
        self._paused = not self._paused
        self.capture.set_paused(self._paused)
        self.depth_worker.set_paused(self._paused)
        self.statusBar().showMessage("Paused" if self._paused else "Running")

    def _configure_capture(self) -> bool:
        mode = self.controls.source.currentData()
        width, height = _parse_resolution(self.controls.resolution.currentData())
        camera = self.controls.camera.currentData()
        if mode == "webcam" and (camera is None or int(camera) < 0):
            QMessageBox.information(self, "Camera", "No webcam was found.")
            return False
        if mode == "video" and not self._video_path:
            self._browse()
            if not self._video_path:
                return False
        if mode == "image" and not self._image_path:
            self._browse()
            if not self._image_path:
                return False
        self.capture.configure(
            mode=mode,
            camera_index=int(camera or 0),
            width=width,
            height=height,
            video_path=self._video_path,
            image_path=self._image_path,
        )
        self.depth_worker.set_scale(float(self.controls.scale.currentData()))
        self._log.info("Camera selected %s resolution %sx%s", camera, width, height)
        return True

    def _rescan_cameras(self) -> None:
        if self._running:
            self.statusBar().showMessage("Stop the source before rescanning cameras.")
            return
        self.capture.request_scan()

    def _restart_if_running(self) -> None:
        if self._running:
            self._start()

    def _on_source_changed(self, source: str) -> None:
        self.controls.set_transport_enabled(source == "video")
        self.depth_worker.reset_filter()
        self.hub.clear_depth()
        self._depth_seen = -1
        self.current_depth = None
        if self._running:
            self._start()

    def _on_camera_changed(self, _index: int) -> None:
        self.depth_worker.reset_filter()
        if self._running and self.controls.source.currentData() == "webcam":
            self._start()

    def _on_scale_changed(self, scale: float) -> None:
        self.depth_worker.set_scale(scale)
        self.depth_worker.reset_filter()
        self.depth_worker.reprocess()

    def _on_options(self) -> None:
        self.video.roi_mode = self.info.roi_check.isChecked()
        self._push_smoothing()

    def _push_smoothing(self) -> None:
        enabled = self.info.smooth_check.isChecked() and self.info.smooth_preset.currentData() != "OFF"
        self.depth_worker.set_smoothing(enabled, self.info.smoothing_alpha())

    def _browse(self) -> None:
        mode = self.controls.source.currentData()
        if mode == "video":
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Open video",
                "",
                "Video (*.mp4 *.avi *.mov *.mkv)",
            )
            if path:
                self._video_path = path
                self.depth_worker.reset_filter()
                if self._running:
                    self._start()
        elif mode == "image":
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Open image",
                "",
                "Image (*.jpg *.jpeg *.png *.webp)",
            )
            if path:
                self._image_path = path
                self.depth_worker.reset_filter()
                self._start()
        else:
            self.statusBar().showMessage("Webcam does not use a file. Press START.")

    def _restart_video(self) -> None:
        self.depth_worker.reset_filter()
        self.capture.request_restart()
        if not self._running:
            self._start()
        else:
            self.capture.set_paused(False)
            self._paused = False

    def _seek_video(self, fraction: float) -> None:
        self.depth_worker.reset_filter()
        self.capture.request_seek(fraction)

    def _on_source_opened(self, message: str) -> None:
        self.statusBar().showMessage(message)

    def _on_source_failed(self, message: str) -> None:
        self._log.error("Source error: %s", message)
        self.statusBar().showMessage(message)
        QMessageBox.warning(self, "Source", message)

    def _on_cursor(self, x: float, y: float) -> None:
        self.cursor = (x, y)
        self.info.set_cursor_xy(x, y)

    def _on_cursor_cleared(self) -> None:
        self.cursor = None
        self.info.set_cursor_xy(None, None)

    def _on_point(self, x: float, y: float) -> None:
        limit = int(self.settings.get("max_points", 10))
        if len(self.points) >= limit:
            self.statusBar().showMessage(f"Maximum {limit} points. Clear points to add more.")
            return
        self.points.append({"name": f"P{len(self.points) + 1}", "x": x, "y": y})

    def _on_roi(self, x: float, y: float, w: float, h: float) -> None:
        self.roi = (x, y, w, h)

    def _clear_points(self) -> None:
        self.points.clear()
        self.info.set_points([])

    def _clear_roi(self) -> None:
        self.roi = None

    def _on_inference_failed(self, message: str) -> None:
        self.statusBar().showMessage(message)
        if "out of memory" in message.lower() and not self._oom_reported:
            self._oom_reported = True
            QMessageBox.warning(
                self,
                "GPU memory",
                "Depth Pro ran out of GPU memory. Choose the GPU with more VRAM, "
                "switch Precision to FP16, or use CPU.",
            )

    def _record(self) -> None:
        if self._recording or self._countdown_value:
            return
        if self.current_rgb is None:
            QMessageBox.information(self, "Demo", "Start a webcam, video, or image before recording.")
            return
        self._countdown_value = 3
        self._countdown_started = time.perf_counter()
        self.controls.set_recording_text("STARTING 3")
        self.controls.set_cancel_enabled(True)
        self.statusBar().showMessage("Demo countdown")

    def _begin_recording(self) -> None:
        width, height = _parse_resolution(self.controls.demo_resolution.currentData())
        fps = int(self.settings.get("demo_fps", 30))
        self._recording = True
        self._record_remaining = 15.0
        path = self.recorder.request_start(width, height, fps, 15.0)
        self._log.info("Recording started %s", path)
        self.controls.set_recording_text("RECORDING 00:15")

    def _cancel_record(self) -> None:
        if self._countdown_value or self._arm_recording:
            self._countdown_value = 0
            self._arm_recording = False
            self.controls.set_recording_text("RECORD 15s DEMO")
            self.controls.set_cancel_enabled(False)
            self.statusBar().showMessage("Demo cancelled")
            return
        if self._recording:
            self.recorder.request_cancel()

    def _on_record_progress(self, remaining: float) -> None:
        self._record_remaining = remaining
        seconds = max(0, int(np.ceil(remaining - 1e-6)))
        self.controls.set_recording_text(f"RECORDING 00:{seconds:02d}")

    def _on_record_finished(self, info: dict) -> None:
        self._recording = False
        self._record_remaining = 0
        self.controls.set_recording_text("RECORD 15s DEMO")
        self.controls.set_cancel_enabled(False)
        self._log.info("Recording completed %s", info.get("path"))
        DemoSavedDialog(info, self).exec()

    def _on_record_cancelled(self, info: dict) -> None:
        self._recording = False
        self.controls.set_recording_text("RECORD 15s DEMO")
        self.controls.set_cancel_enabled(False)
        self._log.info("Recording cancelled %s", info.get("path"))
        answer = QMessageBox.question(
            self,
            "Demo cancelled",
            "Delete the unfinished demo file?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        path = info.get("path")
        if answer == QMessageBox.Yes and path:
            try:
                from pathlib import Path

                Path(path).unlink(missing_ok=True)
            except OSError as exc:
                self._log.error("Could not delete cancelled demo: %s", exc)
        self.statusBar().showMessage("Demo cancelled")

    def _on_record_failed(self, message: str) -> None:
        self._recording = False
        self._countdown_value = 0
        self.controls.set_recording_text("RECORD 15s DEMO")
        self.controls.set_cancel_enabled(False)
        self._log.error("Recording error: %s", message)
        QMessageBox.warning(self, "Recording", message)

    def _save_screenshot(self) -> None:
        image, _layout = self._render_canvas(*_parse_resolution(self.controls.demo_resolution.currentData()))
        path = SCREENSHOT_DIR / f"depth_pro_{_stamp()}.png"
        imwrite(path, image)
        self._log.info("Screenshot saved %s", path)
        self.statusBar().showMessage(f"Screenshot saved {path}")

    def _save_rgb(self) -> None:
        if self.current_rgb is None:
            QMessageBox.information(self, "Save RGB", "There is no camera frame yet.")
            return
        path = SCREENSHOT_DIR / f"depth_pro_rgb_{_stamp()}.png"
        imwrite(path, self.current_rgb)
        self._log.info("Screenshot saved %s", path)
        self.statusBar().showMessage(f"RGB frame saved {path}")

    def _save_depth(self) -> None:
        colored, _near, _far = colorize(self.current_depth, self.controls.colormap.currentData())
        if colored is None:
            QMessageBox.information(self, "Save depth", "There is no depth map yet.")
            return
        path = DEPTH_MAP_DIR / f"depth_pro_{_stamp()}.png"
        imwrite(path, colored)
        self._log.info("Depth map saved %s", path)
        self.statusBar().showMessage(f"Depth map saved {path}")

    def _save_raw(self) -> None:
        if self.current_depth is None:
            QMessageBox.information(self, "Save raw depth", "There is no depth map yet.")
            return
        path = RAW_DEPTH_DIR / f"depth_pro_{_stamp()}.npz"
        focal = self.depth_meta.get("focal_px")
        np.savez_compressed(
            path,
            depth_meters=np.asarray(self.current_depth, dtype=np.float32),
            focal_length_px=np.float32(focal if focal is not None else np.nan),
            timestamp=np.array(_now_iso()),
        )
        self._log.info("Raw depth saved %s", path)
        self.statusBar().showMessage(f"Raw depth saved {path}")

    def _open_model_manager(self) -> None:
        dialog = ModelManagerDialog(self)
        dialog.checkpoint_ready.connect(self._reload_model)
        dialog.exec()
        self._refresh_model_labels()

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec():
            self.settings.update(dialog.values())
            self.controls.select_value(self.controls.demo_resolution, self.settings["demo_resolution"])
            save_settings(self._collect_settings())

    def _toggle_cloud(self, enabled: bool) -> None:
        if not enabled:
            if self._cloud is not None:
                self._cloud.hide()
            return
        try:
            if self._cloud is None:
                from app.point_cloud_dialog import PointCloudDialog

                self._cloud = PointCloudDialog(self)
            self._cloud.show()
            self._cloud.raise_()
            self._cloud.update_cloud(self.current_depth, self.depth_meta.get("focal_px"))
        except Exception as exc:
            self._log.exception("Point cloud view failed")
            self.info.set_point_cloud_checked(False)
            QMessageBox.warning(
                self,
                "3D point cloud",
                "The point cloud view could not start. The rest of the app is unaffected.\n\n"
                f"{exc}",
            )

    def _on_tick(self) -> None:
        self._update_countdown()
        rgb_pack = self.hub.latest_rgb()
        if rgb_pack is not None:
            frame_id, frame, fps, capture_ms = rgb_pack
            if frame_id != self._rgb_id:
                self._rgb_id = frame_id
                self.current_rgb = frame
            self.camera_fps = fps
            self.capture_ms = capture_ms
        depth_pack = self.hub.latest_depth_since(self._depth_seen)
        if depth_pack is not None:
            depth_id, depth, meta = depth_pack
            self._depth_seen = depth_id
            self.current_depth = depth
            self.depth_meta = meta
            if meta.get("vram_total"):
                self.model_status["vram_used"] = meta["vram_used"]
                self.model_status["vram_total"] = meta["vram_total"]
                self.model_status["gpu_name"] = meta.get("gpu_name") or self.model_status.get("gpu_name")
                self._refresh_model_labels()

        src_w, src_h = self._source_size()
        self.analysis = analyze(
            self.current_depth,
            src_w,
            src_h,
            self.cursor,
            self.points,
            self.roi,
            float(self.info.near_spin.value()),
            float(self.info.mid_spin.value()),
            float(self.info.obstacle_spin.value()),
        )
        self.points = list(self.analysis.get("points", self.points))
        self.info.update_stats(self.analysis, self.depth_meta, self.camera_fps)
        signature = tuple(
            (
                point["name"],
                None if point.get("depth") is None else round(float(point["depth"]), 2),
                round(point["x"]),
                round(point["y"]),
            )
            for point in self.points
        )
        if signature != self._point_sig:
            self._point_sig = signature
            self.info.set_points(self.points)
        self._update_debug()
        now = time.perf_counter()
        if self.info.histogram_check.isChecked() and now - self._last_histogram > 0.25:
            self.info.histogram.set_depth(self.current_depth)
            self._last_histogram = now
        if self._cloud is not None and self._cloud.isVisible() and now - self._last_cloud > 0.7:
            self._cloud.update_cloud(self.current_depth, self.depth_meta.get("focal_px"))
            self._last_cloud = now

        canvas = self._canvas_size()
        started = time.perf_counter()
        image, layout = self._render_canvas(*canvas)
        self._render_ms = (time.perf_counter() - started) * 1000.0
        self.video.set_frame(image, layout)
        self.hub.set_composite(image, layout)
        if self._arm_recording:
            self._arm_recording = False
            self._begin_recording()

    def _update_countdown(self) -> None:
        if not self._countdown_value:
            return
        elapsed = time.perf_counter() - self._countdown_started
        remaining = 3 - int(elapsed)
        if elapsed >= 3.0:
            self._countdown_value = 0
            self._recording = True
            self._record_remaining = 15.0
            self._arm_recording = True
            self.controls.set_recording_text("RECORDING 00:15")
            self.controls.set_cancel_enabled(True)
            return
        self._countdown_value = remaining
        self.controls.set_recording_text(f"STARTING {remaining}")

    def _canvas_size(self) -> tuple[int, int]:
        if self._recording or self._countdown_value:
            return _parse_resolution(self.controls.demo_resolution.currentData())
        width = max(640, self.video.width())
        height = max(360, self.video.height())
        return width - width % 2, height - height % 2

    def _source_size(self) -> tuple[int, int]:
        if self.current_rgb is not None:
            height, width = self.current_rgb.shape[:2]
            return width, height
        return _parse_resolution(self.controls.resolution.currentData())

    def _render_canvas(self, width: int, height: int):
        view = self.controls.view.currentData() or "split"
        state = RenderState(
            rgb=self.current_rgb,
            depth=self.current_depth,
            source_size=self._source_size(),
            view_mode=view,
            colormap=self.controls.colormap.currentData() or "turbo",
            overlay_alpha=self.info.overlay_alpha(),
            legend=bool(self.info.legend_check.isChecked() or view == "legend"),
            show_cursor=self.info.cursor_check.isChecked(),
            cursor_xy=self.cursor,
            cursor_depth=self.analysis.get("cursor_depth"),
            show_center=self.info.center_check.isChecked(),
            center_depth=self.analysis.get("center"),
            points=list(self.analysis.get("points") or []),
            roi=self.roi,
            roi_stats=self.analysis.get("roi_stats"),
            show_fps=self.info.fps_check.isChecked(),
            show_focal=self.info.focal_check.isChecked(),
            show_metrics=self.info.metrics_check.isChecked(),
            camera_fps=self.camera_fps,
            depth_fps=float(self.depth_meta.get("depth_fps") or 0),
            focal_px=self.depth_meta.get("focal_px"),
            inference_ms=float(self.depth_meta.get("inference_ms") or 0),
            nearest=self.analysis.get("nearest"),
            farthest=self.analysis.get("farthest"),
            near_m=float(self.info.near_spin.value()),
            mid_m=float(self.info.mid_spin.value()),
            bands=self.info.bands_check.isChecked(),
            threshold_on=self.info.threshold_check.isChecked(),
            threshold_m=self.info.threshold_meters(),
            obstacle_on=self.info.obstacle_check.isChecked(),
            obstacle_m=float(self.info.obstacle_spin.value()),
            obstacle_detected=bool(self.analysis.get("obstacle_detected")),
            countdown=self._countdown_value or None,
            recording=self._recording,
            record_remaining=self._record_remaining,
            record_duration=15.0,
        )
        return render(state, width, height)

    def _update_debug(self) -> None:
        if not self.info.debug_check.isChecked():
            self.info.set_debug("")
            return
        meta = self.depth_meta
        input_hw = meta.get("input_hw")
        text = "\n".join(
            [
                f"Frame ID: {self._rgb_id}",
                f"Camera FPS: {self.camera_fps:.2f}",
                f"Depth FPS: {float(meta.get('depth_fps') or 0):.2f}",
                f"Capture ms: {self.capture_ms:.1f}",
                f"Preprocess ms: {float(meta.get('preprocess_ms') or 0):.1f}",
                f"Inference ms: {float(meta.get('model_ms') or 0):.1f}",
                f"Postprocess ms: {float(meta.get('post_ms') or 0):.1f}",
                f"Render ms: {self._render_ms:.1f}",
                f"Recording ms: {self.recorder.write_ms:.1f}",
                f"Dropped frames: {meta.get('dropped', 0)}",
                f"GPU: {meta.get('gpu_name') or meta.get('device') or '--'}",
                f"VRAM: {vram_text(int(meta.get('vram_used') or 0), int(meta.get('vram_total') or 0))}",
                f"Input tensor size: {input_hw} model {meta.get('model_img_size')}",
                f"Depth min/max: {self.analysis.get('nearest')} / {self.analysis.get('farthest')}",
            ]
        )
        self.info.set_debug(text)

    def closeEvent(self, event) -> None:
        self._log.info("Shutdown")
        try:
            save_settings(self._collect_settings())
        except Exception:
            self._log.exception("Could not save settings")
        self._timer.stop()
        if self._recording:
            self.recorder.request_cancel()
        self.capture.stop()
        self.depth_worker.stop()
        self.recorder.shutdown()
        self.capture.wait(3000)
        self.depth_worker.wait(4000)
        self.recorder.wait(4000)
        event.accept()


def _parse_resolution(value: str) -> tuple[int, int]:
    text = str(value or "1280x720")
    width, height = text.lower().split("x")
    return int(width), int(height)


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")
