"""Right-hand depth, measurement, and model panel."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter, QColor, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from utils.gpu_info import format_bytes


def _meters(value) -> str:
    if value is None:
        return "--"
    return f"{float(value):.2f} m"


class HistogramWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(120)
        self._counts: np.ndarray | None = None
        self._edges: np.ndarray | None = None

    def set_depth(self, depth) -> None:
        if depth is None:
            self._counts = None
            self.update()
            return
        values = depth[np.isfinite(depth) & (depth > 0)]
        if values.size == 0:
            self._counts = None
            self.update()
            return
        high = float(np.percentile(values, 99))
        high = max(0.5, min(20.0, high))
        counts, edges = np.histogram(values, bins=36, range=(0.0, high))
        self._counts = counts
        self._edges = edges
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(16, 18, 22))
        if self._counts is None or self._edges is None or self._counts.max() <= 0:
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignCenter, "No depth histogram")
            painter.end()
            return
        left, right, top, bottom = 36, 8, 8, 22
        width = max(1, self.width() - left - right)
        height = max(1, self.height() - top - bottom)
        peak = float(self._counts.max())
        bar_w = width / len(self._counts)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(90, 170, 230))
        for index, count in enumerate(self._counts):
            bar_h = int(height * (count / peak))
            painter.drawRect(int(left + index * bar_w), top + height - bar_h, max(1, int(bar_w - 1)), bar_h)
        painter.setPen(QPen(QColor(180, 180, 180)))
        painter.drawText(4, self.height() - 6, "0 m")
        painter.drawText(self.width() - 48, self.height() - 6, f"{self._edges[-1]:.1f} m")
        painter.drawText(4, 14, "px")
        painter.end()


class DepthInfoPanel(QWidget):
    options_changed = Signal()
    clear_points_clicked = Signal()
    clear_roi_clicked = Signal()
    point_cloud_toggled = Signal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.nearest = QLabel("--")
        self.farthest = QLabel("--")
        self.center = QLabel("--")
        self.cursor_depth = QLabel("--")
        self.cursor_xy = QLabel("X --   Y --")
        self.focal = QLabel("--")
        self.inference = QLabel("--")
        self.depth_fps = QLabel("--")
        self.camera_fps = QLabel("--")
        self.dropped = QLabel("0")

        self.cursor_check = QCheckBox("Show Cursor Depth")
        self.cursor_check.setChecked(True)
        self.center_check = QCheckBox("Show Center Depth")
        self.center_check.setChecked(True)
        self.fps_check = QCheckBox("Show FPS")
        self.fps_check.setChecked(True)
        self.focal_check = QCheckBox("Show Focal Length")
        self.focal_check.setChecked(True)
        self.legend_check = QCheckBox("Depth Legend")
        self.metrics_check = QCheckBox("Show metric labels")
        self.metrics_check.setChecked(True)

        self.points = QListWidget()
        self.points.setMaximumHeight(120)
        self.clear_points_btn = QPushButton("CLEAR POINTS")
        self.clear_roi_btn = QPushButton("CLEAR ROI")
        self.roi_check = QCheckBox("ROI Measurement")

        self.roi_min = QLabel("--")
        self.roi_median = QLabel("--")
        self.roi_mean = QLabel("--")
        self.roi_max = QLabel("--")

        self.bands_check = QCheckBox("Near/Far Analysis")
        self.bands_check.setChecked(True)
        self.near_spin = QDoubleSpinBox()
        self.mid_spin = QDoubleSpinBox()
        for spin in (self.near_spin, self.mid_spin):
            spin.setRange(0.1, 50.0)
            spin.setSingleStep(0.1)
            spin.setDecimals(2)
        self.near_spin.setValue(1.0)
        self.mid_spin.setValue(3.0)

        self.threshold_check = QCheckBox("Depth Threshold")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(50, 1000)
        self.threshold_slider.setValue(150)
        self.threshold_label = QLabel("Show objects closer than: 1.50 m")

        self.obstacle_check = QCheckBox("Obstacle Highlight")
        self.obstacle_spin = QDoubleSpinBox()
        self.obstacle_spin.setRange(0.2, 10.0)
        self.obstacle_spin.setSingleStep(0.1)
        self.obstacle_spin.setValue(1.0)
        self.obstacle_state = QLabel("Clear")
        self.obstacle_note = QLabel("Monocular estimate only. Not a safety sensor.")
        self.obstacle_note.setWordWrap(True)

        self.smooth_check = QCheckBox("Temporal Smoothing")
        self.smooth_check.setChecked(True)
        self.smooth_preset = QComboBox()
        for name in ("OFF", "LOW", "MEDIUM", "HIGH", "CUSTOM"):
            self.smooth_preset.addItem(name, name)
        self.smooth_preset.setCurrentText("MEDIUM")
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(35)
        self.alpha_label = QLabel("Smoothing Alpha 0.35")

        self.overlay_slider = QSlider(Qt.Horizontal)
        self.overlay_slider.setRange(0, 100)
        self.overlay_slider.setValue(55)
        self.overlay_label = QLabel("Overlay Alpha 55%")

        self.checkpoint = QLabel("MISSING")
        self.model_name = QLabel("Depth Pro")
        self.model_state = QLabel("NOT LOADED")
        self.gpu = QLabel("--")
        self.vram = QLabel("--")
        self.precision = QLabel("--")

        self.debug_check = QCheckBox("Debug")
        self.debug_label = QLabel("")
        self.debug_label.setWordWrap(True)
        self.debug_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.histogram_check = QCheckBox("Depth Histogram")
        self.histogram = HistogramWidget()
        self.histogram.setVisible(False)
        self.cloud_check = QCheckBox("3D Point Cloud")

        info = QGroupBox("DEPTH INFO")
        form = QFormLayout(info)
        form.addRow("Nearest", self.nearest)
        form.addRow("Farthest", self.farthest)
        form.addRow("Center", self.center)
        form.addRow("Cursor", self.cursor_depth)
        form.addRow("Position", self.cursor_xy)
        form.addRow("Focal Length", self.focal)
        form.addRow("Inference", self.inference)
        form.addRow("Depth FPS", self.depth_fps)
        form.addRow("Camera FPS", self.camera_fps)
        form.addRow("Dropped Frames", self.dropped)

        display = QGroupBox("DISPLAY")
        display_layout = QVBoxLayout(display)
        for widget in (
            self.cursor_check,
            self.center_check,
            self.fps_check,
            self.focal_check,
            self.legend_check,
            self.metrics_check,
            self.overlay_label,
            self.overlay_slider,
        ):
            display_layout.addWidget(widget)

        points_box = QGroupBox("MEASUREMENT POINTS")
        points_layout = QVBoxLayout(points_box)
        points_layout.addWidget(self.points)
        row = QHBoxLayout()
        row.addWidget(self.clear_points_btn)
        row.addWidget(self.clear_roi_btn)
        points_layout.addLayout(row)

        roi = QGroupBox("ROI")
        roi_form = QFormLayout(roi)
        roi_form.addRow(self.roi_check)
        roi_form.addRow("Min", self.roi_min)
        roi_form.addRow("Median", self.roi_median)
        roi_form.addRow("Mean", self.roi_mean)
        roi_form.addRow("Max", self.roi_max)

        bands = QGroupBox("NEAR / FAR")
        bands_form = QFormLayout(bands)
        bands_form.addRow(self.bands_check)
        bands_form.addRow("Near up to", self.near_spin)
        bands_form.addRow("Medium up to", self.mid_spin)

        threshold = QGroupBox("DEPTH THRESHOLD")
        threshold_layout = QVBoxLayout(threshold)
        threshold_layout.addWidget(self.threshold_check)
        threshold_layout.addWidget(self.threshold_label)
        threshold_layout.addWidget(self.threshold_slider)

        obstacle = QGroupBox("OBSTACLE")
        obstacle_layout = QVBoxLayout(obstacle)
        obstacle_layout.addWidget(self.obstacle_check)
        obstacle_form = QFormLayout()
        obstacle_form.addRow("Obstacle Distance", self.obstacle_spin)
        obstacle_layout.addLayout(obstacle_form)
        obstacle_layout.addWidget(self.obstacle_state)
        obstacle_layout.addWidget(self.obstacle_note)

        smooth = QGroupBox("STABILIZATION")
        smooth_form = QFormLayout(smooth)
        smooth_form.addRow(self.smooth_check)
        smooth_form.addRow("Mode", self.smooth_preset)
        smooth_form.addRow(self.alpha_label)
        smooth_form.addRow(self.alpha_slider)

        model = QGroupBox("MODEL STATUS")
        model_form = QFormLayout(model)
        model_form.addRow("Checkpoint", self.checkpoint)
        model_form.addRow("Model", self.model_name)
        model_form.addRow("Status", self.model_state)
        model_form.addRow("Precision", self.precision)
        model_form.addRow("GPU", self.gpu)
        model_form.addRow("VRAM", self.vram)

        debug = QGroupBox("DEBUG")
        debug_layout = QVBoxLayout(debug)
        debug_layout.addWidget(self.debug_check)
        debug_layout.addWidget(self.debug_label)

        extra = QGroupBox("EXTRA VIEWS")
        extra_layout = QVBoxLayout(extra)
        extra_layout.addWidget(self.histogram_check)
        extra_layout.addWidget(self.histogram)
        extra_layout.addWidget(self.cloud_check)

        content = QWidget()
        column = QVBoxLayout(content)
        for box in (info, display, points_box, roi, bands, threshold, obstacle, smooth, model, debug, extra):
            column.addWidget(box)
        column.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        self.setMinimumWidth(320)

        self.clear_points_btn.clicked.connect(self.clear_points_clicked)
        self.clear_roi_btn.clicked.connect(self.clear_roi_clicked)
        self.cloud_check.toggled.connect(self.point_cloud_toggled)
        self.histogram_check.toggled.connect(self.histogram.setVisible)
        self.threshold_slider.valueChanged.connect(self._refresh_threshold_label)
        self.alpha_slider.valueChanged.connect(self._refresh_alpha_label)
        self.overlay_slider.valueChanged.connect(self._refresh_overlay_label)
        self.smooth_preset.currentIndexChanged.connect(self._preset_changed)
        for widget in (
            self.cursor_check,
            self.center_check,
            self.fps_check,
            self.focal_check,
            self.legend_check,
            self.metrics_check,
            self.roi_check,
            self.bands_check,
            self.threshold_check,
            self.obstacle_check,
            self.smooth_check,
            self.debug_check,
        ):
            widget.toggled.connect(self.options_changed)
        for widget in (self.near_spin, self.mid_spin, self.obstacle_spin):
            widget.valueChanged.connect(self.options_changed)
        self.threshold_slider.valueChanged.connect(self.options_changed)
        self.alpha_slider.valueChanged.connect(self._alpha_edited)
        self.overlay_slider.valueChanged.connect(self.options_changed)

    def update_stats(self, analysis: dict, meta: dict, camera_fps: float) -> None:
        self.nearest.setText(_meters(analysis.get("nearest")))
        self.farthest.setText(_meters(analysis.get("farthest")))
        self.center.setText(_meters(analysis.get("center")))
        self.cursor_depth.setText(_meters(analysis.get("cursor_depth")))
        focal = meta.get("focal_px")
        self.focal.setText("--" if focal is None else f"{focal:.0f} px")
        inference = meta.get("inference_ms")
        self.inference.setText("--" if not inference else f"{inference:.0f} ms")
        depth_fps = meta.get("depth_fps") or 0
        self.depth_fps.setText(f"{depth_fps:.1f}")
        self.camera_fps.setText(f"{camera_fps:.1f}")
        self.dropped.setText(str(meta.get("dropped", 0)))
        roi = analysis.get("roi_stats")
        if roi is None:
            for label in (self.roi_min, self.roi_median, self.roi_mean, self.roi_max):
                label.setText("--")
        else:
            self.roi_min.setText(_meters(roi["min"]))
            self.roi_median.setText(_meters(roi["median"]))
            self.roi_mean.setText(_meters(roi["mean"]))
            self.roi_max.setText(_meters(roi["max"]))
        if analysis.get("obstacle_detected") and self.obstacle_check.isChecked():
            fraction = 100.0 * float(analysis.get("obstacle_fraction") or 0)
            self.obstacle_state.setText(f"OBSTACLE DETECTED ({fraction:.1f}%)")
        elif self.obstacle_check.isChecked():
            self.obstacle_state.setText("Clear")
        else:
            self.obstacle_state.setText("Off")

    def set_cursor_xy(self, x: float | None, y: float | None) -> None:
        if x is None or y is None:
            self.cursor_xy.setText("X --   Y --")
            return
        self.cursor_xy.setText(f"X {x:.0f}   Y {y:.0f}")

    def set_points(self, points: list[dict]) -> None:
        self.points.clear()
        for point in points:
            depth = point.get("depth")
            depth_text = "--" if depth is None else f"{depth:.2f} m"
            self.points.addItem(
                f"{point['name']}  {depth_text}  ({point['x']:.0f}, {point['y']:.0f})"
            )

    def set_model(self, checkpoint: str, status: str, precision: str, gpu: str, vram: str) -> None:
        self.checkpoint.setText(checkpoint)
        self.model_state.setText(status)
        self.precision.setText(precision)
        self.gpu.setText(gpu or "--")
        self.vram.setText(vram or "--")

    def set_debug(self, text: str) -> None:
        self.debug_label.setText(text if self.debug_check.isChecked() else "")

    def set_point_cloud_checked(self, checked: bool) -> None:
        self.cloud_check.blockSignals(True)
        self.cloud_check.setChecked(checked)
        self.cloud_check.blockSignals(False)

    def overlay_alpha(self) -> float:
        return self.overlay_slider.value() / 100.0

    def smoothing_alpha(self) -> float:
        return self.alpha_slider.value() / 100.0

    def threshold_meters(self) -> float:
        return self.threshold_slider.value() / 100.0

    def _refresh_threshold_label(self, value: int) -> None:
        self.threshold_label.setText(f"Show objects closer than: {value / 100.0:.2f} m")

    def _refresh_alpha_label(self, value: int) -> None:
        self.alpha_label.setText(f"Smoothing Alpha {value / 100.0:.2f}")

    def _refresh_overlay_label(self, value: int) -> None:
        self.overlay_label.setText(f"Overlay Alpha {value}%")

    def _alpha_edited(self) -> None:
        if self.smooth_preset.currentData() != "CUSTOM":
            known = {"OFF": None, "LOW": 65, "MEDIUM": 35, "HIGH": 15}
            current = known.get(self.smooth_preset.currentData())
            if current != self.alpha_slider.value():
                self.smooth_preset.blockSignals(True)
                self.smooth_preset.setCurrentText("CUSTOM")
                self.smooth_preset.blockSignals(False)
        self.options_changed.emit()

    def _preset_changed(self) -> None:
        preset = self.smooth_preset.currentData()
        values = {"LOW": 65, "MEDIUM": 35, "HIGH": 15}
        if preset == "OFF":
            self.smooth_check.setChecked(False)
        elif preset in values:
            self.smooth_check.setChecked(True)
            self.alpha_slider.blockSignals(True)
            self.alpha_slider.setValue(values[preset])
            self.alpha_slider.blockSignals(False)
            self._refresh_alpha_label(values[preset])
        self.options_changed.emit()


def vram_text(used: int, total: int) -> str:
    if total <= 0:
        return "--"
    return f"{format_bytes(used)} / {format_bytes(total)}"
