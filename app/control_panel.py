"""Top control bar: source, device, view, and the main actions."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class ControlPanel(QWidget):
    start_clicked = Signal()
    stop_clicked = Signal()
    pause_clicked = Signal()
    record_clicked = Signal()
    cancel_clicked = Signal()
    screenshot_clicked = Signal()
    save_rgb_clicked = Signal()
    save_depth_clicked = Signal()
    save_raw_clicked = Signal()
    model_clicked = Signal()
    settings_clicked = Signal()
    browse_clicked = Signal()
    rescan_clicked = Signal()
    source_changed = Signal(str)
    camera_changed = Signal(int)
    resolution_changed = Signal(str)
    device_changed = Signal(str)
    precision_changed = Signal(str)
    view_changed = Signal(str)
    colormap_changed = Signal(str)
    scale_changed = Signal(float)
    demo_resolution_changed = Signal(str)
    play_clicked = Signal()
    video_stop_clicked = Signal()
    restart_clicked = Signal()
    seek_changed = Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.source = QComboBox()
        self.source.addItem("Webcam", "webcam")
        self.source.addItem("Video File", "video")
        self.source.addItem("Image", "image")

        self.camera = QComboBox()
        self.camera.addItem("Camera 0", 0)
        self.resolution = QComboBox()
        for item in ("640x480", "1280x720", "1920x1080"):
            self.resolution.addItem(item, item)
        self.device = QComboBox()
        self.device.addItem("AUTO", "auto")
        self.device.addItem("CPU", "cpu")
        self.precision = QComboBox()
        for label, value in (("AUTO", "auto"), ("FP32", "fp32"), ("FP16", "fp16")):
            self.precision.addItem(label, value)
        self.view = QComboBox()
        for label, value in (
            ("RGB", "rgb"),
            ("DEPTH", "depth"),
            ("SPLIT", "split"),
            ("OVERLAY", "overlay"),
            ("DEPTH + LEGEND", "legend"),
        ):
            self.view.addItem(label, value)
        self.colormap = QComboBox()
        for label, value in (
            ("Inferno", "inferno"),
            ("Turbo", "turbo"),
            ("Viridis", "viridis"),
            ("Magma", "magma"),
            ("Grayscale", "grayscale"),
        ):
            self.colormap.addItem(label, value)
        self.scale = QComboBox()
        self.scale.addItem("100%", 1.0)
        self.scale.addItem("75%", 0.75)
        self.scale.addItem("50%", 0.5)
        self.demo_resolution = QComboBox()
        self.demo_resolution.addItem("1280x720", "1280x720")
        self.demo_resolution.addItem("1920x1080", "1920x1080")

        self.start_btn = QPushButton("START")
        self.stop_btn = QPushButton("STOP")
        self.pause_btn = QPushButton("PAUSE")
        self.record_btn = QPushButton("RECORD 15s DEMO")
        self.record_btn.setObjectName("recordButton")
        self.cancel_btn = QPushButton("CANCEL")
        self.cancel_btn.setEnabled(False)
        self.shot_btn = QPushButton("SAVE SCREENSHOT")
        self.rgb_btn = QPushButton("SAVE RGB FRAME")
        self.depth_btn = QPushButton("SAVE DEPTH MAP")
        self.raw_btn = QPushButton("SAVE RAW DEPTH")
        self.model_btn = QPushButton("MODEL MANAGER")
        self.settings_btn = QPushButton("SETTINGS")
        self.browse_btn = QPushButton("OPEN FILE")
        self.rescan_btn = QPushButton("RESCAN")

        self.play_btn = QPushButton("PLAY")
        self.restart_btn = QPushButton("RESTART")
        self.timeline = QSlider(Qt.Horizontal)
        self.timeline.setRange(0, 1000)
        self._slider_hold = False

        row1 = QHBoxLayout()
        for label, widget in (
            ("Source", self.source),
            ("Camera", self.camera),
            ("Resolution", self.resolution),
            ("Device", self.device),
            ("Precision", self.precision),
            ("View", self.view),
            ("Colormap", self.colormap),
            ("Scale", self.scale),
        ):
            row1.addWidget(QLabel(label))
            row1.addWidget(widget)
        row1.addWidget(self.rescan_btn)
        row1.addWidget(self.browse_btn)
        row1.addStretch(1)

        row2 = QHBoxLayout()
        for button in (
            self.start_btn,
            self.stop_btn,
            self.pause_btn,
            self.record_btn,
            self.cancel_btn,
            self.shot_btn,
            self.rgb_btn,
            self.depth_btn,
            self.raw_btn,
            self.model_btn,
            self.settings_btn,
        ):
            row2.addWidget(button)
        row2.addWidget(QLabel("Demo"))
        row2.addWidget(self.demo_resolution)
        row2.addStretch(1)

        row3 = QHBoxLayout()
        row3.addWidget(self.play_btn)
        row3.addWidget(self.restart_btn)
        row3.addWidget(QLabel("Timeline"))
        row3.addWidget(self.timeline, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 4)
        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addLayout(row3)

        self.source.currentIndexChanged.connect(lambda: self.source_changed.emit(self.source.currentData()))
        self.camera.currentIndexChanged.connect(self._emit_camera)
        self.resolution.currentIndexChanged.connect(
            lambda: self.resolution_changed.emit(self.resolution.currentData())
        )
        self.device.currentIndexChanged.connect(lambda: self.device_changed.emit(self.device.currentData()))
        self.precision.currentIndexChanged.connect(
            lambda: self.precision_changed.emit(self.precision.currentData())
        )
        self.view.currentIndexChanged.connect(lambda: self.view_changed.emit(self.view.currentData()))
        self.colormap.currentIndexChanged.connect(
            lambda: self.colormap_changed.emit(self.colormap.currentData())
        )
        self.scale.currentIndexChanged.connect(lambda: self.scale_changed.emit(float(self.scale.currentData())))
        self.demo_resolution.currentIndexChanged.connect(
            lambda: self.demo_resolution_changed.emit(self.demo_resolution.currentData())
        )
        self.start_btn.clicked.connect(self.start_clicked)
        self.stop_btn.clicked.connect(self.stop_clicked)
        self.pause_btn.clicked.connect(self.pause_clicked)
        self.record_btn.clicked.connect(self.record_clicked)
        self.cancel_btn.clicked.connect(self.cancel_clicked)
        self.shot_btn.clicked.connect(self.screenshot_clicked)
        self.rgb_btn.clicked.connect(self.save_rgb_clicked)
        self.depth_btn.clicked.connect(self.save_depth_clicked)
        self.raw_btn.clicked.connect(self.save_raw_clicked)
        self.model_btn.clicked.connect(self.model_clicked)
        self.settings_btn.clicked.connect(self.settings_clicked)
        self.browse_btn.clicked.connect(self.browse_clicked)
        self.rescan_btn.clicked.connect(self.rescan_clicked)
        self.play_btn.clicked.connect(self.play_clicked)
        self.restart_btn.clicked.connect(self.restart_clicked)
        self.timeline.sliderPressed.connect(lambda: setattr(self, "_slider_hold", True))
        self.timeline.sliderReleased.connect(self._emit_seek)

        self.set_transport_enabled(False)

    def _emit_camera(self) -> None:
        data = self.camera.currentData()
        if data is not None:
            self.camera_changed.emit(int(data))

    def _emit_seek(self) -> None:
        self._slider_hold = False
        self.seek_changed.emit(self.timeline.value() / 1000.0)

    def set_cameras(self, indices: list[int]) -> None:
        current = self.camera.currentData()
        self.camera.blockSignals(True)
        self.camera.clear()
        if not indices:
            self.camera.addItem("No camera", -1)
        for index in indices:
            self.camera.addItem(f"Camera {index}", index)
        self._restore_combo(self.camera, current if current is not None else 0)
        self.camera.blockSignals(False)

    def set_devices(self, devices: list[dict], current: str) -> None:
        self.device.blockSignals(True)
        self.device.clear()
        self.device.addItem("AUTO", "auto")
        for item in devices:
            label = f"CUDA:{item['index']} {item['name']}"
            self.device.addItem(label, f"cuda:{item['index']}")
        self.device.addItem("CPU", "cpu")
        self._restore_combo(self.device, current)
        self.device.blockSignals(False)

    def set_recording_text(self, text: str) -> None:
        self.record_btn.setText(text)

    def set_cancel_enabled(self, enabled: bool) -> None:
        self.cancel_btn.setEnabled(enabled)

    def set_transport_enabled(self, enabled: bool) -> None:
        for widget in (self.play_btn, self.restart_btn, self.timeline):
            widget.setEnabled(enabled)

    def set_video_position(self, fraction: float) -> None:
        if self._slider_hold:
            return
        self.timeline.blockSignals(True)
        self.timeline.setValue(int(max(0.0, min(1.0, fraction)) * 1000))
        self.timeline.blockSignals(False)

    def select_value(self, combo: QComboBox, value) -> None:
        combo.blockSignals(True)
        self._restore_combo(combo, value)
        combo.blockSignals(False)

    @staticmethod
    def _restore_combo(combo: QComboBox, value) -> None:
        for index in range(combo.count()):
            if combo.itemData(index) == value:
                combo.setCurrentIndex(index)
                return
        if combo.count():
            combo.setCurrentIndex(0)
