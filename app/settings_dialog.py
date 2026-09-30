"""Recording and measurement defaults."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QSpinBox,
    QVBoxLayout,
)


class SettingsDialog(QDialog):
    def __init__(self, settings: dict, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(420, 220)
        self.demo_resolution = QComboBox()
        self.demo_resolution.addItem("1280x720", "1280x720")
        self.demo_resolution.addItem("1920x1080", "1920x1080")
        self._select(self.demo_resolution, settings.get("demo_resolution", "1920x1080"))
        self.demo_fps = QSpinBox()
        self.demo_fps.setRange(10, 60)
        self.demo_fps.setValue(int(settings.get("demo_fps", 30)))
        self.max_points = QSpinBox()
        self.max_points.setRange(1, 10)
        self.max_points.setValue(int(settings.get("max_points", 10)))
        form = QFormLayout()
        form.addRow("Demo recording", self.demo_resolution)
        form.addRow("Demo FPS", self.demo_fps)
        form.addRow("Max measurement points", self.max_points)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def values(self) -> dict:
        return {
            "demo_resolution": self.demo_resolution.currentData(),
            "demo_fps": int(self.demo_fps.value()),
            "max_points": int(self.max_points.value()),
        }

    @staticmethod
    def _select(combo: QComboBox, value) -> None:
        for index in range(combo.count()):
            if combo.itemData(index) == value:
                combo.setCurrentIndex(index)
                return
