"""Saved-demo summary."""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from utils.gpu_info import format_bytes


class DemoSavedDialog(QDialog):
    def __init__(self, info: dict, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("DEMO SAVED")
        self.resize(560, 240)
        self._path = Path(info["path"])
        lines = [
            "DEMO SAVED",
            f"Duration: {info['duration']:.2f} s (wall {info['wall_duration']:.2f} s)",
            f"Resolution: {info['resolution']}",
            f"FPS: {info['fps']}",
            f"File size: {format_bytes(int(info['size']))}",
            f"Codec: {info.get('codec', '')}",
            f"Path: {info['path']}",
        ]
        label = QLabel("\n".join(lines))
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        open_btn = QPushButton("OPEN FOLDER")
        close_btn = QPushButton("CLOSE")
        open_btn.clicked.connect(self._open)
        close_btn.clicked.connect(self.accept)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(open_btn)
        row.addWidget(close_btn)
        layout = QVBoxLayout(self)
        layout.addWidget(label)
        layout.addLayout(row)

    def _open(self) -> None:
        os.startfile(self._path.parent)
