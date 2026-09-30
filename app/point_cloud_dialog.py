"""Optional OpenGL point cloud. The main app keeps running if OpenGL is unavailable."""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from depth.point_cloud import build_point_cloud


class PointCloudDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("3D Point Cloud")
        self.resize(900, 640)
        import pyqtgraph.opengl as gl

        self._gl = gl
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor((16, 18, 22))
        self.scatter = gl.GLScatterPlotItem(size=2, pxMode=True)
        self.view.addItem(self.scatter)
        self.status = QLabel("Waiting for a depth map with a valid focal length.")
        self.status.setWordWrap(True)
        orbit = QPushButton("Orbit")
        zoom_in = QPushButton("Zoom +")
        zoom_out = QPushButton("Zoom -")
        rotate = QPushButton("Rotate")
        reset = QPushButton("Reset View")
        hint = QLabel("Left drag orbits. Mouse wheel zooms.")
        row = QHBoxLayout()
        for button in (orbit, zoom_in, zoom_out, rotate, reset):
            row.addWidget(button)
        row.addWidget(hint)
        row.addStretch(1)
        layout = QVBoxLayout(self)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.status)
        layout.addLayout(row)
        orbit.clicked.connect(lambda: self._nudge(azimuth=25))
        rotate.clicked.connect(lambda: self._nudge(elevation=10))
        zoom_in.clicked.connect(lambda: self._nudge(distance_scale=0.8))
        zoom_out.clicked.connect(lambda: self._nudge(distance_scale=1.25))
        reset.clicked.connect(self.reset_view)
        self._framed = False
        self.reset_view()

    def reset_view(self) -> None:
        self.view.setCameraPosition(distance=4.0, elevation=18, azimuth=40)

    def _nudge(self, azimuth: float = 0, elevation: float = 0, distance_scale: float = 1.0) -> None:
        opts = self.view.opts
        self.view.setCameraPosition(
            distance=max(0.2, float(opts.get("distance", 4.0)) * distance_scale),
            elevation=float(opts.get("elevation", 18)) + elevation,
            azimuth=float(opts.get("azimuth", 40)) + azimuth,
        )

    def update_cloud(self, depth, focal_px) -> None:
        built = build_point_cloud(depth, focal_px)
        if built is None:
            self.scatter.setData(pos=np.zeros((0, 3), dtype=np.float32))
            self.status.setText(
                "Focal length is not available, so XYZ is not estimated. "
                "A point cloud is shown only from the Depth Pro focal-length estimate."
            )
            return
        points, z_values = built
        colors = _colors(z_values)
        self.scatter.setData(pos=points, color=colors, size=2)
        if not self._framed:
            distance = float(np.median(z_values)) * 1.8
            self.view.setCameraPosition(distance=max(0.5, distance))
            self._framed = True
        self.status.setText(f"Points: {len(points)}    focal {focal_px:.0f} px")


def _colors(z_values: np.ndarray) -> np.ndarray:
    import cv2

    span = max(float(z_values.max() - z_values.min()), 1e-6)
    norm = np.clip((z_values - float(z_values.min())) / span, 0, 1)
    gray = (norm * 255).astype(np.uint8).reshape(-1, 1)
    bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO).reshape(-1, 3)
    rgba = np.ones((len(z_values), 4), dtype=np.float32)
    rgba[:, 0] = bgr[:, 2] / 255.0
    rgba[:, 1] = bgr[:, 1] / 255.0
    rgba[:, 2] = bgr[:, 0] / 255.0
    return rgba
