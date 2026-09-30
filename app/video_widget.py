"""Viewport widget. Mouse coordinates are mapped back to the source frame."""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from visualization.renderer import map_to_source


def _bgr_to_qimage(bgr: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    height, width, _ = rgb.shape
    image = QImage(rgb.data, width, height, width * 3, QImage.Format_RGB888)
    return image.copy()


class VideoWidget(QWidget):
    cursor_at = Signal(float, float)
    cursor_cleared = Signal()
    point_clicked = Signal(float, float)
    roi_finished = Signal(float, float, float, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(640, 360)
        self.roi_mode = False
        self._image: QImage | None = None
        self._layout = None
        self._press_canvas = None
        self._rubber = None

    def set_frame(self, bgr: np.ndarray, layout) -> None:
        self._image = _bgr_to_qimage(bgr)
        self._layout = layout
        self.update()

    def leaveEvent(self, _event) -> None:
        self.cursor_cleared.emit()
        self._rubber = None
        self.update()

    def mouseMoveEvent(self, event) -> None:
        canvas = self._to_canvas(event.position().toPoint())
        if canvas is None:
            self.cursor_cleared.emit()
            return
        if self._press_canvas is not None and self.roi_mode:
            self._rubber = (self._press_canvas, canvas)
            self.update()
        source = self._to_source(canvas)
        if source is not None:
            self.cursor_at.emit(source[0], source[1])

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        self._press_canvas = self._to_canvas(event.position().toPoint())

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.LeftButton or self._press_canvas is None:
            return
        end = self._to_canvas(event.position().toPoint())
        start = self._press_canvas
        self._press_canvas = None
        self._rubber = None
        self.update()
        if end is None:
            return
        if self.roi_mode and (abs(end[0] - start[0]) > 6 or abs(end[1] - start[1]) > 6):
            src_a = self._to_source(start)
            src_b = self._to_source(end)
            if src_a is not None and src_b is not None:
                x = min(src_a[0], src_b[0])
                y = min(src_a[1], src_b[1])
                self.roi_finished.emit(x, y, abs(src_b[0] - src_a[0]), abs(src_b[1] - src_a[1]))
            return
        source = self._to_source(end)
        if source is not None:
            self.point_clicked.emit(source[0], source[1])

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        if self._image is None or self._image.isNull():
            painter.setPen(Qt.gray)
            painter.drawText(self.rect(), Qt.AlignCenter, "Depth Pro Vision Studio")
            painter.end()
            return
        target = self._display_rect()
        painter.drawImage(target, self._image)
        if self._rubber is not None:
            a, b = self._rubber
            rect = self._canvas_rect_to_widget(a, b)
            painter.setPen(QPen(Qt.green, 2))
            painter.drawRect(rect)
        painter.end()

    def _display_rect(self) -> QRect:
        if self._image is None:
            return QRect()
        iw, ih = self._image.width(), self._image.height()
        if iw <= 0 or ih <= 0:
            return QRect()
        scale = min(self.width() / iw, self.height() / ih)
        dw, dh = int(iw * scale), int(ih * scale)
        return QRect((self.width() - dw) // 2, (self.height() - dh) // 2, max(1, dw), max(1, dh))

    def _to_canvas(self, pos: QPoint):
        rect = self._display_rect()
        if self._image is None or not rect.contains(pos):
            return None
        ix = (pos.x() - rect.x()) / rect.width() * self._image.width()
        iy = (pos.y() - rect.y()) / rect.height() * self._image.height()
        return ix, iy

    def _to_source(self, canvas_xy):
        if self._layout is None or canvas_xy is None:
            return None
        return map_to_source(self._layout, canvas_xy[0], canvas_xy[1])

    def _canvas_rect_to_widget(self, a, b) -> QRect:
        rect = self._display_rect()
        if self._image is None:
            return QRect()

        def conv(pt):
            x = rect.x() + pt[0] / self._image.width() * rect.width()
            y = rect.y() + pt[1] / self._image.height() * rect.height()
            return int(x), int(y)

        ax, ay = conv(a)
        bx, by = conv(b)
        return QRect(min(ax, bx), min(ay, by), abs(bx - ax), abs(by - ay))
