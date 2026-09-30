"""Vertical metric legend. Labels are meters; colors match the display colormap."""

from __future__ import annotations

import cv2
import numpy as np

from visualization.depth_colormap import legend_colors


def draw_legend(
    canvas: np.ndarray,
    rect: tuple[int, int, int, int],
    colormap: str,
    near_m: float | None,
    far_m: float | None,
    scale: float,
) -> None:
    x, y, w, h = rect
    if w < 8 or h < 8:
        return
    bar_w = max(12, int(18 * scale))
    bar = legend_colors(colormap, h)
    bar = cv2.resize(bar, (bar_w, h), interpolation=cv2.INTER_NEAREST)
    canvas[y : y + h, x : x + bar_w] = bar
    cv2.rectangle(canvas, (x, y), (x + bar_w - 1, y + h - 1), (220, 220, 220), 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_scale = max(0.4, 0.45 * scale)
    near_label = "near" if near_m is None else f"{near_m:.2f} m"
    far_label = "far" if far_m is None else f"{far_m:.2f} m"
    cv2.putText(canvas, near_label, (x + bar_w + 6, y + int(16 * scale)), font, text_scale, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(canvas, far_label, (x + bar_w + 6, y + h - 6), font, text_scale, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(canvas, "m", (x + bar_w + 6, y + h // 2), font, text_scale, (180, 180, 180), 1, cv2.LINE_AA)
