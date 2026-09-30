"""Recording badges drawn on the visualization, not on the window chrome."""

from __future__ import annotations

import cv2
import numpy as np


def draw_countdown(image: np.ndarray, number: int) -> None:
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], image.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, image, 0.65, 0, image)
    label = str(int(number))
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(2.0, image.shape[0] / 280.0)
    thickness = max(4, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    x = (image.shape[1] - tw) // 2
    y = (image.shape[0] + th) // 2
    cv2.putText(image, label, (x, y), font, scale, (245, 245, 245), thickness, cv2.LINE_AA)


def draw_recording(image: np.ndarray, remaining: float, duration: float) -> None:
    h, w = image.shape[:2]
    elapsed = max(0.0, duration - remaining)
    fraction = 0.0 if duration <= 0 else np.clip(elapsed / duration, 0.0, 1.0)
    bar_h = max(4, h // 140)
    cv2.rectangle(image, (0, 0), (w, bar_h), (40, 40, 40), -1)
    cv2.rectangle(image, (0, 0), (int(w * fraction), bar_h), (40, 40, 220), -1)
    radius = max(7, h // 70)
    center = (18 + radius, bar_h + 16 + radius)
    cv2.circle(image, center, radius, (30, 30, 230), -1, cv2.LINE_AA)
    cv2.putText(
        image,
        "REC",
        (center[0] + radius + 8, center[1] + 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.6, h / 900.0),
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
