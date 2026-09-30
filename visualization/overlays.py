"""HUD, analytical tints, countdown and recording badges burned into the viewport."""

from __future__ import annotations

import cv2
import numpy as np


def apply_bands(color: np.ndarray, depth: np.ndarray, near_m: float, mid_m: float) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    tint = np.zeros_like(color)
    tint[valid & (depth <= near_m)] = (90, 200, 110)
    tint[valid & (depth > near_m) & (depth <= mid_m)] = (50, 190, 255)
    tint[valid & (depth > mid_m)] = (230, 140, 70)
    blended = cv2.addWeighted(color, 0.62, tint, 0.38, 0)
    out = color.copy()
    out[valid] = blended[valid]
    return out


def apply_threshold(color: np.ndarray, depth: np.ndarray, threshold_m: float) -> np.ndarray:
    out = color.copy()
    far = np.isfinite(depth) & (depth > threshold_m)
    out[far] = (out[far].astype(np.float32) * 0.22).astype(np.uint8)
    return out


def apply_obstacle(color: np.ndarray, depth: np.ndarray, obstacle_m: float) -> np.ndarray:
    out = color.copy()
    close = np.isfinite(depth) & (depth > 0) & (depth <= obstacle_m)
    if not np.any(close):
        return out
    red = np.zeros_like(color)
    red[:] = (40, 40, 220)
    blended = cv2.addWeighted(color, 0.45, red, 0.55, 0)
    out[close] = blended[close]
    return out


def draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = (240, 240, 240),
    bg: bool = True,
    thickness: int = 1,
) -> tuple[int, int]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    if bg:
        cv2.rectangle(
            image,
            (x - 4, y - th - 6),
            (x + tw + 6, y + baseline + 4),
            (16, 18, 22),
            -1,
        )
    cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    return tw, th


def draw_banner(image: np.ndarray, rect: tuple[int, int, int, int], text: str, scale: float, text_x: int | None = None) -> None:
    x, y, w, h = rect
    cv2.rectangle(image, (x, y), (x + w, y + h), (22, 24, 30), -1)
    baseline = y + h - max(8, int(h * 0.28))
    cv2.putText(
        image,
        text,
        (x + 12 if text_x is None else text_x, baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
