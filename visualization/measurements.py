"""Measurement markers drawn in canvas coordinates."""

from __future__ import annotations

import cv2
import numpy as np


def _map_point(
    sx: float,
    sy: float,
    content: tuple[int, int, int, int],
    source_size: tuple[int, int],
) -> tuple[int, int]:
    x, y, w, h = content
    sw, sh = source_size
    px = int(x + (sx / max(sw, 1)) * w)
    py = int(y + (sy / max(sh, 1)) * h)
    return px, py


def draw_crosshair(image: np.ndarray, px: int, py: int, label: str, scale: float) -> None:
    color = (240, 240, 240)
    arm = max(6, int(10 * scale))
    cv2.line(image, (px - arm, py), (px + arm, py), color, 1, cv2.LINE_AA)
    cv2.line(image, (px, py - arm), (px, py + arm), color, 1, cv2.LINE_AA)
    cv2.circle(image, (px, py), max(3, int(4 * scale)), (40, 220, 255), 1, cv2.LINE_AA)
    if label:
        cv2.putText(
            image,
            label,
            (px + arm + 4, py - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.45, 0.5 * scale),
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_center(image: np.ndarray, content: tuple[int, int, int, int], depth_m: float | None, scale: float) -> None:
    x, y, w, h = content
    px, py = x + w // 2, y + h // 2
    arm = max(8, int(14 * scale))
    color = (80, 220, 255)
    cv2.line(image, (px - arm, py), (px - 4, py), color, 1, cv2.LINE_AA)
    cv2.line(image, (px + 4, py), (px + arm, py), color, 1, cv2.LINE_AA)
    cv2.line(image, (px, py - arm), (px, py - 4), color, 1, cv2.LINE_AA)
    cv2.line(image, (px, py + 4), (px, py + arm), color, 1, cv2.LINE_AA)
    label = "CENTER" if depth_m is None else f"CENTER {depth_m:.2f} m"
    cv2.putText(
        image,
        label,
        (px - int(70 * scale), py + arm + int(16 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.45, 0.55 * scale),
        color,
        1,
        cv2.LINE_AA,
    )


def draw_points(
    image: np.ndarray,
    points: list[dict],
    content: tuple[int, int, int, int],
    source_size: tuple[int, int],
    scale: float,
) -> None:
    for point in points:
        px, py = _map_point(point["x"], point["y"], content, source_size)
        cv2.circle(image, (px, py), max(4, int(6 * scale)), (0, 200, 255), -1, cv2.LINE_AA)
        cv2.circle(image, (px, py), max(6, int(9 * scale)), (255, 255, 255), 1, cv2.LINE_AA)
        depth = point.get("depth")
        label = point["name"] if depth is None else f"{point['name']} {depth:.2f} m"
        cv2.putText(
            image,
            label,
            (px + 8, py - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.4, 0.48 * scale),
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_roi(
    image: np.ndarray,
    roi: tuple[float, float, float, float],
    stats: dict | None,
    content: tuple[int, int, int, int],
    source_size: tuple[int, int],
    scale: float,
) -> None:
    x0, y0 = _map_point(roi[0], roi[1], content, source_size)
    x1, y1 = _map_point(roi[0] + roi[2], roi[1] + roi[3], content, source_size)
    cv2.rectangle(image, (x0, y0), (x1, y1), (80, 255, 180), 2, cv2.LINE_AA)
    if stats is None:
        label = "ROI"
    else:
        label = f"ROI median {stats['median']:.2f} m"
    cv2.putText(
        image,
        label,
        (min(x0, x1) + 4, max(16, min(y0, y1) - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.45, 0.5 * scale),
        (80, 255, 180),
        1,
        cv2.LINE_AA,
    )
