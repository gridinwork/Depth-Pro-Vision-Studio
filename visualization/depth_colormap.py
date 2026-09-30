"""Display-only depth colormaps. Metric values are not modified."""

from __future__ import annotations

import cv2
import numpy as np


COLORMAPS = {
    "inferno": cv2.COLORMAP_INFERNO,
    "turbo": cv2.COLORMAP_TURBO,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "magma": cv2.COLORMAP_MAGMA,
}


def colorize(depth: np.ndarray | None, name: str = "turbo") -> tuple[np.ndarray | None, float | None, float | None]:
    """Colorize inverse depth for display. Near distances map to the hot end of the scale."""
    if depth is None:
        return None, None, None
    valid = np.isfinite(depth) & (depth > 1e-4)
    if not np.any(valid):
        blank = np.zeros((*depth.shape[:2], 3), dtype=np.uint8)
        return blank, None, None

    meters = depth[valid]
    near = float(np.percentile(meters, 2))
    far = float(np.percentile(meters, 98))
    if far <= near:
        far = near + 1e-3

    inv = np.zeros(depth.shape[:2], dtype=np.float32)
    inv[valid] = 1.0 / np.clip(depth[valid], 1e-4, None)
    inv_near = 1.0 / far
    inv_far = 1.0 / near
    span = max(inv_far - inv_near, 1e-6)
    norm = np.clip((inv - inv_near) / span, 0.0, 1.0)
    gray = (norm * 255.0).astype(np.uint8)
    key = (name or "turbo").lower()
    if key == "grayscale":
        colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        colored = cv2.applyColorMap(gray, COLORMAPS.get(key, cv2.COLORMAP_TURBO))
    colored[~valid] = 0
    return colored, near, far


def legend_colors(name: str, height: int) -> np.ndarray:
    height = max(8, int(height))
    ramp = np.linspace(255, 0, height, dtype=np.uint8).reshape(height, 1)
    if (name or "").lower() == "grayscale":
        bar = cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR)
    else:
        bar = cv2.applyColorMap(ramp, COLORMAPS.get(name.lower(), cv2.COLORMAP_TURBO))
    return bar
