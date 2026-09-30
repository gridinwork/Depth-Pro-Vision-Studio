"""Metric statistics computed only from finite, positive depth values."""

from __future__ import annotations

import numpy as np


def sample_depth(
    depth: np.ndarray | None,
    x: float,
    y: float,
    src_w: int,
    src_h: int,
) -> float | None:
    if depth is None or src_w <= 0 or src_h <= 0:
        return None
    height, width = depth.shape[:2]
    ix = int(np.clip(x / src_w * width, 0, width - 1))
    iy = int(np.clip(y / src_h * height, 0, height - 1))
    value = float(depth[iy, ix])
    if not np.isfinite(value) or value <= 0:
        return None
    return value


def valid_values(depth: np.ndarray | None) -> np.ndarray:
    if depth is None:
        return np.empty((0,), dtype=np.float32)
    mask = np.isfinite(depth) & (depth > 0)
    return depth[mask]


def analyze(
    depth: np.ndarray | None,
    src_w: int,
    src_h: int,
    cursor_xy: tuple[float, float] | None,
    points: list[dict],
    roi: tuple[float, float, float, float] | None,
    near_m: float,
    mid_m: float,
    obstacle_m: float,
    obstacle_fraction: float = 0.05,
) -> dict:
    values = valid_values(depth)
    nearest = float(values.min()) if values.size else None
    farthest = float(values.max()) if values.size else None
    center = None
    if src_w > 0 and src_h > 0:
        center = sample_depth(depth, (src_w - 1) / 2.0, (src_h - 1) / 2.0, src_w, src_h)

    cursor_depth = None
    if cursor_xy is not None:
        cursor_depth = sample_depth(depth, cursor_xy[0], cursor_xy[1], src_w, src_h)

    annotated = []
    for point in points:
        distance = sample_depth(depth, point["x"], point["y"], src_w, src_h)
        annotated.append({**point, "depth": distance})

    roi_stats = None
    if roi is not None and depth is not None and src_w > 0 and src_h > 0:
        roi_stats = _roi_stats(depth, roi, src_w, src_h)

    bands = {"near": 0, "mid": 0, "far": 0}
    obstacle_hit = False
    obstacle_ratio = 0.0
    if values.size:
        bands["near"] = int(np.count_nonzero(values <= near_m))
        bands["mid"] = int(np.count_nonzero((values > near_m) & (values <= mid_m)))
        bands["far"] = int(np.count_nonzero(values > mid_m))
        close = int(np.count_nonzero(values <= obstacle_m))
        obstacle_ratio = close / float(values.size)
        obstacle_hit = obstacle_ratio >= obstacle_fraction and close >= 500

    return {
        "nearest": nearest,
        "farthest": farthest,
        "center": center,
        "cursor_depth": cursor_depth,
        "points": annotated,
        "roi_stats": roi_stats,
        "bands": bands,
        "valid_pixels": int(values.size),
        "obstacle_detected": obstacle_hit,
        "obstacle_fraction": obstacle_ratio,
    }


def _roi_stats(
    depth: np.ndarray,
    roi: tuple[float, float, float, float],
    src_w: int,
    src_h: int,
) -> dict | None:
    height, width = depth.shape[:2]
    x, y, rw, rh = roi
    x0 = int(np.clip(min(x, x + rw) / src_w * width, 0, width - 1))
    x1 = int(np.clip(max(x, x + rw) / src_w * width, 0, width))
    y0 = int(np.clip(min(y, y + rh) / src_h * height, 0, height - 1))
    y1 = int(np.clip(max(y, y + rh) / src_h * height, 0, height))
    if x1 <= x0 or y1 <= y0:
        return None
    patch = depth[y0:y1, x0:x1]
    values = patch[np.isfinite(patch) & (patch > 0)]
    if values.size == 0:
        return None
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "pixels": int(values.size),
    }
