"""Build a metric point cloud from a depth map and a focal length in pixels.

XYZ is returned only when a positive focal length is available.
"""

from __future__ import annotations

import numpy as np


def build_point_cloud(
    depth: np.ndarray | None,
    focal_px: float | None,
    max_points: int = 80000,
    z_min: float = 0.05,
    z_max: float = 30.0,
):
    if depth is None or focal_px is None or not np.isfinite(focal_px) or focal_px <= 1.0:
        return None
    height, width = depth.shape[:2]
    if height < 2 or width < 2:
        return None
    stride = max(1, int(np.ceil(np.sqrt((height * width) / max_points))))
    z = depth[::stride, ::stride]
    vs, us = np.mgrid[0:height:stride, 0:width:stride]
    valid = np.isfinite(z) & (z >= z_min) & (z <= z_max)
    if int(valid.sum()) < 10:
        return None
    z = z[valid].astype(np.float32)
    us = us[valid].astype(np.float32)
    vs = vs[valid].astype(np.float32)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    x = (us - cx) * z / float(focal_px)
    y = (vs - cy) * z / float(focal_px)
    points = np.stack([x, -y, -z], axis=1).astype(np.float32)
    return points, z
