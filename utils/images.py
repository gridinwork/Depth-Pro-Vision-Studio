"""Unicode-safe image IO for Windows paths."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread(path: str | Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    encoded.tofile(str(path))


def fit_cover(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Center-crop to the target aspect, then resize. Does not stretch."""
    src_h, src_w = frame.shape[:2]
    if src_w <= 0 or src_h <= 0 or width <= 0 or height <= 0:
        return frame
    target_aspect = width / height
    src_aspect = src_w / src_h
    if src_aspect > target_aspect:
        crop_w = max(1, int(round(src_h * target_aspect)))
        x0 = max(0, (src_w - crop_w) // 2)
        cropped = frame[:, x0 : x0 + crop_w]
    else:
        crop_h = max(1, int(round(src_w / target_aspect)))
        y0 = max(0, (src_h - crop_h) // 2)
        cropped = frame[y0 : y0 + crop_h, :]
    interpolation = cv2.INTER_AREA if cropped.shape[1] > width else cv2.INTER_LINEAR
    return cv2.resize(cropped, (width, height), interpolation=interpolation)


def fit_inside(frame: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    src_h, src_w = frame.shape[:2]
    scale = min(max_w / src_w, max_h / src_h, 1.0)
    if scale >= 0.999:
        return frame
    width = max(2, int(src_w * scale))
    height = max(2, int(src_h * scale))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
