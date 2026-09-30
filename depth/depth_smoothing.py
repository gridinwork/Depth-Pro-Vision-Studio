"""Temporal smoothing applied after inference. The model output is not modified in-place."""

from __future__ import annotations

import numpy as np


class DepthSmoother:
    def __init__(self) -> None:
        self.previous: np.ndarray | None = None

    def reset(self) -> None:
        self.previous = None

    def apply(self, depth: np.ndarray, enabled: bool, alpha: float) -> np.ndarray:
        current = np.array(depth, dtype=np.float32, copy=True)
        if not enabled:
            self.previous = current
            return current
        alpha = float(np.clip(alpha, 0.0, 1.0))
        if self.previous is None or self.previous.shape != current.shape:
            self.previous = current
            return current
        if alpha >= 0.999:
            self.previous = current
            return current

        mixed = current.copy()
        both = np.isfinite(self.previous) & np.isfinite(current)
        mixed[both] = alpha * current[both] + (1.0 - alpha) * self.previous[both]
        only_old = ~np.isfinite(current) & np.isfinite(self.previous)
        mixed[only_old] = self.previous[only_old]
        self.previous = mixed
        return mixed
