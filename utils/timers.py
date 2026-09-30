"""Small timing helpers. Recording duration uses time.perf_counter."""

from __future__ import annotations

import time


class FpsMeter:
    def __init__(self, smoothing: float = 0.85) -> None:
        self.smoothing = smoothing
        self.fps = 0.0
        self._last: float | None = None

    def reset(self) -> None:
        self.fps = 0.0
        self._last = None

    def tick(self, now: float | None = None) -> float:
        now = time.perf_counter() if now is None else now
        if self._last is not None:
            dt = now - self._last
            if dt > 1e-6:
                instant = 1.0 / dt
                if self.fps <= 0:
                    self.fps = instant
                else:
                    self.fps = self.smoothing * self.fps + (1.0 - self.smoothing) * instant
        self._last = now
        return self.fps


class Stopwatch:
    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def restart(self) -> None:
        self._t0 = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self._t0

    def ms(self) -> float:
        return self.elapsed() * 1000.0
