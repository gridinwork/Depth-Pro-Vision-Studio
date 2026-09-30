"""Still-image source."""

from __future__ import annotations

from pathlib import Path

from utils.images import fit_inside, imread


class ImageSource:
    def __init__(self) -> None:
        self.frame = None
        self.path = ""

    def open(self, path: str, max_w: int, max_h: int):
        image = imread(path)
        if image is None:
            raise RuntimeError(f"Could not read image: {path}")
        self.frame = fit_inside(image, max_w, max_h)
        self.path = str(Path(path))
        return self.frame

    def close(self) -> None:
        self.frame = None
        self.path = ""
