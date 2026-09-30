"""MP4 writer. Prefers H.264 through the bundled ffmpeg, then OpenCV."""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from utils.logger import get_logger


def fit_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    width -= width % 2
    height -= height % 2
    width = max(2, width)
    height = max(2, height)
    src_h, src_w = frame.shape[:2]
    if src_w == width and src_h == height:
        return frame
    scale = min(width / src_w, height / src_h)
    nw = max(2, int(src_w * scale) // 2 * 2)
    nh = max(2, int(src_h * scale) // 2 * 2)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - nw) // 2
    y = (height - nh) // 2
    canvas[y : y + nh, x : x + nw] = resized
    return canvas


class VideoEncoder:
    def __init__(self) -> None:
        self.path: Path | None = None
        self.size = (0, 0)
        self.codec = ""
        self._proc: subprocess.Popen | None = None
        self._writer: cv2.VideoWriter | None = None
        self._stderr = None
        self._logger = get_logger("encoder")

    def open(self, path: str | Path, width: int, height: int, fps: float) -> str:
        self.close()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        width -= int(width) % 2
        height -= int(height) % 2
        self.size = (width, height)
        fps = float(fps)
        if self._open_ffmpeg(self.path, width, height, fps):
            self.codec = "h264"
            return self.codec
        self._open_opencv(self.path, width, height, fps)
        self.codec = "mp4v"
        return self.codec

    def write(self, frame: np.ndarray) -> None:
        fitted = fit_frame(frame, self.size[0], self.size[1])
        if self._proc is not None and self._proc.stdin is not None:
            try:
                self._proc.stdin.write(fitted.tobytes())
            except BrokenPipeError as exc:
                raise RuntimeError(self._ffmpeg_error()) from exc
            return
        if self._writer is not None:
            self._writer.write(fitted)
            return
        raise RuntimeError("Video encoder is not open.")

    def close(self) -> None:
        if self._proc is not None:
            proc = self._proc
            self._proc = None
            if proc.stdin is not None:
                try:
                    proc.stdin.close()
                except OSError:
                    pass
            try:
                code = proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                code = proc.wait(timeout=5)
            if self._stderr is not None:
                self._stderr.close()
                self._stderr = None
            if code != 0:
                raise RuntimeError(f"ffmpeg exited with code {code}")
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def _open_ffmpeg(self, path: Path, width: int, height: int, fps: float) -> bool:
        try:
            import imageio_ffmpeg
        except Exception:
            self._logger.info("imageio-ffmpeg is not installed, using OpenCV writer")
            return False
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        log_path = path.with_suffix(".ffmpeg.log")
        self._stderr = open(log_path, "wb")
        command = [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:.4f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "ultrafast",
            "-crf",
            "20",
            "-movflags",
            "+faststart",
            str(path),
        ]
        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr,
        )
        return True

    def _open_opencv(self, path: Path, width: int, height: int, fps: float) -> None:
        writer = None
        for fourcc_name in ("avc1", "H264", "mp4v"):
            fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
            candidate = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
            if candidate.isOpened():
                writer = candidate
                self._logger.info("OpenCV video codec %s", fourcc_name)
                break
            candidate.release()
        if writer is None:
            raise RuntimeError("No MP4 codec is available in OpenCV or ffmpeg.")
        self._writer = writer

    def _ffmpeg_error(self) -> str:
        return "ffmpeg failed while writing the demo video. See the .ffmpeg.log next to the file."
