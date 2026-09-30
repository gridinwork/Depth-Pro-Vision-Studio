"""Record a 15 second demo through DemoRecorder and check the MP4."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capture.frame_hub import FrameHub
from recording.demo_recorder import DemoRecorder
from visualization.renderer import RenderState, render


def main() -> int:
    app = QApplication([])
    hub = FrameHub()
    recorder = DemoRecorder(hub)
    done = {}

    def finish(info):
        done["info"] = info

    def fail(message):
        done["error"] = message

    recorder.finished_ok.connect(finish)
    recorder.failed.connect(fail)
    recorder.start()
    path = recorder.request_start(1920, 1080, 30, 15.0)
    started = time.perf_counter()
    while "info" not in done and "error" not in done:
        frame = _frame(time.perf_counter() - started)
        hub.set_composite(frame, None)
        app.processEvents()
        time.sleep(0.01)
        if time.perf_counter() - started > 25:
            done["error"] = "timed out"
            break
    recorder.shutdown()
    recorder.wait(5000)
    if "error" in done:
        print(done["error"])
        return 1
    info = done["info"]
    print(info)
    cap = cv2.VideoCapture(info["path"])
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    ok, sample = cap.read()
    cap.release()
    duration = frames / fps if fps else 0
    print(f"opened {width}x{height} fps {fps:.2f} frames {frames} duration {duration:.2f}s readable {ok}")
    if not ok or width != 1920 or height != 1080 or not 14.5 <= duration <= 15.6:
        print("Demo duration or resolution is outside the expected range")
        return 1
    if float(sample.std()) < 1:
        print("Demo frame looks empty")
        return 1
    print("DEMO RECORD CHECK PASSED", path)
    return 0


def _frame(elapsed: float) -> np.ndarray:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    image[:] = (70, 80, 90)
    shift = int((elapsed % 4) * 80)
    cv2.rectangle(image, (100 + shift, 180), (340 + shift, 460), (40, 140, 220), -1)
    depth = np.full((720, 1280), 3.5, dtype=np.float32)
    depth[180:460, 100 + shift : 340 + shift] = 0.8
    state = RenderState(
        rgb=image,
        depth=depth,
        source_size=(1280, 720),
        view_mode="split",
        legend=True,
        show_center=True,
        center_depth=0.8,
        nearest=0.8,
        farthest=3.5,
        focal_px=900,
        camera_fps=30,
        depth_fps=3.2,
        inference_ms=280,
        bands=True,
        recording=True,
        record_remaining=max(0.0, 15 - elapsed),
        record_duration=15,
    )
    canvas, _layout = render(state, 1920, 1080)
    return canvas


if __name__ == "__main__":
    raise SystemExit(main())
