"""Open the real window, run webcam depth, save outputs, and record a 15s demo."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow
from app.theme import apply_theme


def main() -> int:
    app = QApplication(sys.argv)
    apply_theme(app)
    window = MainWindow()
    window.show()

    def dismiss_dialogs() -> None:
        for widget in app.topLevelWidgets():
            if widget.__class__.__name__ == "DemoSavedDialog":
                widget.accept()

    closer = QTimer()
    closer.timeout.connect(dismiss_dialogs)
    closer.start(400)

    deadline = time.perf_counter() + 90
    while time.perf_counter() < deadline and not window.model_status.get("ready"):
        app.processEvents()
        time.sleep(0.05)
    if not window.model_status.get("ready"):
        print("Model did not become ready")
        window.close()
        return 1
    print("model", window.model_status.get("device"), window.model_status.get("precision"))

    window._start()
    deadline = time.perf_counter() + 40
    while time.perf_counter() < deadline and window.current_depth is None:
        app.processEvents()
        time.sleep(0.05)
    if window.current_depth is None:
        print("Webcam depth was not produced")
        window.close()
        return 1
    print(
        "live depth center",
        window.analysis.get("center"),
        "nearest",
        window.analysis.get("nearest"),
        "focal",
        window.depth_meta.get("focal_px"),
        "cam_fps",
        round(window.camera_fps, 2),
        "depth_fps",
        round(float(window.depth_meta.get("depth_fps") or 0), 2),
    )

    width, height = window.current_rgb.shape[1], window.current_rgb.shape[0]
    window._on_cursor(width * 0.5, height * 0.5)
    window._on_point(width * 0.3, height * 0.4)
    window._on_roi(width * 0.2, height * 0.2, width * 0.25, height * 0.25)
    app.processEvents()
    time.sleep(0.2)
    app.processEvents()
    print(
        "cursor",
        window.analysis.get("cursor_depth"),
        "points",
        window.points,
        "roi",
        window.analysis.get("roi_stats"),
    )
    if window.analysis.get("cursor_depth") is None or not window.points:
        print("Cursor or point measurement failed")
        window.close()
        return 1

    from depth.point_cloud import build_point_cloud

    cloud = build_point_cloud(window.current_depth, window.depth_meta.get("focal_px"))
    print("point cloud", "missing focal" if cloud is None else len(cloud[0]))
    window._save_screenshot()
    window._save_rgb()
    window._save_depth()
    window._save_raw()
    before = set(Path(ROOT / "demo_videos").glob("depth_pro_demo_*.mp4"))
    window._record()
    deadline = time.perf_counter() + 30
    finished = False
    while time.perf_counter() < deadline:
        app.processEvents()
        time.sleep(0.05)
        created = set(Path(ROOT / "demo_videos").glob("depth_pro_demo_*.mp4")) - before
        if created and not window._recording and not window._countdown_value:
            finished = True
            break
    dismiss_dialogs()
    app.processEvents()
    if not finished:
        print("GUI demo recording did not finish")
        window.close()
        return 1
    newest = max(set(Path(ROOT / "demo_videos").glob("depth_pro_demo_*.mp4")) - before, key=lambda p: p.stat().st_mtime)
    import cv2

    cap = cv2.VideoCapture(str(newest))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(60, max(0, frames // 2)))
    ok, sample = cap.read()
    cap.release()
    duration = frames / fps if fps else 0
    print(f"gui demo {newest.name} {w}x{h} {duration:.2f}s frames {frames} readable {ok}")
    window.close()
    app.processEvents()
    if not ok or w != 1920 or h != 1080 or not 14.5 <= duration <= 16.0:
        print("GUI demo file is outside the expected range")
        return 1
    if float(sample.std()) < 1:
        print("GUI demo frame looks empty")
        return 1
    print("GUI CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
