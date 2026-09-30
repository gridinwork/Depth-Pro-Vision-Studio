"""Install-time checks: imports, Depth Pro inference, depth export, video writer."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np


def main() -> int:
    print("Python", sys.version.replace("\n", " "))
    try:
        import cv2
        import torch
        from PIL import Image
        from PySide6 import QtCore
        import depth_pro
    except Exception:
        traceback.print_exc()
        return 1

    print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
    print("opencv", cv2.__version__)
    print("PySide6", QtCore.__version__)
    print("depth_pro", depth_pro.__file__)
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            print(f"GPU {index}: {props.name} {props.total_memory / 1024**3:.1f} GB")

    from utils.downloader import checkpoint_info
    from utils.images import imwrite
    from utils.paths import RAW_DEPTH_DIR, SCREENSHOT_DIR, ensure_dirs
    from visualization.depth_colormap import colorize
    from visualization.renderer import RenderState, render

    ensure_dirs()
    info = checkpoint_info()
    print("checkpoint", info["status"], info["size"])
    if info["status"] != "INSTALLED":
        print("Checkpoint is missing. Model inference was not run.")
        return 1

    from depth.depth_analyzer import analyze
    from depth.depth_pro_backend import DepthProBackend
    from recording.video_encoder import VideoEncoder

    image = _synthetic_scene()
    state = RenderState(rgb=image, depth=None, source_size=(image.shape[1], image.shape[0]), view_mode="split")
    preview, _layout = render(state, 1280, 720)
    if preview.shape != (720, 1280, 3):
        print("Renderer produced an unexpected shape", preview.shape)
        return 1

    backend = DepthProBackend()
    status = backend.load("auto", "auto")
    print("model", status["device"], status["precision"], "img", status["img_size"])
    result = backend.infer(image, scale=1.0)
    depth = result["depth"]
    valid = np.isfinite(depth) & (depth > 0)
    print(
        "depth",
        depth.shape,
        "valid",
        float(valid.mean()),
        "median_m",
        float(np.median(depth[valid])) if valid.any() else None,
        "focal_px",
        result["focal_px"],
        "ms",
        round(result["inference_ms"], 1),
    )
    if depth.shape[:2] != image.shape[:2] or valid.mean() < 0.5 or result["focal_px"] is None:
        print("Inference result is not usable.")
        return 1
    stats = analyze(depth, image.shape[1], image.shape[0], None, [], None, 1.0, 3.0, 1.0)
    print("nearest", stats["nearest"], "farthest", stats["farthest"], "center", stats["center"])

    colored, _near, _far = colorize(depth, "turbo")
    shot = SCREENSHOT_DIR / "install_check_depth.png"
    imwrite(shot, colored)
    raw_path = RAW_DEPTH_DIR / "install_check.npz"
    np.savez_compressed(
        raw_path,
        depth_meters=depth.astype(np.float32),
        focal_length_px=np.float32(result["focal_px"]),
        timestamp=np.array("install-check"),
    )
    loaded = np.load(raw_path)
    if loaded["depth_meters"].shape != depth.shape:
        print("Raw depth reload failed")
        return 1
    print("saved", shot)
    print("saved", raw_path)

    video_path = ROOT / "demo_videos" / "install_check.mp4"
    encoder = VideoEncoder()
    codec = encoder.open(video_path, 1280, 720, 30)
    frame, _layout = render(
        RenderState(
            rgb=image,
            depth=depth,
            source_size=(image.shape[1], image.shape[0]),
            view_mode="split",
            legend=True,
            focal_px=result["focal_px"],
            nearest=stats["nearest"],
            farthest=stats["farthest"],
            center_depth=stats["center"],
            show_center=True,
            camera_fps=30,
            depth_fps=1,
            inference_ms=result["inference_ms"],
        ),
        1280,
        720,
    )
    for _ in range(30):
        encoder.write(frame)
    encoder.close()
    cap = cv2.VideoCapture(str(video_path))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    ok, sample = cap.read()
    cap.release()
    print("video", codec, video_path, f"{width}x{height}", "fps", fps, "frames", frames, "readable", ok)
    if not ok or frames < 20 or width != 1280 or height != 720:
        print("Video writer check failed")
        return 1

    webcam = _try_webcam(backend)
    print("webcam", webcam)
    backend.unload()
    print("INSTALL CHECK PASSED")
    return 0


def _synthetic_scene() -> np.ndarray:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:] = (180, 170, 160)
    cv2 = __import__("cv2")
    cv2.rectangle(image, (80, 120), (260, 360), (40, 90, 200), -1)
    cv2.circle(image, (430, 220), 90, (200, 180, 40), -1)
    cv2.putText(image, "Depth Pro", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
    return image


def _try_webcam(backend) -> str:
    import cv2

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        return "no camera opened"
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return "camera opened but returned no frame"
    result = backend.infer(frame, scale=0.5)
    depth = result["depth"]
    valid = np.isfinite(depth) & (depth > 0)
    if valid.mean() < 0.5:
        return "camera frame inference had too few valid depths"
    return f"ok median {float(np.median(depth[valid])):.2f} m focal {result['focal_px']:.0f} px"


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
