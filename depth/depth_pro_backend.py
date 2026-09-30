"""Apple Depth Pro backend.

Uses the official API from ml-depth-pro:

    depth_pro.create_model_and_transforms()
    model.infer(...)
    prediction["depth"]           # meters
    prediction["focallength_px"]  # pixels
"""

from __future__ import annotations

import gc
from dataclasses import replace

import cv2
import numpy as np
import torch

from utils.gpu_info import best_cuda_index, cuda_devices, describe_runtime
from utils.paths import checkpoint_path
from utils.timers import Stopwatch


class DepthProBackend:
    def __init__(self) -> None:
        self.model = None
        self.transform = None
        self.device: torch.device | None = None
        self.precision: torch.dtype | None = None
        self.precision_label = "FP32"
        self.device_label = "CPU"
        self.ready = False
        self.checkpoint = str(checkpoint_path())

    def runtime_info(self) -> dict:
        info = describe_runtime()
        info["devices"] = cuda_devices()
        return info

    def resolve_device(self, preference: str) -> torch.device:
        preference = (preference or "auto").lower()
        if preference == "cpu":
            return torch.device("cpu")
        if preference.startswith("cuda:"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available on this machine.")
            index = int(preference.split(":", 1)[1])
            if index < 0 or index >= torch.cuda.device_count():
                raise RuntimeError(f"CUDA device {index} is not available.")
            return torch.device(f"cuda:{index}")
        index = best_cuda_index()
        if index is None:
            return torch.device("cpu")
        return torch.device(f"cuda:{index}")

    def resolve_precision(self, preference: str, device: torch.device) -> tuple[torch.dtype, str]:
        preference = (preference or "auto").upper()
        if device.type != "cuda":
            return torch.float32, "FP32"
        if preference == "FP32":
            return torch.float32, "FP32"
        # Official CLI uses precision=torch.half on CUDA.
        return torch.half, "FP16"

    def load(self, device_pref: str = "auto", precision_pref: str = "auto") -> dict:
        self.unload()
        path = checkpoint_path()
        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}. Open Model Manager and download it."
            )

        from depth_pro.depth_pro import (
            DEFAULT_MONODEPTH_CONFIG_DICT,
            create_model_and_transforms,
        )

        device = self.resolve_device(device_pref)
        precision, precision_label = self.resolve_precision(precision_pref, device)
        if device.type == "cuda":
            torch.cuda.set_device(device)

        config = replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=str(path))
        model, transform = create_model_and_transforms(
            config=config,
            device=device,
            precision=precision,
        )
        model.eval()

        self.model = model
        self.transform = transform
        self.device = device
        self.precision = precision
        self.precision_label = precision_label
        self.device_label = self._device_label(device)
        self.checkpoint = str(path)
        self.ready = True
        return self.status()

    def status(self) -> dict:
        gpu_name = ""
        vram_used = 0
        vram_total = 0
        if self.device is not None and self.device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(self.device)
            free, total = torch.cuda.mem_get_info(self.device)
            vram_total = int(total)
            vram_used = int(total) - int(free)
        return {
            "ready": self.ready,
            "checkpoint": self.checkpoint,
            "device": self.device_label,
            "device_type": None if self.device is None else self.device.type,
            "device_index": None if self.device is None or self.device.index is None else int(self.device.index),
            "precision": self.precision_label,
            "gpu_name": gpu_name,
            "vram_used": vram_used,
            "vram_total": vram_total,
            "model_name": "Depth Pro",
            "img_size": None if self.model is None else int(self.model.img_size),
        }

    def unload(self) -> None:
        self.ready = False
        self.model = None
        self.transform = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def infer(self, bgr: np.ndarray, scale: float = 1.0) -> dict:
        if not self.ready or self.model is None or self.transform is None:
            raise RuntimeError("Depth Pro model is not loaded.")

        clock = Stopwatch()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        src_h, src_w = rgb.shape[:2]
        scale = float(scale)
        if scale < 0.999:
            width = max(32, int(round(src_w * scale)))
            height = max(32, int(round(src_h * scale)))
            rgb_in = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
        else:
            rgb_in = rgb
        rgb_in = np.ascontiguousarray(rgb_in)
        preprocess_ms = clock.ms()

        model_clock = Stopwatch()
        tensor = self.transform(rgb_in)
        with torch.inference_mode():
            prediction = self.model.infer(tensor)
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        model_ms = model_clock.ms()

        post_clock = Stopwatch()
        depth_tensor = prediction["depth"].detach().to(dtype=torch.float32).cpu()
        depth = np.squeeze(depth_tensor.numpy()).astype(np.float32)
        focal_value = prediction.get("focallength_px")
        focal_px = _as_float(focal_value)

        in_h, in_w = rgb_in.shape[:2]
        if depth.shape[:2] != (src_h, src_w):
            depth = cv2.resize(depth, (src_w, src_h), interpolation=cv2.INTER_LINEAR)
            if focal_px is not None and in_w > 0:
                focal_px = focal_px * (src_w / in_w)

        depth = np.ascontiguousarray(depth, dtype=np.float32)
        invalid = ~np.isfinite(depth) | (depth <= 0)
        depth[invalid] = np.nan
        post_ms = post_clock.ms()

        return {
            "depth": depth,
            "focal_px": focal_px,
            "preprocess_ms": preprocess_ms,
            "model_ms": model_ms,
            "post_ms": post_ms,
            "inference_ms": preprocess_ms + model_ms + post_ms,
            "input_hw": (in_h, in_w),
            "source_hw": (src_h, src_w),
            "model_img_size": int(self.model.img_size),
        }

    @staticmethod
    def _device_label(device: torch.device) -> str:
        if device.type == "cpu":
            return "CPU"
        name = torch.cuda.get_device_name(device)
        return f"CUDA:{device.index} {name}"


def _as_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        value = float(value.detach().to(dtype=torch.float32).cpu().reshape(-1)[0].item())
    else:
        value = float(value)
    if not np.isfinite(value) or value <= 1.0:
        return None
    return value
