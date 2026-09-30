"""Project paths for Depth Pro Vision Studio."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
CONFIG_DIR = ROOT / "config"
CHECKPOINT_DIR = ROOT / "checkpoints"
SCREENSHOT_DIR = ROOT / "screenshots"
DEPTH_MAP_DIR = ROOT / "depth_maps"
RAW_DEPTH_DIR = ROOT / "raw_depth"
DEMO_DIR = ROOT / "demo_videos"
LOG_DIR = ROOT / "logs"
VENDOR_DIR = ROOT / "_vendor" / "ml-depth-pro"

CHECKPOINT_NAME = "depth_pro.pt"
SETTINGS_PATH = CONFIG_DIR / "settings.json"
LOG_PATH = LOG_DIR / "app.log"

CHECKPOINT_URL = "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt"
# Official file is about 1.82 GB. Anything smaller is an incomplete download.
MIN_CHECKPOINT_BYTES = 1_500_000_000


def checkpoint_path() -> Path:
    return CHECKPOINT_DIR / CHECKPOINT_NAME


def ensure_dirs() -> None:
    for path in (
        CONFIG_DIR,
        CHECKPOINT_DIR,
        SCREENSHOT_DIR,
        DEPTH_MAP_DIR,
        RAW_DEPTH_DIR,
        DEMO_DIR,
        LOG_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
