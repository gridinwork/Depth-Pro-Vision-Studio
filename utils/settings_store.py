"""Persist GUI settings in config/settings.json."""

from __future__ import annotations

import json

from utils.paths import SETTINGS_PATH, ensure_dirs


DEFAULT_SETTINGS: dict = {
    "source": "webcam",
    "camera_index": 0,
    "resolution": "1280x720",
    "device": "auto",
    "precision": "auto",
    "view_mode": "split",
    "colormap": "turbo",
    "processing_scale": 1.0,
    "overlay_alpha": 0.55,
    "smoothing_enabled": True,
    "smoothing_preset": "MEDIUM",
    "smoothing_alpha": 0.35,
    "near_m": 1.0,
    "mid_m": 3.0,
    "bands_enabled": True,
    "threshold_enabled": False,
    "threshold_m": 1.5,
    "obstacle_enabled": False,
    "obstacle_m": 1.0,
    "show_cursor": True,
    "show_center": True,
    "show_fps": True,
    "show_focal": True,
    "legend_enabled": False,
    "debug": False,
    "histogram": False,
    "demo_resolution": "1920x1080",
    "demo_fps": 30,
    "max_points": 10,
    "geometry": None,
}


def load_settings() -> dict:
    ensure_dirs()
    settings = dict(DEFAULT_SETTINGS)
    if not SETTINGS_PATH.exists():
        return settings
    try:
        stored = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return settings
    if isinstance(stored, dict):
        for key in DEFAULT_SETTINGS:
            if key in stored:
                settings[key] = stored[key]
    return settings


def save_settings(settings: dict) -> None:
    ensure_dirs()
    payload = dict(DEFAULT_SETTINGS)
    payload.update(settings)
    SETTINGS_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
