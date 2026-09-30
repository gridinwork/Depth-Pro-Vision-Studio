"""Compose the viewport that is shown in the GUI and written into the demo video."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from recording.recording_overlay import draw_countdown, draw_recording
from visualization.depth_colormap import colorize
from visualization.depth_legend import draw_legend
from visualization.measurements import draw_center, draw_crosshair, draw_points, draw_roi
from visualization.overlays import apply_bands, apply_obstacle, apply_threshold, draw_banner


@dataclass
class Layout:
    image_size: tuple[int, int]
    source_size: tuple[int, int]
    rgb_rect: tuple[int, int, int, int] | None = None
    depth_rect: tuple[int, int, int, int] | None = None


@dataclass
class RenderState:
    rgb: np.ndarray | None = None
    depth: np.ndarray | None = None
    source_size: tuple[int, int] = (1280, 720)
    view_mode: str = "split"
    colormap: str = "turbo"
    overlay_alpha: float = 0.55
    legend: bool = False
    show_cursor: bool = True
    cursor_xy: tuple[float, float] | None = None
    cursor_depth: float | None = None
    show_center: bool = True
    center_depth: float | None = None
    points: list = field(default_factory=list)
    roi: tuple[float, float, float, float] | None = None
    roi_stats: dict | None = None
    show_fps: bool = True
    show_focal: bool = True
    show_metrics: bool = True
    camera_fps: float = 0.0
    depth_fps: float = 0.0
    focal_px: float | None = None
    inference_ms: float = 0.0
    nearest: float | None = None
    farthest: float | None = None
    near_m: float = 1.0
    mid_m: float = 3.0
    bands: bool = False
    threshold_on: bool = False
    threshold_m: float = 1.5
    obstacle_on: bool = False
    obstacle_m: float = 1.0
    obstacle_detected: bool = False
    countdown: int | None = None
    recording: bool = False
    record_remaining: float = 0.0
    record_duration: float = 15.0


def map_to_source(layout: Layout, ix: float, iy: float) -> tuple[float, float] | None:
    rects = []
    if layout.rgb_rect is not None:
        rects.append(layout.rgb_rect)
    if layout.depth_rect is not None and layout.depth_rect != layout.rgb_rect:
        rects.append(layout.depth_rect)
    elif layout.depth_rect is not None and layout.rgb_rect is None:
        rects.append(layout.depth_rect)
    sw, sh = layout.source_size
    for rect in rects:
        x, y, w, h = rect
        if w <= 0 or h <= 0:
            continue
        if x <= ix < x + w and y <= iy < y + h:
            return (ix - x) / w * sw, (iy - y) / h * sh
    if layout.rgb_rect is not None and layout.depth_rect == layout.rgb_rect:
        x, y, w, h = layout.rgb_rect
        if w > 0 and h > 0 and x <= ix < x + w and y <= iy < y + h:
            return (ix - x) / w * sw, (iy - y) / h * sh
    return None


def render(state: RenderState, canvas_w: int, canvas_h: int) -> tuple[np.ndarray, Layout]:
    canvas_w = max(320, int(canvas_w))
    canvas_h = max(240, int(canvas_h))
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = (18, 20, 24)
    scale = max(0.65, canvas_h / 900.0)
    margin = int(10 * scale)
    top = int(52 * scale)
    bottom = int(58 * scale)
    legend_on = state.legend and state.view_mode != "rgb"
    legend_w = int(118 * scale) if legend_on else 0
    view = (
        margin,
        top,
        max(20, canvas_w - margin * 2 - legend_w),
        max(20, canvas_h - top - bottom),
    )
    mode = state.view_mode
    gap = int(8 * scale)
    rgb_pane = None
    depth_pane = None
    if mode == "rgb":
        rgb_pane = view
    elif mode == "split":
        pane_w = max(10, (view[2] - gap) // 2)
        rgb_pane = (view[0], view[1], pane_w, view[3])
        depth_pane = (view[0] + pane_w + gap, view[1], max(10, view[2] - pane_w - gap), view[3])
    elif mode == "overlay":
        rgb_pane = view
        depth_pane = view
    else:
        depth_pane = view

    depth_color, disp_near, disp_far = colorize(state.depth, state.colormap)
    if depth_color is not None and state.depth is not None:
        if state.bands:
            depth_color = apply_bands(depth_color, state.depth, state.near_m, state.mid_m)
        if state.threshold_on:
            depth_color = apply_threshold(depth_color, state.depth, state.threshold_m)
        if state.obstacle_on:
            depth_color = apply_obstacle(depth_color, state.depth, state.obstacle_m)

    rgb_rect = _blit_labeled(canvas, state.rgb, rgb_pane, "RGB CAMERA", scale, placeholder="No camera frame")
    if mode == "overlay" and rgb_rect is not None and depth_color is not None:
        _blend_depth(canvas, depth_color, rgb_rect, state.overlay_alpha)
        depth_rect = rgb_rect
    else:
        depth_rect = _blit_labeled(
            canvas,
            depth_color,
            depth_pane,
            "DEPTH MAP",
            scale,
            placeholder="Waiting for depth",
        )

    content_rects = [rect for rect in (rgb_rect, depth_rect) if rect is not None]
    unique_rects = []
    for rect in content_rects:
        if rect not in unique_rects:
            unique_rects.append(rect)
    for rect in unique_rects:
        _draw_measurements(canvas, state, rect, scale)

    if legend_on:
        legend_rect = (view[0] + view[2] + int(8 * scale), view[1], legend_w, view[3])
        near = state.nearest if state.nearest is not None else disp_near
        far = state.farthest if state.farthest is not None else disp_far
        draw_legend(canvas, legend_rect, state.colormap, near, far, scale)

    _draw_top(canvas, state, scale)
    _draw_bottom(canvas, state, scale)
    if state.obstacle_on and state.obstacle_detected:
        _draw_obstacle_banner(canvas, scale)
    if state.recording:
        draw_recording(canvas, state.record_remaining, state.record_duration)
    if state.countdown is not None:
        draw_countdown(canvas, state.countdown)

    layout = Layout(
        image_size=(canvas_w, canvas_h),
        source_size=state.source_size,
        rgb_rect=rgb_rect,
        depth_rect=depth_rect,
    )
    return canvas, layout


def _blend_depth(canvas: np.ndarray, depth_color: np.ndarray, rect: tuple[int, int, int, int], alpha: float) -> None:
    x, y, w, h = rect
    resized = cv2.resize(depth_color, (w, h), interpolation=cv2.INTER_LINEAR)
    roi = canvas[y : y + h, x : x + w]
    alpha = float(np.clip(alpha, 0.0, 1.0))
    canvas[y : y + h, x : x + w] = cv2.addWeighted(roi, 1.0 - alpha, resized, alpha, 0)


def _blit_labeled(canvas, image, pane, title, scale, placeholder: str) -> tuple[int, int, int, int] | None:
    if pane is None:
        return None
    x, y, w, h = pane
    cv2.rectangle(canvas, (x, y), (x + w - 1, y + h - 1), (48, 52, 60), 1)
    if image is None:
        _center_text(canvas, placeholder, pane, scale)
        return (x, y, w, h)
    rect = _blit_fit(canvas, image, pane)
    cv2.putText(
        canvas,
        title,
        (rect[0] + 8, rect[1] + int(22 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.45, 0.55 * scale),
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return rect


def _blit_fit(canvas: np.ndarray, image: np.ndarray, pane: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x, y, w, h = pane
    ih, iw = image.shape[:2]
    scale = min(w / max(iw, 1), h / max(ih, 1))
    nw = max(1, int(iw * scale))
    nh = max(1, int(ih * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    ox = x + (w - nw) // 2
    oy = y + (h - nh) // 2
    canvas[oy : oy + nh, ox : ox + nw] = resized
    return ox, oy, nw, nh


def _center_text(canvas, text, pane, scale) -> None:
    x, y, w, h = pane
    cv2.putText(
        canvas,
        text,
        (x + 16, y + h // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.5, 0.6 * scale),
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )


def _draw_measurements(canvas, state: RenderState, rect, scale) -> None:
    if state.show_center:
        draw_center(canvas, rect, state.center_depth, scale)
    if state.show_cursor and state.cursor_xy is not None:
        sw, sh = state.source_size
        px = int(rect[0] + state.cursor_xy[0] / max(sw, 1) * rect[2])
        py = int(rect[1] + state.cursor_xy[1] / max(sh, 1) * rect[3])
        label = "DEPTH --" if state.cursor_depth is None else f"DEPTH {state.cursor_depth:.2f} m"
        draw_crosshair(canvas, px, py, label, scale)
    if state.points:
        draw_points(canvas, state.points, rect, state.source_size, scale)
    if state.roi is not None:
        draw_roi(canvas, state.roi, state.roi_stats, rect, state.source_size, scale)


def _fmt_m(value: float | None) -> str:
    return "--" if value is None else f"{value:.2f} m"


def _draw_top(canvas, state: RenderState, scale) -> None:
    font_scale = max(0.45, 0.5 * scale)
    parts = ["Depth Pro Vision Studio", "Depth Pro"]
    if state.show_fps:
        parts.append(f"Cam {state.camera_fps:.1f} FPS")
        parts.append(f"Depth {state.depth_fps:.1f} FPS")
    if state.show_focal:
        focal = "--" if state.focal_px is None else f"{state.focal_px:.0f} px"
        parts.append(f"f {focal}")
    text_x = int(150 * scale) if state.recording else 16
    bar_h = int(46 * scale)
    draw_banner(canvas, (0, 0, canvas.shape[1], bar_h), "   ".join(parts), font_scale, text_x=text_x)


def _draw_bottom(canvas, state: RenderState, scale) -> None:
    if not state.show_metrics:
        return
    h = canvas.shape[0]
    bar_h = int(50 * scale)
    y = h - bar_h
    parts = [
        f"Nearest {_fmt_m(state.nearest)}",
        f"Center {_fmt_m(state.center_depth)}",
        f"Farthest {_fmt_m(state.farthest)}",
        f"Inference {state.inference_ms:.0f} ms",
    ]
    if state.bands:
        parts.append(f"Near 0-{state.near_m:.1f}m")
        parts.append(f"Mid {state.near_m:.1f}-{state.mid_m:.1f}m")
        parts.append(f"Far >{state.mid_m:.1f}m")
    if state.threshold_on:
        parts.append(f"Closer than {state.threshold_m:.2f} m")
    draw_banner(canvas, (0, y, canvas.shape[1], bar_h), "   ".join(parts), max(0.42, 0.46 * scale))


def _draw_obstacle_banner(canvas, scale) -> None:
    text = "OBSTACLE DETECTED"
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_scale = max(0.7, 0.85 * scale)
    (tw, th), _ = cv2.getTextSize(text, font, text_scale, 2)
    x = canvas.shape[1] - tw - 28
    y = int(70 * scale)
    cv2.rectangle(canvas, (x - 10, y - th - 10), (x + tw + 10, y + 12), (20, 20, 140), -1)
    cv2.putText(canvas, text, (x, y), font, text_scale, (255, 255, 255), 2, cv2.LINE_AA)
