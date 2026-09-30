# Depth Pro Vision Studio

A local Windows computer-vision application for **monocular metric depth estimation** from a standard RGB camera, video file or still image using Apple's **Depth Pro** model.

The application estimates a metric depth map in meters and adds an engineering-oriented desktop interface for live depth visualization, cursor/ROI measurements, near/far analysis, thresholding, temporal smoothing, obstacle highlighting, raw depth export, 3D point-cloud viewing and demo recording.

> **Important:** monocular depth is an AI estimate, not a certified distance sensor. This application is not a safety-rated collision-avoidance system or calibrated metrology instrument.

## What is monocular metric depth?

Monocular depth estimates distance using a single RGB image rather than stereo cameras or a dedicated ToF/LiDAR sensor. **Metric** depth means the model attempts to return absolute distance values (meters), rather than only a relative near/far ordering.

Depth Pro also estimates focal length in pixels when it is not supplied from image metadata. The application uses the official Depth Pro inference API and preserves depth values separately from visualization colors.

## Main features

- Webcam, video-file and image input
- Metric depth estimation in meters
- RGB, DEPTH, SPLIT, OVERLAY and DEPTH + LEGEND views
- Multiple depth colormaps
- Center-depth crosshair
- Cursor depth measurement
- Up to 10 persistent point measurements
- ROI depth statistics
- Near / Medium / Far zone analysis
- Adjustable depth threshold
- Experimental obstacle highlighting
- Temporal depth smoothing
- Camera FPS and Depth FPS shown separately
- Dropped-frame monitoring
- GPU / CPU device selection
- FP16 / FP32 precision control
- Integrated model/checkpoint manager
- Raw metric depth export to NPZ
- Color depth-map export
- RGB screenshot export
- 15-second demo recording
- Depth histogram
- Interactive 3D point-cloud visualization
- Debug timing and GPU/VRAM diagnostics

## Depth views

### RGB
Displays the original camera/video frame.

### DEPTH
Displays only the generated metric depth map.

### SPLIT
Shows RGB and depth side by side. This is useful for comparing visual structures against predicted distance.

### OVERLAY
Blends the depth colormap over the RGB image with adjustable opacity.

### DEPTH + LEGEND
Displays the depth map together with a vertical scale in meters.

Available palettes include Inferno, Turbo, Viridis, Magma and Grayscale. The colormap affects visualization only; exported/raw depth remains numeric metric data.

## Point and cursor measurements

With **Show Cursor Depth**, moving the mouse over the frame reports X/Y location and predicted depth in meters. Clicking can place up to ten labeled points such as:

```text
P1 0.82 m
P2 1.47 m
```

A center crosshair can continuously show the estimated distance at the center of the frame.

## ROI measurements

The user can draw a rectangular region of interest and inspect:

- minimum depth
- maximum depth
- mean depth
- median depth

The median is especially useful for reducing sensitivity to isolated depth outliers inside the selected region.

## Near / Far analysis

The visualization can divide the metric depth map into configurable distance zones:

- Near
- Medium
- Far

The thresholds affect only the analysis/visualization layer and do not modify the underlying Depth Pro prediction.

## Depth threshold and obstacle highlighting

A depth threshold can hide/dim pixels beyond a selected distance. The experimental obstacle mode highlights sufficiently large image regions predicted to be closer than a configured threshold.

This feature is intended for visualization and prototyping only. It must not be treated as a certified collision-prevention function.

## Temporal smoothing

Live monocular depth can fluctuate frame to frame. The application provides OFF / LOW / MEDIUM / HIGH smoothing modes based on exponential temporal filtering after inference.

The smoothing stage does not alter or retrain the Depth Pro model; it filters the resulting depth sequence for a more stable visualization.

## Raw depth export

**SAVE RAW DEPTH** stores the numerical result as an `.npz` file containing:

- `depth_meters`
- `focal_length_px`
- `timestamp`

This makes the application useful as a prototyping front end for downstream robotics, scene-analysis and measurement experiments.

## 3D point cloud

When a valid focal-length estimate is available, the current depth map can be projected into an interactive point cloud. The viewer supports orbit, zoom, rotate and reset controls.

No invented fallback focal length is used for this feature; if valid camera geometry is unavailable, point-cloud creation is disabled rather than fabricating coordinates.

## Performance architecture

Camera capture and model inference are decoupled. While inference is busy, newer camera frames can replace older queued frames instead of growing an unbounded backlog. The interface therefore reports both:

- Camera FPS
- Depth FPS
- Dropped Frames

This is important because a heavy depth model may update more slowly than the live camera stream.

## GPU and precision

Supported device modes:

- `AUTO` — CUDA when available, otherwise CPU
- explicit `CUDA:N`
- CPU

Precision modes:

- `AUTO` — FP16 on CUDA and FP32 on CPU
- FP16
- FP32

The status panel can report selected GPU and VRAM usage where CUDA is available.

## Model Manager

The application expects the official Depth Pro checkpoint in `checkpoints/depth_pro.pt` and provides controls to download, verify and inspect the model file.

The checkpoint is intentionally **not committed to this GitHub repository**.

## Installation

Recommended environment:

- Windows 10/11
- Python 3.11
- NVIDIA GPU recommended for practical performance
- Git available during first installation so the official Depth Pro package can be installed from Apple's repository

Run:

```bat
install.bat
```

The installer:

1. creates `.venv`;
2. installs PyTorch and application dependencies;
3. installs the official Depth Pro package from Apple's GitHub repository;
4. downloads the official Depth Pro checkpoint when missing;
5. runs installation and inference checks.

Then launch:

```bat
start.bat
```

After packages and weights are installed, normal inference is local.

## Repository structure

```text
app/            PySide6 GUI, panels and dialogs
capture/        webcam, video, image and frame handling
depth/          Depth Pro backend, worker, statistics and depth utilities
visualization/  colormaps, legends, overlays and rendering
recording/      demonstration video recording
utils/          downloader, GPU, settings, paths and timing helpers
scripts/        installation and smoke tests
checkpoints/    Depth Pro weights (not committed)
screenshots/    runtime screenshots (not committed)
depth_maps/     rendered depth maps (not committed)
raw_depth/      metric NPZ exports (not committed)
demo_videos/    demo recordings (not committed)
```

## Planned media

Screenshots and demonstration video will be added separately.

## Privacy / local processing

After the model and dependencies are downloaded, RGB frames and depth inference are processed locally by the application. No cloud inference API is required.

## Depth Pro attribution and license

Depth estimation is powered by Apple's open-source **Depth Pro** project:

https://github.com/apple/ml-depth-pro

Depth Pro is distributed under Apple's license included in its upstream repository. This repository does **not** vendor Apple's Depth Pro source code or model checkpoint; the installation script obtains the official package/checkpoint separately.

Original application code in this repository is released under the **Apache License 2.0**. See [LICENSE](LICENSE) and [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

## Limitations

Monocular depth may be inaccurate around reflective or transparent surfaces, textureless walls, unusual lenses, extreme close-ups, thin structures and scenes unlike the model's training distribution. Always validate predictions against a suitable physical sensor when absolute accuracy or safety matters.
