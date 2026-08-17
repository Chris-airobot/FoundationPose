# FoundationPose for G1 Box Pose Tracking

This repository adapts [NVIDIA FoundationPose](https://github.com/NVlabs/FoundationPose) for **6D pose estimation and continuous tracking of a package box from the Unitree G1 RGB-D camera stream**.

The project is the perception front end of a larger humanoid box-manipulation pipeline. Given a known box mesh, RGB-D observations, camera intrinsics, and a one-time initialization mask, the system estimates the box pose in the camera frame and continuously tracks it while the box or robot moves.

```text
G1 D435i RGB-D stream
        ↓
depth-to-color alignment
        ↓
one-time box mask
        ↓
FoundationPose register()
        ↓
FoundationPose track_one()
        ↓
6D box pose: T_camera_box
        ↓
robot-frame transform / downstream manipulation
```

## Demo video

🎥 **G1 live box-tracking demo:** _video link to be added_

The README is ready for the video. A short recording showing the real RGB image with the tracked 3D box/axes while the box is moved through several positions is sufficient; the link can then be inserted here.

## What has been completed

The current implementation has been tested beyond the original FoundationPose demo:

- [x] Model-based pose estimation using a custom **40 × 30 × 30 cm** box mesh.
- [x] FoundationPose running on the Samsung RTX workstation with a recent CUDA/PyTorch stack.
- [x] Unitree G1 RGB-D camera access through the existing ZMQ camera server.
- [x] RealSense **depth-to-color alignment** before pose estimation.
- [x] One-time pose registration from an initialization image and mask.
- [x] Continuous live box tracking with `track_one()`.
- [x] Latest 4×4 box pose written for downstream use.
- [x] Live FPS/timing measurement.
- [x] Offline RGB-D recording and replay utilities.
- [x] Tracking tests while the box is moved to different positions/ranges.
- [x] Range/stability analysis tools.
- [x] AprilTag-based geometry, distance-reference, and sanity-check tools.
- [x] Camera-to-robot transform and offline root-target utilities for later G1 integration.

The main remaining work is **system integration** with the humanoid manipulation controller, not basic FoundationPose bring-up.

---

## Box model

For the current task the manipulated object is approximated as a rectangular package box:

```text
Length: 0.40 m
Width:  0.30 m
Height: 0.30 m
```

The mesh is stored as:

```text
box.obj
```

and can be regenerated with:

```bash
python makee_box.py
```

The mesh origin is at the center of the box and all dimensions are in metres.

Because a CAD/mesh model is available, this project uses the **model-based** FoundationPose path. No object-specific network training is required.

---

## Live G1 pipeline

The main live script is:

```text
g1/scripts/live_foundationpose_box.py
```

It connects to the G1 camera server at:

```text
tcp://192.168.123.164:5555
```

and expects aligned RGB and depth images from the G1 RealSense camera.

### Initialization

FoundationPose requires an object mask for the first registration frame. The current live setup uses:

```text
g1/data/live_init/
├── rgb/000000.png
├── depth/000000.png
├── masks/000000.png
└── cam_K.txt
```

The initial mask can be prepared with the provided mask utility:

```bash
python g1/scripts/make_first_mask.py
```

Only the initialization/re-registration frame requires this mask. After registration, tracking proceeds from RGB-D frames without a new mask on every frame.

### Run live tracking

On the RTX workstation:

```bash
cd ~/Chris/FoundationPose
conda activate foundationpose5080
python -u g1/scripts/live_foundationpose_box.py
```

Controls in the visualization window:

```text
q  quit
r  freeze the current frame, redraw the box mask, and re-register
```

The script performs:

```python
pose = estimator.register(...)
```

once during initialization, followed by:

```python
pose = estimator.track_one(...)
```

for live tracking.

The latest estimated pose is saved to:

```text
g1/results/live_foundationpose/latest_pose.txt
```

This is a 4×4 homogeneous transform representing the box pose in the camera frame.

---

## RGB-D alignment

Correct RGB/depth registration was important for this project. Early tests showed that using RGB and depth with mismatched camera geometry noticeably degraded the estimated box overlay.

The G1 RealSense driver backup used for this work is under:

```text
g1/camera_server/g1_camera_server_backup/
```

The RealSense stream is aligned with:

```python
self.align = rs.align(rs.stream.color)
frames = self.align.process(frames)
```

so the depth image corresponds to the RGB/color frame used by FoundationPose.

The live camera data used by the tracker is 640 × 480 RGB-D.

---

## Runtime performance

A dedicated core benchmark is provided at:

```text
g1/scripts/benchmark_foundationpose_core.py
```

It preloads RGB-D frames into RAM, removes disk I/O and visualization overhead, performs one registration, warms up the tracker, and then measures repeated `track_one()` calls.

On the Samsung RTX 5080 workstation, the measured core tracking result was approximately:

| Metric | Result |
|---|---:|
| Core tracking throughput | **104.7 FPS** |
| Mean tracking time | **9.55 ms/frame** |
| p95 tracking time | **12.51 ms/frame** |

This shows that the FoundationPose tracking computation itself has enough headroom for a high-rate perception loop. The complete live rate can still be lower because it also includes RealSense capture/alignment, encoding, ZMQ transfer, decoding, depth conversion, and visualization.

The live tracker therefore uses several simple latency controls:

- `TRACK_ITERS = 1`
- ZMQ `CONFLATE=1`, so old frames are dropped rather than queued
- visualization only every third pose frame
- pose file writing only about once per second

The terminal reports:

```text
camera-arrival FPS=... | pose-output FPS=... | track=... ms
```

---

## Offline recording and replay

The repository also contains an offline workflow so perception experiments can be repeated without continuously operating the robot.

Important scripts include:

| Script | Purpose |
|---|---|
| `g1/scripts/record_offline_bundle.py` | Record reusable RGB-D bundles from the G1 stream |
| `g1/scripts/verify_offline_bundle.py` | Verify captured RGB-D data and extract AprilTag information |
| `g1/scripts/foundationpose_offline_bundle.py` | Run FoundationPose over a recorded bundle |
| `g1/scripts/replay_foundationpose_benchmark.py` | Replay recorded sequences for benchmarking |
| `g1/scripts/compare_offline_fp_apriltag.py` | Visual comparison between FoundationPose and AprilTag reference geometry |
| `g1/scripts/offline_fp_range_analysis.py` | Analyze FoundationPose continuity/stability versus object distance |

This workflow was used to test the tracker across recorded motion instead of relying only on a single static image.

---

## Range and stability testing

The live range benchmark is:

```bash
python -u g1/scripts/range_benchmark.py
```

The intended procedure is:

1. Draw the mask once at a good initial view.
2. Register FoundationPose once.
3. Continuously move the box closer/farther while tracking remains active.
4. Stop the box at selected distances and record stationary samples.
5. Analyze translation/rotation continuity, tracking time, projected box visibility, and depth support.

For offline analysis, `offline_fp_range_analysis.py` uses AprilTag detections primarily as a **distance reference**, while FoundationPose continuity is evaluated with frame-to-frame translation and symmetry-aware rotation changes.

This distinction is important: the AprilTag tools in this repo are useful for sanity checking and distance/reference measurements, but the current experiments should not be described as a calibrated absolute 6D ground-truth accuracy benchmark.

The box has a square 30 cm × 30 cm cross-section, so the analysis includes symmetry handling to avoid incorrectly treating equivalent box orientations as large tracking errors.

---

## Robot-frame integration

FoundationPose directly produces:

```text
T_camera_box
```

For robot control, this can be transformed into the robot/base frame using a calibrated camera transform:

```text
T_robot_box = T_robot_camera × T_camera_box
```

Utilities related to this stage are included in:

```text
g1/scripts/derive_robot_camera_tf.py
g1/scripts/offline_root_pipeline.py
g1/scripts/visualize_root_pipeline.py
g1/config/root_pipeline_demo.json
```

These scripts support the next system stage: turning perceived box poses into target information for humanoid approach, grasping, carrying, and placement.

---

## Project structure

```text
FoundationPose/
├── box.obj                         # 40 × 30 × 30 cm box mesh
├── makee_box.py                    # box mesh generator
├── estimater.py                    # FoundationPose estimator/tracker
├── run_demo.py                     # original model-based demo entry point
├── ENV_SETUP.md                    # local environment notes
│
└── g1/
    ├── camera_server/              # G1 camera-server backup / RealSense alignment
    ├── config/                     # transform / offline pipeline configs
    ├── data/                       # small tracked configuration/data files
    └── scripts/
        ├── live_foundationpose_box.py
        ├── make_first_mask.py
        ├── record_offline_bundle.py
        ├── foundationpose_offline_bundle.py
        ├── benchmark_foundationpose_core.py
        ├── range_benchmark.py
        ├── offline_fp_range_analysis.py
        ├── apriltag_gt_benchmark.py
        ├── compare_offline_fp_apriltag.py
        ├── derive_robot_camera_tf.py
        └── offline_root_pipeline.py
```

Large recordings, logs, and generated result folders are intentionally not treated as source files and should remain outside normal Git commits.

---

## Environment

The Samsung development machine uses a recent NVIDIA GPU/CUDA stack. The working FoundationPose environment used during the G1 tests is named:

```bash
conda activate foundationpose5080
```

The project was brought up with CUDA 12.8-compatible PyTorch and locally built CUDA extensions needed by FoundationPose/PyTorch3D/NVDiffRast.

For detailed installation notes, see:

```text
ENV_SETUP.md
```

The exact environment may need small changes depending on GPU generation and CUDA installation.

---

## Current limitation

The main perception limitation is initialization:

> **The first box mask is manually supplied.**

After initialization, FoundationPose tracks continuously without requiring a fresh mask each frame. A future version could replace manual initialization with an object detector/segmenter if fully automatic startup is required.

Other downstream work, such as camera-to-robot calibration validation and connecting the pose output to a humanoid controller, is considered integration work rather than a missing FoundationPose tracking capability.

---

## Upstream FoundationPose

This project is based on the official NVIDIA Research implementation:

- FoundationPose repository: https://github.com/NVlabs/FoundationPose
- Project page: https://nvlabs.github.io/FoundationPose/
- Paper: *FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects*, CVPR 2024 Highlight

Please refer to the upstream repository for the original pretrained weights, public-dataset instructions, model-free pipeline, and full FoundationPose documentation.

### Citation

```bibtex
@InProceedings{foundationposewen2024,
  author    = {Bowen Wen and Wei Yang and Jan Kautz and Stan Birchfield},
  title     = {FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects},
  booktitle = {CVPR},
  year      = {2024}
}
```

## License

The upstream FoundationPose code and data are released under the NVIDIA Source Code License. See `LICENSE` and the original NVIDIA repository for details.
