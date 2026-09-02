#!/usr/bin/env python3
"""Snapshot-only FoundationPose validation for the RBY1 cuboid.

This script intentionally reads a previously captured aligned RGB-D frame from
rby1/data/cuboid_single. It is NOT the live-camera path. Use it only to verify
that FoundationPose registration and the cuboid mesh overlay work on a known
aligned frame.
"""

from pathlib import Path
import sys

import cv2
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from estimater import *
from Utils import *

DATA = ROOT / "rby1/data/cuboid_single"
OUT = ROOT / "rby1/results/cuboid_single"

MESH = Path(
    "/home/samsung/Chris/placement-generalization-execution-aware/"
    "assets/real_robot_meshes/061_foam_brick_final30_exact_mm.obj"
)

OUT.mkdir(parents=True, exist_ok=True)

bgr = cv2.imread(str(DATA / "rgb.ppm"))
if bgr is None:
    raise RuntimeError(f"Could not read {DATA / 'rgb.ppm'}")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

depth_raw = np.fromfile(DATA / "depth_u16.raw", dtype=np.uint16)
if depth_raw.size != 640 * 480:
    raise RuntimeError(f"Unexpected depth size: {depth_raw.size}")
depth_raw = depth_raw.reshape(480, 640)

scale = float((DATA / "depth_scale.txt").read_text().strip())
depth = depth_raw.astype(np.float32) * scale
depth[(depth < 0.05) | (depth > 2.0)] = 0

K = np.loadtxt(DATA / "K.txt").reshape(3, 3)

print("K:")
print(K)
print("valid depth pixels:", np.count_nonzero(depth))

x, y, w, h = map(
    int,
    cv2.selectROI(
        "Draw tight rectangle around cuboid -> ENTER",
        bgr,
        showCrosshair=True,
        fromCenter=False,
    ),
)
cv2.destroyAllWindows()

if w <= 0 or h <= 0:
    raise RuntimeError("No cuboid selected")

mask = np.zeros((480, 640), dtype=np.uint8)
mask[y : y + h, x : x + w] = 1
cv2.imwrite(str(OUT / "mask.png"), mask * 255)

# Physical mesh is stored in millimetres; FoundationPose uses metres.
mesh = trimesh.load(str(MESH), force="mesh")
mesh.apply_scale(0.001)
print("mesh extents [m]:", mesh.extents)

to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

est = FoundationPose(
    model_pts=mesh.vertices,
    model_normals=mesh.vertex_normals,
    mesh=mesh,
    scorer=ScorePredictor(),
    refiner=PoseRefinePredictor(),
    debug_dir=str(OUT),
    debug=0,
    glctx=dr.RasterizeCudaContext(),
)

pose = est.register(
    K=K,
    rgb=rgb,
    depth=depth,
    ob_mask=mask.astype(bool),
    iteration=5,
)

np.savetxt(OUT / "camera_T_object.txt", pose)

print()
print("===== FOUNDATIONPOSE RESULT =====")
print(pose)
print("xyz [m]:", pose[:3, 3])

center_pose = pose @ np.linalg.inv(to_origin)

vis = draw_posed_3d_box(
    K,
    img=rgb,
    ob_in_cam=center_pose,
    bbox=bbox,
)

vis = draw_xyz_axis(
    vis,
    ob_in_cam=center_pose,
    scale=0.05,
    K=K,
    thickness=3,
    transparency=0,
    is_input_rgb=True,
)

vis_bgr = vis[..., ::-1].copy()
cv2.imwrite(str(OUT / "overlay.png"), vis_bgr)

cv2.imshow("FoundationPose cuboid snapshot", vis_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
