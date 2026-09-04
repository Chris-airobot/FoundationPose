#!/usr/bin/env python3
"""Freeze one atomic RBY1 RGB-D bridge snapshot for FoundationPose."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path.home() / "rby1_ros2_ws/foundationpose_bridge/latest"
DEFAULT_DESTINATION = ROOT / "rby1/live_input"


def read_stable_frame(source: Path, max_tries: int = 20):
    for _ in range(max_tries):
        ready_path = source / "READY"
        if not ready_path.exists():
            time.sleep(0.05)
            continue
        try:
            ready_before = ready_path.read_text(encoding="utf-8").strip()
            bgr = cv2.imread(str(source / "rgb.png"), cv2.IMREAD_COLOR)
            depth = cv2.imread(str(source / "depth_u16.png"), cv2.IMREAD_UNCHANGED)
            K = np.loadtxt(source / "K.txt").reshape(3, 3)
            metadata = json.loads(
                (source / "metadata.json").read_text(encoding="utf-8")
            )
            ready_after = ready_path.read_text(encoding="utf-8").strip()
        except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
            time.sleep(0.05)
            continue
        if ready_before == ready_after:
            return ready_before, bgr, depth, K, metadata
        time.sleep(0.05)
    raise RuntimeError("Could not obtain a stable RGB-D bridge frame")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    snapshot_id, bgr, depth_raw, K, metadata = read_stable_frame(source)

    if bgr is None or depth_raw is None:
        raise RuntimeError("Bridge snapshot contains an unreadable RGB or depth image")
    expected_shape = (int(metadata["height"]), int(metadata["width"]))
    if bgr.shape[:2] != expected_shape or depth_raw.shape != expected_shape:
        raise RuntimeError(
            f"Bridge image shape mismatch: RGB={bgr.shape[:2]} "
            f"depth={depth_raw.shape} metadata={expected_shape}"
        )
    if depth_raw.dtype != np.uint16:
        raise RuntimeError(f"Expected uint16 depth, got {depth_raw.dtype}")
    if metadata.get("depth_aligned_to_color") is not True:
        raise RuntimeError("Bridge depth is not marked aligned to color")
    if metadata.get("frame_id") != "right_d405_optical_frame":
        raise RuntimeError(f"Unexpected camera frame: {metadata.get('frame_id')}")

    depth_scale = float(metadata["depth_scale_m_per_unit"])
    if not np.isfinite(depth_scale) or depth_scale <= 0.0:
        raise RuntimeError(f"Invalid depth scale: {depth_scale}")
    depth_m = depth_raw.astype(np.float32) * depth_scale
    valid = depth_m > 0.0
    if not np.any(valid):
        raise RuntimeError("Bridge snapshot contains no valid depth")

    destination.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination / "rgb.png"), bgr):
        raise RuntimeError("Failed to write frozen RGB image")
    if not cv2.imwrite(str(destination / "depth_u16.png"), depth_raw):
        raise RuntimeError("Failed to write frozen depth image")
    np.savetxt(destination / "K.txt", K, fmt="%.10f")
    shutil.copy2(source / "metadata.json", destination / "metadata.json")

    visualization = np.zeros_like(depth_m)
    low, high = np.percentile(depth_m[valid], [2, 98])
    visualization[valid] = np.clip(
        (depth_m[valid] - low) / max(float(high - low), 1e-6), 0.0, 1.0
    )
    cv2.imwrite(str(destination / "depth_vis.png"), (visualization * 255).astype(np.uint8))

    print("===== RBY1 RGB-D BRIDGE INPUT =====")
    print("Bridge snapshot:", snapshot_id)
    print("RGB:", bgr.shape, bgr.dtype)
    print("Depth:", depth_raw.shape, depth_raw.dtype)
    print("Valid depth:", int(np.count_nonzero(depth_raw)))
    print("Depth range [m]:", float(depth_m[valid].min()), float(depth_m.max()))
    print("K:")
    print(K)
    print("Frame:", metadata["frame_id"])
    print("Aligned:", metadata["depth_aligned_to_color"])
    print("Stable FoundationPose input written to:", destination)


if __name__ == "__main__":
    main()
