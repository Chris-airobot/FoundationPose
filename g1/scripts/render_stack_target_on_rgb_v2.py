#!/usr/bin/env python3
"""Corrected real-RGB stack overlay.

Unlike V1, this does NOT assume FoundationPose local +Z is the physical top.
For the 40x30x30 cm box, Y and Z are both 30 cm, so FoundationPose may use
either short axis (and either sign) as the upward direction. We choose the
upward-facing short axis from the support pose projection, then place the
desired final pose B one box height along that top-face normal.

Visualization only: saved RGB + saved FoundationPose poses; no GPU inference,
Isaac Sim, or robot commands.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

import render_stack_target_on_rgb as base

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = ROOT / "g1/data/offline_bundle_20260813_140853"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    p.add_argument("--every", type=int, default=3)
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--box-dims", type=float, nargs=3, default=(0.40, 0.30, 0.30))
    p.add_argument("--box-height", type=float, default=0.30)
    p.add_argument("--preplace-clearance", type=float, default=0.12)
    p.add_argument("--support-record-index", type=int, default=-1)
    p.add_argument(
        "--top-axis", default="auto",
        choices=["auto", "+y", "-y", "+z", "-z"],
        help="Default auto chooses the upward-facing short axis.",
    )
    return p.parse_args()


def parse_axis(name):
    axis = np.zeros(3, dtype=np.float64)
    idx = {"y": 1, "z": 2}[name[1].lower()]
    axis[idx] = 1.0 if name[0] == "+" else -1.0
    return axis


def axis_name(a):
    idx = int(np.argmax(np.abs(a)))
    return ("+" if a[idx] > 0 else "-") + "XYZ"[idx]


def project_cam_point(p, K):
    p = np.asarray(p, dtype=np.float64)
    if p[2] <= 1e-6:
        return None
    return np.array([
        K[0, 0] * p[0] / p[2] + K[0, 2],
        K[1, 1] * p[1] / p[2] + K[1, 2],
    ])


def choose_top_axis(T_support, K, forced="auto"):
    """Choose +/-Y or +/-Z that points most upward in the RGB image."""
    if forced != "auto":
        return parse_axis(forced), 1.0

    center3 = T_support[:3, 3]
    center2 = project_cam_point(center3, K)
    if center2 is None:
        raise RuntimeError("support center behind camera")

    candidates = [
        np.array([0., 1., 0.]), np.array([0., -1., 0.]),
        np.array([0., 0., 1.]), np.array([0., 0., -1.]),
    ]
    best = None
    for a_obj in candidates:
        p3 = center3 + T_support[:3, :3] @ (0.12 * a_obj)
        p2 = project_cam_point(p3, K)
        if p2 is None:
            continue
        d = p2 - center2
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            continue
        # Image y grows downward. +1 means straight upward on image.
        upness = float(-d[1] / n)
        sideways = abs(float(d[0])) / n
        score = upness - 0.10 * sideways
        if best is None or score > best[0]:
            best = (score, a_obj)

    if best is None:
        raise RuntimeError("could not infer top direction")
    return best[1], float(best[0])


def shift_pose(T, axis_obj, distance):
    out = T.copy()
    direction_cam = T[:3, :3] @ axis_obj
    direction_cam = direction_cam / np.linalg.norm(direction_cam)
    out[:3, 3] += float(distance) * direction_cam
    return out


def geometry(T_support, K, box_height, clearance, top_axis):
    a_obj, score = choose_top_axis(T_support, K, top_axis)
    T_B = shift_pose(T_support, a_obj, box_height)
    T_pre = shift_pose(T_B, a_obj, clearance)
    a_cam = T_support[:3, :3] @ a_obj
    a_cam /= np.linalg.norm(a_cam)
    return a_obj, a_cam, score, T_B, T_pre


def choose_support(records, K, dims, box_height, clearance, w, h, top_axis):
    pts = base.box_corners(dims)
    best = None
    stride = max(1, len(records) // 300)

    for idx in range(0, len(records), stride):
        T = records[idx]["pose"]
        try:
            a_obj, _, up_score, B, pre = geometry(
                T, K, box_height, clearance, top_axis
            )
        except RuntimeError:
            continue

        poses = [T, B, pre]
        fractions = [
            base.visible_corner_fraction(base.project(pts, x, K), w, h)
            for x in poses
        ]
        centers = [base.project_center(x, K) for x in poses]
        if min(fractions) < 0.50:
            continue
        if not all(base.in_image(c, w, h) for c in centers):
            continue

        score = 300.0 * up_score + 100.0 * sum(fractions)
        if best is None or score > best[0]:
            best = (score, idx, axis_name(a_obj))

    return best[1] if best is not None else len(records) // 2


def render_frame(img, T_A, T_support, T_B, T_pre, K, pts, idx, total, top_label):
    vis = img.copy()

    for label, T, key, alpha, thick in [
        ("VIRTUAL SUPPORT", T_support, "support", 0.62, 2),
        ("DESIRED FINAL B - ON TOP", T_B, "target", 0.78, 3),
        ("PRE-PLACE", T_pre, "preplace", 0.62, 2),
    ]:
        base.draw_wireframe(vis, base.project(pts, T, K), base.COLORS[key], thick, alpha)
        base.draw_label(vis, base.project_center(T, K), label, base.COLORS[key])

    base.draw_wireframe(vis, base.project(pts, T_A, K), base.COLORS["tracked"], 3, 1.0)
    base.draw_label(vis, base.project_center(T_A, K), "TRACKED A", base.COLORS["tracked"])

    cA = base.project_center(T_A, K)
    cB = base.project_center(T_B, K)
    cS = base.project_center(T_support, K)
    if cA is not None and cB is not None:
        cv2.arrowedLine(
            vis, tuple(np.round(cA).astype(int)), tuple(np.round(cB).astype(int)),
            base.COLORS["path"], 2, cv2.LINE_AA, tipLength=0.04
        )
    if cS is not None and cB is not None:
        cv2.arrowedLine(
            vis, tuple(np.round(cS).astype(int)), tuple(np.round(cB).astype(int)),
            base.COLORS["target"], 2, cv2.LINE_AA, tipLength=0.10
        )

    base.put_text(vis, "CORRECTED: desired B is ON TOP of support", 26)
    base.put_text(vis, f"frame {idx}/{total} | top direction {top_label}", 50, scale=0.44)
    base.put_text(
        vis,
        "green=A | cyan=support | magenta=desired final B | yellow=pre-place",
        74, scale=0.40
    )
    return vis


def main():
    args = parse_args()
    bundle = args.bundle
    K = np.loadtxt(bundle / "cam_K.txt").reshape(3, 3)
    timestamps = list(csv.DictReader((bundle / "timestamps.csv").open()))
    fp_csv = bundle / "foundationpose_offline" / "foundationpose_poses.csv"
    fp_rows = {str(r["frame"]): r for r in csv.DictReader(fp_csv.open())}

    records = []
    for ts in timestamps:
        row = fp_rows.get(str(ts["frame"]))
        rgb_path = bundle / ts["rgb_file"]
        if row is None or not rgb_path.exists():
            continue
        records.append({
            "frame_id": str(ts["frame"]),
            "rgb_path": rgb_path,
            "pose": base.load_pose(row),
        })
    if len(records) < 2:
        raise RuntimeError("not enough saved RGB + pose records")

    first = cv2.imread(str(records[0]["rgb_path"]))
    if first is None:
        raise RuntimeError("could not read first RGB frame")
    h, w = first.shape[:2]
    dims = np.asarray(args.box_dims, dtype=np.float64)
    pts = base.box_corners(dims)

    if args.support_record_index >= 0:
        support_idx = args.support_record_index
    else:
        support_idx = choose_support(
            records, K, dims, args.box_height, args.preplace_clearance,
            w, h, args.top_axis
        )
    if not (0 <= support_idx < len(records)):
        raise IndexError(f"support index out of range: {support_idx}")

    T_support = records[support_idx]["pose"].copy()
    a_obj, a_cam, up_score, T_B, T_pre = geometry(
        T_support, K, args.box_height, args.preplace_clearance, args.top_axis
    )
    top_label = axis_name(a_obj)

    out_dir = bundle / "stack_target_rgb_overlay_v2"
    out_dir.mkdir(exist_ok=True)
    out_video = out_dir / "stack_target_rgb_overlay_v2.mp4"

    writer = cv2.VideoWriter(
        str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h)
    )
    if not writer.isOpened():
        raise RuntimeError("could not open video writer")

    indices = list(range(0, len(records), max(1, args.every)))
    snap = {
        indices[0]: "start",
        indices[len(indices)//2]: "mid",
        indices[-1]: "end",
    }

    written = 0
    for ridx in indices:
        img = cv2.imread(str(records[ridx]["rgb_path"]))
        if img is None:
            continue
        vis = render_frame(
            img, records[ridx]["pose"], T_support, T_B, T_pre,
            K, pts, ridx, len(records)-1, top_label
        )
        writer.write(vis)
        written += 1
        if ridx in snap:
            cv2.imwrite(str(out_dir / f"stack_target_preview_{snap[ridx]}.png"), vis)

    writer.release()

    summary = {
        "mode": "corrected_on_top_v2",
        "support_record_index": int(support_idx),
        "support_frame_id": records[support_idx]["frame_id"],
        "top_axis_object_frame": top_label,
        "top_axis_camera_vector": [float(x) for x in a_cam],
        "top_axis_image_up_score": float(up_score),
        "support_center_camera_m": [float(x) for x in T_support[:3, 3]],
        "desired_final_B_center_camera_m": [float(x) for x in T_B[:3, 3]],
        "preplace_center_camera_m": [float(x) for x in T_pre[:3, 3]],
        "stack_center_distance_m": float(args.box_height),
        "preplace_clearance_m": float(args.preplace_clearance),
        "written_frames": int(written),
        "note": (
            "B is shifted along the upward-facing short-axis/top-face normal. "
            "It is not generated using a blind local +Z offset."
        ),
    }
    (out_dir / "stack_target_rgb_overlay_v2_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print("CORRECTED REAL-RGB STACK OVERLAY V2")
    print("support record:", support_idx, "frame:", records[support_idx]["frame_id"])
    print("detected top axis:", top_label, "image-up score:", round(up_score, 3))
    print("support center:", np.round(T_support[:3, 3], 4))
    print("desired final B:", np.round(T_B[:3, 3], 4))
    print("pre-place:", np.round(T_pre[:3, 3], 4))
    print("written:", written)
    print("saved:", out_video)


if __name__ == "__main__":
    main()
