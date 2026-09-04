#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from estimater import *
from Utils import *
from automatic_object_mask import generate_object_mask, save_mask_debug


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing rgb.png, depth_u16.png, metadata.json",
    )

    p.add_argument(
        "--mesh",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--mesh-scale",
        type=float,
        default=1.0,
        help="Scale applied to mesh vertices. Use 0.001 for mm -> m.",
    )

    mask_source = p.add_mutually_exclusive_group(required=True)

    mask_source.add_argument(
        "--mask",
        type=Path,
        default=None,
        help="Binary mask image.",
    )

    mask_source.add_argument(
        "--automatic-mask",
        action="store_true",
        help="Generate a first-frame mask with the existing depth/table-plane mask utility.",
    )

    mask_source.add_argument(
        "--fake-depth-mask",
        action="store_true",
        help="TEST ONLY: create mask from fake depth range.",
    )

    p.add_argument(
        "--fake-depth-min",
        type=float,
        default=0.70,
    )

    p.add_argument(
        "--fake-depth-max",
        type=float,
        default=0.80,
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--register-iters",
        type=int,
        default=5,
    )

    return p.parse_args()


def load_input(input_dir):
    rgb_path = input_dir / "rgb.png"
    depth_path = input_dir / "depth_u16.png"
    meta_path = input_dir / "metadata.json"

    bgr = cv2.imread(
        str(rgb_path),
        cv2.IMREAD_COLOR,
    )

    if bgr is None:
        raise RuntimeError(
            f"Cannot read RGB: {rgb_path}"
        )

    depth_raw = cv2.imread(
        str(depth_path),
        cv2.IMREAD_UNCHANGED,
    )

    if depth_raw is None:
        raise RuntimeError(
            f"Cannot read depth: {depth_path}"
        )

    if depth_raw.dtype != np.uint16:
        raise RuntimeError(
            f"Expected uint16 depth, got {depth_raw.dtype}"
        )

    meta = json.loads(
        meta_path.read_text()
    )

    if not meta.get(
        "depth_aligned_to_color",
        False,
    ):
        raise RuntimeError(
            "Depth is not marked aligned to color"
        )

    if meta.get("frame_id") != "right_d405_optical_frame":
        raise RuntimeError(
            f"Unexpected camera frame: {meta.get('frame_id')}"
        )

    scale = float(
        meta["depth_scale_m_per_unit"]
    )
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError(
            f"Invalid depth scale: {scale}"
        )

    depth_m = (
        depth_raw.astype(np.float32)
        * scale
    )

    depth_m[
        (depth_m < 0.05)
        | (depth_m > 2.0)
    ] = 0.0

    K = np.array(
        [
            [
                float(meta["fx"]),
                0.0,
                float(meta["cx"]),
            ],
            [
                0.0,
                float(meta["fy"]),
                float(meta["cy"]),
            ],
            [
                0.0,
                0.0,
                1.0,
            ],
        ],
        dtype=np.float64,
    )

    rgb = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB,
    )

    return (
        bgr,
        rgb,
        depth_raw,
        depth_m,
        K,
        meta,
    )


def load_mask(args, depth_m, bgr, K):
    if args.fake_depth_mask:
        mask = (
            (depth_m >= args.fake_depth_min)
            &
            (depth_m <= args.fake_depth_max)
        ).astype(np.uint8)

        print(
            "MASK SOURCE: fake depth range "
            f"{args.fake_depth_min:.3f}-"
            f"{args.fake_depth_max:.3f} m"
        )

    elif args.automatic_mask:
        raw = generate_object_mask(depth_m, K)
        save_mask_debug(
            args.output_dir / "automatic_mask_debug.png",
            bgr,
            raw,
        )
        mask = (raw > 0).astype(np.uint8)
        print("MASK SOURCE: automatic_object_mask.py")

    else:
        if args.mask is None:
            raise RuntimeError(
                "Provide --mask for real data, "
                "or --fake-depth-mask for the offline fake test."
            )

        raw = cv2.imread(
            str(args.mask),
            cv2.IMREAD_GRAYSCALE,
        )

        if raw is None:
            raise RuntimeError(
                f"Cannot read mask: {args.mask}"
            )

        mask = (raw > 0).astype(np.uint8)

        print(
            "MASK SOURCE:",
            args.mask,
        )

    pixels = int(
        np.count_nonzero(mask)
    )

    if pixels < 20:
        raise RuntimeError(
            f"Mask too small: {pixels} pixels"
        )

    return mask


def main():
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        bgr,
        rgb,
        depth_raw,
        depth_m,
        K,
        meta,
    ) = load_input(args.input_dir)

    mask = load_mask(
        args,
        depth_m,
        bgr,
        K,
    )

    cv2.imwrite(
        str(args.output_dir / "mask.png"),
        mask * 255,
    )

    # --------------------------------------------------------
    # Mesh
    # --------------------------------------------------------
    mesh = trimesh.load(
        str(args.mesh),
        force="mesh",
    )

    mesh.apply_scale(
        args.mesh_scale
    )

    if len(mesh.vertices) == 0:
        raise RuntimeError(
            "Mesh contains no vertices"
        )

    to_origin, extents = (
        trimesh.bounds.oriented_bounds(mesh)
    )

    center_tf = np.linalg.inv(
        to_origin
    )

    bbox = np.stack(
        [
            -extents / 2.0,
            extents / 2.0,
        ],
        axis=0,
    ).reshape(2, 3)

    print()
    print("===== INPUT CHECK =====")
    print("RGB:", rgb.shape)
    print(
        "Depth valid:",
        int(np.count_nonzero(depth_m)),
    )
    print("K:")
    print(K)
    print(
        "Camera frame:",
        meta["frame_id"],
    )
    print(
        "Mesh:",
        args.mesh,
    )
    print(
        "Mesh scale:",
        args.mesh_scale,
    )
    print(
        "Mesh extents [m]:",
        extents,
    )
    print(
        "Mask pixels:",
        int(mask.sum()),
    )

    # --------------------------------------------------------
    # FoundationPose
    # --------------------------------------------------------
    set_logging_format()
    set_seed(0)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()

    glctx = dr.RasterizeCudaContext()

    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=str(args.output_dir),
        debug=0,
        glctx=glctx,
    )

    print()
    print("Running FoundationPose registration...")

    pose = est.register(
        K=K,
        rgb=rgb,
        depth=depth_m,
        ob_mask=mask.astype(bool),
        iteration=args.register_iters,
    )

    # pose = object mesh frame -> camera frame
    camera_T_object = pose

    # centered mesh pose, useful for visualization
    camera_T_object_center = (
        pose @ center_tf
    )

    np.savetxt(
        args.output_dir
        / "camera_T_object.txt",
        camera_T_object,
        fmt="%.10f",
    )

    np.savetxt(
        args.output_dir
        / "camera_T_object_center.txt",
        camera_T_object_center,
        fmt="%.10f",
    )

    # --------------------------------------------------------
    # Headless visualization
    # --------------------------------------------------------
    vis = draw_posed_3d_box(
        K,
        img=rgb,
        ob_in_cam=camera_T_object_center,
        bbox=bbox,
    )

    vis = draw_xyz_axis(
        vis,
        ob_in_cam=camera_T_object_center,
        scale=0.05,
        K=K,
        thickness=3,
        transparency=0,
        is_input_rgb=True,
    )

    vis_bgr = cv2.cvtColor(
        vis,
        cv2.COLOR_RGB2BGR,
    )

    cv2.imwrite(
        str(
            args.output_dir
            / "overlay.png"
        ),
        vis_bgr,
    )

    result = {
        "source_frame":
            "object_mesh_frame",

        "target_frame":
            meta["frame_id"],

        "camera_T_object":
            camera_T_object.tolist(),

        "object_center_xyz_camera_m":
            camera_T_object_center[
                :3,
                3,
            ].tolist(),

        "distance_camera_m":
            float(
                np.linalg.norm(
                    camera_T_object_center[
                        :3,
                        3,
                    ]
                )
            ),

        "mesh_path":
            str(args.mesh),

        "mesh_scale":
            args.mesh_scale,

        "mesh_extents_m":
            extents.tolist(),

        "mask_pixels":
            int(mask.sum()),

        "input_timestamp_ros":
            meta.get(
                "timestamp_ros"
            ),

        "input_bridge_frame":
            meta.get(
                "bridge_frame_count"
            ),
    }

    (
        args.output_dir
        / "result.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
        )
        + "\n"
    )

    print()
    print(
        "===================================="
    )
    print(
        " FOUNDATIONPOSE REGISTER: COMPLETE"
    )
    print(
        "===================================="
    )

    print(
        "camera_T_object:"
    )
    print(
        camera_T_object
    )

    print(
        "center xyz camera [m]:",
        result[
            "object_center_xyz_camera_m"
        ],
    )

    print(
        "distance camera [m]:",
        result[
            "distance_camera_m"
        ],
    )

    print()
    print(
        "Outputs:",
        args.output_dir,
    )
    print(
        "  camera_T_object.txt"
    )
    print(
        "  camera_T_object_center.txt"
    )
    print(
        "  mask.png"
    )
    print(
        "  overlay.png"
    )
    print(
        "  result.json"
    )


if __name__ == "__main__":
    main()
