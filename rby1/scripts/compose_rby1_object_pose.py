#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def quat_to_R(q):
    q = np.asarray(q, dtype=np.float64)

    norm = np.linalg.norm(q)

    if norm < 1e-12:
        raise RuntimeError("Invalid zero quaternion")

    q /= norm

    x, y, z, w = q

    return np.array([
        [
            1 - 2*(y*y + z*z),
            2*(x*y - z*w),
            2*(x*z + y*w),
        ],
        [
            2*(x*y + z*w),
            1 - 2*(x*x + z*z),
            2*(y*z - x*w),
        ],
        [
            2*(x*z - y*w),
            2*(y*z + x*w),
            1 - 2*(x*x + y*y),
        ],
    ])


def make_T(t, q):
    T = np.eye(4)

    T[:3, :3] = quat_to_R(q)
    T[:3, 3] = np.asarray(
        t,
        dtype=np.float64,
    )

    return T


def rotation_checks(T):
    R = T[:3, :3]

    return {
        "determinant": float(
            np.linalg.det(R)
        ),
        "orthonormal_error": float(
            np.linalg.norm(
                R.T @ R - np.eye(3)
            )
        ),
    }


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--handeye",
        required=True,
    )

    p.add_argument(
        "--robot-tf",
        required=True,
    )

    p.add_argument(
        "--foundationpose-result",
        required=True,
    )

    p.add_argument(
        "--output-dir",
        required=True,
    )

    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # base_T_ee_right
    # --------------------------------------------------------
    robot = json.loads(
        Path(args.robot_tf).read_text()
    )

    base_T_ee = np.asarray(
        robot["base_T_ee_right"],
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # ee_right_T_camera
    # --------------------------------------------------------
    handeye = yaml.safe_load(
        Path(args.handeye).read_text()
    )["camera_handeye"]

    if not handeye["calibration"]["accepted"]:
        raise RuntimeError(
            "Hand-eye calibration is not accepted"
        )

    if robot.get("parent_frame") != "base":
        raise RuntimeError(
            f"Expected robot TF parent base, got {robot.get('parent_frame')}"
        )
    if handeye.get("parent_frame") != "ee_right":
        raise RuntimeError(
            f"Expected hand-eye parent ee_right, got {handeye.get('parent_frame')}"
        )
    if handeye.get("child_frame") != "right_d405_optical_frame":
        raise RuntimeError(
            "Expected hand-eye child right_d405_optical_frame, got "
            f"{handeye.get('child_frame')}"
        )

    t = handeye["translation_m"]
    q = handeye["quaternion_xyzw"]

    ee_T_camera = make_T(
        [
            t["x"],
            t["y"],
            t["z"],
        ],
        [
            q["x"],
            q["y"],
            q["z"],
            q["w"],
        ],
    )

    # --------------------------------------------------------
    # camera_T_object from FoundationPose
    # --------------------------------------------------------
    fp = json.loads(
        Path(
            args.foundationpose_result
        ).read_text()
    )

    camera_T_object = np.asarray(
        fp["camera_T_object"],
        dtype=np.float64,
    )

    if fp["target_frame"] != handeye["child_frame"]:
        raise RuntimeError(
            "Frame mismatch: FoundationPose target "
            f"{fp['target_frame']} != "
            f"{handeye['child_frame']}"
        )

    if fp.get("source_frame") != "object_mesh_frame":
        raise RuntimeError(
            f"Expected FoundationPose source object_mesh_frame, got "
            f"{fp.get('source_frame')}"
        )

    if robot["child_frame"] != handeye["parent_frame"]:
        raise RuntimeError(
            "Frame mismatch: robot TF child "
            f"{robot['child_frame']} != "
            f"{handeye['parent_frame']}"
        )

    # --------------------------------------------------------
    # Exact transform chain
    # --------------------------------------------------------
    base_T_camera = (
        base_T_ee
        @ ee_T_camera
    )

    base_T_object = (
        base_T_camera
        @ camera_T_object
    )

    # --------------------------------------------------------
    # Numerical consistency check:
    #
    # Recover camera_T_object from base_T_object.
    # --------------------------------------------------------
    recovered_camera_T_object = (
        np.linalg.inv(base_T_camera)
        @ base_T_object
    )

    chain_error = float(
        np.max(
            np.abs(
                recovered_camera_T_object
                - camera_T_object
            )
        )
    )

    np.savetxt(
        out / "base_T_ee_right.txt",
        base_T_ee,
        fmt="%.10f",
    )

    np.savetxt(
        out / "ee_right_T_camera.txt",
        ee_T_camera,
        fmt="%.10f",
    )

    np.savetxt(
        out / "base_T_camera.txt",
        base_T_camera,
        fmt="%.10f",
    )

    np.savetxt(
        out / "camera_T_object.txt",
        camera_T_object,
        fmt="%.10f",
    )

    np.savetxt(
        out / "base_T_object.txt",
        base_T_object,
        fmt="%.10f",
    )

    result = {
        "base_frame":
            robot["parent_frame"],

        "ee_frame":
            robot["child_frame"],

        "camera_frame":
            handeye["child_frame"],

        "object_frame":
            fp["source_frame"],

        "base_T_ee_right":
            base_T_ee.tolist(),

        "ee_right_T_camera":
            ee_T_camera.tolist(),

        "base_T_camera":
            base_T_camera.tolist(),

        "camera_T_object":
            camera_T_object.tolist(),

        "base_T_object":
            base_T_object.tolist(),

        "object_xyz_base_m":
            base_T_object[
                :3,
                3,
            ].tolist(),

        "checks": {
            "base_T_ee":
                rotation_checks(base_T_ee),

            "ee_T_camera":
                rotation_checks(ee_T_camera),

            "camera_T_object":
                rotation_checks(
                    camera_T_object
                ),

            "base_T_object":
                rotation_checks(
                    base_T_object
                ),

            "chain_reconstruction_max_error":
                chain_error,
        },

    }

    (
        out / "base_object_result.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
        ) + "\n"
    )

    print(
        "===================================="
    )
    print(
        " RBY1 OBJECT POSE COMPOSITION: PASS"
    )
    print(
        "===================================="
    )

    print()
    print(
        "Transform chain:"
    )
    print(
        f"{robot['parent_frame']}"
        f"_T_{robot['child_frame']}"
        " @ "
        f"{handeye['parent_frame']}"
        f"_T_{handeye['child_frame']}"
        " @ camera_T_object"
    )

    print()
    print("base_T_object:")
    print(base_T_object)

    print()
    print(
        "Object xyz in base [m]:",
        base_T_object[:3, 3].tolist(),
    )

    print()
    print(
        "Chain reconstruction max error:",
        chain_error,
    )

    if chain_error > 1e-8:
        raise RuntimeError(
            "Transform-chain consistency check failed"
        )

    print()
    print(
        "Result:",
        out / "base_object_result.json",
    )


if __name__ == "__main__":
    main()
