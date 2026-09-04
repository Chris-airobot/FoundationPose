#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

BRIDGE_ROOT = Path(
    os.environ.get(
        "RBY1_FOUNDATIONPOSE_BRIDGE_ROOT",
        Path.home() / "rby1_ros2_ws/foundationpose_bridge",
    )
).expanduser().resolve()

BRIDGE_RGBD = BRIDGE_ROOT / "latest"
ROBOT_TF = BRIDGE_ROOT / "base_T_ee_right.json"

LIVE_INPUT = ROOT / "rby1" / "live_input"

PLACEMENT_ROOT = Path(
    os.environ.get(
        "PLACEMENT_REPO",
        ROOT.parent / "placement-generalization-execution-aware",
    )
).expanduser().resolve()

HAND_EYE = (
    PLACEMENT_ROOT
    / "deployment"
    / "rby1"
    / "config"
    / "right_d405_handeye.yaml"
)

DEFAULT_MESH = (
    PLACEMENT_ROOT
    / "assets"
    / "real_robot_meshes"
    / "061_foam_brick_final30_exact_mm.obj"
)


VERBOSE = False


def run(cmd):
    if VERBOSE:
        print()
        print(">", " ".join(str(x) for x in cmd))
        subprocess.run(cmd, check=True)
        return

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if result.returncode != 0:
        print()
        print("===== SUBPROCESS FAILED =====")
        print(">", " ".join(str(x) for x in cmd))
        print(result.stdout)
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
        )


def check_fresh(path, max_age):
    if not path.exists():
        raise RuntimeError(
            f"Missing required live input: {path}"
        )

    age = time.time() - path.stat().st_mtime

    if age > max_age:
        raise RuntimeError(
            f"Stale input: {path}\n"
            f"age={age:.1f}s, allowed={max_age:.1f}s"
        )

    return age


def matrix_to_quaternion_xyzw(R):
    R = np.asarray(R, dtype=np.float64)

    trace = np.trace(R)

    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s

    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(
            1.0 + R[0, 0]
            - R[1, 1]
            - R[2, 2]
        ) * 2

        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s

    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(
            1.0 + R[1, 1]
            - R[0, 0]
            - R[2, 2]
        ) * 2

        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s

    else:
        s = np.sqrt(
            1.0 + R[2, 2]
            - R[0, 0]
            - R[1, 1]
        ) * 2

        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array(
        [x, y, z, w],
        dtype=np.float64,
    )

    q /= np.linalg.norm(q)

    return q


def atomic_json(path, data):
    tmp = path.with_suffix(".json.tmp")

    tmp.write_text(
        json.dumps(data, indent=2) + "\n"
    )

    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--object-id",
        default="061_foam_brick",
    )

    p.add_argument(
        "--mesh",
        type=Path,
        default=DEFAULT_MESH,
    )

    p.add_argument(
        "--mesh-scale",
        type=float,
        default=0.001,
    )

    mask_source = p.add_mutually_exclusive_group(required=True)

    mask_source.add_argument(
        "--mask",
        type=Path,
        default=None,
    )

    mask_source.add_argument(
        "--automatic-mask",
        action="store_true",
        help="Use the repository's depth/table-plane first-frame mask.",
    )

    mask_source.add_argument(
        "--fake-depth-mask",
        action="store_true",
        help="OFFLINE TEST ONLY",
    )

    p.add_argument(
        "--register-iters",
        type=int,
        default=5,
    )

    p.add_argument(
        "--max-input-age-sec",
        type=float,
        default=10.0,
    )

    p.add_argument(
        "--verbose",
        action="store_true",
        help="Show FoundationPose and subprocess debug output.",
    )

    args = p.parse_args()

    global VERBOSE
    VERBOSE = args.verbose

    runtime = ROOT / "rby1" / "runtime" / "latest"
    fp_dir = runtime / "foundationpose"
    compose_dir = runtime / "compose"

    # --------------------------------------------------------
    # 1. Make sure ROS-side inputs are alive/fresh
    # --------------------------------------------------------
    rgbd_meta = BRIDGE_RGBD / "metadata.json"

    rgbd_age = check_fresh(
        rgbd_meta,
        args.max_input_age_sec,
    )

    tf_age = check_fresh(
        ROBOT_TF,
        args.max_input_age_sec,
    )

    if not args.mesh.exists():
        raise RuntimeError(
            f"Mesh does not exist: {args.mesh}"
        )

    print("====================================")
    print(" RBY1 PERCEPTION PIPELINE")
    print("====================================")
    print("object:", args.object_id)
    print("RGB-D age:", f"{rgbd_age:.2f}s")
    print("robot TF age:", f"{tf_age:.2f}s")

    # --------------------------------------------------------
    # 2. Freeze one synchronized RGB-D frame
    # --------------------------------------------------------
    run([
        sys.executable,
        "-u",
        str(
            ROOT
            / "rby1"
            / "scripts"
            / "read_rby1_rgbd_bridge.py"
        ),
        "--source",
        str(BRIDGE_RGBD),
        "--destination",
        str(LIVE_INPUT),
    ])

    # --------------------------------------------------------
    # 3. FoundationPose
    # --------------------------------------------------------
    shutil.rmtree(
        fp_dir,
        ignore_errors=True,
    )

    fp_cmd = [
        sys.executable,
        "-u",
        str(
            ROOT
            / "rby1"
            / "scripts"
            / "run_rby1_foundationpose.py"
        ),
        "--input-dir",
        str(LIVE_INPUT),
        "--mesh",
        str(args.mesh),
        "--mesh-scale",
        str(args.mesh_scale),
        "--output-dir",
        str(fp_dir),
        "--register-iters",
        str(args.register_iters),
    ]

    if args.fake_depth_mask:
        fp_cmd += [
            "--fake-depth-mask",
            "--fake-depth-min",
            "0.70",
            "--fake-depth-max",
            "0.80",
        ]

        mode = "offline_fake"

    elif args.automatic_mask:
        fp_cmd += ["--automatic-mask"]
        mode = "real_automatic_mask"

    else:
        if not args.mask.exists():
            raise RuntimeError(f"Object mask does not exist: {args.mask}")
        fp_cmd += ["--mask", str(args.mask)]
        mode = "real_manual_mask"

    run(fp_cmd)

    # --------------------------------------------------------
    # 4. Compose:
    #
    # base_T_object =
    #   base_T_ee_right
    #   @ ee_right_T_camera
    #   @ camera_T_object
    # --------------------------------------------------------
    shutil.rmtree(
        compose_dir,
        ignore_errors=True,
    )

    run([
        sys.executable,
        "-u",
        str(
            ROOT
            / "rby1"
            / "scripts"
            / "compose_rby1_object_pose.py"
        ),
        "--handeye",
        str(HAND_EYE),
        "--robot-tf",
        str(ROBOT_TF),
        "--foundationpose-result",
        str(fp_dir / "result.json"),
        "--output-dir",
        str(compose_dir),
    ])

    # --------------------------------------------------------
    # 5. Create ONE simple final output contract
    # --------------------------------------------------------
    composed = json.loads(
        (
            compose_dir
            / "base_object_result.json"
        ).read_text()
    )

    fp = json.loads(
        (
            fp_dir
            / "result.json"
        ).read_text()
    )

    robot_tf = json.loads(
        ROBOT_TF.read_text()
    )

    T = np.asarray(
        composed["base_T_object"],
        dtype=np.float64,
    )

    q = matrix_to_quaternion_xyzw(
        T[:3, :3]
    )

    if not np.all(np.isfinite(T)):
        raise RuntimeError(
            "Non-finite base_T_object"
        )

    det = np.linalg.det(
        T[:3, :3]
    )

    if abs(det - 1.0) > 1e-3:
        raise RuntimeError(
            f"Invalid rotation determinant: {det}"
        )

    final = {
        "valid": True,
        "mode": mode,

        "object_id":
            args.object_id,

        "frame_id":
            composed["base_frame"],

        "object_frame":
            composed["object_frame"],

        "position_m":
            T[:3, 3].tolist(),

        "quaternion_xyzw":
            q.tolist(),

        "base_T_object":
            T.tolist(),

        "camera_frame":
            composed["camera_frame"],

        "camera_timestamp_ros":
            fp.get(
                "input_timestamp_ros"
            ),

        "robot_tf_timestamp_ros":
            robot_tf.get(
                "timestamp_ros"
            ),

        "mesh":
            str(args.mesh),

        "foundationpose_overlay":
            str(
                fp_dir
                / "overlay.png"
            ),
    }

    final_path = (
        runtime
        / "object_pose.json"
    )

    atomic_json(
        final_path,
        final,
    )

    print()
    print("====================================")
    print(" RBY1 PERCEPTION: PASS")
    print("====================================")
    print("object:", final["object_id"])
    print("frame:", final["frame_id"])
    print(
        "position [m]:",
        final["position_m"],
    )
    print(
        "quaternion xyzw:",
        final["quaternion_xyzw"],
    )
    print()
    print("FINAL OUTPUT:")
    print(final_path)


if __name__ == "__main__":
    main()
