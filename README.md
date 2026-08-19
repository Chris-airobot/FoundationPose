# Samsung G1 FoundationPose Development

Current focus: use the existing FoundationPose box pose as **object A**, define a temporary/mock top-of-stack pose, and compute the desired stacking pose **B** plus a pre-place pose. This step is software-only and sends **no robot commands**.

## Current implemented step: A -> stacking target B

The new script is:

```text
g1/scripts/stack_target_demo.py
```

Inputs:

- `T_robot_box.txt`: current/selected box pose in the G1 robot frame, produced by the existing root pipeline.
- Mock top-box pose: temporary stand-in for the future second FoundationPose detection.
- Box height: `0.30 m` for the current 40 x 30 x 30 cm box.
- Pre-place clearance: default `0.12 m`.

Outputs:

```text
g1/results/stack_target_demo/T_robot_carried_box.txt
g1/results/stack_target_demo/T_robot_top_box.txt
g1/results/stack_target_demo/T_robot_place.txt
g1/results/stack_target_demo/T_robot_preplace.txt
g1/results/stack_target_demo/stack_target_summary.json
```

The placement transform is computed from the top box as:

```text
T_robot_place = T_robot_top_box * Translation(0, 0, box_height)
```

and the pre-place pose is:

```text
T_robot_preplace = T_robot_place * Translation(0, 0, preplace_clearance)
```

The translations are applied in the box/target local frame so the same logic can later use a real FoundationPose estimate of the stack orientation.

## Run this on the Samsung PC

First update the repo:

```bash
cd ~/Chris/FoundationPose
git pull
```

If `g1/results/root_pipeline_demo/T_robot_box.txt` already exists from the previous root-pipeline work, go directly to the stack-target test below.

If it does not exist, regenerate it from the existing latest FoundationPose pose (no new recording):

```bash
cd ~/Chris/FoundationPose
python g1/scripts/offline_root_pipeline.py \
  --box-pose-camera g1/results/live_foundationpose/latest_pose.txt \
  --out g1/results/root_pipeline_demo
```

Then run the new stack-target test:

```bash
cd ~/Chris/FoundationPose
python g1/scripts/stack_target_demo.py \
  --carried-pose g1/results/root_pipeline_demo/T_robot_box.txt \
  --top-xyz 1.20 0.30 0.15 \
  --top-yaw-deg 20 \
  --box-height-m 0.30 \
  --preplace-clearance-m 0.12 \
  --out g1/results/stack_target_demo
```

For this mock stack pose, the important sanity checks should be:

```text
top -> B distance:     0.3000 m
pre-place clearance:   0.1200 m
```

The mock top-box center is at `z = 0.15 m`, so the generated new-box center should be at approximately:

```text
placement B z = 0.45 m
pre-place z   = 0.57 m
```

Inspect the complete result with:

```bash
cat g1/results/stack_target_demo/stack_target_summary.json
```

## What this proves

This verifies the target-generation part of the final pipeline:

```text
existing FoundationPose result
        |
        v
current/carried box A

mock top-of-stack pose
        |
        v
calculate final placement pose B
        |
        v
calculate pre-place pose
```

The next perception step will replace the mock `T_robot_top_box` with the pose of the real detected top/support box from a multi-box RGB-D scene. The target-generation math should stay the same.
