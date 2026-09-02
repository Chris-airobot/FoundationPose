# RBY1 right D405 live aligned ROS2 publisher

Verified on the Nvidia U-PC on 2026-09-03.

This node opens right D405 serial `230422272237`, enables 640x480 RGB + Z16 depth, aligns depth into the color geometry with `rs2::align(RS2_STREAM_COLOR)`, and publishes:

```text
/right_d405/color/image_raw
/right_d405/aligned_depth_to_color/image_raw
/right_d405/color/camera_info
```

Observed rate during the first test was about 15 Hz for both image streams.

## Important camera ownership rule

Stop only the right Iris service before starting this node:

```bash
sudo systemctl stop rs-right-iris.service
```

Restore it after custom-camera work:

```bash
sudo systemctl start rs-right-iris.service
```

## Build on Nvidia U-PC

Copy or symlink this package into a ROS2 workspace, for example:

```bash
mkdir -p ~/right_d405_live_ws/src
cp -r right_d405_live_ros2 ~/right_d405_live_ws/src/right_d405_live
```

The Nvidia machine's Conda installation can hijack the Python used by `ament_cmake`. Build with system Python:

```bash
conda deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:$PATH

source /opt/ros/humble/setup.bash
cd ~/right_d405_live_ws
rm -rf build install log

colcon build --symlink-install \
  --cmake-args \
  -DPYTHON_EXECUTABLE=/usr/bin/python3 \
  -DPython3_EXECUTABLE=/usr/bin/python3
```

The package deliberately links the working local RealSense installation:

```text
/usr/local/include
/usr/local/lib/librealsense2.so
```

rather than the ROS Humble librealsense package that failed to detect this D405 during the first integration session.

## Run

```bash
source /opt/ros/humble/setup.bash
source ~/right_d405_live_ws/install/setup.bash

export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH:-}

ros2 run right_d405_live right_d405_live_node
```

Successful startup looked like:

```text
Opening right D405 serial 230422272237
Color intrinsics: fx=434.641174 fy=434.212646 cx=323.722992 cy=241.269516
Depth scale: 0.001000000 m/unit
Publishing:
  /right_d405/color/image_raw
  /right_d405/aligned_depth_to_color/image_raw
  /right_d405/color/camera_info
```

See `../../STATUS_2026-09-03.md` before integrating FoundationPose, especially the documented runtime-intrinsics discrepancy.
