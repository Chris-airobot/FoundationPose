#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="${FOUNDATIONPOSE_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
PLACEMENT_ROOT="${PLACEMENT_REPO:-$(dirname "$ROOT")/placement-generalization-execution-aware}"
BRIDGE_SOURCE="$PLACEMENT_ROOT/deployment/rby1/perception/foundationpose_bridge"
BRIDGE="${RBY1_FOUNDATIONPOSE_BRIDGE_ROOT:-${TMPDIR:-/tmp}/rby1_foundationpose_offline_bridge_${UID}}"
PID_FILE="$BRIDGE/offline_test_processes.pid"
LOG_DIR="$BRIDGE/offline_test_logs"
DOMAIN="${RBY1_PERCEPTION_TEST_DOMAIN_ID:-47}"
PYTHON="${FOUNDATIONPOSE_PYTHON:-$(command -v python3)}"
ROS_PYTHON="${RBY1_ROS_PYTHON:-/usr/bin/python3}"
WEIGHTS_SOURCE="${FOUNDATIONPOSE_WEIGHTS_SOURCE:-}"
WEIGHTS_LINK="$ROOT/weights"
MYCPP_BUILD_SOURCE="${FOUNDATIONPOSE_MYCPP_BUILD_SOURCE:-}"
MYCPP_BUILD_LINK="$ROOT/mycpp/build"
OWNED_PIDS=()
OWNED_WEIGHTS_LINK=0
OWNED_MYCPP_BUILD_LINK=0

cleanup() {
    local pid status=0
    RBY1_FOUNDATIONPOSE_BRIDGE_ROOT="$BRIDGE" \
        "$ROOT/rby1/scripts/rby1_perception_cleanup.sh" >/dev/null 2>&1 || status=$?
    for pid in "${OWNED_PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    OWNED_PIDS=()
    if [[ "$OWNED_WEIGHTS_LINK" == "1" && -L "$WEIGHTS_LINK" ]]; then
        rm -f "$WEIGHTS_LINK"
        OWNED_WEIGHTS_LINK=0
    fi
    if [[ "$OWNED_MYCPP_BUILD_LINK" == "1" && -L "$MYCPP_BUILD_LINK" ]]; then
        rm -f "$MYCPP_BUILD_LINK"
        OWNED_MYCPP_BUILD_LINK=0
    fi
    return "$status"
}

cleanup
trap 'cleanup || true' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

[[ -x "$PYTHON" ]] || {
    echo "ERROR: FoundationPose Python is not executable: $PYTHON" >&2
    echo "Set FOUNDATIONPOSE_PYTHON to the validated environment interpreter." >&2
    exit 1
}
[[ -x "$ROS_PYTHON" ]] || {
    echo "ERROR: ROS Python is not executable: $ROS_PYTHON" >&2
    exit 1
}
[[ -f /opt/ros/humble/setup.bash ]] || {
    echo "ERROR: host ROS Humble setup is unavailable" >&2
    exit 1
}

if [[ ! -f "$WEIGHTS_LINK/2024-01-11-20-02-45/config.yml" \
      || ! -f "$WEIGHTS_LINK/2023-10-28-18-33-37/config.yml" ]]; then
    [[ ! -e "$WEIGHTS_LINK" && ! -L "$WEIGHTS_LINK" ]] || {
        echo "ERROR: incomplete FoundationPose weights at $WEIGHTS_LINK" >&2
        exit 1
    }
    [[ -f "$WEIGHTS_SOURCE/2024-01-11-20-02-45/config.yml" \
       && -f "$WEIGHTS_SOURCE/2023-10-28-18-33-37/config.yml" ]] || {
        echo "ERROR: complete FoundationPose weights are unavailable" >&2
        echo "Set FOUNDATIONPOSE_WEIGHTS_SOURCE to an existing weight directory." >&2
        exit 1
    }
    ln -s "$WEIGHTS_SOURCE" "$WEIGHTS_LINK"
    OWNED_WEIGHTS_LINK=1
fi

if ! compgen -G "$MYCPP_BUILD_LINK/mycpp*.so" >/dev/null; then
    [[ ! -e "$MYCPP_BUILD_LINK" && ! -L "$MYCPP_BUILD_LINK" ]] || {
        echo "ERROR: incomplete FoundationPose mycpp build at $MYCPP_BUILD_LINK" >&2
        exit 1
    }
    compgen -G "$MYCPP_BUILD_SOURCE/mycpp*.so" >/dev/null || {
        echo "ERROR: compiled FoundationPose mycpp extension is unavailable" >&2
        echo "Set FOUNDATIONPOSE_MYCPP_BUILD_SOURCE to a compatible build directory." >&2
        exit 1
    }
    ln -s "$MYCPP_BUILD_SOURCE" "$MYCPP_BUILD_LINK"
    OWNED_MYCPP_BUILD_LINK=1
fi

for script in \
    foundationpose_rgbd_bridge.py \
    tf_pose_bridge.py \
    fake_rgbd_publisher.py \
    fake_base_ee_tf.py; do
    [[ -f "$BRIDGE_SOURCE/$script" ]] || {
        echo "ERROR: missing versioned bridge source: $BRIDGE_SOURCE/$script" >&2
        exit 1
    }
done

mkdir -p "$BRIDGE/latest" "$LOG_DIR"
rm -f \
    "$BRIDGE/base_T_ee_right.json" \
    "$BRIDGE/latest/rgb.png" \
    "$BRIDGE/latest/depth_u16.png" \
    "$BRIDGE/latest/K.txt" \
    "$BRIDGE/latest/metadata.json" \
    "$BRIDGE/latest/READY" \
    "$LOG_DIR/foundationpose_rgbd_bridge.log" \
    "$LOG_DIR/fake_rgbd_publisher.log" \
    "$LOG_DIR/foundationpose_tf_bridge.log" \
    "$LOG_DIR/fake_base_ee_tf.log"

set +u
source /opt/ros/humble/setup.bash
set -u
export ROS_DOMAIN_ID="$DOMAIN"
export ROS_LOCALHOST_ONLY=1
export RBY1_FOUNDATIONPOSE_BRIDGE_ROOT="$BRIDGE"
export RBY1_FOUNDATIONPOSE_BRIDGE_OUTPUT="$BRIDGE/latest"

launch_owned() {
    local log_name="$1"
    local pid start_time script_path
    shift
    script_path="$1"
    "$ROS_PYTHON" "$@" >"$LOG_DIR/$log_name" 2>&1 &
    pid="$!"
    OWNED_PIDS+=("$pid")
    start_time="$(process_start_time "$pid")"
    printf '%s\t%s\t%s\n' "$pid" "$start_time" "$script_path" >> "$PID_FILE"
}

process_start_time() {
    local stat_line rest
    IFS= read -r stat_line < "/proc/$1/stat"
    rest="${stat_line##*) }"
    set -- $rest
    printf '%s\n' "${20}"
}

launch_owned foundationpose_rgbd_bridge.log \
    "$BRIDGE_SOURCE/foundationpose_rgbd_bridge.py"
launch_owned fake_rgbd_publisher.log \
    "$BRIDGE_SOURCE/fake_rgbd_publisher.py"
launch_owned foundationpose_tf_bridge.log \
    "$BRIDGE_SOURCE/tf_pose_bridge.py" \
    --base-frame base \
    --ee-frame ee_right \
    --output "$BRIDGE/base_T_ee_right.json"
launch_owned fake_base_ee_tf.log \
    "$BRIDGE_SOURCE/fake_base_ee_tf.py"

ready=0
for _ in $(seq 1 50); do
    if [[ -f "$BRIDGE/latest/metadata.json" \
          && -f "$BRIDGE/latest/READY" \
          && -f "$BRIDGE/base_T_ee_right.json" ]]; then
        ready=1
        break
    fi
    sleep 0.2
done

if [[ "$ready" != "1" ]]; then
    echo "ERROR: fake perception inputs did not become ready." >&2
    for log in \
        foundationpose_rgbd_bridge.log \
        fake_rgbd_publisher.log \
        foundationpose_tf_bridge.log \
        fake_base_ee_tf.log; do
        echo "===== $log =====" >&2
        tail -30 "$LOG_DIR/$log" >&2 || true
    done
    exit 1
fi

cd "$ROOT"
"$PYTHON" -u rby1/scripts/run_rby1_perception_once.py --fake-depth-mask

cleanup
trap - EXIT INT TERM HUP

echo "RBY1 PERCEPTION: PASS"
echo
echo "===================================="
echo " OFFLINE PIPELINE TEST COMPLETE"
echo "===================================="
echo
echo "Final output:"
echo "$ROOT/rby1/runtime/latest/object_pose.json"
echo
echo "Background-process check:"
echo "CLEAN: no owned RBY1 perception test processes remain."
echo
echo "Final object pose:"
cat "$ROOT/rby1/runtime/latest/object_pose.json"
