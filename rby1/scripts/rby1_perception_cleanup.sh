#!/usr/bin/env bash
set -u

BRIDGE="${RBY1_FOUNDATIONPOSE_BRIDGE_ROOT:-${TMPDIR:-/tmp}/rby1_foundationpose_offline_bridge_${UID}}"
PID_FILE="$BRIDGE/offline_test_processes.pid"

process_start_time() {
    local stat_line rest
    [[ -r "/proc/$1/stat" ]] || return 1
    IFS= read -r stat_line < "/proc/$1/stat" || return 1
    rest="${stat_line##*) }"
    set -- $rest
    [[ "$#" -ge 20 ]] || return 1
    printf '%s\n' "${20}"
}

record_matches() {
    local pid="$1" expected_start="$2" script_path="$3"
    local current_start command_line

    [[ "$pid" =~ ^[0-9]+$ && "$expected_start" =~ ^[0-9]+$ ]] || return 1
    case "$script_path" in
        */foundationpose_rgbd_bridge.py|*/tf_pose_bridge.py|*/fake_rgbd_publisher.py|*/fake_base_ee_tf.py)
            ;;
        *)
            return 1
            ;;
    esac

    current_start="$(process_start_time "$pid" 2>/dev/null)" || return 1
    [[ "$current_start" == "$expected_start" ]] || return 1
    command_line="$(ps -p "$pid" -o args= 2>/dev/null)" || return 1
    [[ "$command_line" == *"$script_path"* ]]
}

if [[ -f "$PID_FILE" ]]; then
    while IFS=$'\t' read -r pid expected_start script_path; do
        if record_matches "$pid" "$expected_start" "$script_path"; then
            kill "$pid" 2>/dev/null || true
        fi
    done < "$PID_FILE"

    for _ in $(seq 1 20); do
        remaining=0
        while IFS=$'\t' read -r pid expected_start script_path; do
            if record_matches "$pid" "$expected_start" "$script_path"; then
                remaining=1
            fi
        done < "$PID_FILE"
        [[ "$remaining" == "0" ]] && break
        sleep 0.05
    done

    failed=0
    while IFS=$'\t' read -r pid expected_start script_path; do
        if record_matches "$pid" "$expected_start" "$script_path"; then
            command_line="$(ps -p "$pid" -o args= 2>/dev/null || true)"
            echo "ERROR: owned perception test process remains: $pid $command_line" >&2
            failed=1
        fi
    done < "$PID_FILE"
    rm -f "$PID_FILE"
    [[ "$failed" == "0" ]] || exit 1
fi

echo "Perception cleanup: CLEAN"
