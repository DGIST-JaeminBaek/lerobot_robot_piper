#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${ENV_FILE:-${REPO_DIR}/configs/recording.env}"

load_recording_env() {
  if [[ -f "${ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
  else
    echo "[WARN] Missing env file: ${ENV_FILE}" >&2
    echo "[WARN] Copy configs/recording.env.example to configs/recording.env for persistent settings." >&2
  fi
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[ERROR] Required command not found: $1" >&2
    return 1
  fi
}

bool_default() {
  local value="${1:-}"
  local fallback="$2"
  if [[ -n "${value}" ]]; then
    printf '%s' "${value}"
  else
    printf '%s' "${fallback}"
  fi
}

robot_camera_args() {
  # LeRobot 0.4.4 dict 파서 우회용 카메라 인자
  local camera_type="${CAMERA_TYPE:-opencv}"
  local top_cam_type="${TOP_CAM_TYPE:-${camera_type}}"
  local wrist_cam_type="${WRIST_CAM_TYPE:-${camera_type}}"
  local top_cam="${TOP_CAM-0}"
  local wrist_cam="${WRIST_CAM-1}"
  local width="${CAM_WIDTH:-640}"
  local height="${CAM_HEIGHT:-480}"
  local fps="${FPS:-30}"
  local realsense_use_depth="${REALSENSE_USE_DEPTH:-false}"
  local realsense_warmup_s="${REALSENSE_WARMUP_S:-5.0}"
  local camera_connect_warmup="${CAMERA_CONNECT_WARMUP:-false}"
  local camera_post_connect_wait_s="${CAMERA_POST_CONNECT_WAIT_S:-2.0}"

  printf '%s\n' \
    "--robot.camera_type=${camera_type}" \
    "--robot.top_cam_type=${top_cam_type}" \
    "--robot.wrist_cam_type=${wrist_cam_type}" \
    "--robot.top_cam=${top_cam}" \
    "--robot.wrist_cam=${wrist_cam}" \
    "--robot.cam_width=${width}" \
    "--robot.cam_height=${height}" \
    "--robot.camera_fps=${fps}" \
    "--robot.realsense_use_depth=${realsense_use_depth}" \
    "--robot.realsense_warmup_s=${realsense_warmup_s}" \
    "--robot.camera_connect_warmup=${camera_connect_warmup}" \
    "--robot.camera_post_connect_wait_s=${camera_post_connect_wait_s}" \
    "--robot.top_realsense_use_depth=${TOP_REALSENSE_USE_DEPTH:-${realsense_use_depth}}" \
    "--robot.wrist_realsense_use_depth=${WRIST_REALSENSE_USE_DEPTH:-${realsense_use_depth}}"
}

robot_action_offset_args() {
  # leader/follower 시작 자세 차이 보정 인자
  printf '%s\n' \
    "--robot.park_on_connect=$(bool_default "${PARK_ON_CONNECT:-}" false)" \
    "--robot.use_action_offset=$(bool_default "${USE_ACTION_OFFSET:-}" true)" \
    "--robot.use_manual_action_offset=$(bool_default "${USE_MANUAL_ACTION_OFFSET:-}" false)" \
    "--robot.action_offset_report_threshold=${ACTION_OFFSET_REPORT_THRESHOLD:-3.0}" \
    "--robot.action_offset_joint1=${ACTION_OFFSET_JOINT1:-0.0}" \
    "--robot.action_offset_joint2=${ACTION_OFFSET_JOINT2:-0.0}" \
    "--robot.action_offset_joint3=${ACTION_OFFSET_JOINT3:-0.0}" \
    "--robot.action_offset_joint4=${ACTION_OFFSET_JOINT4:-0.0}" \
    "--robot.action_offset_joint5=${ACTION_OFFSET_JOINT5:-0.0}" \
    "--robot.action_offset_joint6=${ACTION_OFFSET_JOINT6:-0.0}" \
    "--robot.action_offset_gripper=${ACTION_OFFSET_GRIPPER:-0.0}"
}

print_command() {
  printf '[CMD]'
  for part in "$@"; do
    printf ' %q' "${part}"
  done
  printf '\n'
}

run_or_print() {
  print_command "$@"
  if [[ "${DRY_RUN:-false}" == "true" ]]; then
    return 0
  fi
  "$@"
}

robot_safety_args() {
  # DISABLE_TORQUE_ON_DISCONNECT=false로 두면 disconnect() 시 parking 자세로는
  # 이동하되 torque는 자동으로 풀지 않음 — scripts/tools/safe_release_torque.py로
  # 사람이 팔을 잡은 상태에서 수동으로 torque를 해제하는 루틴과 짝을 이룸.
  printf '%s\n' \
    "--robot.max_relative_target=${MAX_RELATIVE_TARGET:-5.0}" \
    "--robot.disable_torque_on_disconnect=$(bool_default "${DISABLE_TORQUE_ON_DISCONNECT:-}" true)"
}

plugin_discovery_args() {
  printf '%s\n' \
    "--robot.discover_packages_path=lerobot_robot_piper" \
    "--teleop.discover_packages_path=lerobot_robot_piper"
}
