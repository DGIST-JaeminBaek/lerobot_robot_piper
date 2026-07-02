#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UGRP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PLUGIN_DIR="$(cd "${UGRP_DIR}/.." && pwd)"
ENV_FILE="${ENV_FILE:-${UGRP_DIR}/configs/recording.env}"

load_recording_env() {
  if [[ -f "${ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
  else
    echo "[WARN] Missing env file: ${ENV_FILE}" >&2
    echo "[WARN] Copy ugrp/configs/recording.env.example to ugrp/configs/recording.env for persistent settings." >&2
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

camera_config_arg() {
  local top_cam="${TOP_CAM:-0}"
  local wrist_cam="${WRIST_CAM:-1}"
  local width="${CAM_WIDTH:-640}"
  local height="${CAM_HEIGHT:-480}"
  local fps="${FPS:-30}"

  printf '{ top: {type: opencv, index_or_path: %s, width: %s, height: %s, fps: %s}, wrist: {type: opencv, index_or_path: %s, width: %s, height: %s, fps: %s} }' \
    "${top_cam}" "${width}" "${height}" "${fps}" \
    "${wrist_cam}" "${width}" "${height}" "${fps}"
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

plugin_discovery_args() {
  printf '%s\n' \
    "--robot.discover_packages_path=lerobot_robot_piper" \
    "--teleop.discover_packages_path=lerobot_robot_piper"
}
