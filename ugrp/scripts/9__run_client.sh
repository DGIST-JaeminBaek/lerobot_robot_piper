#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=run_common.sh
source "${SCRIPT_DIR}/run_common.sh"

load_recording_env

if [[ "${DRY_RUN:-false}" != "true" ]] && ! python -c "import grpc" >/dev/null 2>&1; then
  echo "[ERROR] grpc is not installed in this environment." >&2
  echo "[HINT] Install LeRobot async dependencies before using async inference." >&2
  exit 1
fi

FOLLOWER_PORT="${FOLLOWER_PORT:-can_follower1}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-8088}"
PRETRAINED_NAME_OR_PATH="${PRETRAINED_NAME_OR_PATH:-}"
POLICY_TYPE="${POLICY_TYPE:-smolvla}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
TASK="${TASK:-write AIIII}"
FPS="${FPS:-30}"
ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-50}"
CHUNK_SIZE_THRESHOLD="${CHUNK_SIZE_THRESHOLD:-0.8}"
AGGREGATE_FN_NAME="${AGGREGATE_FN_NAME:-average}"

if [[ -z "${PRETRAINED_NAME_OR_PATH}" && "${DRY_RUN:-false}" != "true" ]]; then
  echo "[ERROR] PRETRAINED_NAME_OR_PATH is required for async client." >&2
  echo "[HINT] Set it in ugrp/configs/recording.env or pass PRETRAINED_NAME_OR_PATH=... before this script." >&2
  exit 1
fi
PRETRAINED_NAME_OR_PATH="${PRETRAINED_NAME_OR_PATH:-your_hf_or_local_policy_path}"
mapfile -t CAMERA_ARGS < <(robot_camera_args)

cmd=(
  python -m lerobot.async_inference.robot_client
  "--server_address=${SERVER_HOST}:${SERVER_PORT}"
  "--robot.type=piper_follower"
  "--robot.port=${FOLLOWER_PORT}"
  "${CAMERA_ARGS[@]}"
  "--robot.discover_packages_path=lerobot_robot_piper"
  "--pretrained_name_or_path=${PRETRAINED_NAME_OR_PATH}"
  "--policy_type=${POLICY_TYPE}"
  "--policy_device=${POLICY_DEVICE}"
  "--actions_per_chunk=${ACTIONS_PER_CHUNK}"
  "--chunk_size_threshold=${CHUNK_SIZE_THRESHOLD}"
  "--aggregate_fn_name=${AGGREGATE_FN_NAME}"
  "--task=${TASK}"
  "--fps=${FPS}"
)

run_or_print "${cmd[@]}" "$@"
