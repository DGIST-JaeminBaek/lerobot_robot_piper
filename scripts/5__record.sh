#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/run_common.sh
source "${SCRIPT_DIR}/lib/run_common.sh"

load_recording_env
require_cmd lerobot-record

LEADER_PORT="${LEADER_PORT:-can_leader1}"
FOLLOWER_PORT="${FOLLOWER_PORT:-can_follower1}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/piper_write_light}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_DIR}/records/${DATASET_REPO_ID}}"
NUM_EPISODES="${NUM_EPISODES:-5}"
EPISODE_TIME_S="${EPISODE_TIME_S:-60}"
RESET_TIME_S="${RESET_TIME_S:-60}"
FPS="${FPS:-30}"
TASK="${TASK:-write AIIII}"
PUSH_TO_HUB="$(bool_default "${PUSH_TO_HUB:-}" false)"
DISPLAY_DATA="$(bool_default "${DISPLAY_DATA:-}" true)"
RESUME="$(bool_default "${RESUME:-}" false)"

mapfile -t DISCOVERY_ARGS < <(plugin_discovery_args)
mapfile -t CAMERA_ARGS < <(robot_camera_args)
mapfile -t OFFSET_ARGS < <(robot_action_offset_args)

# LeRobot dataset 부모 저장 위치
mkdir -p "$(dirname "${DATASET_ROOT}")"

cmd=(
  lerobot-record
  "--robot.type=piper_follower"
  "--robot.port=${FOLLOWER_PORT}"
  "${CAMERA_ARGS[@]}"
  "${OFFSET_ARGS[@]}"
  "--teleop.type=piper_leader"
  "--teleop.port=${LEADER_PORT}"
  "--display_data=${DISPLAY_DATA}"
  "--dataset.repo_id=${DATASET_REPO_ID}"
  "--dataset.root=${DATASET_ROOT}"
  "--dataset.fps=${FPS}"
  "--dataset.num_episodes=${NUM_EPISODES}"
  "--dataset.episode_time_s=${EPISODE_TIME_S}"
  "--dataset.reset_time_s=${RESET_TIME_S}"
  "--dataset.single_task=${TASK}"
  "--dataset.push_to_hub=${PUSH_TO_HUB}"
  "--resume=${RESUME}"
  "${DISCOVERY_ARGS[@]}"
)

run_or_print "${cmd[@]}" "$@"
