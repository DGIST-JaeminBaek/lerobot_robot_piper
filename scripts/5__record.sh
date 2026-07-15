#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/run_common.sh
source "${SCRIPT_DIR}/lib/run_common.sh"

load_recording_env
require_cmd lerobot-record

LEADER_PORT="${LEADER_PORT:-can_leader1}"
FOLLOWER_PORT="${FOLLOWER_PORT:-can_follower1}"
NUM_EPISODES="${NUM_EPISODES:-5}"
EPISODE_TIME_S="${EPISODE_TIME_S:-60}"
RESET_TIME_S="${RESET_TIME_S:-60}"
FPS="${FPS:-30}"
TASK="${TASK:-write AIIII}"
PUSH_TO_HUB="$(bool_default "${PUSH_TO_HUB:-}" false)"
DISPLAY_DATA="$(bool_default "${DISPLAY_DATA:-}" true)"
RESUME="$(bool_default "${RESUME:-}" false)"

DATASET_REPO_ID_BASE="${DATASET_REPO_ID:-local/piper_write_light}"
DATASET_ROOT_BASE="${DATASET_ROOT:-${REPO_DIR}/records/${DATASET_REPO_ID_BASE}}"
if [[ "${RESUME}" == "true" ]]; then
  # 이어서 녹화하는 경우엔 기존 폴더를 그대로 찾아야 하므로 타임스탬프를 붙이지 않음
  DATASET_REPO_ID="${DATASET_REPO_ID_BASE}"
  DATASET_ROOT="${DATASET_ROOT_BASE}"
else
  # 매 녹화마다 월일-시분초를 붙여서 폴더가 서로 겹치지 않게 함(같은 이름으로
  # 다시 녹화해도 이전 녹화를 덮어쓰지 않음).
  TIMESTAMP="$(date +%m%d-%H%M%S)"
  DATASET_REPO_ID="${DATASET_REPO_ID_BASE}_${TIMESTAMP}"
  DATASET_ROOT="${DATASET_ROOT_BASE}_${TIMESTAMP}"
fi

mapfile -t DISCOVERY_ARGS < <(plugin_discovery_args)
mapfile -t CAMERA_ARGS < <(robot_camera_args)
mapfile -t OFFSET_ARGS < <(robot_action_offset_args)
mapfile -t SAFETY_ARGS < <(robot_safety_args)

# LeRobot dataset 부모 저장 위치
mkdir -p "$(dirname "${DATASET_ROOT}")"

cmd=(
  lerobot-record
  "--robot.type=piper_follower"
  "--robot.port=${FOLLOWER_PORT}"
  "${CAMERA_ARGS[@]}"
  "${OFFSET_ARGS[@]}"
  "${SAFETY_ARGS[@]}"
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
