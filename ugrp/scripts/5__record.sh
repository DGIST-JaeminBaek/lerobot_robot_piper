#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=run_common.sh
source "${SCRIPT_DIR}/run_common.sh"

load_recording_env
require_cmd lerobot-record

LEADER_PORT="${LEADER_PORT:-can_leader1}"
FOLLOWER_PORT="${FOLLOWER_PORT:-can_follower1}"
ROBOT_ID="${ROBOT_ID:-piper_follower1}"
TELEOP_ID="${TELEOP_ID:-piper_leader1}"
MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-5}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/piper_write_light}"
NUM_EPISODES="${NUM_EPISODES:-5}"
EPISODE_TIME_S="${EPISODE_TIME_S:-60}"
TASK="${TASK:-write AIIII}"
PUSH_TO_HUB="$(bool_default "${PUSH_TO_HUB:-}" false)"
DISPLAY_DATA="$(bool_default "${DISPLAY_DATA:-}" true)"
RESUME="$(bool_default "${RESUME:-}" false)"

mapfile -t DISCOVERY_ARGS < <(plugin_discovery_args)

cmd=(
  lerobot-record
  "--robot.type=piper_follower"
  "--robot.port=${FOLLOWER_PORT}"
  "--robot.id=${ROBOT_ID}"
  "--robot.max_relative_target=${MAX_RELATIVE_TARGET}"
  "--robot.cameras=$(camera_config_arg)"
  "--teleop.type=piper_leader"
  "--teleop.port=${LEADER_PORT}"
  "--teleop.id=${TELEOP_ID}"
  "--display_data=${DISPLAY_DATA}"
  "--dataset.repo_id=${DATASET_REPO_ID}"
  "--dataset.num_episodes=${NUM_EPISODES}"
  "--dataset.episode_time_s=${EPISODE_TIME_S}"
  "--dataset.single_task=${TASK}"
  "--dataset.push_to_hub=${PUSH_TO_HUB}"
  "--resume=${RESUME}"
  "${DISCOVERY_ARGS[@]}"
)

run_or_print "${cmd[@]}" "$@"
