#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=run_common.sh
source "${SCRIPT_DIR}/run_common.sh"

load_recording_env
require_cmd lerobot-replay

FOLLOWER_PORT="${FOLLOWER_PORT:-can_follower1}"
ROBOT_ID="${ROBOT_ID:-piper_follower1}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/piper_write_light}"
DATASET_EPISODE="${DATASET_EPISODE:-0}"
FPS="${FPS:-30}"

cmd=(
  lerobot-replay
  "--robot.type=piper_follower"
  "--robot.port=${FOLLOWER_PORT}"
  "--robot.id=${ROBOT_ID}"
  "--dataset.repo_id=${DATASET_REPO_ID}"
  "--dataset.episode=${DATASET_EPISODE}"
  "--dataset.fps=${FPS}"
  "--robot.discover_packages_path=lerobot_robot_piper"
)

run_or_print "${cmd[@]}" "$@"
