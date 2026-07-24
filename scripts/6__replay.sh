#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/run_common.sh
source "${SCRIPT_DIR}/lib/run_common.sh"

load_recording_env
require_cmd lerobot-replay

FOLLOWER_PORT="${FOLLOWER_PORT:-can_follower1}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/piper_write_light}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_DIR}/records/${DATASET_REPO_ID}}"
DATASET_EPISODE="${DATASET_EPISODE:-0}"
FPS="${FPS:-30}"

mapfile -t SAFETY_ARGS < <(robot_safety_args)

cmd=(
  lerobot-replay
  "--robot.type=piper_follower"
  "--robot.port=${FOLLOWER_PORT}"
  "${SAFETY_ARGS[@]}"
  # recorded action은 send_action()이 실제로 follower에 보낸 값(offset 이미 적용된
  # follower 좌표계 절대 목표값) — replay 때 use_action_offset이 켜져 있으면 또
  # 한 번 보정이 얹혀서 이중 보정이 되므로 꺼야 함.
  "--robot.use_action_offset=false"
  "--dataset.repo_id=${DATASET_REPO_ID}"
  "--dataset.root=${DATASET_ROOT}"
  "--dataset.episode=${DATASET_EPISODE}"
  "--dataset.fps=${FPS}"
  "--robot.discover_packages_path=lerobot_robot_piper"
)

run_or_print "${cmd[@]}" "$@"
