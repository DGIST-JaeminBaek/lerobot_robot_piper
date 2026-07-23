#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/run_common.sh
source "${SCRIPT_DIR}/lib/run_common.sh"

load_recording_env
require_cmd lerobot-teleoperate

LEADER_PORT="${LEADER_PORT:-can_leader1}"
FOLLOWER_PORT="${FOLLOWER_PORT:-can_follower1}"
DISPLAY_DATA="$(bool_default "${DISPLAY_DATA:-}" true)"

mapfile -t DISCOVERY_ARGS < <(plugin_discovery_args)
mapfile -t OFFSET_ARGS < <(robot_action_offset_args)
mapfile -t SAFETY_ARGS < <(robot_safety_args)

cmd=(
  lerobot-teleoperate
  "--robot.type=piper_follower"
  "--robot.port=${FOLLOWER_PORT}"
  "${OFFSET_ARGS[@]}"
  "${SAFETY_ARGS[@]}"
  "--teleop.type=piper_leader"
  "--teleop.port=${LEADER_PORT}"
  "--display_data=${DISPLAY_DATA}"
  "${DISCOVERY_ARGS[@]}"
)

run_or_print "${cmd[@]}" "$@"
