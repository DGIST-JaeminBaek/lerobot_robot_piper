#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=run_common.sh
source "${SCRIPT_DIR}/run_common.sh"

load_recording_env

if [[ "${DRY_RUN:-false}" != "true" ]] && ! python -c "import grpc" >/dev/null 2>&1; then
  echo "[ERROR] grpc is not installed in this environment." >&2
  echo "[HINT] Install LeRobot async dependencies, for example: python -m pip install 'grpcio>=1.0' grpcio-tools" >&2
  exit 1
fi

SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-8088}"
FPS="${FPS:-30}"

cmd=(
  python -m lerobot.async_inference.policy_server
  "--host=${SERVER_HOST}"
  "--port=${SERVER_PORT}"
  "--fps=${FPS}"
)

run_or_print "${cmd[@]}" "$@"
