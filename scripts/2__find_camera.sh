#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/run_common.sh
source "${SCRIPT_DIR}/lib/run_common.sh"

load_recording_env

if command -v lerobot-find-cameras >/dev/null 2>&1; then
  run_or_print lerobot-find-cameras "$@"
else
  run_or_print python "${REPO_DIR}/scripts/tools/camera_check.py" "$@"
fi
