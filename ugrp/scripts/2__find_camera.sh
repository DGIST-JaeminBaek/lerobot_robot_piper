#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=run_common.sh
source "${SCRIPT_DIR}/run_common.sh"

load_recording_env

if command -v lerobot-find-cameras >/dev/null 2>&1; then
  run_or_print lerobot-find-cameras "$@"
else
  run_or_print python "${PLUGIN_DIR}/camera_check.py" "$@"
fi
