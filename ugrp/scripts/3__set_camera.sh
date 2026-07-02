#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=run_common.sh
source "${SCRIPT_DIR}/run_common.sh"

TOP="${1:-}"
WRIST="${2:-}"

if [[ -z "${TOP}" || -z "${WRIST}" ]]; then
  echo "Usage: bash ugrp/scripts/3__set_camera.sh TOP_CAM WRIST_CAM"
  echo "Example: bash ugrp/scripts/3__set_camera.sh 6 0"
  exit 2
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  cp "${UGRP_DIR}/configs/recording.env.example" "${ENV_FILE}"
fi

set_key() {
  local key="$1"
  local value="$2"
  if grep -q "^${key}=" "${ENV_FILE}"; then
    sed -i "s|^${key}=.*|${key}=${value}|" "${ENV_FILE}"
  else
    printf '%s=%s\n' "${key}" "${value}" >> "${ENV_FILE}"
  fi
}

set_key TOP_CAM "${TOP}"
set_key WRIST_CAM "${WRIST}"

echo "[INFO] Updated ${ENV_FILE}"
echo "[INFO] TOP_CAM=${TOP}"
echo "[INFO] WRIST_CAM=${WRIST}"
