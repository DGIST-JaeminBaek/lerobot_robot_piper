#!/usr/bin/env bash
# 녹화 QC + 자를 구간 확인 GUI (scripts/tools/qc_studio.py) 실행.
# 로봇/CAN과 무관한 후처리 도구라 CAN 초기화나 관절 체크는 하지 않음 —
# conda 환경만 잡아주는 얇은 래퍼임.
#
#   ./10__qc_studio.sh                          # records/local 검토
#   ./10__qc_studio.sh --folder records/0727/erase_the_circle
#   ./10__qc_studio.sh --report                 # 창 없이 표만 출력
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/run_common.sh
source "${SCRIPT_DIR}/lib/run_common.sh"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-ugrp}"

echo "=== [1/2] conda 가상환경 활성화 (${CONDA_ENV_NAME}) ==="
if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV_NAME}" ]]; then
  echo "[INFO] 이미 ${CONDA_ENV_NAME} 활성화됨 — 건너뜀"
else
  # 0__launch_gui.sh와 같은 이유로 PATH 대신 흔한 설치 위치를 직접 탐색
  conda_base=""
  for candidate in "${CONDA_EXE:+$(dirname "$(dirname "${CONDA_EXE}")")}" \
                   "${HOME}/miniconda3" "${HOME}/anaconda3" "${HOME}/miniforge3" \
                   "/opt/miniconda3" "/opt/anaconda3"; do
    if [[ -n "${candidate}" && -f "${candidate}/etc/profile.d/conda.sh" ]]; then
      conda_base="${candidate}"
      break
    fi
  done

  if [[ -z "${conda_base}" ]]; then
    echo "[ERROR] conda 설치 위치를 찾을 수 없음 — 수동으로 '${CONDA_ENV_NAME}' 환경을 활성화한 뒤 재실행하세요." >&2
    exit 1
  fi

  set +u
  # shellcheck disable=SC1091
  source "${conda_base}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
  set -u
fi

echo "[OK] python = $(command -v python)"

echo
echo "=== [2/2] QC Studio 실행 ==="
exec python "${REPO_DIR}/scripts/tools/qc_studio.py" "$@"
