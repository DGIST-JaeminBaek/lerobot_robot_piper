#!/usr/bin/env bash
# CAN 인터페이스 초기화 → 관절값 프리플라이트 체크 → teleop_ui GUI 실행을 한 번에 수행.
# 개별 단계만 필요하면 기존 1__init_can.sh / scripts/tools/piper_session.py를
# 직접 실행할 것 — 이 스크립트는 그 둘을 그대로 호출하는 얇은 래퍼임.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/run_common.sh
source "${SCRIPT_DIR}/lib/run_common.sh"

load_recording_env

SKIP_CAN_INIT="$(bool_default "${SKIP_CAN_INIT:-}" false)"
SKIP_JOINT_CHECK="$(bool_default "${SKIP_JOINT_CHECK:-}" false)"
ROS_DISTRO_NAME="${ROS_DISTRO_NAME:-humble}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ugrp}"

echo "=== [0/3] conda 가상환경 활성화 (${CONDA_ENV_NAME}) ==="
if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV_NAME}" ]]; then
  echo "[INFO] 이미 ${CONDA_ENV_NAME} 활성화됨 — 건너뜀"
else
  # Nautilus "Run in Terminal" 등 비로그인 셸에서는 ~/.bashrc의 conda init이
  # 실행되지 않아 PATH에 conda가 없을 수 있음 — PATH 대신 흔한 설치 위치를 직접 탐색
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

  # conda.sh/activate가 내부적으로 미설정 변수를 참조해 set -u와 충돌하므로 잠시 해제
  set +u
  # shellcheck disable=SC1091
  source "${conda_base}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
  set -u
fi

echo "[OK] conda env  = ${CONDA_DEFAULT_ENV:-<none>}"
echo "[OK] python     = $(command -v python)"
echo "[OK] python ver = $(python --version 2>&1)"

echo "=== [1/4] CAN 인터페이스 초기화 ==="
if [[ "${SKIP_CAN_INIT}" == "true" ]]; then
  echo "[INFO] SKIP_CAN_INIT=true — 건너뜀"
else
  sudo modprobe gs_usb
  "${REPO_DIR}/scripts/1__init_can.sh"

  # USB 재연결/재부팅 직후에는 커널이 can0/can1 같은 기본 이름을 붙이는데,
  # 1__init_can.sh는 LEADER_PORT/FOLLOWER_PORT(can_leader/can_follower) 이름의
  # 인터페이스만 다루므로 이 경우 아무것도 못 찾고 넘어감. 그 이름들이 아직 없으면
  # 지금 보이는 raw CAN 인터페이스를 이름은 그대로 둔 채 bitrate만 맞춰 올려서,
  # 최소한 GUI의 CAN Setup에서 Detect/Init All(및 손으로 흔들어 실측 확인)로
  # leader/follower를 새로 배정할 수 있게 함. 역할 자동 배정은 여기서 하지 않음 —
  # USB 포트가 바뀌면 어느 쪽이 leader/follower인지는 실제로 흔들어봐야 확실함.
  if ! ip link show "${LEADER_PORT}" >/dev/null 2>&1 || ! ip link show "${FOLLOWER_PORT}" >/dev/null 2>&1; then
    echo "[INFO] ${LEADER_PORT}/${FOLLOWER_PORT} 이름의 인터페이스가 아직 없음 — raw CAN 인터페이스를 이름 그대로 bring-up"
    for iface in $(ip -br link show type can | awk '{print $1}'); do
      sudo ip link set "${iface}" down 2>/dev/null || true
      sudo ip link set "${iface}" type can bitrate "${BITRATE:-1000000}"
      sudo ip link set "${iface}" up
      echo "[OK] ${iface} up (bitrate ${BITRATE:-1000000})"
    done
    echo "[INFO] GUI의 CAN Setup 패널에서 Detect -> (필요시 손으로 팔 흔들어 역할 확인) -> Init All로 이름을 배정하세요."
  fi
fi

echo
echo "=== [2/4] 관절값 프리플라이트 체크 (follower + leader) ==="
if [[ "${SKIP_JOINT_CHECK}" == "true" ]]; then
  echo "[INFO] SKIP_JOINT_CHECK=true — 건너뜀"
elif ! ip link show "${LEADER_PORT}" >/dev/null 2>&1 || ! ip link show "${FOLLOWER_PORT}" >/dev/null 2>&1; then
  echo "[INFO] ${LEADER_PORT}/${FOLLOWER_PORT} 이름이 아직 없어 건너뜀 — GUI에서 이름 배정 후 CAN Monitor로 직접 확인하세요."
else
  if ! python "${REPO_DIR}/scripts/tools/piper_session.py" --step joint_check --check_leader \
      --follower_can_interface="${FOLLOWER_PORT}" --leader_can_interface="${LEADER_PORT}"; then
    echo "[WARN] joint_check 실패 — CAN 배선/전원을 확인하세요." >&2
    read -r -p "그래도 GUI를 실행할까요? (y/N): " reply
    case "${reply}" in
      y|Y|yes|YES) ;;
      *) exit 1 ;;
    esac
  fi
fi

ros_setup="/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
if [[ -f "${ros_setup}" ]]; then
  # ROS2 setup.bash 내부가 set -u와 안 맞는 미설정 변수(AMENT_TRACE_SETUP_FILES 등)를
  # 참조해서 죽으므로 conda.sh와 마찬가지로 잠시 해제.
  set +u
  # shellcheck disable=SC1090
  source "${ros_setup}"
  set -u
else
  echo "[WARN] ${ros_setup} 없음 — RViz 관련 기능(rviz_preview/Replay/Infer Preview)은 이 세션에서 동작하지 않음" >&2
fi

echo
echo "=== [3/4] teleop_ui GUI 실행 ==="
exec python -m lerobot_robot_piper.teleop_ui
