#!/usr/bin/env bash
# 추론 실시간 GUI(smoothing / RViz / E-STOP)를 실행한다.
#
# RViz는 이 스크립트가 띄우지 않는다. 궤적을 보려면 먼저 다른 터미널에서
#   ros2 launch agx_arm_description display_piper.launch.py
# 를 실행할 것. ROS2 setup.bash를 여기서 source하므로, RViz가 없어도 GUI 자체는
# 뜨고 (RViz publish만 실패 로그를 남기고 비활성화된다).
#
# 기본은 source=dataset 안전 모드다. 실물 전송은 GUI 안에서 source=robot +
# 확인 문구 입력을 해야만 켜진다.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/run_common.sh
source "${SCRIPT_DIR}/lib/run_common.sh"

load_recording_env

ROS_DISTRO_NAME="${ROS_DISTRO_NAME:-humble}"
ROS_SETUP_PATH="${ROS_SETUP_PATH:-/opt/ros/${ROS_DISTRO_NAME}/setup.bash}"

if [[ -f "${ROS_SETUP_PATH}" ]]; then
  # ROS2 setup.bash가 미설정 변수를 참조해 set -u와 충돌하므로 잠시 해제
  set +u
  # shellcheck disable=SC1090
  source "${ROS_SETUP_PATH}"
  set -u
  echo "[OK] ROS2 ${ROS_DISTRO_NAME} 환경 로드"
else
  echo "[WARN] ${ROS_SETUP_PATH} 없음 — GUI는 뜨지만 RViz publish는 비활성화됩니다" >&2
fi

# ros2_ws에 agx_arm_description이 빌드돼 있으면 함께 로드해서 RViz launch를
# 이 셸에서도 바로 쓸 수 있게 한다.
AGX_WS_SETUP="${AGX_WS_SETUP:-${HOME}/UGRP/ros2_ws/install/setup.bash}"
if [[ -f "${AGX_WS_SETUP}" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "${AGX_WS_SETUP}"
  set -u
  echo "[OK] ${AGX_WS_SETUP} 로드 — 'ros2 launch agx_arm_description display_piper.launch.py' 사용 가능"
fi

run_or_print python -m lerobot_robot_piper.infer_gui "$@"
