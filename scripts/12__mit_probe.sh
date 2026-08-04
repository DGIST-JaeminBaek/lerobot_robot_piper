#!/usr/bin/env bash
# MIT(임피던스) 제어를 관절 하나씩 확인한다.
#
# ⚠ 토크 제어다. kp가 낮으면 팔이 중력에 무너지고 높으면 진동한다.
# 팔 주변을 비우고 비상 정지가 가능한 상태에서만 실행할 것.
#
# 권장 순서:
#   1) 중력 부담이 적은 관절부터 — joint1, 4, 5, 6
#   2) joint2 / joint3은 마지막에, kp 10부터
#
# 사용 예:
#   ./scripts/12__mit_probe.sh --joint 1 --kp 5
#   ./scripts/12__mit_probe.sh --joint 1 --kp 10 --amplitude 3 --cycles 2
#
# 확인 문구는 이 스크립트가 자동으로 붙인다 — 위 주의사항을 읽었다는 전제다.
set -euo pipefail

# run_common.sh가 source되면서 자기 위치(scripts/lib)로 SCRIPT_DIR을 덮어쓴다 —
# 별도 이름으로 잡아둬야 tools/ 경로가 scripts/lib/tools로 어긋나지 않는다.
PROBE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/run_common.sh
source "${PROBE_SCRIPT_DIR}/lib/run_common.sh"

load_recording_env

# lerobot_robot_piper는 conda env에만 설치돼 있다 — base에서 실행하면
# ModuleNotFoundError로 죽는다.
activate_conda_env

run_or_print python "${PROBE_SCRIPT_DIR}/tools/piper_mit_probe.py" \
  --confirm I_UNDERSTAND_TORQUE_CONTROL "$@"
