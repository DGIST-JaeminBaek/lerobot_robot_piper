#!/usr/bin/env bash
set -euo pipefail

# 녹화 전 텔레옵 안전 점검 스크립트
# 목적:
# - leader 입력이 follower에 정상 전달되는지 확인
# - CAN 포트 방향이 뒤바뀌지 않았는지 확인
# - max_relative_target 설정으로 갑작스러운 이동을 줄인 상태에서 테스트

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 실제 동작은 Python 스크립트에 위임
# Bash wrapper: Linux 서버에서 짧게 실행하기 위한 진입점
python3 "${SCRIPT_DIR}/wego_teleop_smoke.py" "$@"
