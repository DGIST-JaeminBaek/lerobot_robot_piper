#!/usr/bin/env bash
set -euo pipefail

# write AIIII 데이터셋 녹화 스크립트
# - leader Piper: 사람이 직접 움직이는 마스터 팔
# - follower Piper: 실제 펜을 잡고 화이트보드에 쓰는 슬레이브 팔
# - top/wrist camera: LeRobot dataset의 image observation으로 저장
# - task text: LeRobot dataset metadata의 single_task로 저장

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 실제 동작은 Python 스크립트에 위임
# Bash wrapper: Linux 서버에서 짧게 실행하기 위한 진입점
python3 "${SCRIPT_DIR}/wego_record_write_light.py" "$@"
