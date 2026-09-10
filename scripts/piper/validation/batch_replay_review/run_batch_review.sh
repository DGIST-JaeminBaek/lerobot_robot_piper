#!/usr/bin/env bash
# 0802_joint4_corrected + 0804_joint4_corrected 전량(90건)을 리플레이로 눈검사한다.
#
# 기본은 녹화된 영상/관절값 재생만 — 하드웨어를 전혀 건드리지 않는다.
#   ./run_batch_review.sh              # 처음부터
#   ./run_batch_review.sh --resume     # 미판정만 이어서
#   ./run_batch_review.sh --summary    # 진행률만 확인
#
# 실물 팔에 재생하려면 (창 안에서 x 두 번):
#   ./run_batch_review.sh --real-robot --real-robot-confirm I_UNDERSTAND_REAL_ROBOT
#   ./run_batch_review.sh --real-robot --real-robot-confirm I_UNDERSTAND_REAL_ROBOT --dry-run
set -euo pipefail

# 변수명에 BRR_ 접두어를 붙인다 — run_common.sh가 source될 때 SCRIPT_DIR/REPO_DIR을
# 자기 위치(scripts/lib) 기준으로 무조건 덮어쓰기 때문. 그냥 SCRIPT_DIR을 쓰면
# 소스 이후에 scripts/lib/batch_replay_review.py를 찾다가 죽는다.
BRR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRR_REPO_DIR="$(cd "${BRR_DIR}/../../../.." && pwd)"

# lerobot_robot_piper는 conda env(기본 ugrp)에만 설치돼 있다 — run_common.sh의
# activate_conda_env()와 같은 이유로 여기서도 먼저 활성화한다.
# shellcheck source=../../lib/run_common.sh
source "${BRR_REPO_DIR}/scripts/lib/run_common.sh"
activate_conda_env

cd "${BRR_REPO_DIR}"
exec python "${BRR_DIR}/batch_replay_review.py" "$@"
