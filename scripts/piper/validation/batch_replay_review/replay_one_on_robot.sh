#!/usr/bin/env bash
# batch_replay_review.py의 x 키가 호출하는 "녹화 1건을 실물 follower에 재생" 래퍼.
#
# 왜 scripts/6__replay.sh를 직접 안 쓰는가 —
# 6__replay.sh는 load_recording_env()로 configs/recording.env를 `set -a`로 source한다.
# recording.env에는 DATASET_REPO_ID/DATASET_ROOT가 들어 있어서, 바깥에서 export한
# 값이 그 source에 덮여버린다(실측: 0802 녹화를 넘겼는데 커맨드에는
# local/piper_pick_pen_sample이 찍혔다). 엉뚱한 궤적이 팔에 나가는 사고라
# 여기서는 env를 source '이후'에 적용한다.
#
# 안전 인자(max_relative_target, effort 컷오프, torque 해제 방식)는 그대로
# run_common.sh의 robot_safety_args()에서 가져온다 — 정의를 복제하지 않는다.
#
# 필수 입력: REVIEW_DATASET_ROOT (녹화 폴더 절대경로)
# 선택 입력: REVIEW_FOLLOWER_PORT, REVIEW_FPS, DRY_RUN
set -euo pipefail

# BRR_ 접두어 — run_common.sh가 SCRIPT_DIR/REPO_DIR을 자기 위치 기준으로 덮어쓴다.
BRR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRR_REPO_DIR="$(cd "${BRR_DIR}/../../../.." && pwd)"
# shellcheck source=../../lib/run_common.sh
source "${BRR_REPO_DIR}/scripts/lib/run_common.sh"

load_recording_env          # recording.env를 먼저 읽고 (안전 기본값 확보)
require_cmd lerobot-replay

# ---- recording.env가 덮을 수 없도록 여기서 마지막에 결정한다 ----
DATASET_ROOT="${REVIEW_DATASET_ROOT:?REVIEW_DATASET_ROOT is required}"
if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
  echo "[ERROR] LeRobot 데이터셋이 아닙니다: ${DATASET_ROOT}" >&2
  exit 2
fi
DATASET_REPO_ID="local/$(basename "${DATASET_ROOT}")"
FOLLOWER_PORT="${REVIEW_FOLLOWER_PORT:-${FOLLOWER_PORT:-can_follower}}"
FPS="${REVIEW_FPS:-${FPS:-30}}"

mapfile -t SAFETY_ARGS < <(robot_safety_args)

cmd=(
  lerobot-replay
  "--robot.type=piper_follower"
  "--robot.port=${FOLLOWER_PORT}"
  "${SAFETY_ARGS[@]}"
  # recorded action은 send_action()이 실제로 follower에 보낸 값(offset이 이미 적용된
  # 절대 목표값)이라, replay 때 use_action_offset이 켜져 있으면 보정이 두 번 얹힌다
  # — 6__replay.sh와 동일하게 반드시 꺼야 한다.
  "--robot.use_action_offset=false"
  "--dataset.repo_id=${DATASET_REPO_ID}"
  "--dataset.root=${DATASET_ROOT}"
  # 이 프로젝트의 원본 녹화는 1폴더 = 1에피소드다
  "--dataset.episode=0"
  "--dataset.fps=${FPS}"
  "--robot.discover_packages_path=lerobot_robot_piper"
)

run_or_print "${cmd[@]}"
