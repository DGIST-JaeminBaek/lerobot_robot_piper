#!/usr/bin/env bash
# 지우기 종료 게이트 실행 (+ 선택적 HIL 개입).
#
# 설계·검증 절차는 docs/erase_run_design.md. 이 스크립트는 §7.2의 4~7단계를
# 환경변수 하나로 고르게 만든 것이다.
#
#   STAGE=dry    인자·경로만 검증. 로봇에 명령 안 보냄 (기본값)
#   STAGE=gate   게이트만, HIL 없음. ★ 팔이 실제로 움직인다
#   STAGE=hil    게이트 + 리더암 개입. ★ 팔이 실제로 움직인다
#
# 조건 비교 실험을 할 때는 CONDITION으로 로그를 나눠 담는다. 그래야
# erase_eval.py가 조건별로 읽는다:
#
#   CONDITION=gate STAGE=gate TARGET=triangle bash scripts/13__erase_gate.sh
#   ...20회 반복 (매번 사람이 도형을 새로 그린다 — 자동 리셋 불가)
#   python scripts/tools/erase_eval.py \
#       --condition baseline:runs/baseline/*.json \
#       --condition gate:runs/gate/*.json --plot
#
# ★ 실물 안전: 첫 실행은 반드시 STAGE=dry로 확인하고, ATTEMPTS=1로 시작한다.
#   Ctrl+C를 누르면 러너가 parking을 끝낸 뒤에 토크를 푼다(3f6d1ad).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=lib/run_common.sh
source "${SCRIPT_DIR}/lib/run_common.sh"

load_recording_env
activate_conda_env

ROS_DISTRO_NAME="${ROS_DISTRO_NAME:-humble}"
for setup in "/opt/ros/${ROS_DISTRO_NAME}/setup.bash" "${HOME}/UGRP/ros2_ws/install/setup.bash"; do
  if [[ -f "${setup}" ]]; then
    set +u
    # shellcheck disable=SC1090
    source "${setup}"
    set -u
  fi
done

cd "${REPO_DIR}"

# ── 실험 대상 ──────────────────────────────────────────────
# state 7차원 / top+wrist 512x512 / task 문자열이 erase_run 기본값과 일치하는
# 조합이다. 다른 체크포인트로 바꾸면 --task와 --wrist-crop도 같이 확인할 것
# (top_only 체크포인트면 WRIST_CROP을 비운다).
POLICY="${POLICY:-outputs/train/pick_up_the_eraser_and_erase_the_shape/smolvla_topwrist_0802_0804_0805_0813pm_135/checkpoints/last/pretrained_model}"
DATASET="${DATASET:-records/outputs/pick_up_the_eraser_and_erase_the_shape/pick_up_the_eraser_0802_0804_0805_0813pm_135}"
# ★ ERASE_TASK — 이름을 TASK로 두면 안 된다. recording.env가 녹화용 TASK를
#   export하고 있어서(현재 "erase the circle") 기본값이 조용히 덮인다. 그 문장은
#   이 체크포인트의 학습 데이터에 없어서 분포 밖 입력이 되고, 정책이 본 적 없는
#   프롬프트로 돌게 된다 (erase_run.py의 --task 주석 참고).
#   판정 대상(TARGET)은 우리가 쥔 라벨이라 task 문장과 별개다.
ERASE_TASK="${ERASE_TASK:-pick up the eraser and erase the shape}"
TARGET="${TARGET:-triangle}"

STAGE="${STAGE:-dry}"
ATTEMPTS="${ATTEMPTS:-1}"
# 정책에게 주는 스텝 예산. 개입 스텝은 여기서 빠지므로(러너가 제외) 사람이
# 오래 조작해도 정책 시간이 줄지 않는다.
MAX_STEPS="${MAX_STEPS:-940}"
# demo   = 기록 없음. 게이트/HIL 동작만 볼 때.
# augment= 롤아웃을 LeRobotDataset으로 기록 + 원본 프레임 저장.
#          ★ 개입 궤적을 학습에 쓰려면 반드시 이쪽이어야 한다. demo로 돌리면
#            JSON/npz만 남고 데이터셋이 안 생겨서 나중에 학습에 못 쓴다.
MODE="${MODE:-demo}"
# HIL/게이트 산출물이 쌓이는 곳. 녹화 데이터셋(records/local/) 바로 옆이라
# 나중에 찾기 쉽다. 실행마다 records/hil/<시각>/ 하위에 로그·궤적·판정 프레임이 남는다.
HIL_DIR="${HIL_DIR:-records/hil}"
# RECORD=manual: 패널 버튼으로 구간을 골라 녹화한다(개입 구간만 담고 싶을 때).
#                데이터셋은 records/hil/<시각>/dataset 에 생긴다.
# RECORD=off   : 녹화 안 함 (기본)
# 전 구간 자동 녹화는 MODE=augment 쪽이다.
RECORD="${RECORD:-off}"
# aggregate_fn은 조건 전체에서 고정해야 한다 — 안 그러면 게이트 효과와
# 스무딩 방식 효과가 섞인다 (설계 문서 §5.5, §8.1).
AGGREGATE="${AGGREGATE:-weighted_average}"
CONDITION="${CONDITION:-}"

TOP_CROP="${TOP_CROP:-280,0,720}"
WRIST_CROP="${WRIST_CROP:-0,0,720}"

if [[ ! -d "${POLICY}" ]]; then
  echo "[ERROR] 체크포인트가 없다: ${POLICY}" >&2
  exit 1
fi
if [[ ! -d "${DATASET}" ]]; then
  echo "[ERROR] 데이터셋이 없다: ${DATASET}" >&2
  exit 1
fi

# 로그 경로 — 조건을 주면 조건별 디렉터리에 타임스탬프로 쌓는다
if [[ -n "${CONDITION}" ]]; then
  OUT_DIR="${OUT_DIR:-runs/${CONDITION}}"
  mkdir -p "${OUT_DIR}"
  OUT="${OUT_DIR}/$(date +%Y%m%d-%H%M%S).json"
else
  OUT="${OUT:-erase_run_log.json}"
fi

ARGS=(
  --policy_path "${POLICY}"
  --dataset_root "${DATASET}"
  --task "${ERASE_TASK}"
  --target "${TARGET}"
  --max-attempts "${ATTEMPTS}"
  --max-steps "${MAX_STEPS}"
  --mode "${MODE}"
  --aggregate-fn "${AGGREGATE}"
  --top-crop "${TOP_CROP}"
  --top-cam "${TOP_CAM:-327122074262}"
  --out "${OUT}"
  --hil-dir "${HIL_DIR}"
)
[[ "${RECORD}" == manual ]] && ARGS+=(--record-manual)
[[ -n "${WRIST_CROP}" ]] && ARGS+=(--wrist-crop "${WRIST_CROP}")

case "${STAGE}" in
  dry)
    echo "[STAGE=dry] 인자만 검증한다 — 로봇에 명령을 보내지 않는다."
    ;;
  gate)
    echo "[STAGE=gate] ★ 팔이 움직인다. 게이트만, HIL 없음."
    ARGS+=(--confirm)
    ;;
  hil)
    echo "[STAGE=hil] ★ 팔이 움직인다. space=개입 전환, q=시도 중단."
    echo "            개입 전에 리더암을 팔로워와 대충 맞춰두면 조작감이 낫다"
    echo "            (필수는 아니다 — 클러치라 어긋나도 튐은 0. 설계 문서 §5.3.1)"
    ARGS+=(--confirm --hil --leader-port "${LEADER_PORT:-can_leader}")
    ;;
  *)
    echo "[ERROR] STAGE는 dry | gate | hil 중 하나여야 한다 (받은 값: ${STAGE})" >&2
    exit 2
    ;;
esac

echo "  정책   : ${POLICY}"
echo "  데이터 : ${DATASET}"
echo "  task   : ${ERASE_TASK}"
echo "  target : ${TARGET}   시도 상한: ${ATTEMPTS}   aggregate: ${AGGREGATE}"
if [[ "${MODE}" == demo && "${RECORD}" != manual ]]; then
  echo "  mode   : ${MODE}   (기록 안 함 — 개입 궤적을 담으려면 RECORD=manual 또는 MODE=augment)"
else
  echo "  mode   : ${MODE}"
fi
echo "  로그   : ${OUT}"
echo "  산출물 : ${HIL_DIR}/<시각>/  (log.json, *.steps.npz, 판정 프레임 png)"
echo "  녹화   : ${RECORD}$([ "${RECORD}" = manual ] && echo '   (패널 버튼으로 구간 선택)')"
echo

exec python scripts/tools/erase_run.py "${ARGS[@]}"
