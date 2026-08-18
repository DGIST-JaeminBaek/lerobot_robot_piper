#!/usr/bin/env bash
# 평가 세션 창을 띄운다 — 롤아웃 실행 → 자동 채점 → 사람 확인 → 다음.
#
# 이 스크립트는 로봇을 직접 건드리지 않는다. 창 안의 [시작] 버튼이 EVAL_ROLLOUT_CMD를
# 실행하고, 그 명령이 만든 에피소드를 채점한다. 실물 전송 여부는 전적으로 그 명령이
# 정한다(erase_run.py는 --confirm 없이는 명령을 보내지 않는다).
#
# 사용 전 확인:
#   1) 하드웨어 없이 UI 점검  : DRY_RUN='<데이터셋>/erase_the_*' bash scripts/13__eval_session.sh
#   2) 현장 카메라 확인       : python scripts/tools/ink_metric.py <에피소드> --dump-roi /tmp/roi.png
#   3) 실제 세션              : 아래 변수를 채우고 실행
#
# 환경변수:
#   EVAL_MODEL       smolvla / pi0 …            (집계 축 1)
#   EVAL_CONDITION   async / sync …             (집계 축 2)
#   EVAL_TARGET      circle / rectangle / triangle (생략 시 폴더명에서 추론)
#   EVAL_TRIALS      목표 시행 수 (표시용)
#   EVAL_CUTOFF      에피소드 컷오프 초 (기본 60)
#   EVAL_ALIGN_CMD   시도 확인 직후 띄울 정렬 확인 명령 (도형·지우개 위치 점검).
#                    창을 닫을 때까지 [시작]이 잠긴다.
#   EVAL_OUT_DIR     결과 폴더 (기본 outputs/eval/<오늘>)
#   EVAL_WATCH_DIR   롤아웃이 새 에피소드를 만드는 폴더
#   EVAL_ROLLOUT_CMD 롤아웃 1회를 도는 명령
#   EVAL_BOARD       "x y w h" — 카메라를 옮겼으면 반드시 지정
#   DRY_RUN          지정하면 로봇 없이 기존 에피소드로 UI만 점검
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# run_common.sh는 자기 자신(scripts/lib) 기준으로 SCRIPT_DIR을 다시 정의한다 —
# source 뒤에 여기 SCRIPT_DIR을 그대로 쓰면 scripts/lib/tools/... 로 잘못 잡힌다
# (13__erase_gate.sh는 REPO_DIR을 미리 빼놔서 이 문제를 피한다. 2026-08-18 실물
# 첫 실행에서 TOOL 경로가 깨져 즉시 실패하는 것으로 발견).
EVAL_SESSION_SCRIPT_DIR="${SCRIPT_DIR}"
# shellcheck source=lib/run_common.sh
source "${SCRIPT_DIR}/lib/run_common.sh"

load_recording_env
activate_conda_env

TOOL="${EVAL_SESSION_SCRIPT_DIR}/tools/erase_eval_ui.py"
OUT_DIR="${EVAL_OUT_DIR:-}"

args=()
[[ -n "${OUT_DIR}" ]] && args+=(--out-dir "${OUT_DIR}")
[[ -n "${EVAL_MODEL:-}" ]] && args+=(--model "${EVAL_MODEL}")
[[ -n "${EVAL_CONDITION:-}" ]] && args+=(--condition "${EVAL_CONDITION}")
[[ -n "${EVAL_TARGET:-}" ]] && args+=(--target "${EVAL_TARGET}")
[[ -n "${EVAL_TRIALS:-}" ]] && args+=(--trials "${EVAL_TRIALS}")
[[ -n "${EVAL_CUTOFF:-}" ]] && args+=(--cutoff "${EVAL_CUTOFF}")
[[ -n "${EVAL_ALIGN_CMD:-}" ]] && args+=(--align-cmd "${EVAL_ALIGN_CMD}")
# shellcheck disable=SC2206
[[ -n "${EVAL_BOARD:-}" ]] && args+=(--board ${EVAL_BOARD})

if [[ -n "${DRY_RUN:-}" ]]; then
  echo "[DRY] 로봇 없이 UI만 점검합니다: ${DRY_RUN}"
  # shellcheck disable=SC2086
  run_or_print python "${TOOL}" "${args[@]}" --dry-run ${DRY_RUN}
  exit 0
fi

if [[ -z "${EVAL_ROLLOUT_CMD:-}" || -z "${EVAL_WATCH_DIR:-}" ]]; then
  echo "[ERROR] EVAL_ROLLOUT_CMD 와 EVAL_WATCH_DIR 를 설정하거나 DRY_RUN 으로 실행하세요" >&2
  exit 1
fi

echo "[RUN] 롤아웃: ${EVAL_ROLLOUT_CMD}"
echo "[RUN] 감시 폴더: ${EVAL_WATCH_DIR}"
echo "[RUN] 결과: ${OUT_DIR:-evaluation/<MMDD>_<model>_<condition>}"
run_or_print python "${TOOL}" "${args[@]}" \
  --rollout-cmd "${EVAL_ROLLOUT_CMD}" --watch-dir "${EVAL_WATCH_DIR}"
