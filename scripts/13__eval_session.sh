#!/usr/bin/env bash
# 평가 세션 창을 띄운다 — 롤아웃 실행 → 자동 채점 → 사람 확인 → 다음.
#
# 이 스크립트는 로봇을 직접 건드리지 않는다. 창 안의 [시작] 버튼이 EVAL_ROLLOUT_CMD를
# 실행하고, 그 명령이 만든 에피소드를 채점한다. 실물 전송 여부는 전적으로 그 명령이
# 정한다(erase_run.py는 --confirm 없이는 명령을 보내지 않는다).
#
# 사용 전 확인:
#   1) 하드웨어 없이 UI 점검  : DRY_RUN='<데이터셋>/erase_the_*' bash scripts/13__eval_session.sh
#   2) 현장 카메라 확인       : python scripts/tasks/erase_shape/lib/ink_metric.py <에피소드> --dump-roi /tmp/roi.png
#   3) 실제 세션              : 아래 변수를 채우고 실행
#
# 환경변수:
#   EVAL_MODEL       smolvla / pi0 …            (집계 축 1)
#   EVAL_CONDITION   async / sync …             (집계 축 2)
#   EVAL_TARGET      circle / rectangle / triangle (생략 시 폴더명에서 추론)
#   EVAL_TRIALS      목표 시행 수 (표시용)
#   EVAL_CUTOFF      에피소드 컷오프 초 (기본 60)
#   EVAL_ALIGN_CMD   [정렬 도구 켜기] 버튼으로 띄울 명령 직접 지정 (escape hatch).
#                    필수 아님 — 안 주면 GUI가 EVAL_TARGET/EVAL_BOARD 등으로
#                    block_alignment_tool.py --shape-zones 315를 자동 구성해서 쓴다.
#                    켜져 있는 동안 [시작]이 잠긴다.
#   EVAL_OUT_DIR     결과 폴더 (기본 outputs/evaluation/erase_shape/<오늘>_<model>_<condition>)
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

# conda ugrp 환경의 기본 Tk는 Xft/fontconfig 없이 빌드되어 한글 글리프가 있는
# 글꼴이 아예 없다 — erase_eval_ui.py의 한글 라벨이 □로 나온다. teleop_ui.py와
# 동일하게(scripts/0__launch_gui.sh, docs/operations.md §7) 시스템 Tcl/Tk(Xft 있음,
# Noto Sans CJK KR 인식)를 이 GUI 프로세스에만 LD_PRELOAD로 얹는다. Python·프로젝트
# 패키지는 여전히 ugrp 환경 것을 쓴다.
SYSTEM_TK_PRELOAD="/lib/x86_64-linux-gnu/libtcl8.6.so /lib/x86_64-linux-gnu/libtk8.6.so"
if [[ -r /lib/x86_64-linux-gnu/libtcl8.6.so && -r /lib/x86_64-linux-gnu/libtk8.6.so ]]; then
  echo "[INFO] 시스템 Tk 사용 — Noto 한글 글꼴 렌더링 적용"
  export LD_PRELOAD="${SYSTEM_TK_PRELOAD}${LD_PRELOAD:+ ${LD_PRELOAD}}"
else
  echo "[WARN] 시스템 Tcl/Tk를 찾지 못해 conda Tk로 실행합니다 — 한글 가독성이 낮을 수 있음" >&2
fi

TOOL="${EVAL_SESSION_SCRIPT_DIR}/tasks/erase_shape/evaluation/erase_eval_ui.py"
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
echo "[RUN] 결과: ${OUT_DIR:-outputs/evaluation/erase_shape/<MMDD>_<model>_<condition>}"
run_or_print python "${TOOL}" "${args[@]}" \
  --rollout-cmd "${EVAL_ROLLOUT_CMD}" --watch-dir "${EVAL_WATCH_DIR}"
