#!/usr/bin/env bash
# 도형 선택(도달) 평가 세션 창을 띄운다 — 롤아웃 → 어느 도형으로 갔나 → 사람 판정 → 다음.
#
# 13__eval_session.sh("끝까지 다 지웠나")의 축소판이다. 여기서는 영상을 안 남기고
# 완주 여부도 안 본다 — 맞는 도형을 골랐는지만 본다. 그래서 롤아웃 명령은 반드시
# --mode demo 로 준다(러너가 데이터셋·영상을 애초에 안 만든다).
#
# 이 스크립트는 로봇을 직접 건드리지 않는다. 창 안의 [시작] 버튼이 REACH_ROLLOUT_CMD를
# 실행하고, 그 명령이 만든 시도 폴더를 본다. 실물 전송 여부는 전적으로 그 명령이
# 정한다(erase_run.py는 --confirm 없이는 명령을 보내지 않는다).
#
# 환경변수:
#   REACH_TARGET       circle / rectangle / triangle  (필수)
#   REACH_MODEL        smolvla / pi0 …                (집계 축 1)
#   REACH_CONDITION    distractor / shape_prompt …    (집계 축 2)
#   REACH_TRIALS       목표 시행 수 (표시용)
#   REACH_CUTOFF       시도당 컷오프 초 (기본 45 — 도달만 보므로 짧게)
#   REACH_WATCH_DIR    러너가 시도 폴더를 만드는 곳 (--mode demo 기본은 records/hil)
#   REACH_ROLLOUT_CMD  롤아웃 1회를 도는 명령 (--mode demo 로 줄 것)
#   REACH_OUT_DIR      결과 폴더 (기본 outputs/evaluation/erase_shape_reach/<오늘>_<model>_<condition>)
#   REACH_ALIGN_CMD    [정렬 도구 켜기] 명령 직접 지정 (escape hatch, 필수 아님)
#   REACH_BOARD        "x y w h" — 카메라를 옮겼으면 반드시 지정
#   DRY_RUN            지정하면 로봇 없이 기존 시도 폴더로 UI만 점검
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# run_common.sh는 자기 자신(scripts/lib) 기준으로 SCRIPT_DIR을 다시 정의한다 —
# source 뒤에 여기 SCRIPT_DIR을 그대로 쓰면 경로가 깨진다(13__eval_session.sh와 동일).
REACH_SESSION_SCRIPT_DIR="${SCRIPT_DIR}"
# shellcheck source=lib/run_common.sh
source "${SCRIPT_DIR}/lib/run_common.sh"

load_recording_env
activate_conda_env

# conda ugrp 환경의 기본 Tk는 Xft/fontconfig 없이 빌드되어 한글 글리프가 있는 글꼴이
# 아예 없다 — GUI의 한글 라벨이 □로 나온다. teleop_ui.py·13__eval_session.sh와 동일하게
# 시스템 Tcl/Tk(Xft 있음, Noto Sans CJK KR 인식)를 이 프로세스에만 LD_PRELOAD로 얹는다.
if [[ -r /lib/x86_64-linux-gnu/libtcl8.6.so && -r /lib/x86_64-linux-gnu/libtk8.6.so ]]; then
  echo "[INFO] 시스템 Tk 사용 — Noto 한글 글꼴 렌더링 적용"
  export LD_PRELOAD="/lib/x86_64-linux-gnu/libtcl8.6.so /lib/x86_64-linux-gnu/libtk8.6.so${LD_PRELOAD:+ ${LD_PRELOAD}}"
else
  echo "[WARN] 시스템 Tcl/Tk를 찾지 못해 conda Tk로 실행합니다 — 한글 가독성이 낮을 수 있음" >&2
fi

TOOL="${REACH_SESSION_SCRIPT_DIR}/tasks/erase_shape/evaluation/reach_eval_ui.py"

if [[ -z "${REACH_TARGET:-}" ]]; then
  echo "[ERROR] REACH_TARGET 을 지정하세요 (circle / rectangle / triangle)" >&2
  exit 2
fi

args=(--target "${REACH_TARGET}")
[[ -n "${REACH_OUT_DIR:-}" ]] && args+=(--out-dir "${REACH_OUT_DIR}")
[[ -n "${REACH_MODEL:-}" ]] && args+=(--model "${REACH_MODEL}")
[[ -n "${REACH_CONDITION:-}" ]] && args+=(--condition "${REACH_CONDITION}")
[[ -n "${REACH_TRIALS:-}" ]] && args+=(--trials "${REACH_TRIALS}")
[[ -n "${REACH_CUTOFF:-}" ]] && args+=(--cutoff "${REACH_CUTOFF}")
[[ -n "${REACH_ALIGN_CMD:-}" ]] && args+=(--align-cmd "${REACH_ALIGN_CMD}")
# shellcheck disable=SC2206
[[ -n "${REACH_BOARD:-}" ]] && args+=(--board ${REACH_BOARD})

if [[ -n "${DRY_RUN:-}" ]]; then
  echo "[DRY] 로봇 없이 UI만 점검합니다: ${DRY_RUN}"
  # shellcheck disable=SC2086
  run_or_print python "${TOOL}" "${args[@]}" --dry-run ${DRY_RUN}
  exit 0
fi

if [[ -z "${REACH_ROLLOUT_CMD:-}" || -z "${REACH_WATCH_DIR:-}" ]]; then
  echo "[ERROR] REACH_ROLLOUT_CMD 와 REACH_WATCH_DIR 를 함께 지정하세요" >&2
  exit 2
fi
# 영상을 안 남기는 게 이 도구의 전제다. --mode demo가 아니면 러너가 데이터셋을
# 만들기 시작하고, 그러면 시도마다 수십 초 인코딩이 붙는다(이 창은 그걸 기다리지 않는다).
if [[ "${REACH_ROLLOUT_CMD}" != *"--mode demo"* ]]; then
  echo "[WARN] REACH_ROLLOUT_CMD 에 '--mode demo' 가 없습니다 — 영상/데이터셋이 남고" >&2
  echo "       시도마다 인코딩 시간이 붙습니다. 의도한 게 아니면 추가하세요." >&2
fi

run_or_print python "${TOOL}" "${args[@]}" \
  --watch-dir "${REACH_WATCH_DIR}" \
  --rollout-cmd "${REACH_ROLLOUT_CMD}"
