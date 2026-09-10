#!/usr/bin/env bash
# 지정한 PID가 끝날 때까지 기다렸다가 명령을 실행한다 — GPU를 쓰는 학습을 줄 세울 때 쓴다.
#
# 두 학습을 같은 GPU에 동시에 올리면 둘 다 느려지므로(실측: 처리량 절반), 앞 학습이
# 끝난 뒤 자동으로 다음이 시작되게 한다. tmux 세션 안에서 돌리면 터미널을 닫아도 살아 있다.
#
# 사용:
#   scripts/training/queue_after_pid.sh <기다릴PID> <실행할스크립트> [로그파일]
set -uo pipefail

WAIT_PID="${1:?기다릴 PID}"
NEXT_CMD="${2:?실행할 스크립트 경로}"
LOG="${3:-}"

echo "[QUEUE] PID ${WAIT_PID} 종료 대기 중… (시작 $(date '+%F %T'))"
if [[ -d "/proc/${WAIT_PID}" ]]; then
  # 다른 셸이 띄운 프로세스라 wait를 못 쓴다 — /proc 폴링으로 기다린다.
  while [[ -d "/proc/${WAIT_PID}" ]]; do
    sleep 60
  done
else
  echo "[QUEUE] PID ${WAIT_PID}가 이미 없음 — 바로 시작한다"
fi

echo "[QUEUE] 앞 작업 종료 확인 ($(date '+%F %T')) — 30초 뒤 시작 (GPU 메모리 반환 대기)"
sleep 30

echo "[QUEUE] 실행: ${NEXT_CMD}"
if [[ -n "${LOG}" ]]; then
  mkdir -p "$(dirname "${LOG}")"
  bash "${NEXT_CMD}" 2>&1 | tee -a "${LOG}"
else
  bash "${NEXT_CMD}"
fi
