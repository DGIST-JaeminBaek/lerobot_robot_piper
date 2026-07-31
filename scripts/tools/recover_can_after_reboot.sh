#!/usr/bin/env bash
# 재부팅 후 CAN 복구 — can0/can1로 돌아간 인터페이스를 bring-up하고, 실제 역할을
# 판별해서 can_leader/can_follower로 이름을 바꾼 뒤 다시 UP 시킨다.
#
# 이 시스템에는 CAN 이름을 고정하는 udev 규칙이 없어서 재부팅할 때마다 이름이 풀린다.
# scripts/1__init_can.sh는 recording.env의 LEADER_USB_BUS/FOLLOWER_USB_BUS가 비어 있으면
# "이미 그 이름인 인터페이스"만 초기화하므로, 이름이 풀린 직후에는 이 스크립트를 쓴다.
# (역할 판별 결과로 USB bus-info도 같이 출력하니, recording.env의 *_USB_BUS에 넣어두면
# 다음부터는 1__init_can.sh만으로 끝난다.)
#
# 사용: bash scripts/tools/recover_can_after_reboot.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_DIR}/scripts/lib/run_common.sh"
load_recording_env

BITRATE="${BITRATE:-1000000}"
LEADER_PORT="${LEADER_PORT:-can_leader}"
FOLLOWER_PORT="${FOLLOWER_PORT:-can_follower}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "=== [1/4] gs_usb 로드 + 원시 인터페이스 bring-up (bitrate ${BITRATE}) ==="
sudo modprobe gs_usb
mapfile -t RAW < <(ip -br link show type can | awk '{print $1}')
if [[ "${#RAW[@]}" -eq 0 ]]; then
  echo "[ERROR] CAN 인터페이스가 하나도 없습니다 — USB 연결과 로봇 전원을 확인하세요." >&2
  exit 1
fi
echo "[INFO] 발견된 인터페이스: ${RAW[*]}"

for iface in "${RAW[@]}"; do
  sudo ip link set "${iface}" down 2>/dev/null || true
  sudo ip link set "${iface}" type can bitrate "${BITRATE}"
  sudo ip link set "${iface}" up
done

echo
echo "=== [2/4] 역할 판별 (ctrl_mode 실측) ==="
mapfile -t ROLES < <("${PYTHON_BIN}" "${SCRIPT_DIR}/detect_can_roles.py" "${RAW[@]}")
printf '%s\n' "${ROLES[@]}"

echo
echo "=== [3/4] 이름 변경 ==="
for line in "${ROLES[@]}"; do
  iface="${line%% *}"
  role="${line##* }"
  case "${role}" in
    leader) target="${LEADER_PORT}" ;;
    follower) target="${FOLLOWER_PORT}" ;;
    *)
      echo "[WARN] ${iface}: 역할 판별 실패 — 이름을 바꾸지 않습니다."
      echo "       (로봇 전원이 꺼져 있거나 팔이 응답하지 않는 경우입니다)"
      continue
      ;;
  esac

  if [[ "${iface}" == "${target}" ]]; then
    echo "[OK] ${iface}: 이미 ${target}"
    continue
  fi
  # 이름 변경은 DOWN 상태에서만 가능
  sudo ip link set "${iface}" down
  sudo ip link set "${iface}" name "${target}"
  sudo ip link set "${target}" up
  bus_info="$(sudo ethtool -i "${target}" 2>/dev/null | awk -F': ' '/bus-info/{print $2}')"
  echo "[OK] ${iface} -> ${target} (${role}, USB ${bus_info})"
  echo "     recording.env에 넣어두면 다음부터 1__init_can.sh만으로 됩니다:"
  if [[ "${role}" == "leader" ]]; then
    echo "       LEADER_USB_BUS=${bus_info}"
  else
    echo "       FOLLOWER_USB_BUS=${bus_info}"
  fi
done

echo
echo "=== [4/4] 결과 ==="
ip -br link show type can
