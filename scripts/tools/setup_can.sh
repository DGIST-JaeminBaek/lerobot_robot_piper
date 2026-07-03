#!/usr/bin/env bash
set -euo pipefail

# Linux 서버에서 CAN-USB 인터페이스를 1 Mbps로 초기화하는 스크립트
# Piper SDK: CAN 포트 up 상태에서 안정 연결
# 기본값 can0/can1, 다른 이름은 환경변수로 덮어쓰기
# 예: CAN0=can_leader1 CAN1=can_follower1 bash scripts/tools/setup_can.sh

# 초기화할 CAN 인터페이스 이름
CAN0="${CAN0:-can0}"
CAN1="${CAN1:-can1}"

# Piper arm 통신에 사용하는 기본 bitrate
BITRATE="${BITRATE:-1000000}"

# gs_usb: 일반 USB-CAN 어댑터용 커널 모듈
# 이미 로드된 경우에도 modprobe 재호출 가능
sudo modprobe gs_usb

# CAN 포트별 down -> type can bitrate 설정 -> up 순서 재초기화
# down 실패: 포트가 아직 up 상태가 아닐 때 흔함, 무시
for iface in "${CAN0}" "${CAN1}"; do
  sudo ip link set "${iface}" down 2>/dev/null || true
  sudo ip link set "${iface}" type can bitrate "${BITRATE}"
  sudo ip link set "${iface}" up

  # 최종 상태 출력: bitrate와 state 확인
  ip -details link show "${iface}"
done
