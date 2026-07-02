#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=run_common.sh
source "${SCRIPT_DIR}/run_common.sh"

load_recording_env

BITRATE="${BITRATE:-1000000}"
LEADER_PORT="${LEADER_PORT:-can_leader1}"
FOLLOWER_PORT="${FOLLOWER_PORT:-can_follower1}"
LEADER_USB_BUS="${LEADER_USB_BUS:-}"
FOLLOWER_USB_BUS="${FOLLOWER_USB_BUS:-}"
IGNORE_CHECK="${IGNORE_CHECK:-false}"

sudo modprobe gs_usb

declare -A USB_PORTS=()
if [[ -n "${LEADER_USB_BUS}" ]]; then
  USB_PORTS["${LEADER_USB_BUS}"]="${LEADER_PORT}:${BITRATE}"
fi
if [[ -n "${FOLLOWER_USB_BUS}" ]]; then
  USB_PORTS["${FOLLOWER_USB_BUS}"]="${FOLLOWER_PORT}:${BITRATE}"
fi

if [[ "${#USB_PORTS[@]}" -eq 0 ]]; then
  echo "[INFO] LEADER_USB_BUS/FOLLOWER_USB_BUS are not set; initializing existing named interfaces."
  for iface in "${LEADER_PORT}" "${FOLLOWER_PORT}"; do
    if ! ip link show "${iface}" >/dev/null 2>&1; then
      echo "[WARN] CAN interface not found: ${iface}"
      continue
    fi
    sudo ip link set "${iface}" down 2>/dev/null || true
    sudo ip link set "${iface}" type can bitrate "${BITRATE}"
    sudo ip link set "${iface}" up
    ip -details link show "${iface}"
  done
  exit 0
fi

current_can_count="$(ip link show type can | grep -c "link/can" || true)"
if [[ "${IGNORE_CHECK}" != "true" && "${current_can_count}" -ne "${#USB_PORTS[@]}" ]]; then
  echo "[WARN] Detected ${current_can_count} CAN interfaces, expected ${#USB_PORTS[@]}."
  read -r -p "Continue anyway? (y/N): " reply
  case "${reply}" in
    y|Y|yes|YES) ;;
    *) exit 1 ;;
  esac
fi

declare -A TARGET_NAMES=()
for bus in "${!USB_PORTS[@]}"; do
  IFS=':' read -r target_name _target_bitrate <<< "${USB_PORTS[$bus]}"
  if [[ -n "${TARGET_NAMES[$target_name]:-}" ]]; then
    echo "[ERROR] Duplicate target CAN name: ${target_name}" >&2
    exit 1
  fi
  TARGET_NAMES["${target_name}"]=1
done

success_count=0
for iface in $(ip -br link show type can | awk '{print $1}'); do
  echo "--------------------------- ${iface} ------------------------------"
  bus_info="$(sudo ethtool -i "${iface}" | awk '/bus-info/ {print $2}')"
  if [[ -z "${bus_info}" ]]; then
    echo "[WARN] Could not read USB bus-info for ${iface}"
    continue
  fi
  echo "[INFO] ${iface} is on USB bus ${bus_info}"

  if [[ -z "${USB_PORTS[$bus_info]:-}" ]]; then
    echo "[WARN] USB bus ${bus_info} is not configured."
    continue
  fi

  IFS=':' read -r target_name target_bitrate <<< "${USB_PORTS[$bus_info]}"
  sudo ip link set "${iface}" down 2>/dev/null || true
  sudo ip link set "${iface}" type can bitrate "${target_bitrate}"
  if [[ "${iface}" != "${target_name}" ]]; then
    if ip link show "${target_name}" >/dev/null 2>&1; then
      echo "[ERROR] Target interface already exists: ${target_name}" >&2
      exit 1
    fi
    sudo ip link set "${iface}" name "${target_name}"
  fi
  sudo ip link set "${target_name}" up
  ip -details link show "${target_name}"
  success_count=$((success_count + 1))
done

echo "[RESULT] Processed ${success_count} configured CAN interface(s)."
