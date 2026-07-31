#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${ENV_FILE:-${REPO_DIR}/configs/recording.env}"

load_recording_env() {
  if [[ -f "${ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
  else
    echo "[WARN] Missing env file: ${ENV_FILE}" >&2
    echo "[WARN] Copy configs/recording.env.example to configs/recording.env for persistent settings." >&2
  fi
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[ERROR] Required command not found: $1" >&2
    return 1
  fi
}

bool_default() {
  local value="${1:-}"
  local fallback="$2"
  if [[ -n "${value}" ]]; then
    printf '%s' "${value}"
  else
    printf '%s' "${fallback}"
  fi
}

robot_camera_args() {
  # LeRobot 0.4.4 dict 파서 우회용 카메라 인자
  local camera_type="${CAMERA_TYPE:-opencv}"
  local top_cam_type="${TOP_CAM_TYPE:-${camera_type}}"
  local wrist_cam_type="${WRIST_CAM_TYPE:-${camera_type}}"
  local top_cam="${TOP_CAM-0}"
  local wrist_cam="${WRIST_CAM-1}"
  local width="${CAM_WIDTH:-640}"
  local height="${CAM_HEIGHT:-480}"
  local fps="${FPS:-30}"
  local realsense_use_depth="${REALSENSE_USE_DEPTH:-false}"
  local realsense_warmup_s="${REALSENSE_WARMUP_S:-5.0}"
  local camera_connect_warmup="${CAMERA_CONNECT_WARMUP:-false}"
  local camera_post_connect_wait_s="${CAMERA_POST_CONNECT_WAIT_S:-2.0}"

  printf '%s\n' \
    "--robot.camera_type=${camera_type}" \
    "--robot.top_cam_type=${top_cam_type}" \
    "--robot.wrist_cam_type=${wrist_cam_type}" \
    "--robot.top_cam=${top_cam}" \
    "--robot.wrist_cam=${wrist_cam}" \
    "--robot.cam_width=${width}" \
    "--robot.cam_height=${height}" \
    "--robot.camera_fps=${fps}" \
    "--robot.realsense_use_depth=${realsense_use_depth}" \
    "--robot.realsense_warmup_s=${realsense_warmup_s}" \
    "--robot.camera_connect_warmup=${camera_connect_warmup}" \
    "--robot.camera_post_connect_wait_s=${camera_post_connect_wait_s}" \
    "--robot.top_realsense_use_depth=${TOP_REALSENSE_USE_DEPTH:-${realsense_use_depth}}" \
    "--robot.wrist_realsense_use_depth=${WRIST_REALSENSE_USE_DEPTH:-${realsense_use_depth}}"
}

robot_observation_args() {
  # observation 확장(effort/velocity) 인자. depth는 robot_camera_args()의
  # realsense_use_depth 계열이 담당한다.
  # teleop_ui.py의 _observation_toggle_args()와 같은 값을 내보내야 한다 —
  # GUI로 찍은 데이터셋과 셸로 찍은 데이터셋의 features가 다르면 나중에 머지가 안 된다
  # (lerobot/datasets/aggregate.py의 validate_all_metadata가 ValueError를 던짐).
  #
  # USE_EFFORT는 기본 true. 안 찍은 effort는 되살릴 수 없고, 켜도 프레임당 52바이트에
  # 비디오 인코딩과 무관하므로 "일단 항상 찍는다"가 기본 방침이다.
  # 7차원 state로 학습한 옛 체크포인트를 돌릴 때만 false로 내릴 것.
  printf '%s\n' \
    "--robot.use_effort=$(bool_default "${USE_EFFORT:-}" true)"
}

robot_action_offset_args() {
  # leader/follower 시작 자세 차이 보정 인자
  printf '%s\n' \
    "--robot.park_on_connect=$(bool_default "${PARK_ON_CONNECT:-}" false)" \
    "--robot.use_action_offset=$(bool_default "${USE_ACTION_OFFSET:-}" true)" \
    "--robot.use_manual_action_offset=$(bool_default "${USE_MANUAL_ACTION_OFFSET:-}" false)" \
    "--robot.action_offset_warmup_s=${ACTION_OFFSET_WARMUP_S:-1.5}" \
    "--robot.action_offset_report_threshold=${ACTION_OFFSET_REPORT_THRESHOLD:-3.0}" \
    "--robot.action_offset_joint1=${ACTION_OFFSET_JOINT1:-0.0}" \
    "--robot.action_offset_joint2=${ACTION_OFFSET_JOINT2:-0.0}" \
    "--robot.action_offset_joint3=${ACTION_OFFSET_JOINT3:-0.0}" \
    "--robot.action_offset_joint4=${ACTION_OFFSET_JOINT4:-0.0}" \
    "--robot.action_offset_joint5=${ACTION_OFFSET_JOINT5:-0.0}" \
    "--robot.action_offset_joint6=${ACTION_OFFSET_JOINT6:-0.0}" \
    "--robot.action_offset_gripper=${ACTION_OFFSET_GRIPPER:-0.0}"
}

print_command() {
  printf '[CMD]'
  for part in "$@"; do
    printf ' %q' "${part}"
  done
  printf '\n'
}

run_or_print() {
  print_command "$@"
  if [[ "${DRY_RUN:-false}" == "true" ]]; then
    return 0
  fi
  "$@"
}

robot_safety_args() {
  # DISABLE_TORQUE_ON_DISCONNECT=false로 두면 disconnect() 시 parking 자세로는
  # 이동하되 torque는 자동으로 풀지 않음 — scripts/tools/safe_release_torque.py로
  # 사람이 팔을 잡은 상태에서 수동으로 torque를 해제하는 루틴과 짝을 이룸.
  #
  # MAX_RELATIVE_TARGET=off(또는 none/null/disabled, 대소문자 무관)로 두면 상대
  # 이동량 제한을 완전히 끔(draccus에 --robot.max_relative_target=null로 전달 —
  # 이 필드가 float | dict | None이라 파이썬 문자열 "None"은 안 먹히고 YAML
  # null/~ 표기만 None으로 디코딩됨). 리더-팔로워 괴리감이 큰 상황에서 5.0이
  # 너무 자주 걸릴 때 임시로 끄고 조심스럽게 테스트할 용도 — 상시 끄기보다는
  # 필요할 때만 켰다 끄는 걸 권장.
  local max_rel="${MAX_RELATIVE_TARGET:-5.0}"
  case "${max_rel,,}" in
    off | none | null | disabled) max_rel="null" ;;
  esac
  # SAFETY_ON_OVERLOAD=park면 임계값 초과 시 parking 자세로 복귀하고 그 뒤 명령을
  # 전부 차단(래치), hold면 기존처럼 그 자리에서 명령만 보류한다.
  #
  # PARK_RELEASE_MODE는 종료 시 "어떤 자세에서 torque를 푸는지"를 정한다 —
  # lower(팔은 그대로, 손목만 PARK_RELEASE_WRIST_REST_DEG 각도까지 미리 내린 뒤 해제) /
  # in_place(그 자리에서 바로) / park(기존: 파킹 자세로 이동 후).
  # 실측상 torque를 풀 때 떨어지는 건 손목뿐이라(joint1~4/6은 0.00도) lower가 기본.
  printf '%s\n' \
    "--robot.max_relative_target=${max_rel}" \
    "--robot.disable_torque_on_disconnect=$(bool_default "${DISABLE_TORQUE_ON_DISCONNECT:-}" true)" \
    "--robot.safety_enabled=$(bool_default "${SAFETY_ENABLED:-}" true)" \
    "--robot.safety_effort_limit=${SAFETY_EFFORT_LIMIT:-8.0}" \
    "--robot.safety_on_overload=${SAFETY_ON_OVERLOAD:-park}" \
    "--robot.safety_park_ramp_s=${SAFETY_PARK_RAMP_S:-4.0}" \
    "--robot.safety_hold_resend=$(bool_default "${SAFETY_HOLD_RESEND:-}" true)" \
    "--robot.park_release_mode=${PARK_RELEASE_MODE:-lower}" \
    "--robot.park_release_wrist_rest_deg=${PARK_RELEASE_WRIST_REST_DEG:-24.4}" \
    "--robot.park_release_gripper_cycle=$(bool_default "${PARK_RELEASE_GRIPPER_CYCLE:-}" true)" \
    "--robot.park_release_gripper_open=${PARK_RELEASE_GRIPPER_OPEN:-100.0}" \
    "--robot.park_release_gripper_wait_s=${PARK_RELEASE_GRIPPER_WAIT_S:-1.5}" \
    "--robot.park_release_ramp_s=${PARK_RELEASE_RAMP_S:-2.0}" \
    "--robot.park_release_settle_s=${PARK_RELEASE_SETTLE_S:-0.5}"
}

plugin_discovery_args() {
  printf '%s\n' \
    "--robot.discover_packages_path=lerobot_robot_piper" \
    "--teleop.discover_packages_path=lerobot_robot_piper"
}

task_slug() {
  # TASK 문자열을 폴더/repo_id 세그먼트로 쓸 수 있게 슬러그화.
  # 소문자로 바꾸고 영숫자 외 문자는 전부 "_"로(연속 문자는 하나로 뭉침),
  # 앞뒤 "_"는 제거. tr의 [:alnum:]은 한글 등 비-ASCII를 문자로 인식 못 해서
  # 통째로 사라짐(빈 문자열이 됨) — 이 프로젝트는 영어 TASK 문자열만 쓰므로
  # 문제없음. teleop_ui.py의 _task_slug()와 동일한 규칙을 유지할 것.
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '_' | sed 's/^_*//;s/_*$//'
}

replace_last_path_segment() {
  # "$1"의 마지막 "/" 다음 세그먼트를 "$2"로 교체. "/"가 없으면 통째로 "$2"로 교체.
  # 예: local/piper_write_light + pick_up_the_pen -> local/pick_up_the_pen
  local path="$1" replacement="$2"
  if [[ "${path}" == */* ]]; then
    printf '%s/%s' "${path%/*}" "${replacement}"
  else
    printf '%s' "${replacement}"
  fi
}
