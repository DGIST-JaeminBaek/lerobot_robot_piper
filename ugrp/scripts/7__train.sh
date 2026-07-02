#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=run_common.sh
source "${SCRIPT_DIR}/run_common.sh"

load_recording_env
require_cmd lerobot-train

DATASET_REPO_ID="${DATASET_REPO_ID:-local/piper_write_light}"
POLICY_TYPE="${POLICY_TYPE:-smolvla}"
POLICY_REPO_ID="${POLICY_REPO_ID:-}"
POLICY_PRETRAINED_PATH="${POLICY_PRETRAINED_PATH:-}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/piper_${POLICY_TYPE}}"
JOB_NAME="${JOB_NAME:-piper_${POLICY_TYPE}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS="${STEPS:-5000}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
EVAL_FREQ="${EVAL_FREQ:-0}"
WANDB_ENABLE="$(bool_default "${WANDB_ENABLE:-}" false)"
RESUME="$(bool_default "${RESUME:-}" false)"
PUSH_POLICY_TO_HUB="$(bool_default "${PUSH_POLICY_TO_HUB:-}" false)"

cmd=(
  lerobot-train
  "--policy.type=${POLICY_TYPE}"
  "--policy.device=${POLICY_DEVICE}"
  "--dataset.repo_id=${DATASET_REPO_ID}"
  "--output_dir=${OUTPUT_DIR}"
  "--job_name=${JOB_NAME}"
  "--batch_size=${BATCH_SIZE}"
  "--steps=${STEPS}"
  "--save_freq=${SAVE_FREQ}"
  "--eval_freq=${EVAL_FREQ}"
  "--wandb.enable=${WANDB_ENABLE}"
  "--resume=${RESUME}"
  "--policy.push_to_hub=${PUSH_POLICY_TO_HUB}"
)

if [[ -n "${POLICY_REPO_ID}" ]]; then
  cmd+=("--policy.repo_id=${POLICY_REPO_ID}")
fi
if [[ -n "${POLICY_PRETRAINED_PATH}" ]]; then
  cmd+=("--policy.pretrained_path=${POLICY_PRETRAINED_PATH}")
fi

run_or_print "${cmd[@]}" "$@"
