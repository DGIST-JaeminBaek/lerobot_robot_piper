#!/usr/bin/env bash
# GPU1에서 도는 erase_prompt 132->135 체인(PID 2985491, train_gpu1_erase_132_135.sh)이
# 완전히 끝날 때까지 기다렸다가, 315개 pickup(신) 프롬프트 top-only SmolVLA를 이어서
# 돌린다. 개별 lerobot-train PID가 아니라 체인 스크립트 자체의 PID를 기다려야
# 132만 끝나고 135와 동시에 도는 사고를 막는다.
set -euo pipefail

WAIT_PID=2985491
echo "[WAIT] PID ${WAIT_PID}(train_gpu1_erase_132_135.sh, 132->135 체인) 종료 대기 중..."
while kill -0 "${WAIT_PID}" 2>/dev/null; do
  sleep 30
done
echo "[WAIT] 종료 확인. GPU1 315개 pickup top-only 학습 시작."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ugrp
cd /home/ugrp43/UGRP/lerobot_robot_piper

CUDA_VISIBLE_DEVICES=1 \
ACCELERATE_MIXED_PRECISION=bf16 \
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.input_features=null \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/pick_up_the_eraser_315_top_only \
  --dataset.root=/home/ugrp43/UGRP/lerobot_robot_piper/records/outputs/pick_up_the_eraser_and_erase_the_shape/pick_up_the_eraser_315_top_only \
  --dataset.video_backend=pyav \
  --output_dir=/home/ugrp43/UGRP/lerobot_robot_piper/outputs/train/pick_up_the_eraser_and_erase_the_shape/smolvla_pickup_prompt_toponly_315 \
  --job_name=smolvla_pickup_prompt_toponly_315 \
  --batch_size=8 \
  --num_workers=2 \
  --steps=168000 \
  --log_freq=100 \
  --save_freq=15000 \
  --eval_freq=0 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=168000 \
  --wandb.enable=true \
  --wandb.project=smolvla-erase-shape \
  --wandb.run_id=smolvla_pickup_prompt_toponly_315 \
  --wandb.disable_artifact=true

echo "[DONE] smolvla_pickup_prompt_toponly_315 완료."
