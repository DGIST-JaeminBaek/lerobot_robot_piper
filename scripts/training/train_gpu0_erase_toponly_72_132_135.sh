#!/usr/bin/env bash
# 2026-08-19 00:00까지 대기했다가, GPU0에서 72 -> 132 -> 135 순으로
# erase(구) 프롬프트/top-only SmolVLA를 순차 학습한다. 각 job의 steps/save_freq/
# scheduler_decay_steps는 같은 데이터셋의 다른 런과 동일하게 맞췄다.
set -euo pipefail

TARGET_EPOCH=$(date -d "2026-08-19 00:00:00" +%s)
echo "[WAIT] $(date -d @${TARGET_EPOCH}) 까지 대기 중..."
while [ "$(date +%s)" -lt "${TARGET_EPOCH}" ]; do
  sleep 30
done
echo "[WAIT] 시각 도달. GPU0 학습 시작."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ugrp
cd /home/ugrp43/UGRP/lerobot_robot_piper

echo "[1/3] smolvla_erase_prompt_toponly_0812_0813am_72 시작"
CUDA_VISIBLE_DEVICES=0 \
ACCELERATE_MIXED_PRECISION=bf16 \
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.input_features=null \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/erase_the_shape_0812_0813am_72_top_only \
  --dataset.root=/home/ugrp43/UGRP/lerobot_robot_piper/records/outputs/erase_the_shape/erase_the_shape_0812_0813am_72_top_only \
  --dataset.video_backend=pyav \
  --output_dir=/home/ugrp43/UGRP/lerobot_robot_piper/outputs/train/erase_the_shape/smolvla_erase_prompt_toponly_0812_0813am_72 \
  --job_name=smolvla_erase_prompt_toponly_0812_0813am_72 \
  --batch_size=8 \
  --num_workers=2 \
  --steps=33280 \
  --log_freq=100 \
  --save_freq=4160 \
  --eval_freq=0 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=33280 \
  --wandb.enable=true \
  --wandb.project=smolvla-erase-shape \
  --wandb.run_id=smolvla_erase_prompt_toponly_0812_0813am_72 \
  --wandb.disable_artifact=true

echo "[2/3] smolvla_erase_prompt_toponly_0727_0812_0813am_132 시작"
CUDA_VISIBLE_DEVICES=0 \
ACCELERATE_MIXED_PRECISION=bf16 \
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.input_features=null \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/erase_the_shape_0727_0812_0813am_132_top_only \
  --dataset.root=/home/ugrp43/UGRP/lerobot_robot_piper/records/outputs/erase_the_shape/erase_the_shape_0727_0812_0813am_132_top_only \
  --dataset.video_backend=pyav \
  --output_dir=/home/ugrp43/UGRP/lerobot_robot_piper/outputs/train/erase_the_shape/smolvla_erase_prompt_toponly_0727_0812_0813am_132 \
  --job_name=smolvla_erase_prompt_toponly_0727_0812_0813am_132 \
  --batch_size=8 \
  --num_workers=2 \
  --steps=65265 \
  --log_freq=100 \
  --save_freq=5000 \
  --eval_freq=0 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=65265 \
  --wandb.enable=true \
  --wandb.project=smolvla-erase-shape \
  --wandb.run_id=smolvla_erase_prompt_toponly_0727_0812_0813am_132 \
  --wandb.disable_artifact=true

echo "[3/3] smolvla_erase_prompt_toponly_0802_0804_0805_0813pm_135 시작"
CUDA_VISIBLE_DEVICES=0 \
ACCELERATE_MIXED_PRECISION=bf16 \
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.input_features=null \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/erase_the_shape_0802_0804_0805_0813pm_135_top_only \
  --dataset.root=/home/ugrp43/UGRP/lerobot_robot_piper/records/outputs/erase_the_shape/erase_the_shape_0802_0804_0805_0813pm_135_top_only \
  --dataset.video_backend=pyav \
  --output_dir=/home/ugrp43/UGRP/lerobot_robot_piper/outputs/train/erase_the_shape/smolvla_erase_prompt_toponly_0802_0804_0805_0813pm_135 \
  --job_name=smolvla_erase_prompt_toponly_0802_0804_0805_0813pm_135 \
  --batch_size=8 \
  --num_workers=2 \
  --steps=71909 \
  --log_freq=100 \
  --save_freq=8989 \
  --eval_freq=0 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=71909 \
  --wandb.enable=true \
  --wandb.project=smolvla-erase-shape \
  --wandb.run_id=smolvla_erase_prompt_toponly_0802_0804_0805_0813pm_135 \
  --wandb.disable_artifact=true

echo "GPU0 세 학습 모두 완료"
