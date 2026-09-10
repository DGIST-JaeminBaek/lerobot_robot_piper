#!/usr/bin/env bash
set -uo pipefail

source /home/ugrp43/miniconda3/etc/profile.d/conda.sh
conda activate ugrp
cd /home/ugrp43/UGRP/lerobot_robot_piper

RUN_NAME="smolvla_erase_shape_augmented_276_trimmed_8ep_gpu1"
OUTPUT_DIR="/home/ugrp43/UGRP/lerobot_robot_piper/outputs/train/${RUN_NAME}"
LOG_PATH="/home/ugrp43/UGRP/lerobot_robot_piper/outputs/train/${RUN_NAME}.log"

exec > >(tee "${LOG_PATH}") 2>&1

echo "GPU=1"
echo "DATASET=records/outputs/erase_shape_augmented_276_trimmed"
echo "FRAMES=147159 BATCH=8 STEPS=147159 EPOCHS=8.0000"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES=1 lerobot-train \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.train_state_proj=true \
  --policy.optimizer_lr=1e-4 \
  --policy.optimizer_weight_decay=1e-10 \
  --policy.optimizer_grad_clip_norm=10 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=147159 \
  --policy.scheduler_decay_lr=2.5e-6 \
  --dataset.repo_id=local/erase_shape_augmented_276_trimmed \
  --dataset.root=/home/ugrp43/UGRP/lerobot_robot_piper/records/outputs/erase_shape_augmented_276_trimmed \
  --dataset.video_backend=pyav \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${RUN_NAME}" \
  --batch_size=8 \
  --num_workers=2 \
  --steps=147159 \
  --seed=1000 \
  --log_freq=100 \
  --save_freq=15000 \
  --eval_freq=0 \
  --wandb.enable=true \
  --wandb.project=smolvla-erase-shape \
  --wandb.run_id="${RUN_NAME}" \
  --wandb.disable_artifact=true

status=$?
echo "TRAIN_EXIT_STATUS=${status}"
exit "${status}"
