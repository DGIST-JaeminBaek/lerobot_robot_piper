#!/usr/bin/env bash
# GPU0에서 72개(구 프롬프트 "erase the shape") -> 135개(신규 "return to position"
# 프롬프트, kwak 폴더)를 순차로 학습한다. 같은 GPU라 동시 실행하면 서로 느려지므로
# 앞 job이 끝난 뒤 뒤 job이 시작되도록 &&로 묶는다. 앞이 실패하면 뒤는 안 돈다.
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ugrp
cd /home/ugrp43/UGRP/lerobot_robot_piper

echo "[1/2] smolvla_erase_prompt_0812_0813am_72 시작"
CUDA_VISIBLE_DEVICES=0 \
ACCELERATE_MIXED_PRECISION=bf16 \
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.input_features=null \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/erase_the_shape_0812_0813am_72 \
  --dataset.root=/home/ugrp43/UGRP/lerobot_robot_piper/records/outputs/erase_the_shape/erase_the_shape_0812_0813am_72 \
  --dataset.video_backend=pyav \
  --output_dir=/home/ugrp43/UGRP/lerobot_robot_piper/outputs/train/erase_the_shape/smolvla_erase_prompt_0812_0813am_72 \
  --job_name=smolvla_erase_prompt_0812_0813am_72 \
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
  --wandb.run_id=smolvla_erase_prompt_0812_0813am_72 \
  --wandb.disable_artifact=true

echo "[2/2] smolvla_return_prompt_0802_0804_0805_0813pm_135 시작"
CUDA_VISIBLE_DEVICES=0 \
ACCELERATE_MIXED_PRECISION=bf16 \
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.input_features=null \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/kwak_pick_up_the_eraser_0802_0804_0805_0813pm_135 \
  --dataset.root=/home/ugrp43/UGRP/lerobot_robot_piper/records/outputs/kwak/pick_up_the_eraser_0802_0804_0805_0813pm_135 \
  --dataset.video_backend=pyav \
  --output_dir=/home/ugrp43/UGRP/lerobot_robot_piper/outputs/train/kwak/smolvla_return_prompt_0802_0804_0805_0813pm_135 \
  --job_name=smolvla_return_prompt_0802_0804_0805_0813pm_135 \
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
  --wandb.run_id=smolvla_return_prompt_0802_0804_0805_0813pm_135 \
  --wandb.disable_artifact=true

echo "GPU0 두 학습 모두 완료"
