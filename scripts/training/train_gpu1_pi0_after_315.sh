#!/usr/bin/env bash
# GPU1에서 도는 smolvla_pickup_prompt_315(PID 1644167)가 끝날 때까지 기다렸다가,
# 같은 GPU1에 315개 pi0 LoRA(신규 프롬프트 "pick up the eraser and erase the shape")를
# 이어서 돌린다. steps/scheduler_decay_steps는 방금 그 SmolVLA 315 런과 동일하게
# 168000으로 맞춰서 나중에 비교할 수 있게 했다.
set -euo pipefail

WAIT_PID=1644167
echo "[WAIT] PID ${WAIT_PID}(smolvla_pickup_prompt_315) 종료 대기 중..."
while kill -0 "${WAIT_PID}" 2>/dev/null; do
  sleep 30
done
echo "[WAIT] 종료 확인. pi0 315개 학습 시작."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pi0
cd /home/ugrp43/UGRP/lerobot_robot_piper

CUDA_VISIBLE_DEVICES=1 \
lerobot-train \
  --policy.path=/home/ugrp43/.cache/huggingface/hub/models--lerobot--pi0_base/snapshots/26b99b9439acb1e352439e34ee9c67af0d76efa3 \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.push_to_hub=false \
  --peft.method_type=LORA \
  --peft.r=32 \
  --peft.lora_alpha=64 \
  --dataset.repo_id=local/pick_up_the_eraser_315 \
  --dataset.root=/home/ugrp43/UGRP/lerobot_robot_piper/records/outputs/pick_up_the_eraser_and_erase_the_shape/pick_up_the_eraser_315 \
  --dataset.video_backend=pyav \
  --rename_map='{"observation.images.top": "observation.images.base_0_rgb", "observation.images.wrist": "observation.images.left_wrist_0_rgb"}' \
  --output_dir=/home/ugrp43/UGRP/lerobot_robot_piper/outputs/train/pick_up_the_eraser_and_erase_the_shape/pi0_lora_pickup_prompt_315 \
  --job_name=pi0_lora_pickup_prompt_315 \
  --batch_size=8 \
  --num_workers=2 \
  --steps=168000 \
  --log_freq=100 \
  --save_freq=5000 \
  --eval_freq=0 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=168000 \
  --wandb.enable=true \
  --wandb.project=smolvla-erase-shape \
  --wandb.run_id=pi0_lora_pickup_prompt_315 \
  --wandb.disable_artifact=true

echo "[DONE] pi0_lora_pickup_prompt_315 완료."
