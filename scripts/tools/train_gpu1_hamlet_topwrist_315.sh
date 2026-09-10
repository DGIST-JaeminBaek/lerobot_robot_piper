#!/usr/bin/env bash
# GPU1: SmolVLA+HAMLET, 315개, pickup(신) 프롬프트, top+wrist.
# 하이퍼파라미터 출처는 /home/ugrp43/jmbaek/smolvla_hamlet/README.md의
# "lerobot-train CLI로 직접 돌리기" 예시 그대로(HAMLET 132와 동일 레시피).
# steps/scheduler_decay_steps는 다른 315 런들과 동일하게 168000으로 맞췄다.
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ugrp
cd /home/ugrp43/UGRP/lerobot_robot_piper

CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=/home/ugrp43/jmbaek:${PYTHONPATH:-} \
ACCELERATE_MIXED_PRECISION=bf16 \
lerobot-train \
  --policy.type=smolvla_hamlet \
  --policy.discover_packages_path=smolvla_hamlet \
  --policy.pretrained_path=lerobot/smolvla_base \
  --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --policy.load_vlm_weights=true \
  --policy.num_vlm_layers=16 \
  --policy.self_attn_every_n_layers=2 \
  --policy.expert_width_multiplier=0.75 \
  --policy.attention_mode=cross_attn \
  --policy.prefix_length=0 \
  --policy.pad_language_to=max_length \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.max_state_dim=32 \
  --policy.max_action_dim=32 \
  --policy.n_moment_tokens=4 \
  --policy.memory_window=4 \
  --policy.memory_num_layers=2 \
  --policy.input_features='{"observation.images.top": {"type": "VISUAL", "shape": [3, 512, 512]}, "observation.images.wrist": {"type": "VISUAL", "shape": [3, 512, 512]}, "observation.state": {"type": "STATE", "shape": [7]}}' \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/pick_up_the_eraser_315 \
  --dataset.root=/home/ugrp43/UGRP/lerobot_robot_piper/records/outputs/pick_up_the_eraser_and_erase_the_shape/pick_up_the_eraser_315 \
  --dataset.video_backend=pyav \
  --output_dir=/home/ugrp43/UGRP/lerobot_robot_piper/outputs/train/pick_up_the_eraser_and_erase_the_shape/smolvla_hamlet_pickup_prompt_315 \
  --job_name=smolvla_hamlet_pickup_prompt_315 \
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
  --wandb.run_id=smolvla_hamlet_pickup_prompt_315 \
  --wandb.disable_artifact=true

echo "[DONE] smolvla_hamlet_pickup_prompt_315 완료."
