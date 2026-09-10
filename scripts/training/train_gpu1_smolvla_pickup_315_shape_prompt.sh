#!/usr/bin/env bash
# GPU1: SmolVLA를 315개/top+wrist로 학습하되, **프롬프트를 도형별로 쪼갠** 데이터셋을 쓴다.
#   "pick up the eraser and erase the shape"
#     -> "... erase the circle" / "... erase the triangle" / "... erase the rectangle"
#
# 목적: 데이터 증강 없이 프롬프트만 바꿔서, 다중 도형 보드에서 지정한 도형만 지우는
# zero-shot 선택성이 나오는지 본다. 학습 데이터는 전부 도형 1개짜리라(315개 전수 확인)
# 프롬프트의 도형 단어가 시각 입력과 100% 중복이다 — 즉 언어를 접지할 압력이 없어서
# 선택성이 안 나올 가능성이 높다. 그래도 "프롬프트만으로는 안 된다"의 바닥값을 재는
# 대조군으로 필요하다(나중에 distractor 데이터를 넣었을 때 비교 기준).
#
# 하이퍼파라미터는 기준 모델 smolvla_pickup_prompt_315의 train_config.json에서 그대로
# 가져왔다(steps/batch/lr/스케줄러/시드까지 동일). 바꾼 것은 dataset.root(도형별 프롬프트
# 데이터셋), output_dir, job_name, wandb.run_id 뿐이다 — 그래야 프롬프트 효과만 분리된다.
#
# 데이터셋 생성:
#   scripts/tasks/erase_shape/dataset/make_shape_prompt_dataset.py
#   (영상은 하드링크라 용량 추가 없음, task_index/tasks 메타만 다시 씀)
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ugrp
cd /home/ugrp43/UGRP/lerobot_robot_piper

CUDA_VISIBLE_DEVICES=1 \
ACCELERATE_MIXED_PRECISION=bf16 \
lerobot-train \
  --policy.type=smolvla \
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
  --policy.device=cuda \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.push_to_hub=false \
  --policy.optimizer_lr=1e-4 \
  --dataset.repo_id=local/pick_up_the_eraser_315_shape_prompt \
  --dataset.root=/home/ugrp43/UGRP/lerobot_robot_piper/records/outputs/pick_up_the_eraser_and_erase_the_shape/pick_up_the_eraser_315_shape_prompt \
  --dataset.video_backend=pyav \
  --output_dir=/home/ugrp43/UGRP/lerobot_robot_piper/outputs/train/pick_up_the_eraser_and_erase_the_shape/smolvla_pickup_315_shape_prompt \
  --job_name=smolvla_pickup_315_shape_prompt \
  --seed=1000 \
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
  --wandb.run_id=smolvla_pickup_315_shape_prompt \
  --wandb.disable_artifact=true

echo "[DONE] smolvla_pickup_315_shape_prompt 완료."
