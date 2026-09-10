#!/usr/bin/env bash
# GPU1: SmolVLA를 315개/top+wrist로 학습하되, **top 화면에 방해 도형이 하나 더 있는**
# 데이터셋을 쓴다. 프롬프트는 도형별 3종("... erase the circle/triangle/rectangle").
#
# 목적: 선택적 지우기를 실제로 학습시킨다. 기존 학습셋은 보드에 도형이 하나뿐이라
# 프롬프트의 도형 단어가 시각 입력과 100% 중복이었다 — 언어를 접지할 압력이 없었다
# (그 대조군이 smolvla_pickup_315_shape_prompt다). 여기서는 타겟이 아닌 도형이 항상
# 하나 더 있으므로, "어느 걸 지울지"가 프롬프트로만 결정된다.
#
# 데이터: 관측 중 top 영상만 증강본으로 갈아끼웠고 관절 상태·행동·wrist 영상은 원본
# 그대로다. 방해 도형은 타겟과 절대 겹치지 않게 배정돼 있다(circle 타겟 -> rectangle
# 또는 triangle 방해, 105개씩 균등).
#
# 하이퍼파라미터는 기준 모델 smolvla_pickup_prompt_315의 train_config.json에서 그대로
# 가져왔다(steps/batch/lr/스케줄러/시드까지 동일). 바꾼 것은 dataset.root, output_dir,
# job_name, wandb.run_id 뿐이다 — 그래야 데이터 차이만 분리된다.
#
# 데이터셋 생성:
#   scripts/tasks/erase_shape/dataset/make_shape_prompt_dataset.py   (프롬프트 3종으로 분리)
#   scripts/tasks/erase_shape/dataset/make_distractor_dataset.py     (그 위에 top만 증강본으로 교체)
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
  --dataset.repo_id=local/pick_up_the_eraser_315_distractor \
  --dataset.root=/home/ugrp43/UGRP/lerobot_robot_piper/records/outputs/pick_up_the_eraser_and_erase_the_shape/pick_up_the_eraser_315_distractor \
  --dataset.video_backend=pyav \
  --output_dir=/home/ugrp43/UGRP/lerobot_robot_piper/outputs/train/pick_up_the_eraser_and_erase_the_shape/smolvla_pickup_315_distractor \
  --job_name=smolvla_pickup_315_distractor \
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
  --wandb.run_id=smolvla_pickup_315_distractor \
  --wandb.disable_artifact=true

echo "[DONE] smolvla_pickup_315_distractor 완료."
