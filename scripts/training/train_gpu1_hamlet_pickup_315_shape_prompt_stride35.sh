#!/usr/bin/env bash
# GPU1: SmolVLA+HAMLET(메모리 모듈)을 315개/pickup 프롬프트/top+wrist, memory_stride=35로 학습한다.
# stride=35는 piper_infer_runner.py(기본 threshold 트리거, chunk_threshold=0.3, horizon=50)의
# 정상 상태 재추론 간격 (1-0.3)*50=35 raw 스텝에 맞춘 값이다 — 자세한 도출 과정은
# /home/ugrp43/jmbaek/smolvla_hamlet/docs/ARCHITECTURE_DIFFERENCES.md의 "부록" 참고.
#
# 참고: raw 프레임 버퍼(hamlet_history.py) 방식이 이미 재학습 없이 grasp 문제를 해결했다 —
# 이 재학습은 "제대로 학습 시부터 맞는 시간축을 가르치면 더 나아지는지" 비교용 실험이다.
#
# 하이퍼파라미터는 train_gpu1_hamlet_pickup_132.sh(기존 132개 HAMLET 학습)와 동일하게 맞추고
# dataset/steps/memory_stride/output 이름만 315개 데이터셋 기준으로 바꿨다. steps는 기존 132개
# 학습 관례(steps == dataset.total_frames)를 그대로 따라 167589로 설정.
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ugrp
cd /home/ugrp43/UGRP/lerobot_robot_piper

CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=/home/ugrp43/jmbaek \
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
  --policy.memory_stride=35 \
  --policy.memory_num_layers=2 \
  --policy.input_features='{"observation.images.top": {"type": "VISUAL", "shape": [3, 512, 512]}, "observation.images.wrist": {"type": "VISUAL", "shape": [3, 512, 512]}, "observation.state": {"type": "STATE", "shape": [7]}}' \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/pick_up_the_eraser_315_shape_prompt \
  --dataset.root=/home/ugrp43/UGRP/lerobot_robot_piper/records/outputs/pick_up_the_eraser_and_erase_the_shape/pick_up_the_eraser_315_shape_prompt \
  --dataset.video_backend=pyav \
  --output_dir=/home/ugrp43/UGRP/lerobot_robot_piper/outputs/train/pick_up_the_eraser_and_erase_the_shape/smolvla_hamlet_pickup_315_shape_prompt_stride35 \
  --job_name=smolvla_hamlet_pickup_315_shape_prompt_stride35 \
  --batch_size=8 \
  --num_workers=8 \
  --steps=168000 \
  --log_freq=100 \
  --save_freq=15000 \
  --eval_freq=0 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=168000 \
  --wandb.enable=true \
  --wandb.project=smolvla-erase-shape \
  --wandb.run_id=smolvla_hamlet_pickup_315_shape_prompt_stride35 \
  --wandb.disable_artifact=true

echo "[DONE] smolvla_hamlet_pickup_315_shape_prompt_stride35 완료."
