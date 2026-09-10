#!/usr/bin/env bash
# GPU1: SmolVLA+HAMLET(메모리 모듈)을 315개/pickup 프롬프트/top+wrist, memory_stride=16으로 학습한다.
#
# memory_stride 비교 실험의 세 번째 값이다. 앞선 두 값의 근거:
#   35 = piper_infer_runner.py의 정상 상태 재추론 간격 (1-chunk_threshold)*horizon
#        = (1-0.3)*50 raw 스텝에 맞춘 값 (도출 과정은
#        /home/ugrp43/jmbaek/smolvla_hamlet/docs/ARCHITECTURE_DIFFERENCES.md의 "부록")
#   50 = chunk 하나를 통째로 건너뛰는 간격(메모리가 가장 성기다)
#   16 = 그보다 촘촘하게, 30Hz에서 약 0.53초 간격으로 과거를 되짚는다. 메모리 토큰이
#        덮는 시간 범위가 좁아지는 대신 최근 구간을 더 세밀히 본다.
# 나머지 조건은 stride35_v4 / stride50과 **완전히 동일**하다(dataset, steps, batch,
# 스케줄러, memory_window=4, n_moment_tokens=4) — 그래야 stride 효과만 분리된다.
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
  --policy.memory_stride=16 \
  --policy.memory_num_layers=2 \
  --policy.input_features='{"observation.images.top": {"type": "VISUAL", "shape": [3, 512, 512]}, "observation.images.wrist": {"type": "VISUAL", "shape": [3, 512, 512]}, "observation.state": {"type": "STATE", "shape": [7]}}' \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/pick_up_the_eraser_315 \
  --dataset.root=/home/ugrp43/UGRP/lerobot_robot_piper/records/outputs/pick_up_the_eraser_and_erase_the_shape/pick_up_the_eraser_315 \
  --dataset.video_backend=pyav \
  --output_dir=/home/ugrp43/UGRP/lerobot_robot_piper/outputs/train/pick_up_the_eraser_and_erase_the_shape/smolvla_hamlet_pickup_prompt_315_stride16 \
  --job_name=smolvla_hamlet_pickup_prompt_315_stride16 \
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
  --wandb.run_id=smolvla_hamlet_pickup_prompt_315_stride16 \
  --wandb.disable_artifact=true

echo "[DONE] smolvla_hamlet_pickup_prompt_315_stride16 완료."
