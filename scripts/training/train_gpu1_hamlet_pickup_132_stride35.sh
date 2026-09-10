#!/usr/bin/env bash
# GPU1: SmolVLA+HAMLET(메모리 모듈)을 132개/pickup 프롬프트/top+wrist, memory_stride=35로 학습한다.
# stride=35는 piper_infer_runner.py(기본 threshold 트리거, chunk_threshold=0.3, horizon=50)의
# 정상 상태 재추론 간격 (1-0.3)*50=35 raw 스텝에 맞춘 값이다 — 자세한 도출 과정은
# /home/ugrp43/jmbaek/smolvla_hamlet/docs/ARCHITECTURE_DIFFERENCES.md의 "부록" 참고.
#
# 기존 smolvla_hamlet_pickup_prompt_132(memory_stride 필드 생기기 전 학습, 사실상 stride=1)와
# 비교하기 위한 짝 — 315개 데이터셋 stride=35 실험(smolvla_hamlet_pickup_prompt_315_stride35_v4,
# 완료됨)에 이어 132개 규모에서도 재현되는지 보는 실험.
#
# 하이퍼파라미터는 train_gpu1_hamlet_pickup_132.sh(기존 132개 stride=1 학습)와 동일하게 맞추고
# memory_stride만 추가, 이름만 바꿨다. steps/save_freq는 132개 관례(training_status.md §3) 그대로
# 65265 / 5000. num_workers는 315 stride35 v3→v4 재학습 때 확인된 이유(넓은 stride 때문에 영상
# 디코딩 seek 비용이 커짐)로 8로 설정 — 132 규모라 8이 과할 수도 있지만 안전하게 맞춤.
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
  --dataset.repo_id=local/pick_up_the_eraser_0727_0812_0813am_132 \
  --dataset.root=/home/ugrp43/UGRP/lerobot_robot_piper/records/outputs/pick_up_the_eraser_and_erase_the_shape/pick_up_the_eraser_0727_0812_0813am_132 \
  --dataset.video_backend=pyav \
  --output_dir=/home/ugrp43/UGRP/lerobot_robot_piper/outputs/train/pick_up_the_eraser_and_erase_the_shape/smolvla_hamlet_pickup_prompt_132_stride35 \
  --job_name=smolvla_hamlet_pickup_prompt_132_stride35 \
  --batch_size=8 \
  --num_workers=8 \
  --steps=65265 \
  --log_freq=100 \
  --save_freq=5000 \
  --eval_freq=0 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=65265 \
  --wandb.enable=true \
  --wandb.project=smolvla-erase-shape \
  --wandb.run_id=smolvla_hamlet_pickup_prompt_132_stride35 \
  --wandb.disable_artifact=true

echo "[DONE] smolvla_hamlet_pickup_prompt_132_stride35 완료."
