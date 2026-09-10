#!/usr/bin/env bash
# GPU1: stride35_v4와 **한 가지만 다른** 실행 — 모먼트 토큰을 TCL(Stage 1) 결과로 초기화한다.
#
# 왜 비동결(freeze_moment_tokens=false)인가:
#   논문 A.4는 GR00T에 대해 동결을 적는데, 저자 README는 비동결을 권장하며 동결을
#   "cost-saving option, not the setting for peak quality"로 규정한다. 그리고 비교 설계상
#   비동결이어야 v4와 조건이 **하나만**(TCL 초기화 유무) 다르다. 동결하면 초기화와 동결이
#   동시에 바뀌어 효과가 섞인다. 동결판(논문 재현)은 별도로 돌린다.
#
# 사전 측정 결과 주의: TCL은 m′_t를 개선하지 못했다(탐침 5종 전부에서 행동 loss 학습보다 낮음).
# 원인 후보는 사영 헤드가 학습 대상의 480배라는 점. 상세는
# /home/ugrp43/jmbaek/smolvla_hamlet/docs/ATTENTION_ANALYSIS.md 3-4·3-5절.
# 즉 이 실행은 "표현 지표가 안 올라도 실제 성능은 달라지는가"를 보는 것이다.
#
# 나머지 하이퍼파라미터는 v4와 완전히 동일하다.
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
  --policy.load_moment_tokens_from=/home/ugrp43/UGRP/lerobot_robot_piper/outputs/train/tcl/tcl_315/moment_tokens.pt \
  --policy.freeze_moment_tokens=false \
  --policy.input_features='{"observation.images.top": {"type": "VISUAL", "shape": [3, 512, 512]}, "observation.images.wrist": {"type": "VISUAL", "shape": [3, 512, 512]}, "observation.state": {"type": "STATE", "shape": [7]}}' \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/pick_up_the_eraser_315 \
  --dataset.root=/home/ugrp43/UGRP/lerobot_robot_piper/records/outputs/pick_up_the_eraser_and_erase_the_shape/pick_up_the_eraser_315 \
  --dataset.video_backend=pyav \
  --output_dir=/home/ugrp43/UGRP/lerobot_robot_piper/outputs/train/pick_up_the_eraser_and_erase_the_shape/smolvla_hamlet_pickup_prompt_315_stride35_tcl \
  --job_name=smolvla_hamlet_pickup_prompt_315_stride35_tcl \
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
  --wandb.run_id=smolvla_hamlet_pickup_prompt_315_stride35_tcl \
  --wandb.disable_artifact=true

echo "[DONE] smolvla_hamlet_pickup_prompt_315_stride35_tcl 완료."
