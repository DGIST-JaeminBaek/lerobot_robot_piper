#!/usr/bin/env bash
# GPU1 큐의 4번째: 132->135->315toponly 체인 전체(PID 3695285,
# train_gpu1_pickup_toponly_315_after_chain.sh)가 끝날 때까지 기다렸다가,
# SmolVLA+HAMLET(메모리 모듈)을 132개/pickup(신) 프롬프트/top+wrist로 학습한다.
# 무-메모리 기준선(smolvla_pickup_prompt_132_v2, 같은 데이터셋)과 직접 비교하기 위한 짝.
#
# 하이퍼파라미터 출처: /home/ugrp43/jmbaek/smolvla_hamlet/README.md의
# "lerobot-train CLI로 직접 돌리기" 예시 그대로. --policy.path가 아니라
# --policy.type=smolvla_hamlet + --policy.pretrained_path를 같이 써야
# HAMLET config가 실제로 적용된다(--policy.path만 쓰면 type이 "smolvla"로
# 고정되어 무시됨). input_features도 자동추론이 아니라 명시해야 한다.
# 메모리 모듈 하이퍼파라미터(n_moment_tokens/memory_window/memory_num_layers)는
# README 자체가 "튜닝된 값 아님, 일단 합리적인 값"이라고 명시한 기본값이다.
set -euo pipefail

WAIT_PID=3695285
echo "[WAIT] PID ${WAIT_PID}(132->135->315toponly 체인) 종료 대기 중..."
while kill -0 "${WAIT_PID}" 2>/dev/null; do
  sleep 30
done
echo "[WAIT] 종료 확인. GPU1 HAMLET 132개 학습 시작."

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
  --policy.memory_num_layers=2 \
  --policy.input_features='{"observation.images.top": {"type": "VISUAL", "shape": [3, 512, 512]}, "observation.images.wrist": {"type": "VISUAL", "shape": [3, 512, 512]}, "observation.state": {"type": "STATE", "shape": [7]}}' \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/pick_up_the_eraser_0727_0812_0813am_132 \
  --dataset.root=/home/ugrp43/UGRP/lerobot_robot_piper/records/outputs/pick_up_the_eraser_and_erase_the_shape/pick_up_the_eraser_0727_0812_0813am_132 \
  --dataset.video_backend=pyav \
  --output_dir=/home/ugrp43/UGRP/lerobot_robot_piper/outputs/train/pick_up_the_eraser_and_erase_the_shape/smolvla_hamlet_pickup_prompt_132 \
  --job_name=smolvla_hamlet_pickup_prompt_132 \
  --batch_size=8 \
  --num_workers=2 \
  --steps=65265 \
  --log_freq=100 \
  --save_freq=5000 \
  --eval_freq=0 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=65265 \
  --wandb.enable=true \
  --wandb.project=smolvla-erase-shape \
  --wandb.run_id=smolvla_hamlet_pickup_prompt_132 \
  --wandb.disable_artifact=true

echo "[DONE] smolvla_hamlet_pickup_prompt_132 완료."
