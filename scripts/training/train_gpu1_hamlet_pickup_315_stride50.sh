#!/usr/bin/env bash
# GPU1 큐: smolvla_hamlet_pickup_prompt_132_stride35(PID 855164) 끝날 때까지 기다렸다가,
# SmolVLA+HAMLET(메모리 모듈)을 315개/pickup 프롬프트/top+wrist, memory_stride=50로 학습한다.
#
# stride=50은 n_action_steps(=chunk_size=50)와 그대로 맞춘 값 — GR00T 원본이 가정하는 단순
# 규칙(정책 호출 → 청크 전체 실행 → 재추론)에 해당하고, 우리 배포 환경에서는
# piper_human_approved_inference.py(사람이 청크를 다 소비할 때까지 재추론 안 함) 경로에 정확히
# 대응한다. threshold 트리거 러너(기본, 비동기)엔 stride=35가 대응 — 두 배포 패턴을 각각 커버.
# 자세한 도출 과정은 /home/ugrp43/jmbaek/smolvla_hamlet/docs/ARCHITECTURE_DIFFERENCES.md의 "부록" 참고.
#
# 하이퍼파라미터는 train_gpu1_hamlet_pickup_315_stride35_v4.sh와 동일(steps=168000,
# save_freq=15000, num_workers=8), memory_stride만 50으로 바꿨다.
set -euo pipefail

WAIT_PID=855164
echo "[WAIT] PID ${WAIT_PID}(smolvla_hamlet_pickup_prompt_132_stride35) 종료 대기 중..."
while kill -0 "${WAIT_PID}" 2>/dev/null; do
  sleep 30
done
echo "[WAIT] 종료 확인. GPU1 315개 stride=50 학습 시작."

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
  --policy.memory_stride=50 \
  --policy.memory_num_layers=2 \
  --policy.input_features='{"observation.images.top": {"type": "VISUAL", "shape": [3, 512, 512]}, "observation.images.wrist": {"type": "VISUAL", "shape": [3, 512, 512]}, "observation.state": {"type": "STATE", "shape": [7]}}' \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/pick_up_the_eraser_315 \
  --dataset.root=/home/ugrp43/UGRP/lerobot_robot_piper/records/outputs/pick_up_the_eraser_and_erase_the_shape/pick_up_the_eraser_315 \
  --dataset.video_backend=pyav \
  --output_dir=/home/ugrp43/UGRP/lerobot_robot_piper/outputs/train/pick_up_the_eraser_and_erase_the_shape/smolvla_hamlet_pickup_prompt_315_stride50 \
  --job_name=smolvla_hamlet_pickup_prompt_315_stride50 \
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
  --wandb.run_id=smolvla_hamlet_pickup_prompt_315_stride50 \
  --wandb.disable_artifact=true

echo "[DONE] smolvla_hamlet_pickup_prompt_315_stride50 완료."
