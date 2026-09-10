#!/usr/bin/env bash
# HAMLET Stage 1 (TCL) — 315개 데이터셋으로 모먼트 토큰 사전학습.
#
# 하이퍼파라미터는 논문 Appendix A.4의 GR00T N1.5 항목 그대로:
#   30k step / batch 64 / lr 1e-5 / 대조온도 0.07 / 모먼트 토큰 4개
# 단 두 가지만 우리 설정:
#   - tcl_neg_min_gap=35 : 논문은 "액션 호라이즌"(GR00T=16)을 쓰라고 하고 참고 구현도
#     len(action_indices)에서 유도한다. 우리 실효 재추론 주기가 35라 그 값을 쓴다.
#     (config 주석에 근거 정리)
#   - --amp : Stage 2가 ACCELERATE_MIXED_PRECISION=bf16으로 학습되므로 정밀도를 맞춘다.
#     부수적으로 fp32 대비 4.4배 빠름(실측 0.40 -> 1.80 step/s @batch32).
#
# wandb: smolvla-erase-shape / run id `tcl_315` (기존 Stage 2 런들과 같은 프로젝트)
#
# 산출물: moment_tokens_step{N}.pt (2k마다) + moment_tokens.pt(최신) + train_log.json
# Stage 2에서 --policy.load_moment_tokens_from=<경로> 로 불러 쓴다.
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ugrp
cd /home/ugrp43/jmbaek

CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=/home/ugrp43/jmbaek \
python -m smolvla_hamlet.scripts.train_tcl \
  --dataset-root /home/ugrp43/UGRP/lerobot_robot_piper/records/outputs/pick_up_the_eraser_and_erase_the_shape/pick_up_the_eraser_315 \
  --repo-id local/pick_up_the_eraser_315 \
  --output-dir /home/ugrp43/UGRP/lerobot_robot_piper/outputs/train/tcl/tcl_315 \
  --image-keys observation.images.top observation.images.wrist \
  --steps 30000 \
  --batch-size 64 \
  --lr 1e-5 \
  --num-workers 12 \
  --log-freq 100 \
  --save-freq 2000 \
  --wandb-project smolvla-erase-shape \
  --wandb-run-id tcl_315 \
  --amp

echo "[DONE] TCL 315 완료."
