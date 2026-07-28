#!/usr/bin/env python3
"""piper_offline_chunk_rollout.py가 저장한 action 궤적을 RViz에서 재생한다.

CAN이나 실제 Piper에는 연결하지 않고 ROS2 /joint_states만 publish한다.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="저장된 offline rollout action을 RViz에서 재생")
    parser.add_argument("--rollout", type=pathlib.Path, required=True, help="rollout_actions.npz")
    parser.add_argument(
        "--trajectory",
        choices=["predicted", "expert"],
        default="predicted",
        help="예측 궤적 또는 dataset 정답 궤적",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--joint-state-topic", default="/joint_states")
    args = parser.parse_args()

    if not args.rollout.exists():
        print(f"[ERROR] file not found: {args.rollout}", file=sys.stderr)
        return 1
    if args.fps <= 0:
        print("[ERROR] --fps must be positive", file=sys.stderr)
        return 1

    key = "predicted_actions" if args.trajectory == "predicted" else "expert_actions"
    with np.load(args.rollout) as data:
        if key not in data:
            print(f"[ERROR] {key!r} is not present in {args.rollout}", file=sys.stderr)
            return 1
        actions = np.asarray(data[key], dtype=np.float32)

    if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) == 0:
        print(f"[ERROR] invalid action shape: {actions.shape}; expected (frames, 7)", file=sys.stderr)
        return 1

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from piper_infer_preview import run_rviz

    print(
        f"[RVIZ] trajectory={args.trajectory}, frames={len(actions)}, "
        f"fps={args.fps}, source={args.rollout}"
    )
    run_rviz(actions, 1.0 / args.fps, args.joint_state_topic)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
