#!/usr/bin/env python3
"""실제 replay 전에 follower를 선택 episode의 첫 유효 action까지 저속 정렬한다.

이 도구는 replay를 실행하지 않는다. 정렬에 성공한 뒤에 호출자가 표준
``lerobot-replay``를 실행한다. 정렬 자체가 실제 팔을 움직이므로, 기본 실행은
명시적 확인 토큰을 요구한다.

사용법:
    # 현재-시작 자세 차이만 확인 (명령 전송 없음)
    python scripts/piper/hardware/align_replay_start.py \\
      --dataset-root records/local/example --episode 0 --port can_follower --dry-run

    # 실제 정렬 후 별도로 lerobot-replay 실행
    python scripts/piper/hardware/align_replay_start.py \\
      --dataset-root records/local/example --episode 0 --port can_follower \\
      --confirm ALIGN_REPLAY_START
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np


CONFIRM_TOKEN = "ALIGN_REPLAY_START"


def first_unclipped(actions: np.ndarray) -> int:
    """파킹 과정에서 정규화 한계에 붙은 선두 action은 정렬 목표에서 제외한다."""
    clipped = (np.abs(actions) >= 99.99).any(axis=1)
    return int(np.argmax(~clipped)) if (~clipped).any() else len(actions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="LeRobot dataset root")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--port", default="can_follower", help="follower CAN 인터페이스")
    parser.add_argument("--fps", type=float, default=30.0, help="정렬 명령 주기")
    parser.add_argument("--align-step", type=float, default=1.0, help="프레임당 최대 이동량(정규화 단위)")
    parser.add_argument("--align-tol", type=float, default=1.0, help="정렬 완료 최대 오차(정규화 단위)")
    parser.add_argument("--align-timeout", type=float, default=30.0, help="정렬 제한 시간(초)")
    parser.add_argument("--max-relative-target", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true", help="현재 자세와 목표 차이만 출력하고 명령은 보내지 않음")
    parser.add_argument("--confirm", default="", help=f"실제 정렬에는 {CONFIRM_TOKEN} 필요")
    return parser.parse_args()


def load_target(root: pathlib.Path, episode: int) -> tuple[np.ndarray, list[str], int]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id="local/replay", root=root, episodes=[episode])
    frames = dataset.hf_dataset.filter(lambda frame: frame["episode_index"] == episode)
    actions = np.asarray(frames.select_columns("action")["action"], dtype=np.float32)
    names = list(dataset.features["action"]["names"])
    if actions.ndim != 2 or actions.shape[1] != 7 or len(names) != 7:
        raise ValueError("Piper 7축 action 데이터셋만 지원합니다")
    skipped = first_unclipped(actions)
    if skipped == len(actions):
        raise ValueError("전 구간이 정규화 한계에 잘린 값이라 정렬 목표를 정할 수 없습니다")
    return actions[skipped], names, skipped


def main() -> int:
    args = parse_args()
    if not args.dry_run and args.confirm != CONFIRM_TOKEN:
        print(f"[FAIL] 실제 정렬에는 --confirm {CONFIRM_TOKEN} 이 필요합니다", file=sys.stderr)
        return 2
    if args.fps <= 0 or args.align_step <= 0 or args.align_tol < 0 or args.align_timeout <= 0:
        print("[FAIL] fps/align-step/align-timeout은 양수, align-tol은 0 이상이어야 합니다", file=sys.stderr)
        return 2

    target, names, skipped = load_target(pathlib.Path(args.dataset_root), args.episode)
    from lerobot_robot_piper.config_piper import PiperFollowerConfig
    from lerobot_robot_piper.piper_follower import PiperFollower

    follower = PiperFollower(PiperFollowerConfig(
        id="align_replay_start",
        port=args.port,
        park_on_connect=False,
        use_action_offset=False,
        disable_torque_on_disconnect=False,
        max_relative_target=args.max_relative_target,
    ))
    follower.connect()
    try:
        current = np.asarray([follower.get_observation()[name] for name in names], dtype=np.float32)
        initial_gap = float(np.abs(target - current).max())
        print(f"[ALIGN] episode={args.episode}, skipped_clipped_frames={skipped}")
        print(f"[ALIGN] 현재→첫 유효 action 최대 차이: {initial_gap:.2f}")
        if args.dry_run:
            print("[DRY-RUN] 명령을 전송하지 않았습니다")
            return 0

        deadline = time.monotonic() + args.align_timeout
        while time.monotonic() < deadline:
            current = np.asarray([follower.get_observation()[name] for name in names], dtype=np.float32)
            gap = target - current
            maximum = float(np.abs(gap).max())
            if maximum <= args.align_tol:
                print(f"[OK] 정렬 완료 (최대 차이 {maximum:.2f})")
                return 0
            command = current + np.clip(gap, -args.align_step, args.align_step)
            follower.send_action({name: float(value) for name, value in zip(names, command, strict=True)})
            time.sleep(1 / args.fps)

        print("[FAIL] 정렬 시간 초과 — replay를 시작하지 마세요", file=sys.stderr)
        return 1
    finally:
        # 이어지는 lerobot-replay가 현재 자세를 그대로 이어받도록 parking/torque 해제를 하지 않는다.
        follower.disconnect(disable_torque=False, park=False)


if __name__ == "__main__":
    raise SystemExit(main())
