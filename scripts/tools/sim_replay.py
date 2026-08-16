#!/usr/bin/env python
"""리플레이 software-in-the-loop 시뮬. 하드웨어 없이 실데이터로 돌린다.

3D 시뮬레이터는 필요 없다. 이 버그를 만드는 요소는 셋뿐이고 전부 여기 있다:
  1. send_action의 max_relative_target 클램프 (스텝당 5.0)
  2. 팔의 실제 속도 한계 (녹화 데이터에서 실측 — 지어내지 않는다)
  3. 30fps로 계속 흐르는 재생 타임라인

lerobot-replay(정렬 없음)와 replay_safe(정렬 후 재생)를 같은 에피소드로 돌려
추종 오차를 비교한다.

usage: python sim_replay.py <DATASET_ROOT> [--start-gap 30]
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

CLAMP = 5.0  # config_piper.py: max_relative_target
FPS = 30


class ArmModel:
    """follower의 send_action 거동 + 팔의 물리 속도 한계."""

    def __init__(self, pos: np.ndarray, max_speed: float):
        self.pos = pos.astype(float).copy()
        self.max_speed = max_speed

    def send_action(self, goal: np.ndarray) -> None:
        goal = self.pos + np.clip(goal - self.pos, -CLAMP, CLAMP)  # 안전 클램프
        self.pos += np.clip(goal - self.pos, -self.max_speed, self.max_speed)  # 물리 한계


def measure_max_speed(state: np.ndarray) -> float:
    """녹화 중 실제로 관측된 스텝당 최대 이동량 = 팔이 낼 수 있는 속도의 하한."""
    return float(np.percentile(np.abs(np.diff(state, axis=0)).max(axis=1), 99))


def run(actions: np.ndarray, start: np.ndarray, max_speed: float, align: bool,
        use_offset: bool = False, warmup_s: float = 1.5) -> dict:
    """use_offset=True는 리플레이를 --robot.use_action_offset=false 없이 돌린 경우.
    follower가 시작 자세 기준 상대 추종으로 동작해 궤적 전체가 통째로 밀린다."""
    arm = ArmModel(start, max_speed)
    align_steps = 0
    if align:
        # replay_safe: 첫 자세에 닿을 때까지 타임라인을 시작하지 않는다
        while np.abs(actions[0] - arm.pos).max() > 1.0 and align_steps < FPS * 30:
            arm.send_action(arm.pos + np.clip(actions[0] - arm.pos, -1.0, 1.0))
            align_steps += 1

    err = []
    offset = None
    for i, a in enumerate(actions):
        goal = a
        if use_offset:
            # piper_follower.py:229-256 재현 — warmup 동안 offset 재계산 후 고정
            if offset is None or i < warmup_s * FPS:
                offset = arm.pos - a
            goal = a + offset
        arm.send_action(goal)
        err.append(np.abs(a - arm.pos).max())
    err = np.array(err)
    return {
        "align_s": align_steps / FPS,
        "err_mean": err.mean(),
        "err_max": err.max(),
        "err_final": err[-1],
        "bad_frac": float((err > 5).mean()),
    }


def self_check() -> None:
    """모델이 맞는지 확인: 정렬된 채 시작하면 오차 0, offset이 켜지면 오차 = 시작 차이."""
    rng = np.random.default_rng(0)
    traj = np.cumsum(rng.normal(0, 0.5, (600, 7)), axis=0)
    speed = measure_max_speed(traj)

    assert run(traj, traj[0], speed, align=False)["err_max"] < 1.0
    gap = np.zeros(7)
    gap[1] = 20.0
    r_off = run(traj, traj[0] + gap, speed, align=False, use_offset=True)
    # 시작 차이만큼(그 이상으로 — warmup 동안 추종 지연까지 offset에 흡수된다)
    # 영구히 밀린 채 끝까지 회복하지 않는다
    assert r_off["err_final"] >= 20.0 and r_off["bad_frac"] == 1.0, r_off
    assert run(traj, traj[0] + gap, speed, align=True)["err_max"] < 1.0  # 정렬하면 회복
    print("ok")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path, nargs="?")
    p.add_argument("--self-check", action="store_true")
    p.add_argument("--start-gap", type=float, default=30.0,
                   help="리플레이 시작 시 팔이 첫 자세에서 얼마나 떨어져 있다고 볼지 (정규화 단위)")
    args = p.parse_args()
    if args.self_check:
        self_check()
        return

    f = glob.glob(str(args.root / "data/**/*.parquet"), recursive=True)[0]
    d = pd.read_parquet(f)
    actions = np.stack(d["action"].to_numpy()).astype(float)
    state = np.stack(d["observation.state"].to_numpy()).astype(float)[:, : actions.shape[1]]

    max_speed = measure_max_speed(state)
    print(f"{args.root.name}: {len(actions)} frames ({len(actions) / FPS:.1f}s)")
    print(f"실측 팔 속도 상한: {max_speed:.2f} 단위/프레임 (클램프 {CLAMP} 보다 "
          f"{'빡빡함 — 클램프는 사실상 무의미' if max_speed < CLAMP else '여유 있음'})")

    # 팔이 첫 자세에서 start_gap 만큼 떨어진 상태로 리플레이를 시작한다고 가정
    offset = np.zeros(actions.shape[1])
    offset[1] = args.start_gap  # 어깨(joint2) 기준 — 영상에서 크게 돌아간 축
    start = actions[0] + offset

    print(f"\n시작 자세 차이 {args.start_gap:.0f} 단위일 때:")
    print(f"{'':20s} {'정렬':>6s} {'평균오차':>8s} {'최대오차':>8s} {'종료오차':>8s} {'오차>5 비율':>10s}")
    for label, align, off in (
        ("lerobot-replay", False, False),
        ("  + offset 켜짐", False, True),
        ("replay_safe", True, False),
    ):
        r = run(actions, start, max_speed, align, use_offset=off)
        print(f"{label:20s} {r['align_s']:5.1f}s {r['err_mean']:8.2f} {r['err_max']:8.2f} "
              f"{r['err_final']:8.2f} {r['bad_frac']:9.0%}")


if __name__ == "__main__":
    main()
