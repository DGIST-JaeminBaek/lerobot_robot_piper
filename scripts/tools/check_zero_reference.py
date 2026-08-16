#!/usr/bin/env python
"""현재 팔의 영점이 어느 녹화 세션과 같은지 확인한다. 팔을 움직이지 않는다.

정규화 값(-100~100)은 관절의 기계적 영점을 기준으로 계산된다. PiPER는 SDK의
JointConfig(joint_num=N, set_zero=0xAE)로 영점을 다시 잡을 수 있고, 그 값은 팔
펌웨어에 저장된다. 영점이 바뀌면 같은 정규화 값이 다른 물리 자세를 가리키므로,
영점이 바뀐 뒤 옛 데이터를 리플레이하면 팔이 엉뚱한 자세로 간다.

이 스크립트는 지금 팔의 파킹 자세를 읽어서 각 세션의 기록된 파킹 자세와 비교한다.

usage:
  # 팔을 파킹 자세에 둔 상태로 실행
  python check_zero_reference.py --datasets ~/UGRP/0802 --port can_follower1
  python check_zero_reference.py --datasets ~/UGRP/0802 --port can_leader1 --leader
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]


def session_parking(datasets: Path) -> dict[str, np.ndarray]:
    """세션(날짜)별 파킹 자세 = 각 에피소드 첫 프레임의 중앙값."""
    out = {}
    for day_dir in sorted(p for p in datasets.iterdir() if p.is_dir()):
        first = []
        for f in sorted(glob.glob(str(day_dir / "*/data/**/*.parquet"), recursive=True)):
            s = np.stack(pd.read_parquet(f)["observation.state"].to_numpy())
            first.append(np.asarray(s[0], dtype=float)[: len(JOINTS)])
        if first:
            out[day_dir.name] = np.median(first, axis=0)
    return out


def read_parking(port: str, leader: bool) -> np.ndarray:
    if leader:
        from lerobot_robot_piper.config_piper_leader import PiperLeaderConfig
        from lerobot_robot_piper.piper_leader import PiperLeader

        dev = PiperLeader(PiperLeaderConfig(id="zerocheck", port=port))
        dev.connect()
        try:
            act = dev.get_action()
        finally:
            dev.disconnect()
        return np.array([act[f"{j}.pos"] for j in JOINTS], dtype=float)

    from lerobot_robot_piper.config_piper import PiperFollowerConfig
    from lerobot_robot_piper.piper_follower import PiperFollower

    dev = PiperFollower(PiperFollowerConfig(id="zerocheck", port=port, use_action_offset=False))
    dev.connect()
    try:
        obs = dev.get_observation()
    finally:
        dev.disconnect()
    return np.array([obs[f"{j}.pos"] for j in JOINTS], dtype=float)


def report(now: np.ndarray, sessions: dict[str, np.ndarray], tol: float) -> None:
    print("\n현재 파킹 자세:", np.round(now, 1))
    print("\n세션별 기록된 파킹 자세와의 차이 (관절별):")
    best, best_d = None, np.inf
    for day, ref in sessions.items():
        diff = now - ref
        d = float(np.abs(diff).max())
        flag = "일치" if d <= tol else "불일치"
        print(f"  {day}  최대차 {d:6.1f}  [{flag}]  {np.round(diff, 1)}")
        if d < best_d:
            best, best_d = day, d

    print()
    if best_d <= tol:
        print(f"→ 현재 영점은 {best} 세션과 같습니다. 그 세션 데이터는 리플레이해도 됩니다.")
        others = [d for d in sessions if d != best]
        if others:
            print(f"  나머지 세션({', '.join(others)}) 데이터는 좌표계가 다릅니다 — "
                  "그대로 리플레이하면 팔이 다른 물리 자세로 갑니다.")
    else:
        print(f"→ 어느 세션과도 일치하지 않습니다 (가장 가까운 것도 {best} / {best_d:.1f}).")
        print("  영점이 다시 잡혔거나 파킹 자세가 달라진 상태입니다. "
              "어떤 세션 데이터도 그대로 리플레이하지 마세요.")

    # 클리핑된 축은 비교가 무의미하다는 걸 명시
    clipped = [JOINTS[i] for i in range(len(JOINTS)) if abs(now[i]) >= 99.99]
    if clipped:
        print(f"\n  참고: {', '.join(clipped)}는 정규화 한계에 잘려 있어 이 비교로는 "
              "영점 변화를 감지할 수 없습니다. 잘리지 않은 축만 신뢰할 것.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", type=Path, required=True, help="날짜별 폴더를 담은 상위 디렉터리")
    p.add_argument("--port", default="can_follower1")
    p.add_argument("--leader", action="store_true", help="leader 팔을 읽는다")
    p.add_argument("--tol", type=float, default=8.0, help="같은 영점으로 볼 최대 차이")
    args = p.parse_args()

    sessions = session_parking(args.datasets)
    if not sessions:
        raise SystemExit(f"{args.datasets} 에서 데이터셋을 찾지 못했다")
    report(read_parking(args.port, args.leader), sessions, args.tol)


if __name__ == "__main__":
    main()
