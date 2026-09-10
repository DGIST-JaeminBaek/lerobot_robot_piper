#!/usr/bin/env python3
"""Piper follower/leader의 관절값이 수신되고 정상 정규화 범위인지 점검한다.

사용법:
    python scripts/piper/hardware/joint_check.py --check-leader \\
      --follower-can-interface can_follower --leader-can-interface can_leader
"""

from __future__ import annotations

import argparse
import sys


JOINT_KEYS = [f"joint{index}.pos" for index in range(1, 7)]
GRIPPER_KEY = "gripper.pos"


def check_arm(role: str, port: str) -> bool:
    if role == "follower":
        from lerobot_robot_piper import PiperFollower, PiperFollowerConfig

        arm = PiperFollower(PiperFollowerConfig(port=port))
        read = arm.get_observation
    else:
        from lerobot_robot_piper import PiperLeader, PiperLeaderConfig

        arm = PiperLeader(PiperLeaderConfig(port=port))
        read = arm.get_action

    print(f"[INFO] {role} 관절값 읽는 중 (port={port})")
    arm.connect()
    try:
        values = read()
    finally:
        arm.disconnect()

    selected = {key: values.get(key) for key in [*JOINT_KEYS, GRIPPER_KEY]}
    print("vals:", selected)
    if any(value is None for value in selected.values()):
        print(f"[FAIL] {role} 관절값 key 누락", file=sys.stderr)
        return False
    if all(value == 0 for value in selected.values()):
        print(f"[FAIL] {role} 관절값 all-zero — CAN 수신 문제", file=sys.stderr)
        return False
    in_range = all(-100 <= selected[key] <= 100 for key in JOINT_KEYS) and 0 <= selected[GRIPPER_KEY] <= 100
    if not in_range:
        print(f"[FAIL] {role} 관절값이 정규화 범위를 벗어남", file=sys.stderr)
        return False
    print(f"[OK] {role} 관절값 정상")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--follower-can-interface", default="can_follower")
    parser.add_argument("--leader-can-interface", default="can_leader")
    parser.add_argument("--check-leader", action="store_true", help="leader도 함께 검사")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    follower_ok = check_arm("follower", args.follower_can_interface)
    leader_ok = True if not args.check_leader else check_arm("leader", args.leader_can_interface)
    return 0 if follower_ok and leader_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
