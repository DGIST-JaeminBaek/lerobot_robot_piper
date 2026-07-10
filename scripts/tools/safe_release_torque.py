#!/usr/bin/env python3
"""안전한 수동 torque 해제 루틴.

Record/Teleoperate 종료 후 DISABLE_TORQUE_ON_DISCONNECT=false로 두면 팔은
parking 자세로 이동한 채 torque가 걸린 상태로 남는다. 이 스크립트는:

1. joint1~6을 0으로 이동 (gripper는 건드리지 않음 — 잡고 있을 때 손이 끼지
   않도록 현재 상태 유지)
2. 사람이 팔을 안전하게 붙잡을 때까지 Enter 입력을 기다림
3. Enter를 누르면 그 자리에서 torque를 해제 (parking 재이동 없음 — 사람이
   잡고 있는 도중에 팔이 움직이면 위험하므로)

사용 예:
    python3 scripts/tools/safe_release_torque.py --port can_follower
"""

import argparse
import sys
import time

from lerobot_robot_piper import PiperFollowerConfig, PiperFollower
from lerobot_robot_piper.motors.tables import INITIALIZE_POSITION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="can_follower", help="follower CAN 포트 (기본: can_follower)")
    parser.add_argument("--max-relative-target", type=float, default=15.0,
                         help="joint1~6을 0으로 이동할 때 timestep별 최대 이동량")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = PiperFollowerConfig(
        port=args.port,
        max_relative_target=args.max_relative_target,
        park_on_connect=False,
        use_action_offset=False,
        disable_torque_on_disconnect=False,
    )
    follower = PiperFollower(cfg)
    follower.connect()

    try:
        obs = follower.get_observation()
        action = {k: v for k, v in obs.items() if k.endswith(".pos")}
        print(f"현재 위치: {action}")

        # INITIALIZE_POSITION은 각 joint의 calibration 범위(비대칭 포함)로 역산해서
        # "정규화 0"이 아니라 "실제 물리 각도 0도"에 정확히 대응하도록 맞춰진 값.
        # (joint2/joint3/joint6은 calibration이 0 기준 비대칭이라 정규화 0 그대로 쓰면
        # 물리 각도가 0도가 아니게 됨 — motors/tables.py 주석 참고)
        for joint in ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]:
            action[f"{joint}.pos"] = INITIALIZE_POSITION[joint]
        print("joint1~6을 물리 각도 0도로 이동합니다 (gripper는 유지)...")

        # max_relative_target 제한 때문에 여러 스텝에 걸쳐 반복 전송해야 목표에 수렴함
        for _ in range(30):
            follower.send_action(action)
            time.sleep(0.1)
        time.sleep(0.5)

        cur = follower.get_observation()
        print("이동 후 위치: " + ", ".join(
            f"{j}={cur.get(f'{j}.pos', float('nan')):.1f}" for j in
            ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        ))

        print()
        print("!!! 지금부터 사람이 로봇 팔을 안전하게 붙잡아주세요 !!!")
        print("팔을 잡은 상태가 확실하면 Enter를 누르세요. 누르는 즉시 torque가 풀리며")
        print("팔이 늘어질 수 있습니다 (Ctrl+C로 취소 시 torque는 켜진 채로 남습니다).")
        input("준비되면 Enter > ")

        follower.bus.disable_torque()
        print("torque 해제 완료.")

    except KeyboardInterrupt:
        print("\n취소됨 — torque는 켜진 상태로 유지됩니다.", file=sys.stderr)
        follower.bus.disconnect(disable_torque=False)
        sys.exit(1)
    else:
        follower.bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
