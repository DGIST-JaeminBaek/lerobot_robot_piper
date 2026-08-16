#!/usr/bin/env python3
"""안전한 수동 torque 해제 루틴.

Record/Teleoperate 종료 후 DISABLE_TORQUE_ON_DISCONNECT=false로 두면 팔은
torque가 걸린 상태로 남는다. 이 스크립트는:

1. --mode에 따라 자세를 잡음 (gripper는 어느 모드에서도 안 건드림 — 잡고 있을
   때 손이 끼지 않도록 현재 상태 유지)
     lower    — 팔은 그대로 두고 손목(joint5)만 미리 내림 (기본). torque를 풀 때
                실제로 떨어지는 건 손목뿐이라(joint1~4/6은 0.00도) 그 낙차를 없앰
     in_place — 이동 없음. 사람이 이미 팔을 원하는 자리에 둔 경우
     park     — 기존 동작: parking 자세(물리 각도 0도, 팔이 위로 뻗음)로 이동.
                놓는 높이가 제일 높아서 떨어질 때 충격도 제일 크다
2. 사람이 팔을 안전하게 붙잡을 때까지 Enter 입력을 기다림
3. Enter를 누르면 그 자리에서 torque를 해제 (여기서 추가 이동 없음 — 사람이
   잡고 있는 도중에 팔이 움직이면 위험하므로)

사용 예:
    python3 scripts/tools/safe_release_torque.py --port can_follower --mode lower
"""

import argparse
import sys
import time

from lerobot_robot_piper import PiperFollowerConfig, PiperFollower
from lerobot_robot_piper.motors.tables import INITIALIZE_POSITION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="can_follower", help="follower CAN 포트 (기본: can_follower)")
    parser.add_argument("--mode", default="lower", choices=["lower", "in_place", "park"],
                         help="해제 전 자세 (기본: lower)")
    parser.add_argument("--ramp-s", type=float, default=2.0,
                         help="lower 모드에서 손목을 내리는 데 걸리는 시간(초)")
    parser.add_argument("--wrist-rest-deg", type=float, default=24.4,
                         help="lower 모드에서 손목을 내려둘 각도(도, 절대값=자연 정지각)")
    parser.add_argument("--max-relative-target", type=float, default=15.0,
                         help="자세 이동 시 timestep별 최대 이동량")
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
        print("현재 위치: " + ", ".join(
            f"{j}={obs.get(f'{j}.pos', float('nan')):.1f}" for j in
            ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        ))

        # 자세 이동만 여기서 하고 torque 해제는 사람 확인 후에 한다 — 그래서
        # release_torque_safely()를 통째로 부르지 않고 이동 부분만 직접 호출.
        if args.mode == "park":
            # parking = 물리 각도 0도(팔이 위로 뻗은 자세). INITIALIZE_POSITION은 각
            # joint의 calibration 범위(비대칭 포함)로 역산한 값이라 정규화 0이 아니다
            # (joint2/joint3/joint6 — motors/tables.py 주석 참고).
            print("parking 자세(물리 각도 0도)로 이동합니다 (gripper는 유지)...")
            follower.bus.ramp_to(
                {j: INITIALIZE_POSITION[j] for j in
                 ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]},
                ramp_s=args.ramp_s,
            )
        elif args.mode == "lower":
            print(f"손목(joint5)을 {args.wrist_rest_deg}도(자연 정지각)까지 내립니다 (팔은 그대로)...")
            follower.bus.ramp_to(
                follower.bus.wrist_rest_target(args.wrist_rest_deg), ramp_s=args.ramp_s
            )
        else:
            print("이동 없이 현재 자세에서 해제합니다.")
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
