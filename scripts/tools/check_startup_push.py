#!/usr/bin/env python3
"""시작 직후 팔로워가 "아래로 누르는 힘"이 생기는 원인 진단 (팔 이동 명령 없음).

가설: 컨트롤러는 마지막으로 받은 JointCtrl 목표를 기억한다. torque를 켜는 순간
그 목표가 현재 실제 자세보다 낮으면(예: 지난 세션 종료 때 손목을 미리 내려놓고
껐다면) 팔이 그 목표로 가려고 하면서 아래로 누른다.

그래서 다음 세 가지를 이동 명령 없이 읽기만 한다:
  1. torque 켜기 전 실제 자세
  2. 컨트롤러가 들고 있는 목표(GetArmJointCtrl) — bus.get_control()
  3. torque 켠 뒤 자세/effort 변화

목표(2)가 실제 자세(1)보다 낮은 쪽으로 차이가 나면 그게 누르는 힘의 원인이다.

사용: PYTHONPATH=. python scripts/tools/check_startup_push.py --port can_follower
"""

import argparse
import time

from lerobot_robot_piper import PiperFollowerConfig, PiperFollower

JOINTS = [f"joint{n}" for n in range(1, 7)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="can_follower")
    parser.add_argument("--settle-s", type=float, default=3.0,
                        help="torque 켠 뒤 관찰 시간(초)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PiperFollowerConfig(
        port=args.port, park_on_connect=False, use_action_offset=False,
        disable_torque_on_disconnect=False,
    )
    f = PiperFollower(cfg)
    f.bus.connect()  # 포트만 열기 — 아직 torque 안 켬

    def deg(m, n):
        c = f.bus.calibration[m]
        return (((n + 100) / 200) * (c.range_max - c.range_min) + c.range_min) / 1000.0

    # connect 직후엔 SDK 피드백 구조체가 아직 비어 있어서 모든 값이 0으로 읽힌다 —
    # 그대로 읽으면 "현재 자세 0도"라는 가짜 값이 나온다(진단 자체가 틀어짐).
    # 값이 두 번 연속 같아질 때까지(=실제 피드백이 들어와 안정될 때까지) 기다린다.
    prev, stable = None, 0
    for _ in range(50):  # 최대 5초
        time.sleep(0.1)
        now = f.bus.get_action()
        if prev is not None and all(abs(now[j] - prev[j]) < 1e-6 for j in JOINTS):
            stable += 1
            if stable >= 3 and any(abs(now[j]) > 1e-6 for j in JOINTS):
                break
        else:
            stable = 0
        prev = now
    else:
        print("[WARN] 피드백이 안정되지 않았거나 전 관절이 정확히 0입니다 — 값 해석 주의")

    before = f.bus.get_action()
    control = f.bus.get_control()
    eff_before = f.bus.get_effort()

    print(f"{'joint':8} {'현재자세(deg)':>13} {'컨트롤러목표(deg)':>17} {'차이':>8} {'effort':>8}")
    for j in JOINTS:
        d = deg(j, control[j]) - deg(j, before[j])
        print(f"{j:8} {deg(j, before[j]):13.1f} {deg(j, control[j]):17.1f} "
              f"{d:+8.1f} {eff_before[j]:8.2f}")
    worst = max(JOINTS, key=lambda j: abs(deg(j, control[j]) - deg(j, before[j])))
    gap = deg(worst, control[worst]) - deg(worst, before[worst])
    print(f"\n가장 큰 목표-실제 차이: {worst} {gap:+.1f}도")
    print("  -> 이 값이 크면, torque를 켜는 순간 그 방향으로 밀면서 '누르는 힘'이 생긴다.")

    print(f"\ntorque 켜고 {args.settle_s:.0f}초 관찰...")
    f.bus.enable_torque()
    time.sleep(args.settle_s)
    after = f.bus.get_action()
    eff_after = f.bus.get_effort()
    print(f"{'joint':8} {'변화(deg)':>10} {'effort 전':>10} {'effort 후':>10}")
    for j in JOINTS:
        print(f"{j:8} {deg(j, after[j]) - deg(j, before[j]):10.2f} "
              f"{eff_before[j]:10.2f} {eff_after[j]:10.2f}")

    f.bus.disconnect(disable_torque=False, park=False)


if __name__ == "__main__":
    main()
