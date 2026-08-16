#!/usr/bin/env python3
"""piper_mit_probe.py — MIT(임피던스) 제어를 관절 하나씩 안전하게 확인한다.

MIT는 토크 제어다. `토크 = kp*(pos_ref - pos) + kd*(vel_ref - vel) + t_ref` 이므로
kp가 낮으면 팔이 중력에 무너지고, 높으면 격렬하게 진동한다. 정책 추론 전체를
MIT로 돌리기 전에 이 도구로 **관절 하나씩, 낮은 게인부터** 확인할 것.

무엇을 하나:
  1. 현재 자세를 읽어 그 자리를 목표로 삼는다(움직일 이유가 없는 상태에서 시작).
  2. 지정한 관절 하나만 MIT로 잡고, 나머지는 건드리지 않는다.
  3. 목표를 제자리에 둔 채 hold_s 동안 유지하며 위치 오차를 관찰한다
     — 여기서 무너지거나 떨면 그 게인은 쓰면 안 된다.
  4. --amplitude를 주면 그 관절만 사인파로 작게 왕복시켜 추종을 본다.

무너짐 감지: 목표에서 collapse_deg 이상 벗어나면 즉시 위치 제어로 되돌린다.

사용 예 (권장 순서):
    # 1) 가장 낮은 게인으로 제자리 유지만
    python scripts/tools/piper_mit_probe.py --joint 1 --kp 5 --hold-s 3

    # 2) 괜찮으면 게인을 올려가며
    python scripts/tools/piper_mit_probe.py --joint 1 --kp 10
    python scripts/tools/piper_mit_probe.py --joint 1 --kp 20

    # 3) 작은 왕복으로 추종 확인 (정규화 단위 진폭)
    python scripts/tools/piper_mit_probe.py --joint 1 --kp 10 --amplitude 3 --cycles 2

⚠ 팔 주변을 비우고 비상 정지가 가능한 상태에서만 실행할 것. joint2/joint3처럼
중력 모멘트가 큰 관절은 특히 조심 — 낮은 kp에서 팔이 떨어질 수 있다.
"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import sys
import time

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for path in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

CONFIRM = "I_UNDERSTAND_TORQUE_CONTROL"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MIT 제어를 관절 하나씩 확인한다 (토크 제어 — 주의)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--joint", type=int, required=True, choices=range(1, 7))
    parser.add_argument("--kp", type=float, default=5.0, help="낮은 값부터 시작할 것 (SDK 참고값 10)")
    parser.add_argument("--kd", type=float, default=0.8)
    parser.add_argument("--hold-s", type=float, default=3.0, help="제자리 유지 시간")
    parser.add_argument("--amplitude", type=float, default=0.0, help="사인파 진폭(정규화 단위). 0=제자리만")
    parser.add_argument("--period-s", type=float, default=4.0)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--collapse-deg",
        type=float,
        default=8.0,
        help="목표에서 이만큼(정규화 단위) 벗어나면 무너짐으로 보고 즉시 중단",
    )
    parser.add_argument(
        "--goto",
        type=float,
        help="MIT를 켜기 전에 이 관절을 여기(정규화 -100~100)로 먼저 이동시킨다. "
        "중력 부담이 큰 자세에서 재기 위한 것 — probe가 끝날 때마다 파킹하므로 "
        "그냥 실행하면 항상 접힌 자세(중력 모멘트≈0)에서 재게 되어 결과가 무의미하다",
    )
    parser.add_argument("--goto-ramp-s", type=float, default=3.0, help="--goto 이동 시간")
    parser.add_argument(
        "--no-park",
        dest="park",
        action="store_false",
        help="끝나고 파킹하지 않고 그 자세에 둔다 — 같은 자세에서 게인을 바꿔가며 잴 때",
    )
    parser.add_argument("--confirm", default="", help=f"실행하려면 {CONFIRM}")
    parser.add_argument("--port", default=os.environ.get("FOLLOWER_PORT", "can_follower"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.confirm != CONFIRM:
        print(f"[ERROR] 토크 제어입니다. --confirm={CONFIRM} 을 함께 주세요.", file=sys.stderr)
        return 2

    from lerobot_robot_piper.config_piper import PiperFollowerConfig
    from lerobot_robot_piper.motors.piper_motors_bus import JOINT_NAMES
    from lerobot_robot_piper.piper_follower import PiperFollower

    motor = JOINT_NAMES[args.joint - 1]
    # 카메라 없이, parking 없이 — 팔만 잡는다.
    config = PiperFollowerConfig(port=args.port, park_on_connect=False, use_effort=True)
    robot = PiperFollower(config)

    print(f"[CONNECT] {args.port}")
    robot.connect(calibrate=False)
    bus = robot.bus
    status = 0
    try:
        if args.goto is not None:
            # 위치 제어로 먼저 이동한다. MIT를 켠 상태로는 큰 이동을 시키면
            # 오차가 커져 토크가 튀므로, 이동은 기존 위치 제어에 맡긴다.
            print(f"[GOTO] {motor} → {args.goto:.2f} ({args.goto_ramp_s:g}초에 걸쳐 이동)")
            goal = bus.get_action()
            goal[motor] = args.goto
            bus.ramp_to(goal, ramp_s=args.goto_ramp_s)
            time.sleep(0.5)

        start = bus.get_action()
        target = dict(start)
        print(f"[START] {motor} = {start[motor]:.2f} (정규화)")
        if abs(start[motor]) > 97.0:
            print(
                f"[WARN] {motor}가 범위 끝({start[motor]:.1f})에 있습니다 — 접힌 자세라 "
                "중력 모멘트가 거의 0이라 이 결과는 실제 작업 조건을 대표하지 않습니다. "
                "--goto로 팔을 뻗은 자세로 보낸 뒤 다시 재세요.",
                file=sys.stderr,
            )
        print(f"[MIT] kp={args.kp:g} kd={args.kd:g} — 이 관절만 잡습니다")

        period = 1.0 / args.fps
        total_s = args.hold_s + (args.period_s * args.cycles if args.amplitude else 0.0)
        began = time.perf_counter()
        worst = 0.0

        while True:
            now = time.perf_counter() - began
            if now >= total_s:
                break

            velocity = {motor: 0.0}
            if args.amplitude and now > args.hold_s:
                phase = 2 * math.pi * (now - args.hold_s) / args.period_s
                target[motor] = start[motor] + args.amplitude * math.sin(phase)
                # 사인파의 해석적 미분 — MIT의 vel_ref로 넘긴다.
                velocity[motor] = (
                    args.amplitude * 2 * math.pi / args.period_s * math.cos(phase)
                )

            bus.set_action_mit({motor: target[motor]}, velocity, kp=args.kp, kd=args.kd)

            measured = bus.get_action()[motor]
            error = abs(measured - target[motor])
            worst = max(worst, error)
            if error > args.collapse_deg:
                print(
                    f"\n[ABORT] 오차 {error:.2f} > {args.collapse_deg:g} — "
                    f"게인이 이 관절을 지탱하지 못합니다 (kp={args.kp:g})",
                    file=sys.stderr,
                )
                status = 1
                break

            if int(now * 5) != int((now - period) * 5):
                print(
                    f"  t={now:5.1f}s  목표 {target[motor]:7.2f}  실측 {measured:7.2f}  "
                    f"오차 {error:5.2f}",
                    flush=True,
                )
            time.sleep(max(0.0, period - ((time.perf_counter() - began) - now)))

        print(f"\n[RESULT] 최대 오차 {worst:.2f} (정규화 단위)")
        if status == 0:
            print("[RESULT] 무너짐 없음 — 이 게인은 이 관절에서 유지 가능합니다")
    except KeyboardInterrupt:
        print("\n[INTERRUPT] 중단", file=sys.stderr)
        status = 1
    finally:
        # 반드시 위치 제어로 되돌린 뒤 정리한다 — MIT 상태에서는 parking()이
        # 위치 명령이라 먹지 않는다.
        try:
            bus.leave_mit_mode()
            print("[ROBOT] MIT 해제 — 위치 제어로 복귀")
        except Exception as error:
            print(f"[WARN] MIT 해제 실패: {error}", file=sys.stderr)
        try:
            if args.park:
                robot.disconnect(park=True)
            else:
                # 같은 자세에서 게인을 바꿔가며 잴 때 매번 파킹으로 돌아가면
                # 조건이 달라진다. 다만 뻗은 자세에서 토크를 풀면 팔이 떨어지므로
                # 토크는 켜둔 채로 나간다 — 해제는 사람이 따로 해야 한다.
                robot.disconnect(disable_torque=False, park=False)
                print(
                    "[ROBOT] 파킹 없이 종료 — 팔이 현재 자세에 남고 "
                    "\033[1m토크가 켜진 상태\033[0m입니다."
                )
                print(
                    "        해제하려면: python scripts/tools/safe_release_torque.py"
                )
        except Exception as error:
            print(f"[WARN] disconnect 실패: {error}", file=sys.stderr)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
