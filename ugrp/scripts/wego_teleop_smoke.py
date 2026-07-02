from __future__ import annotations

"""WeGo Piper 텔레옵 점검 실행 스크립트"""

import argparse
import subprocess

from common_env import env_bool, env_value, load_env_file, print_command


def build_command(args: argparse.Namespace) -> list[str]:
    """recording.env와 CLI 인자를 합쳐 lerobot-teleoperate 명령 생성"""
    values = load_env_file(args.env_file)

    leader_port = args.leader_port or env_value(values, "LEADER_PORT", "can_leader1")
    follower_port = args.follower_port or env_value(values, "FOLLOWER_PORT", "can_follower1")
    robot_id = args.robot_id or env_value(values, "ROBOT_ID", "piper_follower1")
    teleop_id = args.teleop_id or env_value(values, "TELEOP_ID", "piper_leader1")
    max_relative_target = args.max_relative_target or env_value(values, "MAX_RELATIVE_TARGET", "5")
    display_data = str(args.display_data if args.display_data is not None else env_bool(values, "DISPLAY_DATA", True)).lower()

    return [
        "lerobot-teleoperate",
        "--robot.type=piper_follower",
        f"--robot.port={follower_port}",
        f"--robot.id={robot_id}",
        f"--robot.max_relative_target={max_relative_target}",
        "--teleop.type=piper_leader",
        f"--teleop.port={leader_port}",
        f"--teleop.id={teleop_id}",
        f"--display_data={display_data}",
        "--robot.discover_packages_path=lerobot_robot_piper",
        "--teleop.discover_packages_path=lerobot_robot_piper",
    ]


def parse_args() -> argparse.Namespace:
    """CLI 옵션 정의"""
    parser = argparse.ArgumentParser(
        description="Run a safe WeGo Piper leader/follower teleoperation smoke test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-file", default=None, help="recording.env 경로")
    parser.add_argument("--leader-port", default=None, help="leader CAN 포트")
    parser.add_argument("--follower-port", default=None, help="follower CAN 포트")
    parser.add_argument("--robot-id", default=None, help="LeRobot robot id")
    parser.add_argument("--teleop-id", default=None, help="LeRobot teleop id")
    parser.add_argument("--max-relative-target", default=None, help="follower step 이동 제한")
    parser.add_argument("--display-data", type=lambda x: x.lower() in {"1", "true", "yes"}, default=None)
    parser.add_argument("--dry-run", action="store_true", help="명령 출력만 수행")
    return parser.parse_args()


def main() -> None:
    """명령 생성 후 실행"""
    args = parse_args()
    command = build_command(args)
    print_command(command)
    if args.dry_run:
        return
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
