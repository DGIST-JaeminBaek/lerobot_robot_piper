from __future__ import annotations

"""WeGo Piper write AIIII 데이터셋 녹화 스크립트"""

import argparse
import subprocess

from common_env import camera_config, env_bool, env_value, load_env_file, print_command


def build_command(args: argparse.Namespace) -> list[str]:
    """recording.env와 CLI 인자를 합쳐 lerobot-record 명령 생성"""
    values = load_env_file(args.env_file)

    leader_port = args.leader_port or env_value(values, "LEADER_PORT", "can_leader1")
    follower_port = args.follower_port or env_value(values, "FOLLOWER_PORT", "can_follower1")
    robot_id = args.robot_id or env_value(values, "ROBOT_ID", "piper_follower1")
    teleop_id = args.teleop_id or env_value(values, "TELEOP_ID", "piper_leader1")
    dataset_repo_id = args.dataset_repo_id or env_value(values, "DATASET_REPO_ID", "local/piper_write_light")
    num_episodes = str(args.num_episodes or env_value(values, "NUM_EPISODES", "5"))
    episode_time_s = str(args.episode_time_s or env_value(values, "EPISODE_TIME_S", "60"))
    max_relative_target = str(args.max_relative_target or env_value(values, "MAX_RELATIVE_TARGET", "5"))
    task = args.task or env_value(values, "TASK", "write AIIII")
    push_to_hub = str(args.push_to_hub if args.push_to_hub is not None else env_bool(values, "PUSH_TO_HUB", False)).lower()
    display_data = str(args.display_data if args.display_data is not None else env_bool(values, "DISPLAY_DATA", True)).lower()

    return [
        "lerobot-record",
        "--robot.type=piper_follower",
        f"--robot.port={follower_port}",
        f"--robot.id={robot_id}",
        f"--robot.max_relative_target={max_relative_target}",
        f"--robot.cameras={camera_config(values)}",
        "--teleop.type=piper_leader",
        f"--teleop.port={leader_port}",
        f"--teleop.id={teleop_id}",
        f"--display_data={display_data}",
        f"--dataset.repo_id={dataset_repo_id}",
        f"--dataset.num_episodes={num_episodes}",
        f"--dataset.episode_time_s={episode_time_s}",
        f"--dataset.single_task={task}",
        f"--dataset.push_to_hub={push_to_hub}",
        "--robot.discover_packages_path=lerobot_robot_piper",
        "--teleop.discover_packages_path=lerobot_robot_piper",
    ]


def parse_args() -> argparse.Namespace:
    """CLI 옵션 정의"""
    parser = argparse.ArgumentParser(
        description="Record write AIIII demonstrations with WeGo Piper LeRobot plugin.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-file", default=None, help="recording.env 경로")
    parser.add_argument("--leader-port", default=None, help="leader CAN 포트")
    parser.add_argument("--follower-port", default=None, help="follower CAN 포트")
    parser.add_argument("--robot-id", default=None, help="LeRobot robot id")
    parser.add_argument("--teleop-id", default=None, help="LeRobot teleop id")
    parser.add_argument("--dataset-repo-id", default=None, help="owner/dataset_name")
    parser.add_argument("--num-episodes", type=int, default=None, help="녹화 episode 수")
    parser.add_argument("--episode-time-s", type=int, default=None, help="episode 제한 시간")
    parser.add_argument("--max-relative-target", default=None, help="follower step 이동 제한")
    parser.add_argument("--task", default=None, help="dataset single_task")
    parser.add_argument("--push-to-hub", type=lambda x: x.lower() in {"1", "true", "yes"}, default=None)
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
