from __future__ import annotations

"""WeGo Piper write AIIII 데이터셋 녹화 스크립트"""

import argparse
import subprocess
from pathlib import Path

from common_env import action_offset_args, camera_args, env_bool, env_value, load_env_file, print_command


def build_command(args: argparse.Namespace) -> list[str]:
    """recording.env와 CLI 인자를 합쳐 lerobot-record 명령 생성"""
    values = load_env_file(args.env_file)

    leader_port = args.leader_port or env_value(values, "LEADER_PORT", "can_leader1")
    follower_port = args.follower_port or env_value(values, "FOLLOWER_PORT", "can_follower1")
    dataset_repo_id = args.dataset_repo_id or env_value(values, "DATASET_REPO_ID", "local/piper_write_light")
    dataset_root = args.dataset_root or env_value(
        values,
        "DATASET_ROOT",
        str(Path(__file__).resolve().parents[1] / "records" / dataset_repo_id),
    )
    num_episodes = str(args.num_episodes or env_value(values, "NUM_EPISODES", "5"))
    episode_time_s = str(args.episode_time_s or env_value(values, "EPISODE_TIME_S", "60"))
    reset_time_s = str(args.reset_time_s or env_value(values, "RESET_TIME_S", "60"))
    fps = str(args.fps or env_value(values, "FPS", "30"))
    task = args.task or env_value(values, "TASK", "write AIIII")
    push_to_hub = str(args.push_to_hub if args.push_to_hub is not None else env_bool(values, "PUSH_TO_HUB", False)).lower()
    display_data = str(args.display_data if args.display_data is not None else env_bool(values, "DISPLAY_DATA", True)).lower()

    return [
        "lerobot-record",
        "--robot.type=piper_follower",
        f"--robot.port={follower_port}",
        *camera_args(values),
        *action_offset_args(values),
        "--teleop.type=piper_leader",
        f"--teleop.port={leader_port}",
        f"--display_data={display_data}",
        f"--dataset.repo_id={dataset_repo_id}",
        f"--dataset.root={dataset_root}",
        f"--dataset.fps={fps}",
        f"--dataset.num_episodes={num_episodes}",
        f"--dataset.episode_time_s={episode_time_s}",
        f"--dataset.reset_time_s={reset_time_s}",
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
    parser.add_argument("--dataset-repo-id", default=None, help="owner/dataset_name")
    parser.add_argument("--dataset-root", default=None, help="LeRobot dataset 저장 위치")
    parser.add_argument("--num-episodes", type=int, default=None, help="녹화 episode 수")
    parser.add_argument("--episode-time-s", type=int, default=None, help="episode 녹화 시간")
    parser.add_argument("--reset-time-s", type=int, default=None, help="episode 사이 reset 시간")
    parser.add_argument("--fps", type=int, default=None, help="dataset 및 camera fps")
    parser.add_argument("--task", default=None, help="dataset single_task")
    parser.add_argument("--push-to-hub", type=lambda x: x.lower() in {"1", "true", "yes"}, default=None)
    parser.add_argument("--display-data", type=lambda x: x.lower() in {"1", "true", "yes"}, default=None)
    parser.add_argument("--dry-run", action="store_true", help="명령 출력만 수행")
    return parser.parse_args()


def main() -> None:
    """명령 생성 후 실행"""
    args = parse_args()
    command = build_command(args)
    dataset_root = Path(command[command.index(next(part for part in command if part.startswith("--dataset.root=")))].split("=", 1)[1])
    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    print_command(command)
    if args.dry_run:
        return
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
