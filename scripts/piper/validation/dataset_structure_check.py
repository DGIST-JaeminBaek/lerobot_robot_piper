#!/usr/bin/env python3
"""LeRobot Piper 데이터셋의 metadata, feature 이름, episode 기본 구조를 점검한다."""

from __future__ import annotations

import argparse
from pprint import pprint

import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata


EXPECTED_ACTION_NAMES = [
    "joint1.pos",
    "joint2.pos",
    "joint3.pos",
    "joint4.pos",
    "joint5.pos",
    "joint6.pos",
    "gripper.pos",
]


def _feature_names(meta: LeRobotDatasetMetadata, key: str) -> list[str]:
    """LeRobot feature 이름을 안전하게 추출한다."""
    feature = meta.features.get(key, {})
    names = feature.get("names", [])
    return list(names) if names else []


def check_dataset(args: argparse.Namespace) -> None:
    """데이터셋 metadata와 지정 episode의 parquet 기본 구조를 출력한다."""
    meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)

    print("Dataset root:", meta.root)
    print("FPS:", meta.fps)
    print("Features:")
    pprint(meta.features)

    action_names = _feature_names(meta, "action")
    state_names = _feature_names(meta, "observation.state")
    print("Action names:", action_names)
    print("State names:", state_names)
    print("Expected Piper joint names:", EXPECTED_ACTION_NAMES)

    if action_names and action_names != EXPECTED_ACTION_NAMES:
        print("[WARN] action names differ from Piper follower action order")
    if state_names and state_names != EXPECTED_ACTION_NAMES:
        print("[WARN] observation.state names differ from Piper joint-state order")

    if args.episode is None:
        return

    data_path = meta.root / meta.get_data_file_path(args.episode)
    frame = pd.read_parquet(data_path)
    frame = frame[frame["episode_index"] == args.episode].reset_index(drop=True)
    print("Episode:", args.episode)
    print("Rows:", len(frame))
    print("Columns:", list(frame.columns))

    if "action" in frame.columns and len(frame):
        actions = frame["action"].to_list()
        print("First action:", actions[0])
        print("Last action:", actions[-1])
    if "observation.state" in frame.columns and len(frame):
        states = frame["observation.state"].to_list()
        print("First state:", states[0])
        print("Last state:", states[-1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-repo-id", required=True, help="owner/dataset_name 또는 local repo id")
    parser.add_argument("--dataset-root", default=None, help="로컬 dataset root")
    parser.add_argument("--episode", type=int, default=None, help="점검할 episode index")
    return parser.parse_args()


def main() -> None:
    check_dataset(parse_args())


if __name__ == "__main__":
    main()
