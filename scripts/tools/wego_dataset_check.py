from __future__ import annotations

"""WeGo Piper LeRobot dataset 구조 점검 스크립트"""

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
    """LeRobot feature names 안전 추출"""
    feature = meta.features.get(key, {})
    names = feature.get("names", [])
    return list(names) if names else []


def check_dataset(args: argparse.Namespace) -> None:
    """dataset metadata와 episode parquet 기본 점검"""
    meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)

    print("Dataset root:", meta.root)
    print("FPS:", meta.fps)
    print("Features:")
    pprint(meta.features)

    action_names = _feature_names(meta, "action")
    state_names = _feature_names(meta, "observation.state")

    print("Action names:", action_names)
    print("State names:", state_names)
    print("Expected WeGo action names:", EXPECTED_ACTION_NAMES)

    if action_names and action_names != EXPECTED_ACTION_NAMES:
        print("[WARN] action names differ from WeGo piper_follower/piper_leader expectation")
    if state_names and state_names != EXPECTED_ACTION_NAMES:
        print("[WARN] observation.state names differ from WeGo joint-state expectation")

    if args.episode is None:
        return

    data_path = meta.root / meta.get_data_file_path(args.episode)
    df = pd.read_parquet(data_path)
    df = df[df["episode_index"] == args.episode].reset_index(drop=True)
    print("Episode:", args.episode)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    if "action" in df.columns and len(df):
        actions = df["action"].to_list()
        print("First action:", actions[0])
        print("Last action:", actions[-1])
    if "observation.state" in df.columns and len(df):
        states = df["observation.state"].to_list()
        print("First state:", states[0])
        print("Last state:", states[-1])


def parse_args() -> argparse.Namespace:
    """CLI 옵션 정의"""
    parser = argparse.ArgumentParser(
        description="Inspect a LeRobot dataset recorded with WeGo Piper plugin.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-repo-id", required=True, help="owner/dataset_name 또는 local repo id")
    parser.add_argument("--dataset-root", default=None, help="로컬 dataset root")
    parser.add_argument("--episode", type=int, default=None, help="점검할 episode index")
    return parser.parse_args()


def main(args):
    """dataset 점검 실행"""
    check_dataset(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect a LeRobot dataset recorded with WeGo Piper plugin.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-repo-id", required=True, help="owner/dataset_name 또는 local repo id")
    parser.add_argument("--dataset-root", default=None, help="로컬 dataset root")
    parser.add_argument("--episode", type=int, default=None, help="점검할 episode index")
    args = parser.parse_args()
    main(args)

