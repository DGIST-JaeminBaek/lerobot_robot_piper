#!/usr/bin/env python3
"""이미 만든 lerobot 데이터셋을 재인코딩 없이 다른 task 프롬프트로 복사한다.

비디오/state/action은 그대로 두고 task 텍스트만 바꾸는 경우(단일-task 데이터셋 한정)를
위한 지름길. task_index는 안 바뀐다 — 두 데이터셋 다 task가 하나뿐이라 항상 0이라서다.
프레임/카메라 구성을 바꾸는 경우는 prepare_erase_shape_dataset.py로 원본부터 다시
빌드해야 한다.
"""

from __future__ import annotations

import argparse
import glob
import shutil
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--new-task", required=True)
    args = parser.parse_args()

    if args.output.exists():
        print(f"[ERROR] output already exists: {args.output}", file=sys.stderr)
        return 1

    tasks_path = args.source / "meta" / "tasks.parquet"
    old_tasks = pd.read_parquet(tasks_path)
    if len(old_tasks) != 1:
        print(f"[ERROR] expected a single-task dataset, found {len(old_tasks)}", file=sys.stderr)
        return 1
    old_task = old_tasks.index[0]
    print(f"[COPY] {args.source} -> {args.output}")
    shutil.copytree(args.source, args.output)

    new_tasks = pd.DataFrame({"task_index": [0]}, index=pd.Index([args.new_task], name=None))
    new_tasks.to_parquet(args.output / "meta" / "tasks.parquet")
    print(f"[TASKS] {old_task!r} -> {args.new_task!r}")

    episode_files = glob.glob(str(args.output / "meta" / "episodes" / "**" / "*.parquet"), recursive=True)
    for path in episode_files:
        df = pd.read_parquet(path)
        df["tasks"] = df["tasks"].apply(lambda _: [args.new_task])
        df.to_parquet(path)
    print(f"[EPISODES] updated 'tasks' column in {len(episode_files)} file(s)")

    # sanity check
    check = pd.read_parquet(args.output / "meta" / "tasks.parquet")
    assert list(check.index) == [args.new_task], check
    print(f"PASS: {args.output} now has task={args.new_task!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
