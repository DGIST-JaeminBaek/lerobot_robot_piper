#!/usr/bin/env python3
"""LeRobot 데이터셋의 기본적인 수치·프레임 연속성을 빠르게 점검한다.

사용법:
    python scripts/piper/validation/dataset_quick_check.py
    python scripts/piper/validation/dataset_quick_check.py --dataset-root records/local/my_dataset

기본 경로는 ``configs/recording.env``의 ``DATASET_ROOT``다. 이 검사는 데이터셋을
수정하지 않으며, all-zero state와 action 범위 오류는 실패(exit 1), action 급변과
frame_index 불연속은 경고로만 보고한다.
"""

import argparse
import os
import pathlib
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
RECORDING_ENV_PATH = REPO_ROOT / "configs" / "recording.env"
AXIS_RANGES = (
    ("joint1", -100, 100),
    ("joint2", -100, 100),
    ("joint3", -100, 100),
    ("joint4", -100, 100),
    ("joint5", -100, 100),
    ("joint6", -100, 100),
    ("gripper", 0, 100),
)


def recording_env_value(name: str, default: str = "") -> str:
    """Return environment override or a simple KEY=VALUE recording.env entry."""
    if value := os.environ.get(name):
        return os.path.expanduser(os.path.expandvars(value))
    try:
        lines = RECORDING_ENV_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return default
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == name:
            value = value.strip().strip("\"'")
            return os.path.expanduser(os.path.expandvars(value))
    return default


def check_dataset(root: pathlib.Path, max_delta: float) -> bool:
    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        print("[FAIL] pandas와 numpy가 필요합니다: pip install pandas numpy", file=sys.stderr)
        return False

    if not root.exists():
        print(f"[FAIL] 데이터셋 경로 없음: {root}", file=sys.stderr)
        return False
    parquets = sorted(root.glob("data/**/*.parquet"))
    if not parquets:
        print(f"[FAIL] parquet 파일 없음: {root}", file=sys.stderr)
        return False

    print(f"[INFO] {root}")
    print(f"[INFO] parquet {len(parquets)}개 검사")
    total_zero = 0
    range_fails: list[str] = []
    delta_warns: list[str] = []
    frame_gaps: list[str] = []

    for parquet in parquets:
        frame = pd.read_parquet(parquet)
        if "observation.state" in frame.columns:
            states = np.asarray(frame["observation.state"].tolist())
            zero_rows = int((states == 0).all(axis=1).sum())
            total_zero += zero_rows
            if zero_rows:
                print(f"[WARN] {parquet.name}: all-zero state {zero_rows}행")

        if "action" in frame.columns:
            actions = np.asarray(frame["action"].tolist())
            if actions.ndim != 2:
                range_fails.append(f"{parquet.name}: action 차원이 2차원이 아님")
            else:
                for index, (axis, lower, upper) in enumerate(AXIS_RANGES):
                    if index >= actions.shape[1]:
                        break
                    out_of_range = int(((actions[:, index] < lower) | (actions[:, index] > upper)).sum())
                    if out_of_range:
                        range_fails.append(f"{parquet.name}/{axis}: {out_of_range}행 범위 초과")
                if len(actions) > 1:
                    sudden = int((np.abs(np.diff(actions, axis=0)).max(axis=1) > max_delta).sum())
                    if sudden:
                        delta_warns.append(f"{parquet.name}: {sudden}스텝 급격한 action 변화")

        if "frame_index" in frame.columns:
            indexes = frame["frame_index"].to_numpy()
            gap_count = int(((indexes[1:] - indexes[:-1]) != 1).sum())
            if gap_count:
                frame_gaps.append(f"{parquet.name}: frame_index 불연속 {gap_count}곳")

    if total_zero:
        print(f"[FAIL] all-zero state 합계: {total_zero}행", file=sys.stderr)
    else:
        print("[OK] all-zero state 없음")
    for failure in range_fails:
        print(f"[FAIL] {failure}", file=sys.stderr)
    if not range_fails:
        print("[OK] action 범위 정상")
    for warning in delta_warns + frame_gaps:
        print(f"[WARN] {warning}")

    passed = total_zero == 0 and not range_fails
    print("[OK] 데이터셋 빠른 점검 통과" if passed else "[FAIL] 데이터셋 빠른 점검 실패")
    return passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default=recording_env_value("DATASET_ROOT"),
        help="LeRobot 데이터셋 루트 (기본: recording.env의 DATASET_ROOT)",
    )
    parser.add_argument(
        "--max-delta",
        type=float,
        default=20.0,
        help="프레임 간 action 최대 변화량 경고 기준 (기본: 20)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dataset_root:
        print("[FAIL] --dataset-root 또는 configs/recording.env의 DATASET_ROOT가 필요합니다.", file=sys.stderr)
        return 2
    return 0 if check_dataset(pathlib.Path(args.dataset_root), args.max_delta) else 1


if __name__ == "__main__":
    raise SystemExit(main())
