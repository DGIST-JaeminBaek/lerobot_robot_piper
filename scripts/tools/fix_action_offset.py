#!/usr/bin/env python
"""action offset warmup으로 망가진 LeRobotDataset(v3.0) 후처리.

무엇을 고치는가
  - action이 offset warmup 때문에 어긋난 에피소드에서, action을 observation.state로
    재구성한다. state는 follower의 실제 present position이라 offset 오염이 없다.
    action[t] := state[t + lead]  (팔이 실제로 도달한 자세 = 올바른 지도신호)
  - 선두의 무동작/warmup 구간을 잘라낸다.

영상은 건드리지 않는다. v3.0은 에피소드별로 videos/<key>/from_timestamp 오프셋을 두고
mp4를 공유하므로, 앞을 자른 만큼 from_timestamp를 밀면 재인코딩 없이 동기가 맞는다.

usage:
  python fix_action_offset.py SRC_ROOT DST_ROOT [--mode absolute|delta] [--lead 1]
                              [--still 0.3] [--min-frames 30] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def leading_trim(state: np.ndarray, action: np.ndarray, still_thresh: float) -> int:
    """자를 선두 프레임 수. warmup(action==state) 구간과 무동작 구간 중 긴 쪽."""
    same = np.abs(action - state).max(axis=1) < 1e-6
    warmup = int(np.argmax(~same)) if (~same).any() else len(same)

    # 스텝 단위 속도가 아니라 시작 자세로부터의 누적 변위로 판정한다.
    # 천천히 출발하는 궤적을 "정지"로 오판하지 않기 위해서.
    moved = np.abs(state - state[0]).max(axis=1) > still_thresh
    dead = int(np.argmax(moved)) if moved.any() else len(state)

    return max(warmup, dead)


def rebuild_action(state: np.ndarray, lead: int, mode: str) -> np.ndarray:
    """state로부터 action 재구성. lead 프레임 뒤의 실제 자세를 목표값으로 삼는다."""
    idx = np.minimum(np.arange(len(state)) + lead, len(state) - 1)
    target = state[idx]
    return target - state if mode == "delta" else target


def measure_lead(state: np.ndarray, action: np.ndarray, max_lead: int = 10) -> int:
    """실측 추종 지연(프레임). 모델은 action[t] = state[t+k] - offset 이므로,
    각 k마다 상수 offset을 최소제곱으로 빼고 잔차가 가장 작은 k를 고른다.
    (상호상관은 손글씨처럼 매끄러운 궤적에서 거의 평평해져 못 쓴다.)"""
    if len(state) < max_lead + 10:
        return 1
    scores = []
    for k in range(max_lead + 1):
        a, s = action[: len(action) - k], state[k:]
        resid = a - s
        scores.append(float(np.abs(resid - resid.mean(axis=0)).mean()))
    return int(np.argmin(scores))


def process(src: Path, dst: Path, args) -> None:
    info = json.loads((src / "meta" / "info.json").read_text())
    fps = info["fps"]
    eps = pd.read_parquet(src / "meta" / "episodes")
    data = pd.read_parquet(src / "data")
    video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    action_dim = info["features"]["action"]["shape"][0]

    out_rows, out_eps = [], []
    cursor = 0
    for _, ep in eps.iterrows():
        chunk = data.iloc[ep["dataset_from_index"] : ep["dataset_to_index"]].reset_index(drop=True)
        # observation.state는 [pos*7, effort*7, vel*6] 20차원. action은 pos 7차원뿐이라
        # 앞 7개(pos)만 잘라서 맞춘다.
        state = np.stack(chunk["observation.state"].to_numpy()).astype(np.float32)[:, : action_dim]
        action = np.stack(chunk["action"].to_numpy()).astype(np.float32)

        clipped = ((np.abs(state) >= 99.99).mean(axis=0) * 100)
        n_trim = leading_trim(state, action, args.still)
        # 지연 추정은 반드시 트리밍 이후 구간에서. warmup 경계에는 offset이 튀는
        # 한 프레임짜리 스파이크가 있어서 상호상관을 통째로 망친다.
        lead = args.lead if args.lead >= 0 else measure_lead(state[n_trim:], action[n_trim:])
        kept = len(chunk) - n_trim

        print(
            f"ep{int(ep['episode_index']):03d} {len(chunk):4d}f -> {kept:4d}f "
            f"(trim {n_trim}, {n_trim / fps:.1f}s, lead={lead})"
            + (f"  clip%={np.round(clipped, 0)}" if clipped.max() > 1 else "")
            + ("  [DROP: too short]" if kept < args.min_frames else "")
        )
        if kept < args.min_frames:
            continue

        chunk = chunk.iloc[n_trim:].reset_index(drop=True)
        state, action = state[n_trim:], action[n_trim:]
        chunk["action"] = list(rebuild_action(state, lead, args.mode))
        chunk["frame_index"] = np.arange(kept)
        chunk["index"] = cursor + np.arange(kept)
        chunk["timestamp"] = np.arange(kept, dtype=np.float32) / fps

        new_ep = ep.copy()
        new_ep["length"] = kept
        new_ep["dataset_from_index"] = cursor
        new_ep["dataset_to_index"] = cursor + kept
        # 영상은 그대로 두고 시작 오프셋만 민다 — 재인코딩 불필요
        for vk in video_keys:
            new_ep[f"videos/{vk}/from_timestamp"] = ep[f"videos/{vk}/from_timestamp"] + n_trim / fps

        out_rows.append(chunk)
        out_eps.append(new_ep)
        cursor += kept

    if args.dry_run:
        print(f"\n[dry-run] {len(out_eps)}/{len(eps)} eps, {cursor} frames — 쓰지 않음")
        return

    # 영상·메타 복사 후 데이터/에피소드/통계만 덮어쓴다 (원본 보존)
    if dst.exists():
        raise SystemExit(f"{dst} already exists")
    shutil.copytree(src, dst)
    shutil.rmtree(dst / "data")

    new_data = pd.concat(out_rows, ignore_index=True)
    new_eps = pd.DataFrame(out_eps).reset_index(drop=True)
    new_eps["meta/episodes/chunk_index"] = 0
    new_eps["meta/episodes/file_index"] = 0
    new_data["data/chunk_index"] = 0
    new_data["data/file_index"] = 0

    # ponytail: 단일 chunk/file로 몰아 쓴다. 데이터가 커져 파일 분할이 필요해지면
    # lerobot의 update_chunk_file_indices로 나눌 것.
    (dst / "data" / "chunk-000").mkdir(parents=True)
    new_data.to_parquet(dst / "data" / "chunk-000" / "file-000.parquet", index=False)
    shutil.rmtree(dst / "meta" / "episodes")
    (dst / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    new_eps.to_parquet(dst / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False)

    info["total_episodes"] = len(new_eps)
    info["total_frames"] = int(cursor)
    info["splits"] = {"train": f"0:{len(new_eps)}"}
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    update_stats(dst, new_data)
    print(f"\n{dst}: {len(new_eps)} eps / {cursor} frames")
    print("stats.json의 수치형 키만 재계산됨 (이미지 통계는 원본 유지)")


def update_stats(dst: Path, data: pd.DataFrame) -> None:
    """observation.state / action 통계 재계산. 정규화가 여기에 걸리므로 필수."""
    path = dst / "meta" / "stats.json"
    stats = json.loads(path.read_text())
    for key in ("observation.state", "action"):
        if key not in stats or key not in data:
            continue
        arr = np.stack(data[key].to_numpy()).astype(np.float64)
        stats[key] = {
            "min": arr.min(0).tolist(),
            "max": arr.max(0).tolist(),
            "mean": arr.mean(0).tolist(),
            "std": arr.std(0).tolist(),
            "count": [len(arr)],
        }
    path.write_text(json.dumps(stats, indent=4))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    p.add_argument("--mode", choices=["absolute", "delta"], default="absolute",
                   help="absolute=follower 절대 목표(리플레이 호환), delta=상대 변위(에피소드 간 좌표계 통일)")
    p.add_argument("--lead", type=int, default=1, help="목표 자세 선행 프레임. -1이면 원본에서 실측")
    p.add_argument("--still", type=float, default=0.3, help="무동작 판정 임계 (시작 자세로부터의 누적 변위, 정규화 단위)")
    p.add_argument("--min-frames", type=int, default=30)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    process(args.src, args.dst, args)


if __name__ == "__main__":
    main()
