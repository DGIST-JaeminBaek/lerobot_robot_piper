#!/usr/bin/env python3
"""smooth_start_frames.py — 녹화 초반 프레임을 parking에서 시작하도록 보정.

각 에피소드(LeRobotDataset v3.0)의 초반 N 프레임(`observation.state`, `action`)을
parking position에서 시작해 (N+1)번째 프레임의 값까지 **선형 보간**으로 덮어쓴다.
결과적으로 모든 에피소드가 동일한 parking 자세에서 출발해 자연스럽게 실제 시연으로
이어진다.

    v[i] = parking + (v_real[N] - parking) * (i / N)     (i = 0 .. N-1)

  - i=0 은 정확히 parking, i=N (=(N+1)번째 프레임) 은 원본 그대로라 연속적으로 이어짐.
  - N 이 에피소드 길이보다 크면 마지막 프레임을 타깃으로 자동 축소한다.

비디오 프레임은 재생성이 불가능하므로 건드리지 않는다. 실제 녹화가 이미 parking 근처에서
시작(park_on_connect / 텔레옵 시작 위치)하기 때문에 상태-영상 불일치는 초반 소수 프레임에
한정된다.

`--recompute-stats` (기본 켜짐) 시 lerobot 자체 stats 유틸(compute_episode_stats /
aggregate_stats)로 `action`·`observation.state` 의 per-episode / 전역 통계를 다시 계산해
`meta/episodes/*.parquet` 와 `meta/stats.json` 에 반영한다. 이미지/타임스탬프 등 나머지
통계는 값이 바뀌지 않으므로 그대로 둔다.

사용 예:
    python scripts/tools/smooth_start_frames.py records/local/piper_write_light
    python scripts/tools/smooth_start_frames.py <dataset_root> --num-frames 100 --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger("smooth_start_frames")

# 보정 대상 피처. 비디오/인덱스류는 제외.
DEFAULT_FEATURES = ("observation.state", "action")


def _load_initialize_position() -> dict[str, float]:
    """리포 루트를 sys.path 에 넣고 실제 parking 값(단일 소스)을 import.

    import 실패 시 조용히 틀린 parking으로 데이터를 덮어쓰면 안 되므로 명시적으로
    에러를 낸다(파킹 값은 tables.py에서 재보정될 수 있어 하드코딩 fallback 금지)."""
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from lerobot_robot_piper.motors.tables import INITIALIZE_POSITION
    except Exception as e:  # 리포 밖에서 실행 등
        raise RuntimeError(
            "INITIALIZE_POSITION import 실패 — 리포 루트에서 실행하거나 "
            "lerobot_robot_piper 패키지가 import 가능한 환경에서 실행하세요. "
            f"(원인: {e})"
        ) from e
    return {k: float(v) for k, v in INITIALIZE_POSITION.items()}


def _parking_vector(feature_names: list[str], parking: dict[str, float]) -> np.ndarray:
    """피처 이름 순서(예: ['joint1.pos', ..., 'gripper.pos'])에 맞춘 parking 벡터."""
    vec = []
    for name in feature_names:
        key = name.split(".")[0]  # "joint1.pos" -> "joint1"
        if key not in parking:
            raise KeyError(f"parking position에 '{key}' (from '{name}') 없음: {list(parking)}")
        vec.append(parking[key])
    return np.asarray(vec, dtype=np.float32)


def _table_2d(table: pa.Table, col: str) -> np.ndarray:
    """fixed_size_list 컬럼을 (n, dim) float32 2D 배열로."""
    arrs = table.column(col).to_numpy(zero_copy_only=False)
    return np.stack(arrs).astype(np.float32)


def _set_2d_column(table: pa.Table, col: str, data2d: np.ndarray) -> pa.Table:
    """2D 배열을 원래 fixed_size_list 스키마 그대로 컬럼에 되써넣기."""
    field = table.schema.field(col)
    dim = data2d.shape[1]
    flat = pa.array(data2d.reshape(-1), type=field.type.value_type)
    new_col = pa.FixedSizeListArray.from_arrays(flat, dim)
    idx = table.schema.get_field_index(col)
    return table.set_column(idx, field, new_col)


def _interpolate_start(
    values: np.ndarray, parking: np.ndarray, num_frames: int
) -> np.ndarray:
    """단일 에피소드(프레임 순서대로 정렬된) 값 배열의 초반부를 보간해 새 배열 반환.

    values: (L, dim). parking: (dim,). num_frames: 덮어쓸 초반 프레임 수 N.
    타깃 = values[N] (없으면 마지막 프레임). i=0..end-1 을 parking->타깃 선형 보간.
    """
    L = values.shape[0]
    if L <= 1:
        return values  # 보간 불가
    n = min(num_frames, L - 1)  # 타깃 인덱스(경계 clamp)
    target = values[n]
    out = values.copy()
    # i = 0 .. n-1 을 덮어씀. t = i/n -> i=0에서 parking, i=n에서 target(원본 유지)
    ts = (np.arange(n, dtype=np.float32) / float(n))[:, None]
    out[:n] = parking[None, :] * (1.0 - ts) + target[None, :] * ts
    return out


def smooth_start(
    dataset_root: str | pathlib.Path,
    num_frames: int = 100,
    features: tuple[str, ...] = DEFAULT_FEATURES,
    recompute_stats: bool = True,
    parking: dict[str, float] | None = None,
    dry_run: bool = False,
) -> dict:
    """데이터셋의 모든 에피소드 초반 num_frames 프레임을 parking 시작으로 보정.

    반환: {"episodes": n, "frames_edited": m, "files": k} 요약 dict.
    """
    root = pathlib.Path(dataset_root)
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"info.json 없음: {info_path} (LeRobotDataset root 확인)")
    info = json.loads(info_path.read_text())
    all_features = info["features"]

    if parking is None:
        parking = _load_initialize_position()

    # 편집 대상 피처만 (존재하는 것만), 각 피처의 parking 벡터 준비
    edit_features = [f for f in features if f in all_features]
    if not edit_features:
        raise ValueError(f"편집 대상 피처가 데이터셋에 없음: {features}")
    parking_vecs = {
        f: _parking_vector(all_features[f]["names"], parking) for f in edit_features
    }

    data_files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"data parquet 없음: {root/'data'}")

    # per-episode 편집 후 값(통계 재계산용) 보관: {episode_index: {feature: (L,dim)}}
    episode_arrays: dict[int, dict[str, np.ndarray]] = {}
    total_frames_edited = 0

    for f in data_files:
        table = pq.read_table(f)
        ep_idx = table.column("episode_index").to_numpy(zero_copy_only=False)
        frame_idx = table.column("frame_index").to_numpy(zero_copy_only=False)
        cols2d = {feat: _table_2d(table, feat) for feat in edit_features}
        file_changed = False

        for e in np.unique(ep_idx):
            row_sel = np.where(ep_idx == e)[0]
            # 프레임 순서 보장
            order = row_sel[np.argsort(frame_idx[row_sel])]
            L = len(order)
            n = min(num_frames, max(L - 1, 0))
            episode_arrays.setdefault(int(e), {})
            for feat in edit_features:
                orig = cols2d[feat][order]
                new = _interpolate_start(orig, parking_vecs[feat], num_frames)
                if not np.array_equal(orig, new):
                    file_changed = True
                # 원래 행 위치에 되써넣기
                cols2d[feat][order] = new
                episode_arrays[int(e)][feat] = new
            total_frames_edited += n

        if file_changed and not dry_run:
            for feat in edit_features:
                table = _set_2d_column(table, feat, cols2d[feat])
            pq.write_table(table, f)
        logger.info("data 파일 처리: %s (episodes=%d)", f.name, len(np.unique(ep_idx)))

    summary = {
        "episodes": len(episode_arrays),
        "frames_edited": total_frames_edited,
        "files": len(data_files),
        "num_frames": num_frames,
        "features": edit_features,
        "dry_run": dry_run,
    }

    if recompute_stats and not dry_run:
        _recompute_stats(root, info, edit_features, episode_arrays)
        summary["stats_recomputed"] = True
    else:
        summary["stats_recomputed"] = False

    return summary


def _recompute_stats(
    root: pathlib.Path,
    info: dict,
    edit_features: list[str],
    episode_arrays: dict[int, dict[str, np.ndarray]],
) -> None:
    """편집한 피처의 per-episode / 전역 통계를 lerobot 유틸로 재계산해 반영."""
    from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats

    features = info["features"]
    feat_subset = {f: features[f] for f in edit_features}

    # per-episode 통계 (편집 피처만)
    ep_stats: dict[int, dict] = {}
    for e, arrs in episode_arrays.items():
        ep_stats[e] = compute_episode_stats(
            {f: arrs[f] for f in edit_features}, feat_subset
        )

    # ---- meta/episodes/*.parquet 갱신 (편집 피처 stats 컬럼만) ----
    ep_files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    stat_keys = ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"]
    for ef in ep_files:
        et = pq.read_table(ef)
        row_eps = et.column("episode_index").to_numpy(zero_copy_only=False)
        for feat in edit_features:
            for sk in stat_keys:
                col = f"stats/{feat}/{sk}"
                if col not in et.schema.names:
                    continue
                field = et.schema.field(col)
                new_vals = []
                for e in row_eps:
                    v = np.asarray(ep_stats[int(e)][feat][sk]).reshape(-1)
                    new_vals.append(v.tolist())
                new_col = pa.array(new_vals, type=field.type)
                idx = et.schema.get_field_index(col)
                et = et.set_column(idx, field, new_col)
        pq.write_table(et, ef)

    # ---- meta/stats.json 갱신 (편집 피처 전역 통계만) ----
    stats_path = root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    agg = aggregate_stats(list(ep_stats.values()))
    for feat in edit_features:
        stats[feat] = {
            k: np.asarray(v).reshape(-1).tolist() for k, v in agg[feat].items()
        }
    stats_path.write_text(json.dumps(stats, indent=4))
    logger.info("통계 재계산 완료: %s", ", ".join(edit_features))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="LeRobotDataset 각 에피소드 초반 프레임을 parking 시작으로 보정"
    )
    parser.add_argument("dataset_root", help="LeRobotDataset root (meta/info.json 있는 폴더)")
    parser.add_argument(
        "--num-frames", type=int, default=100,
        help="parking에서 보간해 덮어쓸 초반 프레임 수 N (기본 100)",
    )
    parser.add_argument(
        "--no-stats", action="store_true",
        help="meta 통계(stats.json/episodes) 재계산 생략",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="파일을 쓰지 않고 편집 대상만 계산해 요약 출력",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    summary = smooth_start(
        args.dataset_root,
        num_frames=args.num_frames,
        recompute_stats=not args.no_stats,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
