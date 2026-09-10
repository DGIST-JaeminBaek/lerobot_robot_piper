#!/usr/bin/env python3
"""단일 task 데이터셋을 **도형별 프롬프트** 다중 task 데이터셋으로 바꾼다.

목적: 지금 학습셋은 도형이 뭐든 프롬프트가 하나다("... erase the shape"). 이걸
"... erase the circle / triangle / rectangle"로 쪼개서, 프롬프트만으로 다중 도형
보드에서 선택적으로 지우는 게 되는지(zero-shot) 보려는 실험용이다.

**이 변환이 무엇을 바꾸고 무엇을 안 바꾸는가**
  바꾸는 것 : task 문자열과 그 인덱스뿐 — meta/tasks.parquet, data/*.parquet의
              task_index, meta/episodes/*.parquet의 tasks·task_index 통계,
              meta/info.json의 total_tasks, meta/stats.json의 task_index 통계.
  안 바꾸는 것: 영상, 관측/행동 값, 에피소드 순서·길이. 영상은 하드링크로 걸어서
              용량을 새로 먹지 않는다(같은 파일시스템일 때. 아니면 복사한다).

**도형은 어떻게 알아내는가** — 추론하지 않는다. 데이터셋을 만든 매니페스트의
`source_dataset` 경로(예: `records/0812/erase_the_circle_0812-131127`)에서 읽는다.
prepare_erase_shape_dataset.py가 매니페스트 순서대로 에피소드를 넣고 프레임 수도
검증하므로, 실행 전에 **에피소드 길이 수열이 매니페스트와 일치하는지 먼저 확인**하고
어긋나면 아무것도 쓰지 않고 중단한다. 라벨이 한 칸이라도 밀리면 학습이 통째로
오염되기 때문에 이 검증은 건너뛸 수 없다.

사용:
    python scripts/tasks/erase_shape/dataset/make_shape_prompt_dataset.py \\
        --src records/outputs/pick_up_the_eraser_and_erase_the_shape/pick_up_the_eraser_315 \\
        --manifest configs/erase_shape_315_manifest.json \\
        --dst records/outputs/pick_up_the_eraser_and_erase_the_shape/pick_up_the_eraser_315_shape_prompt \\
        --template "pick up the eraser and erase the {shape}"

    # 실제로 쓰기 전에 무엇이 바뀌는지만 보려면 --dry-run
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
SHAPES = ("circle", "triangle", "rectangle")


def episode_shapes(manifest_path: Path) -> list[str]:
    """매니페스트 순서대로 각 에피소드의 도형 이름."""
    entries = [e for e in json.loads(manifest_path.read_text())["episodes"]
               if e.get("enabled", True)]
    shapes = []
    for e in entries:
        m = re.search(r"erase_the_(\w+?)_\d{4}-", e["source_dataset"])
        if not m or m.group(1) not in SHAPES:
            raise ValueError(f"도형을 못 읽었다: {e['source_dataset']}")
        shapes.append(m.group(1))
    return shapes


def manifest_lengths(manifest_path: Path) -> list[int]:
    entries = [e for e in json.loads(manifest_path.read_text())["episodes"]
               if e.get("enabled", True)]
    return [e["end_frame"] - e["start_frame"] for e in entries]


def load_episode_meta(ds: Path) -> tuple[pd.DataFrame, list[Path]]:
    files = sorted(Path(p) for p in
                   glob.glob(str(ds / "meta/episodes/**/*.parquet"), recursive=True))
    return pd.concat([pd.read_parquet(f) for f in files]).sort_values("episode_index"), files


def verify_mapping(ds: Path, manifest_path: Path) -> list[str]:
    """에피소드 길이로 매핑을 검증하고, 통과하면 에피소드별 도형을 돌려준다."""
    ep, _ = load_episode_meta(ds)
    got = list(ep["length"])
    want = manifest_lengths(manifest_path)
    if len(got) != len(want):
        raise SystemExit(f"에피소드 수 불일치: 데이터셋 {len(got)} vs 매니페스트 {len(want)}")
    bad = [(i, w, g) for i, (w, g) in enumerate(zip(want, got)) if w != g]
    if bad:
        for i, w, g in bad[:5]:
            print(f"  ep{i}: 매니페스트 {w} vs 데이터셋 {g}", file=sys.stderr)
        raise SystemExit(f"길이 불일치 {len(bad)}개 — 매핑을 믿을 수 없어 중단한다")
    return episode_shapes(manifest_path)


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)          # 같은 파일시스템이면 용량을 안 먹는다
    except OSError:
        shutil.copy2(src, dst)


STAT_KEYS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")
# meta/episodes parquet의 stats 스키마는 키마다 타입이 다르다
# (min/max/count = list<int64>, 나머지 = list<double>). 안 지키면 parquet 쓰기가
# "Can only convert 1-dimensional array values of list<item: double>"로 실패한다.
INT_STAT_KEYS = ("min", "max", "count")


def const_task_stats(task_idx: int, n_frames: int) -> dict[str, np.ndarray]:
    """task_index는 한 에피소드 안에서 상수라 통계가 자명하다."""
    k = int(task_idx)
    out = {
        "min": np.array([k], dtype=np.int64),
        "max": np.array([k], dtype=np.int64),
        "count": np.array([int(n_frames)], dtype=np.int64),
        "mean": np.array([float(k)], dtype=np.float64),
        "std": np.array([0.0], dtype=np.float64),
    }
    for q in ("q01", "q10", "q50", "q90", "q99"):
        out[q] = np.array([float(k)], dtype=np.float64)
    return out


def scalar_stats_json(values: np.ndarray) -> dict:
    """meta/stats.json용 전역 통계(각 값이 리스트)."""
    v = values.astype(np.float64)
    return {
        "min": [int(v.min())], "max": [int(v.max())], "count": [int(v.size)],
        "mean": [float(v.mean())], "std": [float(v.std())],
        "q01": [float(np.quantile(v, 0.01))], "q10": [float(np.quantile(v, 0.10))],
        "q50": [float(np.quantile(v, 0.50))], "q90": [float(np.quantile(v, 0.90))],
        "q99": [float(np.quantile(v, 0.99))],
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--template", required=True,
                   help="'{shape}'가 circle/triangle/rectangle로 치환된다")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    src, dst = args.src.resolve(), args.dst.resolve()
    if "{shape}" not in args.template:
        raise SystemExit("--template 에 {shape} 가 있어야 한다")
    if dst.exists():
        raise SystemExit(f"이미 있다: {dst} (덮어쓰지 않는다)")

    shapes = verify_mapping(src, args.manifest)
    tasks = [args.template.format(shape=s) for s in SHAPES]
    task_index = {t: i for i, t in enumerate(tasks)}
    ep_task = [args.template.format(shape=s) for s in shapes]
    ep_task_idx = np.array([task_index[t] for t in ep_task], dtype=np.int64)

    print(f"원본 : {src.relative_to(REPO_ROOT)}")
    print(f"대상 : {dst.relative_to(REPO_ROOT)}")
    print(f"에피소드 {len(shapes)}개, 매핑 검증 통과")
    print("새 task:")
    for t, i in task_index.items():
        print(f"  [{i}] {t}   ({shapes.count(SHAPES[i])}개)")
    if args.dry_run:
        print("\n(dry-run — 아무것도 쓰지 않았다)")
        return 0

    # ── 1. 영상: 하드링크 ──────────────────────────────────────────
    n_link = 0
    for f in sorted(src.glob("videos/**/*")):
        if f.is_file():
            link_or_copy(f, dst / f.relative_to(src))
            n_link += 1
    print(f"\n영상 {n_link}개 링크 완료")

    # ── 2. data/*.parquet: task_index 갱신 ────────────────────────
    all_task_idx = []
    for f in sorted(Path(x) for x in glob.glob(str(src / "data/**/*.parquet"), recursive=True)):
        d = pd.read_parquet(f)
        d["task_index"] = ep_task_idx[d["episode_index"].to_numpy()]
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        d.to_parquet(out, index=False)
        all_task_idx.append(d["task_index"].to_numpy())
    all_task_idx = np.concatenate(all_task_idx)
    print(f"data parquet {len(all_task_idx)}행 task_index 갱신")

    # ── 3. meta/tasks.parquet ────────────────────────────────────
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"task_index": list(task_index.values())},
                 index=pd.Index(list(task_index.keys()))).to_parquet(dst / "meta/tasks.parquet")

    # ── 4. meta/episodes: tasks 목록과 task_index 통계 ────────────
    _, ep_files = load_episode_meta(src)
    for f in ep_files:
        e = pd.read_parquet(f)
        idx = e["episode_index"].to_numpy()
        e["tasks"] = [np.array([ep_task[i]], dtype=object) for i in idx]
        # 컬럼 단위로 한 번에 넣는다 — .at으로 셀마다 넣으면 컬럼이 object로 바뀌어
        # parquet 스키마(list<int64>/list<double>)가 깨진다.
        per_key: dict[str, list] = {k: [] for k in STAT_KEYS}
        for i, ei in enumerate(idx):
            st = const_task_stats(int(ep_task_idx[ei]), int(e.iloc[i]["length"]))
            for key in STAT_KEYS:
                per_key[key].append(st[key])
        for key in STAT_KEYS:
            e[f"stats/task_index/{key}"] = pd.Series(per_key[key], index=e.index)
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        e.to_parquet(out, index=False)
    print(f"meta/episodes {len(ep_files)}개 파일 갱신")

    # ── 5. meta/info.json, stats.json, 나머지 메타 파일 ───────────
    for f in sorted(src.glob("meta/*")):
        if f.is_file() and f.name not in ("tasks.parquet", "info.json", "stats.json"):
            link_or_copy(f, dst / f.relative_to(src))
    info = json.loads((src / "meta/info.json").read_text())
    info["total_tasks"] = len(tasks)
    (dst / "meta/info.json").write_text(json.dumps(info, indent=4) + "\n")

    stats_path = src / "meta/stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        stats["task_index"] = scalar_stats_json(all_task_idx)
        (dst / "meta/stats.json").write_text(json.dumps(stats, indent=4) + "\n")
    print("meta/info.json(total_tasks), stats.json(task_index) 갱신")
    print(f"\n완료: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
