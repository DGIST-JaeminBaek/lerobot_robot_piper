#!/usr/bin/env python3
"""정책이 **조건부 정보(도형이 보드 어디에 있나)를 실제로 쓰는지** 측정한다.

배경: 실물에서 pi0는 지우개는 집지만 도형을 못 지웠다(2026-09-01 사용자 정성 관찰).
315개 관절 시계열 분석에서 이 task는 두 국면으로 갈린다 —
진행률 0~40%(집기)는 에피소드마다 거의 같은 궤적이고, 40~90%(지우기)에서만
도형 위치에 따라 joint1이 ±25도까지 갈린다. 즉 "집기는 되는데 지우기는 안 된다"는
**공통 궤적(prior)은 외웠지만 조건부 정보는 못 쓴다**는 가설과 정확히 맞는다.

측정 방법: 각 에피소드의 정해진 진행률 지점에서 observation을 넣고 action chunk를
받아, 같은 지점의 정답 action과 비교한다. teacher-forced라 오차가 누적되지 않으므로
"그 순간 무엇을 하려 했는가"만 본다. 로봇·CAN은 쓰지 않는다.

핵심 지표는 MAE가 아니라 **분산비**다:

    spread_ratio = std_over_episodes(예측) / std_over_episodes(정답)

같은 진행률에서 에피소드끼리 얼마나 다른 값을 내놓는가의 비율이다. 정답은 도형
위치가 달라서 벌어지는데 예측이 안 벌어지면(비율 << 1) 그 정책은 입력을 무시하고
평균 궤적을 재생하고 있다는 뜻이다. MAE만 보면 "평균을 잘 맞춰서" 낮게 나올 수 있어
이 구분이 안 된다.

두 정책은 conda 환경이 달라(pi0 / ugrp) 한 프로세스에서 못 돌린다. 정책마다 따로
실행해 npz로 남기고, 비교·그림은 compare_policy_conditioning.py가 한다.

사용:
    python scripts/tasks/erase_shape/analysis/policy_conditioning_probe.py \\
        --dataset-root records/outputs/pick_up_the_eraser_and_erase_the_shape/pick_up_the_eraser_315 \\
        --policy-path outputs/train/.../checkpoints/last/pretrained_model \\
        --label pi0_315 --episodes all
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
INFERENCE_DIR = REPO_ROOT / "scripts" / "piper" / "inference"
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
# 진행률 지점. 0.10~0.35는 모든 에피소드가 같은 집기 동작을 하는 구간,
# 0.55~0.90은 도형 위치에 따라 갈리는 지우기·복귀 구간이다(315 시계열 분석 근거).
DEFAULT_PROBES = (0.10, 0.25, 0.35, 0.55, 0.70, 0.85)
SHAPES = ("circle", "triangle", "rectangle")


def shape_of(task_or_name: str) -> str:
    lowered = task_or_name.lower()
    for shape in SHAPES:
        if shape in lowered:
            return shape
    return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=pathlib.Path, required=True)
    parser.add_argument("--policy-path", type=str, required=True)
    parser.add_argument("--label", required=True, help="산출물 이름 (예: pi0_315)")
    parser.add_argument("--policy-discover-packages-path", default=None,
                        help="HAMLET 등 외부 policy 패키지명 (예: smolvla_hamlet)")
    parser.add_argument("--episodes", default="all",
                        help="'all' 또는 개수(도형별 균등 표집). 기본 all")
    parser.add_argument("--probes", type=float, nargs="+", default=list(DEFAULT_PROBES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task", default=None, help="기본값: dataset episode의 task")
    parser.add_argument("--seed", type=int, default=1000,
                        help="추론 직전마다 이 seed를 복원한다 — 두 정책의 차이가 "
                             "sampling noise가 아니라 입력 반응 차이가 되게 한다")
    parser.add_argument("--out-dir", type=pathlib.Path,
                        default=REPO_ROOT / "outputs" / "analysis" / "erase_shape" / "policy_conditioning")
    return parser.parse_args()


def select_episodes(meta, wanted: str) -> list[int]:
    total = len(meta.episodes)
    indices = list(range(total))
    if wanted == "all":
        return indices
    count = int(wanted)
    tasks = meta.episodes["tasks"]
    by_shape: dict[str, list[int]] = {shape: [] for shape in SHAPES}
    for index in indices:
        entry = tasks[index]
        text = entry if isinstance(entry, str) else " ".join(entry)
        by_shape.setdefault(shape_of(text), []).append(index)
    # task 문자열에 도형이 없는 데이터셋(pickup 프롬프트)은 균등 표집을 못 하므로
    # 전체에서 고르게 뽑는다.
    if not any(by_shape.get(shape) for shape in SHAPES):
        return list(np.linspace(0, total - 1, min(count, total)).round().astype(int))
    picked: list[int] = []
    per_shape = max(1, count // len(SHAPES))
    for shape in SHAPES:
        pool = by_shape.get(shape, [])
        if pool:
            picked += [pool[i] for i in np.linspace(0, len(pool) - 1, min(per_shape, len(pool))).round().astype(int)]
    return sorted(set(picked))


def main() -> int:
    args = parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.utils import get_safe_torch_device
    from inference_runtime import (
        hamlet_real_history_enabled, load_policy, make_raw_observation, predict_chunk,
    )

    if args.policy_discover_packages_path:
        import importlib
        importlib.import_module(args.policy_discover_packages_path)

    root = args.dataset_root
    print(f"[LOAD] dataset={root}")
    probe_meta = LeRobotDataset(repo_id=f"local/{root.name}", root=root,
                               episodes=[0], video_backend="pyav")
    episodes = select_episodes(probe_meta.meta, args.episodes)
    del probe_meta
    print(f"[LOAD] episodes={len(episodes)}")
    dataset = LeRobotDataset(repo_id=f"local/{root.name}", root=root,
                             episodes=episodes, video_backend="pyav")
    starts = np.asarray(dataset.meta.episodes["dataset_from_index"])[episodes]
    ends = np.asarray(dataset.meta.episodes["dataset_to_index"])[episodes]
    # episodes= 로 부분 로드하면 dataset 인덱스가 0부터 다시 붙는다.
    offsets = np.concatenate([[0], np.cumsum(ends - starts)[:-1]])

    print(f"[LOAD] policy={args.policy_path}")
    config, policy, preprocessor, postprocessor = load_policy(
        args.policy_path, dataset.meta, args.device
    )
    device = get_safe_torch_device(policy.config.device)
    policy.reset()
    chunk_size = int(getattr(config, "chunk_size", 50))
    # 일반 SmolVLA/pi0도 observation_delta_indices가 비어있지 않을 수 있으므로
    # (현재 프레임 [0]만 참조), 실제로 과거 프레임이 필요한 정책만 거른다.
    if hamlet_real_history_enabled(policy):
        raise SystemExit("이 프로브는 이미지 history를 쓰는 정책(HAMLET)을 아직 지원하지 않는다")

    probes = np.asarray(args.probes, dtype=np.float64)
    n_ep, n_probe = len(episodes), len(probes)
    pred = np.full((n_ep, n_probe, chunk_size, 7), np.nan, dtype=np.float32)
    truth = np.full((n_ep, n_probe, chunk_size, 7), np.nan, dtype=np.float32)
    lengths = (ends - starts).astype(np.int32)
    shapes, latencies = [], []

    started_all = time.perf_counter()
    for row, episode in enumerate(episodes):
        length = int(lengths[row])
        base = int(offsets[row])
        item0 = dataset[base]
        task = args.task if args.task is not None else str(item0.get("task", ""))
        shapes.append(shape_of(task) if shape_of(task) != "unknown" else "unknown")
        for column, fraction in enumerate(probes):
            frame = int(round(fraction * (length - 1)))
            available = min(chunk_size, length - frame)
            if available <= 0:
                continue
            raw_observation = make_raw_observation(dataset, dataset[base + frame])
            # 두 정책이 같은 noise를 받도록 매 추론 직전에 seed를 복원한다.
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
            tick = time.perf_counter()
            chunk = predict_chunk(
                raw_observation=raw_observation, dataset_features=dataset.features,
                policy=policy, preprocessor=preprocessor, postprocessor=postprocessor,
                device=device, task=task,
            )
            latencies.append(time.perf_counter() - tick)
            steps = min(available, len(chunk))
            pred[row, column, :steps] = chunk[:steps].numpy()
            truth[row, column, :steps] = np.stack([
                dataset[base + frame + k]["action"].detach().cpu().numpy() for k in range(steps)
            ])
        if (row + 1) % 20 == 0 or row + 1 == n_ep:
            elapsed = time.perf_counter() - started_all
            print(f"  [{row + 1}/{n_ep}] {elapsed:.0f}s  추론 중앙값 "
                  f"{np.median(latencies) * 1000:.0f}ms", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"probe_{args.label}.npz"
    np.savez_compressed(
        out_path, pred=pred, truth=truth, probes=probes,
        episodes=np.asarray(episodes), lengths=lengths,
        shapes=np.asarray(shapes), joint_names=np.asarray(JOINT_NAMES),
        latency_s=np.asarray(latencies), label=args.label,
        policy_path=str(args.policy_path), chunk_size=chunk_size,
    )
    print(f"[SAVE] {out_path}  (추론 {len(latencies)}회, 중앙값 {np.median(latencies) * 1000:.0f}ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
