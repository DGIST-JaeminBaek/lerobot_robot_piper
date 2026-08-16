#!/usr/bin/env python3
"""smoothing 파라미터를 바꿔가며 실제 정책의 궤적 부드러움을 비교한다.

GUI(piper_infer_gui.py)와 완전히 같은 InferenceWorker를 쓰되 RViz/로봇 없이
dataset observation만으로 돌리므로, 하드웨어 없이 "m을 얼마로 둘지"를 먼저
정할 수 있다. 실행 결과는 outputs/smoothing_sweep/에 저장된다.

    python scripts/tools/piper_smoothing_sweep.py \\
        --policy-path outputs/train/smolvla_erase_shape_512/checkpoints/030000/pretrained_model \\
        --dataset-root records/0727/erase_the_shape_512 \\
        --steps 60

지표는 모두 작을수록 부드럽다:
    TV        스텝당 |Δaction| 합의 평균 (정규화 단위/step)
    max_step  한 스텝에서 일어난 최대 변화 — 튀는 지점
    rms_jerk  3차 차분 RMS (정규화 단위/s^3)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import queue
import subprocess
import sys
import time
from dataclasses import asdict

import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from action_smoothing import SmoothingConfig, smoothness_metrics  # noqa: E402
from piper_infer_gui import Event, InferenceWorker, load_env_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="smoothing 파라미터 스윕 (하드웨어 불필요)")
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--dataset-root", type=pathlib.Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--task", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=60, help="설정당 실행할 스텝 수")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument(
        "--m-values",
        type=float,
        nargs="*",
        default=[0.01, 0.1, 0.3, 1.0],
        help="비교할 temporal-ensemble decay 값들",
    )
    parser.add_argument("--ema-alpha", type=float, default=1.0)
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=None,
        help="지정하지 않으면 rate limit 없이 순수 smoothing 효과만 본다",
    )
    parser.add_argument("--output-dir", type=pathlib.Path, default=None)
    # 내부용: 부모가 설정 하나를 자식 프로세스에서 돌릴 때 쓴다.
    parser.add_argument("--single", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-out", type=pathlib.Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def run_one(base: dict, smoothing: SmoothingConfig, label: str) -> tuple[np.ndarray, list[str]]:
    """한 설정을 이 프로세스에서 실행한다 (--single 자식 프로세스에서 호출됨).

    같은 프로세스에서 SmolVLA를 두 번 로드하면 CUDA 컨텍스트가 깨지면서
    segfault가 나므로, 부모는 설정마다 자식 프로세스를 새로 띄운다.
    """
    settings = dict(base)
    settings["smoothing"] = smoothing
    events: queue.Queue = queue.Queue()
    worker = InferenceWorker(settings, events)

    print(f"\n=== {label}: {smoothing.summary()} ===", flush=True)
    worker.start()
    logs: list[str] = []
    status = None
    while status is None:
        try:
            kind, payload = events.get(timeout=600)
        except queue.Empty:
            worker.emergency_stop()
            raise RuntimeError("worker timed out")
        if kind == Event.LOG:
            logs.append(str(payload))
            print(f"  {payload}", flush=True)
        elif kind == Event.FINISHED:
            status = str(payload)
    worker.join(timeout=30)
    if status == "error":
        raise RuntimeError(f"{label} 실행이 오류로 끝났습니다 — 위 로그 참고")
    if not worker.trajectory:
        raise RuntimeError(f"{label}에서 action이 하나도 생성되지 않았습니다")
    return np.stack(worker.trajectory), logs


def run_in_subprocess(
    base: dict,
    smoothing: SmoothingConfig,
    label: str,
    output_dir: pathlib.Path,
    index: int,
) -> np.ndarray:
    """설정 하나를 새 프로세스에서 실행하고 궤적을 읽어온다.

    한 프로세스에서 SmolVLA를 반복 로드하면 죽기 때문에 매번 새로 띄운다.
    부작용으로 설정 간 정책 상태가 완전히 격리되어 비교도 더 깨끗해진다.
    """
    payload = output_dir / f"_single_{index:02d}.npy"
    spec = json.dumps({"base": {k: v for k, v in base.items()}, "smoothing": asdict(smoothing), "label": label})
    command = [
        sys.executable,
        str(pathlib.Path(__file__).resolve()),
        "--policy-path", base["policy_path"],
        "--dataset-root", base["dataset_root"],
        "--single", spec,
        "--single-out", str(payload),
    ]
    completed = subprocess.run(command, cwd=str(REPO_ROOT))
    if completed.returncode != 0 or not payload.exists():
        raise RuntimeError(f"{label} 실행 실패 (exit={completed.returncode})")
    trajectory = np.load(payload)
    payload.unlink(missing_ok=True)
    return trajectory


def run_single(spec_json: str, out_path: pathlib.Path) -> int:
    spec = json.loads(spec_json)
    smoothing = SmoothingConfig(**spec["smoothing"])
    trajectory, _ = run_one(spec["base"], smoothing, spec["label"])
    np.save(out_path, trajectory)
    return 0


def main() -> int:
    args = parse_args()
    load_env_file(REPO_ROOT / "configs" / "recording.env")
    if args.single is not None:
        return run_single(args.single, args.single_out)

    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.exists():
        print(f"[ERROR] dataset 경로 없음: {dataset_root}", file=sys.stderr)
        return 1
    policy_path = pathlib.Path(args.policy_path).expanduser()
    if not policy_path.exists():
        print(f"[ERROR] checkpoint 경로 없음: {policy_path}", file=sys.stderr)
        return 1

    base = {
        "policy_path": str(policy_path),
        "dataset_root": str(dataset_root),
        "episode": args.episode,
        "task": args.task or "",
        "device": args.device,
        "source": "dataset",
        "fps": args.fps,
        "horizon": args.horizon,
        "infer_every": 1,
        "max_steps": args.steps,
        "loop_dataset": False,
        "rviz": False,
        "joint_state_topic": "/joint_states",
        "apply_to_robot": False,
        "park_on_exit": False,
        "camera_output_size": 512,
        "crops": {},
    }

    configurations: list[tuple[str, SmoothingConfig]] = [
        (
            "baseline (no smoothing)",
            SmoothingConfig(
                temporal_ensemble=False, ema_alpha=1.0, rate_limit=None, clip_to_range=True
            ),
        )
    ]
    for m in args.m_values:
        configurations.append(
            (
                f"ensemble m={m:g}",
                SmoothingConfig(
                    temporal_ensemble=True,
                    ensemble_m=m,
                    ema_alpha=args.ema_alpha,
                    rate_limit=args.rate_limit,
                    clip_to_range=True,
                ),
            )
        )

    output_dir = args.output_dir or (REPO_ROOT / "outputs" / "smoothing_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    results = []
    trajectories = {}
    for index, (label, smoothing) in enumerate(configurations):
        trajectory = run_in_subprocess(base, smoothing, label, output_dir, index)
        metrics = smoothness_metrics(trajectory, fps=args.fps)
        results.append({"label": label, "steps": len(trajectory), **metrics})
        trajectories[label] = trajectory

    baseline = results[0]
    print("\n" + "=" * 78)
    print(f"{'설정':<26}{'steps':>6}{'TV':>10}{'max_step':>10}{'rms_jerk':>13}{'TV 감소':>10}")
    print("-" * 78)
    for row in results:
        reduction = (
            f"{(1 - row['total_variation'] / baseline['total_variation']) * 100:8.1f}%"
            if baseline["total_variation"] > 0
            else "     n/a"
        )
        print(
            f"{row['label']:<26}{row['steps']:>6}{row['total_variation']:>10.4f}"
            f"{row['max_step']:>10.3f}{row['rms_jerk']:>13.1f}{reduction:>10}"
        )
    print("=" * 78)

    npz_path = output_dir / f"sweep_{stamp}.npz"
    np.savez_compressed(
        npz_path, **{label.replace(" ", "_"): trajectory for label, trajectory in trajectories.items()}
    )
    (output_dir / f"sweep_{stamp}.json").write_text(
        json.dumps(
            {
                "policy_path": str(policy_path),
                "dataset_root": str(dataset_root),
                "episode": args.episode,
                "fps": args.fps,
                "steps": args.steps,
                "ema_alpha": args.ema_alpha,
                "rate_limit": args.rate_limit,
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[SAVE] {npz_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
