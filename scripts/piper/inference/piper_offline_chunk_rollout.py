#!/usr/bin/env python3
"""학습 데이터 observation을 주기적으로 다시 공급하는 SmolVLA offline chunk rollout.

실제 로봇에는 연결하지 않는다. Episode의 첫 frame에서 action chunk를 예측하고,
설정한 action 개수만큼 예측 궤적에 붙인 뒤 데이터셋 frame도 같은 수만큼 앞으로
이동해 다시 추론한다.

이 방식은 예측 action의 결과로 다음 영상이 생성되는 물리 시뮬레이션이 아니다.
다음 observation은 녹화된 전문가 trajectory에서 가져오는 teacher-forced
observation이다. 정책의 chunk 궤적, 정답 action과의 차이 및 chunk 경계 불연속을
하드웨어 없이 확인하는 용도로만 사용한다.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass

import numpy as np
import torch


JOINT_NAMES = [
    "joint1.pos",
    "joint2.pos",
    "joint3.pos",
    "joint4.pos",
    "joint5.pos",
    "joint6.pos",
    "gripper.pos",
]


@dataclass
class ChunkResult:
    chunk_index: int
    observation_frame: int
    predicted_actions: int
    executed_actions: int
    inference_seconds: float
    mae: float
    rmse: float
    max_abs_error: float
    boundary_jump: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SmolVLA action chunk를 학습 데이터 observation으로 offline rollout"
    )
    parser.add_argument("--dataset-root", type=pathlib.Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--policy-path", type=str, required=True)
    parser.add_argument(
        "--policy-discover-packages-path",
        default=None,
        help="HAMLET 등 외부 policy 패키지명 (예: smolvla_hamlet)",
    )
    parser.add_argument("--task", default=None, help="기본값: dataset episode의 task")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--actions-per-step",
        type=int,
        default=50,
        help="각 observation에서 예측 chunk 중 사용할 action 수이자 다음 dataset frame 이동량",
    )
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="기본값: outputs/offline_rollout/<dataset>_epXXXX_<checkpoint>",
    )
    parser.add_argument(
        "--rviz",
        action="store_true",
        help="결과 저장 후 예측 action 전체를 /joint_states에 재생",
    )
    parser.add_argument("--rviz-rate", type=float, default=None, help="기본값: dataset FPS")
    parser.add_argument("--joint-state-topic", default="/joint_states")
    return parser.parse_args()


def default_output_dir(dataset_root: pathlib.Path, episode: int, policy_path: str) -> pathlib.Path:
    policy_dir = pathlib.Path(policy_path)
    checkpoint = policy_dir.parent.name if policy_dir.name == "pretrained_model" else policy_dir.name
    if checkpoint in {"", ".", "/"}:
        checkpoint = "policy"
    return (
        pathlib.Path("outputs")
        / "offline_rollout"
        / f"{dataset_root.name}_ep{episode:04d}_{checkpoint}"
    )


from inference_runtime import (  # noqa: E402
    hamlet_real_history_enabled,
    load_policy,
    make_raw_observation,
    predict_chunk,
    validate_hamlet_real_history_policy,
)
from hamlet_history import TimestampedImageRingBuffer, history_retention_s  # noqa: E402


def compute_errors(predicted: np.ndarray, expert: np.ndarray) -> tuple[float, float, float]:
    error = predicted - expert
    return (
        float(np.mean(np.abs(error))),
        float(np.sqrt(np.mean(np.square(error)))),
        float(np.max(np.abs(error))),
    )


def save_plot(
    output_path: pathlib.Path,
    predicted: np.ndarray,
    expert: np.ndarray,
    decision_frames: list[int],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(7, 1, figsize=(15, 17), sharex=True)
    frames = np.arange(len(predicted))
    for joint_index, axis in enumerate(axes):
        axis.plot(frames, expert[:, joint_index], color="#505050", linewidth=1.2, label="dataset")
        axis.plot(
            frames,
            predicted[:, joint_index],
            color="#e45756",
            linewidth=1.0,
            label="predicted rollout",
        )
        for decision_frame in decision_frames[1:]:
            axis.axvline(decision_frame, color="#4c78a8", alpha=0.25, linewidth=0.8)
        axis.set_ylabel(JOINT_NAMES[joint_index])
        axis.grid(alpha=0.2)

    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("episode frame")
    fig.suptitle("Teacher-forced offline action-chunk rollout")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_rollout(args: argparse.Namespace) -> tuple[pathlib.Path, np.ndarray, float]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.utils import get_safe_torch_device

    if args.actions_per_step <= 0:
        raise ValueError("--actions-per-step must be positive")
    if not args.dataset_root.exists():
        raise FileNotFoundError(args.dataset_root)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"[LOAD] dataset={args.dataset_root}, episode={args.episode}")
    dataset = LeRobotDataset(
        repo_id=f"local/{args.dataset_root.name}",
        root=args.dataset_root,
        episodes=[args.episode],
        video_backend="pyav",
    )
    if dataset.num_frames == 0:
        raise RuntimeError(f"episode {args.episode} has no frames")

    output_dir = args.output_dir or default_output_dir(
        args.dataset_root, args.episode, args.policy_path
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[LOAD] policy={args.policy_path}, device={args.device}")
    if args.policy_discover_packages_path:
        import importlib

        importlib.import_module(args.policy_discover_packages_path)
    config, policy, preprocessor, postprocessor = load_policy(
        args.policy_path, dataset.meta, args.device
    )
    device = get_safe_torch_device(policy.config.device)
    policy.reset()
    hamlet_history = None
    hamlet_delta_indices: list[int] | None = None
    history_next_frame = 0
    if hamlet_real_history_enabled(policy):
        validate_hamlet_real_history_policy(policy)
        hamlet_delta_indices = list(policy.config.observation_delta_indices)
        hamlet_history = TimestampedImageRingBuffer(
            max_age_s=history_retention_s(hamlet_delta_indices, dataset.fps)
        )
        print(f"[HAMLET] 실제 dataset image history: delta={hamlet_delta_indices}, fps={dataset.fps}")

    configured_chunk_size = int(getattr(config, "chunk_size", 0))
    if args.actions_per_step > configured_chunk_size:
        raise ValueError(
            f"--actions-per-step={args.actions_per_step} exceeds policy chunk_size="
            f"{configured_chunk_size}"
        )

    first_item = dataset[0]
    task = args.task if args.task is not None else str(first_item.get("task", ""))
    print(
        f"[RUN] frames={dataset.num_frames}, fps={dataset.fps}, task={task!r}, "
        f"chunk_size={configured_chunk_size}, actions_per_step={args.actions_per_step}"
    )

    predicted_parts: list[np.ndarray] = []
    expert_parts: list[np.ndarray] = []
    source_frame_parts: list[np.ndarray] = []
    decision_frames: list[int] = []
    chunks: list[ChunkResult] = []
    cursor = 0

    while cursor < dataset.num_frames:
        item = dataset[cursor]
        raw_observation = make_raw_observation(dataset, item)
        if hamlet_history is not None:
            # Decision frame 사이를 건너뛰더라도 실제 dataset의 모든 raw frame을
            # buffer에 넣는다. 그래야 actions_per_step가 35여도 stride=1 history가
            # '정책 호출 4장'이 아닌 학습 때의 30fps frame 간격이 된다.
            while history_next_frame <= cursor:
                history_raw = (
                    raw_observation
                    if history_next_frame == cursor
                    else make_raw_observation(dataset, dataset[history_next_frame])
                )
                hamlet_history.append(
                    history_next_frame / float(dataset.fps),
                    {
                        key.removeprefix("observation.images."): history_raw[
                            key.removeprefix("observation.images.")
                        ]
                        for key in dataset.meta.camera_keys
                    },
                )
                history_next_frame += 1
        history_snapshot = (
            hamlet_history.snapshot(
                current_timestamp_s=cursor / float(dataset.fps),
                delta_indices=hamlet_delta_indices,
                fps=dataset.fps,
            )
            if hamlet_history is not None
            else None
        )
        started = time.perf_counter()
        chunk = predict_chunk(
            raw_observation=raw_observation,
            dataset_features=dataset.features,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=device,
            task=task,
            rollout_history=history_snapshot,
        )
        inference_seconds = time.perf_counter() - started

        remaining = dataset.num_frames - cursor
        executed = min(args.actions_per_step, remaining, len(chunk))
        predicted = chunk[:executed].numpy().astype(np.float32, copy=False)
        expert = np.stack(
            [dataset[index]["action"].detach().cpu().numpy() for index in range(cursor, cursor + executed)]
        ).astype(np.float32, copy=False)

        mae, rmse, max_abs_error = compute_errors(predicted, expert)
        boundary_jump = None
        if predicted_parts:
            boundary_jump = float(np.max(np.abs(predicted[0] - predicted_parts[-1][-1])))

        chunk_result = ChunkResult(
            chunk_index=len(chunks),
            observation_frame=cursor,
            predicted_actions=len(chunk),
            executed_actions=executed,
            inference_seconds=inference_seconds,
            mae=mae,
            rmse=rmse,
            max_abs_error=max_abs_error,
            boundary_jump=boundary_jump,
        )
        chunks.append(chunk_result)
        decision_frames.append(cursor)
        predicted_parts.append(predicted)
        expert_parts.append(expert)
        source_frame_parts.append(np.full(executed, cursor, dtype=np.int64))
        print(
            f"[CHUNK {chunk_result.chunk_index:02d}] obs_frame={cursor:04d}, "
            f"execute={executed:02d}, infer={inference_seconds:.3f}s, "
            f"MAE={mae:.4f}, max={max_abs_error:.4f}"
        )
        cursor += executed

    predicted_rollout = np.concatenate(predicted_parts)
    expert_actions = np.concatenate(expert_parts)
    source_frames = np.concatenate(source_frame_parts)
    overall_mae, overall_rmse, overall_max = compute_errors(predicted_rollout, expert_actions)
    per_joint_mae = np.mean(np.abs(predicted_rollout - expert_actions), axis=0)

    limits_low = np.array([-100.0] * 6 + [0.0], dtype=np.float32)
    limits_high = np.array([100.0] * 7, dtype=np.float32)
    range_violation_count = int(
        np.count_nonzero((predicted_rollout < limits_low) | (predicted_rollout > limits_high))
    )

    np.savez_compressed(
        output_dir / "rollout_actions.npz",
        predicted_actions=predicted_rollout,
        expert_actions=expert_actions,
        observation_source_frames=source_frames,
        decision_frames=np.asarray(decision_frames, dtype=np.int64),
    )
    summary = {
        "mode": "teacher_forced_offline_chunk_rollout",
        "dataset_root": str(args.dataset_root.resolve()),
        "episode": args.episode,
        "policy_path": str(pathlib.Path(args.policy_path).resolve()),
        "task": task,
        "fps": dataset.fps,
        "episode_frames": dataset.num_frames,
        "policy_chunk_size": configured_chunk_size,
        "actions_per_step": args.actions_per_step,
        "decision_frames": decision_frames,
        "overall": {
            "mae": overall_mae,
            "rmse": overall_rmse,
            "max_abs_error": overall_max,
            "range_violation_count": range_violation_count,
            "per_joint_mae": dict(zip(JOINT_NAMES, per_joint_mae.tolist())),
        },
        "chunks": [asdict(chunk) for chunk in chunks],
        "limitations": [
            "The next observation comes from the recorded expert trajectory.",
            "RViz trajectory playback does not simulate contacts, objects, or erasing.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    save_plot(
        output_dir / "trajectory_comparison.png",
        predicted_rollout,
        expert_actions,
        decision_frames,
    )

    print(f"[PASS] rollout saved: {output_dir}")
    print(
        f"[RESULT] MAE={overall_mae:.4f}, RMSE={overall_rmse:.4f}, "
        f"MAX={overall_max:.4f}, range_violations={range_violation_count}"
    )
    return output_dir, predicted_rollout, float(dataset.fps)


def main() -> int:
    args = parse_args()
    try:
        output_dir, predicted_rollout, dataset_fps = run_rollout(args)
        if args.rviz:
            sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
            from piper_infer_preview import run_rviz

            rate = args.rviz_rate if args.rviz_rate is not None else 1.0 / dataset_fps
            print(f"[RVIZ] replaying {len(predicted_rollout)} actions from {output_dir}")
            run_rviz(predicted_rollout, rate, args.joint_state_topic)
        return 0
    except KeyboardInterrupt:
        print("\n[STOP] interrupted")
        return 130
    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
