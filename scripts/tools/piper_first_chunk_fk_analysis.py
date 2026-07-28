#!/usr/bin/env python3
"""모든 episode의 첫 observation에서 SmolVLA action chunk를 생성해 EEF 궤적을 비교한다.

실제 Piper와 CAN은 사용하지 않는다. Policy가 출력한 7차원 절대 joint target을
Piper calibration으로 joint1~6 radian으로 변환하고, piper_sdk의 순수 DH 계산
`C_PiperForwardKinematics.CalFK()`로 EEF pose를 계산한다.

비교 공정성을 위해 기본적으로 각 episode 추론 직전에 같은 RNG seed를 복원한다.
따라서 episode 간 차이는 sampling noise보다 첫 image/state 차이에 주로 대응한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time
from collections import Counter

import numpy as np
import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
MOTOR_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
POSE_NAMES = ["x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg"]
GROUP_ORDER = ["circle", "triangle", "rectangle", "unknown"]
GROUP_COLORS = {
    "circle": "#4c78a8",
    "triangle": "#f58518",
    "rectangle": "#54a24b",
    "unknown": "#777777",
}

# Piper plugin과 RViz player에서 사용하는 MotorCalibration 값과 동일하다.
# Joint raw unit은 0.001 degree, gripper raw unit은 0.001 mm다.
CALIBRATION_RAW = {
    "joint1": (-150_000.0, 150_000.0),
    "joint2": (0.0, 180_000.0),
    "joint3": (-170_000.0, 0.0),
    "joint4": (-100_000.0, 100_000.0),
    "joint5": (-65_000.0, 65_000.0),
    "joint6": (-100_000.0, 130_000.0),
    "gripper": (0.0, 68_000.0),
}


def parse_episode_selection(text: str, total_episodes: int) -> list[int]:
    if text.strip().lower() == "all":
        return list(range(total_episodes))
    selected: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"Invalid episode range {token!r}")
            selected.extend(range(start, end + 1))
        else:
            selected.append(int(token))
    selected = list(dict.fromkeys(selected))
    if not selected:
        raise ValueError("No episodes selected")
    invalid = [episode for episode in selected if not 0 <= episode < total_episodes]
    if invalid:
        raise ValueError(f"Episodes outside 0..{total_episodes - 1}: {invalid}")
    return selected


def checkpoint_name(policy_path: str) -> str:
    path = pathlib.Path(policy_path)
    return path.parent.name if path.name == "pretrained_model" else path.name


def load_groups(manifest_path: pathlib.Path | None, total_episodes: int) -> list[str]:
    groups = ["unknown"] * total_episodes
    if manifest_path is None or not manifest_path.is_file():
        return groups
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("episodes")
    if not isinstance(entries, list):
        raise ValueError(f"{manifest_path}: expected an episodes list")
    for episode, entry in enumerate(entries[:total_episodes]):
        source = str(entry.get("source_dataset", "")).lower()
        if "circle" in source:
            groups[episode] = "circle"
        elif "triangle" in source:
            groups[episode] = "triangle"
        elif "rectangle" in source:
            groups[episode] = "rectangle"
    return groups


def normalized_joint_to_radians(motor: str, normalized: float) -> float:
    """정규화 absolute joint target을 CalFK 입력 radian으로 변환한다."""
    if motor == "gripper":
        raise ValueError("Gripper is not part of Piper arm FK")
    raw_min, raw_max = CALIBRATION_RAW[motor]
    raw_millidegree = ((float(normalized) + 100.0) / 200.0) * (raw_max - raw_min) + raw_min
    return np.deg2rad(raw_millidegree / 1000.0).item()


def actions_to_eef(actions: np.ndarray, fk_solver) -> np.ndarray:
    """(N,7) normalized absolute targets -> (N,6) EEF [mm, deg]."""
    poses = []
    for action in actions:
        joints_rad = [
            normalized_joint_to_radians(motor, value)
            for motor, value in zip(MOTOR_NAMES[:6], action[:6])
        ]
        poses.append(fk_solver.CalFK(joints_rad)[-1])
    return np.asarray(poses, dtype=np.float64)


def reset_inference_rng(seed: int, device: torch.device) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def save_csv(
    path: pathlib.Path,
    episode_ids: list[int],
    groups: list[str],
    raw_actions: np.ndarray,
    global_actions: np.ndarray,
    safe_actions: np.ndarray,
    global_eef: np.ndarray,
    safe_eef: np.ndarray,
    start_eef: np.ndarray,
) -> None:
    fields = ["episode", "group", "action_step"]
    fields += [f"raw_{name}" for name in MOTOR_NAMES]
    fields += [f"global_{name}" for name in MOTOR_NAMES]
    fields += [f"safe_{name}" for name in MOTOR_NAMES]
    fields += [f"global_{name}" for name in POSE_NAMES]
    fields += [f"safe_{name}" for name in POSE_NAMES]
    fields += [f"global_delta_{axis}_mm" for axis in ("x", "y", "z")]
    fields += [f"safe_delta_{axis}_mm" for axis in ("x", "y", "z")]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for local_episode, episode in enumerate(episode_ids):
            for step in range(raw_actions.shape[1]):
                row: dict[str, object] = {
                    "episode": episode,
                    "group": groups[local_episode],
                    "action_step": step,
                }
                row.update(
                    {
                        f"raw_{name}": float(value)
                        for name, value in zip(MOTOR_NAMES, raw_actions[local_episode, step])
                    }
                )
                row.update(
                    {
                        f"global_{name}": float(value)
                        for name, value in zip(MOTOR_NAMES, global_actions[local_episode, step])
                    }
                )
                row.update(
                    {
                        f"safe_{name}": float(value)
                        for name, value in zip(MOTOR_NAMES, safe_actions[local_episode, step])
                    }
                )
                row.update(
                    {
                        f"global_{name}": float(value)
                        for name, value in zip(POSE_NAMES, global_eef[local_episode, step])
                    }
                )
                row.update(
                    {
                        f"safe_{name}": float(value)
                        for name, value in zip(POSE_NAMES, safe_eef[local_episode, step])
                    }
                )
                global_delta = global_eef[local_episode, step, :3] - start_eef[local_episode, :3]
                safe_delta = safe_eef[local_episode, step, :3] - start_eef[local_episode, :3]
                row.update(
                    {
                        f"global_delta_{axis}_mm": float(value)
                        for axis, value in zip(("x", "y", "z"), global_delta)
                    }
                )
                row.update(
                    {
                        f"safe_delta_{axis}_mm": float(value)
                        for axis, value in zip(("x", "y", "z"), safe_delta)
                    }
                )
                writer.writerow(row)


def configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def group_indices(groups: list[str]) -> dict[str, np.ndarray]:
    array = np.asarray(groups)
    return {
        group: np.flatnonzero(array == group)
        for group in GROUP_ORDER
        if np.any(array == group)
    }


def plot_3d_trajectories(
    path: pathlib.Path,
    eef: np.ndarray,
    start_eef: np.ndarray,
    groups: list[str],
    *,
    relative: bool,
    variant: str,
) -> None:
    plt = configure_matplotlib()
    figure = plt.figure(figsize=(12, 10))
    axis = figure.add_subplot(111, projection="3d")
    seen: set[str] = set()
    for episode_index, group in enumerate(groups):
        xyz = eef[episode_index, :, :3].copy()
        if relative:
            xyz -= start_eef[episode_index, None, :3]
        axis.plot(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            color=GROUP_COLORS[group],
            alpha=0.42,
            linewidth=1.1,
            label=group if group not in seen else None,
        )
        axis.scatter(
            xyz[0, 0],
            xyz[0, 1],
            xyz[0, 2],
            color=GROUP_COLORS[group],
            s=7,
            alpha=0.6,
        )
        seen.add(group)
    axis.set_xlabel("X (mm)")
    axis.set_ylabel("Y (mm)")
    axis.set_zlabel("Z (mm)")
    axis.set_title(
        f"First action chunk EEF trajectories ({variant}, "
        f"{'relative to observed start' if relative else 'absolute'})"
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def plot_xyz_by_step(
    path: pathlib.Path,
    eef: np.ndarray,
    start_eef: np.ndarray,
    groups: list[str],
    *,
    relative: bool,
    variant: str,
) -> None:
    plt = configure_matplotlib()
    values = eef[:, :, :3].copy()
    if relative:
        values -= start_eef[:, None, :3]
    indices_by_group = group_indices(groups)
    figure, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    steps = np.arange(values.shape[1])
    for axis_index, axis_name in enumerate(("X", "Y", "Z")):
        plot_axis = axes[axis_index]
        for group, indices in indices_by_group.items():
            group_values = values[indices, :, axis_index]
            mean = group_values.mean(axis=0)
            std = group_values.std(axis=0)
            color = GROUP_COLORS[group]
            plot_axis.plot(steps, mean, color=color, linewidth=2, label=f"{group} mean")
            plot_axis.fill_between(steps, mean - std, mean + std, color=color, alpha=0.16)
        plot_axis.set_ylabel(f"{axis_name} (mm)")
        plot_axis.grid(alpha=0.22)
    axes[0].legend(ncol=3)
    axes[-1].set_xlabel("Action step in first chunk")
    figure.suptitle(
        f"EEF XYZ mean ± std by shape ({variant}, "
        f"{'relative' if relative else 'absolute'})"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def plot_endpoint_projections(
    path: pathlib.Path,
    eef: np.ndarray,
    start_eef: np.ndarray,
    groups: list[str],
    *,
    relative: bool,
    variant: str,
) -> None:
    plt = configure_matplotlib()
    endpoints = eef[:, -1, :3].copy()
    if relative:
        endpoints -= start_eef[:, :3]
    projections = [(0, 1, "X", "Y"), (0, 2, "X", "Z"), (1, 2, "Y", "Z")]
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    for axis, (first, second, first_name, second_name) in zip(axes, projections):
        for group, indices in group_indices(groups).items():
            axis.scatter(
                endpoints[indices, first],
                endpoints[indices, second],
                color=GROUP_COLORS[group],
                s=32,
                alpha=0.75,
                label=group,
            )
        axis.set_xlabel(f"{first_name} (mm)")
        axis.set_ylabel(f"{second_name} (mm)")
        axis.grid(alpha=0.22)
    axes[0].legend()
    figure.suptitle(
        f"First-chunk EEF endpoint distribution ({variant}, "
        f"{'relative' if relative else 'absolute'})"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def plot_joint_targets(
    path: pathlib.Path,
    actions: np.ndarray,
    groups: list[str],
    *,
    variant: str,
) -> None:
    plt = configure_matplotlib()
    figure, axes = plt.subplots(7, 1, figsize=(13, 17), sharex=True)
    steps = np.arange(actions.shape[1])
    for motor_index, axis in enumerate(axes):
        for group, indices in group_indices(groups).items():
            group_values = actions[indices, :, motor_index]
            mean = group_values.mean(axis=0)
            std = group_values.std(axis=0)
            color = GROUP_COLORS[group]
            axis.plot(steps, mean, color=color, linewidth=1.7, label=group)
            axis.fill_between(steps, mean - std, mean + std, color=color, alpha=0.15)
        axis.set_ylabel(MOTOR_NAMES[motor_index])
        axis.grid(alpha=0.2)
    axes[0].legend(ncol=3)
    axes[-1].set_xlabel("Action step in first chunk")
    figure.suptitle(f"First-chunk normalized joint targets ({variant}, mean ± std)")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="모든 episode 첫 observation의 SmolVLA action chunk를 Piper FK로 비교"
    )
    parser.add_argument(
        "--dataset-root",
        type=pathlib.Path,
        default=pathlib.Path("records/0727/erase_the_shape_512"),
    )
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--task", default=None, help="기본값: 각 dataset item의 task")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", default="all", help="all, 0,20,40 또는 0-19")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--same-noise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="각 episode마다 같은 RNG seed 사용(기본 true)",
    )
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=5.0,
        help="safe trajectory 계산에 사용할 preview 상대 제한",
    )
    parser.add_argument(
        "--plot-variant",
        choices=("global", "safe"),
        default="global",
        help="PNG에 표시할 궤적; NPZ/CSV에는 둘 다 저장",
    )
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=pathlib.Path("configs/erase_shape_frame_ranges.json"),
        help="circle/triangle/rectangle label 출처",
    )
    parser.add_argument("--output-dir", type=pathlib.Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.max_relative_target <= 0:
        raise ValueError("--max-relative-target must be positive")

    sys.path.insert(0, str(SCRIPT_DIR))
    from piper_human_approved_inference import (
        apply_global_limits,
        apply_preview_relative_limit,
        validate_chunk,
    )
    from piper_offline_chunk_rollout import (
        load_policy,
        make_raw_observation,
        predict_chunk,
    )
    from piper_sdk.kinematics.piper_fk import C_PiperForwardKinematics
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.utils import get_safe_torch_device

    dataset_root = args.dataset_root.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve() if args.manifest else None
    print(f"[LOAD] dataset={dataset_root}")
    dataset = LeRobotDataset(
        repo_id=f"local/{dataset_root.name}",
        root=dataset_root,
        video_backend="pyav",
    )
    total_episodes = int(dataset.meta.total_episodes)
    episode_ids = parse_episode_selection(args.episodes, total_episodes)
    all_groups = load_groups(manifest_path, total_episodes)
    groups = [all_groups[episode] for episode in episode_ids]

    print(f"[LOAD] policy={args.policy_path}, device={args.device}")
    config, policy, preprocessor, postprocessor = load_policy(
        args.policy_path,
        dataset.meta,
        args.device,
    )
    device = get_safe_torch_device(policy.config.device)
    policy_chunk_size = int(getattr(config, "chunk_size", 0))
    if args.chunk_size > policy_chunk_size:
        raise ValueError(
            f"--chunk-size={args.chunk_size} exceeds policy chunk_size={policy_chunk_size}"
        )

    output_dir = args.output_dir or (
        REPO_ROOT
        / "outputs"
        / "first_chunk_fk"
        / f"{dataset_root.name}_{checkpoint_name(args.policy_path)}"
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fk_solver = C_PiperForwardKinematics(dh_is_offset=1)

    raw_chunks = []
    global_chunks = []
    safe_chunks = []
    global_eef_chunks = []
    safe_eef_chunks = []
    start_states = []
    start_eef_poses = []
    inference_times = []
    global_clamp_counts = []
    relative_clamp_counts = []

    print(
        f"[RUN] episodes={len(episode_ids)}, chunk_size={args.chunk_size}, "
        f"same_noise={args.same_noise}, seed={args.seed}"
    )
    for result_index, episode in enumerate(episode_ids):
        episode_meta = dataset.meta.episodes[episode]
        first_dataset_index = int(episode_meta["dataset_from_index"])
        item = dataset[first_dataset_index]
        item_episode = int(item["episode_index"].item())
        item_frame = int(item["frame_index"].item())
        if item_episode != episode or item_frame != 0:
            raise RuntimeError(
                f"Episode {episode} first index mismatch: "
                f"episode={item_episode}, frame={item_frame}"
            )

        inference_seed = args.seed if args.same_noise else args.seed + episode
        reset_inference_rng(inference_seed, device)
        policy.reset()
        task = args.task if args.task is not None else str(item.get("task", ""))
        raw_observation = make_raw_observation(dataset, item)
        started = time.perf_counter()
        raw_tensor = predict_chunk(
            raw_observation=raw_observation,
            dataset_features=dataset.features,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=device,
            task=task,
        )
        inference_seconds = time.perf_counter() - started
        raw = raw_tensor[: args.chunk_size].numpy().astype(np.float32, copy=False)
        validate_chunk(raw)
        global_limited, global_count, _ = apply_global_limits(raw)
        start_state = item["observation.state"].detach().cpu().numpy().astype(np.float32)
        safe, relative_count, _ = apply_preview_relative_limit(
            start_state,
            global_limited,
            args.max_relative_target,
        )

        raw_chunks.append(raw)
        global_chunks.append(global_limited)
        safe_chunks.append(safe)
        global_eef_chunks.append(actions_to_eef(global_limited, fk_solver))
        safe_eef_chunks.append(actions_to_eef(safe, fk_solver))
        start_states.append(start_state)
        start_eef_poses.append(actions_to_eef(start_state[None, :], fk_solver)[0])
        inference_times.append(inference_seconds)
        global_clamp_counts.append(global_count)
        relative_clamp_counts.append(relative_count)
        print(
            f"[{result_index + 1:02d}/{len(episode_ids):02d}] ep={episode:02d} "
            f"group={groups[result_index]:9s} infer={inference_seconds:.3f}s "
            f"global_clamp={global_count} relative_clamp={relative_count}"
        )

    raw_actions = np.stack(raw_chunks)
    global_actions = np.stack(global_chunks)
    safe_actions = np.stack(safe_chunks)
    global_eef = np.stack(global_eef_chunks)
    safe_eef = np.stack(safe_eef_chunks)
    start_states_array = np.stack(start_states)
    start_eef = np.stack(start_eef_poses)
    selected_eef = global_eef if args.plot_variant == "global" else safe_eef
    selected_actions = global_actions if args.plot_variant == "global" else safe_actions

    np.savez_compressed(
        output_dir / "first_chunk_fk.npz",
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        groups=np.asarray(groups),
        raw_actions=raw_actions,
        global_actions=global_actions,
        safe_actions=safe_actions,
        start_states=start_states_array,
        start_eef=start_eef,
        global_eef=global_eef,
        safe_eef=safe_eef,
        inference_seconds=np.asarray(inference_times),
        global_clamp_counts=np.asarray(global_clamp_counts),
        relative_clamp_counts=np.asarray(relative_clamp_counts),
    )
    save_csv(
        output_dir / "first_chunk_fk.csv",
        episode_ids,
        groups,
        raw_actions,
        global_actions,
        safe_actions,
        global_eef,
        safe_eef,
        start_eef,
    )

    plot_3d_trajectories(
        output_dir / f"eef_3d_absolute_{args.plot_variant}.png",
        selected_eef,
        start_eef,
        groups,
        relative=False,
        variant=args.plot_variant,
    )
    plot_3d_trajectories(
        output_dir / f"eef_3d_relative_{args.plot_variant}.png",
        selected_eef,
        start_eef,
        groups,
        relative=True,
        variant=args.plot_variant,
    )
    plot_xyz_by_step(
        output_dir / f"eef_xyz_absolute_{args.plot_variant}.png",
        selected_eef,
        start_eef,
        groups,
        relative=False,
        variant=args.plot_variant,
    )
    plot_xyz_by_step(
        output_dir / f"eef_xyz_relative_{args.plot_variant}.png",
        selected_eef,
        start_eef,
        groups,
        relative=True,
        variant=args.plot_variant,
    )
    plot_endpoint_projections(
        output_dir / f"eef_endpoints_relative_{args.plot_variant}.png",
        selected_eef,
        start_eef,
        groups,
        relative=True,
        variant=args.plot_variant,
    )
    plot_joint_targets(
        output_dir / f"joint_targets_{args.plot_variant}.png",
        selected_actions,
        groups,
        variant=args.plot_variant,
    )

    relative_xyz = selected_eef[:, :, :3] - start_eef[:, None, :3]
    endpoint_relative = relative_xyz[:, -1]
    summary = {
        "dataset_root": str(dataset_root),
        "policy_path": str(pathlib.Path(args.policy_path).expanduser().resolve()),
        "episodes": episode_ids,
        "group_counts": dict(Counter(groups)),
        "chunk_size": args.chunk_size,
        "seed": args.seed,
        "same_noise": args.same_noise,
        "plot_variant": args.plot_variant,
        "max_relative_target": args.max_relative_target,
        "mean_inference_seconds": float(np.mean(inference_times)),
        "global_clamped_values": int(np.sum(global_clamp_counts)),
        "relative_clamped_values": int(np.sum(relative_clamp_counts)),
        "start_eef_position_std_mm": dict(
            zip(("x", "y", "z"), start_eef[:, :3].std(axis=0).tolist())
        ),
        "relative_endpoint_mean_mm": dict(
            zip(("x", "y", "z"), endpoint_relative.mean(axis=0).tolist())
        ),
        "relative_endpoint_std_mm": dict(
            zip(("x", "y", "z"), endpoint_relative.std(axis=0).tolist())
        ),
        "mean_xyz_spread_by_action_mm": float(
            np.linalg.norm(relative_xyz.std(axis=0), axis=1).mean()
        ),
        "max_xyz_spread_by_action_mm": float(
            np.linalg.norm(relative_xyz.std(axis=0), axis=1).max()
        ),
        "units": {
            "actions": "normalized absolute joint target",
            "eef_position": "mm",
            "eef_orientation": "degree",
        },
        "limitations": [
            "All observations are training-dataset first frames.",
            "CalFK is kinematic only and does not model contact or dynamics.",
            "Safe max-relative preview assumes each previous target was reached exactly.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[PASS] outputs saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
