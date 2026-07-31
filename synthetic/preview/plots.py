#!/usr/bin/env python3
"""Static EEF/joint plots for offline trajectory review (no GUI, no RViz)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from synthetic.kinematics.action_conversion import JOINT_ORDER  # noqa: E402


class PlotError(ValueError):
    """Raised when plot input data is invalid."""


def save_eef_plot(base_xyzrpy_sequence: Any, output_path: Path) -> Path:
    """Plot EEF x/y/z (mm) and roll/pitch/yaw (deg) over frame index."""

    poses = np.asarray(base_xyzrpy_sequence, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 6:
        raise PlotError(f"base_xyzrpy_sequence must have shape (N, 6), got {poses.shape}")

    frames = np.arange(poses.shape[0])
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    for label, values in zip(("x", "y", "z"), poses[:, :3].T, strict=True):
        axes[0].plot(frames, values, label=f"{label}_mm")
    axes[0].set_ylabel("position (mm)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    for label, values in zip(("roll", "pitch", "yaw"), poses[:, 3:].T, strict=True):
        axes[1].plot(frames, values, label=f"{label}_deg")
    axes[1].set_ylabel("orientation (deg)")
    axes[1].set_xlabel("frame")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("EEF base_xyzrpy over the trajectory (unverified)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def save_joint_plot(
    joint_rad_sequence: Any,
    action_sequence: Any,
    output_path: Path,
) -> Path:
    """Plot physical joint angles (deg) and normalized 7-dim action over frame index."""

    joints = np.asarray(joint_rad_sequence, dtype=np.float64)
    action = np.asarray(action_sequence, dtype=np.float64)
    if joints.ndim != 2 or joints.shape[1] != 6:
        raise PlotError(f"joint_rad_sequence must have shape (N, 6), got {joints.shape}")
    if action.ndim != 2 or action.shape[1] != 7:
        raise PlotError(f"action_sequence must have shape (N, 7), got {action.shape}")
    if joints.shape[0] != action.shape[0]:
        raise PlotError(
            f"joint_rad_sequence and action_sequence frame counts differ: "
            f"{joints.shape[0]} vs {action.shape[0]}"
        )

    frames = np.arange(joints.shape[0])
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    for index, name in enumerate(JOINT_ORDER):
        axes[0].plot(frames, np.degrees(joints[:, index]), label=name)
    axes[0].set_ylabel("physical joint (deg)")
    axes[0].legend(loc="upper right", ncol=3, fontsize="small")
    axes[0].grid(True, alpha=0.3)

    for index, name in enumerate((*JOINT_ORDER, "gripper")):
        axes[1].plot(frames, action[:, index], label=name)
    axes[1].axhline(-100.0, color="red", linestyle="--", linewidth=0.7)
    axes[1].axhline(100.0, color="red", linestyle="--", linewidth=0.7)
    axes[1].set_ylabel("normalized action")
    axes[1].set_xlabel("frame")
    axes[1].legend(loc="upper right", ncol=4, fontsize="small")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Physical joints and normalized action over the trajectory (unverified)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path
