#!/usr/bin/env python3
"""Offline numerical IK for the Piper arm via `ikpy` + the validated URDF.

`piper_sdk` has no inverse kinematics of its own (confirmed by inspecting
`piper_sdk/kinematics/`, which only contains `piper_fk.py`). This module
wires up `ikpy` (already installed in the `ugrp` conda env, no new
dependency) against the same URDF RViz uses
(`agx_arm_urdf/piper/urdf/piper_description.urdf`), which
`docs/kinematics/kinematics_check.md` already reports matches
`piper_sdk`'s `CalFK` to <=0.1mm at zero configuration -- independently
reproduced during this stage's research (<=0.1mm across several
non-trivial configurations too).

`ikpy`'s numerical (Jacobian/least-squares) solver is the same family of
method AgileX's own reference IK implementation uses (Jacobian + damped
pseudoinverse; see the discourse.openrobotics.org writeup referenced by the
user), and its DH offset convention matches `piper_sdk`'s
`dh_is_offset=1` (this project's firmware is confirmed on that convention).

Every solved joint configuration is re-validated with `piper_fk.py`'s
`CalFK`-based forward kinematics -- the project's actual trusted FK, not
`ikpy`'s own internal FK -- so a wrong or unreachable IK result fails loudly
rather than being silently accepted.

This module never imports anything CAN/hardware-related.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ikpy.chain import Chain
from scipy.spatial.transform import Rotation

from synthetic.kinematics.piper_fk import forward_kinematics_eef

DEFAULT_URDF_PATH = Path("/home/ugrp43/UGRP/agx_arm_urdf/piper/urdf/piper_description.urdf")
URDF_PATH_ENV_VAR = "PIPER_URDF_PATH"

DEFAULT_POSITION_TOL_MM = 1.0
DEFAULT_ANGLE_TOL_DEG = 1.0
_JOINT_LIMIT_ATOL_RAD = 1e-6


class InverseKinematicsError(ValueError):
    """Raised when an IK target, seed, or solved joint configuration is invalid."""


_chain: Chain | None = None
_chain_urdf_path: Path | None = None


def resolve_urdf_path(urdf_path: Path | None = None) -> Path:
    if urdf_path is not None:
        return Path(urdf_path)
    env_value = os.environ.get(URDF_PATH_ENV_VAR)
    if env_value:
        return Path(env_value)
    return DEFAULT_URDF_PATH


def load_chain(urdf_path: Path | None = None) -> Chain:
    """Load (and cache) the `ikpy` chain for the Piper arm from its URDF."""

    global _chain, _chain_urdf_path
    resolved = resolve_urdf_path(urdf_path).expanduser().resolve()
    if _chain is not None and _chain_urdf_path == resolved:
        return _chain
    if not resolved.is_file():
        raise InverseKinematicsError(
            f"Piper URDF not found at {resolved}. Set {URDF_PATH_ENV_VAR} or "
            "pass urdf_path explicitly; this offline IK stage does not "
            "fabricate a kinematic model."
        )
    chain = Chain.from_urdf_file(str(resolved))
    if [link.name for link in chain.links[1:]] != [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    ]:
        raise InverseKinematicsError(
            "unexpected URDF chain link names: "
            f"{[link.name for link in chain.links]}; expected a fixed base "
            "link followed by joint1..joint6"
        )
    _chain = chain
    _chain_urdf_path = resolved
    return chain


def joint_bounds_rad(urdf_path: Path | None = None) -> np.ndarray:
    """`(6, 2)` array of `[lower, upper]` joint limits (rad), from the URDF."""

    chain = load_chain(urdf_path)
    bounds = np.asarray([link.bounds for link in chain.links[1:]], dtype=np.float64)
    if bounds.shape != (6, 2):
        raise InverseKinematicsError(f"unexpected joint bounds shape: {bounds.shape}")
    return bounds


def _fallback_seeds_rad(urdf_path: Path | None = None) -> list[np.ndarray]:
    bounds = joint_bounds_rad(urdf_path)
    zero_clipped = np.clip(np.zeros(6), bounds[:, 0], bounds[:, 1])
    midpoint = (bounds[:, 0] + bounds[:, 1]) / 2.0
    return [zero_clipped, midpoint]


def xyzrpy_to_matrix(pose_mm_deg: Any) -> np.ndarray:
    """`[x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg]` -> 4x4 matrix (meters)."""

    pose = np.asarray(pose_mm_deg, dtype=np.float64)
    if pose.shape != (6,):
        raise InverseKinematicsError(f"pose must have shape (6,), got {pose.shape}")
    if not np.isfinite(pose).all():
        raise InverseKinematicsError("pose contains NaN or Inf")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_euler("xyz", pose[3:], degrees=True).as_matrix()
    matrix[:3, 3] = pose[:3] / 1000.0
    return matrix


def matrix_to_xyzrpy(matrix: Any) -> np.ndarray:
    """Inverse of `xyzrpy_to_matrix`: 4x4 matrix (meters) -> xyzrpy (mm, deg)."""

    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (4, 4):
        raise InverseKinematicsError(f"matrix must have shape (4, 4), got {array.shape}")
    xyz_mm = array[:3, 3] * 1000.0
    rpy_deg = Rotation.from_matrix(array[:3, :3]).as_euler("xyz", degrees=True)
    return np.concatenate([xyz_mm, rpy_deg])


def _validate_seed(seed_rad: Any) -> np.ndarray:
    seed = np.asarray(seed_rad, dtype=np.float64)
    if seed.shape != (6,):
        raise InverseKinematicsError(f"seed_joint_rad must have shape (6,), got {seed.shape}")
    if not np.isfinite(seed).all():
        raise InverseKinematicsError("seed_joint_rad contains NaN or Inf")
    return seed


def _check_joint_limits(joint_rad: np.ndarray, bounds: np.ndarray) -> None:
    below = joint_rad < bounds[:, 0] - _JOINT_LIMIT_ATOL_RAD
    above = joint_rad > bounds[:, 1] + _JOINT_LIMIT_ATOL_RAD
    if np.any(below) or np.any(above):
        raise InverseKinematicsError(
            f"solved joint configuration violates joint limit: joints={joint_rad.tolist()}, "
            f"bounds={bounds.tolist()}"
        )


@dataclass(frozen=True)
class IKSolution:
    """One IK solve's result and its FK-verified residual."""

    joint_rad: np.ndarray
    position_error_mm: float
    angle_error_deg: float
    seed_index: int
    joint_distance_from_primary_seed_rad: float


def solve_ik(
    target_xyzrpy_mm_deg: Any,
    *,
    seed_joint_rad: Any,
    extra_seeds_rad: list[Any] | None = None,
    position_tol_mm: float = DEFAULT_POSITION_TOL_MM,
    angle_tol_deg: float = DEFAULT_ANGLE_TOL_DEG,
    urdf_path: Path | None = None,
) -> IKSolution:
    """Solve IK for one target pose, verified against `piper_fk.forward_kinematics_eef`.

    Tries `seed_joint_rad` first, then `extra_seeds_rad` (defaulting to a
    couple of generic bounds-derived seeds) if it does not converge within
    tolerance. Among every candidate that *does* converge, the one closest
    (in joint space) to `seed_joint_rad` is returned -- this is what keeps a
    sequence of per-frame solves from jumping to a different IK branch.
    """

    target_pose = np.asarray(target_xyzrpy_mm_deg, dtype=np.float64)
    if target_pose.shape != (6,):
        raise InverseKinematicsError(f"target pose must have shape (6,), got {target_pose.shape}")
    primary_seed = _validate_seed(seed_joint_rad)

    chain = load_chain(urdf_path)
    bounds = joint_bounds_rad(urdf_path)
    target_matrix = xyzrpy_to_matrix(target_pose)

    candidate_seeds = [primary_seed]
    for extra in extra_seeds_rad or _fallback_seeds_rad(urdf_path):
        candidate_seeds.append(_validate_seed(extra))

    converged: list[tuple[int, np.ndarray, float, float]] = []
    attempts: list[str] = []
    for index, seed in enumerate(candidate_seeds):
        full_seed = np.concatenate([[0.0], seed])
        # `orientation_mode` defaults to None in ikpy, which optimizes ONLY
        # position and leaves orientation unconstrained -- "all" is required
        # to actually solve the full 6D pose (verified empirically: without
        # it, position converges to <0.1mm while orientation error can be
        # tens of degrees).
        full_solution = np.asarray(
            chain.inverse_kinematics_frame(
                target_matrix, initial_position=full_seed, orientation_mode="all"
            ),
            dtype=np.float64,
        )
        joint_rad = full_solution[1:]
        try:
            solved_pose = forward_kinematics_eef(joint_rad)
        except Exception as exc:  # noqa: BLE001 - report and keep trying other seeds
            attempts.append(f"seed[{index}] FK-of-solution failed: {exc}")
            continue

        position_error_mm = float(np.linalg.norm(solved_pose[:3] - target_pose[:3]))
        target_matrix_check = xyzrpy_to_matrix(target_pose)
        solved_matrix_check = xyzrpy_to_matrix(solved_pose)
        rotation_diff = target_matrix_check[:3, :3].T @ solved_matrix_check[:3, :3]
        angle_error_deg = float(
            np.degrees(np.arccos(np.clip((np.trace(rotation_diff) - 1.0) / 2.0, -1.0, 1.0)))
        )
        attempts.append(
            f"seed[{index}] position_error_mm={position_error_mm:.4f} "
            f"angle_error_deg={angle_error_deg:.4f}"
        )
        if position_error_mm <= position_tol_mm and angle_error_deg <= angle_tol_deg:
            converged.append((index, joint_rad, position_error_mm, angle_error_deg))

    if not converged:
        raise InverseKinematicsError(
            "IK did not converge within tolerance "
            f"(position_tol_mm={position_tol_mm}, angle_tol_deg={angle_tol_deg}) "
            f"for target pose {target_pose.tolist()} from any of "
            f"{len(candidate_seeds)} seed(s): " + "; ".join(attempts)
        )

    def distance_to_primary(item: tuple[int, np.ndarray, float, float]) -> float:
        _index, joint_rad, _pos_err, _ang_err = item
        return float(np.max(np.abs(joint_rad - primary_seed)))

    best_index, best_joint_rad, best_position_error, best_angle_error = min(
        converged, key=distance_to_primary
    )
    _check_joint_limits(best_joint_rad, bounds)

    return IKSolution(
        joint_rad=best_joint_rad,
        position_error_mm=best_position_error,
        angle_error_deg=best_angle_error,
        seed_index=best_index,
        joint_distance_from_primary_seed_rad=distance_to_primary(
            (best_index, best_joint_rad, best_position_error, best_angle_error)
        ),
    )


def solve_ik_sequence(
    target_xyzrpy_sequence: Any,
    *,
    initial_seed_rad: Any,
    position_tol_mm: float = DEFAULT_POSITION_TOL_MM,
    angle_tol_deg: float = DEFAULT_ANGLE_TOL_DEG,
    max_joint_step_rad: float | None = None,
    urdf_path: Path | None = None,
) -> tuple[np.ndarray, list[IKSolution]]:
    """Solve IK frame-by-frame, warm-starting each solve from the previous frame.

    Returns `(joint_rad_sequence (N, 6), per_frame_solutions)`. Raises if any
    frame fails to converge, or (when `max_joint_step_rad` is given) if the
    joint-space step between consecutive frames exceeds it.
    """

    poses = np.asarray(target_xyzrpy_sequence, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 6:
        raise InverseKinematicsError(
            f"target_xyzrpy_sequence must have shape (N, 6), got {poses.shape}"
        )
    if poses.shape[0] < 1:
        raise InverseKinematicsError("target_xyzrpy_sequence must contain at least 1 pose")

    seed = _validate_seed(initial_seed_rad)
    solutions: list[IKSolution] = []
    joint_rad_sequence = np.empty((poses.shape[0], 6), dtype=np.float64)

    for frame_index, pose in enumerate(poses):
        try:
            solution = solve_ik(
                pose,
                seed_joint_rad=seed,
                position_tol_mm=position_tol_mm,
                angle_tol_deg=angle_tol_deg,
                urdf_path=urdf_path,
            )
        except InverseKinematicsError as exc:
            raise InverseKinematicsError(f"frame {frame_index}: {exc}") from exc

        if max_joint_step_rad is not None and frame_index > 0:
            step = float(np.max(np.abs(solution.joint_rad - joint_rad_sequence[frame_index - 1])))
            if step > max_joint_step_rad:
                raise InverseKinematicsError(
                    f"frame {frame_index}: joint step {step:.6f}rad exceeds "
                    f"max_joint_step_rad={max_joint_step_rad} (possible IK branch jump)"
                )

        joint_rad_sequence[frame_index] = solution.joint_rad
        solutions.append(solution)
        seed = solution.joint_rad

    return joint_rad_sequence, solutions
