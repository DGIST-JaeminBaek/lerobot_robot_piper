#!/usr/bin/env python3
"""Thin, offline wrapper around `piper_sdk`'s validated forward kinematics.

Only imports `piper_sdk.kinematics.piper_fk`, which is pure DH-parameter math
(no CAN/hardware dependency) per `docs/kinematics/kinematics_check.md`. Never
imports `piper_sdk.C_PiperInterface_V2` or anything that could open a CAN
connection.

Convention (all confirmed against `docs/kinematics/kinematics_check.md` and
`scripts/tools/piper_first_chunk_fk_analysis.py`, not re-derived):

- `dh_is_offset=1`: this project's arm firmware is `S-V1.8-2` (new-firmware
  family), which the doc confirms corresponds to this DH offset convention.
- `CalFK(joint_rad)` input: list of 6 joint angles in radians.
- `CalFK(...)` output: list of 6 link poses, each `[x, y, z, roll, pitch,
  yaw]` with position in mm and roll/pitch/yaw in degrees. The EEF pose is
  the last link, `CalFK(...)[-1]`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from piper_sdk.kinematics.piper_fk import C_PiperForwardKinematics

DH_IS_OFFSET = 1


class ForwardKinematicsError(ValueError):
    """Raised when a joint configuration for FK is invalid."""


_fk_solver: C_PiperForwardKinematics | None = None


def _get_solver() -> C_PiperForwardKinematics:
    global _fk_solver
    if _fk_solver is None:
        _fk_solver = C_PiperForwardKinematics(dh_is_offset=DH_IS_OFFSET)
    return _fk_solver


def _validate_joint_rad(joint_rad: Any) -> np.ndarray:
    array = np.asarray(joint_rad, dtype=np.float64)
    if array.shape != (6,):
        raise ForwardKinematicsError(f"joint_rad must have shape (6,), got {array.shape}")
    if not np.isfinite(array).all():
        raise ForwardKinematicsError("joint_rad contains NaN or Inf")
    return array


def forward_kinematics_eef(joint_rad: Any) -> np.ndarray:
    """EEF pose for one joint configuration: `[x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg]`."""

    joints = _validate_joint_rad(joint_rad)
    all_links = _get_solver().CalFK(joints.tolist())
    pose = np.asarray(all_links[-1], dtype=np.float64)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ForwardKinematicsError(f"CalFK returned an invalid EEF pose: {pose}")
    return pose


def forward_kinematics_eef_batch(joint_rad_sequence: Any) -> np.ndarray:
    """EEF poses for `(N, 6)` joint configurations, as `(N, 6)` xyzrpy (mm, deg)."""

    array = np.asarray(joint_rad_sequence, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 6:
        raise ForwardKinematicsError(
            f"joint_rad_sequence must have shape (N, 6), got {array.shape}"
        )
    return np.stack([forward_kinematics_eef(row) for row in array], axis=0)
