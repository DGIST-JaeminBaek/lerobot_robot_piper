#!/usr/bin/env python3
"""Physical joint/gripper <-> normalized 7-dim action, using the project's
existing (hardcoded, not JSON) calibration -- copied verbatim, not
re-derived.

Source of truth (this project does not load calibration from disk; these
constants ARE the calibration):
- `lerobot_robot_piper/piper_follower.py:76-84` (identical in
  `piper_leader.py:37-45`): per-motor `MotorCalibration(id, drive_mode,
  homing_offset, range_min, range_max)`.
- `lerobot_robot_piper/motors/piper_motors_bus.py:363-418`
  (`_normalize`/`_unnormalize`): the exact formulas for `RANGE_M100_100`
  (joints) and `RANGE_0_100` (gripper). `PiperMotorsBus.apply_drive_mode`
  is `False` (class attribute), so no motor's sign is ever inverted here.
- Raw units: joint raw = 0.001 degree, gripper raw = 0.001 mm (comment in
  `scripts/tools/piper_first_chunk_fk_analysis.py:39`, matching the
  `unnormalize_to_physical()` helpers duplicated in
  `piper_first_chunk_fk_analysis.py` and `piper_infer_preview.py` because
  `PiperMotorsBus` cannot be instantiated without hardware).
- `lerobot_robot_piper/motors/tables.py`'s `INITIALIZE_POSITION` comment:
  "gripper는 닫힘 0mm" -- raw=0 is the closed/0mm gripper position, so
  `range_max` (68mm) is the fully-open position.

Unlike the *runtime* `_normalize`/`_unnormalize` (which clamp out-of-range
values for safety during real execution), the generation-time functions
here raise on out-of-calibration-range input -- clamping would silently
hide an invalid generated trajectory, and `max_relative_target` on the real
`PiperFollower` is an execution-time guard, not a trajectory validator (see
`synthetic/CLAUDE_HANDOFF.md` section 6D rules).

This module never imports anything CAN/hardware-related.
"""

from __future__ import annotations

from typing import Any

import numpy as np

JOINT_ORDER = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")

# (range_min, range_max) in raw units (0.001 degree), from
# lerobot_robot_piper/piper_follower.py:76-84.
JOINT_RAW_RANGE_MILLIDEG: dict[str, tuple[int, int]] = {
    "joint1": (-150000, 150000),
    "joint2": (0, 180000),
    "joint3": (-170000, 0),
    "joint4": (-100000, 100000),
    "joint5": (-65000, 65000),
    "joint6": (-100000, 130000),
}

# (range_min, range_max) in raw units (0.001 mm), from
# lerobot_robot_piper/piper_follower.py:82. 0 = closed, 68mm = fully open.
GRIPPER_RAW_RANGE_MICRON: tuple[int, int] = (0, 68000)
GRIPPER_FULLY_OPEN_MM = GRIPPER_RAW_RANGE_MICRON[1] / 1000.0
GRIPPER_CLOSED_MM = GRIPPER_RAW_RANGE_MICRON[0] / 1000.0

_JOINT_RAW_RANGE_DEG: dict[str, tuple[float, float]] = {
    name: (low / 1000.0, high / 1000.0) for name, (low, high) in JOINT_RAW_RANGE_MILLIDEG.items()
}


class ActionConversionError(ValueError):
    """Raised when a physical or normalized value is invalid or out of calibration range."""


def _require_finite(value: float, *, name: str) -> float:
    if not np.isfinite(value):
        raise ActionConversionError(f"{name} must be finite, got {value}")
    return float(value)


def joint_rad_to_normalized(joint_rad: Any) -> np.ndarray:
    """`(6,)` physical joint radians -> `(6,)` normalized values in [-100, 100].

    Raises if a joint falls outside its calibrated raw range (this is a
    generation-time correctness check, not the runtime bus's clamp).
    """

    joints = np.asarray(joint_rad, dtype=np.float64)
    if joints.shape != (6,):
        raise ActionConversionError(f"joint_rad must have shape (6,), got {joints.shape}")
    if not np.isfinite(joints).all():
        raise ActionConversionError("joint_rad contains NaN or Inf")

    normalized = np.empty(6, dtype=np.float64)
    for index, name in enumerate(JOINT_ORDER):
        degrees = np.degrees(joints[index])
        low, high = _JOINT_RAW_RANGE_DEG[name]
        if degrees < low or degrees > high:
            raise ActionConversionError(
                f"{name}={degrees:.4f}deg is outside its calibrated range "
                f"[{low:.4f}, {high:.4f}]deg"
            )
        normalized[index] = ((degrees - low) / (high - low)) * 200.0 - 100.0
    return normalized


def normalized_to_joint_rad(normalized: Any) -> np.ndarray:
    """Inverse of `joint_rad_to_normalized`: `(6,)` in [-100, 100] -> `(6,)` radians."""

    values = np.asarray(normalized, dtype=np.float64)
    if values.shape != (6,):
        raise ActionConversionError(f"normalized must have shape (6,), got {values.shape}")
    if not np.isfinite(values).all():
        raise ActionConversionError("normalized contains NaN or Inf")
    if np.any(values < -100.0) or np.any(values > 100.0):
        raise ActionConversionError(f"normalized values must be within [-100, 100], got {values}")

    joint_rad = np.empty(6, dtype=np.float64)
    for index, name in enumerate(JOINT_ORDER):
        low, high = _JOINT_RAW_RANGE_DEG[name]
        degrees = ((values[index] + 100.0) / 200.0) * (high - low) + low
        joint_rad[index] = np.radians(degrees)
    return joint_rad


def gripper_mm_to_normalized(gripper_mm: float) -> float:
    """Physical gripper opening (mm, 0=closed) -> normalized value in [0, 100]."""

    value = _require_finite(gripper_mm, name="gripper_mm")
    if value < GRIPPER_CLOSED_MM or value > GRIPPER_FULLY_OPEN_MM:
        raise ActionConversionError(
            f"gripper_mm={value:.4f} is outside its calibrated range "
            f"[{GRIPPER_CLOSED_MM}, {GRIPPER_FULLY_OPEN_MM}]mm"
        )
    return (value - GRIPPER_CLOSED_MM) / (GRIPPER_FULLY_OPEN_MM - GRIPPER_CLOSED_MM) * 100.0


def normalized_to_gripper_mm(normalized: float) -> float:
    """Inverse of `gripper_mm_to_normalized`."""

    value = _require_finite(normalized, name="normalized")
    if value < 0.0 or value > 100.0:
        raise ActionConversionError(f"normalized gripper value must be within [0, 100], got {value}")
    return value / 100.0 * (GRIPPER_FULLY_OPEN_MM - GRIPPER_CLOSED_MM) + GRIPPER_CLOSED_MM


def gripper_fraction_to_mm(closed_fraction: Any) -> np.ndarray:
    """`synthetic.trajectory.schema` gripper fraction (0=open, 1=closed) -> mm.

    `GRIPPER_OPEN`/`GRIPPER_CLOSED` in `synthetic/trajectory/schema.py` are
    0.0/1.0; this maps that abstract closed-fraction directly onto the
    physical gripper travel documented above.
    """

    fraction = np.asarray(closed_fraction, dtype=np.float64)
    if not np.isfinite(fraction).all():
        raise ActionConversionError("closed_fraction contains NaN or Inf")
    if np.any(fraction < 0.0) or np.any(fraction > 1.0):
        raise ActionConversionError(f"closed_fraction must be within [0, 1], got {fraction}")
    return (1.0 - fraction) * (GRIPPER_FULLY_OPEN_MM - GRIPPER_CLOSED_MM) + GRIPPER_CLOSED_MM


def build_normalized_action(
    joint_rad_sequence: Any,
    gripper_closed_fraction_sequence: Any,
) -> np.ndarray:
    """`(N, 6)` joint radians + `(N,)` gripper closed-fraction -> `(N, 7)` normalized action.

    Column order is `joint1, joint2, joint3, joint4, joint5, joint6,
    gripper`, matching every existing use in this repo (confirmed: gripper
    is always last, never reordered).
    """

    joints = np.asarray(joint_rad_sequence, dtype=np.float64)
    if joints.ndim != 2 or joints.shape[1] != 6:
        raise ActionConversionError(
            f"joint_rad_sequence must have shape (N, 6), got {joints.shape}"
        )
    gripper_fraction = np.asarray(gripper_closed_fraction_sequence, dtype=np.float64)
    if gripper_fraction.shape != (joints.shape[0],):
        raise ActionConversionError(
            f"gripper_closed_fraction_sequence must have shape ({joints.shape[0]},), "
            f"got {gripper_fraction.shape}"
        )

    gripper_mm = gripper_fraction_to_mm(gripper_fraction)
    action = np.empty((joints.shape[0], 7), dtype=np.float64)
    for row in range(joints.shape[0]):
        action[row, :6] = joint_rad_to_normalized(joints[row])
        action[row, 6] = gripper_mm_to_normalized(float(gripper_mm[row]))
    return action
