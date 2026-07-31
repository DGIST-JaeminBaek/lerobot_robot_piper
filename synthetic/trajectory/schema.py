#!/usr/bin/env python3
"""Data schema for the erase-task Cartesian trajectory pipeline.

Defines the fixed segment ordering and the per-segment / full-trajectory
containers that carry `board_xy`, `base_xyzrpy`, gripper state, frame index
and time. `base_xyzrpy` columns are `(x_mm, y_mm, z_mm, roll_deg, pitch_deg,
yaw_deg)`, matching the coordinate table in `synthetic/README.md`.

This module is intentionally independent from ROS, CAN, LeRobot, and Piper
hardware. It only handles NumPy arrays and JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

FORMAT_VERSION = 1
SEGMENT_TRAJECTORY_TYPE = "segment_trajectory"
FULL_TRAJECTORY_TYPE = "full_trajectory"

SEGMENT_ORDER = (
    "PARKING_TO_PREGRASP",
    "GRASP",
    "LIFT",
    "TRANSFER_ABOVE_BOARD",
    "DESCEND",
    "ERASE",
    "LIFT_FROM_BOARD",
    "RETURN_ABOVE_ERASER",
    "PLACE",
    "OPEN_GRIPPER",
    "FINISH",
)

# Segments whose Cartesian path this offline stage can fully compute from
# the board path, the board<->base transform, and unverified height/tool
# config (see synthetic/trajectory/compose.py).
SYNTHESIZABLE_SEGMENTS = frozenset(
    {
        "TRANSFER_ABOVE_BOARD",
        "DESCEND",
        "ERASE",
        "LIFT_FROM_BOARD",
        "OPEN_GRIPPER",
        "FINISH",
    }
)

# Segments that require a real recorded human-teleop demonstration. This
# stage never fabricates their poses; it only validates a caller-supplied
# template's schema (see `compose.require_recorded_template`).
TEMPLATE_REQUIRED_SEGMENTS = frozenset(
    {"PARKING_TO_PREGRASP", "GRASP", "LIFT", "RETURN_ABOVE_ERASER", "PLACE"}
)

assert SYNTHESIZABLE_SEGMENTS | TEMPLATE_REQUIRED_SEGMENTS == set(SEGMENT_ORDER)

GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 1.0

_GRIPPER_CONTINUITY_ATOL = 1e-9


class TrajectoryError(ValueError):
    """Raised when trajectory data or a segment schema is invalid."""


def _as_float_array(values: Any, *, ndim: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != ndim:
        raise TrajectoryError(f"{name} must be a {ndim}D array, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise TrajectoryError(f"{name} contains NaN or Inf")
    return array


@dataclass(frozen=True)
class SegmentTrajectory:
    """One segment's per-frame Cartesian path, gripper state, and timing."""

    segment: str
    frame_index: np.ndarray
    time_s: np.ndarray
    base_xyzrpy: np.ndarray
    gripper: np.ndarray
    fps: float
    board_xy: np.ndarray | None = None

    def validate(self) -> None:
        if self.segment not in SEGMENT_ORDER:
            raise TrajectoryError(
                f"unknown segment {self.segment!r}; must be one of {SEGMENT_ORDER}"
            )
        if self.fps <= 0 or not np.isfinite(self.fps):
            raise TrajectoryError(f"fps must be positive and finite, got {self.fps}")

        frame_index = _as_float_array(self.frame_index, ndim=1, name="frame_index")
        time_s = _as_float_array(self.time_s, ndim=1, name="time_s")
        base_xyzrpy = _as_float_array(self.base_xyzrpy, ndim=2, name="base_xyzrpy")
        gripper = _as_float_array(self.gripper, ndim=1, name="gripper")

        if base_xyzrpy.shape[1] != 6:
            raise TrajectoryError(
                f"base_xyzrpy must have shape (N, 6), got {base_xyzrpy.shape}"
            )
        count = base_xyzrpy.shape[0]
        if count < 1:
            raise TrajectoryError(f"segment {self.segment} must contain at least 1 frame")
        for array, name in (
            (frame_index, "frame_index"),
            (time_s, "time_s"),
            (gripper, "gripper"),
        ):
            if array.shape[0] != count:
                raise TrajectoryError(
                    f"{name} length {array.shape[0]} does not match "
                    f"base_xyzrpy rows {count} for segment {self.segment}"
                )
        if not np.array_equal(frame_index, np.arange(count, dtype=np.float64)):
            raise TrajectoryError(
                f"frame_index must be exactly 0..N-1 within a segment "
                f"({self.segment}), got {frame_index.tolist()}"
            )
        if count > 1 and np.any(np.diff(time_s) <= 0):
            raise TrajectoryError(
                f"time_s must be strictly increasing within segment {self.segment}"
            )
        if np.any((gripper < 0.0) | (gripper > 1.0)):
            raise TrajectoryError(
                f"gripper values must be within [0, 1] in segment {self.segment}"
            )

        if self.board_xy is not None:
            board_xy = _as_float_array(self.board_xy, ndim=2, name="board_xy")
            if board_xy.shape != (count, 2):
                raise TrajectoryError(
                    f"board_xy must have shape ({count}, 2), got {board_xy.shape} "
                    f"for segment {self.segment}"
                )

    def start_pose(self) -> np.ndarray:
        self.validate()
        return np.asarray(self.base_xyzrpy, dtype=np.float64)[0]

    def end_pose(self) -> np.ndarray:
        self.validate()
        return np.asarray(self.base_xyzrpy, dtype=np.float64)[-1]

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "segment": self.segment,
            "fps": float(self.fps),
            "frame_index": np.asarray(self.frame_index, dtype=int).tolist(),
            "time_s": np.asarray(self.time_s, dtype=float).tolist(),
            "base_xyzrpy": np.asarray(self.base_xyzrpy, dtype=float).tolist(),
            "gripper": np.asarray(self.gripper, dtype=float).tolist(),
            "board_xy": (
                np.asarray(self.board_xy, dtype=float).tolist()
                if self.board_xy is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SegmentTrajectory":
        required = ["segment", "fps", "frame_index", "time_s", "base_xyzrpy", "gripper"]
        missing = [key for key in required if key not in payload]
        if missing:
            raise TrajectoryError(f"segment trajectory is missing keys: {missing}")
        board_xy = payload.get("board_xy")
        segment = cls(
            segment=str(payload["segment"]),
            frame_index=np.asarray(payload["frame_index"], dtype=np.float64),
            time_s=np.asarray(payload["time_s"], dtype=np.float64),
            base_xyzrpy=np.asarray(payload["base_xyzrpy"], dtype=np.float64),
            gripper=np.asarray(payload["gripper"], dtype=np.float64),
            fps=float(payload["fps"]),
            board_xy=(
                np.asarray(board_xy, dtype=np.float64) if board_xy is not None else None
            ),
        )
        segment.validate()
        return segment


@dataclass(frozen=True)
class FullTrajectory:
    """An ordered, continuity-checked concatenation of segment trajectories."""

    segments: tuple[SegmentTrajectory, ...]
    seed: int
    status: str = "unverified"

    def validate(
        self,
        *,
        position_tol_mm: float = 1e-6,
        angle_tol_deg: float = 1e-6,
    ) -> None:
        if not self.segments:
            raise TrajectoryError("a full trajectory must contain at least one segment")
        names = [segment.segment for segment in self.segments]
        if len(set(names)) != len(names):
            raise TrajectoryError(f"duplicate segments in trajectory: {names}")
        order_index = {name: index for index, name in enumerate(SEGMENT_ORDER)}
        unknown = [name for name in names if name not in order_index]
        if unknown:
            raise TrajectoryError(f"unknown segment(s): {unknown}")
        indices = [order_index[name] for name in names]
        if indices != sorted(indices):
            raise TrajectoryError(
                f"segments must follow {SEGMENT_ORDER} order, got {names}"
            )

        for segment in self.segments:
            segment.validate()

        fps_values = {segment.fps for segment in self.segments}
        if len(fps_values) != 1:
            raise TrajectoryError(f"all segments must share one fps, got {fps_values}")

        for previous, current in zip(self.segments, self.segments[1:]):
            end_pose = previous.end_pose()
            start_pose = current.start_pose()
            position_error = float(np.linalg.norm(end_pose[:3] - start_pose[:3]))
            angle_error = float(np.max(np.abs(end_pose[3:] - start_pose[3:])))
            if position_error > position_tol_mm:
                raise TrajectoryError(
                    f"position discontinuity of {position_error:.6f}mm between "
                    f"{previous.segment} and {current.segment} "
                    f"(tolerance {position_tol_mm}mm)"
                )
            if angle_error > angle_tol_deg:
                raise TrajectoryError(
                    f"orientation discontinuity of {angle_error:.6f}deg between "
                    f"{previous.segment} and {current.segment} "
                    f"(tolerance {angle_tol_deg}deg)"
                )
            gripper_gap = abs(float(previous.gripper[-1]) - float(current.gripper[0]))
            if gripper_gap > _GRIPPER_CONTINUITY_ATOL:
                raise TrajectoryError(
                    f"gripper state jumps from {previous.gripper[-1]} to "
                    f"{current.gripper[0]} between {previous.segment} and "
                    f"{current.segment}"
                )

    def concatenated_frame_index(self) -> np.ndarray:
        self.validate()
        offset = 0
        chunks = []
        for segment in self.segments:
            frame_index = np.asarray(segment.frame_index, dtype=int)
            chunks.append(frame_index + offset)
            offset += frame_index.shape[0]
        return np.concatenate(chunks)

    def concatenated_time_s(self) -> np.ndarray:
        self.validate()
        offset = 0.0
        chunks = []
        for segment in self.segments:
            time_s = np.asarray(segment.time_s, dtype=float)
            chunks.append(time_s + offset)
            offset = float(chunks[-1][-1]) + 1.0 / segment.fps
        return np.concatenate(chunks)

    def concatenated_base_xyzrpy(self) -> np.ndarray:
        self.validate()
        return np.concatenate(
            [np.asarray(segment.base_xyzrpy, dtype=float) for segment in self.segments],
            axis=0,
        )

    def concatenated_gripper(self) -> np.ndarray:
        self.validate()
        return np.concatenate(
            [np.asarray(segment.gripper, dtype=float) for segment in self.segments]
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "format_version": FORMAT_VERSION,
            "type": FULL_TRAJECTORY_TYPE,
            "status": self.status,
            "seed": int(self.seed),
            "segments": [segment.to_dict() for segment in self.segments],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FullTrajectory":
        if payload.get("type") != FULL_TRAJECTORY_TYPE:
            raise TrajectoryError(
                f"expected type={FULL_TRAJECTORY_TYPE!r}, got {payload.get('type')!r}"
            )
        if payload.get("format_version") != FORMAT_VERSION:
            raise TrajectoryError(
                f"unsupported format_version: {payload.get('format_version')!r}"
            )
        segments = tuple(
            SegmentTrajectory.from_dict(item) for item in payload["segments"]
        )
        trajectory = cls(
            segments=segments,
            seed=int(payload["seed"]),
            status=str(payload.get("status", "unverified")),
        )
        trajectory.validate()
        return trajectory
