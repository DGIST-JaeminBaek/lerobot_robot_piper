#!/usr/bin/env python3
"""Compose board-plane shape paths into Cartesian erase-task segments.

This module bridges `synthetic/transforms/board_base.py` (board<->base rigid
transform) and `synthetic/trajectory/schema.py` (segment/full-trajectory
containers). It fully synthesizes the segments that only depend on the
board path, an (unverified) board<->base transform, and (unverified)
hover/contact height and tool-orientation config: `TRANSFER_ABOVE_BOARD`,
`DESCEND`, `ERASE`, `LIFT_FROM_BOARD`, `OPEN_GRIPPER`, `FINISH`.

It never fabricates `PARKING_TO_PREGRASP`/`GRASP`/`LIFT`/
`RETURN_ABOVE_ERASER`/`PLACE`: those require a real recorded human-teleop
demonstration, so this module only validates a caller-supplied template's
schema (`require_recorded_template`).

This module is intentionally independent from ROS, CAN, LeRobot, and Piper
hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from synthetic.transforms.board_base import RigidTransform
from synthetic.trajectory.schema import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    TEMPLATE_REQUIRED_SEGMENTS,
    SEGMENT_ORDER,
    FullTrajectory,
    SegmentTrajectory,
    TrajectoryError,
)
from synthetic.trajectory.timing import (
    frame_and_time,
    interpolate_pose_linear,
    polyline_length,
    resample_polyline_by_arc_length,
    sample_count_for_distance,
)


@dataclass(frozen=True)
class BoardMotionConfig:
    """Unverified height/orientation/speed config for board-plane motion.

    `hover_height_mm` and `contact_height_mm` are measured along the board's
    own +z (plane normal) axis, not the base frame's z axis. `tool_rpy_deg`
    is the fixed EEF orientation held throughout transfer/descend/erase/lift.
    """

    hover_height_mm: float
    contact_height_mm: float
    tool_rpy_deg: tuple[float, float, float]
    fps: float
    transfer_speed_mm_per_s: float
    descend_lift_speed_mm_per_s: float
    erase_speed_mm_per_s: float
    status: str = "unverified"

    def validate(self) -> None:
        for value, name in (
            (self.hover_height_mm, "hover_height_mm"),
            (self.contact_height_mm, "contact_height_mm"),
        ):
            if not np.isfinite(value):
                raise TrajectoryError(f"{name} must be finite, got {value}")
        if self.hover_height_mm <= self.contact_height_mm:
            raise TrajectoryError(
                "hover_height_mm must be greater than contact_height_mm, got "
                f"hover={self.hover_height_mm}, contact={self.contact_height_mm}"
            )
        rpy = np.asarray(self.tool_rpy_deg, dtype=np.float64)
        if rpy.shape != (3,) or not np.isfinite(rpy).all():
            raise TrajectoryError(
                f"tool_rpy_deg must be 3 finite values, got {self.tool_rpy_deg}"
            )
        for value, name in (
            (self.fps, "fps"),
            (self.transfer_speed_mm_per_s, "transfer_speed_mm_per_s"),
            (self.descend_lift_speed_mm_per_s, "descend_lift_speed_mm_per_s"),
            (self.erase_speed_mm_per_s, "erase_speed_mm_per_s"),
        ):
            if not np.isfinite(value) or value <= 0:
                raise TrajectoryError(f"{name} must be positive and finite, got {value}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "status": self.status,
            "hover_height_mm": float(self.hover_height_mm),
            "contact_height_mm": float(self.contact_height_mm),
            "tool_rpy_deg": [float(v) for v in self.tool_rpy_deg],
            "fps": float(self.fps),
            "transfer_speed_mm_per_s": float(self.transfer_speed_mm_per_s),
            "descend_lift_speed_mm_per_s": float(self.descend_lift_speed_mm_per_s),
            "erase_speed_mm_per_s": float(self.erase_speed_mm_per_s),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BoardMotionConfig":
        required = [
            "hover_height_mm",
            "contact_height_mm",
            "tool_rpy_deg",
            "fps",
            "transfer_speed_mm_per_s",
            "descend_lift_speed_mm_per_s",
            "erase_speed_mm_per_s",
        ]
        missing = [key for key in required if key not in payload]
        if missing:
            raise TrajectoryError(f"board motion config is missing keys: {missing}")
        config = cls(
            hover_height_mm=float(payload["hover_height_mm"]),
            contact_height_mm=float(payload["contact_height_mm"]),
            tool_rpy_deg=tuple(float(v) for v in payload["tool_rpy_deg"]),
            fps=float(payload["fps"]),
            transfer_speed_mm_per_s=float(payload["transfer_speed_mm_per_s"]),
            descend_lift_speed_mm_per_s=float(payload["descend_lift_speed_mm_per_s"]),
            erase_speed_mm_per_s=float(payload["erase_speed_mm_per_s"]),
            status=str(payload.get("status", "unverified")),
        )
        config.validate()
        return config


def board_point_to_base_pose(
    board_xy: Any,
    height_mm: float,
    *,
    board_to_base: RigidTransform,
    tool_rpy_deg: Any,
) -> np.ndarray:
    """A single `board_xy` point at `height_mm` (board z), as a base_xyzrpy pose."""

    xy = np.asarray(board_xy, dtype=np.float64)
    if xy.shape != (2,):
        raise TrajectoryError(f"board_xy must have shape (2,), got {xy.shape}")
    rpy = np.asarray(tool_rpy_deg, dtype=np.float64)
    if rpy.shape != (3,):
        raise TrajectoryError(f"tool_rpy_deg must have shape (3,), got {rpy.shape}")
    board_point = np.asarray([xy[0], xy[1], height_mm], dtype=np.float64).reshape(1, 3)
    base_xyz = board_to_base.apply(board_point)[0]
    return np.concatenate([base_xyz, rpy])


def board_path_to_base_poses(
    board_xy_path: Any,
    height_mm: float,
    *,
    board_to_base: RigidTransform,
    tool_rpy_deg: Any,
) -> np.ndarray:
    """A `(N, 2)` board_xy path at `height_mm`, as `(N, 6)` base_xyzrpy poses."""

    path = np.asarray(board_xy_path, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2:
        raise TrajectoryError(f"board_xy_path must have shape (N, 2), got {path.shape}")
    rpy = np.asarray(tool_rpy_deg, dtype=np.float64)
    if rpy.shape != (3,):
        raise TrajectoryError(f"tool_rpy_deg must have shape (3,), got {rpy.shape}")
    heights = np.full((path.shape[0], 1), float(height_mm))
    board_points = np.hstack([path, heights])
    base_xyz = board_to_base.apply(board_points)
    return np.hstack([base_xyz, np.tile(rpy, (path.shape[0], 1))])


def assert_hover_clearance(
    segment: SegmentTrajectory,
    *,
    board_to_base: RigidTransform,
    hover_height_mm: float,
    tolerance_mm: float = 1e-6,
) -> None:
    """Raise unless every pose in `segment` is at/above `hover_height_mm`.

    Height is measured along the board's own +z axis (plane normal), by
    mapping base_xyzrpy positions back into the board frame.
    """

    segment.validate()
    board_from_base = board_to_base.inverse()
    base_points = np.asarray(segment.base_xyzrpy, dtype=np.float64)[:, :3]
    board_points = board_from_base.apply(base_points)
    heights = board_points[:, 2]
    if np.any(heights < hover_height_mm - tolerance_mm):
        raise TrajectoryError(
            f"segment {segment.segment} dips below hover height "
            f"{hover_height_mm}mm (min board-frame z={heights.min():.6f}mm)"
        )


def build_transfer_segment(
    segment_name: str,
    pose_start: Any,
    pose_end: Any,
    *,
    config: BoardMotionConfig,
    gripper_value: float = GRIPPER_CLOSED,
) -> SegmentTrajectory:
    """A linear Cartesian transit segment between two explicit poses.

    `pose_start`/`pose_end` must be supplied by the caller (e.g. the end of a
    recorded pick-up template, or a hover pose derived from the board path);
    this function never invents them.
    """

    config.validate()
    if segment_name not in SEGMENT_ORDER:
        raise TrajectoryError(f"unknown segment {segment_name!r}")
    start = np.asarray(pose_start, dtype=np.float64)
    end = np.asarray(pose_end, dtype=np.float64)
    distance_mm = float(np.linalg.norm(end[:3] - start[:3]))
    num_samples = sample_count_for_distance(
        distance_mm, fps=config.fps, speed_mm_per_s=config.transfer_speed_mm_per_s
    )
    frame_index, time_s = frame_and_time(num_samples, config.fps)
    base_xyzrpy = interpolate_pose_linear(start, end, num_samples)
    gripper = np.full(num_samples, float(gripper_value))
    return SegmentTrajectory(
        segment=segment_name,
        frame_index=frame_index,
        time_s=time_s,
        base_xyzrpy=base_xyzrpy,
        gripper=gripper,
        fps=config.fps,
    )


def build_descend_erase_lift(
    board_xy_path: Any,
    *,
    board_to_base: RigidTransform,
    config: BoardMotionConfig,
    gripper_value: float = GRIPPER_CLOSED,
) -> tuple[SegmentTrajectory, SegmentTrajectory, SegmentTrajectory]:
    """Build `(DESCEND, ERASE, LIFT_FROM_BOARD)` from a closed board_xy path."""

    config.validate()
    path = np.asarray(board_xy_path, dtype=np.float64)
    if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] != 2:
        raise TrajectoryError(f"board_xy_path must have shape (N>=2, 2), got {path.shape}")

    start_xy = path[0]
    end_xy = path[-1]

    descend_start = board_point_to_base_pose(
        start_xy,
        config.hover_height_mm,
        board_to_base=board_to_base,
        tool_rpy_deg=config.tool_rpy_deg,
    )
    descend_end = board_point_to_base_pose(
        start_xy,
        config.contact_height_mm,
        board_to_base=board_to_base,
        tool_rpy_deg=config.tool_rpy_deg,
    )
    descend_distance = config.hover_height_mm - config.contact_height_mm
    descend_samples = sample_count_for_distance(
        descend_distance, fps=config.fps, speed_mm_per_s=config.descend_lift_speed_mm_per_s
    )
    descend_frame_index, descend_time_s = frame_and_time(descend_samples, config.fps)
    descend = SegmentTrajectory(
        segment="DESCEND",
        frame_index=descend_frame_index,
        time_s=descend_time_s,
        base_xyzrpy=interpolate_pose_linear(descend_start, descend_end, descend_samples),
        gripper=np.full(descend_samples, float(gripper_value)),
        fps=config.fps,
    )

    erase_length_mm = polyline_length(path, closed=False)
    erase_samples = sample_count_for_distance(
        erase_length_mm, fps=config.fps, speed_mm_per_s=config.erase_speed_mm_per_s
    )
    erase_path = resample_polyline_by_arc_length(path, erase_samples, closed=False)
    erase_frame_index, erase_time_s = frame_and_time(erase_samples, config.fps)
    erase = SegmentTrajectory(
        segment="ERASE",
        frame_index=erase_frame_index,
        time_s=erase_time_s,
        base_xyzrpy=board_path_to_base_poses(
            erase_path,
            config.contact_height_mm,
            board_to_base=board_to_base,
            tool_rpy_deg=config.tool_rpy_deg,
        ),
        gripper=np.full(erase_samples, float(gripper_value)),
        fps=config.fps,
        board_xy=erase_path,
    )

    lift_start = board_point_to_base_pose(
        end_xy,
        config.contact_height_mm,
        board_to_base=board_to_base,
        tool_rpy_deg=config.tool_rpy_deg,
    )
    lift_end = board_point_to_base_pose(
        end_xy,
        config.hover_height_mm,
        board_to_base=board_to_base,
        tool_rpy_deg=config.tool_rpy_deg,
    )
    lift_samples = descend_samples
    lift_frame_index, lift_time_s = frame_and_time(lift_samples, config.fps)
    lift = SegmentTrajectory(
        segment="LIFT_FROM_BOARD",
        frame_index=lift_frame_index,
        time_s=lift_time_s,
        base_xyzrpy=interpolate_pose_linear(lift_start, lift_end, lift_samples),
        gripper=np.full(lift_samples, float(gripper_value)),
        fps=config.fps,
    )

    return descend, erase, lift


def build_open_gripper_segment(
    pose: Any,
    *,
    fps: float,
    duration_s: float,
) -> SegmentTrajectory:
    """Hold `pose` fixed while ramping the gripper from closed to open."""

    if duration_s <= 0 or not np.isfinite(duration_s):
        raise TrajectoryError(f"duration_s must be positive and finite, got {duration_s}")
    pose_array = np.asarray(pose, dtype=np.float64)
    if pose_array.shape != (6,):
        raise TrajectoryError(f"pose must have shape (6,), got {pose_array.shape}")
    num_samples = max(2, int(np.ceil(duration_s * fps)) + 1)
    frame_index, time_s = frame_and_time(num_samples, fps)
    base_xyzrpy = np.tile(pose_array, (num_samples, 1))
    gripper = np.linspace(GRIPPER_CLOSED, GRIPPER_OPEN, num_samples)
    return SegmentTrajectory(
        segment="OPEN_GRIPPER",
        frame_index=frame_index,
        time_s=time_s,
        base_xyzrpy=base_xyzrpy,
        gripper=gripper,
        fps=fps,
    )


def build_finish_segment(pose: Any, *, fps: float) -> SegmentTrajectory:
    """A single-frame marker segment at `pose` with the gripper open."""

    pose_array = np.asarray(pose, dtype=np.float64)
    if pose_array.shape != (6,):
        raise TrajectoryError(f"pose must have shape (6,), got {pose_array.shape}")
    frame_index, time_s = frame_and_time(1, fps)
    return SegmentTrajectory(
        segment="FINISH",
        frame_index=frame_index,
        time_s=time_s,
        base_xyzrpy=pose_array.reshape(1, 6),
        gripper=np.asarray([GRIPPER_OPEN]),
        fps=fps,
    )


def require_recorded_template(
    segment_name: str,
    template: SegmentTrajectory | None,
) -> SegmentTrajectory:
    """Validate an externally recorded template for a pick-up/return segment.

    This stage never fabricates `PARKING_TO_PREGRASP`/`GRASP`/`LIFT`/
    `RETURN_ABOVE_ERASER`/`PLACE` poses; it only validates a caller-supplied
    template's schema and segment identity.
    """

    if segment_name not in TEMPLATE_REQUIRED_SEGMENTS:
        raise TrajectoryError(
            f"{segment_name!r} is not a template-required segment; expected "
            f"one of {sorted(TEMPLATE_REQUIRED_SEGMENTS)}"
        )
    if template is None:
        raise TrajectoryError(
            f"segment {segment_name!r} requires a recorded human-teleop "
            "template; none was provided. This offline stage does not "
            "fabricate pick-up/return poses."
        )
    template.validate()
    if template.segment != segment_name:
        raise TrajectoryError(
            f"template segment {template.segment!r} does not match requested "
            f"{segment_name!r}"
        )
    return template


def compose_full_trajectory(
    segments_by_name: dict[str, SegmentTrajectory],
    *,
    seed: int,
) -> FullTrajectory:
    """Order and continuity-check a complete set of the 11 task segments."""

    missing = [name for name in SEGMENT_ORDER if name not in segments_by_name]
    if missing:
        raise TrajectoryError(f"missing segments: {missing}")
    ordered = tuple(segments_by_name[name] for name in SEGMENT_ORDER)
    trajectory = FullTrajectory(segments=ordered, seed=seed)
    trajectory.validate()
    return trajectory
