#!/usr/bin/env python3
"""Sample-count, timing, and simple polyline/pose interpolation helpers.

These are placeholder Cartesian-space helpers: they do not check reachability
or joint continuity (that happens in the IK stage). This module is
intentionally independent from ROS, CAN, LeRobot, and Piper hardware.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from synthetic.trajectory.schema import TrajectoryError


def sample_count_for_distance(
    distance_mm: float,
    *,
    fps: float,
    speed_mm_per_s: float,
) -> int:
    """Number of samples to cover `distance_mm` at `speed_mm_per_s` and `fps`."""

    if distance_mm < 0 or not np.isfinite(distance_mm):
        raise TrajectoryError(f"distance_mm must be >=0 and finite, got {distance_mm}")
    if fps <= 0 or not np.isfinite(fps):
        raise TrajectoryError(f"fps must be positive and finite, got {fps}")
    if speed_mm_per_s <= 0 or not np.isfinite(speed_mm_per_s):
        raise TrajectoryError(
            f"speed_mm_per_s must be positive and finite, got {speed_mm_per_s}"
        )
    duration_s = distance_mm / speed_mm_per_s
    return max(2, int(np.ceil(duration_s * fps)) + 1)


def frame_and_time(num_samples: int, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """`(frame_index, time_s)` for `num_samples` frames at `fps`, both 0-based."""

    if num_samples < 1:
        raise TrajectoryError(f"num_samples must be >=1, got {num_samples}")
    if fps <= 0 or not np.isfinite(fps):
        raise TrajectoryError(f"fps must be positive and finite, got {fps}")
    frame_index = np.arange(num_samples)
    time_s = frame_index / fps
    return frame_index, time_s


def polyline_length(points: Any, *, closed: bool = False) -> float:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2:
        raise TrajectoryError(f"points must have shape (N>=2, D), got {array.shape}")
    if not np.isfinite(array).all():
        raise TrajectoryError("points contain NaN or Inf")
    segment_vectors = np.diff(array, axis=0)
    length = float(np.linalg.norm(segment_vectors, axis=1).sum())
    if closed:
        length += float(np.linalg.norm(array[0] - array[-1]))
    return length


def resample_polyline_by_arc_length(
    points: Any,
    num_samples: int,
    *,
    closed: bool = False,
) -> np.ndarray:
    """Resample a polyline to `num_samples` points, evenly spaced by arc length."""

    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2:
        raise TrajectoryError(f"points must have shape (N>=2, D), got {array.shape}")
    if not np.isfinite(array).all():
        raise TrajectoryError("points contain NaN or Inf")
    if num_samples < 2:
        raise TrajectoryError(f"num_samples must be >=2, got {num_samples}")

    vertices = np.vstack([array, array[0:1]]) if closed else array
    deltas = np.diff(vertices, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    if np.any(segment_lengths <= 1e-12):
        raise TrajectoryError(
            "polyline contains a zero-length or duplicate consecutive point"
        )
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = cumulative[-1]
    targets = np.linspace(0.0, total_length, num_samples)

    segment_of_target = np.clip(
        np.searchsorted(cumulative, targets, side="right") - 1,
        0,
        len(segment_lengths) - 1,
    )
    segment_start_len = cumulative[segment_of_target]
    local_fraction = (targets - segment_start_len) / segment_lengths[segment_of_target]
    return vertices[segment_of_target] + local_fraction[:, None] * deltas[segment_of_target]


def interpolate_pose_linear(
    pose_start: Any,
    pose_end: Any,
    num_samples: int,
) -> np.ndarray:
    """Linearly interpolate a `(6,)` base_xyzrpy pose from start to end."""

    start = np.asarray(pose_start, dtype=np.float64)
    end = np.asarray(pose_end, dtype=np.float64)
    if start.shape != (6,) or end.shape != (6,):
        raise TrajectoryError(
            f"pose_start/pose_end must have shape (6,), got {start.shape} "
            f"and {end.shape}"
        )
    if not np.isfinite(start).all() or not np.isfinite(end).all():
        raise TrajectoryError("pose_start/pose_end contain NaN or Inf")
    if num_samples < 2:
        raise TrajectoryError(f"num_samples must be >=2, got {num_samples}")
    fractions = np.linspace(0.0, 1.0, num_samples).reshape(-1, 1)
    return start.reshape(1, -1) * (1.0 - fractions) + end.reshape(1, -1) * fractions
