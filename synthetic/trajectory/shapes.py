#!/usr/bin/env python3
"""Fixed-size shape outlines in `board_xy` (mm), centered and rotatable.

Each function returns a closed polyline (first point == last point) sampled
approximately evenly by arc length, translated to `center_xy` and rotated by
`rotation_deg`. Shapes never assume a contact height; that is applied later
in `synthetic/trajectory/compose.py`. This module is intentionally
independent from ROS, CAN, LeRobot, and Piper hardware.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from synthetic.trajectory.schema import TrajectoryError
from synthetic.trajectory.timing import resample_polyline_by_arc_length


def _validate_center(center_xy: Any) -> np.ndarray:
    center = np.asarray(center_xy, dtype=np.float64)
    if center.shape != (2,):
        raise TrajectoryError(f"center_xy must have shape (2,), got {center.shape}")
    if not np.isfinite(center).all():
        raise TrajectoryError("center_xy contains NaN or Inf")
    return center


def _validate_positive(value: float, *, name: str) -> float:
    if not np.isfinite(value) or value <= 0:
        raise TrajectoryError(f"{name} must be positive and finite, got {value}")
    return float(value)


def _rotation_matrix_2d(rotation_deg: float) -> np.ndarray:
    if not np.isfinite(rotation_deg):
        raise TrajectoryError(f"rotation_deg must be finite, got {rotation_deg}")
    theta = np.deg2rad(rotation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    return np.asarray([[cos_t, -sin_t], [sin_t, cos_t]])


def _place_polygon(
    local_vertices: np.ndarray,
    center: np.ndarray,
    rotation_deg: float,
) -> np.ndarray:
    rotation = _rotation_matrix_2d(rotation_deg)
    rotated = local_vertices @ rotation.T
    return rotated + center


def fixed_circle_path(
    center_xy: Any,
    *,
    radius_mm: float,
    num_points: int,
    rotation_deg: float = 0.0,
) -> np.ndarray:
    """A closed circular outline of `radius_mm` around `center_xy`, in board mm."""

    center = _validate_center(center_xy)
    radius = _validate_positive(radius_mm, name="radius_mm")
    if num_points < 8:
        raise TrajectoryError(f"num_points must be >=8 for a circle, got {num_points}")
    if not np.isfinite(rotation_deg):
        raise TrajectoryError(f"rotation_deg must be finite, got {rotation_deg}")
    angles = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=True) + np.deg2rad(
        rotation_deg
    )
    local = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    return local + center


def fixed_triangle_path(
    center_xy: Any,
    *,
    circumradius_mm: float,
    num_points: int,
    rotation_deg: float = 0.0,
) -> np.ndarray:
    """A closed equilateral-triangle outline around `center_xy`, in board mm."""

    center = _validate_center(center_xy)
    circumradius = _validate_positive(circumradius_mm, name="circumradius_mm")
    if num_points < 4:
        raise TrajectoryError(f"num_points must be >=4 for a triangle, got {num_points}")
    vertex_angles = np.deg2rad(np.asarray([90.0, 210.0, 330.0]))
    local_vertices = np.stack(
        [circumradius * np.cos(vertex_angles), circumradius * np.sin(vertex_angles)],
        axis=1,
    )
    vertices = _place_polygon(local_vertices, center, rotation_deg)
    return resample_polyline_by_arc_length(vertices, num_points, closed=True)


def fixed_rectangle_path(
    center_xy: Any,
    *,
    width_mm: float,
    height_mm: float,
    num_points: int,
    rotation_deg: float = 0.0,
) -> np.ndarray:
    """A closed rectangle outline around `center_xy`, in board mm."""

    center = _validate_center(center_xy)
    width = _validate_positive(width_mm, name="width_mm")
    height = _validate_positive(height_mm, name="height_mm")
    if num_points < 5:
        raise TrajectoryError(f"num_points must be >=5 for a rectangle, got {num_points}")
    half_w, half_h = width / 2.0, height / 2.0
    local_vertices = np.asarray(
        [
            [-half_w, -half_h],
            [half_w, -half_h],
            [half_w, half_h],
            [-half_w, half_h],
        ]
    )
    vertices = _place_polygon(local_vertices, center, rotation_deg)
    return resample_polyline_by_arc_length(vertices, num_points, closed=True)
