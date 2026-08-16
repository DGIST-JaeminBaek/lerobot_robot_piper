#!/usr/bin/env python3
"""Rigid transform between the board plane and the Piper base frame.

The board plane is embedded in 3D as `board_xyz = (board_x, board_y, 0)`.
Given >=3 non-collinear `board_xy <-> base_xyz` correspondences (both in mm),
this module solves the rotation and translation that best aligns the board
plane to the Piper base frame using the Kabsch/Umeyama SVD method.

This is a rigid transform only: no independent scale factor is fit. Physical
scale must already be present in the input mm measurements. `RigidTransform`
always validates that its rotation is a proper orthonormal matrix
(determinant +1); a manually constructed or loaded matrix that is an
improper reflection (determinant -1) is rejected.

This module is intentionally independent from ROS, CAN, LeRobot, and Piper
hardware. It only handles plain numbers and JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from synthetic.calibration.common import read_json, require_keys, write_json

FORMAT_VERSION = 1
CORRESPONDENCE_TYPE = "board_base_correspondence_set"
TRANSFORM_TYPE = "board_base_transform"
REQUIRED_UNIT = "mm"

_ORTHONORMALITY_ATOL = 1e-6
_DETERMINANT_ATOL = 1e-6
_DUPLICATE_ATOL = 1e-9
_COLLINEARITY_RATIO = 1e-9


class TransformError(ValueError):
    """Raised when board<->base correspondences or a transform are invalid."""


def _as_points(points: Any, *, dims: int, name: str) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != dims:
        raise TransformError(f"{name} must have shape (N, {dims}), got {array.shape}")
    if not np.isfinite(array).all():
        raise TransformError(f"{name} contains NaN or Inf")
    return array


def embed_board_xy(board_xy: Any) -> np.ndarray:
    """Embed 2D board-plane points as 3D points with z=0."""

    points = _as_points(board_xy, dims=2, name="board_xy points")
    zeros = np.zeros((points.shape[0], 1), dtype=np.float64)
    return np.hstack([points, zeros])


def require_mm_units(board_unit: str, base_unit: str) -> None:
    if board_unit != REQUIRED_UNIT or base_unit != REQUIRED_UNIT:
        raise TransformError(
            f"board_unit and base_unit must both be {REQUIRED_UNIT!r} "
            f"(a rigid transform does not fit scale); got "
            f"board_unit={board_unit!r}, base_unit={base_unit!r}"
        )


def reject_duplicate_points(points: np.ndarray, *, name: str) -> None:
    count = points.shape[0]
    for i in range(count):
        for j in range(i + 1, count):
            if np.allclose(points[i], points[j], rtol=0, atol=_DUPLICATE_ATOL):
                raise TransformError(
                    f"{name} contains duplicate points at indices {i} and {j}: "
                    f"{points[i].tolist()}"
                )


def reject_collinear_2d(points_xy: np.ndarray) -> None:
    if points_xy.shape[0] < 3:
        raise TransformError("at least 3 correspondences are required")
    centered = points_xy - points_xy.mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    if singular_values[0] <= 0 or singular_values[1] / singular_values[0] < _COLLINEARITY_RATIO:
        raise TransformError(
            "board_xy correspondences are (near-)collinear; rotation about the "
            "in-plane axis is underdetermined. Provide non-collinear points."
        )


@dataclass(frozen=True)
class RigidTransform:
    """A rotation + translation mapping board-frame points to base-frame points."""

    rotation: np.ndarray
    translation: np.ndarray

    def validate(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise TransformError(f"rotation must have shape (3, 3), got {rotation.shape}")
        if translation.shape != (3,):
            raise TransformError(f"translation must have shape (3,), got {translation.shape}")
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise TransformError("rotation/translation contain NaN or Inf")
        should_be_identity = rotation.T @ rotation
        if not np.allclose(should_be_identity, np.eye(3), atol=_ORTHONORMALITY_ATOL):
            raise TransformError(
                "rotation is not orthonormal (R^T R != I within tolerance)"
            )
        determinant = float(np.linalg.det(rotation))
        if determinant < 0:
            raise TransformError(
                f"rotation determinant is {determinant:.6f} (< 0); this is a "
                "mirror reflection, not a proper rotation"
            )
        if abs(determinant - 1.0) > _DETERMINANT_ATOL:
            raise TransformError(
                f"rotation determinant must be 1.0, got {determinant:.6f}"
            )

    def matrix4(self) -> np.ndarray:
        """4x4 homogeneous transform mapping board-frame -> base-frame."""

        self.validate()
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.translation
        return matrix

    def inverse(self) -> "RigidTransform":
        self.validate()
        rotation_inv = self.rotation.T
        translation_inv = -rotation_inv @ self.translation
        return RigidTransform(rotation=rotation_inv, translation=translation_inv)

    def apply(self, points: Any) -> np.ndarray:
        """Map points from this transform's source frame to its target frame."""

        self.validate()
        array = _as_points(points, dims=3, name="points")
        return (self.rotation @ array.T).T + self.translation

    def plane_normal(self) -> np.ndarray:
        """The board's local +z axis (plane normal), expressed in the target frame."""

        self.validate()
        return self.rotation @ np.asarray([0.0, 0.0, 1.0])

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RigidTransform":
        require_keys(payload, ["rotation", "translation"], context="rigid transform")
        rotation = np.asarray(payload["rotation"], dtype=np.float64)
        translation = np.asarray(payload["translation"], dtype=np.float64)
        transform = cls(rotation=rotation, translation=translation)
        transform.validate()
        return transform


def solve_rigid_transform(
    board_xyz: Any,
    base_xyz: Any,
) -> RigidTransform:
    """Solve board-frame -> base-frame rotation/translation via Kabsch/SVD.

    Requires >=3 non-collinear, non-duplicate correspondences. Always returns
    a validated proper rotation (never an improper/reflected matrix).
    """

    board = _as_points(board_xyz, dims=3, name="board points")
    base = _as_points(base_xyz, dims=3, name="base points")
    if board.shape[0] != base.shape[0]:
        raise TransformError(
            f"board and base point counts must match, got {board.shape[0]} "
            f"and {base.shape[0]}"
        )
    if board.shape[0] < 3:
        raise TransformError("at least 3 correspondences are required")
    reject_duplicate_points(board, name="board points")
    reject_duplicate_points(base, name="base points")
    reject_collinear_2d(board[:, :2])

    board_centroid = board.mean(axis=0)
    base_centroid = base.mean(axis=0)
    board_centered = board - board_centroid
    base_centered = base - base_centroid

    covariance = board_centered.T @ base_centered
    u, _, vt = np.linalg.svd(covariance)
    # `board_xyz` is always embedded at z=0, so `covariance` always has rank
    # <=2 (its third row is exactly zero): the sign of the rotation's
    # out-of-plane axis is never determined by the data, and raw SVD output
    # can land on either a proper or an improper matrix depending on
    # implementation-internal sign choices in that null direction. The
    # standard Kabsch/Umeyama correction picks the proper-rotation member of
    # that pair, which fits the (rank-2) correspondences identically either
    # way -- this is not hiding a real reflection, just resolving an
    # otherwise-arbitrary sign in an unconstrained direction.
    sign = float(np.sign(np.linalg.det(vt.T @ u.T))) or 1.0
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    translation = base_centroid - rotation @ board_centroid

    transform = RigidTransform(rotation=rotation, translation=translation)
    transform.validate()
    return transform


def residual_statistics(
    transform: RigidTransform,
    board_xyz: Any,
    base_xyz: Any,
) -> dict[str, Any]:
    """Per-point and summary mm error between transform(board) and base."""

    board = _as_points(board_xyz, dims=3, name="board points")
    base = _as_points(base_xyz, dims=3, name="base points")
    predicted = transform.apply(board)
    errors = np.linalg.norm(predicted - base, axis=1)
    return {
        "per_point_mm": errors.tolist(),
        "mean_mm": float(errors.mean()),
        "max_mm": float(errors.max()),
    }


def parse_correspondences(
    payload: dict[str, Any],
) -> tuple[list[str], np.ndarray, np.ndarray, str, str]:
    """Parse a `board_base_correspondence_set` payload.

    Returns (names, board_xy (N,2), base_xyz (N,3), board_unit, base_unit).
    """

    require_keys(
        payload,
        ["format_version", "type", "board_unit", "base_unit", "correspondences"],
        context="board-base correspondences",
    )
    if payload["format_version"] != FORMAT_VERSION:
        raise TransformError(
            f"unsupported format_version: {payload['format_version']}"
        )
    if payload["type"] != CORRESPONDENCE_TYPE:
        raise TransformError(
            f"expected type={CORRESPONDENCE_TYPE!r}, got {payload['type']!r}"
        )
    board_unit = str(payload["board_unit"])
    base_unit = str(payload["base_unit"])
    require_mm_units(board_unit, base_unit)

    records = payload["correspondences"]
    if not isinstance(records, list) or len(records) < 3:
        raise TransformError(
            "correspondences must be a list of at least 3 points, got "
            f"{records!r}"
        )
    names: list[str] = []
    board_xy: list[list[float]] = []
    base_xyz: list[list[float]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TransformError(f"correspondence {index} must be an object")
        require_keys(
            record, ["name", "board_xy", "base_xyz"], context=f"correspondence {index}"
        )
        names.append(str(record["name"]))
        board_xy.append(list(record["board_xy"]))
        base_xyz.append(list(record["base_xyz"]))

    if len(set(names)) != len(names):
        raise TransformError(f"correspondence names must be unique, got {names}")

    board_xy_array = _as_points(board_xy, dims=2, name="board_xy points")
    base_xyz_array = _as_points(base_xyz, dims=3, name="base_xyz points")
    return names, board_xy_array, base_xyz_array, board_unit, base_unit


def make_correspondence_records(
    names: list[str],
    board_xy: np.ndarray,
    base_xyz: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "board_xy": point_xy.tolist(),
            "base_xyz": point_xyz.tolist(),
        }
        for name, point_xy, point_xyz in zip(names, board_xy, base_xyz, strict=True)
    ]


__all__ = [
    "FORMAT_VERSION",
    "CORRESPONDENCE_TYPE",
    "TRANSFORM_TYPE",
    "REQUIRED_UNIT",
    "TransformError",
    "RigidTransform",
    "embed_board_xy",
    "require_mm_units",
    "reject_duplicate_points",
    "reject_collinear_2d",
    "solve_rigid_transform",
    "residual_statistics",
    "parse_correspondences",
    "make_correspondence_records",
    "read_json",
    "write_json",
]
