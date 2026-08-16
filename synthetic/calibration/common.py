#!/usr/bin/env python3
"""Shared data validation and homography functions.

This module is intentionally independent from ROS, CAN, LeRobot, and Piper
hardware. It only handles images, numeric coordinates, JSON, and homographies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


FORMAT_VERSION = 1
POINT_SELECTION_TYPE = "board_point_selection"
CALIBRATION_TYPE = "board_homography_calibration"
POINT_NAMES = ("top_left", "top_right", "bottom_right", "bottom_left")


class CalibrationError(ValueError):
    """Raised when calibration input is malformed or geometrically invalid."""


@dataclass(frozen=True)
class BoardSpec:
    width: float
    height: float
    unit: str

    def validate(self) -> None:
        if self.unit not in {"normalized", "mm"}:
            raise CalibrationError(
                f"board unit must be 'normalized' or 'mm', got {self.unit!r}"
            )
        if not np.isfinite([self.width, self.height]).all():
            raise CalibrationError("board width and height must be finite")
        if self.width <= 0 or self.height <= 0:
            raise CalibrationError("board width and height must be positive")
        if self.unit == "normalized" and not np.allclose(
            [self.width, self.height], [1.0, 1.0]
        ):
            raise CalibrationError(
                "normalized board coordinates require width=1.0 and height=1.0"
            )

    def corner_coordinates(self) -> np.ndarray:
        self.validate()
        return np.asarray(
            [
                [0.0, 0.0],
                [self.width, 0.0],
                [self.width, self.height],
                [0.0, self.height],
            ],
            dtype=np.float64,
        )


def parse_xy(text: str) -> tuple[float, float]:
    values = [part.strip() for part in text.split(",")]
    if len(values) != 2:
        raise CalibrationError(f"expected x,y, got {text!r}")
    try:
        point = (float(values[0]), float(values[1]))
    except ValueError as exc:
        raise CalibrationError(f"expected numeric x,y, got {text!r}") from exc
    if not np.isfinite(point).all():
        raise CalibrationError(f"point must be finite, got {text!r}")
    return point


def parse_point_list(text: str) -> np.ndarray:
    """Parse `x,y;x,y;x,y;x,y` in TL, TR, BR, BL order."""

    chunks = [chunk.strip() for chunk in text.split(";") if chunk.strip()]
    if len(chunks) != 4:
        raise CalibrationError(
            "exactly four image points are required in "
            "top-left;top-right;bottom-right;bottom-left order"
        )
    return np.asarray([parse_xy(chunk) for chunk in chunks], dtype=np.float64)


def _as_points(points: Any, *, name: str, exact_count: int | None = None) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise CalibrationError(f"{name} must have shape (N, 2), got {array.shape}")
    if exact_count is not None and array.shape[0] != exact_count:
        raise CalibrationError(
            f"{name} must contain {exact_count} points, got {array.shape[0]}"
        )
    if array.shape[0] < 1:
        raise CalibrationError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise CalibrationError(f"{name} contains NaN or Inf")
    return array


def validate_image_quad(points: Any, width: int, height: int) -> np.ndarray:
    quad = _as_points(points, name="image points", exact_count=4)
    if width <= 0 or height <= 0:
        raise CalibrationError("image width and height must be positive")
    if (
        (quad[:, 0] < 0).any()
        or (quad[:, 0] >= width).any()
        or (quad[:, 1] < 0).any()
        or (quad[:, 1] >= height).any()
    ):
        raise CalibrationError(
            f"image points must stay inside 0<=x<{width}, 0<=y<{height}"
        )
    rounded = np.round(quad, decimals=9)
    if np.unique(rounded, axis=0).shape[0] != 4:
        raise CalibrationError("image points must be distinct")
    contour = quad.astype(np.float32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour):
        raise CalibrationError(
            "image points must form a convex quadrilateral in TL, TR, BR, BL order"
        )
    area = abs(float(cv2.contourArea(contour)))
    if area < 100.0:
        raise CalibrationError(
            f"selected quadrilateral is too small or degenerate: area={area:.3f}px^2"
        )
    return quad


def compute_homography(
    image_points: Any,
    board_points: Any,
) -> tuple[np.ndarray, np.ndarray]:
    image = _as_points(image_points, name="image points", exact_count=4)
    board = _as_points(board_points, name="board points", exact_count=4)
    matrix = cv2.getPerspectiveTransform(
        image.astype(np.float32),
        board.astype(np.float32),
    ).astype(np.float64)
    if not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-12:
        raise CalibrationError("homography is singular or non-finite")
    inverse = np.linalg.inv(matrix)
    matrix /= matrix[2, 2]
    inverse /= inverse[2, 2]
    return matrix, inverse


def project_points(points: Any, matrix: Any) -> np.ndarray:
    source = _as_points(points, name="points")
    homography = np.asarray(matrix, dtype=np.float64)
    if homography.shape != (3, 3):
        raise CalibrationError(
            f"homography must have shape (3, 3), got {homography.shape}"
        )
    if not np.isfinite(homography).all():
        raise CalibrationError("homography contains NaN or Inf")
    projected = cv2.perspectiveTransform(
        source.reshape(1, -1, 2),
        homography,
    ).reshape(-1, 2)
    if not np.isfinite(projected).all():
        raise CalibrationError("projected coordinates contain NaN or Inf")
    return projected


def reprojection_statistics(
    image_points: Any,
    board_points: Any,
    image_to_board: Any,
    board_to_image: Any,
) -> dict[str, float | list[float]]:
    image = _as_points(image_points, name="image points", exact_count=4)
    board = _as_points(board_points, name="board points", exact_count=4)
    board_predicted = project_points(image, image_to_board)
    image_predicted = project_points(board, board_to_image)
    board_errors = np.linalg.norm(board_predicted - board, axis=1)
    image_errors = np.linalg.norm(image_predicted - image, axis=1)
    return {
        "board_error_per_point": board_errors.tolist(),
        "board_error_mean": float(board_errors.mean()),
        "board_error_max": float(board_errors.max()),
        "image_error_px_per_point": image_errors.tolist(),
        "image_error_px_mean": float(image_errors.mean()),
        "image_error_px_max": float(image_errors.max()),
    }


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CalibrationError(f"file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CalibrationError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CalibrationError(f"top-level JSON value must be an object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise CalibrationError(
            f"output already exists: {path}; pass --overwrite to replace it"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_source_frame(
    *,
    image_path: Path | None,
    video_path: Path | None,
    frame_index: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if (image_path is None) == (video_path is None):
        raise CalibrationError("provide exactly one of --image or --video")

    if image_path is not None:
        resolved = image_path.expanduser().resolve()
        frame = cv2.imread(str(resolved), cv2.IMREAD_COLOR)
        if frame is None:
            raise CalibrationError(f"could not read image: {resolved}")
        source = {
            "kind": "image",
            "path": str(resolved),
            "frame_index": None,
        }
    else:
        assert video_path is not None
        if frame_index < 0:
            raise CalibrationError("--frame must be zero or greater")
        resolved = video_path.expanduser().resolve()
        capture = cv2.VideoCapture(str(resolved))
        if not capture.isOpened():
            raise CalibrationError(f"could not open video: {resolved}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        actual_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        capture.release()
        if not ok or frame is None:
            raise CalibrationError(
                f"could not decode frame {frame_index} from {resolved}"
            )
        if actual_index != frame_index:
            raise CalibrationError(
                f"decoder returned frame {actual_index}, expected {frame_index}"
            )
        source = {
            "kind": "video",
            "path": str(resolved),
            "frame_index": frame_index,
        }

    height, width = frame.shape[:2]
    source["width"] = int(width)
    source["height"] = int(height)
    return frame, source


def make_correspondences(
    image_points: Any,
    board: BoardSpec,
) -> list[dict[str, Any]]:
    image = _as_points(image_points, name="image points", exact_count=4)
    board_points = board.corner_coordinates()
    return [
        {
            "name": name,
            "image_px": image_point.tolist(),
            "board_xy": board_point.tolist(),
        }
        for name, image_point, board_point in zip(
            POINT_NAMES,
            image,
            board_points,
            strict=True,
        )
    ]


def unpack_correspondences(
    payload: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    records = payload.get("correspondences")
    if not isinstance(records, list) or len(records) != 4:
        raise CalibrationError("correspondences must be a list of four points")
    names = [record.get("name") for record in records if isinstance(record, dict)]
    if names != list(POINT_NAMES):
        raise CalibrationError(
            f"correspondence order must be {list(POINT_NAMES)}, got {names}"
        )
    try:
        image = [record["image_px"] for record in records]
        board = [record["board_xy"] for record in records]
    except (KeyError, TypeError) as exc:
        raise CalibrationError(
            "each correspondence requires image_px and board_xy"
        ) from exc
    return (
        _as_points(image, name="image points", exact_count=4),
        _as_points(board, name="board points", exact_count=4),
    )


def require_keys(payload: dict[str, Any], keys: Iterable[str], *, context: str) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise CalibrationError(f"{context} is missing required keys: {missing}")

