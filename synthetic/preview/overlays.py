#!/usr/bin/env python3
"""Draw the generated board erase path onto the raw TOP frame and the model-input frame.

This is a static image overlay for human review; it never opens a live
camera or RViz.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from synthetic.calibration.common import project_points
from synthetic.preprocessing.image_transform import raw_to_model_points, transform_image
from synthetic.preprocessing.profiles import ImageProfile

_PATH_COLOR = (0, 255, 255)
_OUT_OF_MODEL_COLOR = (0, 0, 255)
_START_COLOR = (0, 255, 0)
_END_COLOR = (255, 0, 255)


def _draw_path(canvas: np.ndarray, points: np.ndarray, *, color: tuple[int, int, int]) -> None:
    rounded = np.rint(points).astype(int)
    for start, end in zip(rounded[:-1], rounded[1:], strict=True):
        cv2.line(canvas, tuple(start), tuple(end), color, 2, cv2.LINE_AA)


def draw_raw_board_overlay(
    frame: np.ndarray,
    board_xy_path: Any,
    *,
    board_to_image_homography: Any,
) -> np.ndarray:
    """Draw the erase path (projected via the board<->image homography) on the raw frame."""

    board_xy = np.asarray(board_xy_path, dtype=np.float64)
    image_px = project_points(board_xy, board_to_image_homography)
    canvas = frame.copy()
    _draw_path(canvas, image_px, color=_PATH_COLOR)
    cv2.circle(canvas, tuple(np.rint(image_px[0]).astype(int)), 6, _START_COLOR, -1, cv2.LINE_AA)
    cv2.circle(canvas, tuple(np.rint(image_px[-1]).astype(int)), 6, _END_COLOR, -1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "unverified calibration - not for real execution",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def draw_model_input_overlay(
    frame: np.ndarray,
    board_xy_path: Any,
    *,
    board_to_image_homography: Any,
    image_profile: ImageProfile,
) -> np.ndarray:
    """Apply `image_profile` to `frame`, then draw the erase path in model-pixel space.

    Points that fall outside the profile's crop are drawn in red past that
    point, so a path that leaves the model's field of view is visible at a
    glance.
    """

    board_xy = np.asarray(board_xy_path, dtype=np.float64)
    image_px = project_points(board_xy, board_to_image_homography)
    model_image = transform_image(frame, image_profile)
    model_px = raw_to_model_points(image_px, image_profile, strict=False)

    canvas = model_image.copy()
    geometry_ok = (
        (model_px[:, 0] >= 0)
        & (model_px[:, 0] < model_image.shape[1])
        & (model_px[:, 1] >= 0)
        & (model_px[:, 1] < model_image.shape[0])
    )
    rounded = np.rint(model_px).astype(int)
    for index in range(len(rounded) - 1):
        color = _PATH_COLOR if (geometry_ok[index] and geometry_ok[index + 1]) else _OUT_OF_MODEL_COLOR
        cv2.line(canvas, tuple(rounded[index]), tuple(rounded[index + 1]), color, 2, cv2.LINE_AA)
    if geometry_ok[0]:
        cv2.circle(canvas, tuple(rounded[0]), 6, _START_COLOR, -1, cv2.LINE_AA)
    if geometry_ok[-1]:
        cv2.circle(canvas, tuple(rounded[-1]), 6, _END_COLOR, -1, cv2.LINE_AA)
    return canvas
