#!/usr/bin/env python3
"""Raw-pixel <-> model-pixel coordinate transforms and actual image resampling.

Given an `ImageProfile` (crop + resize/letterbox), this module derives the
exact same crop/scale/pad geometry for two independent operations:

- transforming point coordinates (`raw_to_model_points`, `model_to_raw_points`)
- transforming the actual image array (`transform_image`)

so a raw calibration point and the corresponding pixel in the resampled image
always agree. This module does not depend on ROS, CAN, LeRobot, or Piper
hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from synthetic.preprocessing.profiles import ImageProfile, PreprocessingError, SourceShape


def _as_points(points: Any) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != 2:
        raise PreprocessingError(f"points must have shape (N, 2), got {array.shape}")
    if array.shape[0] < 1:
        raise PreprocessingError("points must not be empty")
    if not np.isfinite(array).all():
        raise PreprocessingError("points contain NaN or Inf")
    return array


@dataclass(frozen=True)
class ResolvedGeometry:
    """Concrete pixel-space geometry derived from an `ImageProfile`."""

    crop_x: int
    crop_y: int
    crop_width: int
    crop_height: int
    scale_x: float
    scale_y: float
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int
    output_width: int
    output_height: int


def resolve_geometry(profile: ImageProfile) -> ResolvedGeometry:
    """Resolve a profile into pixel-exact crop/scale/pad geometry.

    Integer output dimensions are computed first (rounding the scaled crop),
    and `scale_x`/`scale_y` are then derived from those integers. This keeps
    point transforms exactly consistent with what `transform_image` produces,
    at the cost of a sub-pixel deviation from a mathematically exact aspect
    ratio when rounding is involved.
    """

    profile.validate()
    crop = profile.effective_crop()
    resize = profile.resize

    if resize.mode == "stretch":
        scale_x = resize.width / crop.width
        scale_y = resize.height / crop.height
        return ResolvedGeometry(
            crop_x=crop.x,
            crop_y=crop.y,
            crop_width=crop.width,
            crop_height=crop.height,
            scale_x=scale_x,
            scale_y=scale_y,
            resized_width=resize.width,
            resized_height=resize.height,
            pad_left=0,
            pad_top=0,
            pad_right=0,
            pad_bottom=0,
            output_width=resize.width,
            output_height=resize.height,
        )

    if resize.mode in ("fit", "letterbox"):
        scale = min(resize.width / crop.width, resize.height / crop.height)
        fitted_width = max(1, round(crop.width * scale))
        fitted_height = max(1, round(crop.height * scale))
        scale_x = fitted_width / crop.width
        scale_y = fitted_height / crop.height

        if resize.mode == "fit":
            return ResolvedGeometry(
                crop_x=crop.x,
                crop_y=crop.y,
                crop_width=crop.width,
                crop_height=crop.height,
                scale_x=scale_x,
                scale_y=scale_y,
                resized_width=fitted_width,
                resized_height=fitted_height,
                pad_left=0,
                pad_top=0,
                pad_right=0,
                pad_bottom=0,
                output_width=fitted_width,
                output_height=fitted_height,
            )

        diff_w = resize.width - fitted_width
        diff_h = resize.height - fitted_height
        if diff_w < 0 or diff_h < 0:
            raise PreprocessingError(
                "letterbox target must be at least as large as the "
                f"aspect-preserving fit of the crop: fitted "
                f"{fitted_width}x{fitted_height} does not fit in "
                f"{resize.width}x{resize.height}"
            )
        pad_left = diff_w // 2
        pad_right = diff_w - pad_left
        pad_top = diff_h // 2
        pad_bottom = diff_h - pad_top
        return ResolvedGeometry(
            crop_x=crop.x,
            crop_y=crop.y,
            crop_width=crop.width,
            crop_height=crop.height,
            scale_x=scale_x,
            scale_y=scale_y,
            resized_width=fitted_width,
            resized_height=fitted_height,
            pad_left=pad_left,
            pad_top=pad_top,
            pad_right=pad_right,
            pad_bottom=pad_bottom,
            output_width=resize.width,
            output_height=resize.height,
        )

    raise PreprocessingError(f"unsupported resize mode: {resize.mode!r}")


def _inside_crop_mask(points: np.ndarray, geometry: ResolvedGeometry) -> np.ndarray:
    x, y = points[:, 0], points[:, 1]
    return (
        (x >= geometry.crop_x)
        & (x < geometry.crop_x + geometry.crop_width)
        & (y >= geometry.crop_y)
        & (y < geometry.crop_y + geometry.crop_height)
    )


def is_inside_crop(points: Any, profile: ImageProfile) -> np.ndarray:
    """True where a raw point falls inside the profile's crop region."""

    geometry = resolve_geometry(profile)
    return _inside_crop_mask(_as_points(points), geometry)


def is_inside_raw_bounds(points: Any, profile: ImageProfile) -> np.ndarray:
    """True where a raw point falls inside the canonical source image."""

    profile.source.validate()
    raw = _as_points(points)
    x, y = raw[:, 0], raw[:, 1]
    return (
        (x >= 0)
        & (x < profile.source.width)
        & (y >= 0)
        & (y < profile.source.height)
    )


def is_inside_model_bounds(points: Any, profile: ImageProfile) -> np.ndarray:
    """True where a model-pixel point falls inside the final model image."""

    geometry = resolve_geometry(profile)
    model = _as_points(points)
    x, y = model[:, 0], model[:, 1]
    return (
        (x >= 0)
        & (x < geometry.output_width)
        & (y >= 0)
        & (y < geometry.output_height)
    )


def raw_point_visibility(points: Any, profile: ImageProfile) -> np.ndarray:
    """True where a raw point survives the crop and lands inside the model image.

    A raw point can be lost either by falling outside the crop (cut away
    before resizing) or, in principle, by landing outside the final canvas
    after resize/pad. Both cases mean the corresponding real-world location
    is not visible to the model and must not be treated as a valid model
    input target.
    """

    raw = _as_points(points)
    geometry = resolve_geometry(profile)
    inside_crop = _inside_crop_mask(raw, geometry)
    model_points = raw_to_model_points(raw, profile, strict=False)
    inside_model = is_inside_model_bounds(model_points, profile)
    return inside_crop & inside_model


def raw_to_model_points(
    points: Any,
    profile: ImageProfile,
    *,
    strict: bool = True,
) -> np.ndarray:
    """Map raw canonical-image pixels to model-input pixels.

    With `strict=True` (default), raises if any point falls outside the
    profile's crop region -- such a point has no valid model-pixel location.
    Pass `strict=False` for diagnostics that need the (out-of-model)
    coordinate anyway, e.g. preview overlays.
    """

    geometry = resolve_geometry(profile)
    raw = _as_points(points)
    if strict:
        mask = _inside_crop_mask(raw, geometry)
        if not mask.all():
            rejected = raw[~mask]
            raise PreprocessingError(
                f"{len(rejected)} raw point(s) fall outside the crop region "
                f"x=[{geometry.crop_x}, {geometry.crop_x + geometry.crop_width}), "
                f"y=[{geometry.crop_y}, {geometry.crop_y + geometry.crop_height}): "
                f"{rejected.tolist()}"
            )
    model_x = (raw[:, 0] - geometry.crop_x) * geometry.scale_x + geometry.pad_left
    model_y = (raw[:, 1] - geometry.crop_y) * geometry.scale_y + geometry.pad_top
    return np.stack([model_x, model_y], axis=1)


def model_to_raw_points(points: Any, profile: ImageProfile) -> np.ndarray:
    """Map model-input pixels back to raw canonical-image pixels."""

    geometry = resolve_geometry(profile)
    model = _as_points(points)
    raw_x = (model[:, 0] - geometry.pad_left) / geometry.scale_x + geometry.crop_x
    raw_y = (model[:, 1] - geometry.pad_top) / geometry.scale_y + geometry.crop_y
    return np.stack([raw_x, raw_y], axis=1)


def _validate_image_matches_source(image: np.ndarray, source: SourceShape) -> None:
    if image.ndim not in (2, 3):
        raise PreprocessingError(
            f"image must be a 2D or 3D array, got shape {image.shape}"
        )
    height, width = image.shape[:2]
    if width != source.width or height != source.height:
        raise PreprocessingError(
            f"image shape {width}x{height} does not match profile source "
            f"{source.width}x{source.height}"
        )


def transform_image(image: np.ndarray, profile: ImageProfile) -> np.ndarray:
    """Apply a profile's crop/resize/letterbox to an actual image array."""

    profile.validate()
    _validate_image_matches_source(image, profile.source)
    geometry = resolve_geometry(profile)

    cropped = image[
        geometry.crop_y : geometry.crop_y + geometry.crop_height,
        geometry.crop_x : geometry.crop_x + geometry.crop_width,
    ]
    resized = cv2.resize(
        cropped,
        (geometry.resized_width, geometry.resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    if profile.resize.mode in ("stretch", "fit"):
        return resized

    return cv2.copyMakeBorder(
        resized,
        geometry.pad_top,
        geometry.pad_bottom,
        geometry.pad_left,
        geometry.pad_right,
        cv2.BORDER_CONSTANT,
        value=tuple(int(value) for value in profile.resize.pad_value),
    )
