#!/usr/bin/env python3
"""Data schema for VLA-specific image preprocessing profiles.

Physical calibration is fixed to the canonical TOP image (1280x720). Each VLA
consumes that canonical image differently (crop, resize, letterbox); this
module stores that per-model preprocessing as a versioned, JSON-serializable
profile so calibration never depends on any one model's input convention.

This module is intentionally independent from ROS, CAN, LeRobot, and Piper
hardware. It only handles plain numbers and JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from synthetic.calibration.common import read_json, write_json

FORMAT_VERSION = 1
PROFILE_TYPE = "image_preprocessing_profile"
RESIZE_MODES = ("stretch", "fit", "letterbox")


class PreprocessingError(ValueError):
    """Raised when an image-preprocessing profile or point is invalid."""


@dataclass(frozen=True)
class SourceShape:
    """Canonical raw-image shape a profile was defined against."""

    width: int
    height: int

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise PreprocessingError(
                f"source width/height must be positive, got {self.width}x{self.height}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"width": int(self.width), "height": int(self.height)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SourceShape":
        try:
            shape = cls(width=int(payload["width"]), height=int(payload["height"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise PreprocessingError(f"invalid source shape: {payload!r}") from exc
        shape.validate()
        return shape


@dataclass(frozen=True)
class CropRegion:
    """A rectangle in raw pixel coordinates.

    The right/bottom edge is exclusive: valid columns are
    ``[x, x + width)`` and valid rows are ``[y, y + height)``, matching the
    convention used in ``synthetic/calibration/common.py``.
    """

    x: int
    y: int
    width: int
    height: int

    def right(self) -> int:
        return self.x + self.width

    def bottom(self) -> int:
        return self.y + self.height

    def validate(self, source: SourceShape) -> None:
        if self.width <= 0 or self.height <= 0:
            raise PreprocessingError(
                f"crop width/height must be positive, got {self.width}x{self.height}"
            )
        if self.x < 0 or self.y < 0:
            raise PreprocessingError(
                f"crop origin must be >=0, got x={self.x}, y={self.y}"
            )
        if self.right() > source.width or self.bottom() > source.height:
            raise PreprocessingError(
                "crop must stay inside source bounds: "
                f"0<=x, x+width<={source.width}; 0<=y, y+height<={source.height}; "
                f"got x={self.x}, y={self.y}, width={self.width}, height={self.height} "
                f"on source {source.width}x{source.height}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": int(self.x),
            "y": int(self.y),
            "width": int(self.width),
            "height": int(self.height),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CropRegion":
        try:
            return cls(
                x=int(payload["x"]),
                y=int(payload["y"]),
                width=int(payload["width"]),
                height=int(payload["height"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PreprocessingError(f"invalid crop region: {payload!r}") from exc


@dataclass(frozen=True)
class ResizeSpec:
    """How a (possibly cropped) region is turned into the final model image.

    ``mode``:
      - ``stretch``: resize directly to (width, height); aspect ratio is not
        preserved.
      - ``fit``: aspect-ratio-preserving resize so the region fits inside
        (width, height); the output image is not padded, so its actual size
        may be smaller than (width, height) on one axis.
      - ``letterbox``: same scaling as ``fit``, then pad with ``pad_value``
        so the output image is exactly (width, height).
    """

    mode: str
    width: int
    height: int
    pad_value: tuple[int, int, int] = (0, 0, 0)

    def validate(self) -> None:
        if self.mode not in RESIZE_MODES:
            raise PreprocessingError(
                f"resize mode must be one of {RESIZE_MODES}, got {self.mode!r}"
            )
        if self.width <= 0 or self.height <= 0:
            raise PreprocessingError(
                f"resize width/height must be positive, got {self.width}x{self.height}"
            )
        if len(self.pad_value) != 3 or any(
            not (0 <= int(value) <= 255) for value in self.pad_value
        ):
            raise PreprocessingError(
                f"pad_value must be 3 values in [0, 255], got {self.pad_value!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "width": int(self.width),
            "height": int(self.height),
            "pad_value": [int(value) for value in self.pad_value],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResizeSpec":
        try:
            pad_value = tuple(int(value) for value in payload.get("pad_value", (0, 0, 0)))
            spec = cls(
                mode=str(payload["mode"]),
                width=int(payload["width"]),
                height=int(payload["height"]),
                pad_value=pad_value,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PreprocessingError(f"invalid resize spec: {payload!r}") from exc
        spec.validate()
        return spec


@dataclass(frozen=True)
class ImageProfile:
    """A named, reproducible raw-image -> model-image preprocessing recipe."""

    name: str
    source: SourceShape
    resize: ResizeSpec
    crop: CropRegion | None = None

    def validate(self) -> None:
        if not self.name:
            raise PreprocessingError("profile name must not be empty")
        self.source.validate()
        if self.crop is not None:
            self.crop.validate(self.source)
        self.resize.validate()

    def effective_crop(self) -> CropRegion:
        """The crop actually applied; the full source when none is set."""

        if self.crop is not None:
            return self.crop
        return CropRegion(x=0, y=0, width=self.source.width, height=self.source.height)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "type": PROFILE_TYPE,
            "name": self.name,
            "source": self.source.to_dict(),
            "crop": self.crop.to_dict() if self.crop is not None else None,
            "resize": self.resize.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ImageProfile":
        if payload.get("type") != PROFILE_TYPE:
            raise PreprocessingError(
                f"expected type={PROFILE_TYPE!r}, got {payload.get('type')!r}"
            )
        if payload.get("format_version") != FORMAT_VERSION:
            raise PreprocessingError(
                f"unsupported format_version: {payload.get('format_version')!r}"
            )
        try:
            name = str(payload["name"])
            source = SourceShape.from_dict(payload["source"])
            resize = ResizeSpec.from_dict(payload["resize"])
        except KeyError as exc:
            raise PreprocessingError(f"profile is missing required key: {exc}") from exc
        crop_payload = payload.get("crop")
        crop = CropRegion.from_dict(crop_payload) if crop_payload is not None else None
        profile = cls(name=name, source=source, resize=resize, crop=crop)
        profile.validate()
        return profile


def load_profile(path: Path) -> ImageProfile:
    return ImageProfile.from_dict(read_json(path))


def save_profile(path: Path, profile: ImageProfile, *, overwrite: bool) -> None:
    profile.validate()
    write_json(path, profile.to_dict(), overwrite=overwrite)


def smolvla_v1_profile() -> ImageProfile:
    """The current SmolVLA checkpoint's crop/resize.

    This reproduces one existing checkpoint's input convention. It is not the
    default or canonical calibration profile for the synthetic pipeline as a
    whole -- other VLAs are expected to define their own ``ImageProfile``.
    """

    return ImageProfile(
        name="smolvla_v1",
        source=SourceShape(width=1280, height=720),
        crop=CropRegion(x=280, y=0, width=720, height=720),
        resize=ResizeSpec(mode="stretch", width=512, height=512),
    )
