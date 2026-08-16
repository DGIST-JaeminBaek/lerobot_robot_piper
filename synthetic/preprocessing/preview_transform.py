#!/usr/bin/env python3
"""Preview a model image-preprocessing profile against a real frame.

Applies a profile's crop/resize/letterbox to an actual TOP frame and to any
raw points given on the command line, then writes overlay images so the
raw-image crop region and the resulting model image can be checked by eye.
This tool never connects to ROS, CAN, or the robot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from synthetic.calibration.common import (  # noqa: E402
    CalibrationError,
    load_source_frame,
    parse_xy,
    write_json,
)
from synthetic.preprocessing.image_transform import (  # noqa: E402
    PreprocessingError,
    model_to_raw_points,
    raw_point_visibility,
    raw_to_model_points,
    transform_image,
)
from synthetic.preprocessing.profiles import (  # noqa: E402
    ImageProfile,
    load_profile,
    smolvla_v1_profile,
)


BUILTIN_PROFILES = {
    "smolvla_v1": smolvla_v1_profile,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply an image-preprocessing profile to a real frame and "
            "preview raw/model overlays. No robot connection is used."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--video", type=Path)
    parser.add_argument("--frame", type=int, default=0, help="Frame index for --video")

    profile_source = parser.add_mutually_exclusive_group(required=True)
    profile_source.add_argument("--profile", type=Path, help="Profile JSON path")
    profile_source.add_argument(
        "--profile-name",
        choices=sorted(BUILTIN_PROFILES),
        help="Built-in example profile",
    )

    parser.add_argument(
        "--raw-point",
        type=parse_xy,
        action="append",
        default=[],
        help="Raw image pixel x,y to preview; may be repeated",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def load_selected_profile(args: argparse.Namespace) -> ImageProfile:
    if args.profile is not None:
        return load_profile(args.profile)
    return BUILTIN_PROFILES[args.profile_name]()


def draw_crop_overlay(
    frame: np.ndarray,
    profile: ImageProfile,
    raw_points: list[tuple[float, float]],
    visibility: np.ndarray,
) -> np.ndarray:
    canvas = frame.copy()
    crop = profile.effective_crop()
    cv2.rectangle(
        canvas,
        (crop.x, crop.y),
        (crop.right() - 1, crop.bottom() - 1),
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for point, visible in zip(raw_points, visibility, strict=True):
        location = (round(point[0]), round(point[1]))
        color = (0, 255, 0) if visible else (0, 0, 255)
        cv2.circle(canvas, location, 6, color, -1, cv2.LINE_AA)
    return canvas


def draw_model_overlay(
    model_image: np.ndarray,
    model_points: np.ndarray,
    visibility: np.ndarray,
) -> np.ndarray:
    canvas = model_image.copy()
    for point, visible in zip(model_points, visibility, strict=True):
        location = (round(point[0]), round(point[1]))
        color = (0, 255, 0) if visible else (0, 0, 255)
        cv2.drawMarker(
            canvas,
            location,
            color,
            markerType=cv2.MARKER_CROSS,
            markerSize=12,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
    return canvas


def main() -> int:
    args = build_parser().parse_args()
    try:
        frame, source_info = load_source_frame(
            image_path=args.image,
            video_path=args.video,
            frame_index=args.frame,
        )
        profile = load_selected_profile(args)
        if source_info["width"] != profile.source.width or (
            source_info["height"] != profile.source.height
        ):
            raise PreprocessingError(
                f"frame shape {source_info['width']}x{source_info['height']} "
                f"does not match profile source "
                f"{profile.source.width}x{profile.source.height}"
            )

        model_image = transform_image(frame, profile)

        raw_points = list(args.raw_point)
        report: dict[str, Any] = {
            "profile": profile.to_dict(),
            "source": source_info,
            "model_image_shape": {
                "width": int(model_image.shape[1]),
                "height": int(model_image.shape[0]),
            },
            "points": [],
        }

        output_dir = args.output_dir.expanduser().resolve()

        if raw_points:
            visibility = raw_point_visibility(raw_points, profile)
            model_points = raw_to_model_points(raw_points, profile, strict=False)
            recovered_raw = model_to_raw_points(model_points, profile)
            for raw_point, model_point, recovered, visible in zip(
                raw_points, model_points, recovered_raw, visibility, strict=True
            ):
                round_trip_error = float(
                    np.linalg.norm(np.asarray(raw_point) - recovered)
                )
                report["points"].append(
                    {
                        "raw_px": [float(raw_point[0]), float(raw_point[1])],
                        "model_px": [float(model_point[0]), float(model_point[1])],
                        "recovered_raw_px": [
                            float(recovered[0]),
                            float(recovered[1]),
                        ],
                        "round_trip_error_px": round_trip_error,
                        "visible_in_model": bool(visible),
                    }
                )
                print(
                    f"[POINT] raw={tuple(raw_point)} -> model="
                    f"({model_point[0]:.3f}, {model_point[1]:.3f}) "
                    f"visible={bool(visible)} round_trip_err={round_trip_error:.6f}px"
                )

            raw_overlay = draw_crop_overlay(frame, profile, raw_points, visibility)
            model_overlay = draw_model_overlay(model_image, model_points, visibility)
        else:
            raw_overlay = draw_crop_overlay(frame, profile, [], np.asarray([], dtype=bool))
            model_overlay = model_image.copy()

        output_dir.mkdir(parents=True, exist_ok=True)

        def _write_image(name: str, image: np.ndarray) -> Path:
            path = output_dir / name
            if path.exists() and not args.overwrite:
                raise PreprocessingError(
                    f"output already exists: {path}; pass --overwrite to replace it"
                )
            if not cv2.imwrite(str(path), image):
                raise PreprocessingError(f"could not write image: {path}")
            return path

        _write_image("raw_overlay.png", raw_overlay)
        _write_image("model_overlay.png", model_overlay)
        _write_image("model_image.png", model_image)
        write_json(output_dir / "preview_report.json", report, overwrite=args.overwrite)

        print(f"[OK] preview written to: {output_dir}")
        print(
            "[STATUS] preprocessing preview only; no calibration or robot "
            "validation is implied"
        )
        return 0
    except (CalibrationError, PreprocessingError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
