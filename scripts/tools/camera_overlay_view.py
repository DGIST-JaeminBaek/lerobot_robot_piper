#!/usr/bin/env python3
"""Show a live camera image with labeled rectangular overlays.

The default preset reproduces the four regions used in the reference image.
Coordinates are normalized to the camera frame, so the same preset scales to
any camera resolution.  Press q or Escape to close the window.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class Overlay:
    label: str
    color: tuple[int, int, int]  # BGR, as used by OpenCV
    x: float
    y: float
    width: float
    height: float


# Normalized x, y, width, height values matched to the supplied reference.
REFERENCE_OVERLAYS = (
    Overlay("top 95% range", (0, 255, 255), 0.165, 0.000, 0.370, 0.360),
    Overlay("top dense cluster", (0, 140, 255), 0.220, 0.070, 0.270, 0.200),
    Overlay("bottom 95% range", (255, 255, 0), 0.174, 0.455, 0.374, 0.485),
    Overlay("bottom dense cluster", (255, 0, 255), 0.232, 0.550, 0.268, 0.220),
)

COLOR_NAMES = {
    "yellow": (0, 255, 255),
    "orange": (0, 140, 255),
    "cyan": (255, 255, 0),
    "magenta": (255, 0, 255),
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "white": (255, 255, 255),
}


def parse_overlay(value: str) -> Overlay:
    """Parse LABEL,COLOR,X,Y,WIDTH,HEIGHT with normalized coordinates."""
    fields = [field.strip() for field in value.split(",")]
    if len(fields) != 6:
        raise argparse.ArgumentTypeError(
            "--box must be LABEL,COLOR,X,Y,WIDTH,HEIGHT (for example "
            "'pickup area,green,0.2,0.3,0.4,0.2')"
        )
    label, color_name, *coordinates = fields
    color = COLOR_NAMES.get(color_name.lower())
    if color is None:
        choices = ", ".join(COLOR_NAMES)
        raise argparse.ArgumentTypeError(f"unknown color '{color_name}' (choose: {choices})")
    try:
        x, y, width, height = (float(item) for item in coordinates)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("overlay coordinates must be numbers") from exc
    if not label or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
        raise argparse.ArgumentTypeError("label is required and coordinates must be within 0..1")
    return Overlay(label, color, x, y, width, height)


def draw_overlays(image: np.ndarray, overlays: tuple[Overlay, ...] | list[Overlay]) -> np.ndarray:
    """Return a copy of *image* with normalized overlays rendered on it."""
    output = image.copy()
    height, width = output.shape[:2]
    line_width = max(2, round(min(width, height) / 360))
    font_scale = max(0.5, min(width, height) / 900)

    for overlay in overlays:
        left = round(overlay.x * width)
        top = round(overlay.y * height)
        right = round(min(1.0, overlay.x + overlay.width) * width)
        bottom = round(min(1.0, overlay.y + overlay.height) * height)
        cv2.rectangle(output, (left, top), (right, bottom), overlay.color, line_width)

        text_y = min(max(top + round(26 * font_scale), round(22 * font_scale)), height - 4)
        cv2.putText(
            output,
            overlay.label,
            (left + line_width * 2, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            overlay.color,
            line_width,
            cv2.LINE_AA,
        )
    return output


def realsense_frames(serial: str, width: int, height: int, fps: int) -> Iterator[np.ndarray]:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError("pyrealsense2 is required for --backend realsense") from exc

    pipeline = rs.pipeline()
    config = rs.config()
    if serial:
        config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipeline.start(config)
    try:
        while True:
            color_frame = pipeline.wait_for_frames().get_color_frame()
            if color_frame:
                yield np.asanyarray(color_frame.get_data())
    finally:
        pipeline.stop()


def opencv_frames(camera: int, width: int, height: int, fps: int) -> Iterator[np.ndarray]:
    capture = cv2.VideoCapture(camera)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    if not capture.isOpened():
        raise RuntimeError(f"could not open OpenCV camera index {camera}")
    try:
        while True:
            ok, image = capture.read()
            if ok:
                yield image
    finally:
        capture.release()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("realsense", "opencv"), default="realsense")
    parser.add_argument("--serial", default=os.environ.get("TOP_CAM", ""), help="RealSense serial number")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--box",
        dest="boxes",
        type=parse_overlay,
        action="append",
        help="replace the reference preset; repeat LABEL,COLOR,X,Y,WIDTH,HEIGHT",
    )
    args = parser.parse_args()
    overlays = args.boxes or REFERENCE_OVERLAYS
    window = "Camera Overlay Viewer (q / Esc: quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    frames = (
        realsense_frames(args.serial, args.width, args.height, args.fps)
        if args.backend == "realsense"
        else opencv_frames(args.camera, args.width, args.height, args.fps)
    )
    try:
        for image in frames:
            cv2.imshow(window, draw_overlays(image, overlays))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
