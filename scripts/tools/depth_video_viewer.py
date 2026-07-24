#!/usr/bin/env python3
"""View LeRobot gray12le depth MP4 files as a colorized preview.

The stored MP4 remains unchanged. Conversion to millimetres and the color map
exist only in the display window.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import ExitStack
from pathlib import Path

import av
import cv2
import numpy as np

from lerobot.datasets.depth_utils import (
    DEFAULT_DEPTH_MAX,
    DEFAULT_DEPTH_MIN,
    DEFAULT_DEPTH_SHIFT,
    DEFAULT_DEPTH_USE_LOG,
    dequantize_depth,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Display LeRobot depth MP4 files.")
    parser.add_argument("path", type=Path, help="Dataset root or a depth .mp4 file")
    parser.add_argument("--camera", choices=["top", "wrist", "both"], default="both")
    parser.add_argument("--min-mm", type=float, default=100.0)
    parser.add_argument("--max-mm", type=float, default=3000.0)
    parser.add_argument("--window-width", type=int, default=1600)
    return parser.parse_args()


def depth_params(feature: dict | None) -> dict:
    info = (feature or {}).get("info") or {}
    return {
        "depth_min": info.get("video.depth_min", DEFAULT_DEPTH_MIN),
        "depth_max": info.get("video.depth_max", DEFAULT_DEPTH_MAX),
        "shift": info.get("video.shift", DEFAULT_DEPTH_SHIFT),
        "use_log": info.get("video.use_log", DEFAULT_DEPTH_USE_LOG),
        "output_tensor": False,
    }


def resolve_inputs(path: Path, camera: str) -> list[tuple[str, Path, dict]]:
    path = path.expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() != ".mp4":
            raise ValueError(f"Expected an MP4 file, got {path}")
        return [(path.stem, path, depth_params(None))]

    info_path = path / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found: {info_path}")
    info = json.loads(info_path.read_text())
    cameras = ["top", "wrist"] if camera == "both" else [camera]
    resolved = []
    for name in cameras:
        key = f"observation.images.{name}_depth"
        feature = info["features"].get(key)
        if feature is None:
            raise KeyError(f"Depth feature not found: {key}")
        videos = sorted((path / "videos" / key).glob("**/*.mp4"))
        if not videos:
            raise FileNotFoundError(f"No depth MP4 found for {key}")
        if len(videos) > 1:
            raise RuntimeError(
                f"{key} has {len(videos)} video chunks; select one MP4 path directly"
            )
        resolved.append((name, videos[0], depth_params(feature)))
    return resolved


def colorize_depth(
    quantized: np.ndarray,
    params: dict,
    min_mm: float,
    max_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    depth = dequantize_depth(quantized, **params).squeeze()
    clipped = np.clip(depth, min_mm, max_mm)
    normalized = ((clipped - min_mm) * (255.0 / (max_mm - min_mm))).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[depth <= min_mm] = 0
    return colored, depth


def main() -> int:
    args = parse_args()
    if not os.environ.get("DISPLAY"):
        raise RuntimeError("No graphical DISPLAY. Run this inside the VNC desktop terminal.")
    if args.max_mm <= args.min_mm:
        raise ValueError("--max-mm must be greater than --min-mm")
    if args.window_width <= 0:
        raise ValueError("--window-width must be positive")

    inputs = resolve_inputs(args.path, args.camera)
    window = "Depth video viewer (Space: pause, Q/Esc: quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    with ExitStack() as stack:
        containers = [stack.enter_context(av.open(str(video))) for _, video, _ in inputs]
        streams = [container.streams.video[0] for container in containers]
        decoders = [container.decode(stream) for container, stream in zip(containers, streams, strict=True)]
        fps = float(streams[0].average_rate or 30)
        delay_ms = max(1, round(1000 / fps))
        paused = False
        frame_index = 0

        for frames in zip(*decoders, strict=False):
            panels = []
            for (name, _, params), frame in zip(inputs, frames, strict=True):
                quantized = frame.to_ndarray(format="gray12le")
                panel, depth = colorize_depth(
                    quantized, params, args.min_mm, args.max_mm
                )
                valid = depth[depth > args.min_mm]
                range_text = (
                    f"{valid.min():.0f}-{valid.max():.0f} mm" if valid.size else "no valid depth"
                )
                cv2.putText(
                    panel,
                    f"{name.upper()}  frame={frame_index}  {range_text}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                panels.append(panel)

            preview = np.hstack(panels)
            if preview.shape[1] > args.window_width:
                scale = args.window_width / preview.shape[1]
                preview = cv2.resize(
                    preview,
                    (args.window_width, round(preview.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            cv2.imshow(window, preview)

            while True:
                key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    cv2.destroyAllWindows()
                    return 0
                if key == ord(" "):
                    paused = not paused
                    continue
                break
            frame_index += 1

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
