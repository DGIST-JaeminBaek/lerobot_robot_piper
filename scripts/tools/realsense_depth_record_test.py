#!/usr/bin/env python3
"""Record two RealSense RGB/depth streams without a robot and verify the result.

This exercises the complete depth backport:

RealSense z16 -> quantize -> PNG -> HEVC gray12le lossless -> decode -> pixel comparison

Example:
    conda activate ugrp
    python scripts/tools/realsense_depth_record_test.py --seconds 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import av
import cv2
import numpy as np
import pyrealsense2 as rs

from lerobot.datasets.depth_utils import DepthFeature, quantize_depth
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features

DEFAULT_TOP_SERIAL = "327122074262"
DEFAULT_WRIST_SERIAL = "243322071626"
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class CameraStream:
    name: str
    serial: str
    pipeline: rs.pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record RealSense RGB/depth streams to LeRobotDataset without a robot."
    )
    parser.add_argument("--top-serial", default=os.environ.get("TOP_CAM", DEFAULT_TOP_SERIAL))
    parser.add_argument("--wrist-serial", default=os.environ.get("WRIST_CAM", DEFAULT_WRIST_SERIAL))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--warmup-seconds", type=float, default=3.0)
    parser.add_argument("--timeout-ms", type=int, default=3000)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Dataset directory. Defaults to records/local/realsense_depth_test_<timestamp>.",
    )
    parser.add_argument(
        "--depth-only",
        action="store_true",
        help="Store only depth videos. Color is still enabled to match normal recording settings.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show TOP/WRIST RGB and depth previews. Press q or Esc to stop recording.",
    )
    parser.add_argument(
        "--preview-width",
        type=int,
        default=1200,
        help="Maximum preview window width.",
    )
    parser.add_argument(
        "--encoder-threads",
        type=int,
        default=4,
        help="x265 worker threads per depth video. Lower values keep VNC responsive.",
    )
    return parser.parse_args()


def list_connected_devices() -> dict[str, str]:
    devices = {}
    for device in rs.context().query_devices():
        serial = device.get_info(rs.camera_info.serial_number)
        devices[serial] = device.get_info(rs.camera_info.name)
    return devices


def start_camera(
    name: str, serial: str, width: int, height: int, fps: int
) -> CameraStream:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    pipeline.start(config)
    return CameraStream(name=name, serial=serial, pipeline=pipeline)


def frame_digest(frame: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(frame)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def make_features(
    camera_names: list[str],
    height: int,
    width: int,
    depth_only: bool,
    encoder_threads: int,
) -> dict:
    hardware_features = {}
    for name in camera_names:
        if not depth_only:
            hardware_features[name] = (height, width, 3)
        hardware_features[f"{name}_depth"] = DepthFeature(height, width)
    features = hw_to_dataset_features(hardware_features, "observation", use_video=True)
    for name in camera_names:
        depth_key = f"observation.images.{name}_depth"
        features[depth_key]["info"]["video.extra_options"] = {
            "x265-params": f"lossless=1:pools={encoder_threads}:frame-threads={min(2, encoder_threads)}"
        }
    return features


def decode_depth_hashes(video_path: Path) -> list[str]:
    hashes = []
    with av.open(str(video_path), "r") as container:
        for frame in container.decode(video=0):
            quantized = frame.to_ndarray(format="gray12le")
            hashes.append(frame_digest(quantized))
    return hashes


def make_preview(
    frames: list[tuple[str, np.ndarray, np.ndarray]], max_width: int
) -> np.ndarray:
    rows = []
    for name, color, depth in frames:
        color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        depth_8bit = cv2.convertScaleAbs(np.clip(depth, 0, 10_000), alpha=255.0 / 10_000.0)
        depth_bgr = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_TURBO)
        cv2.putText(
            color_bgr,
            f"{name.upper()} RGB",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            depth_bgr,
            f"{name.upper()} DEPTH (0-10m)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        rows.append(np.hstack((color_bgr, depth_bgr)))

    preview = np.vstack(rows)
    if preview.shape[1] > max_width:
        scale = max_width / preview.shape[1]
        preview = cv2.resize(
            preview,
            (max_width, round(preview.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return preview


def validate_dataset(
    root: Path,
    dataset: LeRobotDataset,
    expected_hashes: dict[str, list[str]],
) -> None:
    print("\n[VERIFY] Decoding depth videos and comparing every pixel...")
    for key, expected in expected_hashes.items():
        video_path = root / dataset.meta.get_video_file_path(0, key)
        actual = decode_depth_hashes(video_path)
        if actual != expected:
            mismatch = next(
                (
                    index
                    for index, (expected_hash, actual_hash) in enumerate(
                        zip(expected, actual, strict=False)
                    )
                    if expected_hash != actual_hash
                ),
                min(len(expected), len(actual)),
            )
            raise AssertionError(
                f"{key}: depth roundtrip mismatch at frame {mismatch}; "
                f"recorded={len(expected)}, decoded={len(actual)}"
            )
        print(f"  PASS {key}: {len(actual)}/{len(expected)} frames pixel-exact")

    info = json.loads((root / "meta" / "info.json").read_text())
    for key in expected_hashes:
        feature = info["features"][key]
        video_info = feature.get("info", {})
        if not video_info.get("video.is_depth_map"):
            raise AssertionError(f"{key}: video.is_depth_map is not true")
        if video_info.get("video.pix_fmt") != "gray12le":
            raise AssertionError(f"{key}: unexpected pixel format {video_info.get('video.pix_fmt')}")
        print(
            f"  META {key}: codec={video_info.get('video.codec')}, "
            f"pix_fmt={video_info.get('video.pix_fmt')}, shape={feature['shape']}"
        )

    loaded = LeRobotDataset(dataset.repo_id, root=root, video_backend="pyav")
    first = loaded[0]
    for key in expected_hashes:
        depth = first[key]
        print(
            f"  LOAD {key}: shape={tuple(depth.shape)}, dtype={depth.dtype}, "
            f"range_mm=[{depth.min().item():.0f}, {depth.max().item():.0f}]"
        )


def main() -> int:
    args = parse_args()
    if args.seconds <= 0 or args.warmup_seconds < 0:
        raise ValueError("seconds must be positive and warmup-seconds must be non-negative")
    if args.fps <= 0 or args.width <= 0 or args.height <= 0:
        raise ValueError("width, height, and fps must be positive")
    if args.preview_width <= 0:
        raise ValueError("preview-width must be positive")
    if args.encoder_threads <= 0:
        raise ValueError("encoder-threads must be positive")
    if args.preview and not os.environ.get("DISPLAY"):
        raise RuntimeError("--preview requires a graphical DISPLAY (run it inside the VNC desktop)")

    requested = {"top": args.top_serial, "wrist": args.wrist_serial}
    connected = list_connected_devices()
    print("[DEVICES]")
    for serial, model in connected.items():
        print(f"  {serial}: {model}")
    missing = {name: serial for name, serial in requested.items() if serial not in connected}
    if missing:
        print(f"[ERROR] Configured cameras not found: {missing}", file=sys.stderr)
        return 2

    timestamp = datetime.now().strftime("%m%d-%H%M%S")
    root = (
        args.output.expanduser().resolve()
        if args.output is not None
        else REPO_ROOT / "records" / "local" / f"realsense_depth_test_{timestamp}"
    )
    repo_id = f"local/realsense_depth_test_{timestamp}"
    if root.exists():
        print(f"[ERROR] Output already exists: {root}", file=sys.stderr)
        return 2

    streams = []
    dataset = None
    try:
        print("\n[START]")
        for name, serial in requested.items():
            print(f"  Starting {name} ({serial})...")
            streams.append(start_camera(name, serial, args.width, args.height, args.fps))

        warmup_frames = max(1, round(args.warmup_seconds * args.fps))
        print(f"\n[WARMUP] {warmup_frames} frames per camera")
        for _ in range(warmup_frames):
            for stream in streams:
                stream.pipeline.wait_for_frames(timeout_ms=args.timeout_ms)

        camera_names = [stream.name for stream in streams]
        features = make_features(
            camera_names,
            args.height,
            args.width,
            args.depth_only,
            args.encoder_threads,
        )
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=args.fps,
            features=features,
            root=root,
            robot_type="realsense_camera_only",
            use_videos=True,
            video_backend="pyav",
            image_writer_threads=max(4, len(features) * 2),
        )

        expected_hashes = {
            f"observation.images.{name}_depth": [] for name in camera_names
        }
        if args.preview:
            cv2.namedWindow("RealSense depth recording test", cv2.WINDOW_NORMAL)
        frame_count = max(1, round(args.seconds * args.fps))
        print(
            f"\n[RECORD] {frame_count} frames "
            f"({args.seconds:.1f}s at {args.fps} fps) -> {root}"
        )
        started = time.perf_counter()
        for frame_index in range(frame_count):
            dataset_frame = {"task": "realsense depth recording test"}
            preview_frames = []
            for stream in streams:
                frames = stream.pipeline.wait_for_frames(timeout_ms=args.timeout_ms)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    raise RuntimeError(
                        f"{stream.name}: missing color or depth at frame {frame_index}"
                    )

                color = np.asanyarray(color_frame.get_data())
                depth = np.asanyarray(depth_frame.get_data())
                if color.shape != (args.height, args.width, 3):
                    raise RuntimeError(f"{stream.name}: unexpected color shape {color.shape}")
                if depth.shape != (args.height, args.width) or depth.dtype != np.uint16:
                    raise RuntimeError(
                        f"{stream.name}: unexpected depth {depth.shape=} {depth.dtype=}"
                    )

                if not args.depth_only:
                    dataset_frame[f"observation.images.{stream.name}"] = color
                depth_key = f"observation.images.{stream.name}_depth"
                dataset_frame[depth_key] = depth[..., None]
                expected_hashes[depth_key].append(frame_digest(quantize_depth(depth)))
                preview_frames.append((stream.name, color, depth))

            dataset.add_frame(dataset_frame)
            if args.preview:
                preview = make_preview(preview_frames, args.preview_width)
                cv2.imshow("RealSense depth recording test", preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print(f"  Preview stopped at frame {frame_index + 1}")
                    break
            if frame_index == 0 or (frame_index + 1) % args.fps == 0:
                elapsed = time.perf_counter() - started
                print(f"  frame {frame_index + 1}/{frame_count}, elapsed={elapsed:.1f}s")

        if args.preview:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        print(
            "\n[SAVE] Preview closed. Waiting for PNG writers, then encoding "
            f"{len(features)} video(s) sequentially."
        )
        print(
            f"  Depth encoder is limited to {args.encoder_threads} threads to keep VNC responsive."
        )
        print("  This can take longer than the recording itself; do not interrupt.")
        save_started = time.perf_counter()
        dataset.save_episode()
        dataset.finalize()
        print(f"  Save/encode completed in {time.perf_counter() - save_started:.1f}s")
        verify_started = time.perf_counter()
        validate_dataset(root, dataset, expected_hashes)
        print(f"  Verification completed in {time.perf_counter() - verify_started:.1f}s")
    finally:
        if dataset is not None:
            dataset.stop_image_writer()
        for stream in streams:
            try:
                stream.pipeline.stop()
            except Exception as error:
                print(f"[WARN] Failed to stop {stream.name}: {error}", file=sys.stderr)
        if args.preview:
            cv2.destroyAllWindows()

    print(f"\n[SUCCESS] Camera-only depth recording test passed: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
