from __future__ import annotations

"""Monitor live RealSense color frames for new drawing in learned board regions.

Press B while the board is clean to capture a baseline.  The tool then marks
each region as DRAWN when enough new dark pixels appear relative to that
baseline.  Regions are calculated from today's cropped-asset manifests.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


DEFAULT_REGIONS = {
    # Derived once from today's 315 cropped assets at 1280x720.
    "top 95%": ((213, 0, 683, 256), (0, 215, 255)),
    "bottom 95%": ((223, 327, 698, 675), (255, 255, 0)),
    "top dense": ((283, 52, 624, 192), (0, 100, 255)),
    "bottom dense": ((298, 398, 637, 553), (255, 0, 255)),
}
TOP_REALSENSE_SERIAL = "327122074262"
WRIST_REALSENSE_SERIAL = "243322071626"


def load_groups(root: Path, split_y: int) -> dict[str, list[tuple[int, int, int, int, float, float]]]:
    groups = {"top": [], "bottom": []}
    for manifest in root.rglob("*manifest.csv"):
        with manifest.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                try:
                    x, y, w, h = (int(row[key]) for key in ("crop_x", "crop_y", "crop_w", "crop_h"))
                    groups["top" if y + h / 2 < split_y else "bottom"].append((x, y, x + w, y + h, x + w / 2, y + h / 2))
                except (KeyError, TypeError, ValueError):
                    continue
    return groups


def percentile_box(boxes: list[tuple[int, int, int, int, float, float]], padding: int) -> tuple[int, int, int, int]:
    return (
        max(0, round(np.percentile([box[0] for box in boxes], 2.5)) - padding),
        max(0, round(np.percentile([box[1] for box in boxes], 2.5)) - padding),
        round(np.percentile([box[2] for box in boxes], 97.5)) + padding,
        round(np.percentile([box[3] for box in boxes], 97.5)) + padding,
    )


def dense_box(boxes: list[tuple[int, int, int, int, float, float]], shape: tuple[int, int], padding: int) -> tuple[int, int, int, int]:
    density = np.zeros(shape, dtype=np.float32)
    for *_unused, cx, cy in boxes:
        density[round(cy), round(cx)] += 1.0
    density = cv2.GaussianBlur(density, (0, 0), 38)
    mask = (density >= density.max() * 0.42).astype(np.uint8)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    label = max(range(1, count), key=lambda idx: stats[idx, cv2.CC_STAT_AREA])
    x, y, w, h = stats[label, :4]
    return max(0, x - padding), max(0, y - padding), x + w + padding, y + h + padding


def region_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    x0, y0, x1, y1 = box
    cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
    return mask


def main() -> int:
    parser = argparse.ArgumentParser(description="Overlay 95%-range/dense regions and detect fresh drawing from a RealSense color stream.")
    parser.add_argument("--assets-root", type=Path, default=None, help="Only used with --recompute-regions")
    parser.add_argument("--recompute-regions", action="store_true", help="Recalculate regions from local candidate crop manifests instead of using the built-in defaults")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--split-y", type=int, default=356)
    parser.add_argument("--pixel-delta", type=int, default=24)
    parser.add_argument("--min-drawing-pixels", type=int, default=220)
    parser.add_argument("--flip-180", action="store_true")
    parser.add_argument("--camera-serial", default=TOP_REALSENSE_SERIAL, help="RealSense serial; defaults to the calibrated Top camera")
    parser.add_argument("--snapshot", type=Path, default=None, help="Save one color frame to this image path and exit; useful over SSH")
    parser.add_argument("--warmup-frames", type=int, default=30, help="Frames to discard before --snapshot (default: 30)")
    args = parser.parse_args()

    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise SystemExit("pyrealsense2 is required. Install it in the ugrp environment: python -m pip install pyrealsense2") from exc

    regions = DEFAULT_REGIONS.copy()
    if args.recompute_regions:
        if args.assets_root is None:
            raise SystemExit("--recompute-regions requires --assets-root")
        groups = load_groups(args.assets_root, args.split_y)
        if not groups["top"] or not groups["bottom"]:
            raise SystemExit(f"No crop-manifest entries found under {args.assets_root}")
        regions = {
            "top 95%": (percentile_box(groups["top"], 14), (0, 215, 255)),
            "bottom 95%": (percentile_box(groups["bottom"], 14), (255, 255, 0)),
            "top dense": (dense_box(groups["top"], (args.height, args.width), 10), (0, 100, 255)),
            "bottom dense": (dense_box(groups["bottom"], (args.height, args.width), 10), (255, 0, 255)),
        }
    masks = {name: region_mask((args.height, args.width), box) for name, (box, _color) in regions.items()}

    pipeline, config = rs.pipeline(), rs.config()
    config.enable_device(args.camera_serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    pipeline.start(config)
    baseline: np.ndarray | None = None
    print(f"Using RealSense serial: {args.camera_serial}")
    if args.snapshot is not None:
        print(f"Capturing one frame after {args.warmup_frames} warmup frames...")
    print("B: capture clean-board baseline | Q/Esc: quit")
    try:
        if args.snapshot is not None:
            image: np.ndarray | None = None
            for _ in range(max(1, args.warmup_frames)):
                frame = pipeline.wait_for_frames().get_color_frame()
                if frame:
                    image = np.asanyarray(frame.get_data())
            if image is None:
                raise RuntimeError("No color frame received from the RealSense camera")
            if args.flip_180:
                image = cv2.rotate(image, cv2.ROTATE_180)
            display = image.copy()
            for name, (box, color) in regions.items():
                x0, y0, x1, y1 = box
                cv2.rectangle(display, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
                cv2.putText(display, f"{name}: clear", (x0 + 5, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            cv2.putText(display, "RealSense region-monitor snapshot", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2, cv2.LINE_AA)
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.snapshot), display):
                raise RuntimeError(f"Failed to write snapshot: {args.snapshot}")
            print(f"Saved snapshot: {args.snapshot} ({image.shape[1]}x{image.shape[0]})")
            return 0
        while True:
            frame = pipeline.wait_for_frames().get_color_frame()
            if not frame:
                continue
            image = np.asanyarray(frame.get_data())
            if args.flip_180:
                image = cv2.rotate(image, cv2.ROTATE_180)
            display = image.copy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if baseline is not None:
                delta = cv2.absdiff(gray, baseline)
                drawing = (delta >= args.pixel_delta).astype(np.uint8) * 255
                drawing = cv2.morphologyEx(drawing, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            else:
                drawing = np.zeros_like(gray)
            for name, (box, color) in regions.items():
                x0, y0, x1, y1 = box
                pixels = int(np.count_nonzero(cv2.bitwise_and(drawing, masks[name])))
                active = pixels >= args.min_drawing_pixels
                draw_color = (0, 0, 255) if active else color
                cv2.rectangle(display, (x0, y0), (x1, y1), draw_color, 2, cv2.LINE_AA)
                label = f"{name}: {'DRAWN' if active else 'clear'} ({pixels})"
                cv2.putText(display, label, (x0 + 5, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, draw_color, 2, cv2.LINE_AA)
            cv2.putText(display, "B: set clean baseline   Q: quit", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2, cv2.LINE_AA)
            cv2.imshow("RealSense drawing-region monitor", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("b"):
                baseline = gray.copy()
                print("Captured clean-board baseline.")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
