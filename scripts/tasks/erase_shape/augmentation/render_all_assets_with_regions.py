from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def composite(frame: np.ndarray, asset: np.ndarray, x: int, y: int) -> None:
    h, w = asset.shape[:2]
    alpha = asset[:, :, 3:4].astype(np.float32) / 255.0
    roi = frame[y : y + h, x : x + w].astype(np.float32)
    frame[y : y + h, x : x + w] = (asset[:, :, :3].astype(np.float32) * alpha + roi * (1 - alpha)).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description="Overlay every extracted RGBA asset at its original crop position and draw top/bottom coverage rectangles.")
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-y", type=int, default=356)
    parser.add_argument("--padding", type=int, default=14)
    parser.add_argument("--lower-percentile", type=float, default=0.0)
    parser.add_argument("--upper-percentile", type=float, default=100.0)
    parser.add_argument("--no-regions", action="store_true")
    args = parser.parse_args()
    canvas = cv2.imread(str(args.background), cv2.IMREAD_COLOR)
    if canvas is None:
        raise FileNotFoundError(args.background)
    groups = {"top": [], "bottom": []}
    total = 0
    for manifest in args.assets_root.rglob("*manifest.csv"):
        with manifest.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                try:
                    asset_path = Path(row["output"])
                    if not asset_path.is_absolute():
                        asset_path = Path.cwd() / asset_path
                    asset = cv2.imread(str(asset_path), cv2.IMREAD_UNCHANGED)
                    x, y, w, h = (int(row[key]) for key in ("crop_x", "crop_y", "crop_w", "crop_h"))
                    if asset is None or asset.shape[2] != 4:
                        continue
                    composite(canvas, asset, x, y)
                    groups["top" if y + h / 2 < args.split_y else "bottom"].append((x, y, x + w, y + h))
                    total += 1
                except (KeyError, TypeError, ValueError):
                    continue
    for name, boxes in groups.items():
        if args.no_regions:
            continue
        x0 = max(0, round(np.percentile([box[0] for box in boxes], args.lower_percentile)) - args.padding)
        y0 = max(0, round(np.percentile([box[1] for box in boxes], args.lower_percentile)) - args.padding)
        x1 = min(canvas.shape[1] - 1, round(np.percentile([box[2] for box in boxes], args.upper_percentile)) + args.padding)
        y1 = min(canvas.shape[0] - 1, round(np.percentile([box[3] for box in boxes], args.upper_percentile)) + args.padding)
        color = (0, 215, 255) if name == "top" else (255, 255, 0)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 3, cv2.LINE_AA)
        cv2.putText(canvas, f"{name}: {len(boxes)} assets", (x0 + 6, y0 + 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), canvas)
    print(f"Overlaid {total} assets: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
