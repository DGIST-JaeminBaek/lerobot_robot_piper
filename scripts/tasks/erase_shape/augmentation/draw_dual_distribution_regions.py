from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def load_groups(root: Path, split_y: int) -> dict[str, list[tuple[int, int, int, int, float, float]]]:
    groups = {"top": [], "bottom": []}
    for manifest in root.rglob("*manifest.csv"):
        with manifest.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                try:
                    x, y, w, h = (int(row[key]) for key in ("crop_x", "crop_y", "crop_w", "crop_h"))
                    cx, cy = x + w / 2, y + h / 2
                    groups["top" if cy < split_y else "bottom"].append((x, y, x + w, y + h, cx, cy))
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
    for *_rest, cx, cy in boxes:
        density[round(cy), round(cx)] += 1.0
    density = cv2.GaussianBlur(density, (0, 0), 38)
    mask = (density >= density.max() * 0.42).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    label = max(range(1, count), key=lambda idx: stats[idx, cv2.CC_STAT_AREA])
    x, y, w, h = stats[label, :4]
    return max(0, x - padding), max(0, y - padding), x + w + padding, y + h + padding


def draw_box(image: np.ndarray, box: tuple[int, int, int, int], color: tuple[int, int, int], label: str, y_offset: int = 0) -> None:
    x0, y0, x1, y1 = box
    cv2.rectangle(image, (x0, y0), (x1, y1), color, 3, cv2.LINE_AA)
    cv2.putText(image, label, (x0 + 6, y0 + 28 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.63, color, 2, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--input-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-y", type=int, default=356)
    args = parser.parse_args()
    image = cv2.imread(str(args.input_image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.input_image)
    groups = load_groups(args.assets_root, args.split_y)
    colors = {"top": ((0, 215, 255), (0, 100, 255)), "bottom": ((255, 255, 0), (255, 0, 255))}
    for name, boxes in groups.items():
        pbox = percentile_box(boxes, 14)
        dbox = dense_box(boxes, image.shape[:2], 10)
        draw_box(image, pbox, colors[name][0], f"{name} 95% range")
        draw_box(image, dbox, colors[name][1], f"{name} dense cluster", 26)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), image)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
