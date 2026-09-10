from __future__ import annotations

"""Create contact sheets and a CSV report for reviewing shape-asset candidates.

The input root may contain circle/, rectangle/, and triangle/ folders.  It also
works for a single folder of PNG files.  RGBA assets receive alpha statistics;
ordinary source frames are listed as opaque images.
"""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


SHAPES = ("circle", "rectangle", "triangle")


def files_by_group(input_root: Path) -> dict[str, list[Path]]:
    groups = {shape: sorted((input_root / shape).glob("*.png")) for shape in SHAPES}
    groups = {shape: files for shape, files in groups.items() if files}
    return groups or {input_root.name: sorted(input_root.glob("*.png"))}


def inspect_image(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError("OpenCV could not read image")

    height, width = image.shape[:2]
    has_alpha = image.ndim == 3 and image.shape[2] == 4
    if has_alpha:
        alpha = image[:, :, 3]
        opaque_ratio = float(np.count_nonzero(alpha == 255)) / alpha.size
        transparent_ratio = float(np.count_nonzero(alpha == 0)) / alpha.size
        visible_ratio = float(np.count_nonzero(alpha > 0)) / alpha.size
        foreground = image[:, :, :3].astype(np.float32)
        checker = np.full((height, width, 3), 224, dtype=np.uint8)
        checker[((np.indices((height, width)).sum(axis=0) // 14) % 2) == 0] = 190
        alpha_float = (alpha.astype(np.float32) / 255.0)[:, :, None]
        preview = (foreground * alpha_float + checker.astype(np.float32) * (1.0 - alpha_float)).astype(np.uint8)
    else:
        opaque_ratio, transparent_ratio, visible_ratio = 1.0, 0.0, 1.0
        preview = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image[:, :, :3]
    return preview, {
        "file": path.name,
        "width": width,
        "height": height,
        "channels": image.shape[2] if image.ndim == 3 else 1,
        "has_alpha": has_alpha,
        "visible_ratio": round(visible_ratio, 6),
        "transparent_ratio": round(transparent_ratio, 6),
        "opaque_ratio": round(opaque_ratio, 6),
    }


def thumbnail(image: np.ndarray, label: str, tile_width: int, tile_height: int) -> np.ndarray:
    canvas = np.full((tile_height, tile_width, 3), 245, dtype=np.uint8)
    usable_height = tile_height - 28
    scale = min(tile_width / image.shape[1], usable_height / image.shape[0])
    resized = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))))
    y = (usable_height - resized.shape[0]) // 2
    x = (tile_width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    cv2.putText(canvas, label, (6, tile_height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)
    return canvas


def write_sheets(group: str, paths: list[Path], output_root: Path, columns: int, rows: int, tile_width: int, tile_height: int) -> list[dict[str, object]]:
    per_sheet = columns * rows
    report_rows: list[dict[str, object]] = []
    for sheet_index, start in enumerate(range(0, len(paths), per_sheet), start=1):
        subset = paths[start : start + per_sheet]
        canvas = np.full((rows * tile_height, columns * tile_width, 3), 220, dtype=np.uint8)
        for offset, path in enumerate(subset):
            try:
                image, record = inspect_image(path)
                record.update({"shape": group, "status": "ok"})
                tile = thumbnail(image, f"{start + offset + 1:03d}  {path.name}", tile_width, tile_height)
            except Exception as exc:
                record = {"shape": group, "file": path.name, "status": f"error: {exc}", "width": "", "height": "", "channels": "", "has_alpha": "", "visible_ratio": "", "transparent_ratio": "", "opaque_ratio": ""}
                tile = thumbnail(np.zeros((1, 1, 3), dtype=np.uint8), f"ERROR  {path.name}", tile_width, tile_height)
            report_rows.append(record)
            y, x = divmod(offset, columns)
            canvas[y * tile_height : (y + 1) * tile_height, x * tile_width : (x + 1) * tile_width] = tile
        sheet_path = output_root / f"{group}_sheet_{sheet_index:02d}.jpg"
        cv2.imwrite(str(sheet_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return report_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate contact sheets and quality_report.csv for PNG asset review.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--tile-width", type=int, default=240)
    parser.add_argument("--tile-height", type=int, default=170)
    args = parser.parse_args()

    groups = files_by_group(args.input_root)
    if not any(groups.values()):
        raise FileNotFoundError(f"No PNG files found under {args.input_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    report_rows = []
    for group, paths in groups.items():
        report_rows.extend(write_sheets(group, paths, args.output_root, args.columns, args.rows, args.tile_width, args.tile_height))
        print(f"{group}: {len(paths)} image(s)")
    fieldnames = ["shape", "file", "status", "width", "height", "channels", "has_alpha", "visible_ratio", "transparent_ratio", "opaque_ratio"]
    with (args.output_root / "quality_report.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)
    print(f"Wrote {len(report_rows)} report rows and contact sheets to {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
