from __future__ import annotations

"""Keyboard reviewer for asset candidates with optional existing-asset references."""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


SHAPES = ("circle", "rectangle", "triangle")
KEY_TO_STATUS = {ord("g"): "good", ord("b"): "bad", ord("s"): "hold", ord("h"): "hold"}


def grouped_pngs(root: Path) -> list[tuple[str, Path]]:
    items = [(shape, path) for shape in SHAPES for path in sorted((root / shape).glob("*.png"))]
    return items or [(root.name, path) for path in sorted(root.glob("*.png"))]


def load_source_positions(input_root: Path) -> dict[str, tuple[int, int]]:
    """Map each cropped candidate to its original crop top-left in the source frame."""
    positions: dict[str, tuple[int, int]] = {}
    for manifest in input_root.rglob("*manifest.csv"):
        with manifest.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                try:
                    output = Path(row["output"])
                    if not output.is_absolute():
                        output = Path.cwd() / output
                    positions[str(output.resolve())] = (int(row["crop_x"]), int(row["crop_y"]))
                except (KeyError, TypeError, ValueError):
                    continue
    return positions


def read_decisions(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as file:
        return {row["candidate"]: row["status"] for row in csv.DictReader(file)}


def write_decisions(path: Path, decisions: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("candidate", "status", "updated_at_utc"))
        writer.writeheader()
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for candidate, status in sorted(decisions.items()):
            writer.writerow({"candidate": candidate, "status": status, "updated_at_utc": now})


def checkerboard(height: int, width: int) -> np.ndarray:
    image = np.full((height, width, 3), 225, dtype=np.uint8)
    image[((np.indices((height, width)).sum(axis=0) // 18) % 2) == 0] = 185
    return image


def display_array(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))))
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    y, x = (height - resized.shape[0]) // 2, (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def display_image(path: Path, width: int, height: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Cannot read {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        background = checkerboard(image.shape[0], image.shape[1]).astype(np.float32)
        image = (image[:, :, :3].astype(np.float32) * alpha + background * (1 - alpha)).astype(np.uint8)
    else:
        image = image[:, :, :3]
    return display_array(image, width, height)


def composite_on_board(background: np.ndarray, asset_path: Path, x0: int, y0: int) -> np.ndarray:
    asset = cv2.imread(str(asset_path), cv2.IMREAD_UNCHANGED)
    if asset is None:
        raise RuntimeError(f"Cannot read {asset_path}")
    if asset.ndim != 3 or asset.shape[2] != 4:
        raise ValueError(f"Candidate must be RGBA PNG: {asset_path}")
    result = background.copy()
    asset_h, asset_w = asset.shape[:2]
    x1, y1 = x0 + asset_w, y0 + asset_h
    if x0 < 0 or y0 < 0 or x1 > result.shape[1] or y1 > result.shape[0]:
        raise ValueError(f"Candidate does not fit at review position: {asset_path.name}")
    alpha = asset[:, :, 3:4].astype(np.float32) / 255.0
    foreground = asset[:, :, :3].astype(np.float32)
    region = result[y0:y1, x0:x1].astype(np.float32)
    result[y0:y1, x0:x1] = (foreground * alpha + region * (1.0 - alpha)).astype(np.uint8)
    return result


def text(canvas: np.ndarray, value: str, origin: tuple[int, int], color: tuple[int, int, int] = (20, 20, 20), scale: float = 0.65) -> None:
    cv2.putText(canvas, value, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def make_candidate_view(shape: str, candidate: Path, position: int, total: int, status: str, background: np.ndarray, review_position: tuple[int, int]) -> np.ndarray:
    width, height, header = 1440, 920, 94
    canvas = np.full((height, width, 3), 238, dtype=np.uint8)
    composited = composite_on_board(background, candidate, *review_position)
    candidate_area = display_array(composited, width - 16, height - header - 18)
    canvas[header + 8 : header + 8 + candidate_area.shape[0], 8 : 8 + candidate_area.shape[1]] = candidate_area
    text(canvas, f"Candidate {position + 1}/{total} | {shape} | original crop {review_position[0]},{review_position[1]} | {candidate.name}", (18, 30), scale=0.48)
    color = {"good": (30, 135, 30), "bad": (30, 30, 210), "hold": (25, 120, 220)}.get(status, (90, 90, 90))
    text(canvas, f"status: {status}", (18, 62), color)
    text(canvas, "G good   B bad   S/H hold   ←/→ move   R clear   V reference   Q quit", (480, 62), scale=0.52)
    return canvas


def make_reference_view(shape: str, references: list[Path]) -> np.ndarray:
    width, height, columns, tile_width, tile_height = 1440, 920, 5, 270, 200
    canvas = np.full((height, width, 3), 238, dtype=np.uint8)
    text(canvas, f"Existing {shape} assets — reference only", (18, 35))
    text(canvas, "Press V to return to the candidate", (18, 67), scale=0.52)
    for index, reference in enumerate(references):
        row, col = divmod(index, columns)
        y, x = 92 + row * 265, 18 + col * 280
        tile = display_image(reference, tile_width, tile_height)
        canvas[y : y + tile_height, x : x + tile_width] = tile
        text(canvas, reference.name, (x + 4, y + tile_height + 24), scale=0.45)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description="Review PNG candidates: G=good, B=bad, S/H=hold; decisions persist to CSV.")
    parser.add_argument("--input-root", type=Path, required=True, help="Candidate folder or root with circle/rectangle/triangle subfolders")
    parser.add_argument("--reference-root", type=Path, default=Path("assets"), help="Existing assets root; press V to show same-shape references")
    parser.add_argument("--background", type=Path, default=Path("review/whiteboard_last_frame.png"), help="Clean whiteboard frame used behind every candidate")
    parser.add_argument("--position", default="450,120", help="Fallback crop top-left x,y when no source position is available")
    parser.add_argument("--decisions", type=Path, default=Path("review/asset_decisions.csv"))
    parser.add_argument("--start-unreviewed", action="store_true", help="Open at the first candidate without a saved decision")
    args = parser.parse_args()

    candidates = grouped_pngs(args.input_root)
    if not candidates:
        raise FileNotFoundError(f"No PNG files found under {args.input_root}")
    background = cv2.imread(str(args.background), cv2.IMREAD_COLOR)
    if background is None:
        raise FileNotFoundError(f"Cannot read review background: {args.background}")
    try:
        review_position = tuple(int(value.strip()) for value in args.position.split(","))
        if len(review_position) != 2:
            raise ValueError
    except ValueError as exc:
        raise ValueError("--position must be x,y (for example 500,140)") from exc
    references = {shape: sorted((args.reference_root / shape).glob("*.png")) for shape in SHAPES}
    source_positions = load_source_positions(args.input_root)
    decisions = read_decisions(args.decisions)
    index = next((i for i, (_, path) in enumerate(candidates) if str(path.resolve()) not in decisions), 0) if args.start_unreviewed else 0
    window = "Asset candidate reviewer"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1440, 920)
    showing_reference = False
    while True:
        shape, path = candidates[index]
        if showing_reference:
            cv2.imshow(window, make_reference_view(shape, references.get(shape, [])))
        else:
            status = decisions.get(str(path.resolve()), "unreviewed")
            candidate_position = source_positions.get(str(path.resolve()), review_position)
            cv2.imshow(window, make_candidate_view(shape, path, index, len(candidates), status, background, candidate_position))
        key = cv2.waitKeyEx(0)
        if key in (ord("q"), 27):
            break
        if key == ord("v"):
            showing_reference = not showing_reference
            continue
        if showing_reference:
            continue
        if key in KEY_TO_STATUS:
            decisions[str(path.resolve())] = KEY_TO_STATUS[key]
            write_decisions(args.decisions, decisions)
            index = min(index + 1, len(candidates) - 1)
        elif key in (ord("r"),):
            decisions.pop(str(path.resolve()), None)
            write_decisions(args.decisions, decisions)
        elif key in (2424832, 81):  # left arrow; Windows / OpenCV variants
            index = max(index - 1, 0)
        elif key in (2555904, 83):  # right arrow; Windows / OpenCV variants
            index = min(index + 1, len(candidates) - 1)
    cv2.destroyAllWindows()
    counts = {status: list(decisions.values()).count(status) for status in ("good", "bad", "hold")}
    print(f"Saved decisions: {counts}; file: {args.decisions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
