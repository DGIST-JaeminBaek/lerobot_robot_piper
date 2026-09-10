from __future__ import annotations

"""0727 첫 프레임에서 원 도형 주변을 자동 crop하는 자산 준비용 스크립트.

여기서 만든 crop 이미지는 이후 배경 제거 웹 도구를 거쳐 assets/circle/*.png로 들어간다.
현재 0802~0805 영상 증강 자체를 수행하는 파일은 아니다.
"""

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class CircleCandidate:
    score: float
    x: int
    y: int
    w: int
    h: int
    fill_ratio: float
    circularity: float


def find_circle_candidate(
    image_bgr: np.ndarray,
    dark_threshold: int,
    min_size: int,
    max_size: int,
    min_aspect: float,
    max_aspect: float,
) -> CircleCandidate:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mask = np.where(gray < dark_threshold, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[CircleCandidate] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < min_size or h < min_size or w > max_size or h > max_size:
            continue
        aspect = w / float(h)
        if not (min_aspect <= aspect <= max_aspect):
            continue

        roi = mask[y : y + h, x : x + w]
        fill_ratio = cv2.countNonZero(roi) / float(w * h)
        if not (0.008 <= fill_ratio <= 0.32):
            continue

        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4.0 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0
        aspect_bonus = max(0.0, 1.0 - abs(aspect - 1.0)) * 2.0
        thin_outline = max(0.0, 0.22 - fill_ratio) * 4.0
        size_bonus = min(w, h) / float(max_size)
        score = aspect_bonus + thin_outline + circularity * 2.0 + size_bonus
        candidates.append(CircleCandidate(score, x, y, w, h, fill_ratio, circularity))

    if not candidates:
        raise RuntimeError("No circle-like dark outline found")
    return max(candidates, key=lambda item: item.score)


def crop_with_margin(image_bgr: np.ndarray, candidate: CircleCandidate, margin: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    img_h, img_w = image_bgr.shape[:2]
    x0 = max(0, candidate.x - margin)
    y0 = max(0, candidate.y - margin)
    x1 = min(img_w, candidate.x + candidate.w + margin)
    y1 = min(img_h, candidate.y + candidate.h + margin)
    return image_bgr[y0:y1, x0:x1], (x0, y0, x1, y1)


def make_transparent_circle(crop_bgr: np.ndarray, dark_threshold: int, dilate: int) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    alpha = np.where(gray < dark_threshold, 255, 0).astype(np.uint8)
    if dilate > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1))
        alpha = cv2.dilate(alpha, kernel)
    out = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
    out[:, :, 3] = alpha
    return out


def process_file(
    input_path: Path,
    output_path: Path,
    dark_threshold: int,
    min_size: int,
    max_size: int,
    min_aspect: float,
    max_aspect: float,
    margin: int,
    transparent: bool,
    alpha_dilate: int,
    backup_dir: Path | None,
    debug_path: Path | None,
) -> dict[str, str | int | float]:
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {input_path}")

    if backup_dir is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / input_path.name
        if not backup_path.exists():
            shutil.copy2(input_path, backup_path)

    candidate = find_circle_candidate(image, dark_threshold, min_size, max_size, min_aspect, max_aspect)
    crop, (x0, y0, x1, y1) = crop_with_margin(image, candidate, margin)
    output = make_transparent_circle(crop, dark_threshold, alpha_dilate) if transparent else crop
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), output):
        raise RuntimeError(f"Could not write output: {output_path}")

    if debug_path is not None:
        preview = image.copy()
        cv2.rectangle(preview, (candidate.x, candidate.y), (candidate.x + candidate.w, candidate.y + candidate.h), (0, 0, 255), 2)
        cv2.rectangle(preview, (x0, y0), (x1, y1), (255, 0, 255), 2)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_path), preview)

    return {
        "input": str(input_path),
        "output": str(output_path),
        "x": candidate.x,
        "y": candidate.y,
        "w": candidate.w,
        "h": candidate.h,
        "crop_x": x0,
        "crop_y": y0,
        "crop_w": x1 - x0,
        "crop_h": y1 - y0,
        "score": round(candidate.score, 6),
        "fill_ratio": round(candidate.fill_ratio, 6),
        "circularity": round(candidate.circularity, 6),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crop circle source frames around the drawn circle without assuming its position.")
    parser.add_argument("--input-dir", type=Path, default=Path("source_frames/0727/circle"))
    parser.add_argument("--output-dir", type=Path, default=Path("source_frames/0727/circle"))
    parser.add_argument("--pattern", default="*.png")
    parser.add_argument("--start-index", type=int, default=11)
    parser.add_argument("--end-index", type=int, default=20)
    parser.add_argument("--dark-threshold", type=int, default=115)
    parser.add_argument("--min-size", type=int, default=35)
    parser.add_argument("--max-size", type=int, default=260)
    parser.add_argument("--min-aspect", type=float, default=0.65)
    parser.add_argument("--max-aspect", type=float, default=1.45)
    parser.add_argument("--margin", type=int, default=18)
    parser.add_argument("--transparent", action="store_true", help="Write BGRA transparent PNGs. Default is regular cropped PNG.")
    parser.add_argument("--alpha-dilate", type=int, default=1)
    parser.add_argument("--backup-dir", type=Path, default=Path("source_frames/0727/circle_raw_11_20"))
    parser.add_argument("--debug-dir", type=Path, default=Path("source_frames/0727/circle_crop_debug"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    files = sorted(args.input_dir.glob(args.pattern))
    selected = [path for path in files if path.name[:2].isdigit() and args.start_index <= int(path.name[:2]) <= args.end_index]
    if not selected:
        raise FileNotFoundError(f"No input PNGs found for index range {args.start_index}-{args.end_index} in {args.input_dir}")

    rows: list[dict[str, str | int | float]] = []
    failures: list[tuple[Path, str]] = []
    for input_path in selected:
        output_path = args.output_dir / input_path.name
        debug_path = args.debug_dir / f"{input_path.stem}_debug.png" if args.debug_dir else None
        try:
            row = process_file(
                input_path,
                output_path,
                args.dark_threshold,
                args.min_size,
                args.max_size,
                args.min_aspect,
                args.max_aspect,
                args.margin,
                args.transparent,
                args.alpha_dilate,
                args.backup_dir,
                debug_path,
            )
            rows.append(row)
            print(f"wrote {output_path}")
        except Exception as exc:
            failures.append((input_path, str(exc)))
            print(f"failed {input_path}: {exc}")

    manifest_path = args.output_dir / f"circle_crop_{args.start_index}_{args.end_index}_manifest.csv"
    fieldnames = ["input", "output", "x", "y", "w", "h", "crop_x", "crop_y", "crop_w", "crop_h", "score", "fill_ratio", "circularity"]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Cropped {len(rows)} file(s), failed {len(failures)}.")
    print(f"Manifest: {manifest_path}")
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
