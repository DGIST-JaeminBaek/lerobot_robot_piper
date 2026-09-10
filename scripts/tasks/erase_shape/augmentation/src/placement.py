from __future__ import annotations

"""투명 PNG asset을 safe_mask 안에 배치할 좌표를 선택하는 모듈.

augment_whiteboard.py가 만든 safe_mask를 입력으로 받아,
PNG alpha footprint 전체가 안전 영역 안에 들어가는 top-left 좌표를 찾는다.
현재 0802~0805 최종 증강에서는 y축 top/bottom 제약도 여기서 bbox 기준으로 한 번 더 검사한다.
"""

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


SHAPES = ("circle", "triangle", "square", "rectangle")


@dataclass(frozen=True)
class AssetPlacement:
    asset_path: Path
    image_rgba: np.ndarray
    footprint: np.ndarray
    x: int
    y: int
    scale: float
    rotation_deg: float
    brightness: float
    blur_sigma: float
    valid_placements: int
    selection_mode: str = "safe_fallback"


def choose_shape(task: str | None, explicit_shape: str | None, rng: random.Random) -> str | None:
    if explicit_shape:
        return explicit_shape.lower()
    if not task:
        return None
    lower = task.lower()
    target = next((shape for shape in SHAPES if shape in lower), None)
    candidates = [shape for shape in SHAPES if shape != target]
    return rng.choice(candidates) if candidates else None


def list_assets(assets_root: Path, shape: str | None) -> list[Path]:
    # shape가 지정되면 assets/<shape> 폴더만 사용한다. 다른 shape로 fallback하지 않는다.
    roots = [assets_root / shape] if shape else [assets_root]
    files: list[Path] = []
    for root in roots:
        if root.is_dir():
            files.extend(sorted(root.glob("*.png")))
    if not files and shape is None and assets_root.is_dir():
        files = sorted(assets_root.rglob("*.png"))
    return files


def load_asset_source_positions(index_path: Path | None) -> dict[Path, tuple[float, float, int, int]]:
    if index_path is None:
        return {}
    positions: dict[Path, tuple[float, float, int, int]] = {}
    with index_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            positions[Path(row["asset_path"]).resolve()] = (
                float(row["source_center_x"]),
                float(row["source_center_y"]),
                int(float(row.get("source_x", row["source_center_x"]))),
                int(float(row.get("source_y", row["source_center_y"]))),
            )
    return positions


def load_rgba_png(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not read PNG asset: {path}")
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f"PNG asset must be RGBA/BGRA with alpha channel: {path}")
    return image


def transform_asset(image_rgba: np.ndarray, scale: float, rotation_deg: float, brightness: float, blur_sigma: float) -> np.ndarray:
    # 실제 촬영 환경처럼 보이도록 크기, 회전, 밝기, blur를 약간 변형한다.
    h, w = image_rgba.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, rotation_deg, scale)
    cos_v, sin_v = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int(h * sin_v + w * cos_v)
    new_h = int(h * cos_v + w * sin_v)
    matrix[0, 2] += new_w / 2.0 - center[0]
    matrix[1, 2] += new_h / 2.0 - center[1]
    warped = cv2.warpAffine(image_rgba, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    warped = warped.astype(np.float32)
    warped[:, :, :3] *= brightness
    warped[:, :, :3] = np.clip(warped[:, :, :3], 0, 255)
    warped = warped.astype(np.uint8)
    if blur_sigma > 0:
        bgr = cv2.GaussianBlur(warped[:, :, :3], (0, 0), blur_sigma)
        warped[:, :, :3] = bgr
    return warped


def footprint_from_alpha(image_rgba: np.ndarray, placement_margin: int) -> np.ndarray:
    # 투명 PNG의 alpha > 0 영역이 실제 충돌 검사 footprint가 된다.
    footprint = np.where(image_rgba[:, :, 3] > 0, 255, 0).astype(np.uint8)
    if placement_margin > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (placement_margin * 2 + 1, placement_margin * 2 + 1))
        footprint = cv2.dilate(footprint, kernel)
    ys, xs = np.where(footprint > 0)
    if len(xs) == 0:
        raise ValueError("PNG alpha footprint is empty")
    return footprint


def find_valid_top_lefts(safe_mask: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    # footprint 전체가 safe_mask 안에 들어갈 수 있는 모든 top-left 후보를 찾는다.
    fh, fw = footprint.shape[:2]
    sh, sw = safe_mask.shape[:2]
    if fh > sh or fw > sw:
        raise ValueError("PNG footprint is larger than the video frame/whiteboard")
    footprint_bool = (footprint > 0).astype(np.uint8)
    safe_bool = (safe_mask > 0).astype(np.uint8)
    required = int(np.count_nonzero(footprint_bool))
    score = cv2.matchTemplate(safe_bool, footprint_bool, cv2.TM_CCORR)
    ys, xs = np.where(score >= required - 0.5)
    return np.column_stack([xs, ys]).astype(np.int32)


def choose_placement(
    assets_root: Path,
    safe_mask: np.ndarray,
    task: str | None,
    shape: str | None,
    seed: int,
    scale_range: tuple[float, float],
    rotation_range: tuple[float, float],
    brightness_range: tuple[float, float],
    blur_range: tuple[float, float],
    placement_margin: int,
    max_asset_tries: int = 50,
    asset_path: Path | None = None,
    min_x: int | None = None,
    max_x: int | None = None,
    min_y: int | None = None,
    max_y: int | None = None,
    asset_index: Path | None = None,
    excluded_assets: set[Path] | None = None,
) -> AssetPlacement:
    rng = random.Random(seed)
    selected_shape = choose_shape(task, shape, rng)
    assets = [asset_path] if asset_path is not None else list_assets(assets_root, selected_shape)
    excluded = {path.resolve() for path in (excluded_assets or set())}
    assets = [path for path in assets if path.resolve() not in excluded]
    if not assets:
        raise FileNotFoundError(f"No PNG assets found under {assets_root}" + (f" for shape {selected_shape}" if selected_shape else ""))

    source_positions = load_asset_source_positions(asset_index)
    shuffled = assets[:]
    rng.shuffle(shuffled)

    # Priority 1: preserve the asset's original board location exactly whenever
    # that complete alpha footprint is safe for this target video.  No visual
    # transform is applied here because it would invalidate that source location.
    if asset_index is not None and asset_path is None:
        exact_matches: list[AssetPlacement] = []
        for candidate_asset in shuffled:
            source = source_positions.get(candidate_asset.resolve())
            if source is None:
                continue
            try:
                image = load_rgba_png(candidate_asset)
                footprint = footprint_from_alpha(image, placement_margin)
                candidates = find_valid_top_lefts(safe_mask, footprint)
                source_x, source_y = source[2], source[3]
                if not np.any((candidates[:, 0] == source_x) & (candidates[:, 1] == source_y)):
                    continue
                image_h, image_w = image.shape[:2]
                if min_x is not None and source_x < min_x:
                    continue
                if max_x is not None and source_x + image_w > max_x:
                    continue
                if min_y is not None and source_y < min_y:
                    continue
                if max_y is not None and source_y + image_h > max_y:
                    continue
                exact_matches.append(AssetPlacement(candidate_asset, image, footprint, source_x, source_y, 1.0, 0.0, 1.0, 0.0, int(len(candidates)), "source_position"))
            except Exception:
                continue
        if exact_matches:
            return rng.choice(exact_matches)
    if asset_path is not None:
        shuffled = assets * max(1, max_asset_tries)
    attempts = max(1, len(shuffled)) if asset_index is not None else min(max_asset_tries, max(1, len(shuffled)))
    last_error: Exception | None = None
    viable: list[tuple[float, AssetPlacement]] = []
    for asset_path in shuffled[:attempts]:
        try:
            image = load_rgba_png(asset_path)
            scale = rng.uniform(*scale_range)
            rotation = rng.uniform(*rotation_range)
            brightness = rng.uniform(*brightness_range)
            blur = rng.uniform(*blur_range)
            transformed = transform_asset(image, scale, rotation, brightness, blur)
            footprint = footprint_from_alpha(transformed, placement_margin)
            candidates = find_valid_top_lefts(safe_mask, footprint)
            if min_x is not None or max_x is not None:
                image_h, image_w = transformed.shape[:2]
                keep = np.ones(len(candidates), dtype=bool)
                if min_x is not None:
                    keep &= candidates[:, 0] >= min_x
                if max_x is not None:
                    keep &= candidates[:, 0] + image_w <= max_x
                candidates = candidates[keep]
            if min_y is not None or max_y is not None:
                # top/bottom 영역 경계선을 PNG bbox가 넘어가지 않도록 한 번 더 필터링한다.
                image_h, image_w = transformed.shape[:2]
                keep = np.ones(len(candidates), dtype=bool)
                if min_y is not None:
                    keep &= candidates[:, 1] >= min_y
                if max_y is not None:
                    keep &= candidates[:, 1] + image_h <= max_y
                candidates = candidates[keep]
            if len(candidates) == 0:
                continue
            if asset_index is None or asset_path.resolve() not in source_positions:
                x, y = map(int, rng.choice(candidates))
                return AssetPlacement(asset_path, transformed, footprint, x, y, scale, rotation, brightness, blur, int(len(candidates)))
            source_x, source_y, _, _ = source_positions[asset_path.resolve()]
            centers = candidates.astype(np.float32) + np.array([transformed.shape[1] / 2.0, transformed.shape[0] / 2.0], dtype=np.float32)
            nearest = int(np.argmin(np.sum((centers - np.array([source_x, source_y], dtype=np.float32)) ** 2, axis=1)))
            x, y = map(int, candidates[nearest])
            distance = float(np.linalg.norm(centers[nearest] - np.array([source_x, source_y], dtype=np.float32)))
            viable.append((distance, AssetPlacement(asset_path, transformed, footprint, x, y, scale, rotation, brightness, blur, int(len(candidates)), "safe_fallback")))
        except Exception as exc:  # keep trying other assets, then report the last useful failure
            last_error = exc
    if viable:
        # Safe locations are computed first; select the asset whose original board
        # position is closest to one of those valid locations.
        return min(viable, key=lambda item: item[0])[1]
    guidance = (
        "No valid safe placement found. Try reducing PNG size, placement margin, motion dilation, "
        "or drawing dilation; also inspect debug masks and manual exclusion regions."
    )
    if last_error is not None:
        raise RuntimeError(f"{guidance} Last asset error: {last_error}") from last_error
    raise RuntimeError(guidance)
