from __future__ import annotations

"""단일 top 영상에 방해 도형 PNG를 합성하는 CLI.

이 파일은 배치 계획을 만들지 않는다. batch_augment_dates.py가 episode별로 이 파일을 호출한다.

주요 역할:
- ROI, motion, 기존 그림, 수동 제외 영역을 합쳐 unsafe/safe mask를 만든다.
- 분석된 target side(top/bottom)의 반대 영역에만 방해 도형을 배치한다.
- 학습 crop에서 방해 도형이 잘리지 않도록 crop-left 제약을 적용한다.
- augmented.mp4, placement_metadata.json, debug 이미지를 저장한다.
"""

import argparse
import json
import json
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.audio import mux_original_audio
from src.compositing import overlay_mask, write_augmented_video
from src.drawing_detection import detect_existing_drawing_mask
from src.metadata import save_metadata
from src.motion import detect_motion_mask
from src.placement import choose_placement
from src.roi import board_mask, border_mask, load_roi, save_roi, select_roi


def parse_range(value: str, name: str) -> tuple[float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"{name} must be formatted like min,max")
    lo, hi = float(parts[0]), float(parts[1])
    if lo > hi:
        raise argparse.ArgumentTypeError(f"{name} minimum cannot be greater than maximum")
    return lo, hi


def float_range(name: str):
    return lambda value: parse_range(value, name)


def read_first_frame(input_video: Path) -> tuple[np.ndarray, dict[str, int | float]]:
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_video}")
    ok, frame = cap.read()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read first frame: {input_video}")
    if width <= 0 or height <= 0 or fps <= 0:
        raise RuntimeError(f"Invalid video properties: width={width}, height={height}, fps={fps}")
    return frame, {"width": width, "height": height, "fps": fps, "frames": frames}


def load_manual_exclusion(path: Path | None, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if path is None:
        return mask
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for polygon in data.get("polygons", []):
        points = np.asarray(polygon, dtype=np.int32)
        if points.ndim == 2 and points.shape[1] == 2 and len(points) >= 3:
            cv2.fillPoly(mask, [points], 255)
    for rect in data.get("rectangles", []):
        x, y, w, h = map(int, rect)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    return mask


def save_debug_images(
    debug_dir: Path,
    first_frame: np.ndarray,
    masks: dict[str, np.ndarray],
    placement_x: int,
    placement_y: int,
    overlay_shape: tuple[int, int],
) -> None:
    # 팀원 검수 때는 debug/placement_preview.png를 가장 먼저 보면 된다.
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / "first_frame.png"), first_frame)
    for name, mask in masks.items():
        cv2.imwrite(str(debug_dir / f"{name}.png"), mask)
    preview = first_frame.copy()
    preview = overlay_mask(preview, masks["unsafe_mask"], (0, 0, 255), 0.35)
    preview = overlay_mask(preview, masks["safe_mask"], (0, 255, 0), 0.25)
    h, w = overlay_shape[:2]
    cv2.rectangle(preview, (placement_x, placement_y), (placement_x + w, placement_y + h), (255, 0, 255), 2)
    cv2.circle(preview, (placement_x, placement_y), 4, (255, 0, 255), -1)
    cv2.imwrite(str(debug_dir / "placement_preview.png"), preview)


def build_placement_constraint_mask(
    shape: tuple[int, int],
    roi_points: np.ndarray,
    placement_side: str | None,
    crop_left: int | None,
    crop_margin: int,
) -> np.ndarray:
    # placement_side가 top/bottom이면 board ROI의 y 중앙선을 기준으로 영역을 나눈다.
    # 현재 0802~0805 최종 기준은 보드가 90도 돌아간 것을 반영한 top/bottom 분할이다.
    height, width = shape
    mask = np.full((height, width), 255, dtype=np.uint8)
    if placement_side:
        board_min_x = float(np.min(roi_points[:, 0]))
        board_max_x = float(np.max(roi_points[:, 0]))
        board_mid_x = int(round((board_min_x + board_max_x) / 2.0))
        board_min_y = float(np.min(roi_points[:, 1]))
        board_max_y = float(np.max(roi_points[:, 1]))
        board_mid_y = int(round((board_min_y + board_max_y) / 2.0))
        if placement_side == "left":
            mask[:, board_mid_x:] = 0
        elif placement_side == "right":
            mask[:, :board_mid_x] = 0
        elif placement_side == "top":
            mask[board_mid_y:, :] = 0
        elif placement_side == "bottom":
            mask[:board_mid_y, :] = 0
        else:
            raise ValueError(f"Unsupported placement side: {placement_side}")
    if crop_left is not None:
        # LeRobot 학습 crop에서 왼쪽이 잘리는 문제를 막기 위한 x축 가시성 제약이다.
        min_visible_x = max(0, crop_left + crop_margin)
        mask[:, :min_visible_x] = 0
    return mask


def visible_ratio_after_crop(overlay_bgra: np.ndarray, x: int, crop_left: int | None, crop_margin: int) -> float | None:
    # metadata 검수용 값이다. 정상 생성 결과에서는 1.0이어야 한다.
    if crop_left is None:
        return None
    alpha = overlay_bgra[:, :, 3] > 0
    total = int(np.count_nonzero(alpha))
    if total == 0:
        return 0.0
    xs = np.where(alpha)[1] + x
    min_visible_x = crop_left + crop_margin
    visible = int(np.count_nonzero(xs >= min_visible_x))
    return visible / float(total)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Add a safe distractor shape to a whiteboard robot video.")
    parser.add_argument("--input", required=True, type=Path, help="Input MP4 path")
    parser.add_argument("--assets", required=True, type=Path, help="PNG asset root, optionally with circle/triangle/square subfolders")
    parser.add_argument("--asset", type=Path, default=None, help="Use this exact RGBA PNG asset instead of random asset selection")
    parser.add_argument("--asset-index", type=Path, default=None, help="CSV with source coordinates; select the asset nearest a valid safe location")
    parser.add_argument("--excluded-assets-file", type=Path, default=None, help="JSON list of assets already used in this batch")
    parser.add_argument("--output", required=True, type=Path, help="Output MP4 path")
    parser.add_argument("--task", default=None, help='Task text, e.g. "erase the circle"; used to avoid selecting the target shape')
    parser.add_argument("--shape", default=None, help="Explicit distractor shape folder/name")
    parser.add_argument("--roi-json", type=Path, default=None, help="ROI JSON to load or save")
    parser.add_argument("--manual-exclusion-json", type=Path, default=None, help="Optional JSON with polygons/rectangles to exclude")
    parser.add_argument("--placement-side", choices=["left", "right", "top", "bottom"], default=None, help="Restrict the distractor to one half of the board ROI")
    parser.add_argument("--target-side", choices=["left", "right", "top", "bottom"], default=None, help="Metadata only: detected side of the target shape")
    parser.add_argument("--target-center-x", type=float, default=None, help="Metadata only: detected target center x")
    parser.add_argument("--target-center-y", type=float, default=None, help="Metadata only: detected target center y")
    parser.add_argument("--crop-left", type=int, default=None, help="Require transformed PNG bbox to start at or after this crop x plus crop margin")
    parser.add_argument("--crop-margin", type=int, default=0, help="Extra pixels added to crop-left visibility constraint")
    parser.add_argument("--sample-stride", type=int, default=3)
    parser.add_argument("--pixel-threshold", type=int, default=18)
    parser.add_argument("--ever-changed-threshold", type=int, default=35)
    parser.add_argument("--changed-ratio", type=float, default=0.005)
    parser.add_argument("--motion-dilate", type=int, default=20)
    parser.add_argument("--drawing-threshold", type=int, default=155)
    parser.add_argument("--drawing-dilate", type=int, default=15)
    parser.add_argument("--drawing-min-area", type=int, default=20)
    parser.add_argument("--placement-margin", type=int, default=20)
    parser.add_argument("--board-border-margin", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale-range", type=float_range("scale-range"), default=(0.90, 1.10), help="min,max")
    parser.add_argument("--rotation-range", type=float_range("rotation-range"), default=(-10.0, 10.0), help="min,max degrees")
    parser.add_argument("--brightness-range", type=float_range("brightness-range"), default=(0.85, 1.10), help="min,max")
    parser.add_argument("--blur-range", type=float_range("blur-range"), default=(0.0, 0.8), help="min,max sigma")
    parser.add_argument("--stabilize", action="store_true", help="Use ECC alignment while analyzing motion")
    parser.add_argument("--keep-audio", action="store_true", help="Mux original audio back with ffmpeg when available")
    parser.add_argument("--debug", action="store_true", help="Write debug masks and placement preview")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    first_frame, video_info = read_first_frame(args.input)
    height, width = first_frame.shape[:2]

    if args.roi_json and args.roi_json.exists():
        roi_points = load_roi(args.roi_json)
    elif args.roi_json:
        roi_points = select_roi(first_frame, args.roi_json)
    else:
        roi_points = select_roi(first_frame)

    if args.roi_json and not args.roi_json.exists():
        save_roi(args.roi_json, roi_points)

    bmask = board_mask((height, width), roi_points)
    if np.count_nonzero(bmask) < 100:
        raise RuntimeError("Whiteboard ROI is too small or invalid")

    motion_mask, motion_stats = detect_motion_mask(
        str(args.input),
        bmask,
        args.sample_stride,
        args.pixel_threshold,
        args.ever_changed_threshold,
        args.changed_ratio,
        args.motion_dilate,
        args.stabilize,
    )
    drawing_mask = detect_existing_drawing_mask(first_frame, bmask, args.drawing_threshold, args.drawing_dilate, args.drawing_min_area)
    manual_mask = load_manual_exclusion(args.manual_exclusion_json, (height, width))
    placement_constraint_mask = build_placement_constraint_mask((height, width), roi_points, args.placement_side, args.crop_left, args.crop_margin)
    bborder = border_mask(bmask, args.board_border_margin)
    board_outside = cv2.bitwise_not(bmask)
    # unsafe_mask는 방해 도형이 절대 들어가면 안 되는 영역들의 합집합이다.
    unsafe_mask = cv2.bitwise_or(board_outside, motion_mask)
    unsafe_mask = cv2.bitwise_or(unsafe_mask, drawing_mask)
    unsafe_mask = cv2.bitwise_or(unsafe_mask, manual_mask)
    unsafe_mask = cv2.bitwise_or(unsafe_mask, bborder)
    safe_mask = cv2.bitwise_and(bmask, cv2.bitwise_not(unsafe_mask))
    # 마지막에 top/bottom 반대 배치와 crop 가시성 제약을 safe_mask에 반영한다.
    safe_mask = cv2.bitwise_and(safe_mask, placement_constraint_mask)

    min_x = None if args.crop_left is None else args.crop_left + args.crop_margin
    max_y = None
    min_y = None
    if args.placement_side in {"top", "bottom"}:
        board_min_y = float(np.min(roi_points[:, 1]))
        board_max_y = float(np.max(roi_points[:, 1]))
        board_mid_y = int(round((board_min_y + board_max_y) / 2.0))
        if args.placement_side == "top":
            max_y = board_mid_y
        else:
            min_y = board_mid_y

    excluded_assets: set[Path] = set()
    if args.excluded_assets_file is not None and args.excluded_assets_file.exists():
        excluded_assets = {Path(path) for path in json.loads(args.excluded_assets_file.read_text(encoding="utf-8"))}

    placement = choose_placement(
        args.assets,
        safe_mask,
        args.task,
        args.shape,
        args.seed,
        args.scale_range,
        args.rotation_range,
        args.brightness_range,
        args.blur_range,
        args.placement_margin,
        asset_path=args.asset,
        min_x=min_x,
        min_y=min_y,
        max_y=max_y,
        asset_index=args.asset_index,
        excluded_assets=excluded_assets,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    final_output = args.output
    silent_output = args.output
    audio_restored = False
    if args.keep_audio:
        silent_output = Path(tempfile.gettempdir()) / f"{args.output.stem}_silent_{args.seed}.mp4"
    write_info = write_augmented_video(str(args.input), str(silent_output), placement.image_rgba, placement.x, placement.y)
    if args.keep_audio:
        try:
            audio_restored = mux_original_audio(silent_output, args.input, final_output)
        except Exception as exc:
            print(f"Warning: FFmpeg audio mux failed; keeping silent video at {silent_output}. Error: {exc}")
            final_output = silent_output

    debug_dir = args.output.parent / "debug"
    if args.debug:
        save_debug_images(
            debug_dir,
            first_frame,
            {
                "board_mask": bmask,
                "motion_mask": motion_mask,
                "existing_drawing_mask": drawing_mask,
                "manual_exclusion_mask": manual_mask,
                "placement_constraint_mask": placement_constraint_mask,
                "board_border_mask": bborder,
                "unsafe_mask": unsafe_mask,
                "safe_mask": safe_mask,
            },
            placement.x,
            placement.y,
            placement.image_rgba.shape,
        )

    stats: dict[str, Any] = {
        **motion_stats,
        "board_area": int(np.count_nonzero(bmask)),
        "drawing_unsafe_area": int(np.count_nonzero(drawing_mask)),
        "final_safe_area": int(np.count_nonzero(safe_mask)),
        "valid_placements": placement.valid_placements,
    }
    visible_ratio = visible_ratio_after_crop(placement.image_rgba, placement.x, args.crop_left, args.crop_margin)
    inferred_shape = args.shape
    try:
        if inferred_shape is None and placement.asset_path.parent.resolve() != args.assets.resolve():
            inferred_shape = placement.asset_path.parent.name
    except OSError:
        inferred_shape = args.shape
    metadata = {
        "input_video": str(args.input),
        "output_video": str(final_output),
        "task": args.task,
        "inserted_shape": inferred_shape,
        "asset": str(placement.asset_path),
        "position": {"x": placement.x, "y": placement.y},
        "target_position": {
            "side": args.target_side,
            "center_x": args.target_center_x,
            "center_y": args.target_center_y,
        },
        "placement_constraints": {
            "placement_side": args.placement_side,
            "crop_left": args.crop_left,
            "crop_margin": args.crop_margin,
            "visible_ratio_after_crop": visible_ratio,
        },
        "scale": placement.scale,
        "rotation_deg": placement.rotation_deg,
        "brightness": placement.brightness,
        "blur_sigma": placement.blur_sigma,
        "selection_mode": placement.selection_mode,
        "board_points": roi_points.astype(int).tolist(),
        "parameters": {
            "sample_stride": args.sample_stride,
            "pixel_threshold": args.pixel_threshold,
            "ever_changed_pixel_threshold": args.ever_changed_threshold,
            "changed_ratio_threshold": args.changed_ratio,
            "motion_dilate_px": args.motion_dilate,
            "drawing_gray_threshold": args.drawing_threshold,
            "drawing_dilate_px": args.drawing_dilate,
            "placement_margin": args.placement_margin,
            "board_border_margin": args.board_border_margin,
            "stabilize": args.stabilize,
        },
        "seed": args.seed,
        "video": {**video_info, **write_info},
        "audio_restored": audio_restored,
        "debug_dir": str(debug_dir) if args.debug else None,
        "stats": stats,
    }
    save_metadata(args.output.parent / "placement_metadata.json", metadata)

    print(f"Video frames: {stats['video_frames']}")
    print(f"Sampled frames: {stats['sampled_frames']}")
    print(f"Board area: {stats['board_area']} px")
    print(f"Motion-unsafe area: {stats['motion_unsafe_area']} px")
    print(f"Drawing-unsafe area: {stats['drawing_unsafe_area']} px")
    print(f"Final safe area: {stats['final_safe_area']} px")
    print(f"Valid placements: {stats['valid_placements']}")
    print(f"Inserted shape: {inferred_shape}")
    print(f"Selected asset: {placement.asset_path.name}")
    print(f"Selected position: x={placement.x}, y={placement.y}")
    print(f"Output: {final_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
