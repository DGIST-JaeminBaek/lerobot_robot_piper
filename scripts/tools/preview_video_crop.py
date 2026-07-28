#!/usr/bin/env python3
"""Create SSH-friendly crop previews for LeRobotDataset camera videos.

This tool never modifies the dataset. It writes PNG contact sheets showing:

1. the original frame with the proposed square crop,
2. the cropped region, and
3. the final 512x512 model input.

Example:
    python scripts/tools/preview_video_crop.py \
        records/0727/erase_the_shape \
        --top-crop 280,0,720 \
        --wrist-crop 200,0,720

To inspect a selected point from every enabled range in a frame-range manifest:
    python scripts/tools/preview_video_crop.py \
        records/0727/erase_the_shape \
        --frame-ranges configs/erase_shape_frame_ranges.json \
        --camera wrist \
        --range-point start
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


CAMERA_KEYS = {
    "top": "observation.images.top",
    "wrist": "observation.images.wrist",
}
MODEL_SIZE = 512
ORIGINAL_PANEL_SIZE = (640, 360)
CROP_PANEL_SIZE = (360, 360)
HEADER_HEIGHT = 34
MIDPOINT_TILE_SIZE = 256
MIDPOINT_TILE_HEADER = 54
MIDPOINT_GRID_COLUMNS = 5
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Crop:
    x: int
    y: int
    size: int


@dataclass(frozen=True)
class VideoPart:
    path: Path
    start: int
    stop: int


class VideoSequence:
    """Treat sorted chunked MP4 files as one global frame sequence."""

    def __init__(self, paths: list[Path]) -> None:
        if not paths:
            raise ValueError("No video files were provided")

        parts: list[VideoPart] = []
        start = 0
        width = height = None

        for path in paths:
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise RuntimeError(f"Failed to open video: {path}")

            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            part_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            part_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            capture.release()

            if count <= 0 or part_width <= 0 or part_height <= 0:
                raise RuntimeError(f"Invalid video metadata: {path}")
            if width is None:
                width, height = part_width, part_height
            elif (part_width, part_height) != (width, height):
                raise RuntimeError(
                    "All chunks for one camera must have the same resolution: "
                    f"expected {width}x{height}, got {part_width}x{part_height} in {path}"
                )

            parts.append(VideoPart(path=path, start=start, stop=start + count))
            start += count

        self.parts = parts
        self.frame_count = start
        self.width = int(width)
        self.height = int(height)

    def read(self, global_frame_index: int) -> np.ndarray:
        if not 0 <= global_frame_index < self.frame_count:
            raise IndexError(
                f"Frame {global_frame_index} is outside 0..{self.frame_count - 1}"
            )

        part = next(
            part
            for part in self.parts
            if part.start <= global_frame_index < part.stop
        )
        local_index = global_frame_index - part.start
        capture = cv2.VideoCapture(str(part.path))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video: {part.path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, local_index)
        ok, frame = capture.read()
        capture.release()
        if not ok or frame is None:
            raise RuntimeError(
                f"Failed to decode global frame {global_frame_index} "
                f"(local frame {local_index}) from {part.path}"
            )
        return frame


def parse_crop(text: str) -> Crop:
    try:
        values = [int(value.strip()) for value in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Crop must contain integers in x,y,size format"
        ) from exc
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Crop must use x,y,size format")
    x, y, size = values
    if x < 0 or y < 0 or size <= 0:
        raise argparse.ArgumentTypeError("x/y must be non-negative and size must be positive")
    return Crop(x=x, y=y, size=size)


def parse_frames(text: str) -> list[int]:
    try:
        frames = [int(value.strip()) for value in text.split(",") if value.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Frames must be comma-separated non-negative integers"
        ) from exc
    if not frames or any(frame < 0 for frame in frames):
        raise argparse.ArgumentTypeError(
            "Frames must be comma-separated non-negative integers"
        )
    return frames


def discover_videos(dataset_root: Path, camera_key: str) -> list[Path]:
    camera_dir = dataset_root / "videos" / camera_key
    paths = sorted(camera_dir.glob("chunk-*/*.mp4"))
    if not paths:
        paths = sorted(camera_dir.rglob("*.mp4"))
    if not paths:
        raise FileNotFoundError(f"No MP4 files found under {camera_dir}")
    return paths


def resolve_manifest_source(dataset_root: Path, source_text: str) -> Path:
    """Resolve a manifest source, including datasets moved as one dated group."""
    source_path = Path(source_text)
    direct = (
        source_path.expanduser().resolve()
        if source_path.is_absolute()
        else (REPO_ROOT / source_path).resolve()
    )
    if (direct / "meta/info.json").is_file():
        return direct

    # The source groups may have been moved together, for example:
    # records/local/erase_the_circle/X -> records/0727/erase_the_circle/X.
    relocated = (
        dataset_root.parent / source_path.parent.name / source_path.name
    ).resolve()
    if (relocated / "meta/info.json").is_file():
        return relocated

    candidates = sorted(
        candidate.resolve()
        for candidate in (REPO_ROOT / "records").rglob(source_path.name)
        if candidate.is_dir()
        and candidate.parent.name == source_path.parent.name
        and (candidate / "meta/info.json").is_file()
    )
    if not candidates:
        raise FileNotFoundError(
            f"Manifest source does not exist and no relocated copy was found: {source_text}"
        )
    if len(candidates) > 1:
        joined = "\n".join(f"- {candidate}" for candidate in candidates)
        raise RuntimeError(
            f"Manifest source is ambiguous: {source_text}\nCandidates:\n{joined}"
        )
    return candidates[0]


def load_manifest_ranges(
    manifest_path: Path,
    dataset_root: Path,
) -> list[dict[str, object]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Frame-range manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("Frame-range manifest must contain an episodes list")

    selected: list[dict[str, object]] = []
    errors: list[str] = []
    for episode_index, entry in enumerate(episodes):
        if not isinstance(entry, dict):
            errors.append(f"episode {episode_index}: entry must be an object")
            continue
        if entry.get("enabled") is False:
            continue

        source_text = entry.get("source_dataset")
        start = entry.get("start_frame")
        end = entry.get("end_frame")
        total_frames = entry.get("total_frames")
        if not isinstance(source_text, str):
            errors.append(f"episode {episode_index}: source_dataset must be a string")
            continue
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start < 0
            or start >= end
        ):
            errors.append(
                f"episode {episode_index} ({source_text}): invalid range [{start}, {end})"
            )
            continue
        if isinstance(total_frames, int) and end > total_frames:
            errors.append(
                f"episode {episode_index} ({source_text}): "
                f"end_frame {end} exceeds total_frames {total_frames}"
            )
            continue

        try:
            source_root = resolve_manifest_source(dataset_root, source_text)
        except (FileNotFoundError, RuntimeError) as exc:
            errors.append(f"episode {episode_index}: {exc}")
            continue

        selected.append(
            {
                "episode_index": episode_index,
                "source_dataset": source_text,
                "source_root": source_root,
                "start_frame": start,
                "end_frame": end,
            }
        )

    if errors:
        raise ValueError(
            "Frame-range manifest contains invalid enabled entries:\n"
            + "\n".join(f"- {error}" for error in errors)
        )
    if not selected:
        raise ValueError("Frame-range manifest has no enabled episodes with valid ranges")
    return selected


def centered_crop(width: int, height: int) -> Crop:
    size = min(width, height)
    return Crop(x=(width - size) // 2, y=(height - size) // 2, size=size)


def validate_crop(crop: Crop, width: int, height: int, camera: str) -> None:
    if crop.x + crop.size > width or crop.y + crop.size > height:
        raise ValueError(
            f"{camera} crop ({crop.x},{crop.y},{crop.size}) exceeds "
            f"the {width}x{height} frame"
        )


def fit_with_padding(
    image: np.ndarray,
    target_width: int,
    target_height: int,
    pad_value: int = 24,
) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    canvas = np.full((target_height, target_width, 3), pad_value, dtype=np.uint8)
    x = (target_width - resized_width) // 2
    y = (target_height - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return canvas


def titled_panel(image: np.ndarray, title: str) -> np.ndarray:
    header = np.full((HEADER_HEIGHT, image.shape[1], 3), 24, dtype=np.uint8)
    cv2.putText(
        header,
        title,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return np.vstack((header, image))


def make_row(frame: np.ndarray, frame_index: int, crop: Crop) -> np.ndarray:
    marked = frame.copy()
    cv2.rectangle(
        marked,
        (crop.x, crop.y),
        (crop.x + crop.size - 1, crop.y + crop.size - 1),
        (0, 255, 255),
        5,
        cv2.LINE_AA,
    )
    original_panel = fit_with_padding(marked, *ORIGINAL_PANEL_SIZE)

    cropped = frame[
        crop.y : crop.y + crop.size,
        crop.x : crop.x + crop.size,
    ]
    crop_panel = cv2.resize(cropped, CROP_PANEL_SIZE, interpolation=cv2.INTER_AREA)
    model_input = cv2.resize(
        cropped,
        (MODEL_SIZE, MODEL_SIZE),
        interpolation=cv2.INTER_AREA if crop.size >= MODEL_SIZE else cv2.INTER_LINEAR,
    )
    model_panel = cv2.resize(model_input, CROP_PANEL_SIZE, interpolation=cv2.INTER_AREA)

    panels = [
        titled_panel(
            original_panel,
            f"FRAME {frame_index} | ORIGINAL | crop={crop.x},{crop.y},{crop.size}",
        ),
        titled_panel(crop_panel, f"CROP {crop.size}x{crop.size}"),
        titled_panel(model_panel, f"MODEL INPUT {MODEL_SIZE}x{MODEL_SIZE}"),
    ]
    return np.hstack(panels)


def make_midpoint_tile(
    frame: np.ndarray,
    crop: Crop,
    episode_index: int,
    source_name: str,
    selected_frame: int,
) -> np.ndarray:
    cropped = frame[
        crop.y : crop.y + crop.size,
        crop.x : crop.x + crop.size,
    ]
    model_input = cv2.resize(
        cropped,
        (MODEL_SIZE, MODEL_SIZE),
        interpolation=cv2.INTER_AREA if crop.size >= MODEL_SIZE else cv2.INTER_LINEAR,
    )
    tile_image = cv2.resize(
        model_input,
        (MIDPOINT_TILE_SIZE, MIDPOINT_TILE_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    header = np.full(
        (MIDPOINT_TILE_HEADER, MIDPOINT_TILE_SIZE, 3),
        24,
        dtype=np.uint8,
    )
    cv2.putText(
        header,
        f"EP {episode_index:02d} | frame {selected_frame}",
        (7, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    display_name = source_name
    if len(display_name) > 31:
        display_name = display_name[-31:]
    cv2.putText(
        header,
        display_name,
        (7, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.37,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )
    return np.vstack((header, tile_image))


def make_tile_grid(
    tiles: list[np.ndarray],
    columns: int = MIDPOINT_GRID_COLUMNS,
) -> np.ndarray:
    if not tiles:
        raise ValueError("Cannot create an empty tile grid")
    tile_height, tile_width = tiles[0].shape[:2]
    rows = (len(tiles) + columns - 1) // columns
    blank = np.full((tile_height, tile_width, 3), 12, dtype=np.uint8)
    padded = tiles + [blank] * (rows * columns - len(tiles))
    return np.vstack(
        [
            np.hstack(padded[row * columns : (row + 1) * columns])
            for row in range(rows)
        ]
    )


def evenly_spaced_frames(frame_count: int, num_samples: int) -> list[int]:
    if num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    count = min(num_samples, frame_count)
    return np.linspace(0, frame_count - 1, count, dtype=np.int64).tolist()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate PNG contact sheets to inspect square camera crops over SSH. "
            "The source dataset is never modified."
        )
    )
    parser.add_argument("dataset_root", type=Path, help="LeRobotDataset root")
    parser.add_argument(
        "--camera",
        choices=("top", "wrist", "both"),
        default="both",
        help="Camera preview to generate (default: both)",
    )
    parser.add_argument(
        "--top-crop",
        type=parse_crop,
        metavar="X,Y,SIZE",
        help="TOP square crop; default is the largest centered square",
    )
    parser.add_argument(
        "--wrist-crop",
        type=parse_crop,
        metavar="X,Y,SIZE",
        help="WRIST square crop; default is the largest centered square",
    )
    parser.add_argument(
        "--frames",
        type=parse_frames,
        help="Global frame indices, for example 0,5000,10000",
    )
    parser.add_argument(
        "--frame-ranges",
        type=Path,
        metavar="JSON",
        help=(
            "Instead of sampling the combined video, show one selected frame from "
            "every enabled source range in this JSON manifest"
        ),
    )
    parser.add_argument(
        "--range-point",
        choices=("start", "midpoint", "end"),
        default="midpoint",
        help=(
            "Frame to show from every --frame-ranges entry: start_frame, midpoint, "
            "or the last included frame (default: midpoint)"
        ),
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of evenly spaced frames when --frames is omitted (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/crop_preview"),
        help="PNG/JSON output directory (default: tmp/crop_preview)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_root}")

    cameras = ("top", "wrist") if args.camera == "both" else (args.camera,)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.frame_ranges is not None and args.frames is not None:
        raise ValueError("--frame-ranges and --frames cannot be used together")

    manifest_entries = None
    manifest_path = None
    if args.frame_ranges is not None:
        manifest_path = args.frame_ranges.expanduser().resolve()
        manifest_entries = load_manifest_ranges(manifest_path, dataset_root)

    config: dict[str, object] = {
        "dataset_root": str(dataset_root),
        "model_input_size": [MODEL_SIZE, MODEL_SIZE],
        "mode": "manifest_ranges" if manifest_entries is not None else "combined_video",
        "cameras": {},
    }
    if manifest_path is not None:
        config["frame_ranges"] = str(manifest_path)
        config["range_point"] = args.range_point

    for camera in cameras:
        camera_key = CAMERA_KEYS[camera]
        first_root = (
            manifest_entries[0]["source_root"]
            if manifest_entries is not None
            else dataset_root
        )
        sequence = VideoSequence(discover_videos(first_root, camera_key))
        requested_crop = args.top_crop if camera == "top" else args.wrist_crop
        crop = requested_crop or centered_crop(sequence.width, sequence.height)
        validate_crop(crop, sequence.width, sequence.height, camera)

        if manifest_entries is not None:
            tiles: list[np.ndarray] = []
            preview_entries: list[dict[str, object]] = []
            for entry in manifest_entries:
                source_root = entry["source_root"]
                source_sequence = VideoSequence(
                    discover_videos(source_root, camera_key)
                )
                validate_crop(
                    crop,
                    source_sequence.width,
                    source_sequence.height,
                    camera,
                )
                start = int(entry["start_frame"])
                end = int(entry["end_frame"])
                if args.range_point == "start":
                    selected_frame = start
                elif args.range_point == "end":
                    selected_frame = end - 1
                else:
                    # end_frame is exclusive. For an even-length range this
                    # chooses the later of the two middle frames.
                    selected_frame = (start + end) // 2
                if selected_frame >= source_sequence.frame_count:
                    raise ValueError(
                        f"{source_root}: selected frame {selected_frame} exceeds "
                        f"decoded video length {source_sequence.frame_count}"
                    )
                tiles.append(
                    make_midpoint_tile(
                        source_sequence.read(selected_frame),
                        crop,
                        int(entry["episode_index"]),
                        source_root.name,
                        selected_frame,
                    )
                )
                preview_entries.append(
                    {
                        "episode_index": entry["episode_index"],
                        "source_dataset": entry["source_dataset"],
                        "resolved_source": str(source_root),
                        "selected_range": [
                            entry["start_frame"],
                            entry["end_frame"],
                        ],
                        "selected_frame": selected_frame,
                    }
                )

            contact_sheet = make_tile_grid(tiles)
            range_suffix = {
                "start": "start_frames",
                "midpoint": "midpoints",
                "end": "end_frames",
            }[args.range_point]
            output_path = output_dir / f"{camera}_crop_{range_suffix}.png"
            if not cv2.imwrite(str(output_path), contact_sheet):
                raise RuntimeError(f"Failed to write preview: {output_path}")
            config["cameras"][camera] = {
                "feature_key": camera_key,
                "source_size": [sequence.width, sequence.height],
                "episode_count": len(preview_entries),
                "range_point": args.range_point,
                "crop": {"x": crop.x, "y": crop.y, "size": crop.size},
                "episodes": preview_entries,
                "preview": str(output_path),
            }
            print(f"[OK] {camera}: {output_path}")
            print(
                f"     source={sequence.width}x{sequence.height}, "
                f"crop={crop.x},{crop.y},{crop.size}, "
                f"episode {args.range_point} frames={len(preview_entries)}"
            )
            continue

        frame_indices = (
            args.frames
            if args.frames is not None
            else evenly_spaced_frames(sequence.frame_count, args.num_samples)
        )
        invalid = [index for index in frame_indices if index >= sequence.frame_count]
        if invalid:
            raise ValueError(
                f"{camera} has {sequence.frame_count} frames, but these were requested: {invalid}"
            )

        rows = [
            make_row(sequence.read(frame_index), frame_index, crop)
            for frame_index in frame_indices
        ]
        separator = np.full((8, rows[0].shape[1], 3), 8, dtype=np.uint8)
        contact_sheet_parts: list[np.ndarray] = []
        for row_index, row in enumerate(rows):
            if row_index:
                contact_sheet_parts.append(separator)
            contact_sheet_parts.append(row)
        contact_sheet = np.vstack(contact_sheet_parts)

        output_path = output_dir / f"{camera}_crop_preview.png"
        if not cv2.imwrite(str(output_path), contact_sheet):
            raise RuntimeError(f"Failed to write preview: {output_path}")

        config["cameras"][camera] = {
            "feature_key": camera_key,
            "source_size": [sequence.width, sequence.height],
            "total_frames": sequence.frame_count,
            "sample_frames": frame_indices,
            "crop": {"x": crop.x, "y": crop.y, "size": crop.size},
            "preview": str(output_path),
        }
        print(f"[OK] {camera}: {output_path}")
        print(
            f"     source={sequence.width}x{sequence.height}, "
            f"crop={crop.x},{crop.y},{crop.size}, frames={frame_indices}"
        )

    config_path = output_dir / "crop_config.json"
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] config: {config_path}")
    print("The source dataset was not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
