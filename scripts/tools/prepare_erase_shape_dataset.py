#!/usr/bin/env python3
"""Create a frame-trimmed ``erase the shape`` LeRobot training dataset.

Workflow:
1. Generate a JSON file containing every source episode with ``--make-template``.
2. Fill in each enabled episode's start/end frame in that JSON file.
3. Run this script normally to decode only those ranges and create a new dataset.

Source datasets are read-only. RGB cameras are retained, ``observation.state``
is reduced to seven joint positions, and the seven-dimensional action is kept.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import av
import cv2
import numpy as np
import pandas as pd

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIRS = (
    REPO_ROOT / "records/0727/erase_the_circle",
    REPO_ROOT / "records/0727/erase_the_triangle",
    REPO_ROOT / "records/0727/erase_the_rectangle",
)
DEFAULT_MANIFEST = REPO_ROOT / "configs/erase_shape_frame_ranges.json"
DEFAULT_OUTPUT = REPO_ROOT / "records/0727/erase_the_shape"
TARGET_TASK = "erase the shape"
POSITION_DIM = 7
RGB_VIDEO_KEYS = ("observation.images.top", "observation.images.wrist")


@dataclass(frozen=True)
class Crop:
    x: int
    y: int
    size: int


def parse_crop(text: str) -> Crop:
    try:
        values = [int(value.strip()) for value in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Crop must use integer x,y,size values") from exc
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Crop must use x,y,size format")
    x, y, size = values
    if x < 0 or y < 0 or size <= 0:
        raise argparse.ArgumentTypeError("x/y must be non-negative and size must be positive")
    return Crop(x=x, y=y, size=size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Frame-range JSON (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output dataset root (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--make-template",
        action="store_true",
        help="Create the frame-range JSON and stop. Existing files are never overwritten.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the completed JSON and source datasets without building.",
    )
    parser.add_argument(
        "--vcodec",
        default=os.environ.get("VCODEC", "hevc"),
        help="RGB output codec (default: VCODEC environment variable or hevc).",
    )
    parser.add_argument(
        "--encoder-threads",
        type=int,
        default=None,
        help="Optional number of encoder threads per camera.",
    )
    parser.add_argument(
        "--gop-size",
        type=int,
        default=250,
        help="RGB keyframe interval (default: 250, matching the source recordings).",
    )
    parser.add_argument(
        "--top-crop",
        type=parse_crop,
        metavar="X,Y,SIZE",
        help="Square crop for observation.images.top before encoding.",
    )
    parser.add_argument(
        "--wrist-crop",
        type=parse_crop,
        metavar="X,Y,SIZE",
        help="Square crop for observation.images.wrist before encoding.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        help="Resize both square crops to this output size, for example 512.",
    )
    return parser.parse_args()


def relative_to_repo(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def resolve_source(source: str) -> Path:
    path = Path(source)
    direct = (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if (direct / "meta/info.json").is_file():
        return direct

    # The selected source groups were moved from records/local to records/0727.
    relocated = (REPO_ROOT / "records/0727" / path.parent.name / path.name).resolve()
    if (relocated / "meta/info.json").is_file():
        return relocated

    candidates = sorted(
        candidate.resolve()
        for candidate in (REPO_ROOT / "records").rglob(path.name)
        if candidate.is_dir()
        and candidate.parent.name == path.parent.name
        and (candidate / "meta/info.json").is_file()
    )
    if not candidates:
        raise FileNotFoundError(f"Source dataset does not exist: {source}")
    if len(candidates) > 1:
        formatted = "\n".join(f"- {candidate}" for candidate in candidates)
        raise RuntimeError(f"Source dataset is ambiguous: {source}\n{formatted}")
    return candidates[0]


def discover_sources() -> list[Path]:
    sources: list[Path] = []
    for group_dir in DEFAULT_SOURCE_DIRS:
        if not group_dir.is_dir():
            raise FileNotFoundError(f"Source group does not exist: {group_dir}")
        group_sources = sorted(path for path in group_dir.iterdir() if (path / "meta/info.json").is_file())
        if not group_sources:
            raise RuntimeError(f"No LeRobot datasets found below: {group_dir}")
        sources.extend(group_sources)
    return sources


def read_info(root: Path) -> dict[str, Any]:
    return json.loads((root / "meta/info.json").read_text())


def source_repo_id(root: Path) -> str:
    return f"local/{root.name}"


def make_template(manifest_path: Path) -> None:
    if manifest_path.exists():
        raise FileExistsError(f"Manifest already exists and will not be overwritten: {manifest_path}")

    episodes = []
    for source in discover_sources():
        info = read_info(source)
        if info["total_episodes"] != 1:
            raise ValueError(f"Each source must contain exactly one episode: {source}")
        episodes.append(
            {
                "source_dataset": relative_to_repo(source),
                "total_frames": info["total_frames"],
                "enabled": True,
                "start_frame": None,
                "end_frame": None,
            }
        )

    manifest = {
        "format_version": 1,
        "target_task": TARGET_TASK,
        "range_semantics": "start_frame is inclusive; end_frame is exclusive",
        "instructions": [
            "For frames 100 through 299, enter start_frame=100 and end_frame=300.",
            "Set enabled=false to exclude an entire episode; its start/end may remain null.",
            "Do not change source_dataset or total_frames.",
        ],
        "episodes": episodes,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Created: {manifest_path}")
    print(f"Episodes awaiting ranges: {len(episodes)}")


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest does not exist: {manifest_path}\n"
            f"Create it first with: {Path(__file__).name} --make-template"
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format_version") != 1:
        raise ValueError(f"Unsupported format_version: {manifest.get('format_version')}")
    if manifest.get("target_task") != TARGET_TASK:
        raise ValueError(f"target_task must be exactly {TARGET_TASK!r}")
    if not isinstance(manifest.get("episodes"), list):
        raise ValueError("Manifest must contain an episodes list.")
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries = manifest["episodes"]
    entry_names = [entry.get("source_dataset") for entry in entries]

    if len(entry_names) != len(set(entry_names)):
        raise ValueError("Manifest contains duplicate source_dataset entries.")

    errors: list[str] = []
    enabled_entries: list[dict[str, Any]] = []
    for entry in entries:
        source_name = entry.get("source_dataset")
        if not isinstance(source_name, str):
            errors.append(f"Invalid source_dataset: {source_name!r}")
            continue
        try:
            source = resolve_source(source_name)
        except (FileNotFoundError, RuntimeError) as exc:
            errors.append(str(exc))
            continue
        actual_frames = read_info(source)["total_frames"]
        if entry.get("total_frames") != actual_frames:
            errors.append(
                f"{source_name}: total_frames changed "
                f"({entry.get('total_frames')} in JSON, {actual_frames} on disk)"
            )
        if not isinstance(entry.get("enabled"), bool):
            errors.append(f"{source_name}: enabled must be true or false")
            continue
        if not entry["enabled"]:
            continue

        start = entry.get("start_frame")
        end = entry.get("end_frame")
        if isinstance(start, bool) or not isinstance(start, int):
            errors.append(f"{source_name}: start_frame must be an integer")
            continue
        if isinstance(end, bool) or not isinstance(end, int):
            errors.append(f"{source_name}: end_frame must be an integer")
            continue
        if not 0 <= start < end <= actual_frames:
            errors.append(
                f"{source_name}: require 0 <= start_frame < end_frame <= {actual_frames}, "
                f"got [{start}, {end})"
            )
            continue

        resolved = dict(entry)
        resolved["_source_path"] = source
        resolved["_selected_frames"] = end - start
        enabled_entries.append(resolved)

    if errors:
        formatted = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"Frame-range JSON is incomplete or invalid:\n{formatted}")
    if not enabled_entries:
        raise ValueError("No episodes are enabled.")
    return enabled_entries


def position_names(features: dict[str, Any]) -> list[str]:
    state = features.get("observation.state")
    action = features.get("action")
    if state is None or action is None:
        raise ValueError("Both observation.state and action are required.")
    state_names = list(state.get("names") or [])
    action_names = list(action.get("names") or [])
    positions = state_names[:POSITION_DIM]
    if len(state_names) < POSITION_DIM or len(action_names) != POSITION_DIM:
        raise ValueError(f"Unexpected state/action names: state={state_names}, action={action_names}")
    if positions != action_names or not all(name.endswith(".pos") for name in positions):
        raise ValueError(f"First seven state values are not the action-position fields: {positions}")
    return positions


def output_features(
    source_features: dict[str, Any],
    crops: dict[str, Crop] | None = None,
    image_size: int | None = None,
) -> dict[str, Any]:
    retained: dict[str, Any] = {}
    for key in ("observation.state", "action", *RGB_VIDEO_KEYS):
        if key not in source_features:
            raise ValueError(f"Required feature is missing: {key}")
        retained[key] = copy.deepcopy(source_features[key])
        retained[key]["shape"] = tuple(retained[key]["shape"])
    positions = position_names(source_features)
    retained["observation.state"]["shape"] = (POSITION_DIM,)
    retained["observation.state"]["names"] = positions
    if crops is not None:
        assert image_size is not None
        for key in RGB_VIDEO_KEYS:
            retained[key]["shape"] = (image_size, image_size, 3)
            video_info = retained[key].get("info")
            if isinstance(video_info, dict):
                video_info["video.height"] = image_size
                video_info["video.width"] = image_size
    return retained


def validate_source_schemas(
    entries: list[dict[str, Any]],
    crops: dict[str, Crop] | None = None,
    image_size: int | None = None,
) -> tuple[dict[str, Any], int, str]:
    reference: dict[str, Any] | None = None
    fps: int | None = None
    robot_type: str | None = None
    for entry in entries:
        info = read_info(entry["_source_path"])
        if info["total_episodes"] != 1:
            raise ValueError(f"Each source must contain one episode: {entry['_source_path']}")
        candidate = output_features(info["features"], crops, image_size)
        if reference is None:
            reference = candidate
            fps = info["fps"]
            robot_type = info["robot_type"]
        elif candidate != reference or info["fps"] != fps or info["robot_type"] != robot_type:
            raise ValueError(f"Source schema differs: {entry['_source_path']}")
    assert reference is not None and fps is not None and robot_type is not None
    return reference, fps, robot_type


def load_source_data(root: Path, total_frames: int) -> pd.DataFrame:
    paths = sorted((root / "data").rglob("*.parquet"))
    if not paths:
        raise RuntimeError(f"No data parquet found: {root}")
    data = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    data = data.sort_values("frame_index").reset_index(drop=True)
    if len(data) != total_frames:
        raise ValueError(f"{root}: parquet rows={len(data)}, info frames={total_frames}")
    if data["frame_index"].tolist() != list(range(total_frames)):
        raise ValueError(f"{root}: frame_index is not contiguous from zero.")
    if set(data["episode_index"].tolist()) != {0}:
        raise ValueError(f"{root}: expected only episode_index 0.")
    return data


def add_selected_episode(
    output: LeRobotDataset,
    entry: dict[str, Any],
    crops: dict[str, Crop] | None = None,
    image_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    source = entry["_source_path"]
    start = entry["start_frame"]
    end = entry["end_frame"]
    data = load_source_data(source, entry["total_frames"])
    selected = data.iloc[start:end]

    meta = LeRobotDatasetMetadata(source_repo_id(source), root=source)
    video_paths = {
        key: source / meta.get_video_file_path(0, key)
        for key in RGB_VIDEO_KEYS
    }

    with ExitStack() as stack:
        containers = {
            key: stack.enter_context(av.open(str(path)))
            for key, path in video_paths.items()
        }
        decoders = {
            key: container.decode(video=0)
            for key, container in containers.items()
        }

        for source_frame_index in range(end):
            decoded: dict[str, np.ndarray] = {}
            for key, decoder in decoders.items():
                try:
                    decoded[key] = next(decoder).to_ndarray(format="rgb24")
                except StopIteration as exc:
                    raise RuntimeError(
                        f"{video_paths[key]} ended before frame {source_frame_index}"
                    ) from exc
            if source_frame_index < start:
                continue

            if crops is not None:
                assert image_size is not None
                for key, crop in crops.items():
                    image = decoded[key]
                    height, width = image.shape[:2]
                    if crop.x + crop.size > width or crop.y + crop.size > height:
                        raise ValueError(
                            f"{source} {key}: crop ({crop.x},{crop.y},{crop.size}) "
                            f"exceeds frame size {width}x{height}"
                        )
                    cropped = image[
                        crop.y : crop.y + crop.size,
                        crop.x : crop.x + crop.size,
                    ]
                    decoded[key] = cv2.resize(
                        cropped,
                        (image_size, image_size),
                        interpolation=(
                            cv2.INTER_AREA
                            if crop.size >= image_size
                            else cv2.INTER_LINEAR
                        ),
                    )

            row = data.iloc[source_frame_index]
            frame: dict[str, Any] = {
                "observation.state": np.asarray(row["observation.state"], dtype=np.float32)[:POSITION_DIM],
                "action": np.asarray(row["action"], dtype=np.float32),
                "task": TARGET_TASK,
            }
            frame.update(decoded)
            output.add_frame(frame)

    output.save_episode()
    return (
        np.stack(selected["observation.state"].map(lambda value: np.asarray(value)[:POSITION_DIM])),
        np.stack(selected["action"].map(np.asarray)),
    )


def load_all_output_data(root: Path) -> pd.DataFrame:
    paths = sorted((root / "data").rglob("*.parquet"))
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)


def validate_output(
    root: Path,
    output_repo_id: str,
    entries: list[dict[str, Any]],
    expected_states: np.ndarray,
    expected_actions: np.ndarray,
) -> None:
    expected_episodes = len(entries)
    expected_frames = sum(entry["_selected_frames"] for entry in entries)
    meta = LeRobotDatasetMetadata(output_repo_id, root=root)
    if meta.total_episodes != expected_episodes or meta.total_frames != expected_frames:
        raise AssertionError(
            f"Count mismatch: episodes={meta.total_episodes}/{expected_episodes}, "
            f"frames={meta.total_frames}/{expected_frames}"
        )
    if list(meta.tasks.index) != [TARGET_TASK]:
        raise AssertionError(f"Unexpected tasks: {list(meta.tasks.index)}")
    if tuple(meta.features["observation.state"]["shape"]) != (POSITION_DIM,):
        raise AssertionError(f"Unexpected state shape: {meta.features['observation.state']['shape']}")
    if set(meta.video_keys) != set(RGB_VIDEO_KEYS):
        raise AssertionError(f"Unexpected video keys: {meta.video_keys}")

    data = load_all_output_data(root)
    states = np.stack(data["observation.state"].map(np.asarray))
    actions = np.stack(data["action"].map(np.asarray))
    if not np.array_equal(states, expected_states):
        raise AssertionError("Output state values differ from the selected source frames.")
    if not np.array_equal(actions, expected_actions):
        raise AssertionError("Output action values differ from the selected source frames.")
    if data["index"].tolist() != list(range(expected_frames)):
        raise AssertionError("Global frame indices are not contiguous.")
    if set(data["task_index"].tolist()) != {0}:
        raise AssertionError("Task indices are not all zero.")

    expected_lengths = [entry["_selected_frames"] for entry in entries]
    actual_lengths = data.groupby("episode_index", sort=True).size().tolist()
    if actual_lengths != expected_lengths:
        raise AssertionError(f"Episode lengths differ: {actual_lengths} != {expected_lengths}")
    for _, episode in data.groupby("episode_index", sort=True):
        if episode["frame_index"].tolist() != list(range(len(episode))):
            raise AssertionError("An episode has non-contiguous frame indices.")

    dataset = LeRobotDataset(output_repo_id, root=root, video_backend="pyav")
    for index in sorted({0, expected_frames // 2, expected_frames - 1}):
        sample = dataset[index]
        if tuple(sample["observation.state"].shape) != (POSITION_DIM,):
            raise AssertionError(f"Bad state shape from loader at frame {index}")
        if tuple(sample["action"].shape) != (POSITION_DIM,):
            raise AssertionError(f"Bad action shape from loader at frame {index}")
        for key in RGB_VIDEO_KEYS:
            height, width, channels = meta.features[key]["shape"]
            expected_shape = (channels, height, width)
            if tuple(sample[key].shape) != expected_shape:
                raise AssertionError(f"Bad {key} shape at frame {index}: {sample[key].shape}")


def build_dataset(
    entries: list[dict[str, Any]],
    output_path: Path,
    vcodec: str,
    encoder_threads: int | None,
    gop_size: int,
    crops: dict[str, Crop] | None = None,
    image_size: int | None = None,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"Output already exists and will not be overwritten: {output_path}")

    output_repo_id = f"local/{output_path.name}"
    features, fps, robot_type = validate_source_schemas(entries, crops, image_size)
    expected_frames = sum(entry["_selected_frames"] for entry in entries)
    print(f"Selected episodes: {len(entries)}")
    print(f"Selected frames: {expected_frames}")
    print(f"Output codec: {vcodec}")
    print(f"GOP size: {gop_size}")
    if crops is not None:
        print(
            f"TOP crop: {crops['observation.images.top'].x},"
            f"{crops['observation.images.top'].y},"
            f"{crops['observation.images.top'].size}"
        )
        print(
            f"WRIST crop: {crops['observation.images.wrist'].x},"
            f"{crops['observation.images.wrist'].y},"
            f"{crops['observation.images.wrist'].size}"
        )
        print(f"Output image size: {image_size}x{image_size}")
    print(f"Output: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = output_path.parent / f".{output_path.name}.building-{uuid4().hex[:8]}"
    writer: LeRobotDataset | None = None
    try:
        writer = LeRobotDataset.create(
            repo_id=output_repo_id,
            fps=fps,
            features=features,
            root=staging,
            robot_type=robot_type,
            use_videos=True,
            video_backend="pyav",
            vcodec=vcodec,
            streaming_encoding=True,
            encoder_threads=encoder_threads,
        )
        # LeRobot 0.4.4 does not expose the streaming GOP through create().
        # Match the existing recordings rather than its GOP=2 default, which
        # torchvision's deprecated HEVC/PyAV seeker can decode one frame late.
        assert writer._streaming_encoder is not None
        writer._streaming_encoder.g = gop_size
        expected_states = []
        expected_actions = []
        for output_episode, entry in enumerate(entries):
            print(
                f"[{output_episode + 1}/{len(entries)}] {entry['source_dataset']} "
                f"[{entry['start_frame']}, {entry['end_frame']})"
            )
            states, actions = add_selected_episode(
                writer,
                entry,
                crops,
                image_size,
            )
            expected_states.append(states)
            expected_actions.append(actions)
        writer.finalize()

        validate_output(
            staging,
            output_repo_id,
            entries,
            np.concatenate(expected_states),
            np.concatenate(expected_actions),
        )
        os.replace(staging, output_path)
    except Exception:
        if writer is not None:
            try:
                writer.finalize()
            except Exception:
                pass
        print(f"Build failed. Partial staging data was kept for inspection: {staging}", file=sys.stderr)
        raise

    print(f"PASS: created {output_path}")
    print(f"PASS: {len(entries)} episodes, {expected_frames} frames, task={TARGET_TASK!r}")
    print("PASS: selected position/action values are pixel-index aligned with both RGB videos.")


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    crop_values = (args.top_crop, args.wrist_crop)
    if any(crop is not None for crop in crop_values) != all(
        crop is not None for crop in crop_values
    ):
        raise ValueError("--top-crop and --wrist-crop must be provided together.")
    if args.image_size is not None and args.image_size <= 0:
        raise ValueError("--image-size must be positive.")
    if all(crop is not None for crop in crop_values) and args.image_size is None:
        raise ValueError("--image-size is required when crop options are used.")
    if args.image_size is not None and not all(
        crop is not None for crop in crop_values
    ):
        raise ValueError("--top-crop and --wrist-crop are required with --image-size.")
    crops = (
        {
            "observation.images.top": args.top_crop,
            "observation.images.wrist": args.wrist_crop,
        }
        if args.top_crop is not None
        else None
    )

    if args.make_template:
        if args.validate_only:
            raise ValueError("--make-template and --validate-only cannot be used together.")
        make_template(manifest_path)
        return 0

    manifest = load_manifest(manifest_path)
    entries = validate_manifest(manifest)
    features, fps, robot_type = validate_source_schemas(
        entries,
        crops,
        args.image_size,
    )
    del features, robot_type
    total_frames = sum(entry["_selected_frames"] for entry in entries)
    print(f"PASS: frame ranges are valid ({len(entries)} episodes, {total_frames} frames, {fps} fps).")
    if args.validate_only:
        return 0

    if args.gop_size <= 0:
        raise ValueError("--gop-size must be positive.")
    build_dataset(
        entries,
        output_path,
        args.vcodec,
        args.encoder_threads,
        args.gop_size,
        crops,
        args.image_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
