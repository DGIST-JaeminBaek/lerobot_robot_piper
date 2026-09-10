from __future__ import annotations

"""0802~0805 분석 결과를 바탕으로 실제 증강 작업을 일괄 실행하는 스크립트.

흐름:
1. analyze_target_positions.py가 만든 target_positions.csv를 읽는다.
2. 목표 도형과 다른 종류의 방해 도형 asset을 episode마다 배정한다.
3. 목표 도형이 top이면 방해 도형은 bottom, 목표 도형이 bottom이면 방해 도형은 top에 배치한다.
4. 각 episode에 대해 augment_whiteboard.py를 subprocess로 호출한다.

주의:
- 예전에 쓰던 left/right 기준 결과는 90도 회전된 보드 방향을 반영하지 못해 잘못된 기준이다.
- 현재 기본 출력 폴더는 outputs/0802_0805_board_y_aware_shape_aug 이다.
"""

import argparse
import csv
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


TARGETS = {
    "circle": {
        "task": "erase the circle",
        "distractors": ("triangle", "rectangle"),
    },
    "rectangle": {
        "task": "erase the rectangle",
        "distractors": ("triangle", "circle"),
    },
    "triangle": {
        "task": "erase the triangle",
        "distractors": ("circle", "rectangle"),
    },
}


def natural_asset_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 10_000, path.name


def opposite_side(side: str) -> str:
    # 현재 0802~0805 최종 기준은 top/bottom이다. left/right는 과거 호환용으로만 남겨둔다.
    if side == "left":
        return "right"
    if side == "right":
        return "left"
    if side == "top":
        return "bottom"
    if side == "bottom":
        return "top"
    raise ValueError(f"Unsupported side: {side}")


def list_shape_assets(assets_root: Path, shape: str) -> list[Path]:
    files = sorted((assets_root / shape).glob("*.png"), key=natural_asset_key)
    if not files:
        raise FileNotFoundError(f"No PNG assets found for {shape}: {assets_root / shape}")
    return files


def read_positions_csv(path: Path, dates: set[str]) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Target position CSV not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["date"] not in dates:
                continue
            target = row["target"]
            if target not in TARGETS:
                continue
            video = Path(row["video"])
            if not video.exists():
                raise FileNotFoundError(f"Video listed in analysis CSV does not exist: {video}")
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No analyzed rows found in {path} for dates {sorted(dates)}")
    return rows


def choose_assets_with_reuse(assets: list[Path], count: int, rng: random.Random) -> list[tuple[Path, int]]:
    # 한 라운드 안에서는 비복원 추출, asset을 다 쓰면 다시 섞어서 재사용한다.
    selected: list[tuple[Path, int]] = []
    reuse_round = 0
    while len(selected) < count:
        pool = assets[:]
        rng.shuffle(pool)
        for asset in pool:
            selected.append((asset, reuse_round))
            if len(selected) == count:
                break
        reuse_round += 1
    return selected


def make_plan(rows: list[dict[str, Any]], assets_root: Path, seed: int) -> list[dict[str, Any]]:
    # plan은 재현성을 위해 seed 기반으로 만든다. 실제 실행 전 CSV/JSON으로 저장된다.
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["target"]].append(row)

    plan: list[dict[str, Any]] = []
    for target in sorted(grouped):
        target_rows = grouped[target][:]
        rng.shuffle(target_rows)
        distractors = TARGETS[target]["distractors"]
        first_count = len(target_rows) // 2
        assignments = [
            (distractors[0], target_rows[:first_count]),
            (distractors[1], target_rows[first_count:]),
        ]
        for shape, shape_rows in assignments:
            asset_choices = choose_assets_with_reuse(list_shape_assets(assets_root, shape), len(shape_rows), rng)
            for row, (asset, reuse_round) in zip(shape_rows, asset_choices):
                plan.append(
                    {
                        "date": row["date"],
                        "target": target,
                        "episode": row["episode"],
                        "target_side": row["side"],
                        "placement_side": opposite_side(row["side"]),
                        "target_center_x": row["target_center_x"],
                        "target_center_y": row["target_center_y"],
                        "shape": shape,
                        "asset": str(asset),
                        "asset_reuse_round": reuse_round,
                        "input_video": row["video"],
                        "seed": rng.randint(1, 2_147_483_647),
                    }
                )
    return sorted(plan, key=lambda item: (item["date"], item["target"], item["episode"]))


def save_plan(plan: list[dict[str, Any]], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "augmentation_plan.json"
    csv_path = output_root / "augmentation_plan.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "date",
            "target",
            "episode",
            "target_side",
            "placement_side",
            "target_center_x",
            "target_center_y",
            "shape",
            "asset",
            "asset_reuse_round",
            "input_video",
            "seed",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(plan)
    print(f"Plan JSON: {json_path}")
    print(f"Plan CSV: {csv_path}")


def output_video_for_row(row: dict[str, Any], output_root: Path) -> Path:
    return output_root / row["date"] / f"erase_{row['target']}" / row["shape"] / row["episode"] / "augmented.mp4"


def metadata_for_row(row: dict[str, Any], output_root: Path) -> Path:
    return output_video_for_row(row, output_root).parent / "placement_metadata.json"


def output_is_complete(row: dict[str, Any], output_root: Path) -> bool:
    output_video = output_video_for_row(row, output_root)
    metadata = metadata_for_row(row, output_root)
    return output_video.exists() and output_video.stat().st_size > 0 and metadata.exists() and metadata.stat().st_size > 0


def command_for_row(args: argparse.Namespace, row: dict[str, Any], output_root: Path, overrides: dict[str, Any] | None = None) -> list[str]:
    values = {
        "sample_stride": args.sample_stride,
        "pixel_threshold": args.pixel_threshold,
        "ever_changed_threshold": args.ever_changed_threshold,
        "changed_ratio": args.changed_ratio,
        "motion_dilate": args.motion_dilate,
        "drawing_threshold": args.drawing_threshold,
        "drawing_dilate": args.drawing_dilate,
        "placement_margin": args.placement_margin,
        "board_border_margin": args.board_border_margin,
        "scale_range": args.scale_range,
        "rotation_range": args.rotation_range,
    }
    if overrides:
        values.update(overrides)

    return [
        sys.executable,
        str(Path(__file__).resolve().parent / "augment_whiteboard.py"),
        "--input",
        row["input_video"],
        "--assets",
        str(args.assets_root),
        *([] if args.asset_index else ["--asset", row["asset"]]),
        *([] if args.asset_index is None else ["--asset-index", str(args.asset_index)]),
        *([] if args.asset_index is None else ["--excluded-assets-file", str(output_root / "used_assets.json")]),
        "--shape",
        row["shape"],
        "--output",
        str(output_video_for_row(row, output_root)),
        "--task",
        TARGETS[row["target"]]["task"],
        "--roi-json",
        str(args.roi_json),
        "--placement-side",
        row["placement_side"],
        "--target-side",
        row["target_side"],
        "--target-center-x",
        str(row["target_center_x"]),
        "--target-center-y",
        str(row["target_center_y"]),
        "--crop-left",
        str(args.crop_left),
        "--crop-margin",
        str(args.crop_margin),
        "--sample-stride",
        str(values["sample_stride"]),
        "--pixel-threshold",
        str(values["pixel_threshold"]),
        "--ever-changed-threshold",
        str(values["ever_changed_threshold"]),
        "--changed-ratio",
        str(values["changed_ratio"]),
        "--motion-dilate",
        str(values["motion_dilate"]),
        "--drawing-threshold",
        str(values["drawing_threshold"]),
        "--drawing-dilate",
        str(values["drawing_dilate"]),
        "--placement-margin",
        str(values["placement_margin"]),
        "--board-border-margin",
        str(values["board_border_margin"]),
        f"--scale-range={values['scale_range']}",
        f"--rotation-range={values['rotation_range']}",
        "--seed",
        str(row["seed"]),
        *(["--debug"] if args.debug else []),
        *(["--keep-audio"] if args.keep_audio else []),
        *(["--stabilize"] if args.stabilize else []),
    ]


def retry_profiles(args: argparse.Namespace) -> list[dict[str, Any]]:
    # retry는 mask/margin/크기만 완화한다. top/bottom 반대 배치 규칙은 command_for_row에서 고정된다.
    return [
        {},
        {
            "changed_ratio": max(args.changed_ratio, 0.03),
            "motion_dilate": min(args.motion_dilate, 6),
            "drawing_threshold": min(args.drawing_threshold, 140),
            "drawing_dilate": min(args.drawing_dilate, 4),
            "placement_margin": min(args.placement_margin, 8),
            "board_border_margin": min(args.board_border_margin, 16),
            "scale_range": "0.80,0.95",
            "rotation_range": "-8,8",
        },
        {
            "sample_stride": max(args.sample_stride, 10),
            "pixel_threshold": max(args.pixel_threshold, 25),
            "ever_changed_threshold": max(args.ever_changed_threshold, 50),
            "changed_ratio": max(args.changed_ratio, 0.04),
            "motion_dilate": min(args.motion_dilate, 4),
            "drawing_threshold": min(args.drawing_threshold, 135),
            "drawing_dilate": min(args.drawing_dilate, 3),
            "placement_margin": min(args.placement_margin, 6),
            "board_border_margin": min(args.board_border_margin, 12),
            "scale_range": "0.70,0.90",
            "rotation_range": "-6,6",
        },
        {
            "sample_stride": max(args.sample_stride, 12),
            "pixel_threshold": max(args.pixel_threshold, 28),
            "ever_changed_threshold": max(args.ever_changed_threshold, 55),
            "changed_ratio": max(args.changed_ratio, 0.05),
            "motion_dilate": 0,
            "drawing_threshold": min(args.drawing_threshold, 130),
            "drawing_dilate": 1,
            "placement_margin": 2,
            "board_border_margin": 4,
            "scale_range": "0.50,0.75",
            "rotation_range": "-4,4",
        },
        {
            "sample_stride": max(args.sample_stride, 14),
            "pixel_threshold": max(args.pixel_threshold, 32),
            "ever_changed_threshold": max(args.ever_changed_threshold, 60),
            "changed_ratio": max(args.changed_ratio, 0.06),
            "motion_dilate": 0,
            "drawing_threshold": min(args.drawing_threshold, 125),
            "drawing_dilate": 0,
            "placement_margin": 0,
            "board_border_margin": 0,
            "scale_range": "0.35,0.55",
            "rotation_range": "-3,3",
        },
    ]


def run_plan(args: argparse.Namespace, plan: list[dict[str, Any]], output_root: Path) -> None:
    failures: list[tuple[str, str]] = []
    used_assets_path = output_root / "used_assets.json"
    used_assets: set[str] = set()
    if args.asset_index is not None:
        for row in plan:
            metadata_path = metadata_for_row(row, output_root)
            if metadata_path.exists():
                try:
                    used_assets.add(json.loads(metadata_path.read_text(encoding="utf-8"))["asset"])
                except (OSError, KeyError, json.JSONDecodeError):
                    pass
        used_assets_path.write_text(json.dumps(sorted(used_assets), indent=2), encoding="utf-8")
    for idx, row in enumerate(plan, start=1):
        output_video = output_video_for_row(row, output_root)
        if output_is_complete(row, output_root) and not args.overwrite:
            print(f"[{idx}/{len(plan)}] skip complete {row['episode']} -> {output_video}")
            continue

        print(
            f"[{idx}/{len(plan)}] {row['episode']} target {row['target_side']} -> "
            f"{row['placement_side']} {row['shape']} using {Path(row['asset']).name}"
        )
        last_error = ""
        for attempt_idx, overrides in enumerate(retry_profiles(args), start=1):
            if overrides:
                print(f"  retry profile {attempt_idx}: {overrides}")
            result = subprocess.run(command_for_row(args, row, output_root, overrides))
            if result.returncode == 0:
                if args.asset_index is not None:
                    try:
                        used_assets.add(json.loads(metadata_for_row(row, output_root).read_text(encoding="utf-8"))["asset"])
                        used_assets_path.write_text(json.dumps(sorted(used_assets), indent=2), encoding="utf-8")
                    except (OSError, KeyError, json.JSONDecodeError) as exc:
                        raise RuntimeError(f"Could not record selected asset for {row['episode']}: {exc}") from exc
                break
            last_error = f"return code {result.returncode}"
        else:
            failures.append((row["episode"], last_error))
            print(f"  failed all retry profiles: {row['episode']} ({last_error})")

    if failures:
        print("Failures:")
        for episode, reason in failures:
            print(f"  {episode}: {reason}")
        raise RuntimeError(f"{len(failures)} augmentation(s) failed")


def summarize_plan(plan: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {
        "by_target": {},
        "by_inserted_shape": {},
        "by_target_side": {},
        "by_placement_side": {},
    }
    for row in plan:
        for source_key, bucket_key in [
            ("target", "by_target"),
            ("shape", "by_inserted_shape"),
            ("target_side", "by_target_side"),
            ("placement_side", "by_placement_side"),
        ]:
            value = row[source_key]
            summary[bucket_key][value] = summary[bucket_key].get(value, 0) + 1
    return summary


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Batch augment analyzed erase-shape videos with side-aware distractor placement.")
    parser.add_argument("--dates", nargs="+", default=["0802", "0804", "0805"])
    parser.add_argument("--target-positions-csv", type=Path, default=project_root / "outputs" / "target_position_analysis_0802_0805_board_y" / "target_positions.csv")
    parser.add_argument("--assets-root", type=Path, default=project_root / "candidate_assets" / "records_105")
    parser.add_argument("--asset-index", type=Path, default=None, help="Select source-position-compatible assets after safe locations are computed")
    parser.add_argument("--output-root", type=Path, default=project_root / "outputs" / "0802_0805_board_y_aware_shape_aug")
    parser.add_argument("--roi-json", type=Path, default=project_root / "board_roi.json")
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--run", action="store_true", help="Actually run augmentation. Without this, only writes the plan.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate outputs even when augmented.mp4 already exists")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--keep-audio", action="store_true")
    parser.add_argument("--stabilize", action="store_true")
    parser.add_argument("--crop-left", type=int, default=280)
    parser.add_argument("--crop-margin", type=int, default=8)
    parser.add_argument("--sample-stride", type=int, default=8)
    parser.add_argument("--pixel-threshold", type=int, default=22)
    parser.add_argument("--ever-changed-threshold", type=int, default=45)
    parser.add_argument("--changed-ratio", type=float, default=0.02)
    parser.add_argument("--motion-dilate", type=int, default=8)
    parser.add_argument("--drawing-threshold", type=int, default=145)
    parser.add_argument("--drawing-dilate", type=int, default=6)
    parser.add_argument("--placement-margin", type=int, default=12)
    parser.add_argument("--board-border-margin", type=int, default=20)
    parser.add_argument("--scale-range", default="0.90,1.10")
    parser.add_argument("--rotation-range", default="-10,10")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = read_positions_csv(args.target_positions_csv, set(args.dates))
    plan = make_plan(rows, args.assets_root, args.seed)
    save_plan(plan, args.output_root)

    summary = summarize_plan(plan)
    print(f"Rows: {len(rows)}")
    print(f"Plan items: {len(plan)}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Placement side is always the opposite half of the detected target region.")
    print("Retry profiles keep the side and crop visibility constraints fixed.")

    if args.run:
        run_plan(args, plan, args.output_root)
    else:
        print("Dry run only. Add --run to create augmented videos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
