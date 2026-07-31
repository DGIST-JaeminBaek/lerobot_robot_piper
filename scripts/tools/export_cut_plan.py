#!/usr/bin/env python3
"""Write a cut plan for a set of episodes without touching anything that exists.

Reads either an existing manifest (``--manifest``) or a directory of recordings
(``--source-dir``), detects each episode's range with ``episode_segmentation``,
and emits three views of the same decision:

``<name>.json``
    Same schema as ``configs/erase_shape_frame_ranges.json``, so
    ``prepare_erase_shape_dataset.py --manifest <name>.json`` consumes it
    directly. When built from a manifest, any label already there is carried
    over into ``manual_start_frame``/``manual_end_frame`` so nothing is lost.
``<name>.csv``
    One row per episode with frames *and* seconds -- the form to open in a
    spreadsheet or hand to a video editor.
``<name>.sh``
    Frame-exact ``ffmpeg`` commands, one per camera per episode. ``trim`` is
    used rather than ``-c copy`` because the recordings have a 250-frame
    keyframe interval, so a stream copy would snap the cut up to 8 seconds away.

The input is only ever read. Outputs refuse to overwrite unless ``--force``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from autofill_frame_ranges import load_action, resolve_source
from episode_segmentation import (
    DEFAULT_END_EVENT,
    END_EVENTS,
    END_MARGIN,
    motion_onset,
    release_frame,
    suggest_range,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "configs/erase_shape_frame_ranges.json"
DEFAULT_OUTPUT = REPO_ROOT / "tmp/cut_plans/cut_plan"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--manifest", type=Path, help=f"Existing manifest to re-cut (default: {DEFAULT_MANIFEST})")
    source.add_argument("--source-dir", type=Path, nargs="+", help="Directories of recordings with no manifest yet.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output path without extension.")
    parser.add_argument("--task", default="erase the shape", help="target_task written into the JSON.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--start-margin", type=int, default=22)
    parser.add_argument("--end-margin", type=int, default=None,
                        help=f"Frames dropped before the end event (default: per event, {END_MARGIN}).")
    parser.add_argument("--end-event", choices=END_EVENTS, default=DEFAULT_END_EVENT,
                        help=f"Which end of the gripper release closes the episode (default: {DEFAULT_END_EVENT}).")
    parser.add_argument("--round-start", type=int, default=10)
    return parser.parse_args()


def discover(directories: list[Path]) -> list[Path]:
    found: list[Path] = []
    for directory in directories:
        directory = directory.expanduser().resolve()
        if (directory / "meta/info.json").is_file():
            found.append(directory)
            continue
        found.extend(sorted(path.parent.parent for path in directory.glob("*/meta/info.json")))
    if not found:
        raise SystemExit(f"No LeRobot datasets found under: {', '.join(str(d) for d in directories)}")
    return found


def episode_rows(sources: list[tuple[Path, dict[str, Any] | None]], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for root, entry in sources:
        action = load_action(root)
        start, end, warnings = suggest_range(action, args.start_margin, args.end_margin,
                                             args.round_start, args.end_event)
        fps = json.loads((root / "meta/info.json").read_text())["fps"]
        release = release_frame(action, args.end_event)
        rows.append(
            {
                "name": root.name,
                "root": root,
                "source_dataset": root.resolve().relative_to(REPO_ROOT).as_posix(),
                "total_frames": len(action),
                "fps": fps,
                "start_frame": start,
                "end_frame": end,
                "kept_frames": end - start,
                "start_sec": round(start / fps, 3),
                "end_sec": round(end / fps, 3),
                "duration_sec": round((end - start) / fps, 3),
                "motion_onset": motion_onset(action),
                "gripper_release": -1 if release is None else release,
                "manual_start_frame": None if entry is None else entry.get("start_frame"),
                "manual_end_frame": None if entry is None else entry.get("end_frame"),
                "warnings": "; ".join(warnings),
            }
        )
    return rows


def write_json(rows: list[dict[str, Any]], path: Path, task: str) -> None:
    document = {
        "format_version": 1,
        "target_task": task,
        "generated_by": "scripts/tools/export_cut_plan.py",
        "range_semantics": "start_frame is inclusive; end_frame is exclusive",
        "instructions": [
            "Ranges are detected: start = motion onset - margin, end = gripper release - margin.",
            "manual_* fields are the previous hand labels, kept for reference only.",
            "Set enabled=false to exclude an entire episode.",
        ],
        "episodes": [
            {
                "source_dataset": row["source_dataset"],
                "total_frames": row["total_frames"],
                "enabled": True,
                "start_frame": row["start_frame"],
                "end_frame": row["end_frame"],
                **(
                    {"manual_start_frame": row["manual_start_frame"], "manual_end_frame": row["manual_end_frame"]}
                    if row["manual_start_frame"] is not None
                    else {}
                ),
            }
            for row in rows
        ],
    }
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "name", "source_dataset", "total_frames", "fps", "start_frame", "end_frame", "kept_frames",
        "start_sec", "end_sec", "duration_sec", "motion_onset", "gripper_release",
        "manual_start_frame", "manual_end_frame", "warnings",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_shell(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "# Frame-exact video cuts. Generated by scripts/tools/export_cut_plan.py -- regenerate, do not edit.",
        "set -euo pipefail",
        'OUT="${1:?usage: $0 <output-dir>}"',
        "",
    ]
    for row in rows:
        lines.append(f"# {row['name']}: frames [{row['start_frame']}, {row['end_frame']}) "
                     f"= {row['duration_sec']}s of {row['total_frames'] / row['fps']:.1f}s")
        for video in sorted(row["root"].glob("videos/*/chunk-*/file-*.mp4")):
            camera = video.parts[-3]
            target = f'"$OUT/{row["name"]}__{camera}.mp4"'
            lines.append(
                f'mkdir -p "$OUT" && ffmpeg -y -i "{video}" '
                f'-vf "trim=start_frame={row["start_frame"]}:end_frame={row["end_frame"]},setpts=PTS-STARTPTS" '
                f"-an {target}"
            )
        lines.append("")
    path.write_text("\n".join(lines))
    path.chmod(0o755)


def main() -> int:
    args = parse_args()

    if args.source_dir:
        sources = [(root, None) for root in discover(args.source_dir)]
    else:
        manifest_path = (args.manifest or DEFAULT_MANIFEST).expanduser().resolve()
        manifest = json.loads(manifest_path.read_text())
        sources = [(resolve_source(entry["source_dataset"]), entry) for entry in manifest["episodes"]]
        print(f"read {manifest_path} ({len(sources)} episodes, unchanged)")

    rows = episode_rows(sources, args)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    targets = {suffix: output.with_suffix(suffix) for suffix in (".json", ".csv", ".sh")}
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing and not args.force:
        raise SystemExit("refusing to overwrite (pass --force):\n  " + "\n  ".join(existing))

    write_json(rows, targets[".json"], args.task)
    write_csv(rows, targets[".csv"])
    write_shell(rows, targets[".sh"])

    kept = sum(row["kept_frames"] for row in rows)
    total = sum(row["total_frames"] for row in rows)
    print(f"\n{len(rows)} episodes: {total} -> {kept} frames ({kept / total * 100:.0f}% kept, "
          f"{(total - kept) / rows[0]['fps'] / 60:.1f} min removed)")

    changed = [row for row in rows if row["manual_start_frame"] is not None
               and (row["start_frame"], row["end_frame"]) != (row["manual_start_frame"], row["manual_end_frame"])]
    if changed:
        print(f"differs from the previous hand labels on {len(changed)}/{len(rows)} episodes "
              f"(both values kept in the JSON)")
    warned = [row for row in rows if row["warnings"]]
    for row in warned:
        print(f"  warning — {row['name']}: {row['warnings']}")

    for path in targets.values():
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
