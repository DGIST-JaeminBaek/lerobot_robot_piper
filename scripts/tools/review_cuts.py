#!/usr/bin/env python3
"""Render a review sheet so a cut can be checked by eye instead of by number.

For each episode it draws the four boundary frames -- the manifest's own
start/end next to the ones ``episode_segmentation`` proposes -- above the two
traces the detector actually reads: per-frame joint step (motion onset) and
gripper command (release). Vertical lines mark both cuts, so a disagreement is
visible as a gap between the red and green line.

    python scripts/tools/review_cuts.py                     # the episodes that disagree
    python scripts/tools/review_cuts.py --all               # every episode in the manifest
    python scripts/tools/review_cuts.py --episode 0726-162803 0727-125938

Sheets land in ``tmp/cut_review/`` as one PNG per episode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import av
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from autofill_frame_ranges import load_action, resolve_source
from episode_segmentation import gripper_plateau, gripper_release, motion_onset, suggest_range


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "configs/erase_shape_frame_ranges.json"
DEFAULT_OUTPUT = REPO_ROOT / "tmp/cut_review"
VIDEO_KEY = "observation.images.top"

LABEL_COLOR = "#d62728"
AUTO_COLOR = "#2ca02c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--all", action="store_true", help="Render every episode, not just the disagreements.")
    parser.add_argument("--episode", nargs="+", default=None, help="Substrings of the episode names to render.")
    parser.add_argument("--tolerance", type=int, default=10, help="Disagreement threshold in frames (default: 10).")
    parser.add_argument("--start-margin", type=int, default=22)
    parser.add_argument("--end-margin", type=int, default=6)
    parser.add_argument("--round-start", type=int, default=10)
    return parser.parse_args()


def grab_frames(root: Path, wanted: list[int]) -> dict[int, np.ndarray]:
    """Decode the requested frame indices from the top camera."""
    videos = sorted((root / "videos" / VIDEO_KEY).rglob("*.mp4"))
    if not videos:
        return {}
    targets = sorted(set(wanted))
    frames: dict[int, np.ndarray] = {}
    with av.open(str(videos[0])) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for index, frame in enumerate(container.decode(stream)):
            if index in targets:
                frames[index] = frame.to_ndarray(format="rgb24")
                if len(frames) == len(targets):
                    break
    return frames


def draw_sheet(name: str, action: np.ndarray, entry: dict[str, Any], auto: tuple[int, int], root: Path, out: Path):
    label_start, label_end = entry.get("start_frame"), entry.get("end_frame")
    auto_start, auto_end = auto
    onset, release = motion_onset(action), gripper_release(action)

    picks = [("manifest start", label_start), ("auto start", auto_start), ("manifest end", label_end), ("auto end", auto_end)]
    picks = [(caption, frame) for caption, frame in picks if frame is not None]
    images = grab_frames(root, [min(frame, len(action) - 1) for _, frame in picks])

    fig = plt.figure(figsize=(16, 8.5))
    grid = fig.add_gridspec(3, 4, height_ratios=[2.4, 1, 1], hspace=0.35, wspace=0.06)

    for column, (caption, frame) in enumerate(picks):
        ax = fig.add_subplot(grid[0, column])
        image = images.get(min(frame, len(action) - 1))
        if image is None:
            ax.text(0.5, 0.5, "no video", ha="center", va="center")
        else:
            ax.imshow(image)
        color = LABEL_COLOR if caption.startswith("manifest") else AUTO_COLOR
        ax.set_title(f"{caption}  ·  frame {frame}", color=color, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)

    step = np.abs(np.diff(action[:, :6], axis=0)).max(axis=1)
    plateau = gripper_plateau(action)

    def mark(ax):
        if label_start is not None:
            ax.axvline(label_start, color=LABEL_COLOR, lw=1.6, label=f"manifest [{label_start}, {label_end})")
            ax.axvline(label_end, color=LABEL_COLOR, lw=1.6)
        ax.axvline(auto_start, color=AUTO_COLOR, lw=1.6, ls="--", label=f"auto [{auto_start}, {auto_end})")
        ax.axvline(auto_end, color=AUTO_COLOR, lw=1.6, ls="--")

    ax_motion = fig.add_subplot(grid[1, :])
    ax_motion.plot(step, color="#444", lw=0.8)
    ax_motion.axhline(0.5, color="#888", lw=0.8, ls=":")
    ax_motion.plot([onset], [step[onset]], "o", color="#1f77b4", ms=8, label=f"motion onset {onset}")
    ax_motion.set_ylabel("joint step\n(deg/frame)")
    ax_motion.set_ylim(0, min(step.max() * 1.1, 12))
    mark(ax_motion)
    ax_motion.legend(loc="upper right", fontsize=9, ncol=3)

    ax_grip = fig.add_subplot(grid[2, :], sharex=ax_motion)
    ax_grip.plot(action[:, 6], color="#444", lw=1.0)
    ax_grip.axhline(plateau, color="#888", lw=0.8, ls=":")
    if release is not None:
        ax_grip.plot([release], [action[release, 6]], "o", color="#ff7f0e", ms=8, label=f"gripper release {release}")
        ax_grip.legend(loc="upper right", fontsize=9)
    ax_grip.set_ylabel(f"gripper\n(hold {plateau:.0f})")
    ax_grip.set_xlabel("frame")
    mark(ax_grip)

    delta = "" if label_start is None else f"   ·   Δstart {auto_start - label_start:+d}, Δend {auto_end - label_end:+d}"
    fig.suptitle(f"{name}   ·   {len(action)} frames{delta}", fontsize=13)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.expanduser().resolve().read_text())

    written = []
    for entry in manifest["episodes"]:
        name = Path(entry["source_dataset"]).name
        if args.episode and not any(pattern in name for pattern in args.episode):
            continue

        root = resolve_source(entry["source_dataset"])
        action = load_action(root)
        start, end, _ = suggest_range(action, args.start_margin, args.end_margin, args.round_start)

        if not args.all and not args.episode:
            label_start, label_end = entry.get("start_frame"), entry.get("end_frame")
            if label_start is None:
                continue
            if max(abs(start - label_start), abs(end - label_end)) <= args.tolerance:
                continue

        path = args.output / f"{name}.png"
        draw_sheet(name, action, entry, (start, end), root, path)
        written.append(path)
        print(f"wrote {path}")

    print(f"\n{len(written)} sheet(s) in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
