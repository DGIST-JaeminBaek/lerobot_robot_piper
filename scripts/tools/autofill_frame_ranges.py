#!/usr/bin/env python3
"""Fill ``start_frame``/``end_frame`` in a frame-range manifest automatically.

The two boundary events come from ``episode_segmentation``; this script turns
them into the cut the operator would have made:

    start = motion_onset - start_margin, rounded down to round_start
    end   = gripper_release - end_margin

The margins are fitted, not guessed. Against the 60 hand-labelled ``erase the
shape`` episodes the defaults give MAE 5.7 frames on start (94% within 10) and
2.7 frames on end (94% within 5) for the 0727 labelling session. The 0726
session -- the first twelve episodes ever labelled -- is roughly twice as noisy
(sd 14.6 vs 8.8 frames), so the defaults are fitted on 0727 and the residual
against 0726 is expected to be larger.

Two things that did *not* help, so they are deliberately absent:

* Learned regressors. Ridge/GBR/RF over the onset, grasp, contact and length
  features were cross-validated (leave-one-out) at MAE 9-10 frames on start
  against the rule's 5.7 -- with 60 samples they fit the labeller's rounding
  noise rather than the boundary. Same story on end (2.95 vs 2.78).
* Alternative onset detectors. Joint velocity, absolute displacement from the
  start pose, windowed path length and follower-state (rather than commanded)
  position all land within half a frame of each other. The start-side spread is
  label noise, not detector error, and no detector can recover it.

Use ``--fit`` to re-derive the margins from the labels already in the manifest;
that is the maintenance path once a new labelling session exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from episode_segmentation import gripper_release, motion_onset, suggest_range


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "configs/erase_shape_frame_ranges.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help=f"(default: {DEFAULT_MANIFEST})")
    parser.add_argument("--write", action="store_true", help="Write the detected ranges back into the manifest.")
    parser.add_argument("--start-margin", type=int, default=22, help="Frames kept before motion onset (default: 22).")
    parser.add_argument("--end-margin", type=int, default=6, help="Frames dropped before gripper release (default: 6).")
    parser.add_argument("--round-start", type=int, default=10, help="Round start down to this multiple (0 disables).")
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Keep existing non-null start/end values and only fill in the blanks.",
    )
    parser.add_argument(
        "--fit",
        action="store_true",
        help="Re-derive the margins from the manifest's existing labels, then use them.",
    )
    parser.add_argument("--tolerance", type=int, default=10, help="Flag |detected - label| above this (default: 10).")
    return parser.parse_args()


def resolve_source(source: str) -> Path:
    """Locate a dataset by manifest path, falling back to a name search under records/."""
    path = Path(source)
    direct = (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if (direct / "meta/info.json").is_file():
        return direct
    matches = sorted(REPO_ROOT.glob(f"records/**/{path.name}/meta/info.json"))
    if not matches:
        raise FileNotFoundError(f"Cannot locate source dataset: {source}")
    return matches[0].parent.parent


def load_action(root: Path) -> np.ndarray:
    """Return the (T, 7) commanded joint+gripper trajectory of a one-episode dataset."""
    parquet_files = sorted(root.glob("data/chunk-*/file-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No episode parquet below: {root}")
    frame = pd.concat([pd.read_parquet(path) for path in parquet_files]).sort_values("frame_index")
    return np.stack(frame["action"].to_numpy())


def fit_margins(episodes: list[dict[str, Any]], round_start: int) -> tuple[int, int]:
    """Grid-search the margins that best reproduce the labels already present."""
    labelled = [entry for entry in episodes if entry.get("start_frame") is not None]
    if len(labelled) < 5:
        raise SystemExit(f"--fit needs at least 5 labelled episodes, found {len(labelled)}")

    onsets, releases, starts, ends = [], [], [], []
    for entry in labelled:
        action = load_action(resolve_source(entry["source_dataset"]))
        release = gripper_release(action)
        if release is None:
            continue
        onsets.append(motion_onset(action))
        releases.append(release)
        starts.append(entry["start_frame"])
        ends.append(entry["end_frame"])
    onsets, releases = np.array(onsets), np.array(releases)
    starts, ends = np.array(starts), np.array(ends)

    def start_error(margin: int) -> float:
        guess = np.maximum(0, onsets - margin)
        if round_start > 0:
            guess -= guess % round_start
        return float(np.abs(guess - starts).mean())

    start_margin = min(range(0, 60), key=start_error)
    end_margin = min(range(-10, 30), key=lambda m: np.abs((releases - m) - ends).mean())
    print(
        f"fitted on {len(starts)} labelled episodes: "
        f"start_margin={start_margin} (MAE {start_error(start_margin):.2f}), "
        f"end_margin={end_margin} (MAE {np.abs((releases - end_margin) - ends).mean():.2f})\n"
    )
    return start_margin, end_margin


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    manifest: dict[str, Any] = json.loads(manifest_path.read_text())
    episodes = manifest["episodes"]

    if args.fit:
        args.start_margin, args.end_margin = fit_margins(episodes, args.round_start)

    deltas: list[tuple[int, int]] = []
    flagged: list[str] = []

    print(f"{'episode':<34}{'frames':>7}{'start':>8}{'end':>7}{'Δstart':>8}{'Δend':>7}")
    for entry in episodes:
        name = Path(entry["source_dataset"]).name
        if not entry.get("enabled", True):
            print(f"{name:<34}{'':>7}{'disabled':>8}")
            continue

        action = load_action(resolve_source(entry["source_dataset"]))
        start, end, warnings = suggest_range(action, args.start_margin, args.end_margin, args.round_start)

        old_start, old_end = entry.get("start_frame"), entry.get("end_frame")
        if args.only_missing and old_start is not None and old_end is not None:
            start, end = old_start, old_end

        d_start = "" if old_start is None else f"{start - old_start:+d}"
        d_end = "" if old_end is None else f"{end - old_end:+d}"
        if old_start is not None and old_end is not None:
            deltas.append((start - old_start, end - old_end))
            if max(abs(start - old_start), abs(end - old_end)) > args.tolerance:
                flagged.append(name)
        for warning in warnings:
            flagged.append(f"{name}: {warning}")

        print(f"{name:<34}{len(action):>7}{start:>8}{end:>7}{d_start:>8}{d_end:>7}")
        if args.write:
            entry["start_frame"] = int(start)
            entry["end_frame"] = int(end)

    if deltas:
        array = np.array(deltas)
        for column, label in ((0, "start"), (1, "end")):
            error = array[:, column]
            print(
                f"\n{label:<5} vs labels: MAE {np.abs(error).mean():.2f}  bias {error.mean():+.1f}  "
                f"sd {error.std():.1f}  within 10: {np.mean(np.abs(error) <= 10) * 100:.0f}%"
            )
    if flagged:
        print(f"\nreview manually ({len(flagged)}):")
        for item in flagged:
            print(f"  {item}")

    if args.write:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        print(f"\nwrote {manifest_path}")
    else:
        print("\ndry run — pass --write to update the manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
