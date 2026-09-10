from __future__ import annotations

"""Build a source-position index for automatically cropped candidate assets."""

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.assets_root / "asset_index.csv"
    rows = []
    for manifest in args.assets_root.rglob("*manifest.csv"):
        with manifest.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                try:
                    asset = Path(row["output"])
                    if not asset.is_absolute():
                        # Crop manifests store paths relative to the project
                        # root (e.g. candidate_assets/...).  Resolve them from
                        # the supplied assets root, not the caller's CWD.
                        asset = args.assets_root.resolve().parent.parent / asset
                    asset = asset.resolve()
                    rows.append({
                        "asset_path": str(asset),
                        "shape": asset.parent.name,
                        "source_center_x": int(row["x"]) + int(row["w"]) / 2.0,
                        "source_center_y": int(row["y"]) + int(row["h"]) / 2.0,
                        "source_width": int(row["w"]),
                        "source_height": int(row["h"]),
                        # crop_x/y is the original padded PNG top-left.  It is
                        # what lets the augmentation pipeline try the exact
                        # source-board location before using a safe fallback.
                        "source_x": int(row.get("crop_x", row["x"])),
                        "source_y": int(row.get("crop_y", row["y"])),
                        "manifest": str(manifest.resolve()),
                    })
                except (KeyError, TypeError, ValueError):
                    continue
    if not rows:
        raise RuntimeError(f"No crop-manifest rows under {args.assets_root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda item: item["asset_path"]))
    print(f"Wrote {len(rows)} indexed assets to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
