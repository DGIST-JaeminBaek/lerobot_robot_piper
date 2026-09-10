from __future__ import annotations

"""Compare bad/hold candidate composites with their original first frames."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from label_asset_candidates import composite_on_board, display_array, text


def load_manifest(input_root: Path) -> dict[str, tuple[Path, tuple[int, int]]]:
    entries: dict[str, tuple[Path, tuple[int, int]]] = {}
    for manifest in input_root.rglob("*manifest.csv"):
        with manifest.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                try:
                    output = Path(row["output"])
                    source = Path(row["input"])
                    if not output.is_absolute():
                        output = Path.cwd() / output
                    if not source.is_absolute():
                        source = Path.cwd() / source
                    entries[str(output.resolve())] = (source.resolve(), (int(row["crop_x"]), int(row["crop_y"])))
                except (KeyError, TypeError, ValueError):
                    continue
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description="Side-by-side original-first-frame comparison for bad/hold assets.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--background", type=Path, default=Path("review/whiteboard_last_frame.png"))
    args = parser.parse_args()

    with args.decisions.open(newline="", encoding="utf-8") as file:
        flagged = [row for row in csv.DictReader(file) if row["status"] in {"bad", "hold", "skip"}]
    manifest = load_manifest(args.input_root)
    items = [(Path(row["candidate"]), row["status"], manifest.get(str(Path(row["candidate"]).resolve()))) for row in flagged]
    items = [(candidate, status, detail) for candidate, status, detail in items if detail is not None]
    if not items:
        raise RuntimeError("No bad/hold decisions with matching crop-manifest entries.")
    background = cv2.imread(str(args.background), cv2.IMREAD_COLOR)
    if background is None:
        raise FileNotFoundError(args.background)

    index, window = 0, "Flagged asset comparison"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1440, 920)
    while True:
        candidate, status, detail = items[index]
        source, top_left = detail
        original = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if original is None:
            raise RuntimeError(f"Cannot read source frame: {source}")
        composited = composite_on_board(background, candidate, *top_left)
        canvas = np.full((920, 1440, 3), 238, dtype=np.uint8)
        text(canvas, f"{status.upper()}  {index + 1}/{len(items)}  |  {candidate.name}", (18, 32))
        text(canvas, "Left: original first frame     Right: clean last frame + extracted asset at original crop position", (18, 63), scale=0.5)
        left, right = display_array(original, 700, 810), display_array(composited, 700, 810)
        canvas[92:902, 10:710] = left
        canvas[92:902, 730:1430] = right
        cv2.imshow(window, canvas)
        key = cv2.waitKeyEx(0)
        if key in (ord("q"), 27):
            break
        if key in (2424832, 81):
            index = max(0, index - 1)
        elif key in (2555904, 83):
            index = min(len(items) - 1, index + 1)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
