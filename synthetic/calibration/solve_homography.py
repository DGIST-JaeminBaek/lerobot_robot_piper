#!/usr/bin/env python3
"""Solve and store image-pixel <-> board-plane homographies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from synthetic.calibration.common import (  # noqa: E402
    CALIBRATION_TYPE,
    FORMAT_VERSION,
    POINT_SELECTION_TYPE,
    CalibrationError,
    compute_homography,
    read_json,
    reprojection_statistics,
    require_keys,
    unpack_correspondences,
    validate_image_quad,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Solve a board homography from a point-selection JSON."
    )
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        input_path = args.points.expanduser().resolve()
        selection = read_json(input_path)
        require_keys(
            selection,
            [
                "format_version",
                "type",
                "status",
                "source",
                "board",
                "correspondences",
            ],
            context="point selection",
        )
        if selection["format_version"] != FORMAT_VERSION:
            raise CalibrationError(
                f"unsupported format_version: {selection['format_version']}"
            )
        if selection["type"] != POINT_SELECTION_TYPE:
            raise CalibrationError(
                f"expected type={POINT_SELECTION_TYPE!r}, got {selection['type']!r}"
            )
        source = selection["source"]
        if not isinstance(source, dict):
            raise CalibrationError("source must be an object")
        require_keys(source, ["width", "height", "path"], context="source")
        image_points, board_points = unpack_correspondences(selection)
        validate_image_quad(
            image_points,
            width=int(source["width"]),
            height=int(source["height"]),
        )
        image_to_board, board_to_image = compute_homography(
            image_points,
            board_points,
        )
        reprojection = reprojection_statistics(
            image_points,
            board_points,
            image_to_board,
            board_to_image,
        )
        calibration = {
            "format_version": FORMAT_VERSION,
            "type": CALIBRATION_TYPE,
            "status": "unverified",
            "source_points_file": str(input_path),
            "source": source,
            "board": selection["board"],
            "correspondences": selection["correspondences"],
            "image_to_board_homography": image_to_board.tolist(),
            "board_to_image_homography": board_to_image.tolist(),
            "reprojection": reprojection,
            "verification": {
                "hardware_verified": False,
                "note": (
                    "Homography math is solved, but physical board dimensions "
                    "and robot-frame alignment are not verified."
                ),
            },
        }
        output = args.output.expanduser().resolve()
        write_json(output, calibration, overwrite=args.overwrite)
        print(f"[OK] calibration: {output}")
        print(
            "[INFO] corner reprojection max="
            f"{reprojection['image_error_px_max']:.6f}px"
        )
        print("[STATUS] unverified; not permitted for real robot execution")
        return 0
    except CalibrationError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

