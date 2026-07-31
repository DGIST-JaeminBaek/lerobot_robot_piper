#!/usr/bin/env python3
"""Query and re-check a solved board-plane <-> Piper-base transform.

Recomputes residuals against the stored correspondences and can convert
individual points in either direction. Never connects to the robot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from synthetic.calibration.common import parse_xy  # noqa: E402
from synthetic.transforms.board_base import (  # noqa: E402
    FORMAT_VERSION,
    TRANSFORM_TYPE,
    RigidTransform,
    TransformError,
    embed_board_xy,
    parse_correspondences,
    read_json,
    require_keys,
    residual_statistics,
    write_json,
)


def parse_xyz(text: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise TransformError(f"expected x,y,z, got {text!r}")
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError as exc:
        raise TransformError(f"expected numeric x,y,z, got {text!r}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute residuals for a solved board-base transform and "
            "query board_xy<->base_xyz points. No robot connection is used."
        )
    )
    parser.add_argument("--transform", type=Path, required=True)
    parser.add_argument(
        "--board-xy",
        type=parse_xy,
        action="append",
        default=[],
        help="board_xy mm to convert to base_xyz; may be repeated",
    )
    parser.add_argument(
        "--base-xyz",
        type=parse_xyz,
        action="append",
        default=[],
        help="base_xyz mm to convert to board_xy (z assumed 0); may be repeated",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path to write a recomputed residual report JSON",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def load_transform_payload(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    require_keys(
        payload,
        [
            "format_version",
            "type",
            "status",
            "board_unit",
            "base_unit",
            "correspondences",
            "transform_base_from_board",
            "transform_board_from_base",
        ],
        context="board-base transform",
    )
    if payload["format_version"] != FORMAT_VERSION:
        raise TransformError(f"unsupported format_version: {payload['format_version']}")
    if payload["type"] != TRANSFORM_TYPE:
        raise TransformError(f"expected type={TRANSFORM_TYPE!r}, got {payload['type']!r}")
    return payload


def main() -> int:
    args = build_parser().parse_args()
    try:
        transform_path = args.transform.expanduser().resolve()
        payload = load_transform_payload(transform_path)
        base_from_board = RigidTransform.from_dict(payload["transform_base_from_board"])
        board_from_base = RigidTransform.from_dict(payload["transform_board_from_base"])

        names, board_xy, base_xyz, _board_unit, _base_unit = parse_correspondences(
            {
                "format_version": payload["format_version"],
                "type": "board_base_correspondence_set",
                "board_unit": payload["board_unit"],
                "base_unit": payload["base_unit"],
                "correspondences": payload["correspondences"],
            }
        )
        board_xyz = embed_board_xy(board_xy)
        residuals = residual_statistics(base_from_board, board_xyz, base_xyz)
        print(
            f"[INFO] recomputed residual mean={residuals['mean_mm']:.6f}mm "
            f"max={residuals['max_mm']:.6f}mm over {len(names)} points"
        )
        print(f"[STATUS] {payload['status']}; recomputation is not robot validation")

        for xy in args.board_xy:
            board_point = embed_board_xy([xy])
            base_point = base_from_board.apply(board_point)[0]
            print(
                f"[QUERY] board_xy=({xy[0]:.3f}, {xy[1]:.3f}) mm -> "
                f"base_xyz=({base_point[0]:.3f}, {base_point[1]:.3f}, "
                f"{base_point[2]:.3f}) mm"
            )

        for xyz in args.base_xyz:
            board_point = board_from_base.apply([list(xyz)])[0]
            print(
                f"[QUERY] base_xyz=({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}) mm -> "
                f"board_xy=({board_point[0]:.3f}, {board_point[1]:.3f}) mm "
                f"(board_z={board_point[2]:.6f} mm, expect ~0)"
            )

        if args.report is not None:
            report = {
                "source_transform_file": str(transform_path),
                "status": payload["status"],
                "recomputed_residuals_mm": residuals,
            }
            report_path = args.report.expanduser().resolve()
            write_json(report_path, report, overwrite=args.overwrite)
            print(f"[OK] report: {report_path}")
        return 0
    except TransformError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
