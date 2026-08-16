#!/usr/bin/env python3
"""Solve and store the board-plane <-> Piper-base rigid transform."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from synthetic.transforms.board_base import (  # noqa: E402
    FORMAT_VERSION,
    TRANSFORM_TYPE,
    TransformError,
    embed_board_xy,
    parse_correspondences,
    read_json,
    residual_statistics,
    solve_rigid_transform,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Solve a board-plane <-> Piper-base rigid transform from mm "
            "correspondences. Never connects to the robot."
        )
    )
    parser.add_argument("--correspondences", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        input_path = args.correspondences.expanduser().resolve()
        payload = read_json(input_path)
        names, board_xy, base_xyz, board_unit, base_unit = parse_correspondences(payload)
        board_xyz = embed_board_xy(board_xy)

        transform = solve_rigid_transform(board_xyz, base_xyz)
        residuals = residual_statistics(transform, board_xyz, base_xyz)

        inverse = transform.inverse()
        output_payload = {
            "format_version": FORMAT_VERSION,
            "type": TRANSFORM_TYPE,
            "status": "unverified",
            "source_correspondences_file": str(input_path),
            "board_unit": board_unit,
            "base_unit": base_unit,
            "correspondences": payload["correspondences"],
            "transform_base_from_board": transform.to_dict(),
            "transform_board_from_base": inverse.to_dict(),
            "plane_normal_in_base": transform.plane_normal().tolist(),
            "residuals_mm": residuals,
            "verification": {
                "hardware_verified": False,
                "note": (
                    "Rigid-alignment math is solved, but the underlying "
                    "base_xyz correspondences are not verified against the "
                    "real robot."
                ),
            },
        }

        output = args.output.expanduser().resolve()
        write_json(output, output_payload, overwrite=args.overwrite)
        print(f"[OK] board-base transform: {output}")
        print(
            f"[INFO] residual mean={residuals['mean_mm']:.6f}mm "
            f"max={residuals['max_mm']:.6f}mm over {len(names)} points"
        )
        print("[STATUS] unverified; not permitted for real robot execution")
        return 0
    except TransformError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
