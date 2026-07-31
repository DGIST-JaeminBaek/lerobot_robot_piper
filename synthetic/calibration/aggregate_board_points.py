#!/usr/bin/env python3
"""Aggregate repeated board-corner selections from a fixed camera session.

The default representative is the per-corner median pixel coordinate, which is
more robust to an occasional inaccurate click than a plain mean. Homography
matrices themselves are never averaged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from synthetic.calibration.common import (  # noqa: E402
    FORMAT_VERSION,
    POINT_NAMES,
    POINT_SELECTION_TYPE,
    CalibrationError,
    read_json,
    require_keys,
    unpack_correspondences,
    validate_image_quad,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Combine repeated board-corner selections from one fixed-camera "
            "session. Homography matrices are not averaged."
        )
    )
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=["median", "mean"],
        default="median",
        help="Representative pixel coordinate per corner (default: median)",
    )
    parser.add_argument(
        "--fail-above-px",
        type=float,
        default=None,
        help=(
            "Fail if any observation is farther than this many pixels from "
            "its representative corner"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _board_identity(board: Any) -> tuple[str, float, float, str, str, str]:
    if not isinstance(board, dict):
        raise CalibrationError("board must be an object")
    require_keys(
        board,
        [
            "unit",
            "width",
            "height",
            "origin",
            "x_direction",
            "y_direction",
        ],
        context="board",
    )
    return (
        str(board["unit"]),
        float(board["width"]),
        float(board["height"]),
        str(board["origin"]),
        str(board["x_direction"]),
        str(board["y_direction"]),
    )


def _source_identity(source: Any) -> tuple[int, int]:
    if not isinstance(source, dict):
        raise CalibrationError("source must be an object")
    require_keys(source, ["width", "height", "path"], context="source")
    return int(source["width"]), int(source["height"])


def load_observations(
    paths: list[Path],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    if len(paths) < 2:
        raise CalibrationError("at least two point-selection files are required")
    selections: list[dict[str, Any]] = []
    image_observations: list[np.ndarray] = []
    expected_board: tuple[str, float, float, str, str, str] | None = None
    expected_image_size: tuple[int, int] | None = None
    board_points_reference: np.ndarray | None = None

    for path in paths:
        resolved = path.expanduser().resolve()
        payload = read_json(resolved)
        require_keys(
            payload,
            [
                "format_version",
                "type",
                "status",
                "source",
                "board",
                "correspondences",
            ],
            context=f"selection {resolved}",
        )
        if payload["format_version"] != FORMAT_VERSION:
            raise CalibrationError(
                f"{resolved}: unsupported format_version "
                f"{payload['format_version']}"
            )
        if payload["type"] != POINT_SELECTION_TYPE:
            raise CalibrationError(
                f"{resolved}: expected type={POINT_SELECTION_TYPE!r}"
            )
        board_identity = _board_identity(payload["board"])
        image_size = _source_identity(payload["source"])
        if expected_board is None:
            expected_board = board_identity
            expected_image_size = image_size
        elif board_identity != expected_board:
            raise CalibrationError(
                f"{resolved}: board definition differs from the first input"
            )
        elif image_size != expected_image_size:
            raise CalibrationError(
                f"{resolved}: image size {image_size} differs from "
                f"{expected_image_size}"
            )

        image_points, board_points = unpack_correspondences(payload)
        validate_image_quad(
            image_points,
            width=image_size[0],
            height=image_size[1],
        )
        if board_points_reference is None:
            board_points_reference = board_points
        elif not np.allclose(board_points, board_points_reference, atol=1e-9):
            raise CalibrationError(
                f"{resolved}: board coordinates differ from the first input"
            )

        payload["_input_file"] = str(resolved)
        selections.append(payload)
        image_observations.append(image_points)

    assert board_points_reference is not None
    return (
        selections,
        np.stack(image_observations, axis=0),
        board_points_reference,
    )


def aggregate_observations(
    observations: np.ndarray,
    *,
    method: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(observations, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (4, 2):
        raise CalibrationError(
            f"observations must have shape (N, 4, 2), got {values.shape}"
        )
    if values.shape[0] < 2:
        raise CalibrationError("at least two observations are required")
    if not np.isfinite(values).all():
        raise CalibrationError("observations contain NaN or Inf")
    if method == "median":
        representative = np.median(values, axis=0)
    elif method == "mean":
        representative = np.mean(values, axis=0)
    else:
        raise CalibrationError(f"unsupported aggregation method: {method}")

    radial = np.linalg.norm(values - representative[None, :, :], axis=2)
    per_point: list[dict[str, Any]] = []
    for point_index, point_name in enumerate(POINT_NAMES):
        point_values = values[:, point_index, :]
        per_point.append(
            {
                "name": point_name,
                "representative_image_px": representative[point_index].tolist(),
                "mean_image_px": np.mean(point_values, axis=0).tolist(),
                "median_image_px": np.median(point_values, axis=0).tolist(),
                "std_image_px": np.std(point_values, axis=0).tolist(),
                "radial_deviation_px": radial[:, point_index].tolist(),
                "radial_deviation_px_mean": float(
                    radial[:, point_index].mean()
                ),
                "radial_deviation_px_max": float(
                    radial[:, point_index].max()
                ),
            }
        )
    statistics = {
        "sample_count": int(values.shape[0]),
        "method": method,
        "global_radial_deviation_px_mean": float(radial.mean()),
        "global_radial_deviation_px_max": float(radial.max()),
        "per_point": per_point,
    }
    return representative, statistics


def build_aggregated_payload(
    *,
    selections: list[dict[str, Any]],
    observations: np.ndarray,
    board_points: np.ndarray,
    method: str,
) -> dict[str, Any]:
    representative, statistics = aggregate_observations(
        observations,
        method=method,
    )
    source = selections[0]["source"]
    validate_image_quad(
        representative,
        width=int(source["width"]),
        height=int(source["height"]),
    )
    correspondences = [
        {
            "name": name,
            "image_px": representative[index].tolist(),
            "board_xy": board_points[index].tolist(),
        }
        for index, name in enumerate(POINT_NAMES)
    ]
    sources = [
        {
            "input_file": selection["_input_file"],
            "source": selection["source"],
            "image_points": observations[index].tolist(),
        }
        for index, selection in enumerate(selections)
    ]
    return {
        "format_version": FORMAT_VERSION,
        "type": POINT_SELECTION_TYPE,
        "status": "unverified",
        "source": source,
        "board": selections[0]["board"],
        "point_order": list(POINT_NAMES),
        "correspondences": correspondences,
        "aggregation": statistics,
        "observations": sources,
        "note": (
            "Representative image points were aggregated before solving one "
            "homography. Homography matrices were not averaged."
        ),
    }


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.fail_above_px is not None and args.fail_above_px <= 0:
            raise CalibrationError("--fail-above-px must be positive")
        selections, observations, board_points = load_observations(args.inputs)
        payload = build_aggregated_payload(
            selections=selections,
            observations=observations,
            board_points=board_points,
            method=args.method,
        )
        max_deviation = float(
            payload["aggregation"]["global_radial_deviation_px_max"]
        )
        if (
            args.fail_above_px is not None
            and max_deviation > args.fail_above_px
        ):
            raise CalibrationError(
                f"maximum point deviation {max_deviation:.3f}px exceeds "
                f"--fail-above-px={args.fail_above_px:.3f}"
            )

        output = args.output.expanduser().resolve()
        write_json(output, payload, overwrite=args.overwrite)
        print(f"[OK] aggregated point selection: {output}")
        print(
            f"[INFO] samples={len(selections)}, method={args.method}, "
            f"mean deviation="
            f"{payload['aggregation']['global_radial_deviation_px_mean']:.3f}px, "
            f"max deviation={max_deviation:.3f}px"
        )
        print("[STATUS] unverified; inspect spread before solving homography")
        return 0
    except CalibrationError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

