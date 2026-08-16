#!/usr/bin/env python3
"""Visualize and query a solved board homography.

This tool produces an overlay and optionally reports board coordinates for a
pixel. With --interactive it reports coordinates for mouse clicks. It never
connects to ROS, CAN, or the robot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from synthetic.calibration.common import (  # noqa: E402
    CALIBRATION_TYPE,
    FORMAT_VERSION,
    CalibrationError,
    load_source_frame,
    parse_xy,
    project_points,
    read_json,
    require_keys,
    unpack_correspondences,
)


WINDOW_NAME = "Synthetic calibration validation"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Draw the calibrated board grid and convert image pixels to "
            "board coordinates. No robot connection is used."
        )
    )
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pixel",
        type=parse_xy,
        action="append",
        default=[],
        help="Pixel x,y to query; may be repeated",
    )
    parser.add_argument(
        "--grid-divisions",
        type=int,
        default=10,
        help="Board grid divisions per axis (default: 10)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open an OpenCV window and print board coordinates on click",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def load_calibration(path: Path) -> dict[str, Any]:
    calibration = read_json(path)
    require_keys(
        calibration,
        [
            "format_version",
            "type",
            "status",
            "source",
            "board",
            "correspondences",
            "image_to_board_homography",
            "board_to_image_homography",
        ],
        context="calibration",
    )
    if calibration["format_version"] != FORMAT_VERSION:
        raise CalibrationError(
            f"unsupported format_version: {calibration['format_version']}"
        )
    if calibration["type"] != CALIBRATION_TYPE:
        raise CalibrationError(
            f"expected type={CALIBRATION_TYPE!r}, got {calibration['type']!r}"
        )
    return calibration


def read_calibration_source(calibration: dict[str, Any]) -> np.ndarray:
    source = calibration["source"]
    if not isinstance(source, dict):
        raise CalibrationError("source must be an object")
    require_keys(source, ["kind", "path", "frame_index"], context="source")
    kind = source["kind"]
    if kind == "image":
        frame, _ = load_source_frame(
            image_path=Path(source["path"]),
            video_path=None,
            frame_index=0,
        )
    elif kind == "video":
        frame, _ = load_source_frame(
            image_path=None,
            video_path=Path(source["path"]),
            frame_index=int(source["frame_index"]),
        )
    else:
        raise CalibrationError(f"unsupported source kind: {kind!r}")
    return frame


def draw_grid(
    frame: np.ndarray,
    calibration: dict[str, Any],
    divisions: int,
) -> np.ndarray:
    if divisions < 1 or divisions > 100:
        raise CalibrationError("--grid-divisions must be between 1 and 100")
    board = calibration["board"]
    if not isinstance(board, dict):
        raise CalibrationError("board must be an object")
    require_keys(board, ["width", "height", "unit"], context="board")
    width = float(board["width"])
    height = float(board["height"])
    board_to_image = np.asarray(
        calibration["board_to_image_homography"],
        dtype=np.float64,
    )
    canvas = frame.copy()

    for index in range(divisions + 1):
        x = width * index / divisions
        vertical = project_points([[x, 0.0], [x, height]], board_to_image)
        p0, p1 = np.rint(vertical).astype(int)
        cv2.line(canvas, tuple(p0), tuple(p1), (60, 200, 60), 1, cv2.LINE_AA)

        y = height * index / divisions
        horizontal = project_points([[0.0, y], [width, y]], board_to_image)
        p0, p1 = np.rint(horizontal).astype(int)
        cv2.line(canvas, tuple(p0), tuple(p1), (60, 200, 60), 1, cv2.LINE_AA)

    image_points, _ = unpack_correspondences(calibration)
    for record, point in zip(
        calibration["correspondences"],
        image_points,
        strict=True,
    ):
        location = tuple(np.rint(point).astype(int))
        cv2.circle(canvas, location, 7, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            record["name"],
            (location[0] + 8, location[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    banner = (
        f"status={calibration['status']} | board={width:g}x{height:g} "
        f"{board['unit']} | grid={divisions}"
    )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 38), (20, 20, 20), -1)
    cv2.putText(
        canvas,
        banner,
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def query_pixels(
    pixels: list[tuple[float, float]],
    image_to_board: np.ndarray,
    *,
    board_unit: str,
) -> list[tuple[float, float]]:
    if not pixels:
        return []
    board_points = project_points(pixels, image_to_board)
    for pixel, board in zip(pixels, board_points, strict=True):
        print(
            f"[QUERY] image_px=({pixel[0]:.3f}, {pixel[1]:.3f}) "
            f"-> board_xy=({board[0]:.6f}, {board[1]:.6f}) {board_unit}"
        )
    return [tuple(point) for point in board_points]


def interactive_query(
    overlay: np.ndarray,
    image_to_board: np.ndarray,
    board_unit: str,
) -> None:
    canvas = overlay.copy()

    def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        nonlocal canvas
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        board = project_points([[x, y]], image_to_board)[0]
        print(
            f"[CLICK] image_px=({x}, {y}) "
            f"-> board_xy=({board[0]:.6f}, {board[1]:.6f}) {board_unit}"
        )
        canvas = overlay.copy()
        cv2.circle(canvas, (x, y), 7, (255, 0, 255), -1, cv2.LINE_AA)
        text = f"({board[0]:.3f}, {board[1]:.3f}) {board_unit}"
        cv2.putText(
            canvas,
            text,
            (x + 10, max(y - 10, 52)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    try:
        while True:
            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in {27, ord("q")}:
                return
    finally:
        cv2.destroyWindow(WINDOW_NAME)


def main() -> int:
    args = build_parser().parse_args()
    try:
        calibration_path = args.calibration.expanduser().resolve()
        calibration = load_calibration(calibration_path)
        frame = read_calibration_source(calibration)
        overlay = draw_grid(frame, calibration, args.grid_divisions)
        image_to_board = np.asarray(
            calibration["image_to_board_homography"],
            dtype=np.float64,
        )
        board_unit = str(calibration["board"]["unit"])
        query_pixels(args.pixel, image_to_board, board_unit=board_unit)

        output = args.output.expanduser().resolve()
        if output.exists() and not args.overwrite:
            raise CalibrationError(
                f"output already exists: {output}; pass --overwrite to replace it"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), overlay):
            raise CalibrationError(f"could not write overlay: {output}")
        print(f"[OK] overlay: {output}")
        print(f"[STATUS] {calibration['status']}; visualization is not robot validation")

        if args.interactive:
            interactive_query(overlay, image_to_board, board_unit)
        return 0
    except CalibrationError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

