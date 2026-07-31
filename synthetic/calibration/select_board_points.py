#!/usr/bin/env python3
"""Select the board corners from an image or one video frame.

The points must be selected in this order:
top-left, top-right, bottom-right, bottom-left.

This tool only writes point correspondences. It never connects to ROS, CAN, or
the robot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from synthetic.calibration.common import (  # noqa: E402
    FORMAT_VERSION,
    POINT_NAMES,
    POINT_SELECTION_TYPE,
    BoardSpec,
    CalibrationError,
    make_correspondences,
    load_source_frame,
    parse_point_list,
    validate_image_quad,
    write_json,
)


WINDOW_NAME = "Synthetic board calibration"
COLORS = (
    (0, 255, 255),
    (0, 255, 0),
    (255, 255, 0),
    (0, 128, 255),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select board corners in TL, TR, BR, BL order and save pixel/board "
            "correspondences. No robot connection is used."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Input image")
    source.add_argument("--video", type=Path, help="Input video")
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Zero-based video frame index (default: 0)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--unit",
        choices=["normalized", "mm"],
        default="normalized",
        help="Board coordinate unit (default: normalized)",
    )
    parser.add_argument(
        "--board-width",
        type=float,
        default=None,
        help="Physical board width; required for --unit mm",
    )
    parser.add_argument(
        "--board-height",
        type=float,
        default=None,
        help="Physical board height; required for --unit mm",
    )
    parser.add_argument(
        "--points",
        help=(
            "Non-interactive image points as "
            "'TL_x,TL_y;TR_x,TR_y;BR_x,BR_y;BL_x,BL_y'"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def resolve_board(args: argparse.Namespace) -> BoardSpec:
    if args.unit == "normalized":
        if args.board_width is not None or args.board_height is not None:
            raise CalibrationError(
                "--board-width/--board-height are only used with --unit mm"
            )
        board = BoardSpec(width=1.0, height=1.0, unit="normalized")
    else:
        if args.board_width is None or args.board_height is None:
            raise CalibrationError(
                "--unit mm requires both --board-width and --board-height"
            )
        board = BoardSpec(
            width=float(args.board_width),
            height=float(args.board_height),
            unit="mm",
        )
    board.validate()
    return board


def fit_for_display(
    frame: np.ndarray,
    max_width: int = 1440,
    max_height: int = 900,
) -> tuple[np.ndarray, float]:
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale == 1.0:
        return frame.copy(), scale
    resized = cv2.resize(
        frame,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def draw_selection(
    display_base: np.ndarray,
    points_original: list[tuple[float, float]],
    scale: float,
) -> np.ndarray:
    canvas = display_base.copy()
    display_points = [
        (int(round(x * scale)), int(round(y * scale)))
        for x, y in points_original
    ]
    if len(display_points) >= 2:
        cv2.polylines(
            canvas,
            [np.asarray(display_points, dtype=np.int32)],
            isClosed=len(display_points) == 4,
            color=(255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
    for index, point in enumerate(display_points):
        cv2.circle(canvas, point, 7, COLORS[index], -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            POINT_NAMES[index],
            (point[0] + 10, point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLORS[index],
            2,
            cv2.LINE_AA,
        )

    next_name = POINT_NAMES[len(points_original)] if len(points_original) < 4 else "done"
    help_text = (
        f"next={next_name} | click: add | u: undo | r: reset | "
        "Enter/s: save | Esc/q: cancel"
    )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 38), (20, 20, 20), -1)
    cv2.putText(
        canvas,
        help_text,
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def interactive_select(frame: np.ndarray) -> np.ndarray:
    display_base, scale = fit_for_display(frame)
    points: list[tuple[float, float]] = []

    def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            original_x = min(max(x / scale, 0.0), frame.shape[1] - 1.0)
            original_y = min(max(y / scale, 0.0), frame.shape[0] - 1.0)
            points.append((original_x, original_y))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    try:
        while True:
            cv2.imshow(WINDOW_NAME, draw_selection(display_base, points, scale))
            key = cv2.waitKey(20) & 0xFF
            if key in {27, ord("q")}:
                raise CalibrationError("selection cancelled; no file was written")
            if key == ord("u") and points:
                points.pop()
            elif key == ord("r"):
                points.clear()
            elif key in {10, 13, ord("s")}:
                if len(points) != 4:
                    print(
                        f"[WAIT] four points are required; currently {len(points)}",
                        file=sys.stderr,
                    )
                    continue
                return np.asarray(points, dtype=np.float64)
    finally:
        cv2.destroyWindow(WINDOW_NAME)


def main() -> int:
    args = build_parser().parse_args()
    try:
        board = resolve_board(args)
        frame, source = load_source_frame(
            image_path=args.image,
            video_path=args.video,
            frame_index=args.frame,
        )
        image_points = (
            parse_point_list(args.points)
            if args.points is not None
            else interactive_select(frame)
        )
        image_points = validate_image_quad(
            image_points,
            width=source["width"],
            height=source["height"],
        )
        payload = {
            "format_version": FORMAT_VERSION,
            "type": POINT_SELECTION_TYPE,
            "status": "unverified",
            "source": source,
            "board": {
                "unit": board.unit,
                "width": board.width,
                "height": board.height,
                "origin": "top_left",
                "x_direction": "top_left_to_top_right",
                "y_direction": "top_left_to_bottom_left",
            },
            "point_order": list(POINT_NAMES),
            "correspondences": make_correspondences(image_points, board),
        }
        output = args.output.expanduser().resolve()
        write_json(output, payload, overwrite=args.overwrite)
        print(f"[OK] point selection: {output}")
        print(f"[INFO] source={source['path']}")
        print(f"[INFO] board={board.width}x{board.height} {board.unit}")
        print("[NEXT] solve_homography.py --points <this file> --output <calibration.json>")
        return 0
    except CalibrationError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

