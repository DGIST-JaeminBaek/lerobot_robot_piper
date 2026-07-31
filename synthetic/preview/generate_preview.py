#!/usr/bin/env python3
"""Generate one offline, hardware-free trajectory preview + validation report.

Ties together: image<->board calibration (`calibration/`), the VLA image
profile (`preprocessing/`), the board<->base rigid transform
(`transforms/`), a shape's erase path and Cartesian motion
(`trajectory/`), and IK + normalized-action conversion (`kinematics/`)
into one `synthetic/outputs/<trajectory_id>/` directory. Only the
`DESCEND`/`ERASE`/`LIFT_FROM_BOARD` segments are generated -- pick-up/return
still require a real recorded template (see `synthetic/trajectory/compose.py`).

Never connects to ROS, CAN, RViz, or a live camera.
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
    CalibrationError,
    load_source_frame,
    parse_xy,
    read_json,
    write_json,
)
from synthetic.kinematics.action_conversion import (  # noqa: E402
    ActionConversionError,
    build_normalized_action,
)
from synthetic.kinematics.piper_ik import (  # noqa: E402
    InverseKinematicsError,
    solve_ik_sequence,
)
from synthetic.preprocessing.profiles import (  # noqa: E402
    ImageProfile,
    load_profile,
    smolvla_v1_profile,
)
from synthetic.preview.overlays import (  # noqa: E402
    draw_model_input_overlay,
    draw_raw_board_overlay,
)
from synthetic.preview.plots import save_eef_plot, save_joint_plot  # noqa: E402
from synthetic.preview.validation import build_validation_report  # noqa: E402
from synthetic.transforms.board_base import RigidTransform, TransformError  # noqa: E402
from synthetic.trajectory.compose import (  # noqa: E402
    BoardMotionConfig,
    build_descend_erase_lift,
)
from synthetic.trajectory.schema import FullTrajectory, TrajectoryError  # noqa: E402
from synthetic.trajectory.shapes import (  # noqa: E402
    fixed_circle_path,
    fixed_rectangle_path,
    fixed_triangle_path,
)

BUILTIN_PROFILES = {"smolvla_v1": smolvla_v1_profile}


class PreviewError(ValueError):
    """Raised when preview generation input or intermediate data is invalid."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an offline preview (overlays, plots, NPZ, validation "
            "report) for one generated erase trajectory. No robot connection "
            "is used."
        )
    )
    parser.add_argument("--calibration", type=Path, required=True, help="image<->board calibration JSON")
    parser.add_argument("--board-base-transform", type=Path, required=True)
    parser.add_argument("--board-motion-config", type=Path, required=True)

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--video", type=Path)
    parser.add_argument("--frame", type=int, default=0)

    profile_source = parser.add_mutually_exclusive_group(required=True)
    profile_source.add_argument("--preprocessing-profile", type=Path)
    profile_source.add_argument("--preprocessing-profile-name", choices=sorted(BUILTIN_PROFILES))

    parser.add_argument("--shape", choices=("circle", "triangle", "rectangle"), required=True)
    parser.add_argument("--center", type=parse_xy, required=True, help="board_xy mm, e.g. 200,150")
    parser.add_argument("--radius-mm", type=float, help="circle radius / triangle circumradius, mm")
    parser.add_argument("--width-mm", type=float, help="rectangle width, mm")
    parser.add_argument("--height-mm", type=float, help="rectangle height, mm")
    parser.add_argument("--rotation-deg", type=float, default=0.0)
    parser.add_argument("--num-points", type=int, default=120)

    parser.add_argument(
        "--initial-seed-joint-rad",
        type=str,
        required=True,
        help="comma-separated 6 joint angles (radians), IK seed for the first frame",
    )
    parser.add_argument("--gripper-closed-fraction", type=float, default=1.0)
    parser.add_argument("--position-tol-mm", type=float, default=1.0)
    parser.add_argument("--angle-tol-deg", type=float, default=1.0)
    parser.add_argument("--max-joint-step-rad", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _parse_joint_rad(text: str) -> np.ndarray:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 6:
        raise PreviewError(f"--initial-seed-joint-rad must have 6 comma-separated values, got {text!r}")
    try:
        return np.asarray([float(part) for part in parts], dtype=np.float64)
    except ValueError as exc:
        raise PreviewError(f"--initial-seed-joint-rad must be numeric, got {text!r}") from exc


def _build_shape_path(args: argparse.Namespace) -> np.ndarray:
    center = np.asarray(args.center, dtype=np.float64)
    if args.shape == "circle":
        if args.radius_mm is None:
            raise PreviewError("--radius-mm is required for --shape circle")
        return fixed_circle_path(
            center, radius_mm=args.radius_mm, num_points=args.num_points, rotation_deg=args.rotation_deg
        )
    if args.shape == "triangle":
        if args.radius_mm is None:
            raise PreviewError("--radius-mm is required for --shape triangle (circumradius)")
        return fixed_triangle_path(
            center,
            circumradius_mm=args.radius_mm,
            num_points=args.num_points,
            rotation_deg=args.rotation_deg,
        )
    if args.width_mm is None or args.height_mm is None:
        raise PreviewError("--width-mm and --height-mm are required for --shape rectangle")
    return fixed_rectangle_path(
        center,
        width_mm=args.width_mm,
        height_mm=args.height_mm,
        num_points=args.num_points,
        rotation_deg=args.rotation_deg,
    )


def _load_profile(args: argparse.Namespace) -> ImageProfile:
    if args.preprocessing_profile is not None:
        return load_profile(args.preprocessing_profile)
    return BUILTIN_PROFILES[args.preprocessing_profile_name]()


def main() -> int:
    args = build_parser().parse_args()
    try:
        calibration_path = args.calibration.expanduser().resolve()
        calibration = read_json(calibration_path)
        board_to_image_homography = np.asarray(
            calibration["board_to_image_homography"], dtype=np.float64
        )

        transform_path = args.board_base_transform.expanduser().resolve()
        transform_payload = read_json(transform_path)
        board_to_base = RigidTransform.from_dict(transform_payload["transform_base_from_board"])

        config_path = args.board_motion_config.expanduser().resolve()
        motion_config = BoardMotionConfig.from_dict(read_json(config_path))

        image_profile = _load_profile(args)
        frame, source_info = load_source_frame(
            image_path=args.image, video_path=args.video, frame_index=args.frame
        )

        board_xy_path = _build_shape_path(args)
        descend, erase, lift = build_descend_erase_lift(
            board_xy_path, board_to_base=board_to_base, config=motion_config
        )
        trajectory = FullTrajectory(segments=(descend, erase, lift), seed=args.seed)
        trajectory.validate()

        base_xyzrpy = trajectory.concatenated_base_xyzrpy()
        gripper_fraction = trajectory.concatenated_gripper()

        initial_seed = _parse_joint_rad(args.initial_seed_joint_rad)
        joint_rad_sequence, ik_solutions = solve_ik_sequence(
            base_xyzrpy,
            initial_seed_rad=initial_seed,
            position_tol_mm=args.position_tol_mm,
            angle_tol_deg=args.angle_tol_deg,
            max_joint_step_rad=args.max_joint_step_rad,
        )
        action = build_normalized_action(joint_rad_sequence, gripper_fraction)

        report = build_validation_report(
            trajectory=trajectory,
            board_xy_path=erase.board_xy,
            board_to_image_homography=board_to_image_homography,
            image_profile=image_profile,
            ik_solutions=ik_solutions,
            joint_rad_sequence=joint_rad_sequence,
            action=action,
            position_tol_mm=args.position_tol_mm,
            angle_tol_deg=args.angle_tol_deg,
            max_joint_step_rad=args.max_joint_step_rad,
            status_by_input={
                "board_calibration": str(calibration.get("status", "unknown")),
                "board_base_transform": str(transform_payload.get("status", "unknown")),
                "board_motion_config": motion_config.status,
            },
        )

        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        request = {
            "shape": args.shape,
            "center_board_xy_mm": list(args.center),
            "radius_mm": args.radius_mm,
            "width_mm": args.width_mm,
            "height_mm": args.height_mm,
            "rotation_deg": args.rotation_deg,
            "num_points": args.num_points,
            "seed": args.seed,
            "initial_seed_joint_rad": initial_seed.tolist(),
            "gripper_closed_fraction": args.gripper_closed_fraction,
            "position_tol_mm": args.position_tol_mm,
            "angle_tol_deg": args.angle_tol_deg,
            "max_joint_step_rad": args.max_joint_step_rad,
            "source": source_info,
        }
        write_json(output_dir / "request.json", request, overwrite=args.overwrite)
        write_json(
            output_dir / "preprocessing_profile.json", image_profile.to_dict(), overwrite=args.overwrite
        )
        write_json(
            output_dir / "calibration_snapshot.json",
            {
                "board_calibration": calibration,
                "board_base_transform": transform_payload,
                "board_motion_config": motion_config.to_dict(),
            },
            overwrite=args.overwrite,
        )
        write_json(output_dir / "validation_report.json", report, overwrite=args.overwrite)

        np.savez(
            output_dir / "cartesian_trajectory.npz",
            frame_index=trajectory.concatenated_frame_index(),
            time_s=trajectory.concatenated_time_s(),
            base_xyzrpy=base_xyzrpy,
            gripper_closed_fraction=gripper_fraction,
            erase_board_xy=erase.board_xy,
            segment_names=np.asarray([segment.segment for segment in trajectory.segments]),
            segment_lengths=np.asarray([segment.base_xyzrpy.shape[0] for segment in trajectory.segments]),
        )
        np.savez(
            output_dir / "joint_actions.npz",
            joint_rad=joint_rad_sequence,
            action_normalized=action,
        )

        raw_overlay = draw_raw_board_overlay(
            frame, erase.board_xy, board_to_image_homography=board_to_image_homography
        )
        model_overlay = draw_model_input_overlay(
            frame,
            erase.board_xy,
            board_to_image_homography=board_to_image_homography,
            image_profile=image_profile,
        )
        cv2.imwrite(str(output_dir / "raw_board_overlay.png"), raw_overlay)
        cv2.imwrite(str(output_dir / "model_input_overlay.png"), model_overlay)

        save_eef_plot(base_xyzrpy, output_dir / "eef_plot.png")
        save_joint_plot(joint_rad_sequence, action, output_dir / "joint_plot.png")

        print(f"[OK] preview written to: {output_dir}")
        print(
            f"[INFO] all_checks_passed={report['all_checks_passed']} "
            f"real_execution_allowed={report['real_execution_allowed']}"
        )
        print("[STATUS] unverified; not permitted for real robot execution")
        return 0
    except (
        PreviewError,
        CalibrationError,
        TransformError,
        TrajectoryError,
        InverseKinematicsError,
        ActionConversionError,
    ) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
