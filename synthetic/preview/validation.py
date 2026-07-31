#!/usr/bin/env python3
"""Static, offline validation checks for a generated trajectory.

Every check returns a plain dict `{"name", "status", "details"}` with
`status` in `{"pass", "fail"}` instead of raising, so a broken trajectory
still produces a full report (with failures called out) rather than
crashing preview generation. `build_validation_report` aggregates them and
always marks generation-stage output as `status="unverified"` /
`real_execution_allowed=False`, since none of this project's calibration
inputs are hardware-verified yet.

This module is intentionally independent from ROS, CAN, LeRobot, and Piper
hardware.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from synthetic.calibration.common import project_points
from synthetic.kinematics.action_conversion import JOINT_ORDER
from synthetic.kinematics.piper_ik import IKSolution, joint_bounds_rad
from synthetic.preprocessing.image_transform import (
    is_inside_raw_bounds,
    raw_point_visibility,
    raw_to_model_points,
)
from synthetic.preprocessing.profiles import ImageProfile
from synthetic.trajectory.schema import FullTrajectory, TrajectoryError

_POSITION_ROUND_TRIP_TOL_PX = 1e-3


def _result(name: str, *, passed: bool, details: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "status": "pass" if passed else "fail", "details": details}


def check_segment_continuity(trajectory: FullTrajectory) -> dict[str, Any]:
    try:
        trajectory.validate()
    except TrajectoryError as exc:
        return _result("segment_continuity", passed=False, details={"error": str(exc)})
    return _result(
        "segment_continuity",
        passed=True,
        details={"segments": [segment.segment for segment in trajectory.segments]},
    )


def check_image_pipeline_consistency(
    board_xy_path: Any,
    *,
    board_to_image_homography: Any,
    image_profile: ImageProfile,
) -> dict[str, Any]:
    """`board_xy -> image_px -> model_px` for the erase path: round-trip and crop visibility.

    Covers both "raw/model/board point consistency" and "model-crop-exit
    detection" for this specific generated path.
    """

    board_xy = np.asarray(board_xy_path, dtype=np.float64)
    image_px = project_points(board_xy, board_to_image_homography)
    inside_raw = is_inside_raw_bounds(image_px, image_profile)
    visible_in_model = raw_point_visibility(image_px, image_profile)
    model_px = raw_to_model_points(image_px, image_profile, strict=False)

    out_of_raw_count = int((~inside_raw).sum())
    out_of_model_count = int((~visible_in_model).sum())
    passed = out_of_raw_count == 0 and out_of_model_count == 0

    return _result(
        "image_pipeline_consistency",
        passed=passed,
        details={
            "num_points": int(board_xy.shape[0]),
            "out_of_raw_image_bounds": out_of_raw_count,
            "out_of_model_crop": out_of_model_count,
            "image_px": image_px.tolist(),
            "model_px": model_px.tolist(),
            "visible_in_model": visible_in_model.tolist(),
        },
    )


def check_ik_residual(
    ik_solutions: list[IKSolution],
    *,
    position_tol_mm: float,
    angle_tol_deg: float,
) -> dict[str, Any]:
    if not ik_solutions:
        return _result("ik_residual", passed=False, details={"error": "no IK solutions given"})
    position_errors = [solution.position_error_mm for solution in ik_solutions]
    angle_errors = [solution.angle_error_deg for solution in ik_solutions]
    max_position = max(position_errors)
    max_angle = max(angle_errors)
    passed = max_position <= position_tol_mm and max_angle <= angle_tol_deg
    return _result(
        "ik_residual",
        passed=passed,
        details={
            "position_error_mm_mean": float(np.mean(position_errors)),
            "position_error_mm_max": float(max_position),
            "angle_error_deg_mean": float(np.mean(angle_errors)),
            "angle_error_deg_max": float(max_angle),
            "position_tol_mm": position_tol_mm,
            "angle_tol_deg": angle_tol_deg,
        },
    )


def check_joint_limits(joint_rad_sequence: Any, *, urdf_path: Any = None) -> dict[str, Any]:
    joints = np.asarray(joint_rad_sequence, dtype=np.float64)
    bounds = joint_bounds_rad(urdf_path)
    below = joints < bounds[None, :, 0]
    above = joints > bounds[None, :, 1]
    violations = int((below | above).sum())
    return _result(
        "joint_limits",
        passed=violations == 0,
        details={
            "violations": violations,
            "bounds_rad": bounds.tolist(),
            "min_per_joint_rad": joints.min(axis=0).tolist(),
            "max_per_joint_rad": joints.max(axis=0).tolist(),
        },
    )


def check_frame_to_frame_joint_change(
    joint_rad_sequence: Any,
    *,
    max_step_rad: float,
) -> dict[str, Any]:
    joints = np.asarray(joint_rad_sequence, dtype=np.float64)
    if joints.shape[0] < 2:
        return _result(
            "frame_to_frame_joint_change",
            passed=True,
            details={"max_step_rad": 0.0, "threshold_rad": max_step_rad},
        )
    steps = np.max(np.abs(np.diff(joints, axis=0)), axis=1)
    max_step = float(steps.max())
    return _result(
        "frame_to_frame_joint_change",
        passed=max_step <= max_step_rad,
        details={
            "max_step_rad": max_step,
            "mean_step_rad": float(steps.mean()),
            "threshold_rad": max_step_rad,
            "worst_frame_index": int(np.argmax(steps)) + 1,
        },
    )


def check_global_action_range(action: Any) -> dict[str, Any]:
    array = np.asarray(action, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 7:
        return _result(
            "global_action_range",
            passed=False,
            details={"error": f"action must have shape (N, 7), got {array.shape}"},
        )
    joint_columns = array[:, :6]
    gripper_column = array[:, 6]
    joint_out_of_range = int(((joint_columns < -100.0) | (joint_columns > 100.0)).sum())
    gripper_out_of_range = int(((gripper_column < 0.0) | (gripper_column > 100.0)).sum())
    per_column = {
        name: {"min": float(array[:, i].min()), "max": float(array[:, i].max())}
        for i, name in enumerate((*JOINT_ORDER, "gripper"))
    }
    return _result(
        "global_action_range",
        passed=(joint_out_of_range == 0 and gripper_out_of_range == 0),
        details={
            "joint_values_out_of_range": joint_out_of_range,
            "gripper_values_out_of_range": gripper_out_of_range,
            "per_column": per_column,
        },
    )


def build_config_status_summary(status_by_input: dict[str, str]) -> dict[str, Any]:
    """Aggregate `status` fields from every calibration/config input.

    `real_execution_allowed` is only ever True if every input is already
    `"verified"` -- currently none are, so this is always False in this
    offline stage.
    """

    unverified = {name: status for name, status in status_by_input.items() if status != "verified"}
    return {
        "status_by_input": status_by_input,
        "real_execution_allowed": len(unverified) == 0,
        "unverified_inputs": sorted(unverified),
    }


def build_validation_report(
    *,
    trajectory: FullTrajectory,
    board_xy_path: Any,
    board_to_image_homography: Any,
    image_profile: ImageProfile,
    ik_solutions: list[IKSolution],
    joint_rad_sequence: Any,
    action: Any,
    position_tol_mm: float,
    angle_tol_deg: float,
    max_joint_step_rad: float,
    status_by_input: dict[str, str],
    urdf_path: Any = None,
) -> dict[str, Any]:
    """Run every required 6E check and assemble one `validation_report.json` payload."""

    checks = [
        check_segment_continuity(trajectory),
        check_image_pipeline_consistency(
            board_xy_path,
            board_to_image_homography=board_to_image_homography,
            image_profile=image_profile,
        ),
        check_ik_residual(
            ik_solutions, position_tol_mm=position_tol_mm, angle_tol_deg=angle_tol_deg
        ),
        check_joint_limits(joint_rad_sequence, urdf_path=urdf_path),
        check_frame_to_frame_joint_change(joint_rad_sequence, max_step_rad=max_joint_step_rad),
        check_global_action_range(action),
    ]
    config_summary = build_config_status_summary(status_by_input)
    all_passed = all(check["status"] == "pass" for check in checks)

    return {
        "status": "unverified",
        "real_execution_allowed": False,
        "all_checks_passed": all_passed,
        "config_status": config_summary,
        "checks": checks,
    }
