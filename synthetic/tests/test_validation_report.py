#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

from synthetic.kinematics.piper_ik import IKSolution, joint_bounds_rad
from synthetic.preprocessing.profiles import CropRegion, ImageProfile, ResizeSpec, SourceShape
from synthetic.preview.validation import (
    build_config_status_summary,
    build_validation_report,
    check_frame_to_frame_joint_change,
    check_global_action_range,
    check_image_pipeline_consistency,
    check_ik_residual,
    check_joint_limits,
    check_segment_continuity,
)
from synthetic.trajectory.compose import BoardMotionConfig, build_descend_erase_lift
from synthetic.trajectory.schema import FullTrajectory
from synthetic.trajectory.shapes import fixed_circle_path
from synthetic.transforms.board_base import RigidTransform

_TRANSLATE_HOMOGRAPHY = np.asarray([[1.0, 0.0, 100.0], [0.0, 1.0, 100.0], [0.0, 0.0, 1.0]])


def _full_source_profile() -> ImageProfile:
    return ImageProfile(
        name="test_full",
        source=SourceShape(width=1000, height=1000),
        resize=ResizeSpec(mode="stretch", width=500, height=500),
    )


def _cropped_profile() -> ImageProfile:
    return ImageProfile(
        name="test_cropped",
        source=SourceShape(width=1000, height=1000),
        crop=CropRegion(x=50, y=0, width=100, height=1000),
        resize=ResizeSpec(mode="stretch", width=100, height=1000),
    )


class SegmentContinuityCheckTest(unittest.TestCase):
    def test_passes_for_a_valid_trajectory(self) -> None:
        transform = RigidTransform(rotation=np.eye(3), translation=np.zeros(3))
        config = BoardMotionConfig(
            hover_height_mm=10.0,
            contact_height_mm=0.0,
            tool_rpy_deg=(180.0, 0.0, 0.0),
            fps=30.0,
            transfer_speed_mm_per_s=100.0,
            descend_lift_speed_mm_per_s=20.0,
            erase_speed_mm_per_s=30.0,
        )
        path = fixed_circle_path([0.0, 0.0], radius_mm=10.0, num_points=16)
        descend, erase, lift = build_descend_erase_lift(path, board_to_base=transform, config=config)
        trajectory = FullTrajectory(segments=(descend, erase, lift), seed=1)
        result = check_segment_continuity(trajectory)
        self.assertEqual(result["status"], "pass")


class ImagePipelineConsistencyCheckTest(unittest.TestCase):
    def test_points_inside_raw_and_model_bounds_pass(self) -> None:
        board_xy = np.asarray([[0.0, 0.0], [10.0, 10.0], [20.0, 0.0]])
        result = check_image_pipeline_consistency(
            board_xy,
            board_to_image_homography=_TRANSLATE_HOMOGRAPHY,
            image_profile=_full_source_profile(),
        )
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["details"]["out_of_raw_image_bounds"], 0)
        self.assertEqual(result["details"]["out_of_model_crop"], 0)

    def test_point_outside_raw_image_is_flagged(self) -> None:
        board_xy = np.asarray([[-200.0, -200.0], [10.0, 10.0], [20.0, 0.0]])
        result = check_image_pipeline_consistency(
            board_xy,
            board_to_image_homography=_TRANSLATE_HOMOGRAPHY,
            image_profile=_full_source_profile(),
        )
        self.assertEqual(result["status"], "fail")
        self.assertGreaterEqual(result["details"]["out_of_raw_image_bounds"], 1)

    def test_point_outside_model_crop_is_flagged(self) -> None:
        # image_px = board_xy + 100; board_xy=[200,0] -> image_px=[300,100],
        # inside raw bounds (0..1000) but outside the crop x in [50,150).
        board_xy = np.asarray([[0.0, 0.0], [200.0, 0.0]])
        result = check_image_pipeline_consistency(
            board_xy,
            board_to_image_homography=_TRANSLATE_HOMOGRAPHY,
            image_profile=_cropped_profile(),
        )
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["details"]["out_of_raw_image_bounds"], 0)
        self.assertGreaterEqual(result["details"]["out_of_model_crop"], 1)


class IkResidualCheckTest(unittest.TestCase):
    def _solution(self, position_error: float, angle_error: float) -> IKSolution:
        return IKSolution(
            joint_rad=np.zeros(6),
            position_error_mm=position_error,
            angle_error_deg=angle_error,
            seed_index=0,
            joint_distance_from_primary_seed_rad=0.0,
        )

    def test_within_tolerance_passes(self) -> None:
        solutions = [self._solution(0.1, 0.1), self._solution(0.5, 0.3)]
        result = check_ik_residual(solutions, position_tol_mm=1.0, angle_tol_deg=1.0)
        self.assertEqual(result["status"], "pass")

    def test_exceeding_tolerance_fails(self) -> None:
        solutions = [self._solution(0.1, 0.1), self._solution(5.0, 0.3)]
        result = check_ik_residual(solutions, position_tol_mm=1.0, angle_tol_deg=1.0)
        self.assertEqual(result["status"], "fail")

    def test_empty_solutions_fails(self) -> None:
        result = check_ik_residual([], position_tol_mm=1.0, angle_tol_deg=1.0)
        self.assertEqual(result["status"], "fail")


class JointLimitCheckTest(unittest.TestCase):
    def test_within_bounds_passes(self) -> None:
        bounds = joint_bounds_rad()
        midpoint = (bounds[:, 0] + bounds[:, 1]) / 2.0
        result = check_joint_limits([midpoint.tolist()])
        self.assertEqual(result["status"], "pass")

    def test_out_of_bounds_fails(self) -> None:
        bounds = joint_bounds_rad()
        violating = bounds[:, 1] + 1.0
        result = check_joint_limits([violating.tolist()])
        self.assertEqual(result["status"], "fail")
        self.assertGreaterEqual(result["details"]["violations"], 1)


class FrameToFrameJointChangeCheckTest(unittest.TestCase):
    def test_small_steps_pass(self) -> None:
        sequence = np.linspace(np.zeros(6), np.full(6, 0.05), 10)
        result = check_frame_to_frame_joint_change(sequence, max_step_rad=0.02)
        self.assertEqual(result["status"], "pass")

    def test_large_step_fails_and_reports_worst_frame(self) -> None:
        sequence = np.zeros((3, 6))
        sequence[2] = 1.0  # a large jump between frame 1 and 2
        result = check_frame_to_frame_joint_change(sequence, max_step_rad=0.1)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["details"]["worst_frame_index"], 2)

    def test_single_frame_passes_trivially(self) -> None:
        result = check_frame_to_frame_joint_change([[0.0] * 6], max_step_rad=0.01)
        self.assertEqual(result["status"], "pass")


class GlobalActionRangeCheckTest(unittest.TestCase):
    def test_in_range_action_passes(self) -> None:
        action = np.zeros((4, 7))
        action[:, :6] = 50.0
        action[:, 6] = 50.0
        result = check_global_action_range(action)
        self.assertEqual(result["status"], "pass")

    def test_out_of_range_joint_column_fails(self) -> None:
        action = np.zeros((2, 7))
        action[0, 0] = 150.0
        result = check_global_action_range(action)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["details"]["joint_values_out_of_range"], 1)

    def test_wrong_shape_fails(self) -> None:
        result = check_global_action_range(np.zeros((3, 6)))
        self.assertEqual(result["status"], "fail")


class ConfigStatusSummaryTest(unittest.TestCase):
    def test_any_unverified_input_blocks_execution(self) -> None:
        summary = build_config_status_summary({"a": "unverified", "b": "verified"})
        self.assertFalse(summary["real_execution_allowed"])
        self.assertEqual(summary["unverified_inputs"], ["a"])

    def test_all_verified_allows_execution(self) -> None:
        summary = build_config_status_summary({"a": "verified", "b": "verified"})
        self.assertTrue(summary["real_execution_allowed"])
        self.assertEqual(summary["unverified_inputs"], [])


class BuildValidationReportTest(unittest.TestCase):
    def test_report_is_always_unverified_and_blocks_execution(self) -> None:
        transform = RigidTransform(rotation=np.eye(3), translation=np.zeros(3))
        config = BoardMotionConfig(
            hover_height_mm=10.0,
            contact_height_mm=0.0,
            tool_rpy_deg=(180.0, 0.0, 0.0),
            fps=30.0,
            transfer_speed_mm_per_s=100.0,
            descend_lift_speed_mm_per_s=20.0,
            erase_speed_mm_per_s=30.0,
        )
        path = fixed_circle_path([0.0, 0.0], radius_mm=10.0, num_points=16)
        descend, erase, lift = build_descend_erase_lift(path, board_to_base=transform, config=config)
        trajectory = FullTrajectory(segments=(descend, erase, lift), seed=1)
        n_frames = trajectory.concatenated_base_xyzrpy().shape[0]

        joint_rad_sequence = np.zeros((n_frames, 6))
        action = np.zeros((n_frames, 7))
        solutions = [
            IKSolution(
                joint_rad=np.zeros(6),
                position_error_mm=0.1,
                angle_error_deg=0.1,
                seed_index=0,
                joint_distance_from_primary_seed_rad=0.0,
            )
            for _ in range(n_frames)
        ]

        report = build_validation_report(
            trajectory=trajectory,
            board_xy_path=erase.board_xy,
            board_to_image_homography=_TRANSLATE_HOMOGRAPHY,
            image_profile=_full_source_profile(),
            ik_solutions=solutions,
            joint_rad_sequence=joint_rad_sequence,
            action=action,
            position_tol_mm=1.0,
            angle_tol_deg=1.0,
            max_joint_step_rad=1.0,
            status_by_input={"board_calibration": "unverified", "board_base_transform": "unverified"},
        )
        self.assertEqual(report["status"], "unverified")
        self.assertFalse(report["real_execution_allowed"])
        self.assertTrue(report["all_checks_passed"])
        self.assertEqual(len(report["checks"]), 6)


if __name__ == "__main__":
    unittest.main()
