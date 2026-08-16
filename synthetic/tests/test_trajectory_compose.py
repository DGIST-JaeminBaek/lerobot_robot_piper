#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

from synthetic.transforms.board_base import RigidTransform
from synthetic.trajectory.compose import (
    BoardMotionConfig,
    assert_hover_clearance,
    board_path_to_base_poses,
    board_point_to_base_pose,
    build_descend_erase_lift,
    build_finish_segment,
    build_open_gripper_segment,
    build_transfer_segment,
    compose_full_trajectory,
    require_recorded_template,
)
from synthetic.trajectory.schema import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    SEGMENT_ORDER,
    FullTrajectory,
    SegmentTrajectory,
    TrajectoryError,
)
from synthetic.trajectory.shapes import fixed_circle_path
from synthetic.trajectory.timing import frame_and_time


def _identity_transform() -> RigidTransform:
    return RigidTransform(rotation=np.eye(3), translation=np.zeros(3))


def _default_config(**overrides) -> BoardMotionConfig:
    values = dict(
        hover_height_mm=50.0,
        contact_height_mm=0.0,
        tool_rpy_deg=(180.0, 0.0, 0.0),
        fps=30.0,
        transfer_speed_mm_per_s=200.0,
        descend_lift_speed_mm_per_s=50.0,
        erase_speed_mm_per_s=80.0,
    )
    values.update(overrides)
    return BoardMotionConfig(**values)


def _constant_segment(segment: str, pose, *, gripper_value: float, num_samples: int = 3) -> SegmentTrajectory:
    frame_index, time_s = frame_and_time(num_samples, 30.0)
    pose_array = np.asarray(pose, dtype=np.float64)
    return SegmentTrajectory(
        segment=segment,
        frame_index=frame_index,
        time_s=time_s,
        base_xyzrpy=np.tile(pose_array, (num_samples, 1)),
        gripper=np.full(num_samples, gripper_value),
        fps=30.0,
    )


class BoardMotionConfigTest(unittest.TestCase):
    def test_hover_must_exceed_contact(self) -> None:
        with self.assertRaisesRegex(TrajectoryError, "hover_height_mm"):
            _default_config(hover_height_mm=0.0, contact_height_mm=10.0).validate()

    def test_non_positive_speed_rejected(self) -> None:
        with self.assertRaisesRegex(TrajectoryError, "transfer_speed_mm_per_s"):
            _default_config(transfer_speed_mm_per_s=0.0).validate()

    def test_json_round_trip(self) -> None:
        config = _default_config()
        loaded = BoardMotionConfig.from_dict(config.to_dict())
        self.assertEqual(loaded, config)


class BoardPointToBasePoseTest(unittest.TestCase):
    def test_identity_transform_places_height_on_board_z(self) -> None:
        transform = _identity_transform()
        pose = board_point_to_base_pose(
            [10.0, 20.0], 5.0, board_to_base=transform, tool_rpy_deg=(180.0, 0.0, 0.0)
        )
        np.testing.assert_allclose(pose, [10.0, 20.0, 5.0, 180.0, 0.0, 0.0])

    def test_path_variant_matches_single_point_variant(self) -> None:
        transform = _identity_transform()
        path = np.asarray([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
        poses = board_path_to_base_poses(
            path, 7.0, board_to_base=transform, tool_rpy_deg=(180.0, 0.0, 0.0)
        )
        for xy, pose in zip(path, poses, strict=True):
            expected = board_point_to_base_pose(
                xy, 7.0, board_to_base=transform, tool_rpy_deg=(180.0, 0.0, 0.0)
            )
            np.testing.assert_allclose(pose, expected)


class DescendEraseLiftTest(unittest.TestCase):
    def setUp(self) -> None:
        self.transform = _identity_transform()
        self.config = _default_config()
        # A fine polyline keeps the chordal (straight-segment) approximation
        # error from re-sampling by arc length well under the test tolerance.
        self.path = fixed_circle_path([100.0, 80.0], radius_mm=30.0, num_points=360)

    def test_segments_are_continuous_with_each_other(self) -> None:
        descend, erase, lift = build_descend_erase_lift(
            self.path, board_to_base=self.transform, config=self.config
        )
        np.testing.assert_allclose(descend.end_pose(), erase.start_pose(), atol=1e-9)
        np.testing.assert_allclose(erase.end_pose(), lift.start_pose(), atol=1e-9)

    def test_descend_goes_from_hover_to_contact_height(self) -> None:
        descend, _erase, _lift = build_descend_erase_lift(
            self.path, board_to_base=self.transform, config=self.config
        )
        self.assertAlmostEqual(descend.start_pose()[2], self.config.hover_height_mm)
        self.assertAlmostEqual(descend.end_pose()[2], self.config.contact_height_mm)

    def test_erase_stays_at_contact_height_and_follows_the_shape(self) -> None:
        _descend, erase, _lift = build_descend_erase_lift(
            self.path, board_to_base=self.transform, config=self.config
        )
        np.testing.assert_allclose(
            erase.base_xyzrpy[:, 2], self.config.contact_height_mm, atol=1e-9
        )
        distances = np.linalg.norm(
            erase.base_xyzrpy[:, :2] - np.asarray([100.0, 80.0]), axis=1
        )
        # Re-sampling by arc length interpolates along straight chords of the
        # input polyline, so a small chordal (sagitta) deviation from the
        # true radius is expected; it shrinks with the input path's density.
        np.testing.assert_allclose(distances, 30.0, atol=2e-3)

    def test_gripper_stays_closed_throughout(self) -> None:
        descend, erase, lift = build_descend_erase_lift(
            self.path, board_to_base=self.transform, config=self.config
        )
        for segment in (descend, erase, lift):
            np.testing.assert_array_equal(
                segment.gripper, np.full(segment.gripper.shape, GRIPPER_CLOSED)
            )

    def test_erase_dips_below_hover_but_transfer_does_not(self) -> None:
        _descend, erase, _lift = build_descend_erase_lift(
            self.path, board_to_base=self.transform, config=self.config
        )
        with self.assertRaisesRegex(TrajectoryError, "dips below hover"):
            assert_hover_clearance(
                erase,
                board_to_base=self.transform,
                hover_height_mm=self.config.hover_height_mm,
            )

        hover_pose = board_point_to_base_pose(
            self.path[0],
            self.config.hover_height_mm,
            board_to_base=self.transform,
            tool_rpy_deg=self.config.tool_rpy_deg,
        )
        transfer = build_transfer_segment(
            "TRANSFER_ABOVE_BOARD", hover_pose, hover_pose, config=self.config
        )
        assert_hover_clearance(
            transfer,
            board_to_base=self.transform,
            hover_height_mm=self.config.hover_height_mm,
        )


class TransferSegmentTest(unittest.TestCase):
    def test_transfer_reaches_start_and_end_exactly(self) -> None:
        config = _default_config()
        start = [0.0, 0.0, 50.0, 180.0, 0.0, 0.0]
        end = [200.0, 0.0, 50.0, 180.0, 0.0, 0.0]
        segment = build_transfer_segment(
            "TRANSFER_ABOVE_BOARD", start, end, config=config
        )
        np.testing.assert_allclose(segment.start_pose(), start)
        np.testing.assert_allclose(segment.end_pose(), end)

    def test_more_samples_for_longer_distance(self) -> None:
        config = _default_config()
        short = build_transfer_segment(
            "TRANSFER_ABOVE_BOARD",
            [0.0, 0.0, 50.0, 0.0, 0.0, 0.0],
            [10.0, 0.0, 50.0, 0.0, 0.0, 0.0],
            config=config,
        )
        long = build_transfer_segment(
            "TRANSFER_ABOVE_BOARD",
            [0.0, 0.0, 50.0, 0.0, 0.0, 0.0],
            [500.0, 0.0, 50.0, 0.0, 0.0, 0.0],
            config=config,
        )
        self.assertLess(short.base_xyzrpy.shape[0], long.base_xyzrpy.shape[0])


class OpenGripperAndFinishTest(unittest.TestCase):
    def test_open_gripper_ramps_from_closed_to_open(self) -> None:
        pose = [0.0, 0.0, 50.0, 180.0, 0.0, 0.0]
        segment = build_open_gripper_segment(pose, fps=30.0, duration_s=0.5)
        self.assertAlmostEqual(float(segment.gripper[0]), GRIPPER_CLOSED)
        self.assertAlmostEqual(float(segment.gripper[-1]), GRIPPER_OPEN)
        np.testing.assert_allclose(segment.base_xyzrpy, np.tile(pose, (segment.gripper.shape[0], 1)))

    def test_finish_is_a_single_open_gripper_frame(self) -> None:
        pose = [0.0, 0.0, 50.0, 180.0, 0.0, 0.0]
        segment = build_finish_segment(pose, fps=30.0)
        self.assertEqual(segment.base_xyzrpy.shape[0], 1)
        self.assertAlmostEqual(float(segment.gripper[0]), GRIPPER_OPEN)


class RequireRecordedTemplateTest(unittest.TestCase):
    def test_none_template_is_rejected(self) -> None:
        with self.assertRaisesRegex(TrajectoryError, "recorded human-teleop"):
            require_recorded_template("GRASP", None)

    def test_non_template_segment_name_is_rejected(self) -> None:
        template = _constant_segment("ERASE", [0, 0, 0, 0, 0, 0], gripper_value=1.0)
        with self.assertRaisesRegex(TrajectoryError, "not a template-required segment"):
            require_recorded_template("ERASE", template)

    def test_mismatched_segment_name_is_rejected(self) -> None:
        template = _constant_segment("GRASP", [0, 0, 0, 0, 0, 0], gripper_value=1.0)
        with self.assertRaisesRegex(TrajectoryError, "does not match requested"):
            require_recorded_template("LIFT", template)

    def test_valid_template_passes_through(self) -> None:
        template = _constant_segment("GRASP", [0, 0, 0, 0, 0, 0], gripper_value=1.0)
        returned = require_recorded_template("GRASP", template)
        self.assertIs(returned, template)


class ComposeFullTrajectoryTest(unittest.TestCase):
    def _build_full_segments(self) -> dict[str, SegmentTrajectory]:
        transform = _identity_transform()
        config = _default_config()
        path = fixed_circle_path([100.0, 80.0], radius_mm=20.0, num_points=24)

        parking_pose = [0.0, 0.0, 200.0, 180.0, 0.0, 0.0]
        pregrasp_pose = [50.0, 0.0, 60.0, 180.0, 0.0, 0.0]
        grasp_pose = [50.0, 0.0, 60.0, 180.0, 0.0, 0.0]
        lift_pose = [50.0, 0.0, 100.0, 180.0, 0.0, 0.0]

        parking_to_pregrasp = _constant_segment(
            "PARKING_TO_PREGRASP", parking_pose, gripper_value=GRIPPER_OPEN, num_samples=2
        )
        # PARKING_TO_PREGRASP must end where GRASP begins.
        parking_to_pregrasp = SegmentTrajectory(
            segment="PARKING_TO_PREGRASP",
            frame_index=parking_to_pregrasp.frame_index,
            time_s=parking_to_pregrasp.time_s,
            base_xyzrpy=np.asarray([parking_pose, pregrasp_pose]),
            gripper=np.full(2, GRIPPER_OPEN),
            fps=30.0,
        )
        grasp = SegmentTrajectory(
            segment="GRASP",
            frame_index=parking_to_pregrasp.frame_index,
            time_s=parking_to_pregrasp.time_s,
            base_xyzrpy=np.asarray([pregrasp_pose, grasp_pose]),
            gripper=np.asarray([GRIPPER_OPEN, GRIPPER_CLOSED]),
            fps=30.0,
        )
        lift = SegmentTrajectory(
            segment="LIFT",
            frame_index=parking_to_pregrasp.frame_index,
            time_s=parking_to_pregrasp.time_s,
            base_xyzrpy=np.asarray([grasp_pose, lift_pose]),
            gripper=np.full(2, GRIPPER_CLOSED),
            fps=30.0,
        )

        hover_pose = board_point_to_base_pose(
            path[0],
            config.hover_height_mm,
            board_to_base=transform,
            tool_rpy_deg=config.tool_rpy_deg,
        )
        transfer_above_board = build_transfer_segment(
            "TRANSFER_ABOVE_BOARD", lift_pose, hover_pose, config=config
        )
        descend, erase, lift_from_board = build_descend_erase_lift(
            path, board_to_base=transform, config=config
        )
        return_above_eraser = SegmentTrajectory(
            segment="RETURN_ABOVE_ERASER",
            frame_index=parking_to_pregrasp.frame_index,
            time_s=parking_to_pregrasp.time_s,
            base_xyzrpy=np.asarray([lift_from_board.end_pose(), lift_pose]),
            gripper=np.full(2, GRIPPER_CLOSED),
            fps=30.0,
        )
        place = SegmentTrajectory(
            segment="PLACE",
            frame_index=parking_to_pregrasp.frame_index,
            time_s=parking_to_pregrasp.time_s,
            base_xyzrpy=np.asarray([lift_pose, grasp_pose]),
            gripper=np.full(2, GRIPPER_CLOSED),
            fps=30.0,
        )
        open_gripper = build_open_gripper_segment(grasp_pose, fps=30.0, duration_s=0.2)
        finish = build_finish_segment(grasp_pose, fps=30.0)

        return {
            "PARKING_TO_PREGRASP": parking_to_pregrasp,
            "GRASP": grasp,
            "LIFT": lift,
            "TRANSFER_ABOVE_BOARD": transfer_above_board,
            "DESCEND": descend,
            "ERASE": erase,
            "LIFT_FROM_BOARD": lift_from_board,
            "RETURN_ABOVE_ERASER": return_above_eraser,
            "PLACE": place,
            "OPEN_GRIPPER": open_gripper,
            "FINISH": finish,
        }

    def test_full_trajectory_composes_and_validates(self) -> None:
        segments = self._build_full_segments()
        trajectory = compose_full_trajectory(segments, seed=1234)
        self.assertEqual([s.segment for s in trajectory.segments], list(SEGMENT_ORDER))
        self.assertEqual(trajectory.status, "unverified")

    def test_missing_segment_is_reported(self) -> None:
        segments = self._build_full_segments()
        del segments["PLACE"]
        with self.assertRaisesRegex(TrajectoryError, "PLACE"):
            compose_full_trajectory(segments, seed=1)

    def test_start_and_end_pose_match_the_built_config(self) -> None:
        segments = self._build_full_segments()
        trajectory = compose_full_trajectory(segments, seed=7)
        np.testing.assert_allclose(
            trajectory.segments[0].start_pose(), [0.0, 0.0, 200.0, 180.0, 0.0, 0.0]
        )
        self.assertAlmostEqual(float(trajectory.segments[-1].gripper[-1]), GRIPPER_OPEN)

    def test_deterministic_rebuild_matches_bit_for_bit(self) -> None:
        first = compose_full_trajectory(self._build_full_segments(), seed=42)
        second = compose_full_trajectory(self._build_full_segments(), seed=42)
        np.testing.assert_array_equal(
            first.concatenated_base_xyzrpy(), second.concatenated_base_xyzrpy()
        )
        np.testing.assert_array_equal(
            first.concatenated_gripper(), second.concatenated_gripper()
        )

    def test_broken_continuity_is_rejected(self) -> None:
        segments = self._build_full_segments()
        broken = segments["FINISH"]
        segments["FINISH"] = SegmentTrajectory(
            segment="FINISH",
            frame_index=broken.frame_index,
            time_s=broken.time_s,
            base_xyzrpy=broken.base_xyzrpy + np.asarray([100.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            gripper=broken.gripper,
            fps=broken.fps,
        )
        with self.assertRaisesRegex(TrajectoryError, "discontinuity"):
            compose_full_trajectory(segments, seed=1)


class FullTrajectoryJsonRoundTripTest(unittest.TestCase):
    def test_round_trip(self) -> None:
        transform = _identity_transform()
        config = _default_config()
        path = fixed_circle_path([0.0, 0.0], radius_mm=10.0, num_points=16)
        descend, erase, lift = build_descend_erase_lift(
            path, board_to_base=transform, config=config
        )
        trajectory = FullTrajectory(segments=(descend, erase, lift), seed=99)
        loaded = FullTrajectory.from_dict(trajectory.to_dict())
        np.testing.assert_allclose(
            loaded.concatenated_base_xyzrpy(), trajectory.concatenated_base_xyzrpy()
        )
        self.assertEqual(loaded.seed, 99)


if __name__ == "__main__":
    unittest.main()
