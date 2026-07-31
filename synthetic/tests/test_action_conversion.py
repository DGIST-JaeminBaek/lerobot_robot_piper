#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

from synthetic.kinematics.action_conversion import (
    ActionConversionError,
    GRIPPER_CLOSED_MM,
    GRIPPER_FULLY_OPEN_MM,
    JOINT_RAW_RANGE_MILLIDEG,
    build_normalized_action,
    gripper_fraction_to_mm,
    gripper_mm_to_normalized,
    joint_rad_to_normalized,
    normalized_to_gripper_mm,
    normalized_to_joint_rad,
)


class JointNormalizationTest(unittest.TestCase):
    def test_range_min_maps_to_minus_100(self) -> None:
        # joint2 range is [0, 180000] millideg -> 0 rad is its range_min.
        normalized = joint_rad_to_normalized([0.0, 0.0, -np.radians(170.0), 0.0, 0.0, 0.0])
        self.assertAlmostEqual(normalized[1], -100.0, places=6)

    def test_range_max_maps_to_plus_100(self) -> None:
        normalized = joint_rad_to_normalized(
            [0.0, np.radians(180.0), 0.0, 0.0, 0.0, 0.0]
        )
        self.assertAlmostEqual(normalized[1], 100.0, places=6)

    def test_midpoint_maps_to_zero(self) -> None:
        # joint1 range is symmetric [-150000, 150000] -> 0 rad is the midpoint.
        normalized = joint_rad_to_normalized([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(normalized[0], 0.0, places=6)

    def test_round_trip(self) -> None:
        joint_rad = [0.1, 0.5, -0.3, 0.2, -0.1, 0.3]
        normalized = joint_rad_to_normalized(joint_rad)
        recovered = normalized_to_joint_rad(normalized)
        np.testing.assert_allclose(recovered, joint_rad, atol=1e-9)

    def test_out_of_calibrated_range_is_rejected(self) -> None:
        # joint2's range is [0, 180] degrees; -10 degrees is out of range.
        with self.assertRaisesRegex(ActionConversionError, "calibrated range"):
            joint_rad_to_normalized([0.0, np.radians(-10.0), 0.0, 0.0, 0.0, 0.0])

    def test_normalized_out_of_bounds_is_rejected(self) -> None:
        with self.assertRaisesRegex(ActionConversionError, r"\[-100, 100\]"):
            normalized_to_joint_rad([101.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_nan_rejected(self) -> None:
        with self.assertRaisesRegex(ActionConversionError, "NaN or Inf"):
            joint_rad_to_normalized([float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_calibration_table_matches_piper_follower_source(self) -> None:
        # Cross-check against the values copied from
        # lerobot_robot_piper/piper_follower.py:76-84.
        self.assertEqual(JOINT_RAW_RANGE_MILLIDEG["joint2"], (0, 180000))
        self.assertEqual(JOINT_RAW_RANGE_MILLIDEG["joint3"], (-170000, 0))


class GripperNormalizationTest(unittest.TestCase):
    def test_closed_is_zero_normalized(self) -> None:
        self.assertAlmostEqual(gripper_mm_to_normalized(GRIPPER_CLOSED_MM), 0.0)

    def test_open_is_hundred_normalized(self) -> None:
        self.assertAlmostEqual(gripper_mm_to_normalized(GRIPPER_FULLY_OPEN_MM), 100.0)

    def test_round_trip(self) -> None:
        for mm in (0.0, 12.5, 34.0, 68.0):
            recovered = normalized_to_gripper_mm(gripper_mm_to_normalized(mm))
            self.assertAlmostEqual(recovered, mm, places=6)

    def test_out_of_range_is_rejected(self) -> None:
        with self.assertRaisesRegex(ActionConversionError, "calibrated range"):
            gripper_mm_to_normalized(-1.0)

    def test_fraction_to_mm_matches_open_closed_endpoints(self) -> None:
        mm = gripper_fraction_to_mm([1.0, 0.0, 0.5])
        np.testing.assert_allclose(
            mm, [GRIPPER_CLOSED_MM, GRIPPER_FULLY_OPEN_MM, GRIPPER_FULLY_OPEN_MM / 2.0]
        )

    def test_fraction_out_of_range_rejected(self) -> None:
        with self.assertRaisesRegex(ActionConversionError, r"\[0, 1\]"):
            gripper_fraction_to_mm([1.5])


class BuildNormalizedActionTest(unittest.TestCase):
    def test_output_shape_is_n_by_7(self) -> None:
        joints = np.zeros((5, 6))
        gripper_fraction = np.full(5, 1.0)
        action = build_normalized_action(joints, gripper_fraction)
        self.assertEqual(action.shape, (5, 7))

    def test_column_order_is_joint1_to_6_then_gripper(self) -> None:
        joints = np.asarray([[0.0, 0.0, -np.radians(170.0), 0.0, 0.0, 0.0]])
        action = build_normalized_action(joints, [1.0])
        self.assertAlmostEqual(action[0, 1], -100.0, places=6)  # joint2 at range_min
        self.assertAlmostEqual(action[0, 6], 0.0, places=6)  # gripper closed

    def test_mismatched_lengths_rejected(self) -> None:
        with self.assertRaisesRegex(ActionConversionError, "shape"):
            build_normalized_action(np.zeros((3, 6)), np.zeros(2))


if __name__ == "__main__":
    unittest.main()
