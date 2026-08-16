#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

from synthetic.preview.rviz_adapter import (
    JOINT_STATE_NAMES,
    RvizAdapterError,
    build_mock_joint_state_sequence,
)


class BuildMockJointStateSequenceTest(unittest.TestCase):
    def test_names_match_piper_infer_preview_convention(self) -> None:
        self.assertEqual(
            JOINT_STATE_NAMES,
            ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"),
        )

    def test_position_uses_radians_for_joints_and_meters_for_gripper(self) -> None:
        joints = np.asarray([[0.1, 0.2, -0.3, 0.0, 0.1, -0.1]])
        gripper_mm = np.asarray([34.0])
        messages = build_mock_joint_state_sequence(joints, gripper_mm)
        self.assertEqual(len(messages), 1)
        message = messages[0]
        self.assertEqual(message.name, JOINT_STATE_NAMES)
        np.testing.assert_allclose(message.position[:6], joints[0])
        self.assertAlmostEqual(message.position[6], 0.034)  # 34mm -> 0.034m

    def test_frame_index_is_sequential(self) -> None:
        joints = np.zeros((3, 6))
        gripper_mm = np.zeros(3)
        messages = build_mock_joint_state_sequence(joints, gripper_mm)
        self.assertEqual([m.frame_index for m in messages], [0, 1, 2])

    def test_to_dict_is_json_serializable_shape(self) -> None:
        joints = np.zeros((1, 6))
        gripper_mm = np.zeros(1)
        message = build_mock_joint_state_sequence(joints, gripper_mm)[0]
        payload = message.to_dict()
        self.assertEqual(payload["name"], list(JOINT_STATE_NAMES))
        self.assertEqual(len(payload["position"]), 7)

    def test_mismatched_shapes_rejected(self) -> None:
        with self.assertRaisesRegex(RvizAdapterError, "shape"):
            build_mock_joint_state_sequence(np.zeros((3, 6)), np.zeros(2))

    def test_gripper_out_of_range_rejected(self) -> None:
        with self.assertRaisesRegex(RvizAdapterError, "mm"):
            build_mock_joint_state_sequence(np.zeros((1, 6)), np.asarray([100.0]))

    def test_nan_rejected(self) -> None:
        joints = np.zeros((1, 6))
        joints[0, 0] = float("nan")
        with self.assertRaisesRegex(RvizAdapterError, "NaN or Inf"):
            build_mock_joint_state_sequence(joints, np.zeros(1))


if __name__ == "__main__":
    unittest.main()
