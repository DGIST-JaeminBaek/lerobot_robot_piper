#!/usr/bin/env python3
"""End-to-end: board shape path -> Cartesian segments -> IK -> (N, 7) action.

Uses a board<->base transform anchored at an actually-reachable EEF pose
(from `forward_kinematics_eef` of a known in-bounds joint config), so a
small circle traced near it stays within the arm's workspace -- this is a
synthetic/test fixture, not a claim about the real board position.
"""

from __future__ import annotations

import unittest

import numpy as np

from synthetic.kinematics.action_conversion import build_normalized_action
from synthetic.kinematics.piper_fk import forward_kinematics_eef
from synthetic.kinematics.piper_ik import solve_ik_sequence
from synthetic.transforms.board_base import RigidTransform
from synthetic.trajectory.compose import BoardMotionConfig, build_descend_erase_lift
from synthetic.trajectory.shapes import fixed_circle_path

_KNOWN_JOINT_RAD = [0.1, 0.6, -0.4, 0.0, 0.2, 0.0]


class TrajectoryToActionIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        reachable_pose = forward_kinematics_eef(_KNOWN_JOINT_RAD)
        self.transform = RigidTransform(rotation=np.eye(3), translation=reachable_pose[:3])
        self.config = BoardMotionConfig(
            hover_height_mm=15.0,
            contact_height_mm=0.0,
            tool_rpy_deg=tuple(reachable_pose[3:]),
            fps=30.0,
            transfer_speed_mm_per_s=200.0,
            descend_lift_speed_mm_per_s=30.0,
            erase_speed_mm_per_s=40.0,
        )
        path = fixed_circle_path([0.0, 0.0], radius_mm=10.0, num_points=16)
        self.descend, self.erase, self.lift = build_descend_erase_lift(
            path, board_to_base=self.transform, config=self.config
        )
        self.poses = np.concatenate(
            [self.descend.base_xyzrpy, self.erase.base_xyzrpy, self.lift.base_xyzrpy], axis=0
        )

    def test_ik_converges_for_the_whole_shape_motion(self) -> None:
        _joint_seq, solutions = solve_ik_sequence(
            self.poses,
            initial_seed_rad=_KNOWN_JOINT_RAD,
            position_tol_mm=1.0,
            angle_tol_deg=1.0,
            max_joint_step_rad=0.3,
        )
        self.assertEqual(len(solutions), self.poses.shape[0])
        self.assertLess(max(s.position_error_mm for s in solutions), 1.0)
        self.assertLess(max(s.angle_error_deg for s in solutions), 1.0)

    def test_output_action_has_shape_n_by_7_and_is_in_range(self) -> None:
        joint_seq, _solutions = solve_ik_sequence(
            self.poses,
            initial_seed_rad=_KNOWN_JOINT_RAD,
            position_tol_mm=1.0,
            angle_tol_deg=1.0,
        )
        gripper_fraction = np.ones(self.poses.shape[0])
        action = build_normalized_action(joint_seq, gripper_fraction)
        self.assertEqual(action.shape, (self.poses.shape[0], 7))
        self.assertTrue(np.all(action[:, :6] >= -100.0) and np.all(action[:, :6] <= 100.0))
        self.assertTrue(np.all(action[:, 6] >= 0.0) and np.all(action[:, 6] <= 100.0))
        # gripper stays closed (fraction=1.0) throughout descend/erase/lift.
        np.testing.assert_allclose(action[:, 6], 0.0, atol=1e-9)

    def test_fk_of_solved_joints_matches_the_requested_cartesian_path(self) -> None:
        joint_seq, _solutions = solve_ik_sequence(
            self.poses,
            initial_seed_rad=_KNOWN_JOINT_RAD,
            position_tol_mm=1.0,
            angle_tol_deg=1.0,
        )
        recovered_poses = np.stack([forward_kinematics_eef(row) for row in joint_seq])
        position_errors = np.linalg.norm(recovered_poses[:, :3] - self.poses[:, :3], axis=1)
        self.assertLess(float(position_errors.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
