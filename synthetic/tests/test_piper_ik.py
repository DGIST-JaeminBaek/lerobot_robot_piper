#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

from synthetic.kinematics.piper_fk import forward_kinematics_eef
from synthetic.kinematics.piper_ik import (
    InverseKinematicsError,
    _check_joint_limits,
    joint_bounds_rad,
    matrix_to_xyzrpy,
    solve_ik,
    solve_ik_sequence,
    xyzrpy_to_matrix,
)

# Within the URDF's declared joint limits (joint2 in [0, pi], joint3 in
# [-2.967, 0], etc.) -- see agx_arm_urdf/piper/urdf/piper_description.urdf.
_KNOWN_JOINT_RAD = [0.1, 0.5, -0.3, 0.05, -0.1, 0.2]
_ZERO_SEED = [0.0, 0.5, -0.3, 0.05, -0.1, 0.2]  # near the known config, a workable seed


class PoseMatrixRoundTripTest(unittest.TestCase):
    def test_xyzrpy_matrix_round_trip(self) -> None:
        pose = np.asarray([100.0, -50.0, 200.0, 10.0, -20.0, 30.0])
        matrix = xyzrpy_to_matrix(pose)
        recovered = matrix_to_xyzrpy(matrix)
        np.testing.assert_allclose(recovered, pose, atol=1e-6)


class KnownJointRoundTripTest(unittest.TestCase):
    def test_fk_ik_fk_round_trip_recovers_the_target_pose(self) -> None:
        target_pose = forward_kinematics_eef(_KNOWN_JOINT_RAD)
        solution = solve_ik(target_pose, seed_joint_rad=_ZERO_SEED)
        self.assertLessEqual(solution.position_error_mm, 1.0)
        self.assertLessEqual(solution.angle_error_deg, 1.0)
        recovered_pose = forward_kinematics_eef(solution.joint_rad)
        np.testing.assert_allclose(recovered_pose[:3], target_pose[:3], atol=1.0)

    def test_solution_picks_the_seed_closest_candidate(self) -> None:
        target_pose = forward_kinematics_eef(_KNOWN_JOINT_RAD)
        solution = solve_ik(target_pose, seed_joint_rad=_KNOWN_JOINT_RAD)
        np.testing.assert_allclose(solution.joint_rad, _KNOWN_JOINT_RAD, atol=1e-2)


class SequenceContinuityTest(unittest.TestCase):
    def test_warm_started_sequence_has_small_frame_to_frame_steps(self) -> None:
        joint_path = np.linspace([0.0, 0.5, -0.2, 0.0, 0.0, 0.0], [0.2, 0.7, -0.4, 0.1, 0.1, 0.1], 6)
        pose_path = np.stack([forward_kinematics_eef(row) for row in joint_path])
        solved, solutions = solve_ik_sequence(
            pose_path, initial_seed_rad=joint_path[0], max_joint_step_rad=0.3
        )
        self.assertEqual(solved.shape, (6, 6))
        self.assertEqual(len(solutions), 6)
        for row, expected in zip(solved, joint_path, strict=True):
            np.testing.assert_allclose(row, expected, atol=0.05)

    def test_excessive_joint_step_is_rejected(self) -> None:
        pose_a = forward_kinematics_eef([0.0, 0.5, -0.2, 0.0, 0.0, 0.0])
        pose_b = forward_kinematics_eef([0.0, 1.4, -1.5, 0.0, 0.0, 0.0])
        with self.assertRaisesRegex(InverseKinematicsError, "joint step"):
            solve_ik_sequence(
                [pose_a, pose_b],
                initial_seed_rad=[0.0, 0.5, -0.2, 0.0, 0.0, 0.0],
                max_joint_step_rad=1e-4,
            )


class FailureModeTest(unittest.TestCase):
    def test_unreachable_pose_is_rejected(self) -> None:
        far_away_pose = [10000.0, 10000.0, 10000.0, 0.0, 0.0, 0.0]
        with self.assertRaisesRegex(InverseKinematicsError, "did not converge"):
            solve_ik(far_away_pose, seed_joint_rad=[0.0, 0.5, -0.3, 0.0, 0.0, 0.0])

    def test_joint_limit_violation_is_rejected(self) -> None:
        bounds = joint_bounds_rad()
        out_of_range = bounds[:, 1] + 1.0
        with self.assertRaisesRegex(InverseKinematicsError, "joint limit"):
            _check_joint_limits(out_of_range, bounds)

    def test_nan_target_pose_is_rejected(self) -> None:
        with self.assertRaisesRegex(InverseKinematicsError, "NaN or Inf"):
            solve_ik(
                [float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0],
                seed_joint_rad=[0.0, 0.5, -0.3, 0.0, 0.0, 0.0],
            )

    def test_nan_seed_is_rejected(self) -> None:
        target_pose = forward_kinematics_eef(_KNOWN_JOINT_RAD)
        with self.assertRaisesRegex(InverseKinematicsError, "NaN or Inf"):
            solve_ik(target_pose, seed_joint_rad=[float("nan")] * 6)

    def test_wrong_shape_target_pose_is_rejected(self) -> None:
        with self.assertRaisesRegex(InverseKinematicsError, "shape"):
            solve_ik([0.0, 0.0, 0.0], seed_joint_rad=[0.0] * 6)


if __name__ == "__main__":
    unittest.main()
