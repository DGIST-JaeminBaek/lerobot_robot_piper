#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

from synthetic.kinematics.piper_fk import (
    ForwardKinematicsError,
    forward_kinematics_eef,
    forward_kinematics_eef_batch,
)


class ForwardKinematicsEefTest(unittest.TestCase):
    def test_zero_configuration_matches_documented_init_pos(self) -> None:
        # docs/kinematics/kinematics_check.md + piper_sdk's own C_PiperForwardKinematics
        # dh_is_offset=1 branch documents init_pos = [56.128, 0.0, 213.266, 0.0, 85.0, 0.0].
        pose = forward_kinematics_eef([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(
            pose, [56.128, 0.0, 213.266, 0.0, 85.0, 0.0], atol=1e-2
        )

    def test_output_is_position_mm_and_orientation_degrees(self) -> None:
        pose = forward_kinematics_eef([0.1, 0.2, -0.3, 0.05, -0.1, 0.2])
        self.assertEqual(pose.shape, (6,))
        self.assertTrue(np.isfinite(pose).all())
        # A non-trivial arm reach should be on the order of hundreds of mm,
        # not meters or single mm -- a coarse sanity check on units.
        self.assertGreater(np.linalg.norm(pose[:3]), 50.0)
        self.assertLess(np.linalg.norm(pose[:3]), 1000.0)

    def test_wrong_shape_rejected(self) -> None:
        with self.assertRaisesRegex(ForwardKinematicsError, "shape"):
            forward_kinematics_eef([0.0, 0.0, 0.0])

    def test_nan_rejected(self) -> None:
        with self.assertRaisesRegex(ForwardKinematicsError, "NaN or Inf"):
            forward_kinematics_eef([0.0, float("nan"), 0.0, 0.0, 0.0, 0.0])

    def test_batch_matches_single_call(self) -> None:
        configs = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.2, -0.3, 0.05, -0.1, 0.2],
        ]
        batch = forward_kinematics_eef_batch(configs)
        self.assertEqual(batch.shape, (2, 6))
        for row, config in zip(batch, configs, strict=True):
            np.testing.assert_allclose(row, forward_kinematics_eef(config))

    def test_batch_wrong_shape_rejected(self) -> None:
        with self.assertRaisesRegex(ForwardKinematicsError, "shape"):
            forward_kinematics_eef_batch([[0.0, 0.0, 0.0]])


if __name__ == "__main__":
    unittest.main()
