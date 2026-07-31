#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

from synthetic.transforms.board_base import (
    RigidTransform,
    TransformError,
    embed_board_xy,
    parse_correspondences,
    reject_collinear_2d,
    reject_duplicate_points,
    residual_statistics,
    solve_rigid_transform,
)


def _rotation_from_euler(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rot_z @ rot_y @ rot_x


_BOARD_XY = np.asarray(
    [
        [0.0, 0.0],
        [400.0, 0.0],
        [400.0, 300.0],
        [0.0, 300.0],
        [150.0, 120.0],
    ]
)


class KnownTransformTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rotation_true = _rotation_from_euler(0.15, -0.25, 1.1)
        self.translation_true = np.asarray([320.0, -40.0, 55.0])
        self.board_xyz = embed_board_xy(_BOARD_XY)
        self.base_xyz = (
            self.rotation_true @ self.board_xyz.T
        ).T + self.translation_true

    def test_solver_recovers_known_rotation_and_translation(self) -> None:
        transform = solve_rigid_transform(self.board_xyz, self.base_xyz)
        np.testing.assert_allclose(
            transform.rotation, self.rotation_true, rtol=0, atol=1e-8
        )
        np.testing.assert_allclose(
            transform.translation, self.translation_true, rtol=0, atol=1e-6
        )

    def test_forward_inverse_round_trip(self) -> None:
        transform = solve_rigid_transform(self.board_xyz, self.base_xyz)
        inverse = transform.inverse()
        recovered_board = inverse.apply(transform.apply(self.board_xyz))
        np.testing.assert_allclose(
            recovered_board, self.board_xyz, rtol=0, atol=1e-6
        )
        recovered_base = transform.apply(inverse.apply(self.base_xyz))
        np.testing.assert_allclose(recovered_base, self.base_xyz, rtol=0, atol=1e-6)

    def test_residuals_are_near_zero_for_exact_correspondences(self) -> None:
        transform = solve_rigid_transform(self.board_xyz, self.base_xyz)
        stats = residual_statistics(transform, self.board_xyz, self.base_xyz)
        self.assertLess(stats["max_mm"], 1e-6)
        self.assertLess(stats["mean_mm"], 1e-6)

    def test_plane_normal_matches_rotated_z_axis(self) -> None:
        transform = solve_rigid_transform(self.board_xyz, self.base_xyz)
        expected_normal = self.rotation_true @ np.asarray([0.0, 0.0, 1.0])
        np.testing.assert_allclose(
            transform.plane_normal(), expected_normal, rtol=0, atol=1e-8
        )

    def test_residual_statistics_grow_with_added_noise(self) -> None:
        rng = np.random.default_rng(0)
        noise = rng.normal(scale=0.5, size=self.base_xyz.shape)
        noisy_base = self.base_xyz + noise
        transform = solve_rigid_transform(self.board_xyz, noisy_base)
        stats = residual_statistics(transform, self.board_xyz, noisy_base)
        self.assertGreater(stats["mean_mm"], 0.0)
        # Least-squares fit of small iid noise should stay within a few
        # multiples of the noise scale, not blow up.
        self.assertLess(stats["max_mm"], 5.0)

    def test_json_round_trip_of_rigid_transform(self) -> None:
        transform = solve_rigid_transform(self.board_xyz, self.base_xyz)
        loaded = RigidTransform.from_dict(transform.to_dict())
        np.testing.assert_allclose(loaded.rotation, transform.rotation, atol=1e-12)
        np.testing.assert_allclose(
            loaded.translation, transform.translation, atol=1e-12
        )


class RejectInvalidCorrespondencesTest(unittest.TestCase):
    def test_fewer_than_three_points_rejected(self) -> None:
        board = embed_board_xy([[0.0, 0.0], [10.0, 0.0]])
        base = board.copy()
        with self.assertRaisesRegex(TransformError, "at least 3"):
            solve_rigid_transform(board, base)

    def test_collinear_points_rejected(self) -> None:
        collinear = np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]])
        with self.assertRaisesRegex(TransformError, "collinear"):
            reject_collinear_2d(collinear)

    def test_solver_rejects_collinear_board_points(self) -> None:
        collinear_xy = np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
        board = embed_board_xy(collinear_xy)
        base = board + np.asarray([100.0, 0.0, 0.0])
        with self.assertRaisesRegex(TransformError, "collinear"):
            solve_rigid_transform(board, base)

    def test_duplicate_points_rejected(self) -> None:
        points = np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        with self.assertRaisesRegex(TransformError, "duplicate"):
            reject_duplicate_points(points, name="test points")

    def test_solver_rejects_duplicate_board_points(self) -> None:
        board = embed_board_xy([[0.0, 0.0], [10.0, 0.0], [0.0, 0.0], [5.0, 8.0]])
        base = board + np.asarray([1.0, 2.0, 3.0])
        with self.assertRaisesRegex(TransformError, "duplicate"):
            solve_rigid_transform(board, base)

    def test_solver_always_returns_a_proper_rotation_for_planar_board_source(
        self,
    ) -> None:
        # board_xyz is always embedded at z=0, so the Kabsch covariance
        # matrix always has rank <=2: the sign of the rotation's
        # out-of-plane axis is not determined by the data, and an
        # in-plane mirror of the board is exactly reproducible by a proper
        # 180-degree rotation about an in-plane axis. The solver must
        # return that proper rotation (det=+1), not an arbitrary
        # SVD-sign-dependent reflection.
        board_xyz = embed_board_xy(_BOARD_XY)
        in_plane_mirror = np.diag([-1.0, 1.0, 1.0])
        mirrored_base = (in_plane_mirror @ board_xyz.T).T + np.asarray(
            [10.0, 20.0, 30.0]
        )
        transform = solve_rigid_transform(board_xyz, mirrored_base)
        transform.validate()
        stats = residual_statistics(transform, board_xyz, mirrored_base)
        self.assertLess(stats["max_mm"], 1e-6)

    def test_mismatched_point_counts_rejected(self) -> None:
        board = embed_board_xy(_BOARD_XY)
        with self.assertRaisesRegex(TransformError, "point counts must match"):
            solve_rigid_transform(board, board[:3])


class RigidTransformValidationTest(unittest.TestCase):
    def test_non_orthonormal_rotation_rejected(self) -> None:
        bad_rotation = np.eye(3) * 1.1
        transform = RigidTransform(rotation=bad_rotation, translation=np.zeros(3))
        with self.assertRaisesRegex(TransformError, "orthonormal"):
            transform.validate()

    def test_reflection_matrix_rejected_by_validate(self) -> None:
        reflection = np.diag([1.0, 1.0, -1.0])
        transform = RigidTransform(rotation=reflection, translation=np.zeros(3))
        with self.assertRaisesRegex(TransformError, "reflection"):
            transform.validate()

    def test_wrong_shape_rejected(self) -> None:
        transform = RigidTransform(
            rotation=np.eye(2), translation=np.zeros(3)
        )
        with self.assertRaisesRegex(TransformError, "shape"):
            transform.validate()

    def test_nan_rejected(self) -> None:
        rotation = np.eye(3)
        rotation[0, 0] = float("nan")
        transform = RigidTransform(rotation=rotation, translation=np.zeros(3))
        with self.assertRaisesRegex(TransformError, "NaN or Inf"):
            transform.validate()


class ParseCorrespondencesTest(unittest.TestCase):
    def _payload(self, **overrides) -> dict:
        payload = {
            "format_version": 1,
            "type": "board_base_correspondence_set",
            "status": "unverified",
            "board_unit": "mm",
            "base_unit": "mm",
            "correspondences": [
                {"name": "a", "board_xy": [0.0, 0.0], "base_xyz": [0.0, 0.0, 0.0]},
                {"name": "b", "board_xy": [10.0, 0.0], "base_xyz": [10.0, 0.0, 0.0]},
                {"name": "c", "board_xy": [0.0, 10.0], "base_xyz": [0.0, 10.0, 0.0]},
            ],
        }
        payload.update(overrides)
        return payload

    def test_valid_payload_parses(self) -> None:
        names, board_xy, base_xyz, board_unit, base_unit = parse_correspondences(
            self._payload()
        )
        self.assertEqual(names, ["a", "b", "c"])
        self.assertEqual(board_xy.shape, (3, 2))
        self.assertEqual(base_xyz.shape, (3, 3))
        self.assertEqual(board_unit, "mm")
        self.assertEqual(base_unit, "mm")

    def test_non_mm_board_unit_rejected(self) -> None:
        with self.assertRaisesRegex(TransformError, "mm"):
            parse_correspondences(self._payload(board_unit="normalized"))

    def test_non_mm_base_unit_rejected(self) -> None:
        with self.assertRaisesRegex(TransformError, "mm"):
            parse_correspondences(self._payload(base_unit="normalized"))

    def test_duplicate_names_rejected(self) -> None:
        payload = self._payload()
        payload["correspondences"][1]["name"] = "a"
        with self.assertRaisesRegex(TransformError, "unique"):
            parse_correspondences(payload)

    def test_too_few_correspondences_rejected(self) -> None:
        payload = self._payload()
        payload["correspondences"] = payload["correspondences"][:2]
        with self.assertRaisesRegex(TransformError, "at least 3"):
            parse_correspondences(payload)

    def test_wrong_type_rejected(self) -> None:
        with self.assertRaisesRegex(TransformError, "expected type"):
            parse_correspondences(self._payload(type="something_else"))


if __name__ == "__main__":
    unittest.main()
