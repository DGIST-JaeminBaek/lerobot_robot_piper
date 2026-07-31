#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

from synthetic.trajectory.schema import TrajectoryError
from synthetic.trajectory.shapes import (
    fixed_circle_path,
    fixed_rectangle_path,
    fixed_triangle_path,
)


class CirclePathTest(unittest.TestCase):
    def test_all_points_are_radius_from_center(self) -> None:
        center = [100.0, 50.0]
        path = fixed_circle_path(center, radius_mm=30.0, num_points=64)
        distances = np.linalg.norm(path - np.asarray(center), axis=1)
        np.testing.assert_allclose(distances, np.full(64, 30.0), atol=1e-9)

    def test_path_is_closed(self) -> None:
        path = fixed_circle_path([0.0, 0.0], radius_mm=10.0, num_points=32)
        np.testing.assert_allclose(path[0], path[-1], atol=1e-9)

    def test_translating_center_translates_the_whole_path(self) -> None:
        base = fixed_circle_path([0.0, 0.0], radius_mm=20.0, num_points=16)
        shifted = fixed_circle_path([50.0, -25.0], radius_mm=20.0, num_points=16)
        np.testing.assert_allclose(shifted - base, np.tile([50.0, -25.0], (16, 1)), atol=1e-9)

    def test_rotation_rotates_the_start_point(self) -> None:
        # 37 points over 360 degrees (endpoint=True) gives an exact 10-degree
        # step, so a 90-degree rotation lands exactly on the 9th sample.
        unrotated = fixed_circle_path([0.0, 0.0], radius_mm=10.0, num_points=37)
        rotated = fixed_circle_path(
            [0.0, 0.0], radius_mm=10.0, num_points=37, rotation_deg=90.0
        )
        np.testing.assert_allclose(rotated[0], unrotated[9], atol=1e-9)

    def test_non_positive_radius_rejected(self) -> None:
        with self.assertRaisesRegex(TrajectoryError, "radius_mm"):
            fixed_circle_path([0.0, 0.0], radius_mm=0.0, num_points=16)

    def test_too_few_points_rejected(self) -> None:
        with self.assertRaisesRegex(TrajectoryError, "num_points"):
            fixed_circle_path([0.0, 0.0], radius_mm=10.0, num_points=4)


class TrianglePathTest(unittest.TestCase):
    def test_path_is_closed(self) -> None:
        path = fixed_triangle_path([0.0, 0.0], circumradius_mm=40.0, num_points=30)
        np.testing.assert_allclose(path[0], path[-1], atol=1e-9)

    def test_vertices_are_circumradius_from_center(self) -> None:
        center = np.asarray([10.0, 20.0])
        # sample coarsely enough that resampled points land on the vertices
        path = fixed_triangle_path(center, circumradius_mm=25.0, num_points=4)
        distances = np.linalg.norm(path[:3] - center, axis=1)
        np.testing.assert_allclose(distances, np.full(3, 25.0), atol=1e-6)

    def test_translating_center_translates_the_whole_path(self) -> None:
        base = fixed_triangle_path([0.0, 0.0], circumradius_mm=15.0, num_points=12)
        shifted = fixed_triangle_path([5.0, 5.0], circumradius_mm=15.0, num_points=12)
        np.testing.assert_allclose(shifted - base, np.tile([5.0, 5.0], (12, 1)), atol=1e-9)

    def test_rotation_changes_vertex_locations(self) -> None:
        unrotated = fixed_triangle_path([0.0, 0.0], circumradius_mm=10.0, num_points=4)
        rotated = fixed_triangle_path(
            [0.0, 0.0], circumradius_mm=10.0, num_points=4, rotation_deg=45.0
        )
        self.assertFalse(np.allclose(unrotated[:3], rotated[:3]))
        distances = np.linalg.norm(rotated[:3], axis=1)
        np.testing.assert_allclose(distances, np.full(3, 10.0), atol=1e-6)

    def test_too_few_points_rejected(self) -> None:
        with self.assertRaisesRegex(TrajectoryError, "num_points"):
            fixed_triangle_path([0.0, 0.0], circumradius_mm=10.0, num_points=2)


class RectanglePathTest(unittest.TestCase):
    def test_path_is_closed(self) -> None:
        path = fixed_rectangle_path(
            [0.0, 0.0], width_mm=40.0, height_mm=20.0, num_points=20
        )
        np.testing.assert_allclose(path[0], path[-1], atol=1e-9)

    def test_corner_extents_match_width_and_height(self) -> None:
        center = np.asarray([0.0, 0.0])
        path = fixed_rectangle_path(center, width_mm=40.0, height_mm=20.0, num_points=5)
        self.assertAlmostEqual(path[:, 0].max() - path[:, 0].min(), 40.0, places=6)
        self.assertAlmostEqual(path[:, 1].max() - path[:, 1].min(), 20.0, places=6)

    def test_translating_center_translates_the_whole_path(self) -> None:
        base = fixed_rectangle_path([0.0, 0.0], width_mm=30.0, height_mm=10.0, num_points=9)
        shifted = fixed_rectangle_path(
            [100.0, 200.0], width_mm=30.0, height_mm=10.0, num_points=9
        )
        np.testing.assert_allclose(
            shifted - base, np.tile([100.0, 200.0], (9, 1)), atol=1e-9
        )

    def test_non_positive_width_rejected(self) -> None:
        with self.assertRaisesRegex(TrajectoryError, "width_mm"):
            fixed_rectangle_path([0.0, 0.0], width_mm=0.0, height_mm=10.0, num_points=8)

    def test_too_few_points_rejected(self) -> None:
        with self.assertRaisesRegex(TrajectoryError, "num_points"):
            fixed_rectangle_path([0.0, 0.0], width_mm=10.0, height_mm=10.0, num_points=3)


class DeterminismTest(unittest.TestCase):
    def test_repeated_calls_are_bit_identical(self) -> None:
        first = fixed_circle_path([12.0, -8.0], radius_mm=17.5, num_points=40)
        second = fixed_circle_path([12.0, -8.0], radius_mm=17.5, num_points=40)
        np.testing.assert_array_equal(first, second)

        first_tri = fixed_triangle_path([1.0, 2.0], circumradius_mm=9.0, num_points=13)
        second_tri = fixed_triangle_path([1.0, 2.0], circumradius_mm=9.0, num_points=13)
        np.testing.assert_array_equal(first_tri, second_tri)

        first_rect = fixed_rectangle_path(
            [3.0, -4.0], width_mm=22.0, height_mm=11.0, num_points=17
        )
        second_rect = fixed_rectangle_path(
            [3.0, -4.0], width_mm=22.0, height_mm=11.0, num_points=17
        )
        np.testing.assert_array_equal(first_rect, second_rect)


if __name__ == "__main__":
    unittest.main()
