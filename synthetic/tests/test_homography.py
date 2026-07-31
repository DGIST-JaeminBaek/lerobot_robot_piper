#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

from synthetic.calibration.common import (
    BoardSpec,
    CalibrationError,
    compute_homography,
    make_correspondences,
    parse_point_list,
    project_points,
    reprojection_statistics,
    validate_image_quad,
)


class HomographyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.image_corners = np.asarray(
            [
                [210.0, 120.0],
                [1080.0, 170.0],
                [990.0, 650.0],
                [140.0, 590.0],
            ],
            dtype=np.float64,
        )
        self.board = BoardSpec(width=900.0, height=500.0, unit="mm")
        self.board_corners = self.board.corner_coordinates()

    def test_corner_round_trip_is_pixel_exact(self) -> None:
        image_to_board, board_to_image = compute_homography(
            self.image_corners,
            self.board_corners,
        )
        board_predicted = project_points(self.image_corners, image_to_board)
        image_round_trip = project_points(board_predicted, board_to_image)
        np.testing.assert_allclose(
            board_predicted,
            self.board_corners,
            rtol=0,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            image_round_trip,
            self.image_corners,
            rtol=0,
            atol=1e-4,
        )

    def test_interior_points_round_trip(self) -> None:
        image_to_board, board_to_image = compute_homography(
            self.image_corners,
            self.board_corners,
        )
        board_points = np.asarray(
            [
                [0.0, 0.0],
                [450.0, 250.0],
                [135.0, 420.0],
                [900.0, 500.0],
            ]
        )
        image_points = project_points(board_points, board_to_image)
        board_round_trip = project_points(image_points, image_to_board)
        np.testing.assert_allclose(
            board_round_trip,
            board_points,
            rtol=0,
            atol=1e-4,
        )

    def test_reprojection_statistics_are_near_zero(self) -> None:
        image_to_board, board_to_image = compute_homography(
            self.image_corners,
            self.board_corners,
        )
        stats = reprojection_statistics(
            self.image_corners,
            self.board_corners,
            image_to_board,
            board_to_image,
        )
        self.assertLess(stats["image_error_px_max"], 1e-4)
        self.assertLess(stats["board_error_max"], 1e-4)

    def test_invalid_crossed_point_order_is_rejected(self) -> None:
        crossed = self.image_corners[[0, 2, 1, 3]]
        with self.assertRaisesRegex(CalibrationError, "convex quadrilateral"):
            validate_image_quad(crossed, width=1280, height=720)

    def test_out_of_image_point_is_rejected(self) -> None:
        outside = self.image_corners.copy()
        outside[0, 0] = -1
        with self.assertRaisesRegex(CalibrationError, "inside"):
            validate_image_quad(outside, width=1280, height=720)

    def test_parse_point_list_uses_required_order(self) -> None:
        points = parse_point_list("10,20;100,20;100,80;10,80")
        np.testing.assert_array_equal(
            points,
            np.asarray([[10, 20], [100, 20], [100, 80], [10, 80]]),
        )

    def test_make_correspondences_preserves_names_and_units(self) -> None:
        records = make_correspondences(self.image_corners, self.board)
        self.assertEqual(
            [record["name"] for record in records],
            ["top_left", "top_right", "bottom_right", "bottom_left"],
        )
        self.assertEqual(records[2]["board_xy"], [900.0, 500.0])

    def test_normalized_board_rejects_non_unit_size(self) -> None:
        with self.assertRaisesRegex(CalibrationError, "width=1.0"):
            BoardSpec(width=2.0, height=1.0, unit="normalized").validate()


if __name__ == "__main__":
    unittest.main()

