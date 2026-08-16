#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

from synthetic.trajectory.schema import TrajectoryError
from synthetic.trajectory.timing import (
    frame_and_time,
    interpolate_pose_linear,
    polyline_length,
    resample_polyline_by_arc_length,
    sample_count_for_distance,
)


class SampleCountTest(unittest.TestCase):
    def test_sample_count_scales_with_distance_and_speed(self) -> None:
        count = sample_count_for_distance(100.0, fps=30.0, speed_mm_per_s=50.0)
        # duration = 100/50 = 2s at 30fps -> 60 frames -> +1 endpoint sample
        self.assertEqual(count, 61)

    def test_sample_count_has_a_minimum_of_two(self) -> None:
        count = sample_count_for_distance(0.0, fps=30.0, speed_mm_per_s=50.0)
        self.assertEqual(count, 2)

    def test_non_positive_speed_rejected(self) -> None:
        with self.assertRaisesRegex(TrajectoryError, "speed_mm_per_s"):
            sample_count_for_distance(10.0, fps=30.0, speed_mm_per_s=0.0)

    def test_non_positive_fps_rejected(self) -> None:
        with self.assertRaisesRegex(TrajectoryError, "fps"):
            sample_count_for_distance(10.0, fps=0.0, speed_mm_per_s=10.0)

    def test_negative_distance_rejected(self) -> None:
        with self.assertRaisesRegex(TrajectoryError, "distance_mm"):
            sample_count_for_distance(-1.0, fps=30.0, speed_mm_per_s=10.0)


class FrameAndTimeTest(unittest.TestCase):
    def test_frame_and_time_are_zero_based_and_evenly_spaced(self) -> None:
        frame_index, time_s = frame_and_time(5, 25.0)
        np.testing.assert_array_equal(frame_index, [0, 1, 2, 3, 4])
        np.testing.assert_allclose(time_s, [0.0, 0.04, 0.08, 0.12, 0.16])

    def test_zero_samples_rejected(self) -> None:
        with self.assertRaisesRegex(TrajectoryError, "num_samples"):
            frame_and_time(0, 30.0)


class PolylineLengthTest(unittest.TestCase):
    def test_open_polyline_length(self) -> None:
        points = [[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]]
        self.assertAlmostEqual(polyline_length(points), 7.0)

    def test_closed_polyline_adds_the_closing_edge(self) -> None:
        points = [[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]]
        self.assertAlmostEqual(polyline_length(points, closed=True), 7.0 + 5.0)

    def test_nan_rejected(self) -> None:
        with self.assertRaisesRegex(TrajectoryError, "NaN or Inf"):
            polyline_length([[0.0, 0.0], [float("nan"), 1.0]])


class ResamplePolylineTest(unittest.TestCase):
    def test_resample_preserves_start_and_end_of_open_polyline(self) -> None:
        points = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]
        resampled = resample_polyline_by_arc_length(points, 5, closed=False)
        np.testing.assert_allclose(resampled[0], [0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(resampled[-1], [10.0, 10.0], atol=1e-9)
        self.assertEqual(resampled.shape, (5, 2))

    def test_resample_is_evenly_spaced_by_arc_length(self) -> None:
        points = [[0.0, 0.0], [10.0, 0.0]]
        resampled = resample_polyline_by_arc_length(points, 6, closed=False)
        deltas = np.linalg.norm(np.diff(resampled, axis=0), axis=1)
        np.testing.assert_allclose(deltas, np.full(5, 2.0), atol=1e-9)

    def test_closed_resample_starts_and_ends_at_same_point(self) -> None:
        square = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
        resampled = resample_polyline_by_arc_length(square, 9, closed=True)
        np.testing.assert_allclose(resampled[0], resampled[-1], atol=1e-9)

    def test_duplicate_consecutive_points_rejected(self) -> None:
        points = [[0.0, 0.0], [0.0, 0.0], [10.0, 0.0]]
        with self.assertRaisesRegex(TrajectoryError, "duplicate"):
            resample_polyline_by_arc_length(points, 4, closed=False)

    def test_too_few_samples_rejected(self) -> None:
        points = [[0.0, 0.0], [10.0, 0.0]]
        with self.assertRaisesRegex(TrajectoryError, "num_samples"):
            resample_polyline_by_arc_length(points, 1, closed=False)


class InterpolatePoseLinearTest(unittest.TestCase):
    def test_interpolation_reaches_start_and_end_exactly(self) -> None:
        start = [0.0, 0.0, 100.0, 0.0, 0.0, 0.0]
        end = [100.0, 50.0, 100.0, 0.0, 0.0, 90.0]
        path = interpolate_pose_linear(start, end, 11)
        np.testing.assert_allclose(path[0], start, atol=1e-9)
        np.testing.assert_allclose(path[-1], end, atol=1e-9)
        self.assertEqual(path.shape, (11, 6))

    def test_midpoint_is_the_average(self) -> None:
        start = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        end = [10.0, 20.0, 30.0, 0.0, 0.0, 0.0]
        path = interpolate_pose_linear(start, end, 3)
        np.testing.assert_allclose(path[1], [5.0, 10.0, 15.0, 0.0, 0.0, 0.0])

    def test_wrong_shape_rejected(self) -> None:
        with self.assertRaisesRegex(TrajectoryError, "shape"):
            interpolate_pose_linear([0.0, 0.0], [1.0, 1.0, 1.0, 0.0, 0.0, 0.0], 3)


if __name__ == "__main__":
    unittest.main()
