#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

from synthetic.calibration.aggregate_board_points import (
    aggregate_observations,
    build_aggregated_payload,
)


class AggregateBoardPointsTest(unittest.TestCase):
    def setUp(self) -> None:
        base = np.asarray(
            [
                [100.0, 80.0],
                [1100.0, 85.0],
                [1090.0, 650.0],
                [90.0, 640.0],
            ]
        )
        self.observations = np.stack(
            [
                base + np.asarray([0.0, 0.0]),
                base + np.asarray([1.0, -1.0]),
                base + np.asarray([-1.0, 1.0]),
            ]
        )
        self.board_points = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        )

    def test_median_recovers_center_observation(self) -> None:
        representative, statistics = aggregate_observations(
            self.observations,
            method="median",
        )
        np.testing.assert_allclose(representative, self.observations[0])
        self.assertEqual(statistics["sample_count"], 3)
        self.assertAlmostEqual(
            statistics["global_radial_deviation_px_max"],
            np.sqrt(2),
        )

    def test_median_resists_one_large_click_error(self) -> None:
        observations = self.observations.copy()
        observations[2, 0] += [80.0, 60.0]
        representative, _ = aggregate_observations(
            observations,
            method="median",
        )
        np.testing.assert_allclose(
            representative[0],
            self.observations[0, 0] + [1.0, 0.0],
        )

    def test_payload_keeps_all_sources_and_observations(self) -> None:
        selections = []
        for index in range(3):
            selections.append(
                {
                    "_input_file": f"/tmp/selection-{index}.json",
                    "source": {
                        "kind": "image",
                        "path": f"/tmp/frame-{index}.png",
                        "frame_index": None,
                        "width": 1280,
                        "height": 720,
                    },
                    "board": {
                        "unit": "normalized",
                        "width": 1.0,
                        "height": 1.0,
                        "origin": "top_left",
                        "x_direction": "top_left_to_top_right",
                        "y_direction": "top_left_to_bottom_left",
                    },
                }
            )
        payload = build_aggregated_payload(
            selections=selections,
            observations=self.observations,
            board_points=self.board_points,
            method="median",
        )
        self.assertEqual(payload["status"], "unverified")
        self.assertEqual(payload["aggregation"]["sample_count"], 3)
        self.assertEqual(len(payload["observations"]), 3)
        self.assertIn("not averaged", payload["note"])


if __name__ == "__main__":
    unittest.main()

