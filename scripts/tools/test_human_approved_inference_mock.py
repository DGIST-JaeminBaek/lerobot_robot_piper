#!/usr/bin/env python3
"""piper_human_approved_inference의 실물 전송 경로를 하드웨어 없이 검증한다."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest

import numpy as np


MODULE_PATH = pathlib.Path(__file__).with_name("piper_human_approved_inference.py")
SPEC = importlib.util.spec_from_file_location("human_approved", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeRobot:
    def __init__(self, trip_after: int | None = None) -> None:
        self.trip_after = trip_after
        self.calls: list[dict[str, float]] = []
        self.safety_tripped = False

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.calls.append(action)
        if self.trip_after is not None and len(self.calls) >= self.trip_after:
            self.safety_tripped = True
        return action


class HumanApprovedExecutionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.actions = np.arange(70, dtype=np.float32).reshape(10, 7)

    def test_no_command_before_execution_function_is_called(self) -> None:
        robot = FakeRobot()
        _preview_only = self.actions.copy()
        self.assertEqual(robot.calls, [])

    def test_only_approved_prefix_is_sent(self) -> None:
        robot = FakeRobot()
        sent, tripped = MODULE.execute_robot_prefix(
            robot,
            self.actions[:4],
            30.0,
            sleep_fn=lambda _seconds: None,
            clock_fn=lambda: 0.0,
        )
        self.assertFalse(tripped)
        self.assertEqual(len(robot.calls), 4)
        np.testing.assert_array_equal(sent, self.actions[:4])

    def test_effort_trip_discards_remaining_prefix(self) -> None:
        robot = FakeRobot(trip_after=3)
        sent, tripped = MODULE.execute_robot_prefix(
            robot,
            self.actions,
            30.0,
            sleep_fn=lambda _seconds: None,
            clock_fn=lambda: 0.0,
        )
        self.assertTrue(tripped)
        self.assertEqual(len(robot.calls), 3)
        # 세 번째 호출에서 trip되어 그 action은 실제 set_action까지 도달하지 않는다.
        self.assertEqual(len(sent), 2)

    def test_normal_segments_keep_and_execute_the_remaining_chunk(self) -> None:
        robot = FakeRobot()
        sent_parts = []
        for start in range(0, len(self.actions), 4):
            sent, tripped = MODULE.execute_robot_prefix(
                robot,
                self.actions[start : start + 4],
                30.0,
                sleep_fn=lambda _seconds: None,
                clock_fn=lambda: 0.0,
            )
            self.assertFalse(tripped)
            sent_parts.append(sent)

        np.testing.assert_array_equal(np.concatenate(sent_parts), self.actions)
        self.assertEqual(len(robot.calls), len(self.actions))

    def test_already_tripped_robot_receives_nothing(self) -> None:
        robot = FakeRobot()
        robot.safety_tripped = True
        sent, tripped = MODULE.execute_robot_prefix(
            robot,
            self.actions,
            30.0,
            sleep_fn=lambda _seconds: None,
        )
        self.assertTrue(tripped)
        self.assertEqual(robot.calls, [])
        self.assertEqual(sent.shape, (0, 7))


class LiveCameraPreprocessTest(unittest.TestCase):
    def setUp(self) -> None:
        y, x = np.indices((720, 1280))
        self.frame = np.stack(
            [
                (x % 256).astype(np.uint8),
                (y % 256).astype(np.uint8),
                ((x + y) % 256).astype(np.uint8),
            ],
            axis=2,
        )
        self.camera_keys = [
            "observation.images.top",
            "observation.images.wrist",
        ]
        self.crops = {
            "top": MODULE.CameraCrop(280, 0, 720),
            "wrist": MODULE.CameraCrop(280, 0, 720),
        }

    def test_exact_crop_coordinates_before_resize(self) -> None:
        raw = {"top": self.frame, "wrist": self.frame.copy(), "joint1.pos": 1.0}
        processed = MODULE.preprocess_live_camera_observation(
            raw,
            self.camera_keys,
            self.crops,
            720,
        )
        np.testing.assert_array_equal(processed["top"], self.frame[:, 280:1000])
        self.assertEqual(processed["joint1.pos"], 1.0)
        self.assertEqual(raw["top"].shape, (720, 1280, 3))

    def test_training_output_shape_is_512_square(self) -> None:
        raw = {"top": self.frame, "wrist": self.frame.copy()}
        processed = MODULE.preprocess_live_camera_observation(
            raw,
            self.camera_keys,
            self.crops,
            512,
        )
        self.assertEqual(processed["top"].shape, (512, 512, 3))
        self.assertEqual(processed["wrist"].shape, (512, 512, 3))
        self.assertEqual(processed["top"].dtype, np.uint8)

    def test_out_of_bounds_crop_fails_closed(self) -> None:
        crops = dict(self.crops)
        crops["top"] = MODULE.CameraCrop(700, 0, 720)
        with self.assertRaisesRegex(ValueError, "exceeds frame"):
            MODULE.preprocess_live_camera_observation(
                {"top": self.frame, "wrist": self.frame},
                self.camera_keys,
                crops,
                512,
            )

    def test_output_size_must_match_training_feature(self) -> None:
        features = {
            key: {"shape": (512, 512, 3)}
            for key in self.camera_keys
        }
        MODULE.validate_live_camera_output_size(features, self.camera_keys, 512)
        with self.assertRaisesRegex(ValueError, "training shape"):
            MODULE.validate_live_camera_output_size(
                features,
                self.camera_keys,
                448,
            )


if __name__ == "__main__":
    unittest.main()
