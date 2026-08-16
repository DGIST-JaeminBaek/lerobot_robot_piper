#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from synthetic.preview.plots import PlotError, save_eef_plot, save_joint_plot


class SaveEefPlotTest(unittest.TestCase):
    def test_writes_a_nonempty_png(self) -> None:
        poses = np.tile(
            np.asarray([[100.0, 0.0, 50.0, 0.0, 0.0, 0.0]]), (10, 1)
        ) + np.arange(10).reshape(-1, 1) * 0.1
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "eef.png"
            save_eef_plot(poses, output_path)
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_wrong_shape_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(PlotError, "shape"):
                save_eef_plot(np.zeros((5, 3)), Path(tmp) / "eef.png")


class SaveJointPlotTest(unittest.TestCase):
    def test_writes_a_nonempty_png(self) -> None:
        joints = np.zeros((8, 6))
        action = np.zeros((8, 7))
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "joint.png"
            save_joint_plot(joints, action, output_path)
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_mismatched_frame_counts_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(PlotError, "frame counts differ"):
                save_joint_plot(np.zeros((5, 6)), np.zeros((4, 7)), Path(tmp) / "joint.png")

    def test_wrong_action_shape_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(PlotError, "shape"):
                save_joint_plot(np.zeros((5, 6)), np.zeros((5, 6)), Path(tmp) / "joint.png")


if __name__ == "__main__":
    unittest.main()
