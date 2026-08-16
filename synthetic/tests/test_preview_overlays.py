#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

from synthetic.preprocessing.profiles import CropRegion, ImageProfile, ResizeSpec, SourceShape
from synthetic.preview.overlays import draw_model_input_overlay, draw_raw_board_overlay

_IDENTITY_HOMOGRAPHY = np.eye(3)


def _profile_with_crop() -> ImageProfile:
    return ImageProfile(
        name="test",
        source=SourceShape(width=200, height=200),
        crop=CropRegion(x=0, y=0, width=200, height=200),
        resize=ResizeSpec(mode="stretch", width=100, height=100),
    )


class DrawRawBoardOverlayTest(unittest.TestCase):
    def test_overlay_keeps_raw_frame_shape(self) -> None:
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        board_xy_path = np.asarray([[20.0, 20.0], [180.0, 20.0], [180.0, 180.0]])
        overlay = draw_raw_board_overlay(
            frame, board_xy_path, board_to_image_homography=_IDENTITY_HOMOGRAPHY
        )
        self.assertEqual(overlay.shape, frame.shape)
        self.assertFalse(np.array_equal(overlay, frame))  # something was drawn

    def test_does_not_mutate_the_input_frame(self) -> None:
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        board_xy_path = np.asarray([[20.0, 20.0], [180.0, 180.0]])
        draw_raw_board_overlay(frame, board_xy_path, board_to_image_homography=_IDENTITY_HOMOGRAPHY)
        np.testing.assert_array_equal(frame, np.zeros((200, 200, 3), dtype=np.uint8))


class DrawModelInputOverlayTest(unittest.TestCase):
    def test_overlay_has_model_output_shape(self) -> None:
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        board_xy_path = np.asarray([[20.0, 20.0], [100.0, 100.0], [180.0, 20.0]])
        profile = _profile_with_crop()
        overlay = draw_model_input_overlay(
            frame,
            board_xy_path,
            board_to_image_homography=_IDENTITY_HOMOGRAPHY,
            image_profile=profile,
        )
        self.assertEqual(overlay.shape[:2], (profile.resize.height, profile.resize.width))

    def test_path_partially_outside_crop_is_still_handled(self) -> None:
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        # -50,-50 is outside the raw source/crop entirely.
        board_xy_path = np.asarray([[-50.0, -50.0], [100.0, 100.0]])
        overlay = draw_model_input_overlay(
            frame,
            board_xy_path,
            board_to_image_homography=_IDENTITY_HOMOGRAPHY,
            image_profile=_profile_with_crop(),
        )
        self.assertEqual(overlay.shape[:2], (100, 100))


if __name__ == "__main__":
    unittest.main()
