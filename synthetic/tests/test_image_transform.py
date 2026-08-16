#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from synthetic.preprocessing.image_transform import (
    is_inside_crop,
    is_inside_model_bounds,
    is_inside_raw_bounds,
    model_to_raw_points,
    raw_point_visibility,
    raw_to_model_points,
    resolve_geometry,
    transform_image,
)
from synthetic.preprocessing.profiles import (
    CropRegion,
    ImageProfile,
    PreprocessingError,
    ResizeSpec,
    SourceShape,
    load_profile,
    save_profile,
    smolvla_v1_profile,
)


def _stretch_profile() -> ImageProfile:
    return ImageProfile(
        name="test_stretch",
        source=SourceShape(width=200, height=100),
        crop=CropRegion(x=50, y=20, width=100, height=60),
        resize=ResizeSpec(mode="stretch", width=50, height=30),
    )


def _fit_profile() -> ImageProfile:
    # crop is 200x60; fitting into 100x100 is limited by width
    # (100/200=0.5 vs height 100/60=1.667), so scale=0.5 and the fitted
    # output is 100x30 with no padding.
    return ImageProfile(
        name="test_fit",
        source=SourceShape(width=200, height=100),
        crop=CropRegion(x=0, y=20, width=200, height=60),
        resize=ResizeSpec(mode="fit", width=100, height=100),
    )


def _letterbox_profile() -> ImageProfile:
    # crop is 100x50; resize target 100x100 -> scale=min(1.0,2.0)=1.0 ->
    # fitted 100x50, padded top/bottom by exactly 25px each.
    return ImageProfile(
        name="test_letterbox",
        source=SourceShape(width=200, height=100),
        crop=CropRegion(x=30, y=10, width=100, height=50),
        resize=ResizeSpec(mode="letterbox", width=100, height=100, pad_value=(7, 8, 9)),
    )


class RoundTripTest(unittest.TestCase):
    def _assert_round_trip(self, profile: ImageProfile, raw_points) -> None:
        model_points = raw_to_model_points(raw_points, profile)
        recovered = model_to_raw_points(model_points, profile)
        np.testing.assert_allclose(recovered, raw_points, rtol=0, atol=1e-9)

    def _corner_and_center_points(self, profile: ImageProfile):
        crop = profile.crop
        eps = 1e-6
        return [
            [crop.x, crop.y],
            [crop.x + crop.width - eps, crop.y],
            [crop.x + crop.width - eps, crop.y + crop.height - eps],
            [crop.x, crop.y + crop.height - eps],
            [crop.x + crop.width / 2, crop.y + crop.height / 2],
        ]

    def test_stretch_corners_and_center_round_trip(self) -> None:
        profile = _stretch_profile()
        self._assert_round_trip(profile, self._corner_and_center_points(profile))

    def test_fit_corners_and_center_round_trip(self) -> None:
        profile = _fit_profile()
        self._assert_round_trip(profile, self._corner_and_center_points(profile))

    def test_letterbox_corners_and_center_round_trip(self) -> None:
        profile = _letterbox_profile()
        self._assert_round_trip(profile, self._corner_and_center_points(profile))

    def test_smolvla_profile_reproduces_current_checkpoint_numbers(self) -> None:
        profile = smolvla_v1_profile()
        self.assertEqual(profile.source.width, 1280)
        self.assertEqual(profile.source.height, 720)
        self.assertEqual((profile.crop.x, profile.crop.y), (280, 0))
        self.assertEqual((profile.crop.width, profile.crop.height), (720, 720))
        self.assertEqual((profile.resize.width, profile.resize.height), (512, 512))
        self.assertEqual(profile.resize.mode, "stretch")
        geometry = resolve_geometry(profile)
        self.assertEqual((geometry.output_width, geometry.output_height), (512, 512))
        self.assertAlmostEqual(geometry.scale_x, 512 / 720)
        self.assertAlmostEqual(geometry.scale_y, 512 / 720)
        self._assert_round_trip(profile, self._corner_and_center_points(profile))


class CropBoundaryTest(unittest.TestCase):
    def test_right_bottom_edge_is_exclusive(self) -> None:
        profile = _stretch_profile()
        crop = profile.crop
        on_edge = is_inside_crop(
            [[crop.x + crop.width, crop.y], [crop.x, crop.y + crop.height]],
            profile,
        )
        self.assertFalse(on_edge.any())
        one_px_inside = is_inside_crop(
            [[crop.x + crop.width - 1, crop.y], [crop.x, crop.y + crop.height - 1]],
            profile,
        )
        self.assertTrue(one_px_inside.all())

    def test_point_outside_crop_is_rejected_in_strict_mode(self) -> None:
        profile = _stretch_profile()
        with self.assertRaisesRegex(PreprocessingError, "outside the crop region"):
            raw_to_model_points([[0.0, 0.0]], profile)

    def test_non_strict_mode_returns_out_of_model_coordinate(self) -> None:
        profile = _stretch_profile()
        model_point = raw_to_model_points([[0.0, 0.0]], profile, strict=False)
        self.assertFalse(is_inside_model_bounds(model_point, profile).all())

    def test_raw_point_visibility_flags_points_outside_crop(self) -> None:
        profile = _stretch_profile()
        crop = profile.crop
        visibility = raw_point_visibility(
            [[0.0, 0.0], [crop.x + 1, crop.y + 1]],
            profile,
        )
        np.testing.assert_array_equal(visibility, [False, True])

    def test_is_inside_raw_bounds(self) -> None:
        profile = _stretch_profile()
        visibility = is_inside_raw_bounds(
            [[-1.0, 0.0], [0.0, 0.0], [199.0, 99.0], [200.0, 99.0]],
            profile,
        )
        np.testing.assert_array_equal(visibility, [False, True, True, False])


class ScalingTest(unittest.TestCase):
    def test_stretch_scale_matches_independent_axis_ratios(self) -> None:
        profile = _stretch_profile()
        geometry = resolve_geometry(profile)
        self.assertAlmostEqual(geometry.scale_x, 50 / 100)
        self.assertAlmostEqual(geometry.scale_y, 30 / 60)
        self.assertEqual((geometry.output_width, geometry.output_height), (50, 30))

    def test_fit_scale_is_limited_by_the_tighter_axis(self) -> None:
        profile = _fit_profile()
        geometry = resolve_geometry(profile)
        self.assertAlmostEqual(geometry.scale_x, 0.5)
        self.assertAlmostEqual(geometry.scale_y, 0.5)
        self.assertEqual((geometry.output_width, geometry.output_height), (100, 30))

    def test_letterbox_padding_offset_is_centered(self) -> None:
        profile = _letterbox_profile()
        geometry = resolve_geometry(profile)
        self.assertEqual((geometry.resized_width, geometry.resized_height), (100, 50))
        self.assertEqual(geometry.pad_left, 0)
        self.assertEqual(geometry.pad_right, 0)
        self.assertEqual(geometry.pad_top, 25)
        self.assertEqual(geometry.pad_bottom, 25)
        self.assertEqual((geometry.output_width, geometry.output_height), (100, 100))


class VariousAspectRatioTest(unittest.TestCase):
    def test_wide_crop_letterboxed_into_square(self) -> None:
        profile = ImageProfile(
            name="wide",
            source=SourceShape(width=400, height=100),
            crop=CropRegion(x=0, y=0, width=400, height=100),
            resize=ResizeSpec(mode="letterbox", width=200, height=200),
        )
        geometry = resolve_geometry(profile)
        self.assertEqual((geometry.resized_width, geometry.resized_height), (200, 50))
        self.assertEqual(geometry.pad_top + geometry.pad_bottom, 150)
        self.assertEqual(geometry.pad_left, 0)

    def test_tall_crop_letterboxed_into_square(self) -> None:
        profile = ImageProfile(
            name="tall",
            source=SourceShape(width=100, height=400),
            crop=CropRegion(x=0, y=0, width=100, height=400),
            resize=ResizeSpec(mode="letterbox", width=200, height=200),
        )
        geometry = resolve_geometry(profile)
        self.assertEqual((geometry.resized_width, geometry.resized_height), (50, 200))
        self.assertEqual(geometry.pad_left + geometry.pad_right, 150)
        self.assertEqual(geometry.pad_top, 0)

    def test_odd_pixel_difference_splits_padding_by_floor(self) -> None:
        profile = ImageProfile(
            name="odd",
            source=SourceShape(width=101, height=100),
            crop=CropRegion(x=0, y=0, width=101, height=100),
            resize=ResizeSpec(mode="letterbox", width=101, height=103),
        )
        geometry = resolve_geometry(profile)
        self.assertEqual(geometry.pad_top, 1)
        self.assertEqual(geometry.pad_bottom, 2)


class ImageAppliesSameTransformAsPointsTest(unittest.TestCase):
    def _coordinate_ramp_image(self, width: int, height: int) -> np.ndarray:
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:, :, 0] = np.arange(width, dtype=np.uint8)[None, :]
        image[:, :, 1] = np.arange(height, dtype=np.uint8)[:, None]
        return image

    def test_stretch_image_pixel_matches_point_transform(self) -> None:
        profile = _stretch_profile()
        image = self._coordinate_ramp_image(profile.source.width, profile.source.height)
        model_image = transform_image(image, profile)
        self.assertEqual(
            model_image.shape[:2][::-1],
            (profile.resize.width, profile.resize.height),
        )

        crop = profile.crop
        raw_point = [crop.x + 10.0, crop.y + 5.0]
        model_point = raw_to_model_points([raw_point], profile)[0]
        mx, my = round(model_point[0]), round(model_point[1])
        raw_pixel = image[int(raw_point[1]), int(raw_point[0])]
        model_pixel = model_image[my, mx]
        # Both channels encode raw-pixel coordinates, so a bilinearly-resized
        # model pixel must stay close to the raw pixel value at the mapped
        # location.
        np.testing.assert_allclose(
            model_pixel[:2].astype(np.float64),
            raw_pixel[:2].astype(np.float64),
            atol=2.0,
        )

    def test_letterbox_image_padding_uses_pad_value(self) -> None:
        profile = _letterbox_profile()
        image = self._coordinate_ramp_image(profile.source.width, profile.source.height)
        model_image = transform_image(image, profile)
        self.assertEqual(model_image.shape[:2], (100, 100))
        expected = np.asarray(profile.resize.pad_value, dtype=np.uint8)
        np.testing.assert_array_equal(model_image[0, 0], expected)

    def test_transform_image_rejects_shape_mismatch(self) -> None:
        profile = _stretch_profile()
        wrong_shape_image = np.zeros((10, 10, 3), dtype=np.uint8)
        with self.assertRaisesRegex(
            PreprocessingError, "does not match profile source"
        ):
            transform_image(wrong_shape_image, profile)


class RejectInvalidInputTest(unittest.TestCase):
    def test_nan_point_is_rejected(self) -> None:
        profile = _stretch_profile()
        with self.assertRaisesRegex(PreprocessingError, "NaN or Inf"):
            raw_to_model_points([[float("nan"), 0.0]], profile)

    def test_inf_point_is_rejected(self) -> None:
        profile = _stretch_profile()
        with self.assertRaisesRegex(PreprocessingError, "NaN or Inf"):
            raw_to_model_points([[float("inf"), 0.0]], profile)

    def test_negative_source_size_is_rejected(self) -> None:
        with self.assertRaisesRegex(PreprocessingError, "positive"):
            SourceShape(width=-1, height=100).validate()

    def test_zero_crop_size_is_rejected(self) -> None:
        source = SourceShape(width=100, height=100)
        with self.assertRaisesRegex(PreprocessingError, "positive"):
            CropRegion(x=0, y=0, width=0, height=10).validate(source)

    def test_crop_outside_source_bounds_is_rejected(self) -> None:
        source = SourceShape(width=100, height=100)
        with self.assertRaisesRegex(PreprocessingError, "source bounds"):
            CropRegion(x=50, y=0, width=60, height=10).validate(source)

    def test_unknown_resize_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(PreprocessingError, "resize mode"):
            ResizeSpec(mode="bogus", width=10, height=10).validate()


class SerializationTest(unittest.TestCase):
    def test_profile_json_round_trip(self) -> None:
        profile = _letterbox_profile()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            save_profile(path, profile, overwrite=False)
            loaded = load_profile(path)
        self.assertEqual(loaded, profile)

    def test_profile_without_crop_uses_full_source(self) -> None:
        profile = ImageProfile(
            name="no_crop",
            source=SourceShape(width=64, height=48),
            resize=ResizeSpec(mode="stretch", width=32, height=24),
        )
        crop = profile.effective_crop()
        self.assertEqual((crop.x, crop.y, crop.width, crop.height), (0, 0, 64, 48))

    def test_load_profile_rejects_wrong_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text('{"format_version": 1, "type": "something_else"}')
            with self.assertRaisesRegex(PreprocessingError, "expected type"):
                load_profile(path)


if __name__ == "__main__":
    unittest.main()
