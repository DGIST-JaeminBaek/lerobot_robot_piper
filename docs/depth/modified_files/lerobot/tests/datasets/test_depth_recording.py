#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from lerobot.datasets.depth_utils import (
    DEFAULT_DEPTH_MIN,
    DepthFeature,
    dequantize_depth,
    quantize_depth,
)
from lerobot.datasets.image_writer import write_image
from lerobot.datasets.video_utils import decode_depth_video_frames, encode_video_frames


def test_default_depth_min_preserves_invalid_zero():
    raw_depth = np.array([[0, 10, 100, 1_000, 10_000]], dtype=np.uint16)

    quantized = quantize_depth(raw_depth)
    restored = dequantize_depth(quantized, output_tensor=False)

    assert DEFAULT_DEPTH_MIN == 0.0
    assert DepthFeature(1, raw_depth.shape[1]).info["video.depth_min"] == 0.0
    assert quantized[0, 0] == 0
    assert restored[0, 0, 0] == 0


def test_explicit_legacy_depth_min_keeps_previous_decoding():
    raw_depth = np.array([[0]], dtype=np.uint16)

    quantized = quantize_depth(raw_depth, depth_min=0.01)
    restored = dequantize_depth(quantized, depth_min=0.01, output_tensor=False)

    assert quantized.item() == 0
    assert restored.item() == 10


def test_depth_quantize_encode_decode_dequantize_roundtrip(tmp_path):
    rng = np.random.default_rng(42)
    raw_depth = [
        rng.integers(0, 10_001, size=(48, 64), dtype=np.uint16) for _ in range(5)
    ]
    for frame in raw_depth:
        frame[0, 0] = 0
    quantized = [quantize_depth(frame) for frame in raw_depth]

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for index, frame in enumerate(quantized):
        write_image(frame, frames_dir / f"frame-{index:06d}.png")

    video_path = tmp_path / "depth.mp4"
    encode_video_frames(
        frames_dir,
        video_path,
        fps=30,
        vcodec="hevc",
        pix_fmt="gray12le",
        crf=None,
        extra_options={"x265-params": "lossless=1"},
        overwrite=True,
    )
    decoded = decode_depth_video_frames(
        video_path,
        timestamps=[index / 30 for index in range(len(quantized))],
        tolerance_s=1e-4,
    ).numpy()[:, 0]

    for expected_codes, actual_codes in zip(quantized, decoded, strict=True):
        np.testing.assert_array_equal(actual_codes, expected_codes)
        np.testing.assert_array_equal(
            dequantize_depth(actual_codes, output_tensor=False),
            dequantize_depth(expected_codes, output_tensor=False),
        )
        assert dequantize_depth(actual_codes, output_tensor=False)[0, 0, 0] == 0
