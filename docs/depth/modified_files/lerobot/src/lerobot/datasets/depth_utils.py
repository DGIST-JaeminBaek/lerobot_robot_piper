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
"""Depth quantization helpers backported from LeRobot 0.6.0."""

import math
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray

from .image_writer import squeeze_single_channel

DEPTH_QUANT_BITS = 12
DEPTH_QMAX = (1 << DEPTH_QUANT_BITS) - 1

# RealSense reports invalid depth pixels as 0 mm. Keep zero inside the
# quantization range so new recordings preserve that sentinel exactly.
DEFAULT_DEPTH_MIN = 0.0
DEFAULT_DEPTH_MAX = 10.0
DEFAULT_DEPTH_SHIFT = 3.5
DEFAULT_DEPTH_USE_LOG = True
DEFAULT_DEPTH_PIX_FMT = "gray12le"

DEPTH_METER_UNIT = "m"
DEPTH_MILLIMETER_UNIT = "mm"
DEFAULT_DEPTH_UNIT = DEPTH_MILLIMETER_UNIT

MM_PER_METRE = 1000.0
_UINT16_MAX = 65535


class DepthFeature(tuple):
    """Camera feature shape carrying depth encoder metadata through recording setup."""

    def __new__(
        cls,
        height: int,
        width: int,
        *,
        depth_min: float = DEFAULT_DEPTH_MIN,
        depth_max: float = DEFAULT_DEPTH_MAX,
        shift: float = DEFAULT_DEPTH_SHIFT,
        use_log: bool = DEFAULT_DEPTH_USE_LOG,
    ):
        obj = super().__new__(cls, (height, width, 1))
        obj.info = {
            "is_depth_map": True,
            "video.is_depth_map": True,
            "video.depth_min": depth_min,
            "video.depth_max": depth_max,
            "video.shift": shift,
            "video.use_log": use_log,
            "video.codec": "hevc",
            "video.pix_fmt": DEFAULT_DEPTH_PIX_FMT,
            "video.extra_options": {"x265-params": "lossless=1"},
        }
        return obj

    def __deepcopy__(self, memo):
        info = self.info
        copied = type(self)(
            self[0],
            self[1],
            depth_min=info["video.depth_min"],
            depth_max=info["video.depth_max"],
            shift=info["video.shift"],
            use_log=info["video.use_log"],
        )
        memo[id(self)] = copied
        return copied


def infer_depth_unit(dtype: np.dtype | type) -> str:
    """Infer metres for floating-point input and millimetres for integer input."""
    return DEPTH_METER_UNIT if np.issubdtype(np.dtype(dtype), np.floating) else DEPTH_MILLIMETER_UNIT


def _validate_quant_params(depth_min: float, depth_max: float, shift: float, use_log: bool) -> None:
    if depth_max <= depth_min:
        raise ValueError(f"depth_max must be greater than depth_min, got {depth_min=} and {depth_max=}")
    if use_log and depth_min + shift <= 0:
        raise ValueError(
            "depth_min + shift must be positive for logarithmic quantization, "
            f"got {depth_min + shift}"
        )


def quantize_depth(
    depth: NDArray[np.uint16] | NDArray[np.float32] | torch.Tensor,
    depth_min: float = DEFAULT_DEPTH_MIN,
    depth_max: float = DEFAULT_DEPTH_MAX,
    shift: float = DEFAULT_DEPTH_SHIFT,
    use_log: bool = DEFAULT_DEPTH_USE_LOG,
    input_unit: Literal["auto", "m", "mm"] = "auto",
) -> NDArray[np.uint16]:
    """Quantize a raw depth map to 12-bit integer codes stored in ``uint16``."""
    if input_unit not in ("auto", DEPTH_METER_UNIT, DEPTH_MILLIMETER_UNIT):
        raise ValueError(f"input_unit must be 'auto', 'm', or 'mm', got {input_unit!r}")
    _validate_quant_params(depth_min, depth_max, shift, use_log)

    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()
    depth = squeeze_single_channel(np.asarray(depth))
    if depth.ndim != 2:
        raise ValueError(f"Depth must be a 2D single-channel image, got shape {depth.shape}")

    resolved_unit = infer_depth_unit(depth.dtype) if input_unit == "auto" else input_unit
    depth_f = depth.astype(np.float32, order="K")
    unit_scale = 1.0 if resolved_unit == DEPTH_METER_UNIT else MM_PER_METRE
    depth_min_u = np.float32(depth_min * unit_scale)
    depth_max_u = np.float32(depth_max * unit_scale)
    shift_u = np.float32(shift * unit_scale)

    if use_log:
        log_min = math.log(float(depth_min_u + shift_u))
        log_max = math.log(float(depth_max_u + shift_u))
        norm = (np.log(depth_f + shift_u) - log_min) / (log_max - log_min)
    else:
        norm = (depth_f - depth_min_u) / (depth_max_u - depth_min_u)

    return np.rint(norm * DEPTH_QMAX).clip(0, DEPTH_QMAX).astype(np.uint16, copy=False)


def _remove_channel_axis(array: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if array.ndim < 3:
        return array
    if array.shape[-3] == 1:
        return array.squeeze(-3)
    if array.shape[-1] == 1:
        return array.squeeze(-1)
    raise ValueError(f"Depth input must have one channel, got shape {tuple(array.shape)}")


def dequantize_depth(
    quantized: NDArray[np.uint16] | torch.Tensor,
    depth_min: float = DEFAULT_DEPTH_MIN,
    depth_max: float = DEFAULT_DEPTH_MAX,
    shift: float = DEFAULT_DEPTH_SHIFT,
    use_log: bool = DEFAULT_DEPTH_USE_LOG,
    output_unit: Literal["m", "mm"] = DEFAULT_DEPTH_UNIT,
    output_tensor: bool = True,
    output_channel_last: bool = False,
) -> NDArray[np.uint16] | NDArray[np.float32] | torch.Tensor:
    """Invert :func:`quantize_depth` using the same quantizer parameters."""
    if output_unit not in (DEPTH_METER_UNIT, DEPTH_MILLIMETER_UNIT):
        raise ValueError(f"output_unit must be 'm' or 'mm', got {output_unit!r}")
    _validate_quant_params(depth_min, depth_max, shift, use_log)

    depth_min_m = float(depth_min)
    depth_max_m = float(depth_max)
    shift_m = float(shift)
    if use_log:
        log_min = math.log(depth_min_m + shift_m)
        log_max = math.log(depth_max_m + shift_m)
        scale = (log_max - log_min) / DEPTH_QMAX
        offset = log_min
    else:
        scale = (depth_max_m - depth_min_m) / DEPTH_QMAX
        offset = depth_min_m

    if isinstance(quantized, torch.Tensor):
        quantized = _remove_channel_axis(quantized)
        buf = quantized.to(dtype=torch.float32, copy=True)
        buf.mul_(scale).add_(offset)
        if use_log:
            buf.exp_().sub_(shift_m)
        buf.clamp_(depth_min_m, depth_max_m)
        buf.unsqueeze_(-1) if output_channel_last else buf.unsqueeze_(-3)
        if output_unit == DEPTH_MILLIMETER_UNIT:
            buf.mul_(MM_PER_METRE).round_().clamp_(0.0, _UINT16_MAX)
        return buf if output_tensor else buf.cpu().numpy().astype(
            np.uint16 if output_unit == DEPTH_MILLIMETER_UNIT else np.float32,
            copy=False,
        )

    array = _remove_channel_axis(np.asarray(quantized))
    buf = np.empty(array.shape, dtype=np.float32)
    np.multiply(array, scale, out=buf)
    np.add(buf, offset, out=buf)
    if use_log:
        np.exp(buf, out=buf)
        np.subtract(buf, shift_m, out=buf)
    np.clip(buf, depth_min_m, depth_max_m, out=buf)
    buf = np.expand_dims(buf, axis=-1 if output_channel_last else -3)
    if output_unit == DEPTH_MILLIMETER_UNIT:
        np.multiply(buf, MM_PER_METRE, out=buf)
        np.rint(buf, out=buf)
        np.clip(buf, 0.0, _UINT16_MAX, out=buf)
    if output_tensor:
        return torch.from_numpy(buf)
    return buf.astype(np.uint16 if output_unit == DEPTH_MILLIMETER_UNIT else np.float32, copy=False)
