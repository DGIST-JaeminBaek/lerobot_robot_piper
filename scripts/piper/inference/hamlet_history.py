"""Timestamped, camera-bundle history selection for HAMLET rollout.

The buffer deliberately stores post-crop/resize HWC uint8 images.  It does
not know about PyTorch or policies: that keeps main-loop work to copying a
camera bundle, and makes frame-time selection testable without a robot.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class TimestampedImageBundle:
    timestamp_s: float
    images: dict[str, np.ndarray]


@dataclass(frozen=True)
class HamletHistorySnapshot:
    """Oldest-first image slots selected for one immutable policy request."""

    frames: tuple[TimestampedImageBundle, ...]
    is_pad: np.ndarray  # (T,), True only for episode-start clamped slots.


class TimestampedImageRingBuffer:
    """Per-episode camera bundle buffer, addressed by monotonic timestamps."""

    def __init__(self, *, max_age_s: float) -> None:
        if max_age_s <= 0:
            raise ValueError(f"max_age_s must be positive, got {max_age_s}")
        self.max_age_s = float(max_age_s)
        self._frames: deque[TimestampedImageBundle] = deque()
        self._camera_names: tuple[str, ...] | None = None

    def clear(self) -> None:
        self._frames.clear()

    def append(self, timestamp_s: float, images: Mapping[str, np.ndarray]) -> TimestampedImageBundle:
        names = tuple(sorted(images))
        if not names:
            raise ValueError("HAMLET history needs at least one camera image")
        if self._camera_names is None:
            self._camera_names = names
        elif names != self._camera_names:
            raise ValueError(
                f"camera set changed inside one HAMLET episode: {names} != {self._camera_names}"
            )
        if self._frames and timestamp_s < self._frames[-1].timestamp_s:
            raise ValueError("history timestamps must be monotonic")

        copied: dict[str, np.ndarray] = {}
        for name, image in images.items():
            array = np.asarray(image)
            if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
                raise ValueError(f"history camera {name!r} must be HWC uint8 RGB, got {array.shape}/{array.dtype}")
            copied[name] = np.ascontiguousarray(array).copy()
            copied[name].setflags(write=False)
        bundle = TimestampedImageBundle(float(timestamp_s), copied)
        self._frames.append(bundle)
        self._discard_old(timestamp_s)
        return bundle

    def _discard_old(self, newest_s: float) -> None:
        # Keep one older guard frame.  It is useful when an interval target sits
        # just before the nominal retention boundary due to camera jitter.
        cutoff = newest_s - self.max_age_s
        while len(self._frames) > 1 and self._frames[1].timestamp_s < cutoff:
            self._frames.popleft()

    def snapshot(
        self,
        *,
        current_timestamp_s: float,
        delta_indices: list[int],
        fps: float,
    ) -> HamletHistorySnapshot:
        if not self._frames:
            raise RuntimeError("cannot snapshot an empty HAMLET history buffer")
        if fps <= 0:
            raise ValueError(f"history fps must be positive, got {fps}")
        if not delta_indices or delta_indices[-1] != 0 or delta_indices != sorted(delta_indices):
            raise ValueError(f"history delta indices must be sorted and end at 0, got {delta_indices}")
        if current_timestamp_s != self._frames[-1].timestamp_s:
            raise ValueError("history snapshot must be requested for the latest appended observation")

        earliest = self._frames[0]
        frames = tuple(self._frames)
        selected: list[TimestampedImageBundle] = []
        padded: list[bool] = []
        for delta in delta_indices:
            target = current_timestamp_s + float(delta) / fps
            candidate = None
            for bundle in reversed(frames):
                if bundle.timestamp_s <= target:
                    candidate = bundle
                    break
            if candidate is None:
                candidate = earliest
                padded.append(True)
            else:
                padded.append(False)
            selected.append(candidate)

        # The newest slot is exactly the observation passed to this request,
        # never an earlier camera frame selected by floating point accident.
        selected[-1] = self._frames[-1]
        padded[-1] = False
        return HamletHistorySnapshot(tuple(selected), np.asarray(padded, dtype=np.bool_))


def history_retention_s(delta_indices: list[int], fps: float, *, guard_frames: int = 2) -> float:
    """Minimum timestamp span required to select the oldest requested slot."""
    if fps <= 0:
        raise ValueError(f"history fps must be positive, got {fps}")
    if not delta_indices:
        raise ValueError("history delta indices must not be empty")
    return (abs(min(delta_indices)) + max(0, guard_frames)) / float(fps)
