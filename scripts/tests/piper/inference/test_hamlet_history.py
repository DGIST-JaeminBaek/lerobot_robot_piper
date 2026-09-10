"""HAMLET real-history selection/preprocessing tests (no camera or robot)."""
from __future__ import annotations

from types import SimpleNamespace
import pathlib
import sys

import numpy as np
import pytest
import torch

SCRIPTS_DIR = next(parent for parent in pathlib.Path(__file__).resolve().parents if (parent / "0__launch_gui.sh").is_file())
INFERENCE_DIR = SCRIPTS_DIR / "piper" / "inference"
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

from hamlet_history import TimestampedImageRingBuffer, history_retention_s  # noqa: E402
import inference_runtime as runtime  # noqa: E402


def _image(value: int) -> np.ndarray:
    return np.full((2, 2, 3), value, dtype=np.uint8)


def test_timestamped_history_uses_training_fps_offsets_not_policy_call_count():
    buf = TimestampedImageRingBuffer(max_age_s=1.0)
    for index in range(4):
        buf.append(index / 30.0, {"top": _image(index)})

    history = buf.snapshot(current_timestamp_s=3 / 30.0, delta_indices=[-3, -2, -1, 0], fps=30)

    assert [int(frame.images["top"][0, 0, 0]) for frame in history.frames] == [0, 1, 2, 3]
    assert history.is_pad.tolist() == [False, False, False, False]


def test_cold_start_repeats_first_frame_and_marks_only_padded_slots():
    buf = TimestampedImageRingBuffer(max_age_s=1.0)
    buf.append(0.0, {"top": _image(9)})

    history = buf.snapshot(current_timestamp_s=0.0, delta_indices=[-3, -2, -1, 0], fps=30)

    assert [int(frame.images["top"][0, 0, 0]) for frame in history.frames] == [9, 9, 9, 9]
    assert history.is_pad.tolist() == [True, True, True, False]


def test_selection_never_looks_into_the_future_and_history_images_are_immutable():
    buf = TimestampedImageRingBuffer(max_age_s=1.0)
    source = _image(1)
    buf.append(0.0, {"top": source})
    buf.append(0.04, {"top": _image(2)})
    buf.append(0.08, {"top": _image(3)})
    source[:] = 99  # append copied it; caller mutation cannot alter a request.

    history = buf.snapshot(current_timestamp_s=0.08, delta_indices=[-2, -1, 0], fps=30)

    assert [int(frame.images["top"][0, 0, 0]) for frame in history.frames] == [1, 2, 3]
    assert history.frames[0].images["top"].flags.writeable is False


def test_retention_covers_oldest_slot_plus_guard_frames():
    assert history_retention_s([-9, -6, -3, 0], 30, guard_frames=2) == pytest.approx(11 / 30)


def _policy(normalization="identity"):
    return SimpleNamespace(
        config=SimpleNamespace(
            hamlet_enabled=True,
            memory_window=2,
            normalization_mapping={"VISUAL": normalization},
            image_features={"observation.images.top": object()},
        )
    )


def test_history_preprocessor_reuses_current_slot_and_stacks_past(monkeypatch):
    buf = TimestampedImageRingBuffer(max_age_s=1.0)
    buf.append(0.0, {"top": _image(10)})
    buf.append(1 / 30, {"top": _image(20)})
    snapshot = buf.snapshot(current_timestamp_s=1 / 30, delta_indices=[-1, 0], fps=30)
    current_raw = {"top": _image(20), "joint1.pos": 0.0}
    current = {"observation.images.top": torch.full((1, 3, 2, 2), 20 / 255)}

    def fake_prepare(**kwargs):
        value = float(kwargs["raw_observation"]["top"][0, 0, 0]) / 255
        return {"observation.images.top": torch.full((1, 3, 2, 2), value)}

    monkeypatch.setattr(runtime, "_prepare_policy_observation", fake_prepare)
    history = runtime.prepare_hamlet_rollout_history(
        snapshot=snapshot,
        current_raw_observation=current_raw,
        current_observation=current,
        dataset_features={"observation.images.top": {"shape": [2, 2, 3]}},
        policy=_policy(),
        preprocessor=object(),
        device=torch.device("cpu"),
        task="test",
    )

    image = history["images"]["observation.images.top"]
    assert image.shape == (1, 2, 3, 2, 2)
    assert image[:, 0].mean().item() == pytest.approx(10 / 255)
    assert torch.equal(image[:, -1], current["observation.images.top"])
    assert history["is_pad"].tolist() == [[False, False]]


def test_history_preprocessor_fails_fast_for_nonidentity_visuals():
    with pytest.raises(ValueError, match="VISUAL normalization=IDENTITY"):
        runtime.validate_hamlet_real_history_policy(_policy("mean_std"))


def test_legacy_hamlet_policy_type_enables_real_history_without_new_flag():
    policy = _policy()
    policy.config.hamlet_enabled = False
    policy.config.type = "smolvla_hamlet"
    assert runtime.hamlet_real_history_enabled(policy)


def test_history_preprocessor_uses_post_rename_policy_visual_key(monkeypatch):
    buf = TimestampedImageRingBuffer(max_age_s=1.0)
    buf.append(0.0, {"top": _image(10)})
    snapshot = buf.snapshot(current_timestamp_s=0.0, delta_indices=[0], fps=30)
    policy = _policy()
    policy.config.memory_window = 1
    policy.config.image_features = {"observation.images.renamed_top": object()}
    current = {"observation.images.renamed_top": torch.full((1, 3, 2, 2), 10 / 255)}

    monkeypatch.setattr(runtime, "_prepare_policy_observation", lambda **_kwargs: current)
    history = runtime.prepare_hamlet_rollout_history(
        snapshot=snapshot,
        current_raw_observation={"top": _image(10)},
        current_observation=current,
        dataset_features={"observation.images.top": {"shape": [2, 2, 3]}},
        policy=policy,
        preprocessor=object(),
        device=torch.device("cpu"),
        task="test",
    )
    assert set(history["images"]) == {"observation.images.renamed_top"}
