"""piper_smoothing_sweep가 GUI 구현에 의존하지 않는지 검증한다."""

from __future__ import annotations

import pathlib
import queue
import sys

import numpy as np

SCRIPTS_DIR = next(
    parent
    for parent in pathlib.Path(__file__).resolve().parents
    if (parent / "0__launch_gui.sh").is_file()
)
INFERENCE_DIR = SCRIPTS_DIR / "piper" / "inference"
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

import piper_smoothing_sweep as sweep  # noqa: E402
from action_smoothing import SmoothingConfig  # noqa: E402


def test_run_one_constructs_runner_settings_without_hardware(monkeypatch):
    seen = {}

    class FakeRunner:
        def __init__(self, settings, events):
            seen["settings"] = settings
            self.events = events
            self.trajectory = [np.zeros(7, dtype=np.float32)]

        def start(self):
            self.events.put((sweep.Event.FINISHED, "finished"))

        def join(self, timeout):
            return None

        def emergency_stop(self):
            raise AssertionError("정상 종료에서는 호출되지 않아야 합니다")

    monkeypatch.setattr(sweep, "InferenceRunner", FakeRunner)
    smoothing = SmoothingConfig(temporal_ensemble=True, ensemble_m=0.3)
    trajectory, logs = sweep.run_one(
        {
            "policy_path": "checkpoint",
            "dataset_root": "dataset",
            "source": "dataset",
            "apply_to_robot": False,
            "rviz": False,
            "max_steps": 1,
        },
        smoothing,
        "test",
    )

    assert trajectory.shape == (1, 7)
    assert logs == []
    assert seen["settings"].smoothing is smoothing
    assert seen["settings"].apply_to_robot is False
    assert seen["settings"].rviz is False
