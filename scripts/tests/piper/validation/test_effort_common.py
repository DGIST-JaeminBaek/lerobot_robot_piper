"""effort CLI가 공유하는 schema·오염 판정의 순수 로직 테스트."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

SCRIPTS_DIR = next(
    parent
    for parent in pathlib.Path(__file__).resolve().parents
    if (parent / "0__launch_gui.sh").is_file()
)
VALIDATION_DIR = SCRIPTS_DIR / "piper" / "validation"
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

from effort_common import effort_schema, is_smooth_start_contaminated  # noqa: E402


def test_effort_schema_finds_each_observation_group():
    schema = effort_schema(
        {"features": {"observation.state": {"names": ["joint1.pos", "joint1.effort", "joint1.vel"]}}}
    )
    assert schema is not None
    assert schema.position_indices == [0]
    assert schema.effort_indices == [1]
    assert schema.velocity_indices == [2]


def test_effort_schema_returns_none_without_observation_state():
    assert effort_schema({"features": {}}) is None


def test_smooth_start_linear_ramp_is_contaminated():
    ramp = np.linspace(-100.0, 2.0, 100)[:, None].repeat(7, axis=1)
    tail = np.random.RandomState(0).randn(30, 7)
    assert is_smooth_start_contaminated(np.vstack([ramp, tail]))


def test_short_or_variable_effort_is_not_contaminated():
    real = np.random.RandomState(1).randn(200, 7)
    assert not is_smooth_start_contaminated(real)
    assert not is_smooth_start_contaminated(real[:50])
