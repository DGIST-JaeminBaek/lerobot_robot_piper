"""통합 replay player의 CLI 경계와 Piper 관절 변환을 하드웨어 없이 검증한다."""

from __future__ import annotations

import pathlib
import sys

import pytest

SCRIPTS_DIR = next(
    parent
    for parent in pathlib.Path(__file__).resolve().parents
    if (parent / "0__launch_gui.sh").is_file()
)
VALIDATION_DIR = SCRIPTS_DIR / "piper" / "validation"
PIPER_DIR = SCRIPTS_DIR / "piper"
if str(PIPER_DIR) not in sys.path:
    sys.path.insert(0, str(PIPER_DIR))
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

import piper_replay_player as player  # noqa: E402
from rviz_joint_state import piper_normalized_to_physical  # noqa: E402


def test_normal_mode_does_not_require_ros_arguments():
    args = player.parse_args(["--dataset-root", "dataset", "--episode", "2"])
    assert args.rviz is False
    assert args.joint_state_topic == "/joint_states"


def test_rviz_mode_has_explicit_joint_state_options():
    args = player.parse_args(
        ["--dataset-root", "dataset", "--rviz", "--joint-state-topic", "/replay/joints"]
    )
    assert args.rviz is True
    assert args.joint_state_topic == "/replay/joints"
    assert args.rviz_key == "action"


def test_piper_joint_conversion_preserves_expected_units():
    assert piper_normalized_to_physical("joint1", -100.0) == pytest.approx(-2.61799387799)
    assert piper_normalized_to_physical("joint1", 100.0) == pytest.approx(2.61799387799)
    assert piper_normalized_to_physical("gripper", 100.0) == pytest.approx(0.068)
