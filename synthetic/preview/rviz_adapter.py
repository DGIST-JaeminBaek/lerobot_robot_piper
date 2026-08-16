#!/usr/bin/env python3
"""Mock RViz `JointState` messages, matching `piper_infer_preview.py`'s wire format.

`scripts/tools/piper_infer_preview.py`'s `run_rviz()`/`InferPreviewNode.publish_frame()`
builds `sensor_msgs.msg.JointState` with `name = JOINT_NAMES + [GRIPPER_NAME]` and
`position` in physical units (joint radians, gripper meters). This module
reproduces that exact shape as a plain dataclass so a sequence can be
generated and inspected without importing `rclpy`/`sensor_msgs` or opening
RViz -- required for this offline stage. Publishing to a real `rclpy` node
is a separate adapter left for the (not-yet-in-scope) execution stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from synthetic.kinematics.action_conversion import (
    GRIPPER_CLOSED_MM,
    GRIPPER_FULLY_OPEN_MM,
    JOINT_ORDER,
)

JOINT_STATE_NAMES = (*JOINT_ORDER, "gripper")


class RvizAdapterError(ValueError):
    """Raised when mock JointState input is invalid."""


@dataclass(frozen=True)
class MockJointStateMessage:
    """Stand-in for `sensor_msgs.msg.JointState`, without any ROS dependency."""

    frame_index: int
    name: tuple[str, ...]
    position: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "name": list(self.name),
            "position": list(self.position),
        }


def build_mock_joint_state_sequence(
    joint_rad_sequence: Any,
    gripper_mm_sequence: Any,
) -> list[MockJointStateMessage]:
    """`(N, 6)` joint radians + `(N,)` gripper mm -> a sequence of mock `JointState`s."""

    joints = np.asarray(joint_rad_sequence, dtype=np.float64)
    gripper_mm = np.asarray(gripper_mm_sequence, dtype=np.float64)
    if joints.ndim != 2 or joints.shape[1] != 6:
        raise RvizAdapterError(f"joint_rad_sequence must have shape (N, 6), got {joints.shape}")
    if gripper_mm.shape != (joints.shape[0],):
        raise RvizAdapterError(
            f"gripper_mm_sequence must have shape ({joints.shape[0]},), got {gripper_mm.shape}"
        )
    if not np.isfinite(joints).all() or not np.isfinite(gripper_mm).all():
        raise RvizAdapterError("joint_rad_sequence/gripper_mm_sequence contain NaN or Inf")
    if np.any(gripper_mm < GRIPPER_CLOSED_MM) or np.any(gripper_mm > GRIPPER_FULLY_OPEN_MM):
        raise RvizAdapterError(
            f"gripper_mm_sequence must be within [{GRIPPER_CLOSED_MM}, "
            f"{GRIPPER_FULLY_OPEN_MM}]mm"
        )

    messages = []
    for frame_index in range(joints.shape[0]):
        gripper_m = float(gripper_mm[frame_index]) / 1000.0
        position = (*joints[frame_index].tolist(), gripper_m)
        messages.append(
            MockJointStateMessage(
                frame_index=frame_index,
                name=JOINT_STATE_NAMES,
                position=position,
            )
        )
    return messages
