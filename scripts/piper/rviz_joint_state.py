"""Piper 정규화 action을 ROS ``JointState`` 물리 단위로 변환한다.

ROS2 의존성 없이 사용할 수 있도록, calibration과 순수 변환 함수만 둔다.
각 실행 도구가 ROS node의 생성·종료와 재생 속도를 직접 관리한다.
"""

from __future__ import annotations

import math


PIPER_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
PIPER_GRIPPER_NAME = "gripper"
PIPER_MOTOR_NAMES = PIPER_JOINT_NAMES + [PIPER_GRIPPER_NAME]

PIPER_CALIBRATION_RAW = {
    "joint1": (-150_000, 150_000),
    "joint2": (0, 180_000),
    "joint3": (-170_000, 0),
    "joint4": (-100_000, 100_000),
    "joint5": (-65_000, 65_000),
    "joint6": (-100_000, 130_000),
    "gripper": (0, 68_000),
}


def piper_normalized_to_physical(motor: str, value: float) -> float:
    """정규화 Piper action을 ROS JointState 단위(rad / m)로 바꾼다."""
    minimum, maximum = PIPER_CALIBRATION_RAW[motor]
    if motor == PIPER_GRIPPER_NAME:
        bounded = min(100.0, max(0.0, value))
        return ((bounded / 100.0) * (maximum - minimum) + minimum) / 1_000_000.0
    bounded = min(100.0, max(-100.0, value))
    raw = ((bounded + 100.0) / 200.0) * (maximum - minimum) + minimum
    return math.radians(raw / 1000.0)
