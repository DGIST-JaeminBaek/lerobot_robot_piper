from __future__ import annotations

"""UGRP recording scripts 공통 설정 로더"""

import os
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = REPO_DIR / "configs" / "recording.env"


def _strip_quotes(value: str) -> str:
    """env 값 양끝 따옴표 제거"""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_env_file(path: str | Path | None = None) -> dict[str, str]:
    """recording.env 형식 파일 로드"""
    env_path = Path(path or os.environ.get("ENV_FILE", DEFAULT_ENV_FILE))
    if not env_path.exists():
        raise FileNotFoundError(
            f"Missing env file: {env_path}\n"
            "Copy configs/recording.env.example to configs/recording.env first."
        )

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = _strip_quotes(value)

    return values


def env_value(values: dict[str, str], key: str, default: str) -> str:
    """환경변수 우선, env 파일 다음, 기본값 마지막"""
    return os.environ.get(key) or values.get(key) or default


def env_bool(values: dict[str, str], key: str, default: bool) -> bool:
    """true/false 계열 문자열을 bool로 변환"""
    raw = env_value(values, key, "true" if default else "false")
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(values: dict[str, str], key: str, default: int) -> int:
    """정수 설정값 변환"""
    return int(env_value(values, key, str(default)))


def camera_args(values: dict[str, str]) -> list[str]:
    """Piper follower 카메라 CLI 인자 생성"""
    camera_type = env_value(values, "CAMERA_TYPE", "opencv").lower()
    top_cam_type = env_value(values, "TOP_CAM_TYPE", camera_type).lower()
    wrist_cam_type = env_value(values, "WRIST_CAM_TYPE", camera_type).lower()
    # 빈 값은 카메라 비활성화로 유지
    top_cam = values.get("TOP_CAM", "0")
    wrist_cam = values.get("WRIST_CAM", "1")
    width = env_value(values, "CAM_WIDTH", "640")
    height = env_value(values, "CAM_HEIGHT", "480")
    fps = env_value(values, "FPS", "30")
    realsense_use_depth = env_value(values, "REALSENSE_USE_DEPTH", "false")
    realsense_warmup_s = env_value(values, "REALSENSE_WARMUP_S", "5.0")
    camera_connect_warmup = env_value(values, "CAMERA_CONNECT_WARMUP", "false")
    camera_post_connect_wait_s = env_value(values, "CAMERA_POST_CONNECT_WAIT_S", "2.0")
    top_realsense_use_depth = env_value(values, "TOP_REALSENSE_USE_DEPTH", realsense_use_depth)
    wrist_realsense_use_depth = env_value(values, "WRIST_REALSENSE_USE_DEPTH", realsense_use_depth)

    return [
        f"--robot.camera_type={camera_type}",
        f"--robot.top_cam_type={top_cam_type}",
        f"--robot.wrist_cam_type={wrist_cam_type}",
        f"--robot.top_cam={top_cam}",
        f"--robot.wrist_cam={wrist_cam}",
        f"--robot.cam_width={width}",
        f"--robot.cam_height={height}",
        f"--robot.camera_fps={fps}",
        f"--robot.realsense_use_depth={realsense_use_depth}",
        f"--robot.realsense_warmup_s={realsense_warmup_s}",
        f"--robot.camera_connect_warmup={camera_connect_warmup}",
        f"--robot.camera_post_connect_wait_s={camera_post_connect_wait_s}",
        f"--robot.top_realsense_use_depth={top_realsense_use_depth}",
        f"--robot.wrist_realsense_use_depth={wrist_realsense_use_depth}",
    ]


def action_offset_args(values: dict[str, str]) -> list[str]:
    """Piper follower 시작 자세 보정 CLI 인자 생성"""
    # shell 스크립트와 동일한 기본값 유지
    return [
        f"--robot.park_on_connect={str(env_bool(values, 'PARK_ON_CONNECT', False)).lower()}",
        f"--robot.use_action_offset={str(env_bool(values, 'USE_ACTION_OFFSET', True)).lower()}",
        f"--robot.use_manual_action_offset={str(env_bool(values, 'USE_MANUAL_ACTION_OFFSET', False)).lower()}",
        f"--robot.action_offset_report_threshold={env_value(values, 'ACTION_OFFSET_REPORT_THRESHOLD', '3.0')}",
        f"--robot.action_offset_joint1={env_value(values, 'ACTION_OFFSET_JOINT1', '0.0')}",
        f"--robot.action_offset_joint2={env_value(values, 'ACTION_OFFSET_JOINT2', '0.0')}",
        f"--robot.action_offset_joint3={env_value(values, 'ACTION_OFFSET_JOINT3', '0.0')}",
        f"--robot.action_offset_joint4={env_value(values, 'ACTION_OFFSET_JOINT4', '0.0')}",
        f"--robot.action_offset_joint5={env_value(values, 'ACTION_OFFSET_JOINT5', '0.0')}",
        f"--robot.action_offset_joint6={env_value(values, 'ACTION_OFFSET_JOINT6', '0.0')}",
        f"--robot.action_offset_gripper={env_value(values, 'ACTION_OFFSET_GRIPPER', '0.0')}",
    ]


def print_command(command: list[str]) -> None:
    """실행 전 사람이 복사 가능한 형태로 명령 출력"""
    print("Command:")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))
