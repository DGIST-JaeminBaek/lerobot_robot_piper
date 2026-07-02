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


def _camera_entry(
    name: str,
    camera_type: str,
    camera_value: str,
    width: str,
    height: str,
    fps: str,
    use_depth: str,
) -> str:
    """LeRobot CLI용 단일 camera config 문자열 생성"""
    camera_type = camera_type.lower()
    if camera_type == "opencv":
        return (
            f"{name}: "
            f"{{type: opencv, index_or_path: {camera_value}, width: {width}, height: {height}, fps: {fps}}}"
        )

    if camera_type in {"intelrealsense", "realsense"}:
        return (
            f"{name}: "
            "{"
            f'type: intelrealsense, serial_number_or_name: "{camera_value}", '
            f"width: {width}, height: {height}, fps: {fps}, use_depth: {use_depth}"
            "}"
        )

    raise ValueError(f"Unsupported camera type '{camera_type}'. Use opencv or intelrealsense.")


def camera_config(values: dict[str, str]) -> str:
    """LeRobot CLI용 top/wrist camera config 문자열 생성"""
    camera_type = env_value(values, "CAMERA_TYPE", "opencv").lower()
    top_cam_type = env_value(values, "TOP_CAM_TYPE", camera_type).lower()
    wrist_cam_type = env_value(values, "WRIST_CAM_TYPE", camera_type).lower()
    top_cam = env_value(values, "TOP_CAM", "0")
    wrist_cam = env_value(values, "WRIST_CAM", "1")
    width = env_value(values, "CAM_WIDTH", "640")
    height = env_value(values, "CAM_HEIGHT", "480")
    fps = env_value(values, "FPS", "30")
    realsense_use_depth = env_value(values, "REALSENSE_USE_DEPTH", "false")
    top_realsense_use_depth = env_value(values, "TOP_REALSENSE_USE_DEPTH", realsense_use_depth)
    wrist_realsense_use_depth = env_value(values, "WRIST_REALSENSE_USE_DEPTH", realsense_use_depth)

    top_config = _camera_entry("top", top_cam_type, top_cam, width, height, fps, top_realsense_use_depth)
    wrist_config = _camera_entry(
        "wrist", wrist_cam_type, wrist_cam, width, height, fps, wrist_realsense_use_depth
    )

    return (
        "{ "
        f"{top_config}, "
        f"{wrist_config}"
        "}"
    )


def print_command(command: list[str]) -> None:
    """실행 전 사람이 복사 가능한 형태로 명령 출력"""
    print("Command:")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))
