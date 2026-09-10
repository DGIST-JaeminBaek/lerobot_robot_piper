#!/usr/bin/env python3
"""SmolVLA action chunk를 사람의 승인 단위로 확인하고 선택적으로 실행하는 도구.

기본값은 dataset observation을 사용하는 RViz-only 안전 모드다. 실제 Piper 실행은
source=robot, apply_to_robot=true, 명시적 확인 문구를 모두 지정해야만 활성화된다.
preview chunk를 execute_actions 크기의 구간으로 나누고, 각 구간을 사람이 따로 승인한다.
정상 승인된 구간 뒤에도 같은 chunk의 나머지를 유지해 다음 승인 대상으로 보여준다.
모든 실물 명령은 PiperFollower.send_action()을 거치므로 실제 위치 기준
max_relative_target 제한과 effort 안전 컷오프가 그대로 적용된다.

흐름:
    dataset 또는 live robot observation
      -> action chunk 추론
      -> 범위 검사 및 preview용 max_relative_target 적용
      -> RViz 1회 재생
      -> 터미널 승인 / 재생 / 폐기 / 종료
      -> 승인 시 현재 구간만 dataset에서 전진하거나 실제 로봇에 전송
      -> 같은 chunk의 다음 구간을 다시 preview/승인
      -> 전체 chunk 완료 또는 사람이 discard한 뒤 새 observation으로 다시 추론

주요 환경변수:
    HUMAN_APPROVED_DATASET_ROOT
    HUMAN_APPROVED_EPISODE
    HUMAN_APPROVED_POLICY_PATH
    HUMAN_APPROVED_TASK
    HUMAN_APPROVED_SOURCE=dataset
    HUMAN_APPROVED_APPLY_TO_ROBOT=false
    HUMAN_APPROVED_REAL_ROBOT_CONFIRM=
    HUMAN_APPROVED_PREVIEW_ACTIONS=50
    HUMAN_APPROVED_EXECUTE_ACTIONS=10
    HUMAN_APPROVED_STALE_STATE_TOLERANCE=2.0
    HUMAN_APPROVED_PARK_ON_EXIT=true
    HUMAN_APPROVED_TOP_CROP=280,0,720
    HUMAN_APPROVED_WRIST_CROP=280,0,720
    HUMAN_APPROVED_CAMERA_OUTPUT_SIZE=512
    HUMAN_APPROVED_RVIZ=true
    HUMAN_APPROVED_SHOW_IMAGES=true
    HUMAN_APPROVED_APPROVAL_TIMEOUT_S=0
    HUMAN_APPROVED_MAX_CHUNKS=0
    HUMAN_APPROVED_REPEAT_LAST_FRAME=false
    MAX_RELATIVE_TARGET=5.0
    FPS=30

기본적으로 configs/recording.env를 읽되, 이미 export된 환경변수를 우선한다.
CLI 인자는 두 값보다 우선한다.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import select
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_ENV_FILE = REPO_ROOT / "configs" / "recording.env"
GLOBAL_LOW = np.asarray([-100.0] * 6 + [0.0], dtype=np.float32)
GLOBAL_HIGH = np.asarray([100.0] * 7, dtype=np.float32)
MOTOR_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
FALSE_VALUES = {"0", "false", "no", "off"}
TRUE_VALUES = {"1", "true", "yes", "on"}
DISABLED_VALUES = {"", "none", "null", "off", "disabled"}

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))
from inference_runtime import (  # noqa: E402
    CameraCrop,
    build_robot_from_env,
    hamlet_real_history_enabled,
    load_policy,
    make_raw_observation,
    parse_camera_crop,
    predict_chunk,
    preprocess_live_camera_observation,
    state_from_raw_observation,
    validate_hamlet_real_history_policy,
    validate_live_camera_output_size,
)
from hamlet_history import TimestampedImageRingBuffer, history_retention_s  # noqa: E402
from rviz_joint_state import piper_normalized_to_physical  # noqa: E402


@dataclass
class ChunkLog:
    chunk_id: int
    segment_id: int
    segment_start_action: int
    segment_end_action: int
    observation_frame: int
    task: str
    predicted_actions: int
    previewed_actions: int
    inference_seconds: float
    global_clamped_values: int
    relative_clamped_values: int
    clamped_values: int
    max_clamp_adjustment: float
    decision: str
    decision_seconds: float
    apply_to_robot: bool
    executed_actions: int
    stale_observation: bool
    safety_tripped: bool


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parse_optional_positive_float(value: str | None) -> float | None:
    if value is None or value.strip().lower() in DISABLED_VALUES:
        return None
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be positive or an off/none sentinel")
    return parsed


def load_env_file(path: pathlib.Path) -> None:
    """간단한 KEY=VALUE 파일을 읽는다. Shell에서 export한 값은 덮어쓰지 않는다."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)


def build_parser() -> argparse.ArgumentParser:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--env-file", type=pathlib.Path, default=DEFAULT_ENV_FILE)
    pre_args, _ = pre_parser.parse_known_args()
    load_env_file(pre_args.env_file.expanduser().resolve())

    env = os.environ
    policy_path = (
        env.get("HUMAN_APPROVED_POLICY_PATH")
        or env.get("PRETRAINED_NAME_OR_PATH")
        or env.get("POLICY_PRETRAINED_PATH")
    )
    parser = argparse.ArgumentParser(description="SmolVLA chunk별 인간 승인 preview/execution")
    parser.add_argument("--env-file", type=pathlib.Path, default=pre_args.env_file)
    parser.add_argument(
        "--dataset-root",
        type=pathlib.Path,
        default=(pathlib.Path(env["HUMAN_APPROVED_DATASET_ROOT"])
                 if env.get("HUMAN_APPROVED_DATASET_ROOT") else None),
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=int(env.get("HUMAN_APPROVED_EPISODE", "0")),
    )
    parser.add_argument("--policy-path", default=policy_path)
    parser.add_argument(
        "--policy-discover-packages-path",
        default=env.get("HUMAN_APPROVED_POLICY_DISCOVER_PACKAGES_PATH") or None,
        help="HAMLET 등 서드파티 policy 타입을 쓸 때 지정. lerobot-train의 "
        "--policy.discover_packages_path와 같은 값(예: smolvla_hamlet)을 주면 "
        "PreTrainedConfig.from_pretrained() 전에 그 패키지를 import해서 등록한다. "
        "PYTHONPATH에 해당 패키지가 있어야 함",
    )
    parser.add_argument("--task", default=env.get("HUMAN_APPROVED_TASK") or None)
    parser.add_argument("--device", default=env.get("POLICY_DEVICE", "cuda"))
    parser.add_argument(
        "--source",
        choices=("dataset", "robot"),
        default=env.get("HUMAN_APPROVED_SOURCE", "dataset").lower(),
        help="observation 출처. 기본값 dataset",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=float(env.get("FPS", "30")),
    )
    parser.add_argument(
        "--preview-actions",
        "--actions-per-chunk",
        dest="preview_actions",
        type=int,
        default=int(
            env.get(
                "HUMAN_APPROVED_PREVIEW_ACTIONS",
                env.get(
                    "HUMAN_APPROVED_ACTIONS_PER_CHUNK",
                    env.get("ACTIONS_PER_CHUNK", "50"),
                ),
            )
        ),
        help="추론 chunk 중 RViz에서 확인할 action 수",
    )
    parser.add_argument(
        "--execute-actions",
        type=int,
        default=int(env.get("HUMAN_APPROVED_EXECUTE_ACTIONS", "10")),
        help="chunk를 나눌 승인/실행 구간 크기",
    )
    parser.add_argument(
        "--sync-eval",
        type=parse_bool,
        default=parse_bool(env.get("HUMAN_APPROVED_SYNC_EVAL", "false")),
        metavar="BOOL",
        help="승인한 prefix 직후 새 observation으로 재추론하고 남은 chunk를 폐기",
    )
    parser.add_argument(
        "--max-relative-target",
        type=parse_optional_positive_float,
        default=parse_optional_positive_float(env.get("MAX_RELATIVE_TARGET", "5.0")),
        help="RViz preview에 순차 적용할 상대 action 제한; off/none으로 비활성화",
    )
    parser.add_argument(
        "--apply-to-robot",
        type=parse_bool,
        default=parse_bool(env.get("HUMAN_APPROVED_APPLY_TO_ROBOT", "false")),
        metavar="BOOL",
        help="true이면 승인된 prefix를 실제 Piper에 전송",
    )
    parser.add_argument(
        "--real-robot-confirm",
        default=env.get("HUMAN_APPROVED_REAL_ROBOT_CONFIRM", ""),
        help="실물 실행 시 I_UNDERSTAND_REAL_ROBOT 필요",
    )
    parser.add_argument(
        "--stale-state-tolerance",
        type=float,
        default=float(env.get("HUMAN_APPROVED_STALE_STATE_TOLERANCE", "2.0")),
        help="승인 직전 재측정 state가 preview 시작점과 달라져도 허용할 최대 정규화 단위",
    )
    parser.add_argument(
        "--park-on-exit",
        type=parse_bool,
        default=parse_bool(env.get("HUMAN_APPROVED_PARK_ON_EXIT", "true")),
        metavar="BOOL",
    )
    parser.add_argument(
        "--top-crop",
        type=parse_camera_crop,
        default=parse_camera_crop(env.get("HUMAN_APPROVED_TOP_CROP", "280,0,720")),
        metavar="X,Y,SIZE",
        help="Robot source TOP crop before policy input",
    )
    parser.add_argument(
        "--wrist-crop",
        type=parse_camera_crop,
        default=parse_camera_crop(env.get("HUMAN_APPROVED_WRIST_CROP", "280,0,720")),
        metavar="X,Y,SIZE",
        help="Robot source WRIST crop before policy input",
    )
    parser.add_argument(
        "--camera-output-size",
        type=int,
        default=int(env.get("HUMAN_APPROVED_CAMERA_OUTPUT_SIZE", "512")),
        help="Robot source square crop resize output",
    )
    parser.add_argument(
        "--rviz",
        type=parse_bool,
        default=parse_bool(env.get("HUMAN_APPROVED_RVIZ", "true")),
        metavar="BOOL",
    )
    parser.add_argument(
        "--show-images",
        type=parse_bool,
        default=parse_bool(env.get("HUMAN_APPROVED_SHOW_IMAGES", "true")),
        metavar="BOOL",
        help="Policy에 입력한 dataset camera frame을 OpenCV 창에 표시",
    )
    parser.add_argument(
        "--approval-timeout-s",
        type=float,
        default=float(env.get("HUMAN_APPROVED_APPROVAL_TIMEOUT_S", "0")),
        help="0이면 timeout 없음",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=int(env.get("HUMAN_APPROVED_MAX_CHUNKS", "0")),
        help="0이면 episode 끝까지",
    )
    parser.add_argument(
        "--repeat-last-frame",
        type=parse_bool,
        default=parse_bool(env.get("HUMAN_APPROVED_REPEAT_LAST_FRAME", "false")),
        metavar="BOOL",
        help="dataset source에서 episode 끝에 도달해도 멈추지 않고 마지막 frame을 "
        "계속 observation으로 재사용 (RViz-only 반복 확인용, source=dataset에서만 동작)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(env.get("HUMAN_APPROVED_SEED", "1000")),
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=(
            pathlib.Path(env["HUMAN_APPROVED_OUTPUT_DIR"])
            if env.get("HUMAN_APPROVED_OUTPUT_DIR")
            else None
        ),
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.apply_to_robot and args.source != "robot":
        raise ValueError("--apply-to-robot=true requires --source=robot")
    if args.apply_to_robot and args.real_robot_confirm != "I_UNDERSTAND_REAL_ROBOT":
        raise ValueError(
            "Real robot execution requires "
            "HUMAN_APPROVED_REAL_ROBOT_CONFIRM=I_UNDERSTAND_REAL_ROBOT"
        )
    if args.apply_to_robot:
        safety_enabled = parse_bool(os.environ.get("SAFETY_ENABLED", "true"))
        safety_mode = os.environ.get("SAFETY_ON_OVERLOAD", "park").strip().lower()
        if not safety_enabled or safety_mode != "park":
            raise ValueError(
                "Real robot execution requires SAFETY_ENABLED=true and "
                "SAFETY_ON_OVERLOAD=park"
            )
    if not args.policy_path:
        raise ValueError(
            "Policy path is required. Set HUMAN_APPROVED_POLICY_PATH or pass --policy-path."
        )
    if args.source == "dataset" and args.dataset_root is None:
        raise ValueError(
            "Dataset source requires HUMAN_APPROVED_DATASET_ROOT or --dataset-root."
        )
    if args.episode < 0:
        raise ValueError("--episode must be non-negative")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.preview_actions <= 0:
        raise ValueError("--preview-actions must be positive")
    if args.execute_actions <= 0:
        raise ValueError("--execute-actions must be positive")
    if args.execute_actions > args.preview_actions:
        raise ValueError("--execute-actions cannot exceed --preview-actions")
    if args.stale_state_tolerance < 0:
        raise ValueError("--stale-state-tolerance cannot be negative")
    if args.camera_output_size <= 0:
        raise ValueError("--camera-output-size must be positive")
    if args.approval_timeout_s < 0:
        raise ValueError("--approval-timeout-s cannot be negative")
    if args.max_chunks < 0:
        raise ValueError("--max-chunks cannot be negative")


def validate_chunk(chunk: np.ndarray) -> None:
    if chunk.ndim != 2 or chunk.shape[1] != 7 or len(chunk) == 0:
        raise ValueError(f"Invalid action chunk shape {chunk.shape}; expected (N, 7)")
    if not np.isfinite(chunk).all():
        bad = np.argwhere(~np.isfinite(chunk))
        raise ValueError(f"Action chunk contains NaN/Inf at {bad[:10].tolist()}")


def apply_global_limits(chunk: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Policy 출력을 Piper의 joint/gripper 정규화 범위로 clamp한다."""
    limited = np.clip(chunk, GLOBAL_LOW, GLOBAL_HIGH)
    adjustment = np.abs(limited - chunk)
    count = int(np.count_nonzero(adjustment > 1e-4))
    max_adjustment = float(adjustment.max(initial=0.0))
    return limited, count, max_adjustment


def apply_preview_relative_limit(
    start_state: np.ndarray,
    chunk: np.ndarray,
    max_relative_target: float | None,
) -> tuple[np.ndarray, int, float]:
    """이상적인 직전 목표 도달을 가정해 preview action을 순차 clamp한다.

    실제 로봇은 각 send_action 시점의 측정 position을 기준으로 clamp하므로 이 결과는
    preview 근사치다. 실물 실행 경로를 구현할 때는 실제로 전송될 safe action을 다시
    기록하고 비교해야 한다.
    """
    if max_relative_target is None:
        return chunk.copy(), 0, 0.0

    current = np.asarray(start_state, dtype=np.float32).copy()
    limited = np.empty_like(chunk)
    clamped_values = 0
    max_adjustment = 0.0
    for index, action in enumerate(chunk):
        safe = np.clip(
            action,
            current - max_relative_target,
            current + max_relative_target,
        )
        adjustment = np.abs(safe - action)
        clamped_values += int(np.count_nonzero(adjustment > 1e-4))
        max_adjustment = max(max_adjustment, float(adjustment.max(initial=0.0)))
        limited[index] = safe
        current = safe
    return limited, clamped_values, max_adjustment


def action_dict(action: np.ndarray) -> dict[str, float]:
    return {
        f"{motor}.pos": float(value)
        for motor, value in zip(MOTOR_NAMES, np.asarray(action).tolist(), strict=True)
    }


def execute_robot_prefix(
    robot,
    actions: np.ndarray,
    fps: float,
    *,
    after_action=None,
    sleep_fn=time.sleep,
    clock_fn=time.perf_counter,
) -> tuple[np.ndarray, bool]:
    """승인 prefix를 전송하고, 필요하면 각 제어 tick 뒤 history 관측을 저장한다.

    ``after_action``은 HAMLET real-history에서만 live camera를 읽는다. action 전송 뒤에
    호출해, 다음 정책 요청의 이미지 window가 실제 실행 중 시간 축을 보존하게 한다.
    """
    sent: list[np.ndarray] = []
    if robot.safety_tripped:
        return np.empty((0, 7), dtype=np.float32), True

    for action in actions:
        started = clock_fn()
        actual = robot.send_action(action_dict(action))
        if robot.safety_tripped:
            break
        sent.append(
            np.asarray(
                [actual[f"{motor}.pos"] for motor in MOTOR_NAMES],
                dtype=np.float32,
            )
        )
        if after_action is not None:
            after_action()
        sleep_fn(max(0.0, 1.0 / fps - (clock_fn() - started)))

    result = np.stack(sent) if sent else np.empty((0, 7), dtype=np.float32)
    return result, bool(robot.safety_tripped)


def append_hamlet_history_observation(
    history: TimestampedImageRingBuffer,
    raw_observation: dict,
    camera_keys: list[str],
    timestamp_s: float,
) -> None:
    """Store exactly the post-crop/resize RGB cameras used by the policy."""
    history.append(
        timestamp_s,
        {
            feature_key.removeprefix("observation.images."): raw_observation[
                feature_key.removeprefix("observation.images.")
            ]
            for feature_key in camera_keys
        },
    )


def wait_for_safety_parking(robot, timeout_s: float = 15.0) -> None:
    thread = getattr(robot, "_safety_park_thread", None)
    if thread is not None and thread.is_alive():
        print("[SAFETY] parking 완료를 기다립니다")
        thread.join(timeout=timeout_s)
        if thread.is_alive():
            print("[WARN] safety parking thread가 timeout 안에 끝나지 않았습니다")


class RvizChunkPreview:
    def __init__(self, topic: str = "/joint_states") -> None:
        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import JointState
        except ImportError as error:
            raise RuntimeError(
                "RViz preview requires rclpy and sensor_msgs. Source ROS2 Humble first."
            ) from error

        self.rclpy = rclpy
        self.JointState = JointState
        self.unnormalize_to_physical = piper_normalized_to_physical
        rclpy.init()
        self.node = Node("piper_human_approved_inference")
        self.publisher = self.node.create_publisher(JointState, topic, 10)
        self.topic = topic

    def publish(self, action: np.ndarray) -> None:
        message = self.JointState()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.name = MOTOR_NAMES
        message.position = [
            self.unnormalize_to_physical(name, float(value))
            for name, value in zip(MOTOR_NAMES, action)
        ]
        self.publisher.publish(message)
        self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def replay(
        self,
        start_state: np.ndarray,
        chunk: np.ndarray,
        fps: float,
        event_pump=None,
    ) -> None:
        print(f"[RVIZ] start state + {len(chunk)} action 재생 ({fps:.2f} FPS)")
        for _ in range(max(1, round(fps * 0.5))):
            self.publish(start_state)
            if event_pump is not None:
                event_pump()
            time.sleep(1.0 / fps)
        for action in chunk:
            started = time.perf_counter()
            self.publish(action)
            if event_pump is not None:
                event_pump()
            time.sleep(max(0.0, 1.0 / fps - (time.perf_counter() - started)))
        # 마지막 자세가 승인 입력 중에도 RViz에 남도록 한 번 더 publish한다.
        self.publish(chunk[-1])

    def close(self) -> None:
        self.node.destroy_node()
        self.rclpy.shutdown()


class ObservationImageViewer:
    WINDOW_NAME = "Policy input observation"

    def __init__(self) -> None:
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError(
                "Image preview requires OpenCV with GUI support (opencv-python)."
            ) from error
        self.cv2 = cv2
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, 1280, 700)

    def show(
        self,
        raw_observation: dict,
        camera_keys: list[str],
        *,
        chunk_id: int,
        observation_frame: int,
        task: str,
    ) -> None:
        panels: list[np.ndarray] = []
        for feature_key in camera_keys:
            short_key = feature_key.removeprefix("observation.images.")
            rgb = np.asarray(raw_observation[short_key])
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            if rgb.ndim == 2:
                rgb = rgb[..., None]
            if rgb.shape[2] == 1:
                bgr = np.repeat(rgb, 3, axis=2)
            else:
                bgr = rgb[:, :, :3][:, :, ::-1].copy()

            label = short_key
            self.cv2.putText(
                bgr,
                label,
                (14, 32),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                self.cv2.LINE_AA,
            )
            panels.append(bgr)

        if not panels:
            raise ValueError("Dataset has no camera features to display")

        target_height = min(panel.shape[0] for panel in panels)
        resized = []
        for panel in panels:
            scale = target_height / panel.shape[0]
            width = max(1, round(panel.shape[1] * scale))
            resized.append(
                self.cv2.resize(
                    panel,
                    (width, target_height),
                    interpolation=self.cv2.INTER_AREA,
                )
            )
        image_strip = self.cv2.hconcat(resized)
        header = np.full((62, image_strip.shape[1], 3), 24, dtype=np.uint8)
        title = (
            f"chunk {chunk_id:04d} | observation frame {observation_frame} | "
            f"task: {task}"
        )
        self.cv2.putText(
            header,
            title,
            (14, 39),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (240, 240, 240),
            2,
            self.cv2.LINE_AA,
        )
        canvas = self.cv2.vconcat([header, image_strip])
        self.cv2.imshow(self.WINDOW_NAME, canvas)
        self.pump()

    def pump(self) -> None:
        # waitKey가 OpenCV/X11 event queue도 처리하므로 승인 입력 대기 중 계속 호출한다.
        self.cv2.waitKey(1)

    def close(self) -> None:
        self.cv2.destroyWindow(self.WINDOW_NAME)
        self.cv2.waitKey(1)


def read_decision(timeout_s: float, event_pump=None) -> tuple[str, float]:
    prompt = (
        "[DECISION] a=approve current segment, r=replay, "
        "d=discard remaining chunk+reinfer, q=quit > "
    )
    started = time.monotonic()
    while True:
        print(prompt, end="", flush=True)
        deadline = started + timeout_s if timeout_s > 0 else None
        while True:
            if event_pump is not None:
                event_pump()
            wait_s = 0.05
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    print("\n[TIMEOUT] chunk discarded")
                    return "timeout", time.monotonic() - started
                wait_s = min(wait_s, remaining)
            ready, _, _ = select.select([sys.stdin], [], [], wait_s)
            if ready:
                break
        line = sys.stdin.readline()
        if line == "":
            print("\n[EOF] quit")
            return "quit", time.monotonic() - started
        decision = line.strip().lower()
        mapping = {
            "a": "approve",
            "approve": "approve",
            "r": "replay",
            "replay": "replay",
            "d": "discard",
            "discard": "discard",
            "q": "quit",
            "quit": "quit",
        }
        if decision in mapping:
            return mapping[decision], time.monotonic() - started
        print("[WARN] a, r, d, q 중 하나를 입력하세요.")


def append_jsonl(path: pathlib.Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    args = build_parser().parse_args()
    try:
        validate_args(args)
    except Exception as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2

    if args.policy_discover_packages_path:
        import importlib

        importlib.import_module(args.policy_discover_packages_path)
        print(f"[PLUGIN] imported {args.policy_discover_packages_path!r}")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.utils import get_safe_torch_device

    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir or (
        REPO_ROOT
        / "outputs"
        / "human_approved_preview"
        / f"{dataset_root.name}_ep{args.episode:04d}"
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    decision_log = output_dir / "decisions.jsonl"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 72)
    if args.apply_to_robot:
        print("[REAL ROBOT MODE] chunk를 나누어 구간별 승인 후 Piper에 전송합니다.")
        print(
            f"[REAL ROBOT MODE] preview={args.preview_actions}, "
            f"execute={args.execute_actions}, effort trip=park+terminate"
        )
    else:
        print(f"[SAFE MODE] source={args.source}, APPLY_TO_ROBOT=false")
        print("[SAFE MODE] robot.send_action()을 호출하지 않습니다.")
    print("=" * 72)
    print(f"[LOAD] dataset={dataset_root}, episode={args.episode}")
    dataset = LeRobotDataset(
        repo_id=f"local/{dataset_root.name}",
        root=dataset_root,
        episodes=[args.episode],
        video_backend="pyav",
    )
    camera_keys = list(dataset.meta.camera_keys)
    live_crops = {
        "top": args.top_crop,
        "wrist": args.wrist_crop,
    }
    if args.source == "robot":
        validate_live_camera_output_size(
            dataset.features,
            camera_keys,
            args.camera_output_size,
        )
        print(
            "[LIVE PREPROCESS] "
            f"top={args.top_crop.x},{args.top_crop.y},{args.top_crop.size}; "
            f"wrist={args.wrist_crop.x},{args.wrist_crop.y},{args.wrist_crop.size}; "
            f"output={args.camera_output_size}x{args.camera_output_size}"
        )
    print(f"[LOAD] policy={args.policy_path}, device={args.device}")
    config, policy, preprocessor, postprocessor = load_policy(
        args.policy_path,
        dataset.meta,
        args.device,
    )
    device = get_safe_torch_device(policy.config.device)
    policy.reset()

    hamlet_history: TimestampedImageRingBuffer | None = None
    hamlet_delta_indices: list[int] = []
    hamlet_fps = float(getattr(dataset.meta, "fps", args.fps))
    if hamlet_real_history_enabled(policy):
        validate_hamlet_real_history_policy(policy)
        if hamlet_fps <= 0:
            raise ValueError(f"HAMLET history requires positive dataset fps, got {hamlet_fps}")
        hamlet_delta_indices = list(policy.config.observation_delta_indices)
        hamlet_history = TimestampedImageRingBuffer(
            max_age_s=history_retention_s(hamlet_delta_indices, hamlet_fps)
        )
        print(
            "[HAMLET] real image history enabled: "
            f"fps={hamlet_fps:g}, deltas={hamlet_delta_indices}"
        )

    policy_chunk_size = int(getattr(config, "chunk_size", 0))
    if args.preview_actions > policy_chunk_size:
        print(
            f"[ERROR] preview_actions={args.preview_actions} exceeds "
            f"policy chunk_size={policy_chunk_size}",
            file=sys.stderr,
        )
        return 2

    first_item = dataset[0]
    task = args.task if args.task is not None else str(first_item.get("task", ""))
    rviz = RvizChunkPreview() if args.rviz else None
    image_viewer = ObservationImageViewer() if args.show_images else None
    robot = None
    cursor = 0
    history_dataset_cursor = 0
    chunk_id = 0
    approved_count = 0
    safety_tripped = False

    try:
        if args.source == "robot":
            print("[CONNECT] Piper follower와 camera를 연결합니다")
            robot = build_robot_from_env(args.max_relative_target)
            robot.connect()

        def capture_live_hamlet_history() -> None:
            """Store one post-action camera frame without issuing any robot command."""
            if hamlet_history is None:
                return
            live_observation = robot.get_observation()
            processed = preprocess_live_camera_observation(
                live_observation,
                camera_keys,
                live_crops,
                args.camera_output_size,
            )
            append_hamlet_history_observation(
                hamlet_history, processed, camera_keys, time.monotonic()
            )

        while (
            args.source == "robot"
            or cursor < dataset.num_frames
            or args.repeat_last_frame
        ):
            if args.max_chunks and approved_count >= args.max_chunks:
                print(f"[STOP] approved chunk limit reached: {args.max_chunks}")
                break

            if args.source == "dataset":
                # repeat_last_frame이면 cursor가 episode 끝을 넘어가도 마지막
                # frame(dataset.num_frames - 1)에 고정해서 계속 같은 observation을 준다.
                frame_idx = min(cursor, dataset.num_frames - 1)
                item = dataset[frame_idx]
                raw_observation = make_raw_observation(dataset, item)
                observation_frame = frame_idx
            else:
                live_observation = robot.get_observation()
                raw_observation = preprocess_live_camera_observation(
                    live_observation,
                    camera_keys,
                    live_crops,
                    args.camera_output_size,
                )
                observation_frame = chunk_id

            history_snapshot = None
            if hamlet_history is not None:
                if args.source == "dataset":
                    # 승인된 action 수만큼 cursor가 전진한 구간의 dataset frame도 모두
                    # 채운다. 승인 chunk의 크기가 image-history 시간축을 바꾸지 않는다.
                    while history_dataset_cursor <= observation_frame:
                        history_item = (
                            item
                            if history_dataset_cursor == observation_frame
                            else dataset[history_dataset_cursor]
                        )
                        history_raw = (
                            raw_observation
                            if history_dataset_cursor == observation_frame
                            else make_raw_observation(dataset, history_item)
                        )
                        append_hamlet_history_observation(
                            hamlet_history,
                            history_raw,
                            camera_keys,
                            history_dataset_cursor / hamlet_fps,
                        )
                        history_dataset_cursor += 1
                    history_timestamp = observation_frame / hamlet_fps
                else:
                    history_timestamp = time.monotonic()
                    append_hamlet_history_observation(
                        hamlet_history, raw_observation, camera_keys, history_timestamp
                    )
                history_snapshot = hamlet_history.snapshot(
                    current_timestamp_s=history_timestamp,
                    delta_indices=hamlet_delta_indices,
                    fps=hamlet_fps,
                )
            start_state = state_from_raw_observation(raw_observation, dataset.features)
            started = time.perf_counter()
            raw_chunk_tensor = predict_chunk(
                raw_observation=raw_observation,
                dataset_features=dataset.features,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                device=device,
                task=task,
                rollout_history=history_snapshot,
            )
            inference_seconds = time.perf_counter() - started
            raw_chunk = raw_chunk_tensor.numpy().astype(np.float32, copy=False)
            repeating_last_frame = (
                args.source == "dataset" and cursor >= dataset.num_frames
            )
            remaining = (
                dataset.num_frames - cursor
                if args.source == "dataset" and not repeating_last_frame
                else args.preview_actions
            )
            take = min(args.preview_actions, len(raw_chunk), remaining)
            raw_chunk = raw_chunk[:take]
            validate_chunk(raw_chunk)
            globally_limited_chunk, global_clamped_values, max_global_adjustment = (
                apply_global_limits(raw_chunk)
            )
            preview_chunk, relative_clamped_values, max_relative_adjustment = (
                apply_preview_relative_limit(
                    start_state,
                    globally_limited_chunk,
                    args.max_relative_target,
                )
            )
            clamped_values = global_clamped_values + relative_clamped_values
            max_adjustment = max(max_global_adjustment, max_relative_adjustment)

            chunk_path = (
                output_dir
                / f"chunk_{chunk_id:04d}_{args.source}_{observation_frame:06d}.npz"
            )
            np.savez_compressed(
                chunk_path,
                raw_actions=raw_chunk,
                globally_limited_actions=globally_limited_chunk,
                preview_actions=preview_chunk,
                start_state=start_state,
                observation_frame=np.asarray(observation_frame, dtype=np.int64),
                top_crop=np.asarray(
                    [args.top_crop.x, args.top_crop.y, args.top_crop.size],
                    dtype=np.int64,
                ),
                wrist_crop=np.asarray(
                    [args.wrist_crop.x, args.wrist_crop.y, args.wrist_crop.size],
                    dtype=np.int64,
                ),
                camera_output_size=np.asarray(args.camera_output_size, dtype=np.int64),
                live_crop_applied=np.asarray(args.source == "robot"),
            )
            print(
                f"\n[CHUNK {chunk_id:04d}] source={args.source}, "
                f"obs={observation_frame}, preview_actions={take}, "
                f"infer={inference_seconds:.3f}s, "
                f"global_clamp={global_clamped_values}, "
                f"relative_clamp={relative_clamped_values}, "
                f"max_adjust={max_adjustment:.4f}"
            )
            print(f"[SAVE] {chunk_path}")
            if image_viewer is not None:
                image_viewer.show(
                    raw_observation,
                    camera_keys,
                    chunk_id=chunk_id,
                    observation_frame=observation_frame,
                    task=task,
                )

            segment_offset = 0
            segment_id = 0
            expected_state = start_state.copy()
            discard_remaining = False
            stop_requested = False

            while segment_offset < take:
                segment_end = min(segment_offset + args.execute_actions, take)

                # 실물 실행 중에는 앞 구간 실행 결과의 실제 state에서 다음 구간을
                # preview한다. RViz-only에서는 직전 preview의 끝을 이어서 사용한다.
                if args.apply_to_robot and segment_offset > 0:
                    segment_observation = robot.get_observation()
                    segment_start_state = state_from_raw_observation(
                        segment_observation, dataset.features
                    )
                else:
                    segment_start_state = expected_state

                segment_actions, segment_relative_clamps, segment_max_adjustment = (
                    apply_preview_relative_limit(
                        segment_start_state,
                        globally_limited_chunk[segment_offset:segment_end],
                        args.max_relative_target,
                    )
                )
                print(
                    f"\n[SEGMENT {segment_id + 1}/"
                    f"{(take + args.execute_actions - 1) // args.execute_actions}] "
                    f"chunk_actions=[{segment_offset}:{segment_end}) "
                    f"relative_clamp={segment_relative_clamps}"
                )

                while True:
                    if rviz is not None:
                        rviz.replay(
                            segment_start_state,
                            segment_actions,
                            args.fps,
                            event_pump=(
                                image_viewer.pump if image_viewer is not None else None
                            ),
                        )
                    else:
                        print("[RVIZ] disabled; current segment was not displayed")

                    decision, decision_seconds = read_decision(
                        args.approval_timeout_s,
                        event_pump=(
                            image_viewer.pump if image_viewer is not None else None
                        ),
                    )
                    if decision == "replay":
                        continue
                    break

                executed_actions = 0
                stale_observation = False
                if decision in {"quit", "timeout"}:
                    print(f"[STOP] decision={decision}")
                    stop_requested = True
                elif decision == "discard":
                    print(
                        f"[DISCARD] chunk {chunk_id}의 남은 action "
                        f"[{segment_offset}:{take})를 폐기하고 새로 추론합니다"
                    )
                    discard_remaining = True
                elif args.apply_to_robot:
                    fresh_observation = robot.get_observation()
                    fresh_state = state_from_raw_observation(
                        fresh_observation, dataset.features
                    )
                    state_delta = float(
                        np.max(np.abs(fresh_state - segment_start_state))
                    )
                    if state_delta > args.stale_state_tolerance:
                        stale_observation = True
                        expected_state = fresh_state
                        print(
                            f"[STALE] 승인 직전 state 변화 {state_delta:.4f} > "
                            f"{args.stale_state_tolerance:.4f}; 명령 0개. "
                            "현재 구간을 다시 판단합니다"
                        )
                    else:
                        actual_actions, safety_tripped = execute_robot_prefix(
                            robot,
                            segment_actions,
                            args.fps,
                            after_action=(
                                capture_live_hamlet_history
                                if hamlet_history is not None
                                else None
                            ),
                        )
                        executed_actions = len(actual_actions)
                        actual_path = (
                            output_dir
                            / f"chunk_{chunk_id:04d}_segment_{segment_id:02d}"
                            "_actual_actions.npy"
                        )
                        np.save(actual_path, actual_actions)
                        print(
                            f"[EXECUTED] segment actions "
                            f"{executed_actions}/{len(segment_actions)}; "
                            f"actual targets={actual_path}"
                        )
                        if safety_tripped:
                            print(
                                "[SAFETY TRIP] 현재 구간의 미실행 action과 chunk의 "
                                "나머지를 폐기하고 parking 후 종료합니다"
                            )
                        elif executed_actions == len(segment_actions):
                            segment_offset = segment_end
                            expected_state = actual_actions[-1]
                            if args.sync_eval:
                                # 마지막 command가 적용된 바로 다음 제어 주기에 새 관측을
                                # 잡는다. 멈춰서 안정화한 프레임이 아니라, 폐루프가 실제로
                                # 이어질 때의 이동 중 상태를 다음 정책 입력으로 쓴다.
                                print("[SYNC] prefix 직후 새 관측으로 재추론 (남은 chunk 폐기)")
                                discard_remaining = True
                else:
                    segment_offset = segment_end
                    expected_state = segment_actions[-1]
                    if args.source == "dataset":
                        if cursor < dataset.num_frames:
                            cursor += len(segment_actions)
                        if args.sync_eval:
                            discard_remaining = True
                        if cursor >= dataset.num_frames and args.repeat_last_frame:
                            print(
                                "[APPROVED/RVIZ-ONLY] dataset episode 끝"
                                f"(frame {dataset.num_frames - 1}) 도달; repeat-last-frame으로 "
                                "마지막 frame을 계속 재사용합니다; physical robot commands sent: 0"
                            )
                        else:
                            print(
                                f"[APPROVED/RVIZ-ONLY] dataset cursor -> {cursor}; "
                                "physical robot commands sent: 0"
                            )
                    else:
                        print(
                            "[APPROVED/LIVE-PREVIEW-ONLY] 다음 구간을 이어서 "
                            "preview합니다; physical robot commands sent: 0"
                        )

                append_jsonl(
                    decision_log,
                    asdict(
                        ChunkLog(
                            chunk_id=chunk_id,
                            segment_id=segment_id,
                            segment_start_action=segment_offset
                            if stale_observation
                            else max(0, segment_end - len(segment_actions)),
                            segment_end_action=segment_end,
                            observation_frame=observation_frame,
                            task=task,
                            predicted_actions=len(raw_chunk_tensor),
                            previewed_actions=len(segment_actions),
                            inference_seconds=inference_seconds,
                            global_clamped_values=global_clamped_values,
                            relative_clamped_values=segment_relative_clamps,
                            clamped_values=global_clamped_values
                            + segment_relative_clamps,
                            max_clamp_adjustment=max(
                                max_global_adjustment, segment_max_adjustment
                            ),
                            decision=decision,
                            decision_seconds=decision_seconds,
                            apply_to_robot=args.apply_to_robot,
                            executed_actions=executed_actions,
                            stale_observation=stale_observation,
                            safety_tripped=safety_tripped,
                        )
                    ),
                )

                if safety_tripped:
                    wait_for_safety_parking(robot)
                    stop_requested = True
                if stop_requested or discard_remaining:
                    break
                if stale_observation:
                    # 남은 chunk는 유지한다. 새 실제 state로 같은 구간을 다시
                    # preview하고 승인받는다.
                    continue
                segment_id += 1

            if stop_requested:
                break
            if segment_offset >= take:
                approved_count += 1
                print(f"[CHUNK COMPLETE] chunk {chunk_id}: {take}/{take} actions approved")
            chunk_id += 1

        print(
            f"[DONE] approved_chunks={approved_count}, final_frame={cursor}, "
            f"safety_tripped={safety_tripped}, "
            f"log={decision_log}"
        )
        return 3 if safety_tripped else 0
    except KeyboardInterrupt:
        print("\n[STOP] interrupted")
        return 130
    finally:
        if image_viewer is not None:
            image_viewer.close()
        if rviz is not None:
            rviz.close()
        if robot is not None and robot.is_connected:
            if robot.safety_tripped:
                wait_for_safety_parking(robot)
                robot.disconnect(park=False)
            else:
                robot.disconnect(park=args.park_on_exit)


if __name__ == "__main__":
    raise SystemExit(main())
