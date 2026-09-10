#!/usr/bin/env python3
"""piper_infer_runner.py — 정책 추론 제어 루프 본체 (GUI 없음).

piper_infer_gui.py의 InferenceWorker를 여기로 뺀 것이다. GUI도, teleop_ui의
Infer 프리셋도, CLI도 전부 이 한 루프를 쓴다. lerobot-record --policy.path를
대체하지만 우회하지는 않는다 — LeRobotDataset / make_policy / PiperFollower 등
lerobot API 위에 그대로 얹혀 있고, 실물 명령은 전부 PiperFollower.send_action()을
지나가므로 max_relative_target과 effort 안전 컷오프가 유지된다.

lerobot-record 대신 우리가 루프를 드는 이유는 temporal ensemble 때문이다.
ensemble은 매 스텝 새로 예측한 action chunk들을 겹쳐서 가중평균하므로 chunk
단위 접근이 필요한데, lerobot-record의 policy 경로는 chunk를 노출하지 않는다.
자세한 근거와 실측값은 docs/policy/smoothing.md 참고.

## 모드

모드는 프리셋일 뿐이다 — 고르면 아래 값들이 세팅되고, 개별 값은 그대로 보이고
따로 덮어쓸 수 있다. 논문에 조건을 명시해야 하므로 모드 뒤에 값을 숨기지 않는다.

  demo(시연용)     기록 없음. 스무딩 강하게. 궤적/지표만 npz로 남긴다.
  augment(증강용)  롤아웃을 LeRobotDataset으로 기록. 카메라는 크롭 전 원본
                   프레임을 저장하고, 끝나면 성공/실패를 물어 sidecar에 남긴다.

## 증강용 기록에서 지키는 것

- 기록되는 action은 정책 raw 출력이 아니라 **스무딩을 거쳐 실제로 send_action()에
  넘어간 값**이다. 그래야 영상과 움직임의 인과가 맞아 BC 학습에 쓸 수 있다.
  raw chunk는 학습 호환성을 깨지 않도록 dataset feature가 아니라 옆의
  `rollout_meta.json` / `raw_actions.npz`에 따로 남긴다.
- 카메라는 정책에 먹인 512 크롭이 아니라 **원본 프레임**을 저장한다. 기존
  Record 경로와 같은 형태라 scripts/tools/prepare_erase_shape_dataset.py로
  똑같이 학습용 변환을 돌릴 수 있다.
- 실행 조건(스무딩 파라미터 전부 + 실제 측정 제어 주기)을 sidecar에 기록한다.

주의: temporal ensemble을 켜고 기록하면 action_t가 과거 스텝의 chunk 예측에도
영향을 받는다. 관찰↔행동 인과가 한 스텝 수준에서 살짝 번지므로, 순수한 BC
데이터가 필요하면 ensemble을 끄고 기록하거나 raw_actions.npz 쪽을 쓸 것.

실행:
    python scripts/tools/piper_infer_runner.py --help
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import pathlib
import queue
import signal
import sys
import threading
import time
from typing import Any, Callable

import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from action_smoothing import (  # noqa: E402
    GLOBAL_HIGH,
    GLOBAL_LOW,
    SmoothingConfig,
    SmoothingPipeline,
    smoothness_metrics,
)

MOTOR_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
REAL_ROBOT_CONFIRM = "I_UNDERSTAND_REAL_ROBOT"
# MIT는 토크 제어라 위치 제어용 안전장치(max_relative_target, effort 컷오프)의
# 의미가 달라진다. 실물 확인 문구와 별개로 하나 더 요구한다.
MIT_CONFIRM = "I_UNDERSTAND_TORQUE_CONTROL"
DEFAULT_ENV_FILE = REPO_ROOT / "configs" / "recording.env"


class RvizPublisher:
    """예측 action을 /joint_states로 publish해서 RViz에서 보게 한다.

    piper_infer_gui.py에 있던 것을 여기로 옮겼다 — GUI가 runner를 import하므로
    반대 방향 import는 순환이 된다. GUI는 이제 여기서 가져다 쓴다.
    """

    def __init__(self, topic: str = "/joint_states", node_name: str = "piper_infer_runner") -> None:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState

        from piper_infer_preview import unnormalize_to_physical

        self._rclpy = rclpy
        self._JointState = JointState
        self._unnormalize = unnormalize_to_physical
        if not rclpy.ok():
            rclpy.init()
        self._node = Node(node_name)
        self._publisher = self._node.create_publisher(JointState, topic, 10)
        self.topic = topic

    def publish(self, action: np.ndarray) -> None:
        message = self._JointState()
        message.header.stamp = self._node.get_clock().now().to_msg()
        message.name = MOTOR_NAMES
        message.position = [
            self._unnormalize(name, float(value)) for name, value in zip(MOTOR_NAMES, action)
        ]
        self._publisher.publish(message)
        self._rclpy.spin_once(self._node, timeout_sec=0.0)

    def close(self) -> None:
        try:
            self._node.destroy_node()
            if self._rclpy.ok():
                self._rclpy.shutdown()
        except Exception:
            pass


class Event:
    """runner가 바깥(GUI/CLI)으로 보내는 이벤트 종류."""

    LOG = "log"
    STEP = "step"
    FINISHED = "finished"


# ═══════════════════════════════════════════════════════════════════
# 모드 프리셋
# ═══════════════════════════════════════════════════════════════════
@dataclasses.dataclass(frozen=True)
class ModePreset:
    """모드를 고르면 적용되는 기본값. 개별 값은 이후 자유롭게 덮어쓸 수 있다."""

    name: str
    label: str
    record_dataset: bool
    record_raw_frames: bool
    prompt_outcome: bool
    smoothing: SmoothingConfig
    description: str


DEMO_MODE = ModePreset(
    name="demo",
    label="시연용",
    record_dataset=False,
    record_raw_frames=False,
    prompt_outcome=False,
    # 실측 최적값. docs/policy/smoothing.md의 TV/RMS jerk 표 참고.
    #
    # ema_alpha는 오랫동안 1.0(꺼짐)이었는데, 실물에서 위치 명령 자체가 스텝의
    # 33.7%에서 방향을 뒤집는 걸 확인하고 켰다. MOVE J는 점대점 플래너가 그
    # 지터를 뭉개줘서 티가 안 났지만, MIT는 충실한 추종기라 그대로 재현해
    # 팔이 ~10Hz로 떤다. alpha=0.2면 방향 반전이 5.8%로 줄고 이동폭은 그대로다
    # (42.28 vs 42.57) — 실제 움직임이 아니라 지터만 걷힌다.
    smoothing=SmoothingConfig(
        temporal_ensemble=True, ensemble_m=0.01, ema_alpha=0.2, rate_limit=5.0
    ),
    description="기록 없이 부드럽게 움직이는 데 집중. 궤적/지표만 npz로 남긴다.",
)

AUGMENT_MODE = ModePreset(
    name="augment",
    label="증강용",
    record_dataset=True,
    record_raw_frames=True,
    prompt_outcome=True,
    smoothing=SmoothingConfig(
        temporal_ensemble=True, ensemble_m=0.01, ema_alpha=0.2, rate_limit=5.0
    ),
    description="롤아웃을 LeRobotDataset으로 기록. 원본 프레임 저장 + 성공/실패 표시.",
)

MODES: dict[str, ModePreset] = {DEMO_MODE.name: DEMO_MODE, AUGMENT_MODE.name: AUGMENT_MODE}
MODE_LABELS = {mode.label: mode.name for mode in MODES.values()}


def mode_preset(name: str) -> ModePreset:
    if name not in MODES:
        raise ValueError(f"알 수 없는 모드 {name!r} — {sorted(MODES)} 중 하나여야 합니다")
    return MODES[name]


# ═══════════════════════════════════════════════════════════════════
# 롤아웃 dataset 기록
# ═══════════════════════════════════════════════════════════════════
def build_rollout_features(
    *,
    camera_shapes: dict[str, tuple[int, int, int]],
    state_names: list[str],
    action_names: list[str],
) -> dict:
    """롤아웃 dataset의 feature 정의를 만든다.

    lerobot-record가 만드는 것과 같은 형태 — observation.state / action은 float32
    벡터, 카메라는 HWC video. camera_shapes는 실제로 저장할 프레임 크기여야 한다
    (증강용에서는 크롭 전 원본).
    """
    features: dict[str, dict] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": list(state_names),
        },
        "action": {
            "dtype": "float32",
            "shape": (len(action_names),),
            "names": list(action_names),
        },
    }
    for camera, shape in camera_shapes.items():
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": tuple(shape),
            "names": ["height", "width", "channels"],
        }
    return features


class RolloutRecorder:
    """추론 롤아웃을 LeRobotDataset으로 기록한다.

    기존 Record 경로가 만드는 데이터셋과 같은 형태를 유지해서, 녹화한 롤아웃을
    prepare_erase_shape_dataset.py로 똑같이 학습용 변환할 수 있게 한다.
    raw chunk와 실행 조건은 dataset feature를 오염시키지 않도록 sidecar로 뺀다.
    """

    def __init__(
        self,
        *,
        root: pathlib.Path,
        repo_id: str,
        fps: int,
        features: dict,
        task: str,
        robot_type: str = "piper_follower",
    ) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.root = pathlib.Path(root)
        self.repo_id = repo_id
        self.task = task
        self.fps = fps
        self.frames_written = 0
        self.episodes_written = 0
        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=self.root,
            robot_type=robot_type,
            use_videos=True,
        )

    def add_frame(
        self,
        *,
        state: np.ndarray,
        action: np.ndarray,
        images: dict[str, np.ndarray],
    ) -> None:
        frame: dict[str, Any] = {
            "observation.state": np.asarray(state, dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "task": self.task,
        }
        for camera, image in images.items():
            frame[f"observation.images.{camera}"] = np.asarray(image)
        self.dataset.add_frame(frame)
        self.frames_written += 1

    def save_episode(self) -> None:
        if self.frames_written == 0:
            return
        self.dataset.save_episode()
        self.episodes_written += 1

    def discard_episode(self) -> None:
        buffer = getattr(self.dataset, "episode_buffer", None)
        if buffer is not None:
            episode_index = self._current_episode_index(buffer)
            self.dataset.clear_episode_buffer()
            self._remove_leftover_frames(episode_index)
        self.frames_written = 0

    @staticmethod
    def _current_episode_index(buffer: dict) -> int:
        index = buffer.get("episode_index", 0)
        if isinstance(index, np.ndarray):
            return int(index.item() if index.size == 1 else index[0])
        if isinstance(index, list):
            return int(index[0]) if index else 0
        return int(index)

    def _remove_leftover_frames(self, episode_index: int) -> None:
        """폐기한 에피소드의 임시 PNG를 지운다.

        lerobot 0.4.4의 clear_episode_buffer()는 meta.image_keys만 정리하는데,
        우리 카메라는 video dtype이라 video_keys로 분류돼서 그 정리를 못 받는다.
        그대로 두면 폐기할 때마다 images/ 밑에 프레임이 쌓인다.
        """
        import shutil

        video_keys = list(getattr(self.dataset.meta, "video_keys", []))
        for camera_key in video_keys:
            try:
                directory = self.dataset._get_image_file_dir(episode_index, camera_key)
            except Exception:
                continue
            if directory.is_dir():
                shutil.rmtree(directory, ignore_errors=True)

    def write_sidecar(self, payload: dict) -> pathlib.Path:
        """실행 조건 / 성공-실패 라벨 / raw chunk를 dataset 옆에 남긴다."""
        path = self.root / "rollout_meta.json"
        existing: list = []
        if path.exists():
            with contextlib.suppress(json.JSONDecodeError, OSError):
                existing = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(existing, list):
            existing = [existing]
        existing.append(payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        return path

    def write_raw_actions(self, raw_actions: np.ndarray, episode_index: int) -> pathlib.Path:
        path = self.root / f"raw_actions_ep{episode_index:04d}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, raw_first_actions=raw_actions)
        return path


# ═══════════════════════════════════════════════════════════════════
# 실행 설정
# ═══════════════════════════════════════════════════════════════════
@dataclasses.dataclass
class RunSettings:
    """runner 한 번의 실행에 필요한 전부. GUI/CLI가 공통으로 채운다."""

    dataset_root: pathlib.Path
    policy_path: str
    episode: int = 0
    task: str = ""
    device: str = "cuda"
    source: str = "dataset"  # dataset | robot
    apply_to_robot: bool = False
    real_robot_confirm: str = ""
    # 학습 데이터 fps와 맞춘다. 정책은 "다음 프레임의 action"을 예측하므로 이 값이
    # 학습 fps보다 낮으면 시연 동작이 그 비율만큼 슬로모션이 된다(30fps 학습을
    # 6Hz로 재생 = 5배 느림). 낮은 명령 주파수는 "움직였다 멈췄다"를 반복해
    # 물리적 끊김/진동도 만든다 — 궤적이 아무리 매끄러워도 티가 안 난다.
    fps: float = 30.0
    # chunk 하나가 이미 chunk_size(=50)스텝 분량을 담고 있어 매 스텝 추론할 필요가
    # 없다. 5스텝마다 추론하면 30Hz에서 추론 간격이 167ms라 115ms 추론이 여유 있게
    # 들어가고, 겹치는 chunk가 50/5=10개라 temporal ensemble도 유지된다.
    infer_every: int = 5
    horizon: int = 50
    max_steps: int = 0
    loop_dataset: bool = False
    rviz: bool = True
    joint_state_topic: str = "/joint_states"
    park_on_exit: bool = True
    crops: dict = dataclasses.field(default_factory=dict)
    camera_output_size: int = 512
    # send_action의 안전 클램프. smoothing.rate_limit과는 다른 것이다 —
    # rate_limit은 "직전 *명령*에서 얼마나 변할 수 있나"이고, 이건 "명령이 *실측
    # 위치*에서 얼마나 떨어질 수 있나"다. 예전에는 둘을 같은 값으로 묶어놨는데,
    # 그러면 스무딩을 조이려고 rate_limit을 줄일 때 로봇 클램프까지 조여져서
    # 팔이 목표를 못 쫓아가고 매 스텝 포화된다. None이면 recording.env의
    # MAX_RELATIVE_TARGET을 쓴다.
    max_relative_target: float | None = None
    # ModeCtrl의 MOVE 모드. 1=MOVE J(점대점), 5=MOVE CPV(연속 위치-속도).
    # 30Hz 스트리밍에서는 MOVE J가 매 스텝 궤적을 다시 계획해 팔이 떤다.
    # None이면 PiperFollowerConfig 기본값(MOVE J)을 그대로 쓴다.
    move_mode: int | None = None
    # ModeCtrl의 속도 백분율(0~100). 팔이 실제로 얼마나 빨리 움직이는지를 정한다 —
    # smoothing의 rate_limit이나 max_relative_target과는 다르다. 그 둘은 명령값의
    # 상한(천장)이라 올려도 팔이 빨라지지 않는다. None이면 config 기본값(30).
    move_speed_rate: int | None = None
    # A. 룩어헤드(초). 0이면 꺼짐. MOVE J는 목표마다 궤적을 계획하는데, 33ms 앞의
    # 목표는 거리가 너무 짧아 가속 초입만 밟다 교체된다. 앞을 보고 쏘면 각 명령에
    # 실제 거리와 일관된 방향이 생긴다(pure pursuit의 carrot). 대신 코너를 자르고
    # 기준 궤적보다 그만큼 뒤처진다.
    lookahead_s: float = 0.0
    # C. MIT(임피던스) 제어. 궤적 재계획이 없어 30Hz 스트리밍에 맞는 인터페이스.
    # ⚠ 토크 제어라 게인이 잘못되면 팔이 무너지거나 진동한다 — 기본 꺼짐이고
    # 켜려면 확인 문구까지 필요하다.
    use_mit: bool = False
    mit_kp: float = 10.0
    mit_kd: float = 0.8
    mit_confirm: str = ""
    # 관절별 kp 덮어쓰기 "joint2=30,joint3=20". 빈 문자열이면 config 기본값을 쓴다.
    mit_kp_overrides: str | None = None
    # vel_ref 다듬기. 30Hz 궤적을 그냥 미분하면((a-prev)*fps) 지터가 30배로
    # 증폭돼 속도 신호가 아니라 노이즈가 된다 — 실측에서 스텝 간 변화가 속도
    # 크기의 72%였고 부호가 3스텝에 1번(28.7%) 뒤집혔다. 그걸 kd*(vel_ref-vel)에
    # 넣으면 초당 10번 방향이 바뀌는 토크가 나가서 팔이 떤다.
    # EMA alpha=0.2면 부호 반전이 8.1%, 스텝 간 변화가 1.55로 줄고 크기는 유지된다.
    mit_vel_smoothing: float = 0.2
    # vel_ref 배율. 0이면 속도 피드포워드를 완전히 끈다(순수 위치 임피던스) —
    # 흔들림 원인이 속도항인지 가려낼 때 쓴다.
    mit_vel_scale: float = 1.0
    smoothing: SmoothingConfig = dataclasses.field(default_factory=SmoothingConfig)
    # 모드 프리셋에서 오는 값들 — 개별 수정 가능
    mode: str = DEMO_MODE.name
    record_dataset: bool = False
    record_raw_frames: bool = False
    prompt_outcome: bool = False
    record_root: pathlib.Path | None = None
    record_repo_id: str = ""

    @classmethod
    def from_mode(cls, mode: str, **overrides: Any) -> "RunSettings":
        """모드 프리셋을 적용한 뒤 overrides로 덮어쓴다."""
        preset = mode_preset(mode)
        base: dict[str, Any] = {
            "mode": preset.name,
            "record_dataset": preset.record_dataset,
            "record_raw_frames": preset.record_raw_frames,
            "prompt_outcome": preset.prompt_outcome,
            "smoothing": preset.smoothing,
        }
        base.update(overrides)
        return cls(**base)

    def real_robot_enabled(self) -> bool:
        """실물 전송이 실제로 열렸는지. 세 조건을 모두 만족해야 한다."""
        return (
            self.source == "robot"
            and self.apply_to_robot
            and self.real_robot_confirm == REAL_ROBOT_CONFIRM
        )

    def describe(self) -> str:
        preset = mode_preset(self.mode)
        record = "on" if self.record_dataset else "off"
        return (
            f"mode={preset.name}({preset.label}) source={self.source} "
            f"record={record} fps={self.fps:g} infer_every={self.infer_every} "
            f"smoothing={self.smoothing.summary()}"
        )


# ═══════════════════════════════════════════════════════════════════
# 제어 루프
# ═══════════════════════════════════════════════════════════════════
class InferenceRunner(threading.Thread):
    """정책 추론 → 스무딩 → (RViz/실물) 전송 → (선택) dataset 기록 루프.

    GUI 없이 단독으로 돈다. 진행 상황은 events 큐로 나가고, GUI는 그걸 그려주기만
    하면 된다. events를 안 주면 큐를 내부에 만들어 CLI가 소비한다.
    """

    def __init__(
        self,
        settings: RunSettings,
        events: "queue.Queue[tuple[str, object]] | None" = None,
        *,
        outcome_prompt: Callable[[], tuple[str, str]] | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self.settings = settings
        self.events: "queue.Queue[tuple[str, object]]" = events or queue.Queue()
        self.outcome_prompt = outcome_prompt

        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.estop_event = threading.Event()

        self.trajectory: list[np.ndarray] = []
        self.raw_trajectory: list[np.ndarray] = []
        self.step_periods: list[float] = []
        self.status = "not_started"
        self.recorded_path: pathlib.Path | None = None

        self._pending_smoothing: SmoothingConfig | None = None
        self._smoothing_lock = threading.Lock()
        self._clamp_window: list[bool] = []
        self._clamp_saturated_reports = 0
        self._phase_totals: dict[str, list[float]] = {
            name: [] for name in ("loop", "observe", "infer", "send", "record")
        }
        self._late_steps = 0
        self._infer_requests: "queue.Queue[dict | None]" = queue.Queue(maxsize=1)
        self._infer_results: "queue.Queue[np.ndarray]" = queue.Queue()
        self._infer_busy = threading.Event()
        self._infer_thread: threading.Thread | None = None
        self._infer_error: str | None = None
        self._last_infer_seconds = 0.0

    # ── 바깥에서 거는 제어 ─────────────────────────────────
    def apply_smoothing(self, config: SmoothingConfig) -> None:
        with self._smoothing_lock:
            self._pending_smoothing = config

    def emergency_stop(self) -> None:
        self.estop_event.set()
        self.stop_event.set()

    def _take_pending_smoothing(self) -> SmoothingConfig | None:
        with self._smoothing_lock:
            pending, self._pending_smoothing = self._pending_smoothing, None
        return pending

    def _log(self, message: str) -> None:
        self.events.put((Event.LOG, message))

    # ── 측정된 실제 제어 주기 ──────────────────────────────
    def measured_fps(self) -> float:
        if len(self.step_periods) < 2:
            return 0.0
        return 1.0 / float(np.mean(self.step_periods))

    def run(self) -> None:  # noqa: C901 — 순차적 셋업이라 나누면 오히려 읽기 나쁨
        settings = self.settings
        robot = None
        rviz = None
        recorder = None
        status = "finished"
        try:
            import torch  # noqa: F401  (정책 로딩 전에 import 비용을 여기서 치른다)

            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            from lerobot.utils.utils import get_safe_torch_device

            from piper_human_approved_inference import (
                preprocess_live_camera_observation,
                state_from_raw_observation,
                validate_live_camera_output_size,
            )
            from piper_offline_chunk_rollout import (
                load_policy,
                make_raw_observation,
                predict_chunk,
            )

            self._log(f"[RUN] {settings.describe()}")

            dataset_root = pathlib.Path(settings.dataset_root).expanduser().resolve()
            self._log(f"[LOAD] dataset={dataset_root} (episode {settings.episode})")
            dataset = LeRobotDataset(
                repo_id=f"local/{dataset_root.name}",
                root=dataset_root,
                episodes=[settings.episode],
                video_backend="pyav",
            )
            camera_keys = list(dataset.meta.camera_keys)
            self._log(f"[LOAD] {dataset.num_frames} frames, cameras={camera_keys}")

            # 정책 로딩(수십 초)과 로봇 연결 전에 확인한다 — 안 맞으면 그 뒤에
            # 텐서 크기 불일치로 죽는데, 그 시점엔 이미 팔이 연결돼 있다.
            check_policy_dataset_match(settings.policy_path, dataset_root)

            self._log(f"[LOAD] policy={settings.policy_path} device={settings.device}")
            config, policy, preprocessor, postprocessor = load_policy(
                settings.policy_path, dataset.meta, settings.device
            )
            device = get_safe_torch_device(policy.config.device)
            policy.reset()
            chunk_size = int(getattr(config, "chunk_size", 0)) or settings.horizon
            horizon = min(settings.horizon, chunk_size)
            self._log(f"[LOAD] policy chunk_size={chunk_size}, using horizon={horizon}")

            if settings.source == "robot":
                validate_live_camera_output_size(
                    dataset.features, camera_keys, settings.camera_output_size
                )
                from piper_human_approved_inference import build_robot_from_env

                self._log("[CONNECT] Piper follower + camera 연결 중…")
                clamp = settings.max_relative_target
                if clamp is None:
                    clamp = float(os.environ.get("MAX_RELATIVE_TARGET", "5.0"))
                self._log(f"[ROBOT] max_relative_target={clamp:g} (실측 위치 기준 클램프)")
                robot_args = argparse.Namespace(max_relative_target=clamp)
                robot = build_robot_from_env(robot_args)
                if settings.use_mit:
                    robot.config.use_mit_control = True
                    robot.config.mit_kp = settings.mit_kp
                    robot.config.mit_kd = settings.mit_kd
                    if settings.mit_kp_overrides is not None:
                        robot.config.mit_kp_overrides = settings.mit_kp_overrides
                    self._log(
                        f"[ROBOT] 관절별 kp: {robot.config.mit_kp_overrides or '(없음, 공통)'}"
                    )
                    if settings.mit_vel_scale == 0:
                        self._log("[ROBOT] 속도 피드포워드 꺼짐 — 순수 위치 임피던스")
                    else:
                        self._log(
                            f"[ROBOT] vel_ref: EMA alpha={settings.mit_vel_smoothing:g}, "
                            f"배율 {settings.mit_vel_scale:g}"
                        )
                    self._log(
                        f"[ROBOT] MIT(임피던스) 제어 kp={settings.mit_kp:g} kd={settings.mit_kd:g} "
                        "— 궤적 재계획 없음, chunk 속도를 vel_ref로 전달"
                    )
                    self._log(
                        "[WARN] 토크 제어입니다. max_relative_target과 effort 컷오프는 "
                        "위치 제어 전제라 의미가 달라집니다 — 이상하면 즉시 E-STOP"
                    )
                if settings.move_speed_rate is not None:
                    robot.config.move_speed_rate = settings.move_speed_rate
                    robot.bus.move_speed_rate = settings.move_speed_rate
                    self._log(f"[ROBOT] move_speed_rate={settings.move_speed_rate}% (컨트롤러 이동 속도)")
                if settings.move_mode is not None:
                    # 연결 전에 바꿔야 첫 명령부터 적용된다.
                    robot.config.move_mode = settings.move_mode
                    robot.bus.move_mode = settings.move_mode
                    label = {1: "MOVE J(점대점)", 5: "MOVE CPV(연속 위치-속도)"}.get(
                        settings.move_mode, str(settings.move_mode)
                    )
                    self._log(f"[ROBOT] move_mode={settings.move_mode} {label}")
                robot.connect()
                self._log("[CONNECT] 연결 완료")

            if settings.rviz:
                try:
                    rviz = RvizPublisher(settings.joint_state_topic)
                    self._log(f"[RVIZ] publishing to {rviz.topic}")
                except Exception as error:
                    self._log(f"[WARN] RViz publisher를 만들 수 없음 — 비활성화: {error}")
                    rviz = None

            task = settings.task or str(dataset[0].get("task", ""))
            self._log(f"[TASK] {task!r}")

            def predict(raw_observation: dict) -> np.ndarray:
                chunk = (
                    predict_chunk(
                        raw_observation=raw_observation,
                        dataset_features=dataset.features,
                        policy=policy,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        device=device,
                        task=task,
                    )
                    .numpy()
                    .astype(np.float32, copy=False)[:horizon]
                )
                if not np.isfinite(chunk).all():
                    raise ValueError("정책이 NaN/Inf를 출력했습니다")
                return chunk

            self._start_inference_worker(predict)

            pipeline = SmoothingPipeline(settings.smoothing)
            self._log(f"[SMOOTH] {settings.smoothing.summary()}")

            # 명령 주파수가 학습 fps보다 낮으면 동작이 그 비율만큼 슬로모션이 되고,
            # 낮은 주파수 자체가 "움직였다 멈췄다"를 만들어 물리적으로 끊겨 보인다.
            # 실측으로 확인된 증상이라 조용히 넘기지 않고 경고한다.
            train_fps = float(getattr(dataset.meta, "fps", 0) or 0)
            if train_fps and settings.fps < train_fps * 0.9:
                self._log(
                    f"[WARN] 명령 주파수 {settings.fps:g}Hz < 학습 fps {train_fps:g} "
                    f"— 동작이 {train_fps / settings.fps:.1f}배 느려지고 끊겨 보입니다. "
                    f"--fps {train_fps:g} --infer-every {max(1, round(chunk_size / 10))} 권장"
                )
            if settings.smoothing.temporal_ensemble:
                votes = max(1, horizon // max(1, settings.infer_every))
                self._log(f"[SMOOTH] infer_every={settings.infer_every} → ensemble 최대 {votes}표")
                if votes < 3:
                    self._log("[WARN] 표수가 3 미만이라 temporal ensemble 효과가 거의 없습니다")

            fps = settings.fps
            period = 1.0 / fps
            infer_every = max(1, settings.infer_every)
            max_steps = settings.max_steps
            cursor = 0
            step = 0
            first_state = None
            previous_action: np.ndarray | None = None
            smoothed_velocity: np.ndarray | None = None
            state_names = list(dataset.features["observation.state"]["names"])
            action_names = list(dataset.features["action"]["names"])
            last_loop_start: float | None = None

            while not self.stop_event.is_set():
                if max_steps and step >= max_steps:
                    self._log(f"[STOP] max_steps({max_steps}) 도달")
                    break
                if self.pause_event.is_set():
                    time.sleep(0.05)
                    continue

                pending = self._take_pending_smoothing()
                if pending is not None:
                    pipeline.update_config(pending, state=first_state)
                    settings.smoothing = pending
                    self._log(f"[SMOOTH] 실행 중 변경 → {pending.summary()}")

                loop_started = time.perf_counter()
                if last_loop_start is not None:
                    self.step_periods.append(loop_started - last_loop_start)
                last_loop_start = loop_started

                # ── observation ────────────────────────────────
                record_images: dict[str, np.ndarray] = {}
                observe_started = time.perf_counter()
                if settings.source == "dataset":
                    if cursor >= dataset.num_frames:
                        if not settings.loop_dataset:
                            self._log("[STOP] dataset episode 끝")
                            break
                        cursor = 0
                    raw_observation = make_raw_observation(dataset, dataset[cursor])
                    observation_frame = cursor
                    if settings.record_dataset:
                        record_images = {
                            key.removeprefix("observation.images."): np.asarray(
                                raw_observation[key.removeprefix("observation.images.")]
                            )
                            for key in camera_keys
                        }
                else:
                    live_observation = robot.get_observation()
                    if settings.record_dataset and settings.record_raw_frames:
                        # 정책에 먹이기 전, 크롭/리사이즈 전 원본 프레임을 따로 잡아둔다.
                        record_images = {
                            key.removeprefix("observation.images."): np.asarray(
                                live_observation[key.removeprefix("observation.images.")]
                            ).copy()
                            for key in camera_keys
                        }
                    raw_observation = preprocess_live_camera_observation(
                        live_observation,
                        camera_keys,
                        settings.crops,
                        settings.camera_output_size,
                    )
                    if settings.record_dataset and not settings.record_raw_frames:
                        record_images = {
                            key.removeprefix("observation.images."): np.asarray(
                                raw_observation[key.removeprefix("observation.images.")]
                            ).copy()
                            for key in camera_keys
                        }
                    observation_frame = step

                self._phase_totals["observe"].append(time.perf_counter() - observe_started)
                measured_state = state_from_raw_observation(raw_observation, dataset.features)
                if first_state is None:
                    first_state = measured_state.copy()
                    pipeline.reset(first_state)
                    if settings.record_dataset:
                        recorder = self._make_recorder(
                            settings=settings,
                            record_images=record_images,
                            state_names=state_names,
                            action_names=action_names,
                            task=task,
                        )

                # ── inference (별도 스레드) ────────────────────
                # 추론은 1회 ~115ms인데 30Hz 명령 주기는 33ms다. 루프 안에서 돌리면
                # "33ms 4번 → 115ms 1번"이 반복돼 6Hz 주기의 규칙적 끊김이 생기고,
                # 팔이 그 리듬으로 진동한다. 그래서 추론을 워커로 넘기고 명령
                # 루프는 일정한 주기를 지킨다. 결과는 준비되는 대로 받아 섞는다.
                infer_seconds = 0.0
                with self._phase("infer"):
                    for chunk in self._collect_chunks():
                        self.raw_trajectory.append(chunk[0].copy())
                        pipeline.add_chunk(chunk)

                    if step % infer_every == 0:
                        self._request_inference(raw_observation)

                    # 쓸 게 떨어졌으면 어쩔 수 없이 기다린다 — 목표 없이 보내느니
                    # 한 스텝 늦는 게 낫다.
                    if pipeline.pending_steps == 0:
                        waited = self._await_chunk(timeout=5.0)
                        if waited is None:
                            self._log("[ERROR] 추론 결과를 기다리다 시간 초과 — 중단합니다")
                            status = "error"
                            break
                        infer_seconds = self._last_infer_seconds
                        self.raw_trajectory.append(waited[0].copy())
                        pipeline.add_chunk(waited)

                if self._infer_error is not None:
                    self._log(f"[ERROR] 추론 스레드: {self._infer_error}")
                    status = "error"
                    break

                votes = pipeline.votes_for_next
                action = pipeline.next_action()
                self.trajectory.append(action.copy())

                # 스무딩된 궤적 자체의 속도(정규화 단위/초). 룩어헤드와 MIT가
                # 둘 다 이걸 쓴다 — 정책 chunk는 시간 매개화된 궤적이라 의도된
                # 속도를 이미 담고 있는데, 위치만 보내면 그 정보를 버리게 된다.
                velocity = np.zeros_like(action)
                if previous_action is not None:
                    velocity = (action - previous_action) * fps
                previous_action = action.copy()

                # 유한차분을 그대로 쓰면 지터가 fps배로 증폭돼 속도가 아니라
                # 노이즈가 된다. EMA로 다듬고 배율을 곱한 값을 vel_ref로 쓴다.
                alpha = float(np.clip(settings.mit_vel_smoothing, 0.0, 1.0))
                smoothed_velocity = (
                    alpha * velocity + (1.0 - alpha) * smoothed_velocity
                    if smoothed_velocity is not None
                    else velocity.copy()
                )
                feedforward_velocity = smoothed_velocity * settings.mit_vel_scale

                # A. 룩어헤드 — 목표를 진행 방향으로 lookahead_s만큼 앞서 보낸다.
                # 기록/지표는 실제 궤적(action) 기준을 유지하고, 로봇에 나가는
                # 목표만 앞당긴다.
                commanded = action
                if settings.lookahead_s > 0:
                    commanded = np.clip(
                        action + velocity * settings.lookahead_s, GLOBAL_LOW, GLOBAL_HIGH
                    )

                # ── 출력 ───────────────────────────────────────
                if self.estop_event.is_set():
                    break
                if rviz is not None:
                    rviz.publish(action)
                safety_tripped = False
                if robot is not None and settings.apply_to_robot:
                    with self._phase("send"):
                        sent = robot.send_action(
                            {
                                f"{name}.pos": float(value)
                                for name, value in zip(MOTOR_NAMES, commanded)
                            },
                            velocity={
                                name: float(value)
                                for name, value in zip(MOTOR_NAMES, feedforward_velocity)
                            },
                        )
                    safety_tripped = bool(robot.safety_tripped)
                    # 클램프에 걸리면 스무딩한 목표가 "실측 위치 + 제한"으로
                    # 통째로 대체된다 — 그 상태가 이어지면 스무딩 파라미터를
                    # 아무리 만져도 소용이 없으므로 눈에 띄게 알린다.
                    self._track_send(commanded, sent)

                # ── 기록 ───────────────────────────────────────
                # 기록하는 action은 스무딩을 거쳐 실제로 나간 값이다(raw 아님).
                if recorder is not None and record_images:
                    with self._phase("record"):
                        recorder.add_frame(
                            state=measured_state, action=action, images=record_images
                        )

                self.events.put(
                    (
                        Event.STEP,
                        {
                            "step": step,
                            "observation_frame": observation_frame,
                            "action": action.copy(),
                            "measured": measured_state.copy(),
                            "votes": votes,
                            "pending": pipeline.pending_steps,
                            "infer_ms": infer_seconds * 1000.0,
                            "rate_clamp": pipeline.last_rate_adjustment,
                            "recorded": recorder.frames_written if recorder else 0,
                        },
                    )
                )

                if safety_tripped:
                    self._log("[SAFETY TRIP] effort 한계 초과 — 명령을 중단하고 parking합니다")
                    status = "safety_tripped"
                    break

                cursor += 1
                step += 1

                elapsed = time.perf_counter() - loop_started
                overrun = elapsed - period
                self._phase_totals["loop"].append(elapsed)
                if step % self.TIMING_REPORT_EVERY == 0:
                    self._report_timing(period)
                if overrun > 0:
                    self._late_steps += 1
                else:
                    self._sleep_until(loop_started + period)

            if self.estop_event.is_set():
                status = "estop"
                self._log("[E-STOP] 명령 전송을 즉시 중단했습니다")

        except Exception as error:  # runner 예외는 그대로 보여준다
            status = "error"
            import traceback

            self._log(f"[ERROR] {type(error).__name__}: {error}")
            self._log(traceback.format_exc())
        finally:
            # 추론 워커를 먼저 세운다 — 로봇 disconnect 뒤에 늦은 결과가
            # 들어오면 이미 닫힌 자원을 건드릴 수 있다.
            self.stop_event.set()
            with contextlib.suppress(queue.Full):
                self._infer_requests.put_nowait(None)
            if self._infer_thread is not None:
                self._infer_thread.join(timeout=10)
            if recorder is not None:
                with contextlib.suppress(Exception):
                    self._finalize_recording(recorder, status)
            if robot is not None:
                try:
                    if getattr(robot, "is_connected", False) and self.settings.use_mit:
                        # parking()은 위치 명령이라 MIT 상태에서는 안 먹는다.
                        robot.config.use_mit_control = False
                        robot.bus.leave_mit_mode()
                        self._log("[ROBOT] MIT 해제 — 위치 제어로 복귀")
                except Exception as error:
                    self._log(f"[WARN] MIT 해제 실패: {error}")
                try:
                    if getattr(robot, "is_connected", False):
                        park = self.settings.park_on_exit and not robot.safety_tripped
                        self._log(f"[DISCONNECT] park={park}")
                        robot.disconnect(park=park)
                except Exception as error:
                    self._log(f"[WARN] disconnect 실패: {error}")
            if rviz is not None:
                rviz.close()
            self.status = status
            self.events.put((Event.FINISHED, status))

    SLEEP_SLICE = 0.005

    def _sleep_until(self, deadline: float, *, clock=time.perf_counter, sleep=time.sleep) -> None:
        """다음 스텝까지 잘게 나눠 잔다. E-stop 반응성을 위해 조각으로 자른다.

        남은 시간은 반드시 매번 다시 재고 음수를 걸러야 한다. 예전 구현은
        `while clock() < deadline: sleep(min(0.005, deadline - clock()))` 였는데,
        while 판정과 인자 계산 사이에 시각이 지나가면 음수가 되고 time.sleep()이
        ValueError를 던진다. 그러면 제어 루프가 통째로 죽고 팔이 park로 내려간다
        — 루프가 빨라져 반복 횟수가 늘어난 뒤 실물에서 실제로 터졌다.
        """
        while not self.stop_event.is_set():
            remaining = deadline - clock()
            if remaining <= 0:
                return
            sleep(min(self.SLEEP_SLICE, remaining))

    # ── 추론 워커 ──────────────────────────────────────────
    def _start_inference_worker(self, predict) -> None:
        """추론을 명령 루프 밖으로 뺀다.

        `predict(raw_observation) -> chunk` 하나만 받는다. 요청은 최대 1건만
        들고 있고(최신 관찰만 의미가 있으므로) 결과는 준비되는 대로 큐로 넘긴다.
        torch 연산은 GIL을 놓으므로 스레드로도 명령 루프를 막지 않는다.
        """

        def loop() -> None:
            while not self.stop_event.is_set():
                try:
                    observation = self._infer_requests.get(timeout=0.1)
                except queue.Empty:
                    continue
                if observation is None:
                    break
                started = time.perf_counter()
                try:
                    chunk = predict(observation)
                except Exception as error:  # 루프가 알아채고 멈추도록 넘긴다
                    self._infer_error = f"{type(error).__name__}: {error}"
                    self._infer_busy.clear()
                    break
                self._last_infer_seconds = time.perf_counter() - started
                self._infer_results.put(chunk)
                self._infer_busy.clear()

        self._infer_thread = threading.Thread(target=loop, name="piper-infer", daemon=True)
        self._infer_thread.start()

    def _request_inference(self, raw_observation: dict) -> None:
        """워커가 놀고 있을 때만 새 관찰을 넘긴다. 밀리면 그냥 건너뛴다 —
        오래된 관찰로 추론해봐야 쓸모가 없다."""
        if self._infer_busy.is_set():
            return
        self._infer_busy.set()
        self._infer_requests.put(raw_observation)

    def _collect_chunks(self) -> list[np.ndarray]:
        chunks = []
        while True:
            try:
                chunks.append(self._infer_results.get_nowait())
            except queue.Empty:
                return chunks

    def _await_chunk(self, timeout: float) -> "np.ndarray | None":
        try:
            return self._infer_results.get(timeout=timeout)
        except queue.Empty:
            return None

    # ── 타이밍 진단 ────────────────────────────────────────
    TIMING_REPORT_EVERY = 90

    def _report_timing(self, period: float) -> None:
        """루프 시간이 어디로 새는지 단계별로 보고한다.

        명령 주파수가 흔들리면(어떤 스텝은 33ms, 어떤 스텝은 115ms) 팔이
        "움직였다 멈췄다"를 반복해 진동한다. 평균만 봐서는 이 지터가 안 보이므로
        단계별 평균과 함께 최대치·지연 스텝 수를 같이 낸다.
        """
        loops = self._phase_totals["loop"]
        if not loops:
            return
        mean = float(np.mean(loops))
        worst = float(np.max(loops))
        parts = []
        for name in ("observe", "infer", "send", "record"):
            samples = self._phase_totals[name]
            if samples:
                parts.append(f"{name} {float(np.mean(samples)) * 1000:.0f}ms")
        late = self._late_steps
        self._log(
            f"[TIMING] 최근 {len(loops)}스텝 평균 {mean * 1000:.0f}ms "
            f"({1.0 / mean if mean else 0:.1f}Hz), 최대 {worst * 1000:.0f}ms, "
            f"목표({period * 1000:.0f}ms) 초과 {late}회"
            + (" | " + ", ".join(parts) if parts else "")
        )
        if worst > period * 2 and mean < period * 1.5:
            self._log(
                "[WARN] 주기가 고르지 않습니다 — 느린 스텝이 섞여 있으면 평균이 맞아도 "
                "팔이 규칙적으로 끊깁니다"
            )
        for samples in self._phase_totals.values():
            samples.clear()
        self._late_steps = 0

    @contextlib.contextmanager
    def _phase(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self._phase_totals[name].append(time.perf_counter() - started)

    # ── 추종 진단 ──────────────────────────────────────────
    CLAMP_REPORT_EVERY = 60

    def _track_send(self, requested: np.ndarray, sent: dict | None) -> None:
        """요청한 목표와 실제로 나간 목표의 차이를 추적한다.

        send_action은 안전 클램프를 적용한 뒤의 값을 돌려준다. 둘이 다르면
        max_relative_target에 걸린 것이고, 그게 계속되면 명령이 사실상
        "실측 위치 + 제한"으로 고정돼 스무딩 결과가 버려진다. 그 상태에서
        smoothing 하이퍼파라미터를 만지는 건 의미가 없으므로 미리 알린다.
        """
        if not sent:
            return
        actual = np.asarray(
            [sent.get(f"{name}.pos", np.nan) for name in MOTOR_NAMES], dtype=np.float32
        )
        if not np.isfinite(actual).all():
            return

        deviation = float(np.abs(actual - requested).max())
        self._clamp_window.append(deviation > 1e-4)
        if len(self._clamp_window) < self.CLAMP_REPORT_EVERY:
            return

        clamped = sum(self._clamp_window)
        ratio = clamped / len(self._clamp_window)
        self._clamp_window.clear()
        if ratio < 0.2:
            return

        message = (
            f"[CLAMP] 최근 {self.CLAMP_REPORT_EVERY}스텝 중 {clamped}회 "
            f"({ratio:.0%}) max_relative_target에 걸림"
        )
        if ratio > 0.9:
            self._clamp_saturated_reports += 1
            message += " — 포화 상태입니다. 명령이 사실상 '실측 위치 + 제한'으로"
            if self._clamp_saturated_reports == 1:
                message += (
                    " 대체되고 있어 smoothing 파라미터는 효과가 없습니다."
                    " max_relative_target을 올리거나 fps를 낮추세요."
                )
        self._log(message)

    # ── 기록 헬퍼 ──────────────────────────────────────────
    def _make_recorder(
        self,
        *,
        settings: RunSettings,
        record_images: dict[str, np.ndarray],
        state_names: list[str],
        action_names: list[str],
        task: str,
    ) -> "RolloutRecorder | None":
        """첫 프레임을 본 뒤에야 실제 카메라 해상도를 알 수 있어 여기서 만든다."""
        if not record_images:
            self._log("[WARN] 기록할 카메라 프레임이 없어 dataset 기록을 건너뜁니다")
            return None
        camera_shapes = {
            camera: tuple(np.asarray(image).shape) for camera, image in record_images.items()
        }
        features = build_rollout_features(
            camera_shapes=camera_shapes,
            state_names=state_names,
            action_names=action_names,
        )
        root = settings.record_root or default_record_root(settings)
        repo_id = settings.record_repo_id or f"local/{pathlib.Path(root).name}"
        # 기록 fps는 설정값이 아니라 실제 제어 주기에 맞춰야 하지만, dataset 생성
        # 시점에는 아직 측정치가 없다. 설정 fps로 만들고 실측치는 sidecar에 남긴다.
        recorder = RolloutRecorder(
            root=pathlib.Path(root),
            repo_id=repo_id,
            fps=int(round(settings.fps)),
            features=features,
            task=task,
        )
        shapes = ", ".join(f"{k}{v}" for k, v in camera_shapes.items())
        self._log(f"[RECORD] {root} (fps={int(round(settings.fps))}, {shapes})")
        if settings.record_raw_frames:
            self._log(
                "[RECORD] 크롭 전 원본 프레임을 저장합니다 — "
                "학습에 쓰려면 prepare_erase_shape_dataset.py로 변환하세요"
            )
        self.recorded_path = pathlib.Path(root)
        return recorder

    def _finalize_recording(self, recorder: RolloutRecorder, status: str) -> None:
        settings = self.settings
        if recorder.frames_written == 0:
            self._log("[RECORD] 기록된 프레임이 없어 저장하지 않습니다")
            return

        outcome, note = "unlabeled", ""
        if settings.prompt_outcome and self.outcome_prompt is not None:
            try:
                outcome, note = self.outcome_prompt()
            except Exception as error:
                self._log(f"[WARN] 성공/실패 입력 실패 — unlabeled로 둡니다: {error}")

        if outcome == "discard":
            recorder.discard_episode()
            self._log("[RECORD] 에피소드를 폐기했습니다")
            return

        episode_index = recorder.episodes_written
        frames = recorder.frames_written
        recorder.save_episode()

        raw = np.stack(self.raw_trajectory) if self.raw_trajectory else np.zeros((0, 7), np.float32)
        raw_path = recorder.write_raw_actions(raw, episode_index)

        measured = self.measured_fps()
        payload = {
            "episode_index": episode_index,
            "frames": frames,
            "outcome": outcome,
            "note": note,
            "status": status,
            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": settings.mode,
            "source": settings.source,
            "policy_path": settings.policy_path,
            "reference_dataset": str(settings.dataset_root),
            "task": recorder.task,
            # 논문에 그대로 인용할 수 있게 스무딩 조건 전부를 남긴다.
            "smoothing": dataclasses.asdict(settings.smoothing),
            "infer_every": settings.infer_every,
            "horizon": settings.horizon,
            "configured_fps": settings.fps,
            "measured_fps": round(measured, 3) if measured else None,
            "raw_actions_file": raw_path.name,
            "camera_frames": "raw" if settings.record_raw_frames else "preprocessed",
        }
        sidecar = recorder.write_sidecar(payload)
        self._log(
            f"[RECORD] 저장 완료 — {frames} frames, outcome={outcome}, "
            f"실측 {measured:.2f}Hz (설정 {settings.fps:g})"
        )
        self._log(f"[RECORD] 실행 조건: {sidecar}")
        if measured and abs(measured - settings.fps) / settings.fps > 0.15:
            self._log(
                f"[WARN] 기록된 dataset의 fps는 {int(round(settings.fps))}로 적혀 있지만 "
                f"실제 제어 주기는 {measured:.2f}Hz였습니다 — 학습에 쓰기 전에 확인하세요"
            )


def training_dataset_of(policy_path: str | pathlib.Path) -> str | None:
    """체크포인트가 어떤 dataset으로 학습됐는지 train_config.json에서 읽는다."""
    for candidate in (
        pathlib.Path(policy_path) / "train_config.json",
        pathlib.Path(policy_path).parent.parent.parent / "train_config.json",
    ):
        if candidate.is_file():
            with contextlib.suppress(json.JSONDecodeError, OSError, KeyError):
                dataset = json.loads(candidate.read_text(encoding="utf-8"))["dataset"]
                return dataset.get("root") or dataset.get("repo_id")
    return None


def check_policy_dataset_match(policy_path: str | pathlib.Path, dataset_root: pathlib.Path) -> None:
    """정책이 기대하는 입력과 참조 dataset의 feature가 맞는지 미리 본다.

    runner는 dataset의 meta로 정규화 통계와 관찰 형태를 만들기 때문에, 정책이
    학습된 것과 다른 dataset을 고르면 정책 로딩(수십 초)과 로봇 연결까지 다
    끝난 뒤에야 텐서 크기 불일치로 죽는다. Dataset Browser에 200개가 다 보이므로
    실수하기 쉽다 — config.json만 읽어서(torch 로딩 없이) 미리 막는다.
    """
    config_path = pathlib.Path(policy_path) / "config.json"
    info_path = pathlib.Path(dataset_root) / "meta" / "info.json"
    if not config_path.is_file() or not info_path.is_file():
        return  # HF repo id 등 로컬에서 확인 불가한 경우는 그냥 통과시킨다

    try:
        expected = json.loads(config_path.read_text(encoding="utf-8"))
        features = json.loads(info_path.read_text(encoding="utf-8"))["features"]
    except (json.JSONDecodeError, OSError, KeyError):
        return

    wanted = dict(expected.get("input_features") or {})
    wanted.update(expected.get("output_features") or {})

    problems: list[str] = []
    for key, spec in wanted.items():
        if key not in features:
            problems.append(f"  {key}: dataset에 없음")
            continue
        # 이미지 shape은 CHW/HWC 표기가 섞여 있어 채널 위치가 다를 수 있다.
        # 정규화 문제를 일으키는 건 벡터 feature이므로 그쪽만 엄격히 본다.
        if key.startswith("observation.images."):
            continue
        want = tuple(spec.get("shape") or ())
        have = tuple(features[key].get("shape") or ())
        if want != have:
            problems.append(f"  {key}: 정책은 {want}, dataset은 {have}")

    if not problems:
        return

    trained_on = training_dataset_of(policy_path)
    lines = [
        "정책과 참조 dataset이 맞지 않습니다:",
        *problems,
        "",
        "runner는 참조 dataset의 meta로 정규화 통계와 관찰 형태를 만듭니다 —",
        "정책을 학습시킨 그 dataset을 고르세요.",
    ]
    if trained_on:
        lines.append(f"이 체크포인트의 학습 dataset: {trained_on}")
    raise ValueError("\n".join(lines))


def load_env_file(path: pathlib.Path) -> dict[str, str]:
    """configs/recording.env 형식(KEY=VALUE)을 읽는다. 없으면 빈 dict."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def resolve_crops(
    env: dict[str, str], top: str | None = None, wrist: str | None = None
) -> dict:
    """live 카메라 crop 설정을 정한다.

    source=robot에서는 학습 때와 똑같이 크롭/리사이즈해야 정책이 제대로 본다.
    GUI는 자기 입력칸에서 받지만 CLI에는 그 값이 없어서, human_approved 도구와
    같은 recording.env 키를 재사용한다(값이 이미 거기 있고 두 벌로 관리할 이유가
    없다). --top-crop / --wrist-crop으로 덮어쓸 수 있다.
    """
    from piper_human_approved_inference import parse_camera_crop

    sources = {
        "top": top or env.get("HUMAN_APPROVED_TOP_CROP", ""),
        "wrist": wrist or env.get("HUMAN_APPROVED_WRIST_CROP", ""),
    }
    return {
        camera: parse_camera_crop(value) for camera, value in sources.items() if value
    }


def default_record_root(settings: RunSettings) -> pathlib.Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = pathlib.Path(settings.dataset_root).name
    return REPO_ROOT / "records" / "rollout" / f"{name}_rollout_{stamp}"


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════
def terminal_outcome_prompt() -> tuple[str, str]:
    """롤아웃이 끝나면 성공/실패를 물어본다. 증강용에서만 호출된다."""
    print("\n" + "=" * 60)
    print("이 롤아웃을 어떻게 기록할까요?")
    print("  s = 성공(success)   f = 실패(failure)")
    print("  u = 판단 보류(unlabeled)   d = 폐기(discard, 저장 안 함)")
    while True:
        try:
            answer = input("선택 [s/f/u/d]: ").strip().lower()
        except EOFError:
            return "unlabeled", ""
        mapping = {"s": "success", "f": "failure", "u": "unlabeled", "d": "discard"}
        if answer in mapping:
            outcome = mapping[answer]
            if outcome == "discard":
                return outcome, ""
            try:
                note = input("메모 (없으면 Enter): ").strip()
            except EOFError:
                note = ""
            return outcome, note
        print("s, f, u, d 중 하나를 입력하세요.")


# 정리(파킹 → 손목 내리기 → 그리퍼 사이클 → 토크 해제)가 끝날 때까지 기다리는 한계.
# park_lower는 parking(최대 10초) + 램프 2초 + 그리퍼 여닫기 1.5초×2 라서 30초로는
# 모자랄 수 있다. 모자라면 데몬 스레드가 파킹 도중에 잘려서 팔이 그 자리에 늘어진다
# — 실제로 Stop을 눌렀을 때 파킹을 안 하고 힘이 풀리던 원인.
SHUTDOWN_TIMEOUT_S = 120.0


def _drain_until_finished(runner: "InferenceRunner") -> str:
    """중단 요청 뒤에도 이벤트를 계속 소비하면서 정리가 끝나기를 기다린다.

    예전에는 KeyboardInterrupt를 받자마자 이벤트 루프를 빠져나가 join만 했다.
    그러면 runner 스레드가 정리하면서 남기는 [MIT 해제] / [DISCONNECT] 로그가
    큐에 쌓인 채 출력되지 않아, 파킹이 됐는지 안 됐는지 알 수 없었다.
    """
    deadline = time.perf_counter() + SHUTDOWN_TIMEOUT_S
    while time.perf_counter() < deadline:
        try:
            kind, payload = runner.events.get(timeout=1.0)
        except queue.Empty:
            if not runner.is_alive():
                break
            continue
        if kind == Event.LOG:
            print(payload, flush=True)
        elif kind == Event.FINISHED:
            return str(payload)
    else:
        print(
            f"[WARN] 정리가 {SHUTDOWN_TIMEOUT_S:g}초 안에 끝나지 않았습니다 — "
            "팔이 파킹되지 않았을 수 있습니다",
            file=sys.stderr,
            flush=True,
        )
    return runner.status


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="정책 추론 제어 루프 (smoothing + 롤아웃 dataset 기록)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        default=DEMO_MODE.name,
        choices=sorted(MODES),
        help="; ".join(f"{m.name}={m.description}" for m in MODES.values()),
    )
    parser.add_argument("--dataset-root", required=True, type=pathlib.Path)
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--task", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source", default="dataset", choices=["dataset", "robot"])
    parser.add_argument(
        "--fps", type=float, default=30.0, help="명령 주파수. 학습 데이터 fps와 맞출 것"
    )
    parser.add_argument(
        "--infer-every", type=int, default=5, help="N 스텝마다 추론. ensemble 표수 = chunk_size/N"
    )
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--loop-dataset", action="store_true")
    parser.add_argument("--no-rviz", dest="rviz", action="store_false")
    parser.add_argument("--joint-state-topic", default="/joint_states")

    smoothing = parser.add_argument_group("smoothing (모드 기본값을 덮어씀)")
    smoothing.add_argument("--no-ensemble", dest="temporal_ensemble", action="store_false")
    smoothing.add_argument("--ensemble-m", type=float)
    smoothing.add_argument("--ema-alpha", type=float)
    smoothing.add_argument("--rate-limit", type=float)
    parser.set_defaults(temporal_ensemble=None)

    record = parser.add_argument_group("기록 (모드 기본값을 덮어씀)")
    record.add_argument("--record", dest="record_dataset", action="store_true", default=None)
    record.add_argument("--no-record", dest="record_dataset", action="store_false")
    record.add_argument("--record-root", type=pathlib.Path)
    record.add_argument("--record-repo-id", default="")
    record.add_argument("--no-prompt-outcome", dest="prompt_outcome", action="store_false", default=None)

    robot = parser.add_argument_group("실물 전송 (셋 다 있어야 열림)")
    robot.add_argument("--apply-to-robot", action="store_true")
    robot.add_argument("--real-robot-confirm", default="")
    robot.add_argument("--no-park-on-exit", dest="park_on_exit", action="store_false")
    robot.add_argument(
        "--move-mode",
        type=int,
        choices=[1, 5],
        help="1=MOVE J(점대점, 기본), 5=MOVE CPV(연속 위치-속도, 30Hz 스트리밍용). "
        "펌웨어 V1.8-1 이상에서만 5를 쓸 수 있다",
    )
    robot.add_argument(
        "--lookahead-s",
        type=float,
        default=0.0,
        help="목표를 이만큼 앞서 보낸다(초). MOVE J의 재계획 문제를 줄인다. 0=꺼짐",
    )
    robot.add_argument("--mit", action="store_true", help="MIT(임피던스) 제어 사용 — 토크 제어")
    robot.add_argument("--mit-kp", type=float, default=10.0)
    robot.add_argument("--mit-kd", type=float, default=0.8)
    robot.add_argument("--mit-confirm", default="", help=f"MIT를 켜려면 {MIT_CONFIRM}")
    robot.add_argument(
        "--mit-vel-smoothing",
        type=float,
        default=0.2,
        help="vel_ref EMA 계수(0~1). 작을수록 부드럽다. 1=다듬지 않음(노이즈)",
    )
    robot.add_argument(
        "--mit-vel-scale",
        type=float,
        default=1.0,
        help="vel_ref 배율. 0이면 속도 피드포워드를 끈다 — 원인 가려낼 때",
    )
    robot.add_argument(
        "--mit-kp-overrides",
        help='관절별 kp, 예: "joint2=30,joint3=20". 중력 부담이 다른 관절에 같은 kp를 '
        "주면 처짐이 제각각이 된다(처짐 = 중력토크/kp)",
    )
    robot.add_argument(
        "--move-speed-rate",
        type=int,
        help="컨트롤러 이동 속도 백분율(0~100, 기본 30). 팔의 실제 속도. "
        "--rate-limit / --max-relative-target은 명령값의 상한이라 이것과 다르다",
    )
    robot.add_argument(
        "--max-relative-target",
        type=float,
        help="명령이 실측 위치에서 벗어날 수 있는 최대치. smoothing의 rate-limit과 다른 것이다. "
        "기본값은 recording.env의 MAX_RELATIVE_TARGET",
    )
    robot.add_argument(
        "--camera-output-size",
        type=int,
        help="기본값은 recording.env의 HUMAN_APPROVED_CAMERA_OUTPUT_SIZE (없으면 512)",
    )
    robot.add_argument(
        "--top-crop", help="X,Y,SIZE. 기본값은 recording.env의 HUMAN_APPROVED_TOP_CROP"
    )
    robot.add_argument(
        "--wrist-crop", help="X,Y,SIZE. 기본값은 recording.env의 HUMAN_APPROVED_WRIST_CROP"
    )
    robot.add_argument("--env-file", type=pathlib.Path, default=DEFAULT_ENV_FILE)

    return parser.parse_args(argv)


def settings_from_args(args: argparse.Namespace) -> RunSettings:
    preset = mode_preset(args.mode)
    smoothing = dataclasses.replace(preset.smoothing)
    if args.temporal_ensemble is False:
        smoothing = dataclasses.replace(smoothing, temporal_ensemble=False)
    if args.ensemble_m is not None:
        smoothing = dataclasses.replace(smoothing, ensemble_m=args.ensemble_m)
    if args.ema_alpha is not None:
        smoothing = dataclasses.replace(smoothing, ema_alpha=args.ema_alpha)
    if args.rate_limit is not None:
        smoothing = dataclasses.replace(smoothing, rate_limit=args.rate_limit)

    # live 카메라 전처리는 recording.env를 기본으로 쓴다 — GUI와 CLI가 서로 다른
    # 크롭으로 돌면 정책이 학습 때와 다른 화면을 보게 된다.
    env = load_env_file(args.env_file)
    crops = resolve_crops(env, args.top_crop, args.wrist_crop)
    camera_output_size = args.camera_output_size or int(
        env.get("HUMAN_APPROVED_CAMERA_OUTPUT_SIZE", "512")
    )
    if args.source == "robot" and not crops:
        raise SystemExit(
            "[ERROR] source=robot에는 카메라 crop이 필요합니다. "
            f"{args.env_file}에 HUMAN_APPROVED_TOP_CROP / HUMAN_APPROVED_WRIST_CROP를 "
            "두거나 --top-crop / --wrist-crop으로 지정하세요."
        )

    overrides: dict[str, Any] = {
        "dataset_root": args.dataset_root,
        "policy_path": args.policy_path,
        "episode": args.episode,
        "task": args.task,
        "device": args.device,
        "source": args.source,
        "fps": args.fps,
        "infer_every": args.infer_every,
        "horizon": args.horizon,
        "max_steps": args.max_steps,
        "loop_dataset": args.loop_dataset,
        "rviz": args.rviz,
        "joint_state_topic": args.joint_state_topic,
        "apply_to_robot": args.apply_to_robot,
        "real_robot_confirm": args.real_robot_confirm,
        "park_on_exit": args.park_on_exit,
        "camera_output_size": camera_output_size,
        "move_mode": args.move_mode,
        "move_speed_rate": args.move_speed_rate,
        "lookahead_s": args.lookahead_s,
        "use_mit": args.mit,
        "mit_kp": args.mit_kp,
        "mit_kd": args.mit_kd,
        "mit_confirm": args.mit_confirm,
        "mit_kp_overrides": args.mit_kp_overrides,
        "mit_vel_smoothing": args.mit_vel_smoothing,
        "mit_vel_scale": args.mit_vel_scale,
        "max_relative_target": args.max_relative_target
        or (float(env["MAX_RELATIVE_TARGET"]) if "MAX_RELATIVE_TARGET" in env else None),
        "crops": crops,
        "smoothing": smoothing,
        "record_root": args.record_root,
        "record_repo_id": args.record_repo_id,
    }
    if args.record_dataset is not None:
        overrides["record_dataset"] = args.record_dataset
    if args.prompt_outcome is not None:
        overrides["prompt_outcome"] = args.prompt_outcome
    return RunSettings.from_mode(args.mode, **overrides)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = settings_from_args(args)

    if settings.apply_to_robot and not settings.real_robot_enabled():
        print(
            "[ERROR] 실물 전송을 켜려면 --source=robot --apply-to-robot "
            f"--real-robot-confirm={REAL_ROBOT_CONFIRM} 를 모두 지정해야 합니다.",
            file=sys.stderr,
        )
        return 2

    if settings.use_mit and settings.mit_confirm != MIT_CONFIRM:
        print(
            f"[ERROR] MIT는 토크 제어입니다. 켜려면 --mit-confirm={MIT_CONFIRM} 도 함께 주세요.\n"
            "        먼저 scripts/tools/piper_mit_probe.py로 관절 하나씩 확인하시길 권합니다.",
            file=sys.stderr,
        )
        return 2

    runner = InferenceRunner(settings, outcome_prompt=terminal_outcome_prompt)

    # SIGTERM으로 죽여도 파킹/토크 해제를 거치도록 KeyboardInterrupt와 같은 경로로
    # 보낸다. 기본 동작(즉시 종료)이면 팔이 그 자리에 늘어진다.
    def _on_terminate(_signum, _frame):
        raise KeyboardInterrupt

    with contextlib.suppress(ValueError):  # 메인 스레드가 아니면 등록 불가
        signal.signal(signal.SIGTERM, _on_terminate)

    runner.start()

    try:
        while True:
            kind, payload = runner.events.get()
            if kind == Event.LOG:
                print(payload, flush=True)
            elif kind == Event.STEP:
                data = payload
                print(
                    f"  step {data['step']:>4}  votes={data['votes']:>3} "
                    f"pending={data['pending']:>3}  infer={data['infer_ms']:>6.1f}ms"
                    + (f"  rec={data['recorded']}" if data["recorded"] else ""),
                    flush=True,
                )
            elif kind == Event.FINISHED:
                status = payload
                break
    except KeyboardInterrupt:
        # teleop_ui의 Stop 버튼이 프로세스 그룹에 SIGINT를 보낸다.
        print("\n[INTERRUPT] 중단 요청 — 정리 중…", flush=True)
        runner.stop_event.set()
        status = _drain_until_finished(runner)

    runner.join(timeout=SHUTDOWN_TIMEOUT_S)

    if runner.trajectory:
        trajectory = np.stack(runner.trajectory)
        metrics = smoothness_metrics(trajectory, fps=settings.fps)
        measured = runner.measured_fps()
        print(f"\n[METRICS] {metrics}")
        if measured:
            print(f"[METRICS] 실측 제어 주기 {measured:.2f}Hz (설정 {settings.fps:g})")
    if runner.recorded_path is not None:
        print(f"[RECORD] {runner.recorded_path}")

    return 0 if status in {"finished", "estop"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
