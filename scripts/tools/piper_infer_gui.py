#!/usr/bin/env python3
"""piper_infer_gui.py — 정책 추론을 실시간으로 보면서 smoothing을 조절하는 전용 GUI.

기존 도구들과의 관계:
  piper_infer_preview.py         전체 episode를 한 번에 추론하고 끝나면 RViz 재생.
                                  실행 중 개입 불가, smoothing 없음.
  piper_offline_chunk_rollout.py 오프라인 분석 + 그래프 저장. 대화형 아님.
  piper_human_approved_inference 구간마다 터미널에서 사람이 승인. 안전하지만 느리고
                                  smoothing 파라미터를 바꾸려면 재시작해야 함.
  이 파일                        추론을 돌리면서 RViz로 궤적을 보고, smoothing
                                  파라미터를 실행 중에 바꾸고, 큰 빨간 버튼으로
                                  즉시 멈춘다.

기본값은 안전하다: source=dataset(로봇 미연결), APPLY_TO_ROBOT 없음. 실물 실행은
Robot 탭에서 source=robot + "실물 전송" 체크 + 확인 문구 입력을 모두 해야 켜진다.
실물 명령은 전부 PiperFollower.send_action()을 지나가므로 max_relative_target과
effort 안전 컷오프가 그대로 적용된다.

RViz는 이 GUI가 띄우지 않는다. 먼저 아래를 실행해 두면 궤적이 보인다:
    ros2 launch agx_arm_description display_piper.launch.py
    (또는 scripts/tools/piper_session.py --step rviz)

실행:
    python -m lerobot_robot_piper.infer_gui        # 래퍼
    python scripts/tools/piper_infer_gui.py        # 직접
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import queue
import sys
import threading
import time
import tkinter as tk
from dataclasses import asdict
from tkinter import filedialog, messagebox, ttk

import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from action_smoothing import SmoothingConfig, SmoothingPipeline, smoothness_metrics  # noqa: E402

MOTOR_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
REAL_ROBOT_CONFIRM = "I_UNDERSTAND_REAL_ROBOT"
DEFAULT_ENV_FILE = REPO_ROOT / "configs" / "recording.env"
GEOMETRY_KEY = "infer_gui"


# ═══════════════════════════════════════════════════════════════════
# worker가 GUI로 보내는 이벤트
# ═══════════════════════════════════════════════════════════════════
class Event:
    LOG = "log"
    STATUS = "status"
    STEP = "step"
    FINISHED = "finished"


# ═══════════════════════════════════════════════════════════════════
# RViz publisher — piper_infer_preview.run_rviz와 같은 변환/토픽을 쓴다.
# ═══════════════════════════════════════════════════════════════════
class RvizPublisher:
    def __init__(self, topic: str = "/joint_states") -> None:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState

        from piper_infer_preview import unnormalize_to_physical

        self._rclpy = rclpy
        self._JointState = JointState
        self._unnormalize = unnormalize_to_physical
        if not rclpy.ok():
            rclpy.init()
        self._node = Node("piper_infer_gui")
        self._publisher = self._node.create_publisher(JointState, topic, 10)
        self.topic = topic

    def publish(self, action: np.ndarray) -> None:
        message = self._JointState()
        message.header.stamp = self._node.get_clock().now().to_msg()
        message.name = MOTOR_NAMES
        message.position = [
            self._unnormalize(name, float(value))
            for name, value in zip(MOTOR_NAMES, action)
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


# ═══════════════════════════════════════════════════════════════════
# 추론 worker — GUI 스레드를 절대 건드리지 않고 큐로만 통신한다.
# ═══════════════════════════════════════════════════════════════════
class InferenceWorker(threading.Thread):
    def __init__(self, settings: dict, events: "queue.Queue[tuple[str, object]]") -> None:
        super().__init__(daemon=True)
        self.settings = settings
        self.events = events

        self.stop_event = threading.Event()
        self.estop_event = threading.Event()
        self.pause_event = threading.Event()
        # GUI가 실행 중에 바꿀 수 있는 값들 — lock으로 감싼다.
        self._config_lock = threading.Lock()
        self._pending_smoothing: SmoothingConfig | None = None

        self.trajectory: list[np.ndarray] = []
        self.raw_trajectory: list[np.ndarray] = []

    # ── GUI 쪽에서 호출 ──────────────────────────────────────────
    def apply_smoothing(self, config: SmoothingConfig) -> None:
        with self._config_lock:
            self._pending_smoothing = config

    def emergency_stop(self) -> None:
        self.estop_event.set()
        self.stop_event.set()
        self.pause_event.clear()

    # ── 내부 ─────────────────────────────────────────────────────
    def _log(self, message: str) -> None:
        self.events.put((Event.LOG, message))

    def _take_pending_smoothing(self) -> SmoothingConfig | None:
        with self._config_lock:
            pending, self._pending_smoothing = self._pending_smoothing, None
        return pending

    def run(self) -> None:  # noqa: C901 — 순차적 셋업이라 나누면 오히려 읽기 나쁨
        settings = self.settings
        robot = None
        rviz = None
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

            dataset_root = pathlib.Path(settings["dataset_root"]).expanduser().resolve()
            self._log(f"[LOAD] dataset={dataset_root} (episode {settings['episode']})")
            dataset = LeRobotDataset(
                repo_id=f"local/{dataset_root.name}",
                root=dataset_root,
                episodes=[settings["episode"]],
                video_backend="pyav",
            )
            camera_keys = list(dataset.meta.camera_keys)
            self._log(f"[LOAD] {dataset.num_frames} frames, cameras={camera_keys}")

            self._log(f"[LOAD] policy={settings['policy_path']} device={settings['device']}")
            config, policy, preprocessor, postprocessor = load_policy(
                settings["policy_path"], dataset.meta, settings["device"]
            )
            device = get_safe_torch_device(policy.config.device)
            policy.reset()
            chunk_size = int(getattr(config, "chunk_size", 0)) or settings["horizon"]
            horizon = min(settings["horizon"], chunk_size)
            self._log(f"[LOAD] policy chunk_size={chunk_size}, using horizon={horizon}")

            if settings["source"] == "robot":
                validate_live_camera_output_size(
                    dataset.features, camera_keys, settings["camera_output_size"]
                )
                from piper_human_approved_inference import build_robot_from_env

                self._log("[CONNECT] Piper follower + camera 연결 중…")
                robot_args = argparse.Namespace(
                    max_relative_target=settings["smoothing"].rate_limit
                )
                robot = build_robot_from_env(robot_args)
                robot.connect()
                self._log("[CONNECT] 연결 완료")

            if settings["rviz"]:
                try:
                    rviz = RvizPublisher(settings["joint_state_topic"])
                    self._log(f"[RVIZ] publishing to {rviz.topic}")
                except Exception as error:
                    self._log(f"[WARN] RViz publisher를 만들 수 없음 — 비활성화: {error}")
                    rviz = None

            task = settings["task"] or str(dataset[0].get("task", ""))
            self._log(f"[TASK] {task!r}")

            pipeline = SmoothingPipeline(settings["smoothing"])
            self._log(f"[SMOOTH] {settings['smoothing'].summary()}")

            fps = settings["fps"]
            period = 1.0 / fps
            infer_every = max(1, settings["infer_every"])
            max_steps = settings["max_steps"]
            cursor = 0
            step = 0
            first_state = None

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
                    self._log(f"[SMOOTH] 실행 중 변경 → {pending.summary()}")

                loop_started = time.perf_counter()

                # ── observation ────────────────────────────────
                if settings["source"] == "dataset":
                    if cursor >= dataset.num_frames:
                        if not settings["loop_dataset"]:
                            self._log("[STOP] dataset episode 끝")
                            break
                        cursor = 0
                    raw_observation = make_raw_observation(dataset, dataset[cursor])
                    observation_frame = cursor
                else:
                    raw_observation = preprocess_live_camera_observation(
                        robot.get_observation(),
                        camera_keys,
                        settings["crops"],
                        settings["camera_output_size"],
                    )
                    observation_frame = step
                measured_state = state_from_raw_observation(raw_observation, dataset.features)
                if first_state is None:
                    first_state = measured_state.copy()
                    pipeline.reset(first_state)

                # ── inference ──────────────────────────────────
                infer_seconds = 0.0
                if step % infer_every == 0 or pipeline.pending_steps == 0:
                    infer_started = time.perf_counter()
                    chunk = predict_chunk(
                        raw_observation=raw_observation,
                        dataset_features=dataset.features,
                        policy=policy,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        device=device,
                        task=task,
                    ).numpy().astype(np.float32, copy=False)[:horizon]
                    infer_seconds = time.perf_counter() - infer_started
                    if not np.isfinite(chunk).all():
                        self._log("[ERROR] 정책이 NaN/Inf를 출력 — 중단합니다")
                        status = "error"
                        break
                    self.raw_trajectory.append(chunk[0].copy())
                    pipeline.add_chunk(chunk)

                votes = pipeline.votes_for_next
                action = pipeline.next_action()
                self.trajectory.append(action.copy())

                # ── 출력 ───────────────────────────────────────
                if self.estop_event.is_set():
                    break
                if rviz is not None:
                    rviz.publish(action)
                safety_tripped = False
                if robot is not None and settings["apply_to_robot"]:
                    robot.send_action(
                        {f"{name}.pos": float(value) for name, value in zip(MOTOR_NAMES, action)}
                    )
                    safety_tripped = bool(robot.safety_tripped)

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
                if overrun > 0:
                    if step % 30 == 0:
                        self._log(
                            f"[TIMING] 루프가 {overrun * 1000:.0f}ms 늦음 "
                            f"— fps를 낮추거나 infer_every를 키우세요"
                        )
                else:
                    # 남은 시간을 잘게 나눠 자면서 E-stop 반응성을 유지한다.
                    deadline = loop_started + period
                    while time.perf_counter() < deadline and not self.stop_event.is_set():
                        time.sleep(min(0.005, deadline - time.perf_counter()))

            if self.estop_event.is_set():
                status = "estop"
                self._log("[E-STOP] 명령 전송을 즉시 중단했습니다")

        except Exception as error:  # worker 예외는 GUI에 그대로 보여준다
            status = "error"
            import traceback

            self._log(f"[ERROR] {type(error).__name__}: {error}")
            self._log(traceback.format_exc())
        finally:
            if robot is not None:
                try:
                    if getattr(robot, "is_connected", False):
                        park = self.settings["park_on_exit"] and not robot.safety_tripped
                        self._log(f"[DISCONNECT] park={park}")
                        robot.disconnect(park=park)
                except Exception as error:
                    self._log(f"[WARN] disconnect 실패: {error}")
            if rviz is not None:
                rviz.close()
            self.events.put((Event.FINISHED, status))


# ═══════════════════════════════════════════════════════════════════
# GUI
# ═══════════════════════════════════════════════════════════════════
class InferGui(tk.Tk):
    POLL_MS = 40
    PLOT_WINDOW = 300  # 그래프에 남길 최근 스텝 수

    def __init__(self, env: dict[str, str]) -> None:
        super().__init__()
        self.title("Piper Inference — smoothing / RViz / E-STOP")
        self.env = env
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: InferenceWorker | None = None
        self.history: list[np.ndarray] = []
        self.step_count = 0

        self._build_variables()
        self._build_widgets()
        self._restore_geometry()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(self.POLL_MS, self._pump_events)
        self.bind("<Escape>", lambda _event: self._emergency_stop())

    # ── 상태 변수 ────────────────────────────────────────────────
    def _build_variables(self) -> None:
        env = self.env
        default_policy = (
            env.get("HUMAN_APPROVED_POLICY_PATH")
            or env.get("PRETRAINED_NAME_OR_PATH")
            or str(
                REPO_ROOT
                / "outputs/train/smolvla_erase_shape_512/checkpoints/030000/pretrained_model"
            )
        )
        default_dataset = env.get(
            "HUMAN_APPROVED_DATASET_ROOT", str(REPO_ROOT / "records/0727/erase_the_shape_512")
        )

        self.var_policy = tk.StringVar(value=default_policy)
        self.var_dataset = tk.StringVar(value=default_dataset)
        self.var_episode = tk.IntVar(value=int(env.get("HUMAN_APPROVED_EPISODE", "0")))
        self.var_task = tk.StringVar(value=env.get("HUMAN_APPROVED_TASK", ""))
        self.var_device = tk.StringVar(value=env.get("POLICY_DEVICE", "cuda"))
        self.var_source = tk.StringVar(value="dataset")
        # 녹화 FPS(보통 30)와 분리한다. 이 머신의 SmolVLA 추론이 1회 ~150ms라
        # 30을 쓰면 매 루프가 밀려서 제어 주기가 들쭉날쭉해지고, 그 자체가 jerk가 된다.
        # 실측(30k 체크포인트, 40스텝): fps=8은 유지 불가(실제 6.65Hz, 주기 125→150ms),
        # fps=6은 5.99Hz에 지터 표준편차 7.5ms로 안정. 그래서 기본값을 6으로 둔다.
        self.var_fps = tk.DoubleVar(value=float(env.get("INFER_FPS", "6")))
        self.var_horizon = tk.IntVar(value=50)
        self.var_infer_every = tk.IntVar(value=1)
        self.var_max_steps = tk.IntVar(value=0)
        self.var_loop_dataset = tk.BooleanVar(value=False)

        self.var_ensemble = tk.BooleanVar(value=True)
        self.var_m = tk.DoubleVar(value=0.01)
        self.var_alpha = tk.DoubleVar(value=1.0)
        self.var_rate_limit = tk.DoubleVar(value=float(env.get("MAX_RELATIVE_TARGET", "5.0")))
        self.var_rate_enabled = tk.BooleanVar(value=True)

        self.var_rviz = tk.BooleanVar(value=True)
        self.var_topic = tk.StringVar(value="/joint_states")
        self.var_apply = tk.BooleanVar(value=False)
        self.var_confirm = tk.StringVar(value="")
        self.var_park = tk.BooleanVar(value=True)
        self.var_camera_size = tk.IntVar(
            value=int(env.get("HUMAN_APPROVED_CAMERA_OUTPUT_SIZE", "512"))
        )
        self.var_top_crop = tk.StringVar(value=env.get("HUMAN_APPROVED_TOP_CROP", "280,0,720"))
        self.var_wrist_crop = tk.StringVar(
            value=env.get("HUMAN_APPROVED_WRIST_CROP", "280,0,720")
        )

        self.var_status = tk.StringVar(value="대기 중")
        self.var_metrics = tk.StringVar(value="—")

    # ── 위젯 ─────────────────────────────────────────────────────
    def _build_widgets(self) -> None:
        root = ttk.Frame(self, padding=8)
        root.pack(fill="both", expand=True)

        self._build_control_bar(root)

        body = ttk.Frame(root)
        body.pack(fill="both", expand=True, pady=(8, 0))

        left = ttk.Frame(body)
        left.pack(side="left", fill="y")
        self._build_setup_panel(left)
        self._build_smoothing_panel(left)
        self._build_robot_panel(left)

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self._build_monitor_panel(right)
        self._build_log_panel(right)

    def _build_control_bar(self, parent: ttk.Frame) -> None:
        bar = ttk.Frame(parent)
        bar.pack(fill="x")

        self.button_start = tk.Button(
            bar, text="▶  Start", command=self._start, width=12,
            bg="#1f7a4d", fg="white", font=("TkDefaultFont", 11, "bold"), relief="raised",
        )
        self.button_start.pack(side="left")

        self.button_pause = tk.Button(
            bar, text="⏸  Pause", command=self._toggle_pause, width=12, state="disabled"
        )
        self.button_pause.pack(side="left", padx=(6, 0))

        self.button_stop = tk.Button(
            bar, text="⏹  Stop", command=self._stop, width=12, state="disabled"
        )
        self.button_stop.pack(side="left", padx=(6, 0))

        self.button_estop = tk.Button(
            bar, text="■  E-STOP  (Esc)", command=self._emergency_stop,
            bg="#b3261e", fg="white", activebackground="#8c1d18", activeforeground="white",
            font=("TkDefaultFont", 13, "bold"), width=18, height=2, relief="raised", bd=4,
        )
        self.button_estop.pack(side="right")

        ttk.Label(bar, textvariable=self.var_status, font=("TkDefaultFont", 10, "bold")).pack(
            side="right", padx=12
        )

    def _build_setup_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="1. 정책 / 관찰", padding=8)
        frame.pack(fill="x")

        def path_row(row: int, label: str, variable: tk.StringVar, directory: bool) -> None:
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w")
            entry = ttk.Entry(frame, textvariable=variable, width=44)
            entry.grid(row=row, column=1, sticky="we", padx=4)
            ttk.Button(
                frame, text="…", width=3,
                command=lambda: self._browse(variable, directory),
            ).grid(row=row, column=2)

        path_row(0, "checkpoint", self.var_policy, True)
        path_row(1, "dataset", self.var_dataset, True)

        ttk.Label(frame, text="episode").grid(row=2, column=0, sticky="w")
        ttk.Spinbox(frame, from_=0, to=9999, textvariable=self.var_episode, width=8).grid(
            row=2, column=1, sticky="w", padx=4
        )

        ttk.Label(frame, text="task").grid(row=3, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.var_task, width=44).grid(
            row=3, column=1, columnspan=2, sticky="we", padx=4
        )
        ttk.Label(frame, text="(비우면 dataset의 task 사용)", foreground="#666").grid(
            row=4, column=1, sticky="w", padx=4
        )

        ttk.Label(frame, text="source").grid(row=5, column=0, sticky="w")
        source_box = ttk.Frame(frame)
        source_box.grid(row=5, column=1, sticky="w", padx=4)
        ttk.Radiobutton(
            source_box, text="dataset (안전)", value="dataset", variable=self.var_source
        ).pack(side="left")
        ttk.Radiobutton(
            source_box, text="robot (실제 카메라)", value="robot", variable=self.var_source
        ).pack(side="left", padx=(8, 0))

        options = ttk.Frame(frame)
        options.grid(row=6, column=0, columnspan=3, sticky="we", pady=(6, 0))
        for column, (label, variable, width) in enumerate(
            [
                ("device", self.var_device, 6),
                ("fps", self.var_fps, 6),
                ("horizon", self.var_horizon, 5),
                ("infer_every", self.var_infer_every, 5),
                ("max_steps", self.var_max_steps, 7),
            ]
        ):
            ttk.Label(options, text=label).grid(row=0, column=column * 2, sticky="e", padx=(0, 2))
            ttk.Entry(options, textvariable=variable, width=width).grid(
                row=0, column=column * 2 + 1, padx=(0, 8)
            )
        ttk.Checkbutton(
            frame, text="dataset 끝에서 처음으로 되감기", variable=self.var_loop_dataset
        ).grid(row=7, column=0, columnspan=3, sticky="w", pady=(4, 0))
        frame.columnconfigure(1, weight=1)

    def _build_smoothing_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="2. Smoothing (실행 중 변경 가능)", padding=8)
        frame.pack(fill="x", pady=(8, 0))

        ttk.Checkbutton(
            frame, text="Temporal ensemble (chunk 간 예측 평균)",
            variable=self.var_ensemble, command=self._on_smoothing_changed,
        ).grid(row=0, column=0, columnspan=3, sticky="w")

        self._slider_row(
            frame, 1, "decay m", self.var_m, 0.005, 2.0,
            "작을수록 부드러움 (0.01=ACT 기본), 클수록 반응 빠름",
        )
        self._slider_row(
            frame, 3, "EMA α", self.var_alpha, 0.05, 1.0,
            "1.0=끔. 0.3~0.5면 고주파 노이즈가 확 줄어듦",
        )

        rate_row = ttk.Frame(frame)
        rate_row.grid(row=5, column=0, columnspan=3, sticky="we", pady=(6, 0))
        ttk.Checkbutton(
            rate_row, text="rate limit", variable=self.var_rate_enabled,
            command=self._on_smoothing_changed,
        ).pack(side="left")
        ttk.Scale(
            rate_row, from_=0.5, to=20.0, variable=self.var_rate_limit,
            orient="horizontal", length=170, command=lambda _v: self._on_smoothing_changed(),
        ).pack(side="left", padx=6)
        self.label_rate = ttk.Label(rate_row, width=6)
        self.label_rate.pack(side="left")
        ttk.Label(
            frame, text="스텝당 최대 변화(정규화 단위) — 튀는 예측을 잘라내는 안전장치",
            foreground="#666", wraplength=380,
        ).grid(row=6, column=0, columnspan=3, sticky="w")

        ttk.Button(
            frame, text="현재 설정을 실행 중인 추론에 적용", command=self._apply_smoothing_now
        ).grid(row=7, column=0, columnspan=3, sticky="we", pady=(8, 0))

        ttk.Label(
            frame,
            text="※ temporal ensemble은 infer_every=1일 때만 제대로 동작합니다 "
                 "(매 스텝 새 chunk를 예측해야 겹치는 예측이 생김).",
            foreground="#8a5a00", wraplength=380,
        ).grid(row=8, column=0, columnspan=3, sticky="w", pady=(6, 0))
        frame.columnconfigure(1, weight=1)
        self._on_smoothing_changed()

    def _slider_row(
        self, parent: ttk.Frame, row: int, label: str, variable: tk.DoubleVar,
        low: float, high: float, hint: str,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        ttk.Scale(
            parent, from_=low, to=high, variable=variable, orient="horizontal", length=170,
            command=lambda _value: self._on_smoothing_changed(),
        ).grid(row=row, column=1, sticky="we", padx=4)
        value_label = ttk.Label(parent, width=6)
        value_label.grid(row=row, column=2, sticky="w")
        setattr(self, f"label_{label.split()[0].lower()}", value_label)
        ttk.Label(parent, text=hint, foreground="#666", wraplength=380).grid(
            row=row + 1, column=0, columnspan=3, sticky="w"
        )

    def _build_robot_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="3. 출력", padding=8)
        frame.pack(fill="x", pady=(8, 0))

        ttk.Checkbutton(frame, text="RViz publish", variable=self.var_rviz).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Entry(frame, textvariable=self.var_topic, width=18).grid(
            row=0, column=1, sticky="w", padx=4
        )

        ttk.Separator(frame, orient="horizontal").grid(
            row=1, column=0, columnspan=3, sticky="we", pady=6
        )

        ttk.Checkbutton(
            frame, text="실물 Piper에 전송 (source=robot 필요)",
            variable=self.var_apply, command=self._on_apply_changed,
        ).grid(row=2, column=0, columnspan=3, sticky="w")
        ttk.Label(frame, text=f"확인 문구 {REAL_ROBOT_CONFIRM}", foreground="#b3261e").grid(
            row=3, column=0, columnspan=3, sticky="w"
        )
        self.entry_confirm = ttk.Entry(frame, textvariable=self.var_confirm, width=34)
        self.entry_confirm.grid(row=4, column=0, columnspan=3, sticky="we", pady=(2, 0))
        ttk.Checkbutton(frame, text="종료 시 parking", variable=self.var_park).grid(
            row=5, column=0, columnspan=3, sticky="w", pady=(4, 0)
        )

        live = ttk.Frame(frame)
        live.grid(row=6, column=0, columnspan=3, sticky="we", pady=(6, 0))
        ttk.Label(live, text="crop top/wrist").pack(side="left")
        ttk.Entry(live, textvariable=self.var_top_crop, width=11).pack(side="left", padx=3)
        ttk.Entry(live, textvariable=self.var_wrist_crop, width=11).pack(side="left")
        ttk.Label(live, text="size").pack(side="left", padx=(6, 2))
        ttk.Entry(live, textvariable=self.var_camera_size, width=5).pack(side="left")
        frame.columnconfigure(2, weight=1)

    def _build_monitor_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="실시간 모니터", padding=8)
        frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(frame, height=230, bg="#12151a", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        ttk.Label(frame, textvariable=self.var_metrics, font=("TkFixedFont", 9)).pack(
            anchor="w", pady=(6, 0)
        )

        self.bars: dict[str, tuple[ttk.Progressbar, ttk.Label]] = {}
        grid = ttk.Frame(frame)
        grid.pack(fill="x", pady=(6, 0))
        for index, name in enumerate(MOTOR_NAMES):
            ttk.Label(grid, text=name, width=8).grid(row=index, column=0, sticky="w")
            bar = ttk.Progressbar(grid, maximum=200, length=180)
            bar.grid(row=index, column=1, sticky="we", padx=4)
            value = ttk.Label(grid, text="—", width=8, font=("TkFixedFont", 9))
            value.grid(row=index, column=2, sticky="w")
            self.bars[name] = (bar, value)
        grid.columnconfigure(1, weight=1)

    def _build_log_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="로그", padding=4)
        frame.pack(fill="both", expand=True, pady=(8, 0))
        self.text_log = tk.Text(frame, height=10, wrap="none", font=("TkFixedFont", 9))
        scroll = ttk.Scrollbar(frame, command=self.text_log.yview)
        self.text_log.configure(yscrollcommand=scroll.set)
        self.text_log.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    # ── 설정 수집/검증 ───────────────────────────────────────────
    def _smoothing_config(self) -> SmoothingConfig:
        return SmoothingConfig(
            temporal_ensemble=self.var_ensemble.get(),
            ensemble_m=float(self.var_m.get()),
            ema_alpha=float(self.var_alpha.get()),
            rate_limit=float(self.var_rate_limit.get()) if self.var_rate_enabled.get() else None,
            clip_to_range=True,
        )

    def _parse_crop(self, text: str) -> object:
        from piper_human_approved_inference import parse_camera_crop

        return parse_camera_crop(text)

    def _collect_settings(self) -> dict:
        policy_path = self.var_policy.get().strip()
        dataset_root = pathlib.Path(self.var_dataset.get().strip()).expanduser()
        if not policy_path:
            raise ValueError("checkpoint 경로를 지정하세요.")
        if not pathlib.Path(policy_path).expanduser().exists():
            raise ValueError(f"checkpoint 경로가 없습니다: {policy_path}")
        if not dataset_root.exists():
            raise ValueError(f"dataset 경로가 없습니다: {dataset_root}")
        if self.var_fps.get() <= 0:
            raise ValueError("fps는 양수여야 합니다.")
        if self.var_horizon.get() <= 0:
            raise ValueError("horizon은 양수여야 합니다.")

        apply_to_robot = self.var_apply.get()
        source = self.var_source.get()
        if apply_to_robot:
            if source != "robot":
                raise ValueError("실물 전송은 source=robot에서만 가능합니다.")
            if self.var_confirm.get().strip() != REAL_ROBOT_CONFIRM:
                raise ValueError(f"실물 전송에는 확인 문구 {REAL_ROBOT_CONFIRM} 입력이 필요합니다.")
            if os.environ.get("SAFETY_ENABLED", "true").lower() in {"0", "false", "no", "off"}:
                raise ValueError("실물 전송에는 SAFETY_ENABLED=true가 필요합니다.")
            # hold와 park 둘 다 "정책 명령은 무시하되 토크는 유지"라 안전하다
            # (config_piper.py의 safety_on_overload 설명 참고). hold는 트립 자세에서
            # 그대로 정지, park는 parking 자세로 천천히 복귀한 뒤 정지.
            overload_mode = os.environ.get("SAFETY_ON_OVERLOAD", "park").strip().lower()
            if overload_mode not in {"hold", "park"}:
                raise ValueError(
                    f"실물 전송에는 SAFETY_ON_OVERLOAD가 hold 또는 park여야 합니다 "
                    f"(현재 {overload_mode!r})."
                )

        return {
            "policy_path": policy_path,
            "dataset_root": str(dataset_root),
            "episode": int(self.var_episode.get()),
            "task": self.var_task.get().strip(),
            "device": self.var_device.get().strip() or "cuda",
            "source": source,
            "fps": float(self.var_fps.get()),
            "horizon": int(self.var_horizon.get()),
            "infer_every": int(self.var_infer_every.get()),
            "max_steps": max(0, int(self.var_max_steps.get())),
            "loop_dataset": self.var_loop_dataset.get(),
            "smoothing": self._smoothing_config(),
            "rviz": self.var_rviz.get(),
            "joint_state_topic": self.var_topic.get().strip() or "/joint_states",
            "apply_to_robot": apply_to_robot,
            "park_on_exit": self.var_park.get(),
            "camera_output_size": int(self.var_camera_size.get()),
            "crops": {
                "top": self._parse_crop(self.var_top_crop.get()),
                "wrist": self._parse_crop(self.var_wrist_crop.get()),
            },
        }

    # ── 액션 ─────────────────────────────────────────────────────
    def _start(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        try:
            settings = self._collect_settings()
        except Exception as error:
            messagebox.showerror("설정 오류", str(error))
            return

        if settings["apply_to_robot"] and not messagebox.askyesno(
            "실물 로봇 실행",
            "실제 Piper에 명령을 전송합니다.\n"
            "주변에 사람이 없고 비상 정지가 가능한지 확인했습니까?",
            icon="warning",
        ):
            return

        self.history.clear()
        self.step_count = 0
        self.text_log.delete("1.0", "end")
        self._log(f"[START] {settings['smoothing'].summary()}")
        if settings["smoothing"].temporal_ensemble and settings["infer_every"] != 1:
            self._log("[WARN] infer_every != 1 — temporal ensemble 효과가 거의 없습니다")

        self.worker = InferenceWorker(settings, self.events)
        self.worker.start()
        self._set_running(True)
        self.var_status.set("실행 중")

    def _toggle_pause(self) -> None:
        if self.worker is None:
            return
        if self.worker.pause_event.is_set():
            self.worker.pause_event.clear()
            self.button_pause.configure(text="⏸  Pause")
            self.var_status.set("실행 중")
        else:
            self.worker.pause_event.set()
            self.button_pause.configure(text="▶  Resume")
            self.var_status.set("일시정지")

    def _stop(self) -> None:
        if self.worker is not None:
            self.worker.stop_event.set()
            self.worker.pause_event.clear()
            self.var_status.set("정지 중…")

    def _emergency_stop(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.worker.emergency_stop()
            self.var_status.set("E-STOP")
            self._log("[E-STOP] 사용자 요청")
        else:
            self._log("[E-STOP] 실행 중인 추론이 없습니다")

    def _apply_smoothing_now(self) -> None:
        config = self._smoothing_config()
        if self.worker is not None and self.worker.is_alive():
            self.worker.apply_smoothing(config)
            self._log(f"[SMOOTH] 적용 요청: {config.summary()}")
        else:
            self._log(f"[SMOOTH] 다음 Start에 적용될 설정: {config.summary()}")

    def _browse(self, variable: tk.StringVar, directory: bool) -> None:
        current = variable.get() or str(REPO_ROOT)
        initial = pathlib.Path(current)
        initial = initial if initial.is_dir() else initial.parent
        chosen = (
            filedialog.askdirectory(initialdir=str(initial))
            if directory
            else filedialog.askopenfilename(initialdir=str(initial))
        )
        if chosen:
            variable.set(chosen)

    def _on_apply_changed(self) -> None:
        if self.var_apply.get():
            self.var_source.set("robot")

    def _on_smoothing_changed(self) -> None:
        # 위젯을 만드는 도중에도 Scale의 command가 발화하므로, 아직 생성되지 않은
        # 라벨은 조용히 건너뛴다.
        if hasattr(self, "label_decay"):
            self.label_decay.configure(text=f"{self.var_m.get():.3f}")
        if hasattr(self, "label_ema"):
            self.label_ema.configure(text=f"{self.var_alpha.get():.2f}")
        if hasattr(self, "label_rate"):
            self.label_rate.configure(
                text=f"{self.var_rate_limit.get():.1f}" if self.var_rate_enabled.get() else "off"
            )

    def _set_running(self, running: bool) -> None:
        self.button_start.configure(state="disabled" if running else "normal")
        self.button_pause.configure(state="normal" if running else "disabled", text="⏸  Pause")
        self.button_stop.configure(state="normal" if running else "disabled")

    # ── 이벤트 펌프 ──────────────────────────────────────────────
    def _pump_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == Event.LOG:
                    self._log(str(payload))
                elif kind == Event.STEP:
                    self._on_step(payload)
                elif kind == Event.FINISHED:
                    self._on_finished(str(payload))
        except queue.Empty:
            pass
        self.after(self.POLL_MS, self._pump_events)

    def _on_step(self, payload: dict) -> None:
        action = payload["action"]
        self.history.append(action)
        if len(self.history) > self.PLOT_WINDOW:
            self.history.pop(0)
        self.step_count = payload["step"]

        for index, name in enumerate(MOTOR_NAMES):
            bar, label = self.bars[name]
            value = float(action[index])
            bar["value"] = value + 100.0 if name != "gripper" else value * 2.0
            label.configure(text=f"{value:7.2f}")

        if len(self.history) >= 4:
            metrics = smoothness_metrics(np.stack(self.history), fps=float(self.var_fps.get()))
            self.var_metrics.set(
                f"step {payload['step']:5d} | obs {payload['observation_frame']:5d} | "
                f"votes {payload['votes']:3d} | pending {payload['pending']:3d} | "
                f"infer {payload['infer_ms']:6.1f}ms | clamp {payload['rate_clamp']:5.2f}\n"
                f"TV {metrics['total_variation']:7.3f} | max step {metrics['max_step']:6.2f} | "
                f"RMS jerk {metrics['rms_jerk']:10.1f}   (작을수록 부드러움)"
            )
        self._redraw_plot()

    def _redraw_plot(self) -> None:
        canvas = self.canvas
        canvas.delete("all")
        if len(self.history) < 2:
            return
        width = max(canvas.winfo_width(), 10)
        height = max(canvas.winfo_height(), 10)
        data = np.stack(self.history)
        colors = ["#e45756", "#f58518", "#eeca3b", "#54a24b", "#4c78a8", "#b279a2", "#9d755d"]

        for value in (-100, -50, 0, 50, 100):
            y = height * (1.0 - (value + 100) / 200.0)
            canvas.create_line(0, y, width, y, fill="#2a2f3a")
            canvas.create_text(4, y - 7, text=str(value), fill="#4a5162", anchor="w", font=("TkFixedFont", 7))

        step_x = width / max(1, len(data) - 1)
        for joint in range(7):
            series = data[:, joint]
            points = []
            for index, value in enumerate(series):
                y = height * (1.0 - (float(np.clip(value, -100, 100)) + 100) / 200.0)
                points.extend([index * step_x, y])
            canvas.create_line(*points, fill=colors[joint], width=1.6, smooth=False)
        canvas.create_text(
            width - 6, 8, text="정규화 action 목표 (최근 %d step)" % len(data),
            fill="#7a828f", anchor="ne", font=("TkDefaultFont", 8),
        )

    def _on_finished(self, status: str) -> None:
        self._set_running(False)
        self.var_status.set({"estop": "E-STOP 후 정지", "error": "오류로 종료"}.get(status, "종료"))
        worker, self.worker = self.worker, None
        if worker is None:
            return
        if worker.trajectory:
            self._save_run(worker, status)

    def _save_run(self, worker: InferenceWorker, status: str) -> None:
        trajectory = np.stack(worker.trajectory)
        output_dir = REPO_ROOT / "outputs" / "infer_gui"
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"run_{stamp}.npz"
        metrics = smoothness_metrics(trajectory, fps=worker.settings["fps"])
        np.savez_compressed(
            path,
            smoothed_actions=trajectory,
            raw_first_actions=(
                np.stack(worker.raw_trajectory) if worker.raw_trajectory else np.zeros((0, 7))
            ),
        )
        summary = {
            "status": status,
            "steps": int(len(trajectory)),
            "smoothing": asdict(worker.settings["smoothing"]),
            "fps": worker.settings["fps"],
            "infer_every": worker.settings["infer_every"],
            "policy_path": worker.settings["policy_path"],
            "metrics": metrics,
        }
        (output_dir / f"run_{stamp}.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self._log(f"[SAVE] {path}")
        self._log(
            f"[METRICS] TV={metrics['total_variation']:.4f} "
            f"max_step={metrics['max_step']:.3f} rms_jerk={metrics['rms_jerk']:.1f}"
        )

    def _log(self, message: str) -> None:
        self.text_log.insert("end", message.rstrip() + "\n")
        self.text_log.see("end")

    # ── 창 위치 저장 ─────────────────────────────────────────────
    def _restore_geometry(self) -> None:
        try:
            from lerobot_robot_piper.ui import _load_geometry

            geometry = _load_geometry(GEOMETRY_KEY)
        except Exception:
            geometry = None
        self.geometry(geometry or "1180x900")

    def _on_close(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            if not messagebox.askyesno("종료", "추론이 실행 중입니다. 정지하고 종료할까요?"):
                return
            self.worker.emergency_stop()
            self.worker.join(timeout=5.0)
        try:
            from lerobot_robot_piper.ui import _save_geometry

            _save_geometry(GEOMETRY_KEY, self.geometry())
        except Exception:
            pass
        self.destroy()


# ═══════════════════════════════════════════════════════════════════
def load_env_file(path: pathlib.Path) -> dict[str, str]:
    """configs/recording.env를 읽어 dict로 돌려주고 os.environ에도 채운다.

    이미 export된 값은 덮어쓰지 않는다(piper_human_approved_inference와 동일).
    """
    env: dict[str, str] = {}
    if not path.is_file():
        return env
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("'\"")
        if key:
            env[key] = value
            os.environ.setdefault(key, value)
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description="Piper 추론 실시간 GUI (smoothing / RViz / E-STOP)")
    parser.add_argument("--env-file", type=pathlib.Path, default=DEFAULT_ENV_FILE)
    args = parser.parse_args()

    env = load_env_file(args.env_file.expanduser().resolve())
    app = InferGui(env)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
