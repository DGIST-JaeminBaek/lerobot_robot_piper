#!/usr/bin/env python3
"""piper_infer_gui.py — 정책 추론을 실시간으로 보면서 smoothing을 조절하는 전용 GUI.

제어 루프 자체는 여기 없다 — piper_infer_runner.InferenceRunner에 있고, 이 파일은
그 위에 붙는 화면일 뿐이다. teleop_ui의 Infer 프리셋과 CLI도 같은 runner를 쓰므로
어느 경로로 돌리든 smoothing과 안전 게이트가 동일하게 적용된다.

기존 도구들과의 관계:
  piper_infer_preview.py         전체 episode를 한 번에 추론하고 끝나면 RViz 재생.
                                  실행 중 개입 불가, smoothing 없음.
  piper_offline_chunk_rollout.py 오프라인 분석 + 그래프 저장. 대화형 아님.
  piper_human_approved_inference 구간마다 터미널에서 사람이 승인. 안전하지만 느리고
                                  smoothing 파라미터를 바꾸려면 재시작해야 함.
  이 파일                        추론을 돌리면서 RViz로 궤적을 보고, smoothing
                                  파라미터를 실행 중에 바꾸고, 큰 빨간 버튼으로
                                  즉시 멈춘다.

모드(시연용/증강용)는 프리셋일 뿐이다 — 고르면 아래 값들이 세팅되지만 전부 화면에
그대로 보이고 개별로 덮어쓸 수 있다. 증강용을 고르면 롤아웃이 LeRobotDataset으로
기록되고 끝날 때 성공/실패를 묻는다.

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
import time
import tkinter as tk
from dataclasses import asdict
from tkinter import filedialog, messagebox, simpledialog, ttk

import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from action_smoothing import SmoothingConfig, smoothness_metrics  # noqa: E402
from piper_infer_runner import (  # noqa: E402
    DEFAULT_ENV_FILE,
    MODES,
    MOTOR_NAMES,
    REAL_ROBOT_CONFIRM,
    Event,
    InferenceRunner,
    RunSettings,
    mode_preset,
)

GEOMETRY_KEY = "infer_gui"


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
        self.worker: InferenceRunner | None = None
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
        # 명령 주파수는 학습 데이터 fps(보통 30)와 맞춘다. 낮추면 정책이 예측한
        # 동작이 그 비율만큼 슬로모션이 되고, 낮은 주파수 자체가 "움직였다 멈췄다"를
        # 반복해 물리적으로 끊겨 보인다 — 실물에서 확인된 증상이다.
        #
        # 예전 기본값은 6이었다. "추론이 1회 ~115ms라 6Hz가 한계"라는 판단이었는데
        # 전제가 틀렸다: chunk 하나가 이미 chunk_size(=50)스텝 분량을 담고 있어
        # 매 스텝 추론할 필요가 없다. infer_every로 추론 빈도만 낮추면 명령은 계속
        # 30Hz로 쏠 수 있고, 겹치는 chunk가 50/5=10개라 ensemble도 유지된다.
        self.var_fps = tk.DoubleVar(value=float(env.get("INFER_FPS", "30")))
        self.var_horizon = tk.IntVar(value=50)
        self.var_infer_every = tk.IntVar(value=int(env.get("INFER_EVERY", "5")))
        self.var_max_steps = tk.IntVar(value=0)
        self.var_loop_dataset = tk.BooleanVar(value=False)

        # 모드는 프리셋일 뿐이다 — 고르면 아래 var들이 세팅되지만 전부 화면에
        # 그대로 보이고 따로 바꿀 수 있다. 논문에 조건을 명시해야 하므로 값을
        # 모드 뒤에 숨기지 않는다.
        default_mode = env.get("INFER_MODE", "demo")
        self.var_mode = tk.StringVar(value=default_mode if default_mode in MODES else "demo")
        preset = mode_preset(self.var_mode.get())
        self.var_record = tk.BooleanVar(value=preset.record_dataset)
        self.var_record_raw = tk.BooleanVar(value=preset.record_raw_frames)
        self.var_prompt_outcome = tk.BooleanVar(value=preset.prompt_outcome)
        self.var_record_root = tk.StringVar(value="")

        self.var_ensemble = tk.BooleanVar(value=True)
        self.var_m = tk.DoubleVar(value=0.01)
        self.var_alpha = tk.DoubleVar(value=0.2)
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
        self._build_mode_panel(left)
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

    def _build_mode_panel(self, parent: ttk.Frame) -> None:
        """모드 프리셋 + 기록 설정.

        모드를 고르면 아래 체크박스들이 프리셋 값으로 바뀌지만, 그 뒤에 개별로
        고쳐도 된다. 값을 숨기지 않는 게 요점이다 — 논문에 실행 조건을 그대로
        옮겨 적어야 하기 때문.
        """
        frame = ttk.LabelFrame(parent, text="0. 모드", padding=8)
        frame.pack(fill="x", pady=(0, 6))

        row = ttk.Frame(frame)
        row.pack(fill="x")
        for preset in MODES.values():
            ttk.Radiobutton(
                row,
                text=f"{preset.label} ({preset.name})",
                value=preset.name,
                variable=self.var_mode,
                command=self._on_mode_changed,
            ).pack(side="left", padx=(0, 12))

        self.var_mode_hint = tk.StringVar()
        ttk.Label(frame, textvariable=self.var_mode_hint, foreground="#555", wraplength=380).pack(
            fill="x", pady=(4, 6)
        )

        ttk.Checkbutton(
            frame, text="롤아웃을 LeRobotDataset으로 기록", variable=self.var_record
        ).pack(anchor="w")
        ttk.Checkbutton(
            frame,
            text="크롭 전 원본 프레임으로 저장 (학습 재사용용)",
            variable=self.var_record_raw,
        ).pack(anchor="w")
        ttk.Checkbutton(
            frame, text="끝나면 성공/실패 묻기", variable=self.var_prompt_outcome
        ).pack(anchor="w")

        path_row = ttk.Frame(frame)
        path_row.pack(fill="x", pady=(4, 0))
        ttk.Label(path_row, text="기록 위치").pack(side="left")
        ttk.Entry(path_row, textvariable=self.var_record_root, width=32).pack(
            side="left", fill="x", expand=True, padx=4
        )
        ttk.Button(
            path_row, text="…", width=3,
            command=lambda: self._browse(self.var_record_root, True),
        ).pack(side="left")
        ttk.Label(
            frame, text="(비워두면 records/rollout/ 아래에 자동 생성)", foreground="#777"
        ).pack(anchor="w")

        self._on_mode_changed()

    def _on_mode_changed(self) -> None:
        preset = mode_preset(self.var_mode.get())
        self.var_record.set(preset.record_dataset)
        self.var_record_raw.set(preset.record_raw_frames)
        self.var_prompt_outcome.set(preset.prompt_outcome)
        self.var_ensemble.set(preset.smoothing.temporal_ensemble)
        self.var_m.set(preset.smoothing.ensemble_m)
        self.var_alpha.set(preset.smoothing.ema_alpha)
        self.var_rate_enabled.set(preset.smoothing.rate_limit is not None)
        if preset.smoothing.rate_limit is not None:
            self.var_rate_limit.set(preset.smoothing.rate_limit)
        self.var_mode_hint.set(preset.description)

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

    def _collect_settings(self) -> RunSettings:
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

        record_root = self.var_record_root.get().strip()
        # 모드는 프리셋일 뿐 — 화면의 값이 항상 이긴다. from_mode에 전부 넘겨서
        # 프리셋 기본값을 덮어쓴다.
        return RunSettings.from_mode(
            self.var_mode.get(),
            policy_path=policy_path,
            dataset_root=dataset_root,
            episode=int(self.var_episode.get()),
            task=self.var_task.get().strip(),
            device=self.var_device.get().strip() or "cuda",
            source=source,
            fps=float(self.var_fps.get()),
            horizon=int(self.var_horizon.get()),
            infer_every=int(self.var_infer_every.get()),
            max_steps=max(0, int(self.var_max_steps.get())),
            loop_dataset=self.var_loop_dataset.get(),
            smoothing=self._smoothing_config(),
            rviz=self.var_rviz.get(),
            joint_state_topic=self.var_topic.get().strip() or "/joint_states",
            apply_to_robot=apply_to_robot,
            real_robot_confirm=self.var_confirm.get().strip(),
            park_on_exit=self.var_park.get(),
            camera_output_size=int(self.var_camera_size.get()),
            crops={
                "top": self._parse_crop(self.var_top_crop.get()),
                "wrist": self._parse_crop(self.var_wrist_crop.get()),
            },
            record_dataset=self.var_record.get(),
            record_raw_frames=self.var_record_raw.get(),
            prompt_outcome=self.var_prompt_outcome.get(),
            record_root=pathlib.Path(record_root).expanduser() if record_root else None,
        )

    def _ask_outcome(self) -> tuple[str, str]:
        """롤아웃이 끝나면 성공/실패를 묻는다. runner 스레드에서 호출되므로
        Tk 위젯을 직접 만들 수 없다 — 메인 스레드에 넘기고 결과를 기다린다."""
        result: "queue.Queue[tuple[str, str]]" = queue.Queue(maxsize=1)

        def ask() -> None:
            answer = messagebox.askyesnocancel(
                "롤아웃 결과",
                "이 롤아웃을 성공으로 기록할까요?\n\n"
                "예 = 성공(success)\n아니오 = 실패(failure)\n취소 = 폐기(저장 안 함)",
            )
            if answer is None:
                result.put(("discard", ""))
                return
            outcome = "success" if answer else "failure"
            note = simpledialog.askstring("메모", "메모 (없으면 비워두세요):", parent=self) or ""
            result.put((outcome, note.strip()))

        self.after(0, ask)
        try:
            return result.get(timeout=300)
        except queue.Empty:
            return "unlabeled", ""

    # ── 액션 ─────────────────────────────────────────────────────
    def _start(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        try:
            settings = self._collect_settings()
        except Exception as error:
            messagebox.showerror("설정 오류", str(error))
            return

        if settings.apply_to_robot and not messagebox.askyesno(
            "실물 로봇 실행",
            "실제 Piper에 명령을 전송합니다.\n"
            "주변에 사람이 없고 비상 정지가 가능한지 확인했습니까?",
            icon="warning",
        ):
            return

        self.history.clear()
        self.step_count = 0
        self.text_log.delete("1.0", "end")
        self._log(f"[START] {settings.describe()}")
        if settings.smoothing.temporal_ensemble and settings.infer_every != 1:
            self._log("[WARN] infer_every != 1 — temporal ensemble 효과가 거의 없습니다")
        if settings.record_dataset and settings.source == "dataset":
            self._log(
                "[WARN] source=dataset을 기록하면 원본 dataset을 정책 출력으로 되쓰는 셈입니다 "
                "— 학습용 증강이 목적이면 source=robot으로 도세요"
            )

        self.worker = InferenceRunner(settings, self.events, outcome_prompt=self._ask_outcome)
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

    def _save_run(self, worker: InferenceRunner, status: str) -> None:
        settings = worker.settings
        trajectory = np.stack(worker.trajectory)
        output_dir = REPO_ROOT / "outputs" / "infer_gui"
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"run_{stamp}.npz"
        metrics = smoothness_metrics(trajectory, fps=settings.fps)
        np.savez_compressed(
            path,
            smoothed_actions=trajectory,
            raw_first_actions=(
                np.stack(worker.raw_trajectory) if worker.raw_trajectory else np.zeros((0, 7))
            ),
        )
        measured = worker.measured_fps()
        summary = {
            "status": status,
            "steps": int(len(trajectory)),
            "mode": settings.mode,
            "smoothing": asdict(settings.smoothing),
            "fps": settings.fps,
            "measured_fps": round(measured, 3) if measured else None,
            "infer_every": settings.infer_every,
            "policy_path": settings.policy_path,
            "recorded_dataset": str(worker.recorded_path) if worker.recorded_path else None,
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
        if measured:
            self._log(f"[METRICS] 실측 제어 주기 {measured:.2f}Hz (설정 {settings.fps:g})")

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
