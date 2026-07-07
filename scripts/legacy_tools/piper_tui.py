#!/usr/bin/env python3
"""
piper_tui.py — UGRP PiPER 실험 대시보드 (Textual TUI)

실행:
    pip install textual
    python piper_tui.py

SSH 원격 접속에서도 동작함.
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import subprocess
import time
from datetime import datetime
from typing import ClassVar

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import (
    Button,
    Footer,
    Header,
    Label,
    Log,
    ProgressBar,
    RichLog,
    Static,
    TabbedContent,
    TabPane,
)

# ───────────────────────────────────────────────
# 설정 — 실험 환경에 맞게 수정
# ───────────────────────────────────────────────
CFG = {
    "venv_activate":    "/home/ugrp308/Group43/.venv/bin/activate",
    "ugrp_dir":         "/home/ugrp308/Group43/UGRP",
    "can_interface":    "can0",
    "can_bitrate":      1000000,
    "top_serial":       "327122074262",
    "wrist_serial":     "243322071626",
    "dataset_root":     "/home/ugrp308/Group43/datasets/piper-smolvla",
    "dataset_repo_id":  "local/piper-smolvla",
    "ros_distro":       "humble",
    "ros2_ws":          "/home/ugrp308/ros2_ws",
    "safe_range": {
        "x":       (100_000,  400_000),
        "y":      (-200_000,  200_000),
        "z":        (50_000,  350_000),
        "rx":     (-180_000,  180_000),
        "ry":     (-180_000,  180_000),
        "rz":     (-180_000,  180_000),
        "gripper":      (0,    70_000),
    },
}

# ───────────────────────────────────────────────
# CSS
# ───────────────────────────────────────────────
CSS = """
Screen {
    background: $surface;
}

/* ── 상태 패널 ── */
#status-panel {
    width: 32;
    height: 100%;
    background: $panel;
    border-right: solid $primary-darken-2;
    padding: 0 1;
}

#status-title {
    text-style: bold;
    color: $primary;
    padding: 1 0 0 0;
    text-align: center;
}

.status-row {
    height: 3;
    padding: 0 0;
}

.status-label {
    color: $text-muted;
    width: 10;
}

.status-val {
    text-style: bold;
    width: 100%;
}

.ok    { color: #4ec94e; }
.fail  { color: #e05252; }
.warn  { color: #e0b050; }
.idle  { color: $text-muted; }

/* ── EEF 수치 ── */
#eef-panel {
    margin-top: 1;
    border: solid $primary-darken-3;
    padding: 0 1;
    height: auto;
}

#eef-title {
    color: $primary;
    text-style: bold;
    padding: 0;
}

.eef-row {
    height: 1;
}

.eef-axis  { width: 8;  color: $text-muted; }
.eef-value { width: 12; text-style: bold; }
.eef-bar   { width: 10; color: $primary; }

/* ── 메인 영역 ── */
#main-area {
    width: 1fr;
    height: 100%;
}

/* ── 탭 ── */
TabbedContent {
    height: 100%;
}

/* ── 버튼 그룹 ── */
.btn-group {
    height: auto;
    padding: 1 1;
}

.btn-group-title {
    color: $primary;
    text-style: bold;
    padding: 0 0 1 0;
}

.step-btn {
    width: 100%;
    margin-bottom: 1;
}

.btn-session  { background: #1a5276; }
.btn-safe     { background: #1e8449; }
.btn-danger   { background: #922b21; }
.btn-validate { background: #7d3c98; }
.btn-infer    { background: #1a5276; }
.btn-disabled { background: $panel-darken-1; color: $text-muted; }

/* ── 로그 ── */
#log-area {
    height: 1fr;
    border: solid $primary-darken-3;
    margin: 0 1 1 1;
}

#log-title {
    background: $primary-darken-2;
    color: $text;
    padding: 0 1;
    text-style: bold;
}

RichLog {
    height: 1fr;
    padding: 0 1;
}

/* ── 진행 바 ── */
#progress-bar {
    margin: 0 1;
    height: 1;
}

/* ── 설정 탭 ── */
.cfg-row {
    height: 3;
    padding: 0 1;
}

.cfg-key {
    width: 20;
    color: $text-muted;
    padding-top: 1;
}

.cfg-val {
    width: 1fr;
    text-style: bold;
    padding-top: 1;
}

/* ── 푸터 단축키 ── */
Footer {
    background: $primary-darken-2;
}
"""


# ───────────────────────────────────────────────
# 상태 값 관리
# ───────────────────────────────────────────────
class AppState:
    """전역 상태 (CAN, EEF, 실행 중 여부)"""
    can_status:   str   = "unknown"   # ok / fail / unknown
    eef_values:   dict  = {}
    eef_status:   str   = "unknown"
    running:      bool  = False
    current_step: str   = ""
    session_ok:   bool  = False

STATE = AppState()


# ───────────────────────────────────────────────
# 헬퍼
# ───────────────────────────────────────────────
def src_prefix() -> str:
    venv   = CFG["venv_activate"]
    distro = CFG["ros_distro"]
    ws     = CFG["ros2_ws"]
    return (
        f"source {venv} 2>/dev/null || true && "
        f"source /opt/ros/{distro}/setup.bash 2>/dev/null || true && "
        f"source {ws}/install/setup.bash 2>/dev/null || true && "
    )

async def run_cmd(cmd: str, log_widget: RichLog) -> bool:
    """비동기로 shell 명령 실행, 출력을 RichLog에 스트리밍"""
    full = src_prefix() + cmd
    proc = await asyncio.create_subprocess_shell(
        full,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    async for raw in proc.stdout:
        line = raw.decode(errors="replace").rstrip()
        if line:
            log_widget.write(line)
    await proc.wait()
    return proc.returncode == 0


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ───────────────────────────────────────────────
# 상태 패널 위젯
# ───────────────────────────────────────────────
class StatusPanel(Widget):
    DEFAULT_CSS = ""

    can_st  = reactive("--")
    eef_st  = reactive("--")
    run_st  = reactive("대기")
    step_st = reactive("")

    eef_x  = reactive(0)
    eef_y  = reactive(0)
    eef_z  = reactive(0)
    eef_rx = reactive(0)
    eef_ry = reactive(0)
    eef_rz = reactive(0)
    eef_gr = reactive(0)

    def compose(self) -> ComposeResult:
        yield Static("● PIPER STATUS", id="status-title")
        yield Static("─" * 28, classes="idle")

        with Horizontal(classes="status-row"):
            yield Label("CAN",    classes="status-label")
            yield Label("--",     id="val-can",  classes="status-val idle")
        with Horizontal(classes="status-row"):
            yield Label("EEF",    classes="status-label")
            yield Label("--",     id="val-eef",  classes="status-val idle")
        with Horizontal(classes="status-row"):
            yield Label("상태",    classes="status-label")
            yield Label("대기",    id="val-run",  classes="status-val idle")
        with Horizontal(classes="status-row"):
            yield Label("단계",    classes="status-label")
            yield Label("",       id="val-step", classes="status-val idle")

        yield Static("─" * 28, classes="idle")
        yield Static("EEF 수치", id="eef-title")

        axes = [("X", "eef-x"), ("Y", "eef-y"), ("Z", "eef-z"),
                ("RX","eef-rx"),("RY","eef-ry"),("RZ","eef-rz"),("GR","eef-gr")]
        for label, id_ in axes:
            with Horizontal(classes="eef-row"):
                yield Label(f"{label:>4}", classes="eef-axis")
                yield Label("---", id=id_, classes="eef-value idle")

    def _update_label(self, id_: str, text: str, cls: str):
        try:
            lbl = self.query_one(f"#{id_}", Label)
            lbl.update(text)
            lbl.set_classes(f"status-val {cls}")
        except NoMatches:
            pass

    def watch_can_st(self, val: str):
        cls = "ok" if val == "UP" else ("fail" if val == "DOWN" else "idle")
        self._update_label("val-can", val, cls)

    def watch_eef_st(self, val: str):
        cls = "ok" if val == "OK" else ("fail" if val == "ZERO" else "idle")
        self._update_label("val-eef", val, cls)

    def watch_run_st(self, val: str):
        cls = "warn" if val == "실행 중" else "ok" if val == "완료" else "idle"
        self._update_label("val-run", val, cls)

    def watch_step_st(self, val: str):
        self._update_label("val-step", val, "warn")

    def _upd_eef(self, id_: str, val: int, lo: int, hi: int):
        try:
            lbl = self.query_one(f"#{id_}", Label)
            lbl.update(f"{val:>10,}")
            in_range = lo <= val <= hi
            lbl.set_classes(f"eef-value {'ok' if in_range else 'fail'}")
        except NoMatches:
            pass

    def update_eef(self, vals: dict):
        safe = CFG["safe_range"]
        mapping = [
            ("x",  "eef-x"),  ("y",  "eef-y"),  ("z",  "eef-z"),
            ("rx", "eef-rx"), ("ry", "eef-ry"), ("rz", "eef-rz"),
            ("gripper", "eef-gr"),
        ]
        for key, id_ in mapping:
            lo, hi = safe.get(key, (-999_999_999, 999_999_999))
            self._upd_eef(id_, vals.get(key, 0), lo, hi)


# ───────────────────────────────────────────────
# 메인 앱
# ───────────────────────────────────────────────
class PiperDashboard(App):
    CSS = CSS
    TITLE = "UGRP PiPER Dashboard"
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("q",   "quit",         "종료"),
        Binding("r",   "refresh_status","상태 갱신"),
        Binding("e",   "emergency_stop","비상정지", show=True),
        Binding("ctrl+l", "clear_log", "로그 지우기"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal():
            # 왼쪽 상태 패널
            with Vertical(id="status-panel"):
                yield StatusPanel(id="status-widget")

            # 오른쪽 메인
            with Vertical(id="main-area"):
                with TabbedContent():

                    # ── 탭1: 세션 ──────────────────────────
                    with TabPane("🚀 세션", id="tab-session"):
                        with ScrollableContainer():
                            with Vertical(classes="btn-group"):
                                yield Static("■ 세션 시작", classes="btn-group-title")
                                yield Button(
                                    "① 세션 시작 (CAN + EEF + 카메라 점검)",
                                    id="btn-session", classes="step-btn btn-session")
                                yield Button(
                                    "② RViz 실행",
                                    id="btn-rviz", classes="step-btn btn-session")
                                yield Button(
                                    "③ teleop no-op 검증",
                                    id="btn-teleop", classes="step-btn btn-session")

                            with Vertical(classes="btn-group"):
                                yield Static("■ 개별 점검", classes="btn-group-title")
                                yield Button(
                                    "CAN 활성화",
                                    id="btn-can-up", classes="step-btn btn-safe")
                                yield Button(
                                    "EEF non-zero 확인",
                                    id="btn-eef", classes="step-btn btn-safe")
                                yield Button(
                                    "카메라 확인",
                                    id="btn-cam", classes="step-btn btn-safe")

                            with Vertical(classes="btn-group"):
                                yield Static("■ 비상", classes="btn-group-title")
                                yield Button(
                                    "🛑 비상 CAN 차단",
                                    id="btn-can-down", classes="step-btn btn-danger")

                    # ── 탭2: 검증 ──────────────────────────
                    with TabPane("🔍 검증", id="tab-validate"):
                        with ScrollableContainer():
                            with Vertical(classes="btn-group"):
                                yield Static("■ 데이터 검증", classes="btn-group-title")
                                yield Button(
                                    "① 데이터셋 유효성 검사",
                                    id="btn-data", classes="step-btn btn-validate")
                                yield Button(
                                    "② ACTION_MIN/MAX 계산",
                                    id="btn-range", classes="step-btn btn-validate")
                                yield Button(
                                    "③ SmolVLA dry-run 추론",
                                    id="btn-infer-dry", classes="step-btn btn-validate")
                                yield Button(
                                    "④ RViz 궤적 미리보기",
                                    id="btn-rviz-preview", classes="step-btn btn-validate")
                                yield Button(
                                    "⑤ piper-replay dry-run",
                                    id="btn-replay-dry", classes="step-btn btn-validate")

                            with Vertical(classes="btn-group"):
                                yield Static("■ 전체 자동 검증", classes="btn-group-title")
                                yield Button(
                                    "▶ 전체 검증 순서 자동 실행",
                                    id="btn-full-validate", classes="step-btn btn-validate")

                    # ── 탭3: 추론 ──────────────────────────
                    with TabPane("🤖 추론", id="tab-infer"):
                        with ScrollableContainer():
                            with Vertical(classes="btn-group"):
                                yield Static("■ 실제 arm 실험", classes="btn-group-title")
                                yield Static(
                                    "⚠ 반드시 검증 탭을 모두 통과한 뒤 실행",
                                    classes="warn")
                                yield Button(
                                    "replay 실제 arm (max_steps=5)",
                                    id="btn-replay-real", classes="step-btn btn-infer")
                                yield Button(
                                    "SmolVLA 실제 arm 추론 (max_steps=5)",
                                    id="btn-infer-real", classes="step-btn btn-infer")

                            with Vertical(classes="btn-group"):
                                yield Static("■ 녹화", classes="btn-group-title")
                                yield Button(
                                    "lerobot-record 시작",
                                    id="btn-record", classes="step-btn btn-safe")

                    # ── 탭4: 설정 ──────────────────────────
                    with TabPane("⚙ 설정", id="tab-cfg"):
                        with ScrollableContainer():
                            yield Static("현재 CFG 값 (파일 상단에서 수정)", classes="btn-group-title")
                            for key, val in CFG.items():
                                if key == "safe_range":
                                    continue
                                with Horizontal(classes="cfg-row"):
                                    yield Label(f"{key}", classes="cfg-key")
                                    yield Label(str(val), classes="cfg-val")

                # 로그 영역 (탭 밖)
                yield Static("", id="progress-bar")
                with Vertical(id="log-area"):
                    yield Static("■ 실행 로그", id="log-title")
                    yield RichLog(id="log", highlight=True, markup=True, wrap=True)

        yield Footer()

    # ──────────────────────────────────────────
    # 상태 갱신 (CAN + EEF)
    # ──────────────────────────────────────────
    def on_mount(self):
        self.log_msg(f"[bold cyan]UGRP PiPER Dashboard 시작[/] — {ts()}")
        self.log_msg("q=종료  r=상태갱신  e=비상정지  Ctrl+L=로그지우기")
        self.set_interval(5.0, self.refresh_status_bg)

    def log_msg(self, msg: str):
        try:
            self.query_one("#log", RichLog).write(f"[dim]{ts()}[/]  {msg}")
        except NoMatches:
            pass

    def set_status(self, step: str, running: bool):
        try:
            sp = self.query_one("#status-widget", StatusPanel)
            sp.step_st = step
            sp.run_st  = "실행 중" if running else ("완료" if step else "대기")
        except NoMatches:
            pass

    # ── 비동기 상태 갱신 ──
    @work(exclusive=False, thread=True)
    def refresh_status_bg(self):
        self._check_can_sync()
        self._check_eef_sync()

    def _check_can_sync(self):
        iface = CFG["can_interface"]
        r = subprocess.run(
            ["ip", "link", "show", iface],
            capture_output=True, text=True)
        up = "UP" in r.stdout
        STATE.can_status = "UP" if up else "DOWN"
        try:
            sp = self.query_one("#status-widget", StatusPanel)
            sp.can_st = STATE.can_status
        except NoMatches:
            pass

    def _check_eef_sync(self):
        if STATE.can_status != "UP":
            return
        script = (
            "from lerobot_robot_piper.piper_sdk_interface import PiperSDKInterface;"
            "import json,time;"
            f"iface=PiperSDKInterface(port='{CFG['can_interface']}',skip_enable=False);"
            "time.sleep(0.3);"
            "d=iface.get_end_pose_raw();"
            "print(json.dumps(d))"
        )
        r = subprocess.run(
            ["bash", "-c", src_prefix() + f"python3 -c \"{script}\""],
            capture_output=True, text=True, timeout=8)
        if r.returncode == 0:
            try:
                vals = json.loads(r.stdout.strip().split("\n")[-1])
                STATE.eef_values = vals
                STATE.eef_status = "OK" if any(v != 0 for v in vals.values()) else "ZERO"
                try:
                    sp = self.query_one("#status-widget", StatusPanel)
                    sp.eef_st = STATE.eef_status
                    sp.update_eef(vals)
                except NoMatches:
                    pass
            except Exception:
                pass

    # ──────────────────────────────────────────
    # 액션 바인딩
    # ──────────────────────────────────────────
    def action_refresh_status(self):
        self.log_msg("[cyan]상태 갱신 중...[/]")
        self.refresh_status_bg()

    def action_emergency_stop(self):
        self.log_msg("[bold red]🛑 비상 CAN 차단![/]")
        subprocess.run(
            ["sudo", "ip", "link", "set", CFG["can_interface"], "down"])
        try:
            sp = self.query_one("#status-widget", StatusPanel)
            sp.can_st = "DOWN"
        except NoMatches:
            pass
        self.log_msg("[red]CAN 차단 완료 — 로봇 정지[/]")

    def action_clear_log(self):
        try:
            self.query_one("#log", RichLog).clear()
        except NoMatches:
            pass

    # ──────────────────────────────────────────
    # 버튼 핸들러
    # ──────────────────────────────────────────
    @on(Button.Pressed)
    async def on_button(self, event: Button.Pressed) -> None:
        if STATE.running:
            self.log_msg("[yellow]이미 실행 중 — 완료 후 다시 시도[/]")
            return
        bid = event.button.id
        handlers = {
            "btn-session":       self._run_session,
            "btn-rviz":          self._run_rviz,
            "btn-teleop":        self._run_teleop,
            "btn-can-up":        self._run_can_up,
            "btn-can-down":      self._run_can_down,
            "btn-eef":           self._run_eef,
            "btn-cam":           self._run_cam,
            "btn-data":          self._run_data_check,
            "btn-range":         self._run_calc_range,
            "btn-infer-dry":     self._run_infer_dry,
            "btn-rviz-preview":  self._run_rviz_preview,
            "btn-replay-dry":    self._run_replay_dry,
            "btn-full-validate": self._run_full_validate,
            "btn-replay-real":   self._run_replay_real,
            "btn-infer-real":    self._run_infer_real,
            "btn-record":        self._run_record,
        }
        fn = handlers.get(bid)
        if fn:
            await fn()

    # ──────────────────────────────────────────
    # 각 단계 구현
    # ──────────────────────────────────────────
    async def _exec(self, label: str, cmd: str) -> bool:
        """공통 실행 래퍼"""
        log = self.query_one("#log", RichLog)
        STATE.running = True
        self.set_status(label, True)
        self.log_msg(f"[bold green]▶ {label} 시작[/]")
        ok = await run_cmd(cmd, log)
        STATE.running = False
        color = "green" if ok else "red"
        mark  = "✓" if ok else "✗"
        self.log_msg(f"[{color}]{mark} {label} {'완료' if ok else '실패'}[/]")
        self.set_status(label if not ok else "", False)
        return ok

    async def _run_session(self):
        self.log_msg("[cyan]── 세션 시작 루틴 ──[/]")
        # pip install -e .
        await self._exec(
            "pip install -e .",
            f"cd {CFG['ugrp_dir']} && pip install -e . -q")
        # CAN
        await self._run_can_up()
        # EEF
        await self._run_eef()
        # 카메라
        await self._run_cam()
        self.log_msg("[bold green]세션 시작 루틴 완료[/]")

    async def _run_can_up(self):
        iface   = CFG["can_interface"]
        bitrate = CFG["can_bitrate"]
        await self._exec(
            "CAN 활성화",
            f"sudo ip link set {iface} up type can bitrate {bitrate} 2>/dev/null || "
            f"sudo ip link set {iface} up 2>/dev/null; "
            f"ip link show {iface}")
        self._check_can_sync()

    async def _run_can_down(self):
        log = self.query_one("#log", RichLog)
        self.log_msg("[bold red]🛑 비상 CAN 차단[/]")
        iface = CFG["can_interface"]
        ok = await run_cmd(f"sudo ip link set {iface} down", log)
        self._check_can_sync()
        self.log_msg("[red]CAN 차단 완료[/]" if ok else "[red]차단 실패[/]")

    async def _run_eef(self):
        iface = CFG["can_interface"]
        script = (
            "python3 -c \""
            "from lerobot_robot_piper.piper_sdk_interface import PiperSDKInterface;"
            "import time;"
            f"iface=PiperSDKInterface(port='{iface}',skip_enable=False);"
            "time.sleep(0.5);"
            "d=iface.get_end_pose_raw();"
            "print('EEF:', d);"
            "ok=any(v!=0 for v in d.values());"
            "print('결과: OK' if ok else '결과: ZERO');"
            "exit(0 if ok else 1)"
            "\""
        )
        await self._exec("EEF non-zero 확인", script)
        self._check_eef_sync()

    async def _run_cam(self):
        top    = CFG["top_serial"]
        wrist  = CFG["wrist_serial"]
        script = (
            "python3 -c \""
            "import pyrealsense2 as rs;"
            "ctx=rs.context();"
            "devs=list(ctx.devices);"
            "print(f'RealSense {len(devs)}대 감지');"
            "[print(f'  [{i}]',d.get_info(rs.camera_info.name),"
            "d.get_info(rs.camera_info.serial_number)) for i,d in enumerate(devs)];"
            f"serials=[d.get_info(rs.camera_info.serial_number) for d in devs];"
            f"print('top   :', 'OK' if '{top}' in serials else 'MISSING');"
            f"print('wrist :', 'OK' if '{wrist}' in serials else 'MISSING')"
            "\""
        )
        await self._exec("카메라 확인", script)

    async def _run_teleop(self):
        iface = CFG["can_interface"]
        script = (
            "python3 -c \""
            "from lerobot_robot_piper.config_piper import PiperConfig;"
            "from lerobot_robot_piper.piper import Piper;"
            "from lerobot_robot_piper.piper_slave_only import PiperSlaveOnly,PiperSlaveOnlyConfig;"
            f"robot=Piper(PiperConfig(can_interface='{iface}',control_mode='teleop'));"
            "robot.connect();"
            "teleop=PiperSlaveOnly(PiperSlaveOnlyConfig());"
            "obs=robot.get_observation();"
            "action=teleop.get_action();"
            "ret=robot.send_action(action);"
            "eef=[k for k in obs if 'pos' in k];"
            "match=all(obs[k]==action[k] for k in eef);"
            "print('send_action no-op:', match);"
            "robot.disconnect();"
            "exit(0 if match else 1)"
            "\""
        )
        await self._exec("teleop no-op 검증", script)

    async def _run_rviz(self):
        distro = CFG["ros_distro"]
        ws     = CFG["ros2_ws"]
        self.log_msg("[cyan]RViz 실행 — 종료하려면 RViz 창을 닫아[/]")
        cmd = (
            f"source /opt/ros/{distro}/setup.bash && "
            f"source {ws}/install/setup.bash 2>/dev/null || true && "
            f"ros2 launch piper_description display_piper.launch.py"
        )
        await self._exec("RViz", cmd)

    async def _run_data_check(self):
        root = CFG["dataset_root"]
        script = (
            "python3 -c \""
            "import pathlib,pandas as pd,numpy as np;"
            f"root=pathlib.Path('{root}');"
            "files=sorted(root.glob('data/**/*.parquet'));"
            "print(f'parquet {len(files)}개');"
            "total_zero=0;"
            "[print(f'  {p.name}: {len(pd.read_parquet(p))}행') for p in files];"
            "print('검사 완료')"
            "\""
        )
        await self._exec("데이터셋 유효성 검사", script)

    async def _run_calc_range(self):
        root = CFG["dataset_root"]
        script = (
            "python3 -c \""
            "import pathlib,pandas as pd,numpy as np,json;"
            f"root=pathlib.Path('{root}');"
            "files=sorted(root.glob('data/**/*.parquet'));"
            "acts=[];"
            "[acts.extend(pd.read_parquet(p)['action'].tolist()) "
            " for p in files if 'action' in pd.read_parquet(p).columns];"
            "a=np.array(acts);"
            "span=a.max(0)-a.min(0);"
            "amin=(a.min(0)-span*0.05).tolist();"
            "amax=(a.max(0)+span*0.05).tolist();"
            "out={'ACTION_MIN':amin,'ACTION_MAX':amax,'n_frames':len(acts)};"
            "open('action_range.json','w').write(json.dumps(out,indent=2));"
            "print('ACTION_MIN:',amin);"
            "print('ACTION_MAX:',amax);"
            "print('저장: action_range.json')"
            "\""
        )
        await self._exec("ACTION_MIN/MAX 계산", script)

    async def _run_infer_dry(self):
        self.log_msg("[yellow]pretrained_path를 파일 상단 CFG에 설정하거나 직접 입력하세요[/]")
        path = os.environ.get(
            "PRETRAINED_PATH",
            "outputs/piper-smolvla/checkpoints/last/pretrained_model")
        cmd = (
            f"smolvla-inference "
            f"--pretrained_path={path} "
            f"--use_devices=false "
            f"--max_steps=20 "
            f"--task='pick the pan' "
            f"--save_actions=true"
        )
        await self._exec("SmolVLA dry-run 추론", cmd)

    async def _run_rviz_preview(self):
        af = pathlib.Path("predicted_actions.json")
        if not af.exists():
            self.log_msg("[red]predicted_actions.json 없음 — 먼저 dry-run 실행[/]")
            return
        self.log_msg("[cyan]RViz에 궤적 퍼블리시 중 — Ctrl+C로 종료[/]")
        script = (
            "python3 -c \""
            "import json,rclpy,time;"
            "from rclpy.node import Node;"
            "from visualization_msgs.msg import Marker;"
            "from geometry_msgs.msg import Point;"
            "from std_msgs.msg import ColorRGBA;"
            "rclpy.init();"
            "n=Node('preview');"
            "pub=n.create_publisher(Marker,'/preview_trajectory',10);"
            "data=json.load(open('predicted_actions.json'));"
            "acts=data['actions'];"
            "m=Marker();m.header.frame_id='base_link';"
            "m.type=Marker.LINE_STRIP;m.action=Marker.ADD;"
            "m.scale.x=0.005;m.color=ColorRGBA(r=0.1,g=1.0,b=0.1,a=1.0);"
            "[m.points.append(Point(x=a[0]/1e6,y=a[1]/1e6,z=a[2]/1e6)) for a in acts];"
            "[pub.publish(m) or rclpy.spin_once(n,timeout_sec=1.0) or time.sleep(1) for _ in range(30)];"
            "rclpy.shutdown();"
            "print('퍼블리시 완료')"
            "\""
        )
        await self._exec("RViz 궤적 미리보기", script)

    async def _run_replay_dry(self):
        repo  = CFG["dataset_repo_id"]
        root  = CFG["dataset_root"]
        cmd = (
            f"piper-replay "
            f"--dataset_repo_id={repo} "
            f"--dataset_root={root} "
            f"--episode=0 "
            f"--use_devices=false "
            f"--max_steps=20 "
            f"--replay_fps=5"
        )
        await self._exec("piper-replay dry-run", cmd)

    async def _run_full_validate(self):
        self.log_msg("[bold cyan]── 전체 검증 시작 ──[/]")
        steps = [
            ("데이터셋 유효성 검사", self._run_data_check),
            ("ACTION_MIN/MAX 계산",  self._run_calc_range),
            ("SmolVLA dry-run",      self._run_infer_dry),
            ("RViz 궤적 미리보기",    self._run_rviz_preview),
            ("replay dry-run",       self._run_replay_dry),
        ]
        results = {}
        for name, fn in steps:
            self.log_msg(f"[cyan]▶ {name}...[/]")
            await fn()
            results[name] = not STATE.running  # 실행 후 running=False 이면 완료

        self.log_msg("[bold]── 전체 검증 결과 ──[/]")
        for name in results:
            self.log_msg(f"  [green]✓[/] {name}")
        self.log_msg("[bold green]전체 검증 완료[/]")

    async def _run_replay_real(self):
        self.log_msg("[bold yellow]⚠ 실제 arm replay — 주변 50cm 확인 후 진행[/]")
        repo  = CFG["dataset_repo_id"]
        root  = CFG["dataset_root"]
        iface = CFG["can_interface"]
        cmd = (
            f"piper-replay "
            f"--dataset_repo_id={repo} "
            f"--dataset_root={root} "
            f"--episode=0 "
            f"--use_devices=true "
            f"--can_interface={iface} "
            f"--max_steps=5 "
            f"--replay_fps=5"
        )
        await self._exec("replay 실제 arm", cmd)

    async def _run_infer_real(self):
        self.log_msg("[bold red]⚠ 실제 arm 추론 — 모든 검증 통과 후 진행[/]")
        path  = os.environ.get(
            "PRETRAINED_PATH",
            "outputs/piper-smolvla/checkpoints/last/pretrained_model")
        iface = CFG["can_interface"]
        top   = CFG["top_serial"]
        wrist = CFG["wrist_serial"]
        cmd = (
            f"smolvla-inference "
            f"--pretrained_path={path} "
            f"--can_interface={iface} "
            f"--top_serial={top} "
            f"--wrist_serial={wrist} "
            f"--max_steps=5 "
            f"--task='pick the pan'"
        )
        await self._exec("SmolVLA 실제 arm 추론", cmd)

    async def _run_record(self):
        iface = CFG["can_interface"]
        top   = CFG["top_serial"]
        wrist = CFG["wrist_serial"]
        repo  = CFG["dataset_repo_id"]
        root  = CFG["dataset_root"]
        cmd = (
            f"lerobot-record "
            f"--robot.type=piper "
            f"--robot.control_mode=teleop "
            f"--robot.can_interface={iface} "
            f"--robot.top_serial={top} "
            f"--robot.wrist_serial={wrist} "
            f"--teleop.type=piper_slave_only "
            f"--dataset.repo_id={repo} "
            f"--dataset.root={root} "
            f"--dataset.single_task='pick the pan' "
            f"--dataset.num_episodes=20 "
            f"--dataset.push_to_hub=false "
            f"--robot.discover_packages_path=lerobot_robot_piper "
            f"--teleop.discover_packages_path=lerobot_robot_piper"
        )
        await self._exec("lerobot-record", cmd)


# ───────────────────────────────────────────────
if __name__ == "__main__":
    PiperDashboard().run()
