"""
Piper Monitor UI — subprocess launcher + read-only CAN monitoring.

Architecture:
  ┌──────────────────────────────────┐
  │ lerobot-teleoperate (subprocess) │  ← any lerobot script, untouched
  │ leader CAN ←→ follower CAN      │
  └──────────────────────────────────┘
  ┌──────────────────────────────────┐
  │ Monitor UI (this process)        │
  │ CAN read-only sniff (no control) │  ← observe only, no interference
  │ joint positions + arm status     │
  └──────────────────────────────────┘
"""

import json
import logging
import os
import pathlib
import shlex
import signal
import subprocess
import threading
import time
import tkinter as tk
from tkinter import ttk

import cv2

from .ui import _load_geometry, _save_geometry

from piper_sdk import C_PiperInterface_V2

logger = logging.getLogger(__name__)

JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]

# lerobot-record 종료 직후, 다음 녹화 시작 전에 카메라 release를 위해 기다리는 시간(초).
# SIGINT로 죽은 뒤 OS가 비디오 디바이스를 완전히 놓아주기까지 약간의 지연이 있는 경우가
# 있어(녹화마다 카메라 open 타임아웃 나는 문제의 원인으로 추정), Launch 버튼을 이 시간
# 동안 비활성화해서 바로 재시작하지 못하게 막음. 실제 하드웨어에서 적정값 조정 필요.
CAMERA_RELEASE_WAIT_S = 1.5

# repo root (teleop_ui.py 기준 두 단계 위) / configs/recording.env
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RECORDING_ENV_PATH = REPO_ROOT / "configs" / "recording.env"


def load_recording_env(path: pathlib.Path = RECORDING_ENV_PATH) -> dict[str, str]:
    """configs/recording.env를 KEY=VALUE 딕셔너리로 파싱. 파일이 없거나
    읽기 실패하면 빈 dict를 반환 (에러 없이 넘어감 — 값은 각 호출부에서
    fallback 기본값을 씀)."""
    env: dict[str, str] = {}
    try:
        text = path.read_text()
    except OSError:
        return env

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            env[key] = value
    return env


def dataset_scan_root(recording_env: dict[str, str]) -> pathlib.Path:
    """recording.env의 DATASET_ROOT 부모 폴더(보통 records/)를 스캔 기준으로 씀.
    DATASET_ROOT가 없으면 REPO_ROOT/records로 fallback."""
    dataset_root = recording_env.get("DATASET_ROOT", "")
    if dataset_root:
        p = pathlib.Path(dataset_root)
        if not p.is_absolute():
            p = REPO_ROOT / p
        return p.parent
    return REPO_ROOT / "records"


def discover_datasets(scan_root: pathlib.Path) -> list[pathlib.Path]:
    """scan_root 아래에서 meta/info.json이 있는 LeRobotDataset 루트를 전부 찾음
    (repo_id가 local/piper_xxx처럼 중첩 폴더라 재귀적으로 찾아야 함)."""
    if not scan_root.exists():
        return []
    return sorted(p.parent.parent for p in scan_root.rglob("meta/info.json"))


def read_episode_count(dataset_root: pathlib.Path) -> int:
    """meta/info.json의 total_episodes 필드를 읽음. 없거나 파싱 실패하면 0."""
    info_path = dataset_root / "meta" / "info.json"
    try:
        with open(info_path) as f:
            info = json.load(f)
        return int(info.get("total_episodes", 0))
    except (OSError, ValueError, json.JSONDecodeError):
        return 0


# ---------------------------------------------------------------- CAN helpers
def _run_cmd(cmd: list[str], sudo: bool = False) -> tuple[int, str, str]:
    if sudo:
        cmd = ["sudo"] + cmd
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def detect_can_interfaces() -> list[dict]:
    rc, out, _ = _run_cmd(["ip", "-br", "link", "show", "type", "can"])
    if rc != 0 or not out:
        return []
    interfaces = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        iface, state = parts[0], parts[1]
        rc2, out2, _ = _run_cmd(["ethtool", "-i", iface], sudo=True)
        bus_info = ""
        if rc2 == 0:
            for l in out2.splitlines():
                if l.startswith("bus-info:"):
                    bus_info = l.split(":", 1)[1].strip()
        rc3, out3, _ = _run_cmd(["ip", "-details", "link", "show", iface])
        bitrate = ""
        if rc3 == 0:
            for l in out3.splitlines():
                if "bitrate" in l:
                    for token in l.split():
                        if token.isdigit() and int(token) > 10000:
                            bitrate = token
                            break
        interfaces.append({"iface": iface, "bus_info": bus_info, "state": state, "bitrate": bitrate})
    return interfaces


def init_can_interface(iface: str, target_name: str, bitrate: int) -> tuple[bool, str]:
    msgs = []
    _run_cmd(["modprobe", "gs_usb"], sudo=True)
    _run_cmd(["ip", "link", "set", iface, "down"], sudo=True)
    rc, _, err = _run_cmd(["ip", "link", "set", iface, "type", "can", "bitrate", str(bitrate)], sudo=True)
    if rc != 0:
        return False, f"Failed to set bitrate: {err}"
    msgs.append(f"Bitrate {bitrate}")
    if iface != target_name:
        rc_chk, _, _ = _run_cmd(["ip", "link", "show", target_name])
        if rc_chk == 0:
            return False, f"'{target_name}' already exists"
        rc, _, err = _run_cmd(["ip", "link", "set", iface, "name", target_name], sudo=True)
        if rc != 0:
            return False, f"Rename failed: {err}"
        msgs.append(f"{iface} -> {target_name}")
    rc, _, err = _run_cmd(["ip", "link", "set", target_name, "up"], sudo=True)
    if rc != 0:
        return False, f"Failed to bring up: {err}"
    msgs.append("UP")
    return True, "; ".join(msgs)


# --------------------------------------------------------- CAN Read-Only Monitor
class CANMonitor:
    """Read-only CAN bus monitor. Connects to CAN and reads joint/status without sending commands."""

    def __init__(self, port: str):
        self.port = port
        self.piper = C_PiperInterface_V2(port)
        self._connected = False

    def connect(self) -> None:
        self.piper.ConnectPort()
        self._connected = True
        time.sleep(0.2)  # wait for first messages

    @property
    def is_connected(self) -> bool:
        return self._connected

    def read_joints(self) -> dict[str, float]:
        """Read joint state (for follower)."""
        msg_j = self.piper.GetArmJointMsgs()
        msg_g = self.piper.GetArmGripperMsgs()
        return {
            "joint1": float(msg_j.joint_state.joint_1),
            "joint2": float(msg_j.joint_state.joint_2),
            "joint3": float(msg_j.joint_state.joint_3),
            "joint4": float(msg_j.joint_state.joint_4),
            "joint5": float(msg_j.joint_state.joint_5),
            "joint6": float(msg_j.joint_state.joint_6),
            "gripper": float(msg_g.gripper_state.grippers_angle),
        }

    def read_control(self) -> dict[str, float]:
        """Read joint control (for leader/master mode)."""
        msg_j = self.piper.GetArmJointCtrl()
        msg_g = self.piper.GetArmGripperCtrl()
        return {
            "joint1": float(msg_j.joint_ctrl.joint_1),
            "joint2": float(msg_j.joint_ctrl.joint_2),
            "joint3": float(msg_j.joint_ctrl.joint_3),
            "joint4": float(msg_j.joint_ctrl.joint_4),
            "joint5": float(msg_j.joint_ctrl.joint_5),
            "joint6": float(msg_j.joint_ctrl.joint_6),
            "gripper": float(msg_g.gripper_ctrl.grippers_angle),
        }

    def read_status(self) -> dict:
        st = self.piper.GetArmStatus()
        en = self.piper.GetArmEnableStatus()
        return {
            "enable": en,
            "motion_status": str(st.arm_status.motion_status),
            "ctrl_mode": str(st.arm_status.ctrl_mode),
            "err_code": st.arm_status.err_code,
        }

    def disconnect(self) -> None:
        self._connected = False


# ---------------------------------------------------------------- Preset commands
# discover_packages_path 없이는 lerobot_robot_piper가 import되지 않아
# piper_follower/piper_leader가 RobotConfig/TeleoperatorConfig registry에
# 등록되지 않음 (config_piper.py / config_piper_leader.py의
# register_subclass 데코레이터는 import 시점에만 실행됨) — 그래서
# Teleoperate/Record 프리셋 모두 discovery 인자를 반드시 포함해야 함.
_DISCOVERY_ARGS = (
    " --robot.discover_packages_path=lerobot_robot_piper"
    " --teleop.discover_packages_path=lerobot_robot_piper"
)

# 프리셋 이름 -> PiperMonitorUI의 커맨드 빌더 메서드 이름.
# 콤보박스 목록도 이 딕셔너리 키에서 그대로 가져옴 — 나중에 추론(smolvla 등) 프리셋을
# 추가할 때는 PiperMonitorUI에 _build_infer_command() 같은 메서드를 만들고 여기 한 줄만
# 추가하면 됨 (_on_preset_selected/_on_launch 쪽은 손댈 필요 없음).
PRESET_BUILDERS: dict[str, str] = {
    "Teleoperate": "_build_teleoperate_command",
    "Record": "_build_record_command",
    "Infer": "_build_infer_command",
    "Replay (RViz)": "_build_replay_command",
}
PRESET_NAMES = list(PRESET_BUILDERS.keys())


# ---------------------------------------------------------------- Main UI
class PiperMonitorUI:
    def __init__(self):
        self.running = True
        self.script_proc: subprocess.Popen | None = None
        self.leader_mon: CANMonitor | None = None
        self.follower_mon: CANMonitor | None = None
        self.monitoring = False

        # configs/recording.env 값 (없거나 읽기 실패해도 빈 dict — 각 필드는 fallback 기본값 사용)
        self.recording_env: dict[str, str] = load_recording_env()

        # Shared data
        self.leader_pos: dict[str, float] = {}
        self.follower_pos: dict[str, float] = {}
        self.follower_status: dict = {}
        self.mon_hz = 0.0

        self.root = tk.Tk()
        self.root.title("Piper Monitor")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.minsize(750, 550)

        geo = _load_geometry("piper-monitor")
        if geo:
            self.root.geometry(geo)

        self._build_ui()
        self.root.update_idletasks()

        self._update_ui()

    # ------------------------------------------------------------ Build UI
    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)

        # -- CAN Setup
        self._build_can_frame()

        # -- Script Launcher
        script_frame = ttk.LabelFrame(self.root, text="Script Launcher", padding=8)
        script_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        script_frame.columnconfigure(1, weight=1)

        # Ports (configs/recording.env의 LEADER_PORT/FOLLOWER_PORT를 기본값으로 사용,
        # 없으면 기존 하드코딩 fallback 유지)
        port_row = ttk.Frame(script_frame)
        port_row.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 4))

        ttk.Label(port_row, text="Leader:").pack(side="left", padx=4)
        self.leader_port_var = tk.StringVar(value=self.recording_env.get("LEADER_PORT") or "can_leader")
        ttk.Entry(port_row, textvariable=self.leader_port_var, width=14).pack(side="left", padx=2)

        ttk.Label(port_row, text="Follower:").pack(side="left", padx=(12, 4))
        self.follower_port_var = tk.StringVar(value=self.recording_env.get("FOLLOWER_PORT") or "can_follower")
        ttk.Entry(port_row, textvariable=self.follower_port_var, width=14).pack(side="left", padx=2)

        # Task / Num Episodes — Record 프리셋의 --dataset.single_task / --dataset.num_episodes로 반영.
        # 기존 Leader/Follower Entry와 같은 레이아웃(Label + Entry, pack side=left) 재사용.
        task_row = ttk.Frame(script_frame)
        task_row.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 4))

        ttk.Label(task_row, text="Task:").pack(side="left", padx=4)
        self.task_var = tk.StringVar(value=self.recording_env.get("TASK") or "")
        ttk.Entry(task_row, textvariable=self.task_var, width=32).pack(side="left", padx=2)

        ttk.Label(task_row, text="Num Episodes:").pack(side="left", padx=(12, 4))
        self.num_episodes_var = tk.StringVar(value=self.recording_env.get("NUM_EPISODES") or "5")
        ttk.Entry(task_row, textvariable=self.num_episodes_var, width=6).pack(side="left", padx=2)

        # Policy Path — Infer 프리셋의 --policy.path로 반영 (체크포인트 로컬 경로 또는 HF repo id).
        # Teleoperate/Record에서는 안 쓰이지만 항상 보이게 둠 (프리셋 전환 시 값 유지).
        policy_row = ttk.Frame(script_frame)
        policy_row.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(0, 4))

        ttk.Label(policy_row, text="Policy Path:").pack(side="left", padx=4)
        self.policy_path_var = tk.StringVar(value=self.recording_env.get("POLICY_PRETRAINED_PATH") or "")
        ttk.Entry(policy_row, textvariable=self.policy_path_var, width=40).pack(side="left", padx=2)

        # Preset + custom command
        ttk.Label(script_frame, text="Preset:").grid(row=3, column=0, padx=4, sticky="e")
        self.preset_var = tk.StringVar(value="Teleoperate")
        preset_combo = ttk.Combobox(
            script_frame, textvariable=self.preset_var,
            values=PRESET_NAMES, state="readonly", width=14,
        )
        preset_combo.grid(row=3, column=1, padx=4, sticky="w")
        preset_combo.bind("<<ComboboxSelected>>", self._on_preset_selected)

        btn_row2 = ttk.Frame(script_frame)
        btn_row2.grid(row=3, column=2, sticky="e")
        self.btn_launch = ttk.Button(btn_row2, text="Launch", command=self._on_launch)
        self.btn_launch.pack(side="left", padx=4)
        self.btn_kill = ttk.Button(btn_row2, text="Stop", command=self._on_kill, state="disabled")
        self.btn_kill.pack(side="left", padx=4)

        ttk.Label(script_frame, text="Command:").grid(row=4, column=0, padx=4, sticky="e")
        self.cmd_var = tk.StringVar()
        self._on_preset_selected(None)  # fill initial command
        cmd_entry = ttk.Entry(script_frame, textvariable=self.cmd_var)
        cmd_entry.grid(row=4, column=1, columnspan=2, padx=4, sticky="ew", pady=(4, 0))

        # 입력값이 바뀔 때마다 Command를 자동으로 다시 조립 — Preset을 재선택 안 해도
        # 항상 최신 값 기준 커맨드가 보이게 해서, 옛날 커맨드로 Launch 누르는 실수를 막음.
        for var in (
            self.leader_port_var, self.follower_port_var,
            self.task_var, self.num_episodes_var, self.policy_path_var,
        ):
            var.trace_add("write", self._refresh_command)

        # -- Dataset Browser (records/ 밑의 기존 dataset/episode 탐색 — 향후 Replay 프리셋이 사용)
        self._build_dataset_browser_frame()

        # -- Monitor Controls
        mon_ctrl = ttk.LabelFrame(self.root, text="CAN Monitor", padding=8)
        mon_ctrl.grid(row=3, column=0, sticky="ew", padx=8, pady=4)

        self.btn_mon_start = ttk.Button(mon_ctrl, text="Start Monitor", command=self._on_mon_start)
        self.btn_mon_start.pack(side="left", padx=4)
        self.btn_mon_stop = ttk.Button(mon_ctrl, text="Stop Monitor", command=self._on_mon_stop, state="disabled")
        self.btn_mon_stop.pack(side="left", padx=4)

        self.mon_status_var = tk.StringVar(value="Monitor stopped")
        ttk.Label(mon_ctrl, textvariable=self.mon_status_var).pack(side="left", padx=12)

        self.mon_hz_var = tk.StringVar(value="")
        ttk.Label(mon_ctrl, textvariable=self.mon_hz_var, anchor="e").pack(side="right", padx=8)

        # -- Joint Monitor
        joint_frame = ttk.LabelFrame(self.root, text="Joint Positions (raw)", padding=8)
        joint_frame.grid(row=4, column=0, sticky="nsew", padx=8, pady=4)
        self.root.rowconfigure(4, weight=1)
        joint_frame.columnconfigure(2, weight=1)
        joint_frame.columnconfigure(5, weight=1)

        headers = [("Joint", 0), ("Leader", 1), ("", 2), ("Follower", 4), ("", 5)]
        for text, col in headers:
            ttk.Label(joint_frame, text=text, font=("", 9, "bold")).grid(row=0, column=col, padx=4)

        ttk.Separator(joint_frame, orient="vertical").grid(row=0, column=3, rowspan=8, sticky="ns", padx=8)

        self.leader_labels: dict[str, tk.StringVar] = {}
        self.follower_labels: dict[str, tk.StringVar] = {}
        self.leader_bars: dict[str, ttk.Progressbar] = {}
        self.follower_bars: dict[str, ttk.Progressbar] = {}

        for i, name in enumerate(JOINTS):
            r = i + 1
            ttk.Label(joint_frame, text=name, width=8, anchor="e").grid(row=r, column=0, padx=4, pady=1)

            lv = tk.StringVar(value="--")
            ttk.Label(joint_frame, textvariable=lv, width=10, anchor="e").grid(row=r, column=1, padx=4)
            self.leader_labels[name] = lv

            lb = ttk.Progressbar(joint_frame, length=100, maximum=200, value=100, mode="determinate")
            lb.grid(row=r, column=2, padx=4, sticky="ew")
            self.leader_bars[name] = lb

            fv = tk.StringVar(value="--")
            ttk.Label(joint_frame, textvariable=fv, width=10, anchor="e").grid(row=r, column=4, padx=4)
            self.follower_labels[name] = fv

            fb = ttk.Progressbar(joint_frame, length=100, maximum=200, value=100, mode="determinate")
            fb.grid(row=r, column=5, padx=4, sticky="ew")
            self.follower_bars[name] = fb

        # -- Arm Status
        status_frame = ttk.LabelFrame(self.root, text="Follower Arm Status", padding=8)
        status_frame.grid(row=5, column=0, sticky="ew", padx=8, pady=4)
        status_frame.columnconfigure(0, weight=1)

        self.status_text_var = tk.StringVar(value="--")
        ttk.Label(status_frame, textvariable=self.status_text_var, anchor="w", width=90).grid(
            row=0, column=0, sticky="ew"
        )

        # -- Bottom status
        self.bottom_var = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.bottom_var, relief="sunken", anchor="w", padding=4).grid(
            row=99, column=0, sticky="ew", padx=8, pady=(4, 8)
        )

    # ---------------------------------------------------------- CAN Setup
    def _build_can_frame(self):
        can_frame = ttk.LabelFrame(self.root, text="CAN Setup", padding=8)
        can_frame.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))

        btn_row = ttk.Frame(can_frame)
        btn_row.pack(fill="x")
        ttk.Button(btn_row, text="Detect", command=self._on_can_detect).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Init All", command=self._on_can_init_all).pack(side="left", padx=4)
        self.can_status_var = tk.StringVar(value="Click 'Detect' to scan")
        ttk.Label(btn_row, textvariable=self.can_status_var).pack(side="left", padx=12)

        self.can_rows_frame = ttk.Frame(can_frame)
        self.can_rows_frame.pack(fill="x", pady=(4, 0))
        self.can_row_widgets: list[dict] = []

    # ---------------------------------------------------------- Dataset Browser
    def _build_dataset_browser_frame(self):
        """records/ 밑의 기존 dataset/episode를 탐색해서 Dataset/Episode 콤보박스로
        선택하게 함. 여기서 고른 값(self.replay_dataset_root_var/replay_episode_var)은
        추후 Replay 프리셋이 그대로 씀."""
        frame = ttk.LabelFrame(self.root, text="Dataset Browser", padding=8)
        frame.grid(row=2, column=0, sticky="ew", padx=8, pady=4)

        ttk.Button(frame, text="Refresh", command=self._on_dataset_browser_refresh).pack(side="left", padx=4)

        ttk.Label(frame, text="Dataset:").pack(side="left", padx=(12, 4))
        self.dataset_label_var = tk.StringVar(value="")
        self.dataset_combo = ttk.Combobox(
            frame, textvariable=self.dataset_label_var, state="readonly", width=36
        )
        self.dataset_combo.pack(side="left", padx=2)
        self.dataset_combo.bind("<<ComboboxSelected>>", self._on_dataset_selected)

        ttk.Label(frame, text="Episode:").pack(side="left", padx=(12, 4))
        self.replay_episode_var = tk.StringVar(value="")
        self.episode_combo = ttk.Combobox(
            frame, textvariable=self.replay_episode_var, state="readonly", width=6
        )
        self.episode_combo.pack(side="left", padx=2)

        self.dataset_status_var = tk.StringVar(value="Click 'Refresh' to scan")
        ttk.Label(frame, textvariable=self.dataset_status_var).pack(side="left", padx=12)

        # label(콤보박스 표시용 상대경로) -> 실제 dataset root 절대경로
        self._dataset_paths: dict[str, pathlib.Path] = {}
        # 선택된 dataset의 절대경로 문자열 — Replay 프리셋이 --dataset_root로 씀
        self.replay_dataset_root_var = tk.StringVar(value="")

        # Dataset/Episode 선택이 바뀌면 Replay 커맨드도 자동 새로고침
        self.replay_dataset_root_var.trace_add("write", self._refresh_command)
        self.replay_episode_var.trace_add("write", self._refresh_command)

        self._on_dataset_browser_refresh()

    def _on_dataset_browser_refresh(self):
        scan_root = dataset_scan_root(self.recording_env)
        datasets = discover_datasets(scan_root)

        self._dataset_paths = {}
        labels = []
        for d in datasets:
            try:
                label = str(d.relative_to(scan_root))
            except ValueError:
                label = str(d)
            self._dataset_paths[label] = d
            labels.append(label)

        self.dataset_combo["values"] = labels
        if labels:
            self.dataset_status_var.set(f"{len(labels)}개 dataset 발견 ({scan_root})")
            if self.dataset_label_var.get() not in labels:
                self.dataset_label_var.set(labels[0])
            self._on_dataset_selected(None)
        else:
            self.dataset_status_var.set(f"dataset 없음 ({scan_root})")
            self.dataset_label_var.set("")
            self.replay_dataset_root_var.set("")
            self.episode_combo["values"] = []
            self.replay_episode_var.set("")

    def _on_dataset_selected(self, _event):
        label = self.dataset_label_var.get()
        dataset_root = self._dataset_paths.get(label)
        if dataset_root is None:
            self.replay_dataset_root_var.set("")
            self.episode_combo["values"] = []
            self.replay_episode_var.set("")
            return

        self.replay_dataset_root_var.set(str(dataset_root))
        n = read_episode_count(dataset_root)
        episodes = [str(i) for i in range(n)]
        self.episode_combo["values"] = episodes
        self.replay_episode_var.set(episodes[0] if episodes else "")

    def _on_can_detect(self):
        for w in self.can_rows_frame.winfo_children():
            w.destroy()
        self.can_row_widgets.clear()
        interfaces = detect_can_interfaces()
        if not interfaces:
            self.can_status_var.set("No CAN interfaces detected")
            return
        self.can_status_var.set(f"{len(interfaces)} interface(s) found")
        for i, info in enumerate(interfaces):
            row = {"info": info}
            f = ttk.Frame(self.can_rows_frame)
            f.pack(fill="x", pady=1)
            ttk.Label(f, text=info["iface"], width=12).pack(side="left", padx=2)
            ttk.Label(f, text=info["bus_info"], width=16).pack(side="left", padx=2)
            state_color = "green" if info["state"] == "UP" else "gray"
            tk.Label(f, text=info["state"], fg=state_color, width=6).pack(side="left", padx=2)
            default_name = f"can_{'leader' if i == 0 else 'follower'}" if len(interfaces) > 1 else "can_follower"
            nv = tk.StringVar(value=default_name)
            ttk.Entry(f, textvariable=nv, width=14).pack(side="left", padx=2)
            row["target_name"] = nv
            bv = tk.StringVar(value="1000000")
            ttk.Entry(f, textvariable=bv, width=10).pack(side="left", padx=2)
            row["target_bitrate"] = bv
            ttk.Button(f, text="Init", command=lambda idx=i: self._on_can_init_single(idx)).pack(side="left", padx=2)
            self.can_row_widgets.append(row)

    def _on_can_init_single(self, idx: int):
        row = self.can_row_widgets[idx]
        iface = row["info"]["iface"]
        target = row["target_name"].get().strip()
        try:
            bitrate = int(row["target_bitrate"].get().strip())
        except ValueError:
            self.can_status_var.set(f"Invalid bitrate for {iface}")
            return
        self.can_status_var.set(f"Initializing {iface}...")
        self.root.update()
        ok, msg = init_can_interface(iface, target, bitrate)
        self.can_status_var.set(f"{'OK' if ok else 'FAIL'}: {msg}")
        if ok:
            self._on_can_detect()

    def _on_can_init_all(self):
        if not self.can_row_widgets:
            self.can_status_var.set("Detect first")
            return
        results = []
        for row in self.can_row_widgets:
            iface = row["info"]["iface"]
            target = row["target_name"].get().strip()
            try:
                bitrate = int(row["target_bitrate"].get().strip())
            except ValueError:
                results.append(f"{iface}: bad bitrate")
                continue
            ok, _ = init_can_interface(iface, target, bitrate)
            results.append(f"{target}: {'OK' if ok else 'FAIL'}")
        self.can_status_var.set(" | ".join(results))
        self._on_can_detect()

    # ---------------------------------------------------------- Script Launcher
    def _on_preset_selected(self, _event):
        self._refresh_command()

    def _refresh_command(self, *_trace_args):
        """현재 선택된 Preset 기준으로 Command 칸을 다시 조립.
        Preset 콤보박스 선택뿐 아니라 Leader/Follower/Task/Num Episodes/
        Policy Path/Dataset Browser 선택이 바뀔 때도 이 메서드가 trace로
        호출돼서, 옛날 값으로 조립된 커맨드로 실수로 Launch 누르는 걸 막음."""
        if not hasattr(self, "cmd_var"):
            return  # 위젯 초기 구성 중 trace가 너무 일찍 불린 경우 (cmd_var 생성 전)
        builder_name = PRESET_BUILDERS.get(self.preset_var.get())
        if builder_name:
            self.cmd_var.set(getattr(self, builder_name)())

    def _build_teleoperate_command(self) -> str:
        leader_port = self.leader_port_var.get().strip()
        follower_port = self.follower_port_var.get().strip()
        return (
            "lerobot-teleoperate"
            f" --robot.type=piper_follower --robot.port={follower_port}"
            f" --teleop.type=piper_leader --teleop.port={leader_port}"
            + _DISCOVERY_ARGS
        )

    # -- lerobot-record 커맨드 조립 공용 헬퍼 (Record/Infer가 공유) -----------
    def _camera_args(self) -> list[str]:
        """robot_camera_args()(run_common.sh)와 동일한 fallback. depth 설정
        (REALSENSE_USE_DEPTH 등)도 여기서 그대로 반영돼서 Record/Infer 둘 다 씀."""
        env = self.recording_env
        camera_type = env.get("CAMERA_TYPE") or "opencv"
        top_cam_type = env.get("TOP_CAM_TYPE") or camera_type
        wrist_cam_type = env.get("WRIST_CAM_TYPE") or camera_type
        realsense_use_depth = env.get("REALSENSE_USE_DEPTH") or "false"
        return [
            f"--robot.camera_type={camera_type}",
            f"--robot.top_cam_type={top_cam_type}",
            f"--robot.wrist_cam_type={wrist_cam_type}",
            f"--robot.top_cam={env.get('TOP_CAM') or '0'}",
            f"--robot.wrist_cam={env.get('WRIST_CAM') or '1'}",
            f"--robot.cam_width={env.get('CAM_WIDTH') or '640'}",
            f"--robot.cam_height={env.get('CAM_HEIGHT') or '480'}",
            f"--robot.camera_fps={env.get('FPS') or '30'}",
            f"--robot.realsense_use_depth={realsense_use_depth}",
            f"--robot.realsense_warmup_s={env.get('REALSENSE_WARMUP_S') or '5.0'}",
            f"--robot.camera_connect_warmup={env.get('CAMERA_CONNECT_WARMUP') or 'false'}",
            f"--robot.camera_post_connect_wait_s={env.get('CAMERA_POST_CONNECT_WAIT_S') or '2.0'}",
            f"--robot.top_realsense_use_depth={env.get('TOP_REALSENSE_USE_DEPTH') or realsense_use_depth}",
            f"--robot.wrist_realsense_use_depth={env.get('WRIST_REALSENSE_USE_DEPTH') or realsense_use_depth}",
        ]

    def _action_offset_args(self) -> list[str]:
        """robot_action_offset_args()(run_common.sh)와 동일한 fallback."""
        env = self.recording_env
        offset_joints = [env.get(f"ACTION_OFFSET_JOINT{n}") or "0.0" for n in range(1, 7)]
        return [
            f"--robot.park_on_connect={env.get('PARK_ON_CONNECT') or 'false'}",
            f"--robot.use_action_offset={env.get('USE_ACTION_OFFSET') or 'true'}",
            f"--robot.use_manual_action_offset={env.get('USE_MANUAL_ACTION_OFFSET') or 'false'}",
            f"--robot.action_offset_report_threshold={env.get('ACTION_OFFSET_REPORT_THRESHOLD') or '3.0'}",
            f"--robot.action_offset_joint1={offset_joints[0]}",
            f"--robot.action_offset_joint2={offset_joints[1]}",
            f"--robot.action_offset_joint3={offset_joints[2]}",
            f"--robot.action_offset_joint4={offset_joints[3]}",
            f"--robot.action_offset_joint5={offset_joints[4]}",
            f"--robot.action_offset_joint6={offset_joints[5]}",
            f"--robot.action_offset_gripper={env.get('ACTION_OFFSET_GRIPPER') or '0.0'}",
        ]

    def _dataset_args(self, fps: str) -> list[str]:
        """5__record.sh의 dataset.* 인자와 동일한 fallback. Task/Num Episodes만 UI 입력값 사용."""
        env = self.recording_env
        task = self.task_var.get().strip()
        num_episodes = self.num_episodes_var.get().strip()
        dataset_repo_id = env.get("DATASET_REPO_ID") or "local/piper_write_light"
        dataset_root = env.get("DATASET_ROOT") or f"records/{dataset_repo_id}"
        return [
            f"--dataset.repo_id={dataset_repo_id}",
            f"--dataset.root={dataset_root}",
            f"--dataset.fps={fps}",
            f"--dataset.num_episodes={num_episodes}",
            f"--dataset.episode_time_s={env.get('EPISODE_TIME_S') or '60'}",
            f"--dataset.reset_time_s={env.get('RESET_TIME_S') or '60'}",
            f"--dataset.single_task={shlex.quote(task)}",
            f"--dataset.push_to_hub={env.get('PUSH_TO_HUB') or 'false'}",
            f"--resume={env.get('RESUME') or 'false'}",
        ]

    def _build_record_command(self) -> str:
        """scripts/5__record.sh(lib/run_common.sh)와 동등한 lerobot-record 커맨드 조립."""
        env = self.recording_env
        follower_port = self.follower_port_var.get().strip()
        leader_port = self.leader_port_var.get().strip()
        fps = env.get("FPS") or "30"

        args = [
            "lerobot-record",
            "--robot.type=piper_follower",
            f"--robot.port={follower_port}",
            *self._camera_args(),
            *self._action_offset_args(),
            "--teleop.type=piper_leader",
            f"--teleop.port={leader_port}",
            f"--display_data={env.get('DISPLAY_DATA') or 'true'}",
            *self._dataset_args(fps),
            "--robot.discover_packages_path=lerobot_robot_piper",
            "--teleop.discover_packages_path=lerobot_robot_piper",
        ]
        return " ".join(args)

    def _build_infer_command(self) -> str:
        """lerobot-record --policy.path=... 로 정책(SmolVLA 등) 추론 실행.
        구 UGRP의 별도 smolvla-inference CLI는 새 레포에 없음 — lerobot 자체가
        lerobot-record에 --policy.path를 지원해서 policy가 action을 생성하고
        teleop은 episode 사이 리셋용으로 병행할 수 있음
        (lerobot/scripts/lerobot_record.py 상단 docstring, RecordConfig 참고).
        카메라 인자를 Record와 공유하므로 depth 설정(REALSENSE_USE_DEPTH 등)도
        recording.env에 넣어두면 그대로 반영됨.
        주의: 새 lerobot-record CLI에는 구 UGRP infer_dry 같은
        --use_devices=false dry-run 옵션이 없음 — Launch 누르면 바로 실제
        로봇에 정책 action이 전송됨."""
        env = self.recording_env
        follower_port = self.follower_port_var.get().strip()
        leader_port = self.leader_port_var.get().strip()
        policy_path = self.policy_path_var.get().strip()
        fps = env.get("FPS") or "30"

        args = [
            "lerobot-record",
            "--robot.type=piper_follower",
            f"--robot.port={follower_port}",
            *self._camera_args(),
            *self._action_offset_args(),
            "--teleop.type=piper_leader",
            f"--teleop.port={leader_port}",
            f"--policy.path={policy_path}",
            f"--display_data={env.get('DISPLAY_DATA') or 'true'}",
            *self._dataset_args(fps),
            "--robot.discover_packages_path=lerobot_robot_piper",
            "--teleop.discover_packages_path=lerobot_robot_piper",
        ]
        return " ".join(args)

    def _build_replay_command(self) -> str:
        """Dataset Browser에서 고른 dataset/episode를 scripts/legacy_tools/
        piper_replay_viz.py(joint_states 기반 RViz 재생, --robot 하드웨어 연결
        없이 동작)로 넘김. 실행 전에 별도 터미널에서 RViz + robot_state_publisher
        (agx_arm_urdf의 display 계열 launch)가 떠 있어야 함 — 스크립트 자체는
        launch를 대신 띄워주지 않음. ROS2 환경(source /opt/ros/humble/setup.bash)도
        이 UI를 실행한 셸에 이미 sourced 되어 있어야 함."""
        script_path = REPO_ROOT / "scripts" / "legacy_tools" / "piper_replay_viz.py"
        dataset_root = self.replay_dataset_root_var.get().strip()
        episode = self.replay_episode_var.get().strip()

        args = [
            "python3", str(script_path),
            f"--dataset_root={dataset_root}",
            f"--episode={episode}",
        ]
        return " ".join(args)

    def _on_launch(self):
        cmd = self.cmd_var.get().strip()
        if not cmd:
            self.bottom_var.set("No command to run")
            return

        # Substitute current port values
        cmd = cmd.replace("{leader_port}", self.leader_port_var.get())
        cmd = cmd.replace("{follower_port}", self.follower_port_var.get())

        self.bottom_var.set(f"Launching: {cmd}")
        self.root.update()

        try:
            self.script_proc = subprocess.Popen(
                cmd, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )
        except Exception as e:
            self.bottom_var.set(f"Launch failed: {e}")
            return

        self.btn_launch.config(state="disabled")
        self.btn_kill.config(state="normal")
        self.bottom_var.set(f"Running (PID {self.script_proc.pid}): {cmd}")

        # Monitor process in background
        threading.Thread(target=self._watch_proc, daemon=True).start()

    def _watch_proc(self):
        """Wait for subprocess to finish and update UI."""
        if self.script_proc:
            self.script_proc.wait()
            rc = self.script_proc.returncode
            self.script_proc = None
            self.root.after(0, self._proc_finished, rc)

    def _proc_finished(self, rc: int):
        # 프로세스는 완전히 죽었지만, 카메라 디바이스가 아직 release 안 됐을 수 있음
        # (SIGINT로 죽을 때 lerobot-record 쪽 cv2.VideoCapture release가 항상
        # 깨끗하게 실행된다는 보장이 없음) — Launch를 바로 켜지 않고, 카메라
        # release 사이클을 먼저 돌린 뒤 켬. Stop 버튼은 이미 끌 게 없으니 비활성화.
        self.btn_launch.config(state="disabled")
        self.btn_kill.config(state="disabled")
        self.bottom_var.set(f"Script exited (code {rc}) — releasing cameras...")
        threading.Thread(target=self._release_cameras_then_ready, args=(rc,), daemon=True).start()

    def _reset_opencv_cameras(self) -> None:
        """녹화 프로세스 종료 직후 OpenCV 카메라 index를 열었다 바로 닫아서
        OS 레벨 release를 유도. RealSense(serial 지정 카메라)는 index 기반이
        아니라 이 방식이 안 맞아서 CAMERA_TYPE=opencv일 때만 수행함 — RealSense는
        pyrealsense2의 hardware_reset()이 필요한데, 이 컴퓨터엔 하드웨어가 없어
        검증 못 해서 이번 변경에는 포함하지 않음."""
        camera_type = (self.recording_env.get("CAMERA_TYPE") or "opencv").lower()
        if camera_type != "opencv":
            return

        indices: list[int] = []
        for key in ("TOP_CAM", "WRIST_CAM"):
            raw = self.recording_env.get(key, "")
            if raw.isdecimal():
                indices.append(int(raw))

        for idx in indices:
            try:
                cap = cv2.VideoCapture(idx)
                time.sleep(0.1)
                cap.release()
            except Exception:
                logger.exception(f"Camera {idx} release probe failed")

    def _release_cameras_then_ready(self, rc: int) -> None:
        self._reset_opencv_cameras()
        time.sleep(CAMERA_RELEASE_WAIT_S)
        self.root.after(0, self._on_proc_fully_finished, rc)

    def _on_proc_fully_finished(self, rc: int) -> None:
        self.btn_launch.config(state="normal")
        self.bottom_var.set(f"Script exited (code {rc}) — cameras reset, ready")

    def _on_kill(self):
        if self.script_proc:
            try:
                os.killpg(os.getpgid(self.script_proc.pid), signal.SIGINT)
            except ProcessLookupError:
                pass
            self.bottom_var.set("Sending SIGINT to script...")

    # ---------------------------------------------------------- CAN Monitor
    def _on_mon_start(self):
        leader_port = self.leader_port_var.get().strip()
        follower_port = self.follower_port_var.get().strip()

        self.mon_status_var.set("Connecting monitors...")
        self.root.update()

        try:
            self.follower_mon = CANMonitor(follower_port)
            self.follower_mon.connect()
        except Exception as e:
            self.mon_status_var.set(f"Follower monitor failed: {e}")
            self.follower_mon = None
            return

        try:
            self.leader_mon = CANMonitor(leader_port)
            self.leader_mon.connect()
        except Exception as e:
            self.mon_status_var.set(f"Leader monitor failed: {e}")
            self.leader_mon = None
            # follower-only is still useful

        self.monitoring = True
        self.btn_mon_start.config(state="disabled")
        self.btn_mon_stop.config(state="normal")
        self.mon_status_var.set("Monitoring active")

        self._mon_thread = threading.Thread(target=self._mon_loop, daemon=True)
        self._mon_thread.start()

    def _mon_loop(self):
        while self.monitoring and self.running:
            t0 = time.perf_counter()
            try:
                if self.follower_mon and self.follower_mon.is_connected:
                    self.follower_pos = self.follower_mon.read_joints()
                    self.follower_status = self.follower_mon.read_status()

                if self.leader_mon and self.leader_mon.is_connected:
                    self.leader_pos = self.leader_mon.read_control()

            except Exception:
                logger.exception("Monitor read error")

            dt = time.perf_counter() - t0
            time.sleep(max(0.05 - dt, 0.0))  # ~20Hz
            total = time.perf_counter() - t0
            self.mon_hz = 1.0 / total if total > 0 else 0

    def _on_mon_stop(self):
        self.monitoring = False
        time.sleep(0.1)

        if self.leader_mon:
            self.leader_mon.disconnect()
            self.leader_mon = None
        if self.follower_mon:
            self.follower_mon.disconnect()
            self.follower_mon = None

        self.leader_pos.clear()
        self.follower_pos.clear()
        self.follower_status.clear()

        self.btn_mon_start.config(state="normal")
        self.btn_mon_stop.config(state="disabled")
        self.mon_status_var.set("Monitor stopped")
        self.mon_hz_var.set("")

    # ---------------------------------------------------------- UI Update
    def _update_ui(self):
        if not self.running:
            return

        for name in JOINTS:
            if name in self.leader_pos:
                val = self.leader_pos[name]
                self.leader_labels[name].set(f"{val:.0f}")
                self.leader_bars[name]["value"] = max(0, min(200, val / 1000 + 100))
            else:
                self.leader_labels[name].set("--")
                self.leader_bars[name]["value"] = 100

            if name in self.follower_pos:
                val = self.follower_pos[name]
                self.follower_labels[name].set(f"{val:.0f}")
                self.follower_bars[name]["value"] = max(0, min(200, val / 1000 + 100))
            else:
                self.follower_labels[name].set("--")
                self.follower_bars[name]["value"] = 100

        if self.follower_status:
            s = self.follower_status
            self.status_text_var.set(
                f"Enable: {s.get('enable', '--')}  |  "
                f"Motion: {s.get('motion_status', '--')}  |  "
                f"Mode: {s.get('ctrl_mode', '--')}  |  "
                f"Error: {s.get('err_code', '--')}"
            )

        if self.monitoring and self.mon_hz > 0:
            self.mon_hz_var.set(f"{self.mon_hz:.0f} Hz")

        # Check if script is still running
        if self.script_proc and self.script_proc.poll() is not None:
            self._proc_finished(self.script_proc.returncode)
            self.script_proc = None

        self.root.after(50, self._update_ui)

    # ---------------------------------------------------------- Close
    def _on_close(self):
        _save_geometry("piper-monitor", self.root.geometry())
        self.running = False
        if self.monitoring:
            self._on_mon_stop()
        if self.script_proc:
            self._on_kill()
            time.sleep(0.5)
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Piper Monitor UI")
    parser.add_argument("--leader-port", default="can_leader")
    parser.add_argument("--follower-port", default="can_follower")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    app = PiperMonitorUI()
    app.leader_port_var.set(args.leader_port)
    app.follower_port_var.set(args.follower_port)
    app.run()


if __name__ == "__main__":
    main()
