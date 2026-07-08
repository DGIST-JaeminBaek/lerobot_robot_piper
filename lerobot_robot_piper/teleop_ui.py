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

from .ui import _load_geometry, _save_geometry

from piper_sdk import C_PiperInterface_V2

logger = logging.getLogger(__name__)

JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]

# configs/recording.env — repo root의 설정 파일 (teleop_ui.py 기준 두 단계 위)
RECORDING_ENV_PATH = pathlib.Path(__file__).resolve().parent.parent / "configs" / "recording.env"


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
# Teleoperate 프리셋에도 discovery 인자를 반드시 포함해야 함.
_DISCOVERY_ARGS = (
    " --robot.discover_packages_path=lerobot_robot_piper"
    " --teleop.discover_packages_path=lerobot_robot_piper"
)

SCRIPT_PRESETS = {
    "Teleoperate": (
        "lerobot-teleoperate"
        " --robot.type=piper_follower --robot.port={follower_port}"
        " --teleop.type=piper_leader --teleop.port={leader_port}"
        + _DISCOVERY_ARGS
    ),
    # "Record"는 configs/recording.env + task/num_episodes 입력값을 조합해야 해서
    # 고정 템플릿이 아니라 PiperMonitorUI._build_record_command()에서 동적으로 생성함.
    # scripts/5__record.sh(lib/run_common.sh의 robot_camera_args/robot_action_offset_args)와
    # 동등한 41개 인자를 그대로 반영.
}

# 콤보박스에 표시할 프리셋 이름 (SCRIPT_PRESETS에 없는 "Record"도 포함)
PRESET_NAMES = ["Teleoperate", "Record"]


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

        # Preset + custom command
        ttk.Label(script_frame, text="Preset:").grid(row=2, column=0, padx=4, sticky="e")
        self.preset_var = tk.StringVar(value="Teleoperate")
        preset_combo = ttk.Combobox(
            script_frame, textvariable=self.preset_var,
            values=PRESET_NAMES, state="readonly", width=14,
        )
        preset_combo.grid(row=2, column=1, padx=4, sticky="w")
        preset_combo.bind("<<ComboboxSelected>>", self._on_preset_selected)

        btn_row2 = ttk.Frame(script_frame)
        btn_row2.grid(row=2, column=2, sticky="e")
        self.btn_launch = ttk.Button(btn_row2, text="Launch", command=self._on_launch)
        self.btn_launch.pack(side="left", padx=4)
        self.btn_kill = ttk.Button(btn_row2, text="Stop", command=self._on_kill, state="disabled")
        self.btn_kill.pack(side="left", padx=4)

        ttk.Label(script_frame, text="Command:").grid(row=3, column=0, padx=4, sticky="e")
        self.cmd_var = tk.StringVar()
        self._on_preset_selected(None)  # fill initial command
        cmd_entry = ttk.Entry(script_frame, textvariable=self.cmd_var)
        cmd_entry.grid(row=3, column=1, columnspan=2, padx=4, sticky="ew", pady=(4, 0))

        # -- Monitor Controls
        mon_ctrl = ttk.LabelFrame(self.root, text="CAN Monitor", padding=8)
        mon_ctrl.grid(row=2, column=0, sticky="ew", padx=8, pady=4)

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
        joint_frame.grid(row=3, column=0, sticky="nsew", padx=8, pady=4)
        self.root.rowconfigure(3, weight=1)
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
        status_frame.grid(row=4, column=0, sticky="ew", padx=8, pady=4)
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
        preset = self.preset_var.get()
        if preset == "Record":
            self.cmd_var.set(self._build_record_command())
        elif preset in SCRIPT_PRESETS:
            cmd = SCRIPT_PRESETS[preset].format(
                leader_port=self.leader_port_var.get(),
                follower_port=self.follower_port_var.get(),
            )
            self.cmd_var.set(cmd)

    def _build_record_command(self) -> str:
        """scripts/5__record.sh(lib/run_common.sh)와 동등한 lerobot-record 커맨드 조립.
        카메라/offset/dataset 값은 configs/recording.env를 우선 쓰고, 없으면
        5__record.sh와 동일한 fallback 기본값을 씀. Task/Num Episodes만 UI 입력값 사용."""
        env = self.recording_env

        follower_port = self.follower_port_var.get().strip()
        leader_port = self.leader_port_var.get().strip()
        task = self.task_var.get().strip()
        num_episodes = self.num_episodes_var.get().strip()

        # -- 카메라 (robot_camera_args()와 동일한 fallback)
        camera_type = env.get("CAMERA_TYPE") or "opencv"
        top_cam_type = env.get("TOP_CAM_TYPE") or camera_type
        wrist_cam_type = env.get("WRIST_CAM_TYPE") or camera_type
        top_cam = env.get("TOP_CAM") or "0"
        wrist_cam = env.get("WRIST_CAM") or "1"
        cam_width = env.get("CAM_WIDTH") or "640"
        cam_height = env.get("CAM_HEIGHT") or "480"
        fps = env.get("FPS") or "30"
        realsense_use_depth = env.get("REALSENSE_USE_DEPTH") or "false"
        realsense_warmup_s = env.get("REALSENSE_WARMUP_S") or "5.0"
        camera_connect_warmup = env.get("CAMERA_CONNECT_WARMUP") or "false"
        camera_post_connect_wait_s = env.get("CAMERA_POST_CONNECT_WAIT_S") or "2.0"
        top_realsense_use_depth = env.get("TOP_REALSENSE_USE_DEPTH") or realsense_use_depth
        wrist_realsense_use_depth = env.get("WRIST_REALSENSE_USE_DEPTH") or realsense_use_depth

        # -- action offset (robot_action_offset_args()와 동일한 fallback)
        park_on_connect = env.get("PARK_ON_CONNECT") or "false"
        use_action_offset = env.get("USE_ACTION_OFFSET") or "true"
        use_manual_action_offset = env.get("USE_MANUAL_ACTION_OFFSET") or "false"
        action_offset_report_threshold = env.get("ACTION_OFFSET_REPORT_THRESHOLD") or "3.0"
        offset_joints = [env.get(f"ACTION_OFFSET_JOINT{n}") or "0.0" for n in range(1, 7)]
        offset_gripper = env.get("ACTION_OFFSET_GRIPPER") or "0.0"

        # -- dataset (5__record.sh와 동일한 fallback)
        dataset_repo_id = env.get("DATASET_REPO_ID") or "local/piper_write_light"
        dataset_root = env.get("DATASET_ROOT") or f"records/{dataset_repo_id}"
        episode_time_s = env.get("EPISODE_TIME_S") or "60"
        reset_time_s = env.get("RESET_TIME_S") or "60"
        push_to_hub = env.get("PUSH_TO_HUB") or "false"
        display_data = env.get("DISPLAY_DATA") or "true"
        resume = env.get("RESUME") or "false"

        args = [
            "lerobot-record",
            "--robot.type=piper_follower",
            f"--robot.port={follower_port}",
            f"--robot.camera_type={camera_type}",
            f"--robot.top_cam_type={top_cam_type}",
            f"--robot.wrist_cam_type={wrist_cam_type}",
            f"--robot.top_cam={top_cam}",
            f"--robot.wrist_cam={wrist_cam}",
            f"--robot.cam_width={cam_width}",
            f"--robot.cam_height={cam_height}",
            f"--robot.camera_fps={fps}",
            f"--robot.realsense_use_depth={realsense_use_depth}",
            f"--robot.realsense_warmup_s={realsense_warmup_s}",
            f"--robot.camera_connect_warmup={camera_connect_warmup}",
            f"--robot.camera_post_connect_wait_s={camera_post_connect_wait_s}",
            f"--robot.top_realsense_use_depth={top_realsense_use_depth}",
            f"--robot.wrist_realsense_use_depth={wrist_realsense_use_depth}",
            f"--robot.park_on_connect={park_on_connect}",
            f"--robot.use_action_offset={use_action_offset}",
            f"--robot.use_manual_action_offset={use_manual_action_offset}",
            f"--robot.action_offset_report_threshold={action_offset_report_threshold}",
            f"--robot.action_offset_joint1={offset_joints[0]}",
            f"--robot.action_offset_joint2={offset_joints[1]}",
            f"--robot.action_offset_joint3={offset_joints[2]}",
            f"--robot.action_offset_joint4={offset_joints[3]}",
            f"--robot.action_offset_joint5={offset_joints[4]}",
            f"--robot.action_offset_joint6={offset_joints[5]}",
            f"--robot.action_offset_gripper={offset_gripper}",
            "--teleop.type=piper_leader",
            f"--teleop.port={leader_port}",
            f"--display_data={display_data}",
            f"--dataset.repo_id={dataset_repo_id}",
            f"--dataset.root={dataset_root}",
            f"--dataset.fps={fps}",
            f"--dataset.num_episodes={num_episodes}",
            f"--dataset.episode_time_s={episode_time_s}",
            f"--dataset.reset_time_s={reset_time_s}",
            f"--dataset.single_task={shlex.quote(task)}",
            f"--dataset.push_to_hub={push_to_hub}",
            f"--resume={resume}",
            "--robot.discover_packages_path=lerobot_robot_piper",
            "--teleop.discover_packages_path=lerobot_robot_piper",
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
        self.btn_launch.config(state="normal")
        self.btn_kill.config(state="disabled")
        self.bottom_var.set(f"Script exited (code {rc})")

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
