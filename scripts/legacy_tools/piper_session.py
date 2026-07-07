#!/usr/bin/env python3
"""
piper_session.py — UGRP PiPER 실험실 세션 자동화 CLI

사용법:
    python piper_session.py --step <단계> [옵션]

단계 목록:
    env_setup       : 가상환경 활성화 + CAN 활성화 + EEF non-zero 확인
    rviz            : agx_arm_urdf 세팅 후 RViz 실행
    can_up          : CAN 인터페이스만 올리기
    can_down        : 비상 CAN 차단
    eef_check       : EEF non-zero 확인
    teleop_check    : send_action no-op 검증
    cam_check       : RealSense 시리얼 및 OpenCV 인덱스 확인
    data_check      : 데이터셋 parquet 유효성 검사
    calc_range      : ACTION_MIN/MAX 자동 계산
    infer_dry       : SmolVLA 추론 (로봇 없이, actions 저장)
    rviz_preview    : 추론 궤적 RViz 시각화 + 사전 안전검사
    replay_dry      : piper-replay dry-run
    replay_real     : piper-replay 실제 arm (확인 프롬프트)
    infer_real      : SmolVLA 실제 arm 추론 (확인 프롬프트)
    session         : env_setup → cam_check → teleop_check 세션 시작 루틴
    full_validate   : data_check → calc_range → infer_dry → rviz_preview → replay_dry

예시:
    # 실험실 켰을 때 가장 먼저
    python piper_session.py --step session

    # RViz 띄우기
    python piper_session.py --step rviz

    # 추론 검증 전체 자동
    python piper_session.py --step full_validate \\
        --dataset_root /home/ugrp308/Group43/datasets/piper-smolvla \\
        --pretrained_path outputs/piper-smolvla/checkpoints/last/pretrained_model
"""

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

# ───────────────────────────────────────────────
# 설정값 — 실험 환경에 맞게 수정
# ───────────────────────────────────────────────
CFG = {
    # 가상환경
    "venv_activate":    "/home/ugrp308/Group43/.venv/bin/activate",
    "lerobot_dir":      "/home/ugrp308/Group43/lerobot",
    "ugrp_dir":         "/home/ugrp308/Group43/UGRP",
    # CAN
    "can_interface":    "can0",
    "can_bitrate":      1000000,
    # 카메라
    "top_serial":       "327122074262",
    "wrist_serial":     "243322071626",
    # URDF
    "urdf_repo":        "https://github.com/agilexrobotics/agx_arm_urdf.git",
    "urdf_local_dir":   "/home/ugrp308/Group43/agx_arm_urdf",
    "ros2_ws":          "/home/ugrp308/ros2_ws",
    "ros_distro":       "humble",
    # 데이터셋
    "dataset_root":     "/home/ugrp308/Group43/datasets/piper-smolvla",
    "dataset_repo_id":  "local/piper-smolvla",
    # EEF 안전 범위 (raw SDK 정수)
    "safe_range": {
        "x":        (100_000,  400_000),
        "y":       (-200_000,  200_000),
        "z":         (50_000,  350_000),
        "rx":      (-180_000,  180_000),
        "ry":      (-180_000,  180_000),
        "rz":      (-180_000,  180_000),
        "gripper":       (0,    70_000),
    },
    "max_delta_per_step": 30_000,
}

# ───────────────────────────────────────────────
# 컬러 출력
# ───────────────────────────────────────────────
class C:
    RESET  = "\033[0m";  BOLD   = "\033[1m"
    RED    = "\033[91m"; GREEN  = "\033[92m"
    YELLOW = "\033[93m"; CYAN   = "\033[96m"
    BLUE   = "\033[94m"

def ok(msg):     print(f"{C.GREEN}[OK]{C.RESET}    {msg}")
def warn(msg):   print(f"{C.YELLOW}[WARN]{C.RESET}  {msg}")
def err(msg):    print(f"{C.RED}[FAIL]{C.RESET}  {msg}")
def info(msg):   print(f"{C.CYAN}[INFO]{C.RESET}  {msg}")
def step_hdr(msg):
    print(f"\n{C.BOLD}{C.BLUE}{'━'*60}{C.RESET}")
    print(f"{C.BOLD}{C.BLUE}  {msg}{C.RESET}")
    print(f"{C.BOLD}{C.BLUE}{'━'*60}{C.RESET}\n")

def confirm(prompt):
    ans = input(f"{C.YELLOW}[확인]{C.RESET} {prompt} (y/n): ").strip().lower()
    return ans == "y"

def run(cmd, check=True, shell=False, capture=False):
    """subprocess 실행. 실패 시 False 반환."""
    kw = dict(shell=shell, capture_output=capture, text=capture)
    try:
        r = subprocess.run(cmd, **kw)
        return r if capture else (r.returncode == 0)
    except FileNotFoundError:
        err(f"명령 없음: {cmd[0] if isinstance(cmd, list) else cmd}")
        return False

def run_sudo(cmd_str):
    """sudo 명령어 실행 (shell=True)"""
    return run(cmd_str, shell=True)


# ═══════════════════════════════════════════════
# 공통 유틸
# ═══════════════════════════════════════════════
def source_prefix():
    """bash에서 venv + ROS2 source하는 prefix 문자열"""
    venv   = CFG["venv_activate"]
    distro = CFG["ros_distro"]
    ws     = CFG["ros2_ws"]
    return (
        f"source {venv} && "
        f"source /opt/ros/{distro}/setup.bash && "
        f"source {ws}/install/setup.bash 2>/dev/null || true && "
    )

def bash(cmd_str, check=True):
    """venv + ROS2 환경에서 bash 명령 실행"""
    full = source_prefix() + cmd_str
    r = subprocess.run(["bash", "-c", full])
    return r.returncode == 0


# ═══════════════════════════════════════════════
# STEP: can_up — CAN 인터페이스 올리기
# ═══════════════════════════════════════════════
def step_can_up(args):
    step_hdr("CAN 인터페이스 활성화")
    iface   = args.can_interface or CFG["can_interface"]
    bitrate = CFG["can_bitrate"]

    # 이미 UP인지 확인
    r = run(["ip", "link", "show", iface], capture=True)
    if r and "UP" in r.stdout:
        ok(f"{iface} 이미 활성화됨")
        return True

    info(f"sudo ip link set {iface} up type can bitrate {bitrate}")
    ok1 = run_sudo(f"sudo ip link set {iface} up type can bitrate {bitrate}")
    if not ok1:
        # bitrate 없이 up만 시도 (이미 설정된 경우)
        run_sudo(f"sudo ip link set {iface} up")

    time.sleep(0.5)
    r2 = run(["ip", "link", "show", iface], capture=True)
    if r2 and "UP" in r2.stdout:
        ok(f"{iface} 활성화 완료")
        return True
    else:
        err(f"{iface} 활성화 실패 — USB-to-CAN 어댑터 연결 확인")
        return False


# ═══════════════════════════════════════════════
# STEP: can_down — 비상 CAN 차단
# ═══════════════════════════════════════════════
def step_can_down(args):
    step_hdr("비상 CAN 차단")
    iface = args.can_interface or CFG["can_interface"]
    warn(f"sudo ip link set {iface} down 실행")
    ok_ = run_sudo(f"sudo ip link set {iface} down")
    if ok_:
        ok(f"{iface} 차단 완료 — 로봇 정지됨")
    else:
        err("CAN 차단 실패 — 전원 차단 권고")
    return ok_


# ═══════════════════════════════════════════════
# STEP: eef_check — EEF non-zero 확인
# ═══════════════════════════════════════════════
def step_eef_check(args):
    step_hdr("EEF non-zero 확인")
    iface = args.can_interface or CFG["can_interface"]

    script = f"""
import sys, time
from lerobot_robot_piper.piper_sdk_interface import PiperSDKInterface
import logging; logging.basicConfig(level=logging.WARNING)

iface = PiperSDKInterface(port='{iface}', skip_enable=False)
time.sleep(0.5)
data = iface.get_end_pose_raw()
print("EEF:", data)
if any(v != 0 for v in data.values()):
    print("RESULT:OK")
    sys.exit(0)
else:
    print("RESULT:ZERO")
    sys.exit(1)
"""
    info("EEF 상태 읽는 중...")
    r = run(["bash", "-c", source_prefix() + f'python3 -c "{script}"'],
            capture=True)
    # capture_output이 있으면 r은 CompletedProcess
    if hasattr(r, 'returncode'):
        output = (r.stdout or "") + (r.stderr or "")
        if "RESULT:OK" in output:
            ok("EEF non-zero 확인")
            return True
        elif "RESULT:ZERO" in output:
            err("EEF all-zero — CAN 수신 문제. 로봇 전원/연결 확인")
            return False

    # subprocess 직접 실행 fallback
    result = subprocess.run(
        ["bash", "-c",
         source_prefix() +
         "python3 -c \"\n"
         "import sys, time\n"
         "from lerobot_robot_piper.piper_sdk_interface import PiperSDKInterface\n"
         f"iface = PiperSDKInterface(port='{iface}', skip_enable=False)\n"
         "time.sleep(0.5)\n"
         "data = iface.get_end_pose_raw()\n"
         "print(data)\n"
         "ok = any(v != 0 for v in data.values())\n"
         "sys.exit(0 if ok else 1)\n"
         "\""],
        capture_output=False
    )
    if result.returncode == 0:
        ok("EEF non-zero 확인")
        return True
    else:
        err("EEF all-zero 또는 연결 실패")
        return False


# ═══════════════════════════════════════════════
# STEP: teleop_check — send_action no-op 검증
# ═══════════════════════════════════════════════
def step_teleop_check(args):
    step_hdr("teleop send_action no-op 검증")
    iface = args.can_interface or CFG["can_interface"]

    cmd = (
        source_prefix() +
        "python3 -c \"\n"
        "from lerobot_robot_piper.config_piper import PiperConfig\n"
        "from lerobot_robot_piper.piper import Piper\n"
        "from lerobot_robot_piper.piper_slave_only import PiperSlaveOnly, PiperSlaveOnlyConfig\n"
        f"robot = Piper(PiperConfig(can_interface='{iface}', control_mode='teleop'))\n"
        "robot.connect()\n"
        "teleop = PiperSlaveOnly(PiperSlaveOnlyConfig())\n"
        "obs    = robot.get_observation()\n"
        "action = teleop.get_action()\n"
        "ret    = robot.send_action(action)\n"
        "eef_keys = [k for k in obs if 'pos' in k]\n"
        "match = all(obs[k] == action[k] for k in eef_keys)\n"
        "print('send_action returned:', ret == action)\n"
        "print('obs == action        :', match)\n"
        "robot.disconnect()\n"
        "import sys; sys.exit(0 if match else 1)\n"
        "\""
    )

    result = subprocess.run(["bash", "-c", cmd])
    if result.returncode == 0:
        ok("send_action no-op 확인 — 녹화 중 arm에 명령 안 보냄")
        return True
    else:
        err("send_action no-op 실패 — 코드 확인 필요")
        return False


# ═══════════════════════════════════════════════
# STEP: cam_check — 카메라 확인
# ═══════════════════════════════════════════════
def step_cam_check(args):
    step_hdr("카메라 확인")
    top_serial   = args.top_serial   or CFG["top_serial"]
    wrist_serial = args.wrist_serial or CFG["wrist_serial"]

    # RealSense
    info("RealSense 시리얼 확인...")
    rs_script = (
        "python3 -c \"\n"
        "import pyrealsense2 as rs\n"
        "ctx = rs.context()\n"
        "devices = list(ctx.devices)\n"
        "print(f'감지된 RealSense {len(devices)}대')\n"
        "serials = []\n"
        "for i, d in enumerate(devices):\n"
        "    name   = d.get_info(rs.camera_info.name)\n"
        "    serial = d.get_info(rs.camera_info.serial_number)\n"
        "    print(f'  [{i}] {name}  serial={serial}')\n"
        "    serials.append(serial)\n"
        f"top_ok   = '{top_serial}'   in serials\n"
        f"wrist_ok = '{wrist_serial}' in serials\n"
        "print('top   OK:', top_ok)\n"
        "print('wrist OK:', wrist_ok)\n"
        "\""
    )
    bash(rs_script)

    # OpenCV
    info("\nOpenCV 카메라 인덱스 확인...")
    cv_script = (
        "python3 -c \"\n"
        "import cv2\n"
        "found = []\n"
        "for idx in range(10):\n"
        "    cap = cv2.VideoCapture(idx)\n"
        "    if cap.isOpened():\n"
        "        ret, frame = cap.read()\n"
        "        shape = frame.shape if ret else 'read failed'\n"
        "        print(f'  index {idx}: OK shape={shape}')\n"
        "        found.append(idx)\n"
        "        cap.release()\n"
        "print('사용 가능한 인덱스:', found)\n"
        "\""
    )
    bash(cv_script)
    ok("카메라 확인 완료")
    return True


# ═══════════════════════════════════════════════
# STEP: rviz — URDF 세팅 + RViz 실행
# ═══════════════════════════════════════════════
def step_rviz(args):
    step_hdr("RViz 세팅 및 실행")

    urdf_dir = pathlib.Path(CFG["urdf_local_dir"])
    ros2_ws  = pathlib.Path(CFG["ros2_ws"])
    distro   = CFG["ros_distro"]

    # 1. agx_arm_urdf 클론
    if not urdf_dir.exists():
        info(f"agx_arm_urdf 클론 중: {urdf_dir}")
        ok_ = run(["git", "clone", CFG["urdf_repo"], str(urdf_dir)])
        if not ok_:
            err("agx_arm_urdf 클론 실패")
            return False
        ok("agx_arm_urdf 클론 완료")
    else:
        ok(f"agx_arm_urdf 이미 존재: {urdf_dir}")

    # 2. piper_description을 ros2_ws/src에 복사
    src_piper = urdf_dir / "piper"
    dst_piper = ros2_ws / "src" / "piper_description"
    if not dst_piper.exists():
        info(f"piper_description → {dst_piper}")
        if src_piper.exists():
            import shutil
            shutil.copytree(str(src_piper), str(dst_piper))
        else:
            # agx_arm_urdf 구조에 따라 경로 탐색
            candidates = list(urdf_dir.glob("**/piper_description"))
            if candidates:
                shutil.copytree(str(candidates[0]), str(dst_piper))
            else:
                warn("piper_description 디렉터리를 찾지 못함 — 수동 확인 필요")
                warn(f"agx_arm_urdf 내용: {list(urdf_dir.iterdir())}")

    # 3. colcon build
    info("colcon build 실행 중...")
    build_cmd = (
        f"source /opt/ros/{distro}/setup.bash && "
        f"cd {ros2_ws} && "
        f"colcon build --packages-select piper_description 2>&1 | tail -5"
    )
    built = run(["bash", "-c", build_cmd])
    if built:
        ok("piper_description 빌드 완료")
    else:
        warn("colcon build 실패 또는 이미 빌드됨 — 계속 진행")

    # 4. RViz 실행
    info("RViz 실행 중... (Ctrl+C로 종료)")
    info("RViz에서 /preview_trajectory (Marker) 토픽을 Add 해야 궤적이 보여")

    rviz_cmd = (
        f"source /opt/ros/{distro}/setup.bash && "
        f"source {ros2_ws}/install/setup.bash && "
        f"ros2 launch piper_description display_piper.launch.py"
    )
    try:
        subprocess.run(["bash", "-c", rviz_cmd])
    except KeyboardInterrupt:
        info("RViz 종료")
    return True


# ═══════════════════════════════════════════════
# STEP: data_check — 데이터셋 유효성 검사
# ═══════════════════════════════════════════════
def step_data_check(args):
    step_hdr("데이터셋 유효성 검사")
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        err("pip install pandas numpy")
        return False

    root = pathlib.Path(args.dataset_root or CFG["dataset_root"])
    if not root.exists():
        err(f"경로 없음: {root}")
        return False

    parquets = sorted(root.glob("data/**/*.parquet"))
    if not parquets:
        err(f"parquet 없음: {root}")
        return False

    info(f"parquet {len(parquets)}개 검사")
    safe = CFG["safe_range"]
    max_delta = CFG["max_delta_per_step"]

    total_zero = 0
    range_fails = []
    delta_warns = []

    for p in parquets:
        df = pd.read_parquet(p)

        # all-zero
        if "observation.state" in df.columns:
            states = np.array(df["observation.state"].tolist())
            nz = (states == 0).all(axis=1).sum()
            total_zero += nz
            if nz:
                warn(f"{p.name}: all-zero {nz}행")

        # action 범위
        if "action" in df.columns:
            acts = np.array(df["action"].tolist())
            for i, (ax, (lo, hi)) in enumerate(safe.items()):
                if i >= acts.shape[1]: break
                out = ((acts[:, i] < lo) | (acts[:, i] > hi)).sum()
                if out:
                    range_fails.append(f"{p.name}/{ax}: {out}행 범위초과")

            # delta
            if len(acts) > 1:
                deltas = np.abs(np.diff(acts, axis=0)).max(axis=1)
                bad = (deltas > max_delta).sum()
                if bad:
                    delta_warns.append(f"{p.name}: {bad}스텝 급격한 delta")

        # frame 연속성
        if "frame_index" in df.columns:
            idx = df["frame_index"].values
            gaps = ((idx[1:] - idx[:-1]) != 1).sum()
            if gaps:
                warn(f"{p.name}: frame_index 불연속 {gaps}곳")

    print()
    ok("all-zero 없음") if total_zero == 0 else err(f"all-zero 합계 {total_zero}행")
    ok("action 범위 정상") if not range_fails else [err(f"범위초과: {v}") for v in range_fails]
    ok("delta 정상") if not delta_warns else [warn(v) for v in delta_warns]

    passed = (total_zero == 0) and (not range_fails)
    ok("데이터 검사 통과") if passed else err("데이터 검사 실패")
    return passed


# ═══════════════════════════════════════════════
# STEP: calc_range — ACTION_MIN/MAX 계산
# ═══════════════════════════════════════════════
def step_calc_range(args):
    step_hdr("ACTION_MIN/MAX 자동 계산")
    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        err("pip install pandas numpy"); return False

    root = pathlib.Path(args.dataset_root or CFG["dataset_root"])
    parquets = sorted(root.glob("data/**/*.parquet"))
    all_actions = []
    for p in parquets:
        df = pd.read_parquet(p)
        if "action" in df.columns:
            all_actions.extend(df["action"].tolist())

    if not all_actions:
        err("action 없음"); return False

    acts = np.array(all_actions)
    margin = 0.05
    span   = acts.max(axis=0) - acts.min(axis=0)
    amin   = (acts.min(axis=0) - span * margin).tolist()
    amax   = (acts.max(axis=0) + span * margin).tolist()

    result = {"ACTION_MIN": amin, "ACTION_MAX": amax,
              "n_frames": len(all_actions)}
    out = pathlib.Path(args.range_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    axis_names = list(CFG["safe_range"].keys())
    for i, name in enumerate(axis_names):
        if i >= len(amin): break
        print(f"  {name:>8}: [{amin[i]:>10.0f}, {amax[i]:>10.0f}]")

    ok(f"저장: {out}")
    return True


# ═══════════════════════════════════════════════
# STEP: infer_dry — SmolVLA 추론 (로봇 없이)
# ═══════════════════════════════════════════════
def step_infer_dry(args):
    step_hdr("SmolVLA 추론 dry-run (use_devices=false)")
    if not args.pretrained_path:
        err("--pretrained_path 필요"); return False

    cmd = (
        source_prefix() +
        f"smolvla-inference "
        f"--pretrained_path={args.pretrained_path} "
        f"--use_devices=false "
        f"--max_steps={args.max_steps} "
        f"--task=\"{args.task}\" "
        f"--save_actions=true"
    )
    info(f"task: {args.task} | max_steps: {args.max_steps}")
    result = subprocess.run(["bash", "-c", cmd])
    if result.returncode != 0:
        err("smolvla-inference 실패")
        warn("smolvla_inference.py에 --save_actions 옵션 추가 필요")
        return False

    af = pathlib.Path(args.actions_file)
    if af.exists():
        ok(f"actions 저장: {af}")
        return True
    else:
        warn(f"{af} 없음 — smolvla_inference.py에 save_actions 로직 확인")
        return False


# ═══════════════════════════════════════════════
# STEP: rviz_preview — 궤적 RViz 시각화 + 안전검사
# ═══════════════════════════════════════════════
def step_rviz_preview(args):
    step_hdr("RViz 궤적 시각화 + 안전검사")

    af = pathlib.Path(args.actions_file)
    if not af.exists():
        err(f"{af} 없음 — 먼저 --step infer_dry 실행")
        return False

    with open(af) as f:
        data = json.load(f)
    actions = data.get("actions", [])
    if not actions:
        err("actions 비어있음"); return False

    try:
        import numpy as np
    except ImportError:
        err("pip install numpy"); return False

    acts = np.array(actions)
    safe = CFG["safe_range"]
    max_delta = CFG["max_delta_per_step"]

    # ── 안전 검사 ──
    info(f"총 {len(actions)} steps 안전검사")
    safety_ok = True

    print(f"\n  {'축':>8}  {'최솟값':>12}  {'최댓값':>12}  {'안전범위':>25}  결과")
    print(f"  {'─'*75}")
    for i, (ax, (lo, hi)) in enumerate(safe.items()):
        if i >= acts.shape[1]: break
        col = acts[:, i]
        vmin, vmax = col.min(), col.max()
        fail = ((col < lo) | (col > hi)).sum()
        result = f"{C.GREEN}OK{C.RESET}" if fail == 0 else f"{C.RED}FAIL({fail}행){C.RESET}"
        print(f"  {ax:>8}  {vmin:>12.0f}  {vmax:>12.0f}  [{lo:>10}, {hi:>10}]  {result}")
        if fail: safety_ok = False

    print()
    if len(acts) > 1:
        deltas = np.abs(np.diff(acts, axis=0)).max(axis=1)
        bad_steps = np.where(deltas > max_delta)[0]
        if len(bad_steps) == 0:
            ok(f"모든 step delta < {max_delta}")
        else:
            for s in bad_steps:
                warn(f"step {s}→{s+1}: delta={deltas[s]:.0f}")
            safety_ok = False

    print()
    if safety_ok:
        ok("사전 안전검사 통과 — RViz에 초록 선으로 표시")
    else:
        err("사전 안전검사 실패 — RViz에 빨간 선으로 표시 / 실제 arm 연결 금지")

    # ── RViz 퍼블리시 ──
    try:
        import rclpy
        from rclpy.node import Node
        from visualization_msgs.msg import Marker, MarkerArray
        from geometry_msgs.msg import Point
        from std_msgs.msg import ColorRGBA
    except ImportError:
        err("rclpy 없음 — source /opt/ros/humble/setup.bash 후 재실행")
        return False

    class PreviewNode(Node):
        def __init__(self):
            super().__init__("piper_preview")
            self.pub  = self.create_publisher(Marker,      "/preview_trajectory", 10)
            self.pubm = self.create_publisher(MarkerArray, "/preview_waypoints",  10)

        def publish(self, acts, safe_flag):
            m = Marker()
            m.header.frame_id = "base_link"
            m.header.stamp    = self.get_clock().now().to_msg()
            m.ns = "traj"; m.id = 0
            m.type = Marker.LINE_STRIP; m.action = Marker.ADD
            m.scale.x = 0.005
            m.color = ColorRGBA(
                r=0.1 if safe_flag else 1.0,
                g=1.0 if safe_flag else 0.1,
                b=0.1, a=1.0)
            for a in acts:
                p = Point()
                p.x = float(a[0]) / 1_000_000.0
                p.y = float(a[1]) / 1_000_000.0
                p.z = float(a[2]) / 1_000_000.0
                m.points.append(p)
            self.pub.publish(m)

            # 시작/끝 구체
            ma = MarkerArray()
            for idx, (label, color, act) in enumerate([
                ("start", ColorRGBA(r=0.0,g=0.5,b=1.0,a=1.0), acts[0]),
                ("end",   ColorRGBA(r=1.0,g=0.5,b=0.0,a=1.0), acts[-1]),
            ]):
                sm = Marker()
                sm.header.frame_id = "base_link"
                sm.header.stamp = self.get_clock().now().to_msg()
                sm.ns = label; sm.id = idx
                sm.type = Marker.SPHERE; sm.action = Marker.ADD
                sm.pose.position.x = float(act[0]) / 1_000_000.0
                sm.pose.position.y = float(act[1]) / 1_000_000.0
                sm.pose.position.z = float(act[2]) / 1_000_000.0
                sm.scale.x = sm.scale.y = sm.scale.z = 0.025
                sm.color = color
                ma.markers.append(sm)
            self.pubm.publish(ma)

    rclpy.init()
    node = PreviewNode()
    color_hint = "초록 선 = 안전" if safety_ok else "빨간 선 = 위험"
    info(f"RViz에서 /preview_trajectory 토픽 Add — {color_hint}")
    info("파란 구체=시작, 주황 구체=끝  |  Ctrl+C로 종료\n")

    try:
        while rclpy.ok():
            node.publish(actions, safety_ok)
            rclpy.spin_once(node, timeout_sec=1.0)
            time.sleep(1.0)
    except KeyboardInterrupt:
        info("RViz 퍼블리시 종료")
    finally:
        node.destroy_node()
        rclpy.shutdown()

    return safety_ok


# ═══════════════════════════════════════════════
# STEP: replay_dry
# ═══════════════════════════════════════════════
def step_replay_dry(args):
    step_hdr("piper-replay dry-run (use_devices=false)")
    repo_id  = args.dataset_repo_id or CFG["dataset_repo_id"]
    root     = args.dataset_root    or CFG["dataset_root"]
    cmd = (
        source_prefix() +
        f"piper-replay "
        f"--dataset_repo_id={repo_id} "
        f"--dataset_root={root} "
        f"--episode={args.episode} "
        f"--use_devices=false "
        f"--start_frame={args.start_frame} "
        f"--max_steps={args.max_steps} "
        f"--replay_fps={args.replay_fps}"
    )
    info(f"episode={args.episode} | max_steps={args.max_steps}")
    result = subprocess.run(["bash", "-c", cmd])
    if result.returncode == 0:
        ok("replay dry-run 통과")
        return True
    err("replay dry-run 실패")
    return False


# ═══════════════════════════════════════════════
# STEP: replay_real — 실제 arm replay
# ═══════════════════════════════════════════════
def step_replay_real(args):
    step_hdr("piper-replay 실제 arm")

    print(f"{C.YELLOW}{'─'*55}{C.RESET}")
    print(f"{C.YELLOW}  실제 로봇이 움직입니다. 확인하세요:{C.RESET}")
    print("  □ arm 주변 50cm 이내 사람/물체 없음")
    print("  □ CAN 활성화 완료")
    print("  □ EEF non-zero 확인 완료")
    print("  □ dry-run 통과 완료")
    print(f"  □ max_steps={args.max_steps} (5 권장)")
    print(f"{C.YELLOW}{'─'*55}{C.RESET}\n")

    if not confirm("실제 arm을 움직이겠습니까?"):
        info("취소"); return False

    repo_id = args.dataset_repo_id or CFG["dataset_repo_id"]
    root    = args.dataset_root    or CFG["dataset_root"]
    iface   = args.can_interface   or CFG["can_interface"]
    cmd = (
        source_prefix() +
        f"piper-replay "
        f"--dataset_repo_id={repo_id} "
        f"--dataset_root={root} "
        f"--episode={args.episode} "
        f"--use_devices=true "
        f"--can_interface={iface} "
        f"--start_frame={args.start_frame} "
        f"--max_steps={args.max_steps} "
        f"--replay_fps={args.replay_fps}"
    )
    info("이상 시 즉시 Ctrl+C, 멈추면 sudo ip link set can0 down\n")
    result = subprocess.run(["bash", "-c", cmd])
    if result.returncode == 0:
        ok("replay 실제 arm 완료"); return True
    err("replay 실패"); return False


# ═══════════════════════════════════════════════
# STEP: infer_real — SmolVLA 실제 arm 추론
# ═══════════════════════════════════════════════
def step_infer_real(args):
    step_hdr("SmolVLA 실제 arm 추론")
    if not args.pretrained_path:
        err("--pretrained_path 필요"); return False

    print(f"{C.RED}{'─'*55}{C.RESET}")
    print(f"{C.RED}  모델 추론 결과를 실제 arm에 전송합니다:{C.RESET}")
    print("  □ arm 주변 50cm 이내 사람/물체 없음")
    print("  □ CAN 활성화 + EEF non-zero 확인")
    print("  □ infer_dry에서 action 범위 정상")
    print("  □ rviz_preview에서 궤적 확인")
    print("  □ replay_dry / replay_real 통과")
    print(f"  □ max_steps={args.max_steps} (5 권장)")
    print(f"{C.RED}{'─'*55}{C.RESET}\n")

    if not confirm("실제 arm에 추론을 전송하겠습니까?"):
        info("취소"); return False

    iface        = args.can_interface   or CFG["can_interface"]
    top_serial   = args.top_serial      or CFG["top_serial"]
    wrist_serial = args.wrist_serial    or CFG["wrist_serial"]
    cmd = (
        source_prefix() +
        f"smolvla-inference "
        f"--pretrained_path={args.pretrained_path} "
        f"--can_interface={iface} "
        f"--top_serial={top_serial} "
        f"--wrist_serial={wrist_serial} "
        f"--max_steps={args.max_steps} "
        f"--task=\"{args.task}\""
    )
    info("이상 시 즉시 Ctrl+C, 멈추면 sudo ip link set can0 down\n")
    result = subprocess.run(["bash", "-c", cmd])
    if result.returncode == 0:
        ok("SmolVLA 실제 arm 추론 완료"); return True
    err("추론 실패"); return False


# ═══════════════════════════════════════════════
# STEP: session — 세션 시작 루틴
# ═══════════════════════════════════════════════
def step_session(args):
    step_hdr("세션 시작 루틴")
    info("실험실 시작 시 매번 실행하는 점검 순서입니다.\n")

    results = {}

    # 1. pip install -e .
    info("패키지 설치 상태 확인 (pip install -e .)")
    install_cmd = (
        source_prefix() +
        f"cd {CFG['ugrp_dir']} && pip install -e . -q"
    )
    r = subprocess.run(["bash", "-c", install_cmd])
    if r.returncode == 0:
        ok("pip install -e . 완료")
    else:
        warn("pip install -e . 실패 — 경로 확인 필요")

    # 2. CAN 활성화
    results["can_up"] = step_can_up(args)
    if not results["can_up"]:
        err("CAN 활성화 실패 — 이후 단계 중단")
        return

    # 3. EEF non-zero
    results["eef_check"] = step_eef_check(args)

    # 4. 카메라 확인
    results["cam_check"] = step_cam_check(args)

    # 5. teleop no-op (EEF OK일 때만)
    if results.get("eef_check"):
        results["teleop_check"] = step_teleop_check(args)

    # ── 결과 요약 ──
    print(f"\n{C.BOLD}{'─'*40}{C.RESET}")
    info("세션 시작 점검 결과:")
    for name, passed in results.items():
        mark  = f"{C.GREEN}✓{C.RESET}" if passed else f"{C.RED}✗{C.RESET}"
        print(f"  {mark} {name}")

    all_ok = all(results.values())
    print()
    if all_ok:
        ok("세션 준비 완료 — 실험 시작 가능")
    else:
        warn("일부 항목 실패 — 확인 후 진행")


# ═══════════════════════════════════════════════
# STEP: full_validate — 검증 전체 자동 실행
# ═══════════════════════════════════════════════
def step_full_validate(args):
    step_hdr("FULL VALIDATE: 전체 검증 자동 실행")
    info("data_check → calc_range → infer_dry → rviz_preview → replay_dry\n")

    steps = [
        ("data_check",   step_data_check),
        ("calc_range",   step_calc_range),
        ("infer_dry",    step_infer_dry),
        ("rviz_preview", step_rviz_preview),
        ("replay_dry",   step_replay_dry),
    ]

    results = {}
    for name, fn in steps:
        results[name] = fn(args)
        if not results[name] and name in ("data_check",):
            err(f"{name} 실패 — 이후 단계 중단")
            break

    # 결과 요약
    step_hdr("전체 검증 결과")
    for name, passed in results.items():
        mark  = f"{C.GREEN}✓{C.RESET}" if passed else f"{C.RED}✗{C.RESET}"
        print(f"  {mark} {name}")

    print()
    if all(results.values()):
        ok("모든 검증 통과 — 실제 arm 실험 준비 완료")
        info("다음: python piper_session.py --step replay_real")
        info("      python piper_session.py --step infer_real")
    else:
        warn("일부 실패 — 실제 arm 실험 전 수정 필요")


# ───────────────────────────────────────────────
# argparse
# ───────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="UGRP PiPER 세션 자동화",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--step", required=True,
        choices=["env_setup","rviz","can_up","can_down","eef_check",
                 "teleop_check","cam_check","data_check","calc_range",
                 "infer_dry","rviz_preview","replay_dry","replay_real",
                 "infer_real","session","full_validate"],
        help="실행할 단계")

    # 경로
    p.add_argument("--dataset_root",    default="", help="데이터셋 루트")
    p.add_argument("--dataset_repo_id", default="", help="LeRobot repo id")
    p.add_argument("--pretrained_path", default="", help="모델 체크포인트")
    p.add_argument("--range_output",    default="action_range.json")
    p.add_argument("--actions_file",    default="predicted_actions.json")

    # 태스크/스텝
    p.add_argument("--task",        default="pick the pan")
    p.add_argument("--max_steps",   type=int, default=5)
    p.add_argument("--episode",     type=int, default=0)
    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--replay_fps",  type=int, default=5)

    # 하드웨어
    p.add_argument("--can_interface", default="")
    p.add_argument("--top_serial",    default="")
    p.add_argument("--wrist_serial",  default="")

    return p.parse_args()


def main():
    args = parse_args()

    dispatch = {
        "env_setup":     step_session,      # alias
        "rviz":          step_rviz,
        "can_up":        step_can_up,
        "can_down":      step_can_down,
        "eef_check":     step_eef_check,
        "teleop_check":  step_teleop_check,
        "cam_check":     step_cam_check,
        "data_check":    step_data_check,
        "calc_range":    step_calc_range,
        "infer_dry":     step_infer_dry,
        "rviz_preview":  step_rviz_preview,
        "replay_dry":    step_replay_dry,
        "replay_real":   step_replay_real,
        "infer_real":    step_infer_real,
        "session":       step_session,
        "full_validate": step_full_validate,
    }

    dispatch[args.step](args)


if __name__ == "__main__":
    main()
