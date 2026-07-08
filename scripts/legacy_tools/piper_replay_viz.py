#!/usr/bin/env python3
"""
piper_replay_viz.py — 녹화된 episode(parquet)를 RViz에서 실제 로봇 모델로 재생

기존 EEF 마커(x/y/z 선) 방식에서 joint_states 기반으로 재설계함:
lerobot_robot_piper는 joint-space(-100~100, gripper 0~100)만 쓰고 EEF 좌표를
전혀 다루지 않으므로, action의 앞 3컬럼(joint1~3)을 x/y/z로 그리던 예전 방식은
관절각을 좌표로 착각해서 그리는 것과 같아 물리적으로 의미가 없었음.

대신 agx_arm_urdf(https://github.com/agilexrobotics/agx_arm_urdf)의
piper/urdf/piper_description.urdf + piper_with_gripper_description.xacro를
직접 확인한 결과:
  - joint1~joint6: revolute, radian 단위, lerobot_robot_piper의 모터 이름과 동일
  - gripper: prismatic(직선), 0~0.1m, gripper_joint1/gripper_joint2는
    <mimic joint="gripper">로 자동 추종 (별도 publish 불필요)
그래서 sensor_msgs/JointState를 /joint_states에 publish하면
robot_state_publisher가 TF를 계산해서 RViz의 로봇 메시 자체가 그대로
움직임 — FK를 직접 구현할 필요 없음.

사용법:
    # 터미널 1: RViz + robot_state_publisher 먼저 실행
    #   (piper_session.py --step rviz 또는 agx_arm_urdf의
    #    display_piper.launch.py 등, /joint_states 토픽을 구독하는 launch)

    # 터미널 2: 이 스크립트 실행
    python piper_replay_viz.py --dataset_root /path/to/dataset --episode 0

옵션:
    --dataset_root   LeRobotDataset 루트 경로 (parquet들이 들어있는 폴더)
    --episode        episode 인덱스 (기본 0)
    --column         재생할 컬럼 이름 (기본 action, observation.state도 가능)
    --rate           프레임 간 퍼블리시 주기 초 (기본 0.1 = 10Hz)
    --loop           끝까지 재생 후 처음부터 반복 (기본은 1회 재생 후 마지막 자세 유지)
    --joint_state_topic   publish할 토픽 (기본 /joint_states)

주의:
    이 스크립트는 lerobot_robot_piper의 PiperMotorsBus를 직접 import하지 않음
    — 그 클래스의 생성자가 C_PiperInterface_V2(port)를 즉시 호출해서 CAN
    소켓이 없으면 바로 ConnectionError가 나기 때문 (하드웨어 없이 순수
    재생/시각화만 하려는 이 스크립트의 목적과 안 맞음). 대신 동일한
    calibration 표를 아래에 그대로 복사해서 순수 변환 함수로만 씀
    (lerobot_robot_piper/piper_follower.py, piper_leader.py에서 확인한 값과 동일).

    joint5/joint6의 URDF <limit>은 calibration 표와 정확히 일치하지 않음
    (예: joint5 calibration은 ±65°, URDF <limit>은 ±70°). 이 스크립트는
    calibration 표(lerobot_robot_piper가 실제로 쓰는 정규화 기준)를 진실로
    삼아 변환하고, URDF <limit>은 RViz 쪽에서 별도로 clamp함 — 실제 하드웨어
    한계는 로봇 PC에서 재확인 필요.

    gripper 물리 단위: README의 Motor Configuration 표는 "0-68 deg"라고
    적혀 있지만, 실제 URDF의 gripper joint는 prismatic(직선, 미터 단위)이라
    이 라벨은 오기로 보임 — raw 값(0~68000)을 mm로 해석해 미터로 변환함
    (68000 raw → 68mm → 0.068m, AgileX Piper 실제 그리퍼 스트로크와 일치).
    로봇 PC에서 실제 그리퍼 열림 정도와 비교해서 맞는지 확인할 것.
"""

import argparse
import math
import pathlib
import sys
import time

import numpy as np
import pandas as pd


class C:
    RESET = "\033[0m"; BOLD = "\033[1m"
    RED = "\033[91m"; GREEN = "\033[92m"; YELLOW = "\033[93m"; CYAN = "\033[96m"

def ok(m):   print(f"{C.GREEN}[OK]{C.RESET} {m}")
def warn(m): print(f"{C.YELLOW}[WARN]{C.RESET} {m}")
def err(m):  print(f"{C.RED}[ERROR]{C.RESET} {m}")
def info(m): print(f"{C.CYAN}[INFO]{C.RESET} {m}")


# ═══════════════════════════════════════════════
# 1. joint-space calibration (lerobot_robot_piper/piper_follower.py,
#    piper_leader.py의 MotorCalibration 값과 동일 — PiperMotorsBus는
#    생성자가 CAN 연결을 즉시 시도해서 하드웨어 없이 못 쓰므로 여기 값만 복사)
# ═══════════════════════════════════════════════
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_NAME = "gripper"

# (motor: (range_min_raw, range_max_raw)) — raw는 0.001deg 단위 (joint) /
# 0.001mm 단위 (gripper)로 추정 (piper_sdk 표준 관례, README 물리 범위와 대조 확인됨)
CALIBRATION_RAW = {
    "joint1": (-150_000, 150_000),
    "joint2": (0, 180_000),
    "joint3": (-170_000, 0),
    "joint4": (-100_000, 100_000),
    "joint5": (-65_000, 65_000),
    "joint6": (-100_000, 130_000),
    "gripper": (0, 68_000),
}


def unnormalize_to_physical(motor: str, normalized_val: float) -> float:
    """정규화값(-100~100, gripper 0~100) -> raw(0.001 단위) -> 물리 단위.
    joint1~6은 라디안, gripper는 미터로 반환.
    PiperMotorsBus._unnormalize()와 동일한 수식 (motors/piper_motors_bus.py 참고)."""
    min_, max_ = CALIBRATION_RAW[motor]

    if motor == GRIPPER_NAME:
        bounded = min(100.0, max(0.0, normalized_val))
        raw = (bounded / 100.0) * (max_ - min_) + min_
        mm = raw / 1000.0
        return mm / 1000.0  # meters

    bounded = min(100.0, max(-100.0, normalized_val))
    raw = ((bounded + 100) / 200) * (max_ - min_) + min_
    degrees = raw / 1000.0
    return math.radians(degrees)


# ═══════════════════════════════════════════════
# 2. parquet 로드
# ═══════════════════════════════════════════════
def load_episode(dataset_root: pathlib.Path, episode: int) -> pd.DataFrame:
    candidates = sorted(dataset_root.glob("data/**/*.parquet"))
    if not candidates:
        err(f"parquet 파일 없음: {dataset_root}/data/**/*.parquet")
        sys.exit(1)

    info(f"전체 {len(candidates)}개 parquet 발견")

    target = None
    for p in candidates:
        if f"{episode:06d}" in p.stem or f"episode_{episode}" in p.stem:
            target = p
            break

    if target is None:
        if episode < len(candidates):
            target = candidates[episode]
            warn(f"파일명으로 episode {episode}를 못 찾음 — 정렬 순서 기준 {target.name} 사용")
        else:
            err(f"episode {episode} 파일 없음 (전체 {len(candidates)}개)")
            sys.exit(1)

    info(f"로드: {target}")
    df = pd.read_parquet(target)
    if "episode_index" in df.columns:
        df = df[df["episode_index"] == episode].reset_index(drop=True)
    info(f"{len(df)} frame 로드 완료, 컬럼: {list(df.columns)[:8]}...")
    return df


# ═══════════════════════════════════════════════
# 3. 컬럼 -> joint position 시퀀스 추출 + 정규화 범위 사전 점검
# ═══════════════════════════════════════════════
def extract_joint_sequence(df: pd.DataFrame, column: str) -> np.ndarray:
    """(frame, 7) 정규화값 배열 반환 — 순서는 [joint1..joint6, gripper]."""
    if column not in df.columns:
        err(f"'{column}' 컬럼 없음. 사용 가능한 컬럼: {list(df.columns)}")
        sys.exit(1)

    arr = np.array(df[column].tolist())
    if arr.ndim != 2 or arr.shape[1] != 7:
        err(f"'{column}' shape={arr.shape} — joint1~6+gripper(7) 형태가 아님. "
            f"이 스크립트는 joint-space 데이터 전용 (EEF 데이터는 지원 안 함)")
        sys.exit(1)

    return arr


def range_check(arr: np.ndarray, label: str) -> bool:
    """joint_check(piper_session.py)와 동일한 논리 — 정상 기록이면 항상
    -100~100(gripper 0~100) 안에 있어야 함. 벗어나면 기록 당시 이상 신호."""
    print()
    info(f"── {label} 정규화 범위 점검 ──────────")
    names = JOINT_NAMES + [GRIPPER_NAME]
    all_ok = True

    for i, name in enumerate(names):
        col = arr[:, i]
        lo, hi = (-100, 100) if name != GRIPPER_NAME else (0, 100)
        vmin, vmax = col.min(), col.max()
        out = ((col < lo) | (col > hi)).sum()
        status = f"{C.GREEN}OK{C.RESET}" if out == 0 else f"{C.RED}범위초과({out}행){C.RESET}"
        print(f"  {name:>8}  min={vmin:>8.2f}  max={vmax:>8.2f}  [{lo},{hi}]  {status}")
        if out:
            all_ok = False

    if all_ok:
        ok(f"{label} 전체 정규화 범위 정상")
    else:
        warn(f"{label} 일부 프레임이 정규화 범위를 벗어남 — 기록 당시 이상 가능성")
    return all_ok


# ═══════════════════════════════════════════════
# 4. RViz 퍼블리시 (joint_states)
# ═══════════════════════════════════════════════
def run_rviz(seq: np.ndarray, rate: float, loop: bool, topic: str):
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
    except ImportError:
        err("rclpy/sensor_msgs 없음 — ROS2 환경에서 실행해야 함")
        err("source /opt/ros/humble/setup.bash 후 재실행")
        sys.exit(1)

    joint_msg_names = JOINT_NAMES + [GRIPPER_NAME]

    class ReplayJointStateNode(Node):
        def __init__(self):
            super().__init__("piper_replay_viz")
            self.pub = self.create_publisher(JointState, topic, 10)

        def publish_frame(self, normalized_frame: np.ndarray):
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = joint_msg_names
            msg.position = [
                unnormalize_to_physical(name, float(val))
                for name, val in zip(joint_msg_names, normalized_frame)
            ]
            self.pub.publish(msg)

    rclpy.init()
    node = ReplayJointStateNode()

    print()
    info(f"토픽: {topic} (JointState) — robot_state_publisher가 이 토픽을 구독해야")
    info("      RViz의 로봇 메시가 움직임 (agx_arm_urdf의 display 계열 launch 실행 필요)")
    info(f"{len(seq)} frame을 {rate}초 간격으로 {'반복' if loop else '1회'} 재생 — Ctrl+C로 종료\n")

    try:
        first = True
        while rclpy.ok() and (first or loop):
            first = False
            for frame in seq:
                if not rclpy.ok():
                    break
                node.publish_frame(frame)
                rclpy.spin_once(node, timeout_sec=0.0)
                time.sleep(rate)
            # 마지막 프레임을 반복 재생 없을 때도 유지해서 보이게 함
            if not loop:
                info("재생 종료 — 마지막 자세 유지 (Ctrl+C로 종료)")
                while rclpy.ok():
                    node.publish_frame(seq[-1])
                    rclpy.spin_once(node, timeout_sec=0.0)
                    time.sleep(0.5)
    except KeyboardInterrupt:
        info("종료")
    finally:
        node.destroy_node()
        rclpy.shutdown()


# ═══════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description="녹화된 episode를 RViz에서 실제 로봇 모델로 재생")
    p.add_argument("--dataset_root", required=True, help="LeRobotDataset 루트 경로")
    p.add_argument("--episode", type=int, default=0, help="episode 인덱스")
    p.add_argument("--column", default="action", help="재생할 컬럼 (action / observation.state)")
    p.add_argument("--rate", type=float, default=0.1, help="프레임 간 퍼블리시 주기(초), 기본 10Hz")
    p.add_argument("--loop", action="store_true", help="끝까지 재생 후 처음부터 반복")
    p.add_argument("--joint_state_topic", default="/joint_states",
                   help="publish할 JointState 토픽 (robot_state_publisher가 구독하는 토픽과 일치해야 함)")
    args = p.parse_args()

    root = pathlib.Path(args.dataset_root)
    if not root.exists():
        err(f"경로 없음: {root}")
        sys.exit(1)

    df = load_episode(root, args.episode)
    seq = extract_joint_sequence(df, args.column)
    range_check(seq, args.column)

    run_rviz(seq, args.rate, args.loop, args.joint_state_topic)


if __name__ == "__main__":
    main()
