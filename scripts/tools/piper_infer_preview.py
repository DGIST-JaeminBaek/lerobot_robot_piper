#!/usr/bin/env python3
"""
piper_infer_preview.py — 정책(SmolVLA 등) 추론 결과를 실제 로봇에 보내기 전에
RViz에서 먼저 확인하는 open-loop 미리보기 도구.

배경: teleop_ui.py의 Infer 프리셋(lerobot-record --policy.path=...)은 Launch를
누르는 즉시 정책이 예측한 action을 실제 로봇에 전송함 — 사전 미리보기 경로가
없음. 이 스크립트는 이미 녹화된 episode의 카메라 프레임을 정책에 순서대로
"지금 보고 있는 화면"인 것처럼 먹여서 예측된 action을 뽑아내고, 실제 로봇 대신
piper_replay_viz.py와 동일한 방식(joint_states publish)으로 RViz에서 재생함.

중요한 한계 — open-loop 미리보기라는 것: 실제 정책 실행(closed-loop)은
"정책이 예측한 action → 로봇이 실제로 움직임 → 그 결과를 다시 관찰 → 다음
action 예측"을 반복함. 이 스크립트는 로봇이 실제로 안 움직이므로, 매 스텝의
관찰이 전부 "녹화 당시 실제로 있었던 다음 프레임"임 — 정책이 예측한 대로
로봇이 움직였다면 봤을 프레임이 아님. 그래서 첫 몇 스텝은 참고할 만하지만,
스텝이 누적될수록 실제 실행 결과와 벌어질 수 있음. 그래도 "이 정책이 이
episode 상황에서 대략 어느 방향으로 움직이려 하는지" 감을 잡는 용도로는 유효함.

사용법:
    # 터미널 1: RViz + robot_state_publisher 먼저 실행
    #   (piper_session.py --step rviz 또는 agx_arm_urdf의 display 계열 launch)

    # 터미널 2:
    python piper_infer_preview.py \\
        --dataset_root record_sample/local/piper_write_light_rs_10s_3eps_v5 \\
        --episode 0 \\
        --policy_path outputs/train/piper_smolvla/checkpoints/last/pretrained_model

옵션:
    --dataset_root   관찰 프레임(카메라+상태)으로 쓸 LeRobotDataset 루트
    --episode        episode 인덱스
    --policy_path    정책 체크포인트 경로 (로컬 경로 또는 HF repo id) —
                      lerobot-record의 --policy.path와 동일한 값
    --task           정책에 넘길 task 문자열 (기본: episode에 기록된 task 그대로 사용)
    --device         cpu/cuda/mps (기본 cpu)
    --rate           RViz 퍼블리시 주기(초), 기본 0.1 (10Hz)
    --joint_state_topic  기본 /joint_states

주의: policy 호출 방식(make_policy/make_pre_post_processors/predict_action)은
lerobot/scripts/lerobot_record.py의 record_loop()를 그대로 따름 — 소스에서
직접 확인한 시그니처 그대로 씀, 추측 없음. 다만 이 프로젝트는 아직 SmolVLA를
실제로 학습시키지 않아서 체크포인트가 없고, 이 스크립트를 end-to-end로
실행해보지는 못했음 — 로봇 PC에서 실제 체크포인트로 검증 필요. 카메라/CAN
하드웨어는 전혀 안 씀 (정책 로딩 + 추론 + RViz publish만 함).
"""

import argparse
import math
import pathlib
import sys
import time

import numpy as np


class C:
    RESET = "\033[0m"; BOLD = "\033[1m"
    RED = "\033[91m"; GREEN = "\033[92m"; YELLOW = "\033[93m"; CYAN = "\033[96m"

def ok(m):   print(f"{C.GREEN}[OK]{C.RESET} {m}")
def warn(m): print(f"{C.YELLOW}[WARN]{C.RESET} {m}")
def err(m):  print(f"{C.RED}[ERROR]{C.RESET} {m}")
def info(m): print(f"{C.CYAN}[INFO]{C.RESET} {m}")


# ═══════════════════════════════════════════════
# joint-space calibration — piper_replay_viz.py와 동일한 표를 그대로 복사.
# PiperMotorsBus는 생성자가 CAN 연결을 즉시 시도해서 하드웨어 없이 못 쓰므로
# lerobot_robot_piper/piper_follower.py, piper_leader.py의 MotorCalibration
# 값만 순수 변환 함수로 옮겨옴.
# ═══════════════════════════════════════════════
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_NAME = "gripper"

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
    """정규화값(-100~100, gripper 0~100) -> 물리 단위 (joint: 라디안, gripper: 미터).
    PiperMotorsBus._unnormalize()와 동일한 수식."""
    min_, max_ = CALIBRATION_RAW[motor]

    if motor == GRIPPER_NAME:
        bounded = min(100.0, max(0.0, normalized_val))
        raw = (bounded / 100.0) * (max_ - min_) + min_
        return (raw / 1000.0) / 1000.0  # meters

    bounded = min(100.0, max(-100.0, normalized_val))
    raw = ((bounded + 100) / 200) * (max_ - min_) + min_
    return math.radians(raw / 1000.0)


# ═══════════════════════════════════════════════
# 정책 로딩 + 프레임별 추론
# lerobot/scripts/lerobot_record.py의 record_loop()가 policy를 호출하는 방식을
# 그대로 따름 (소스 확인: make_policy, make_pre_post_processors, predict_action,
# make_robot_action, build_dataset_frame).
# ═══════════════════════════════════════════════
def load_policy(policy_path: str, ds_meta, device: str):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(policy_path)
    cfg.pretrained_path = policy_path
    cfg.device = device

    policy = make_policy(cfg, ds_meta=ds_meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=cfg.pretrained_path,
        dataset_stats=ds_meta.stats,
        preprocessor_overrides={"device_processor": {"device": cfg.device}},
    )
    return cfg, policy, preprocessor, postprocessor


def frame_to_raw_observation(dataset, item: dict) -> dict:
    """LeRobotDataset.__getitem__() 결과(item)를 build_dataset_frame()이 기대하는
    "하드웨어에서 막 읽은 것 같은" raw dict로 변환.
    - state: dataset.features["observation.state"]["names"] 순서로 평평하게 풀어서
      {"joint1.pos": float, ...} 형태로 (robot.get_observation()과 동일한 key 형태)
    - image: LeRobotDataset은 CHW float[0,1] torch tensor로 주지만,
      prepare_observation_for_inference()는 HWC uint8 [0,255] numpy를 기대함
      (소스 확인: policies/utils.py의 permute(2,0,1)+/255 처리가 여기서 다시
      일어남) — 그래서 여기서 미리 CHW float -> HWC uint8로 역변환해둠.
      카메라 dict key는 full feature명(예: observation.images.top)이 아니라
      build_dataset_frame이 prefix를 떼어내는 short name(예: "top")을 씀."""
    obs: dict = {}

    state_names = dataset.features["observation.state"]["names"]
    state_vals = item["observation.state"].numpy()
    for name, val in zip(state_names, state_vals):
        obs[name] = float(val)

    for full_key in dataset.meta.camera_keys:
        short_key = full_key.removeprefix("observation.images.")
        chw = item[full_key]  # torch float [0,1], (C, H, W)
        hwc_uint8 = (chw.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        obs[short_key] = hwc_uint8

    return obs


def run_inference_sequence(dataset_root: pathlib.Path, episode: int, policy_path: str,
                            task: str | None, device: str) -> np.ndarray:
    """episode의 각 프레임을 정책에 순서대로 먹여서 예측 action 시퀀스((frame, 7)
    정규화값 배열, [joint1..joint6, gripper] 순서)를 뽑아냄."""
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import OBS_STR, build_dataset_frame
    from lerobot.policies.utils import make_robot_action
    from lerobot.utils.control_utils import predict_action
    from lerobot.utils.utils import get_safe_torch_device

    info(f"LeRobotDataset 로드 중: {dataset_root}")
    dataset = LeRobotDataset(repo_id=dataset_root.name, root=dataset_root, episodes=[episode])
    ok(f"{dataset.num_frames} frame 로드 완료 (episode {episode})")

    info(f"정책 로드 중: {policy_path} (device={device})")
    cfg, policy, preprocessor, postprocessor = load_policy(policy_path, dataset.meta, device)
    ok(f"정책 로드 완료: type={cfg.type}")

    joint_order = JOINT_NAMES + [GRIPPER_NAME]
    action_names = [f"{n}.pos" for n in joint_order]

    predicted = []
    torch_device = get_safe_torch_device(policy.config.device)

    for i in range(dataset.num_frames):
        item = dataset[i]
        raw_obs = frame_to_raw_observation(dataset, item)
        observation_frame = build_dataset_frame(dataset.features, raw_obs, prefix=OBS_STR)

        task_str = task if task else str(item.get("task", ""))

        action_values = predict_action(
            observation=observation_frame,
            policy=policy,
            device=torch_device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=getattr(policy.config, "use_amp", False),
            task=task_str,
            robot_type="piper_follower",
        )
        act_dict = make_robot_action(action_values, dataset.features)
        predicted.append([act_dict[name] for name in action_names])

        if (i + 1) % 20 == 0 or i == dataset.num_frames - 1:
            info(f"  {i + 1}/{dataset.num_frames} 프레임 추론 완료")

    return np.array(predicted, dtype=np.float32)


# ═══════════════════════════════════════════════
# RViz 퍼블리시 — piper_replay_viz.py의 run_rviz()와 동일한 방식
# ═══════════════════════════════════════════════
def run_rviz(seq: np.ndarray, rate: float, topic: str):
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
    except ImportError:
        err("rclpy/sensor_msgs 없음 — ROS2 환경에서 실행해야 함")
        err("source /opt/ros/humble/setup.bash 후 재실행")
        sys.exit(1)

    joint_msg_names = JOINT_NAMES + [GRIPPER_NAME]

    class InferPreviewNode(Node):
        def __init__(self):
            super().__init__("piper_infer_preview")
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
    node = InferPreviewNode()

    print()
    info(f"토픽: {topic} (JointState) — 정책이 예측한 action을 재생 (open-loop 미리보기)")
    info(f"{len(seq)} frame을 {rate}초 간격으로 1회 재생 — Ctrl+C로 종료\n")

    try:
        for frame in seq:
            if not rclpy.ok():
                break
            node.publish_frame(frame)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(rate)
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
    p = argparse.ArgumentParser(description="정책 추론 결과를 실제 로봇 전송 전 RViz로 미리보기 (open-loop)")
    p.add_argument("--dataset_root", required=True, help="관찰 프레임으로 쓸 LeRobotDataset 루트")
    p.add_argument("--episode", type=int, default=0, help="episode 인덱스")
    p.add_argument("--policy_path", required=True, help="정책 체크포인트 경로 또는 HF repo id")
    p.add_argument("--task", default=None, help="정책에 넘길 task 문자열 (기본: episode에 기록된 값)")
    p.add_argument("--device", default="cpu", help="cpu/cuda/mps (기본 cpu)")
    p.add_argument("--rate", type=float, default=0.1, help="RViz 퍼블리시 주기(초), 기본 10Hz")
    p.add_argument("--joint_state_topic", default="/joint_states")
    args = p.parse_args()

    root = pathlib.Path(args.dataset_root)
    if not root.exists():
        err(f"경로 없음: {root}")
        sys.exit(1)

    seq = run_inference_sequence(root, args.episode, args.policy_path, args.task, args.device)

    print()
    info(f"── 예측 action 정규화 범위 (참고용) ──────────")
    names = JOINT_NAMES + [GRIPPER_NAME]
    for i, name in enumerate(names):
        col = seq[:, i]
        lo, hi = (-100, 100) if name != GRIPPER_NAME else (0, 100)
        out = ((col < lo) | (col > hi)).sum()
        status = f"{C.GREEN}OK{C.RESET}" if out == 0 else f"{C.RED}범위초과({out}행){C.RESET}"
        print(f"  {name:>8}  min={col.min():>8.2f}  max={col.max():>8.2f}  {status}")

    run_rviz(seq, args.rate, args.joint_state_topic)


if __name__ == "__main__":
    main()
