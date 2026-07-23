#!/usr/bin/env python3
"""
piper_replay_player_rviz.py — 녹화된 episode(parquet)를 RViz + 카메라 영상으로 동시 재생

scripts/legacy_tools/piper_replay_viz.py(joint_states만 RViz로 publish)와
piper_replay_player.py(영상+joint 그래프만 표시, ROS2 없음)를 각각 따로 실행하면
서로 독립된 프로세스라 프레임 단위로 맞물리지 않음. 이 스크립트는 한 루프 안에서
"이번 프레임 joint publish + 이번 프레임 영상 imshow"를 같이 처리해서 RViz 3D
모델과 카메라 영상이 프레임 단위로 정확히 동기화되게 함.

사용법:
    # 터미널 1: RViz + robot_state_publisher 먼저 실행
    #   ros2 launch agx_arm_description display_piper.launch.py

    # 터미널 2: 이 스크립트 실행 (ROS2 source + conda activate 둘 다 필요)
    python scripts/tools/piper_replay_player_rviz.py \
        --dataset_root record_sample/local/piper_write_light_rs_10s_3eps_v5 \
        --episode 0

화면 구성은 piper_replay_player.py(asset/viwer.png)와 동일한 스타일 — 왼쪽에 카메라
영상을 위아래로 쌓고, 오른쪽에 다크 패널(frame/episode 카운터, Joint별 State/Action
표, 컨트롤 안내)을 붙여서 한 창(cv2 window)으로 표시함.

옵션:
    --dataset_root   LeRobotDataset 루트 경로 (parquet/videos가 들어있는 폴더)
    --episode        episode 인덱스 (기본 0)
    --column         RViz에 publish할 컬럼 (기본 action, observation.state도 가능;
                      패널에는 항상 action/observation.state 둘 다 표시됨)
    --rate           프레임 간 간격(초) — 기본은 meta/info.json의 fps로부터 자동 계산
                      (직접 지정하면 그 값 사용, RViz publish와 imshow 둘 다 이 간격을 씀)
    --loop           끝까지 재생 후 처음부터 반복
    --joint_state_topic   publish할 토픽 (기본 /joint_states)
    --video_key      표시할 video feature 이름, 반복 가능 (기본: info.json에서
                      dtype=video인 feature 전부 자동 탐색)
    --video_height   각 영상 표시 높이(px), 기본 300
    --panel_width    오른쪽 정보 패널 너비(px), 기본 360

주의:
    piper_replay_viz.py와 동일하게 lerobot_robot_piper의 PiperMotorsBus는 직접
    import하지 않음(생성자가 CAN 연결을 즉시 시도해서 하드웨어 없이 못 씀).
    calibration 표도 동일하게 그대로 복사해서 씀.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time

import av
import cv2
import numpy as np
import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata


class C:
    RESET = "\033[0m"; BOLD = "\033[1m"
    RED = "\033[91m"; GREEN = "\033[92m"; YELLOW = "\033[93m"; CYAN = "\033[96m"

def ok(m):   print(f"{C.GREEN}[OK]{C.RESET} {m}")
def warn(m): print(f"{C.YELLOW}[WARN]{C.RESET} {m}")
def err(m):  print(f"{C.RED}[ERROR]{C.RESET} {m}")
def info(m): print(f"{C.CYAN}[INFO]{C.RESET} {m}")


# ═══════════════════════════════════════════════
# 1. joint-space calibration (piper_replay_viz.py와 동일한 값)
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
    """정규화값(-100~100, gripper 0~100) -> raw(0.001 단위) -> 물리 단위(rad / m)."""
    min_, max_ = CALIBRATION_RAW[motor]

    if motor == GRIPPER_NAME:
        bounded = min(100.0, max(0.0, normalized_val))
        raw = (bounded / 100.0) * (max_ - min_) + min_
        return raw / 1000.0 / 1000.0  # meters

    bounded = min(100.0, max(-100.0, normalized_val))
    raw = ((bounded + 100) / 200) * (max_ - min_) + min_
    return math.radians(raw / 1000.0)


# ═══════════════════════════════════════════════
# 2. parquet / 영상 로드
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


def try_extract_joint_sequence(df: pd.DataFrame, column: str) -> np.ndarray | None:
    """패널 표시용 — extract_joint_sequence와 같지만 컬럼이 없거나 모양이 안 맞으면
    (프로그램을 죽이지 않고) None을 반환함."""
    if column not in df.columns:
        return None
    arr = np.array(df[column].tolist())
    if arr.ndim != 2 or arr.shape[1] != 7:
        return None
    return arr


def range_check(arr: np.ndarray, label: str) -> None:
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


def load_meta(dataset_root: pathlib.Path) -> LeRobotDatasetMetadata:
    return LeRobotDatasetMetadata(f"local/{dataset_root.name}", root=dataset_root)


def load_video_frames(
    meta: LeRobotDatasetMetadata, episode: int, video_keys: list[str] | None
) -> dict[str, list[np.ndarray] | None]:
    """episode의 카메라 영상을 전부 디코딩해서 {video_key: [frame, ...]}로 반환.
    video_keys가 None이면 info.json에서 dtype=video인 feature를 전부 자동으로 찾음.
    영상을 못 열면 값이 None — 화면에는 "video not found" placeholder로 표시됨
    (해당 카메라만 빠지는 게 아니라 레이아웃은 그대로 유지).

    cv2.VideoCapture(ffmpeg 내장 AV1 디코더)는 이 프로젝트의 mp4(AV1/libdav1d 인코딩)를
    seek할 때 'Missing Sequence Header'로 깨져서 프레임을 못 읽어옴 — 대신 PyAV(av)로
    처음부터 순차 디코딩해서 메모리에 전부 올려놓음(재생 순서가 항상 0..N-1 순차라
    seek이 필요 없음, episode 길이가 짧아서 메모리도 부담 없음)."""
    keys = video_keys or [k for k, v in meta.features.items() if v.get("dtype") == "video"]
    if not keys:
        warn("영상 feature 없음 — RViz만 재생")
        return {}

    videos: dict[str, list[np.ndarray] | None] = {}
    for key in keys:
        path = pathlib.Path(meta.root) / meta.get_video_file_path(episode, key)
        try:
            container = av.open(str(path))
            frames = [f.to_ndarray(format="bgr24") for f in container.decode(video=0)]
            container.close()
        except Exception as e:
            warn(f"영상을 못 엶: {path} ({e})")
            videos[key] = None
            continue
        if not frames:
            warn(f"영상에서 프레임을 못 읽음: {path}")
            videos[key] = None
            continue
        ok(f"영상 로드: {key} ({path.name}, {len(frames)} frame)")
        videos[key] = frames
    return videos


def resize_to_height(frame: np.ndarray, height: int) -> np.ndarray:
    scale = height / frame.shape[0]
    return cv2.resize(frame, (int(frame.shape[1] * scale), height))


def placeholder_frame(height: int, width: int, text: str) -> np.ndarray:
    frame = np.full((height, width, 3), (20, 20, 20), dtype=np.uint8)
    draw_text(frame, text, (18, height // 2), color=(80, 80, 255))
    return frame


# ═══════════════════════════════════════════════
# 2.5. 화면 구성 — piper_replay_player.py(asset/viwer.png)와 동일한 스타일:
#      영상 스트립(왼쪽) + 다크 정보 패널(오른쪽)을 hconcat해서 한 창으로 표시
# ═══════════════════════════════════════════════
def draw_text(img, text, pos, scale=0.55, color=(255, 255, 255), thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def build_panel(
    *,
    width: int,
    height: int,
    idx: int,
    total: int,
    joint_names: list[str],
    state: np.ndarray | None,
    action: np.ndarray | None,
    paused: bool,
    speed: float,
    topic: str,
    loop: bool,
) -> np.ndarray:
    panel = np.full((height, width, 3), (28, 30, 34), dtype=np.uint8)

    y = 28
    draw_text(panel, "Piper replay + RViz", (14, y), scale=0.62, color=(255, 255, 255), thickness=2)
    y += 30
    draw_text(panel, f"frame {idx + 1}/{total}", (14, y))
    y += 22
    draw_text(panel, f"{'PAUSED' if paused else f'{speed:.2f}x'}  {'loop' if loop else 'once'}", (14, y), color=(190, 220, 255))
    y += 28

    x_name, x_state, x_action = 14, 170, 270
    draw_text(panel, "Joint", (x_name, y), color=(160, 200, 170))
    draw_text(panel, "State", (x_state, y), color=(160, 200, 170))
    draw_text(panel, "Action", (x_action, y), color=(160, 200, 170))
    y += 22

    for i, name in enumerate(joint_names):
        draw_text(panel, name[:16], (x_name, y))
        draw_text(panel, f"{state[i]: >7.2f}" if state is not None else "-", (x_state, y))
        draw_text(panel, f"{action[i]: >7.2f}" if action is not None else "-", (x_action, y))
        y += 21

    y += 10
    draw_text(panel, f"topic: {topic}", (14, y), scale=0.45, color=(150, 210, 255))

    y = height - 100
    draw_text(panel, "- CONTROLS -", (14, y), scale=0.45, color=(220, 220, 220))
    y += 18
    draw_text(panel, "space: Pause/Resume | q, esc: Quit", (14, y), scale=0.43, color=(185, 185, 185))
    y += 16
    draw_text(panel, ", . or left/right: Seek Frame | a, d: Seek 10", (14, y), scale=0.43, color=(185, 185, 185))
    y += 16
    draw_text(panel, "+/-: adjust speed by 0.25x", (14, y), scale=0.43, color=(185, 185, 185))

    return panel


def build_canvas(
    *,
    idx: int,
    total: int,
    video_frames: dict[str, list[np.ndarray] | None],
    video_height: int,
    panel_width: int,
    joint_names: list[str],
    state_seq: np.ndarray | None,
    action_seq: np.ndarray | None,
    paused: bool,
    speed: float,
    topic: str,
    loop: bool,
) -> np.ndarray:
    strips = []
    for key, frames in video_frames.items():
        if frames is None or idx >= len(frames):
            frame = placeholder_frame(video_height, 424, f"video not found: {key}")
        else:
            frame = resize_to_height(frames[idx], video_height)
        frame = frame.copy()
        draw_text(frame, key, (12, 24), color=(40, 230, 255), thickness=2)
        strips.append(frame)

    video_strip = cv2.vconcat(strips) if strips else placeholder_frame(video_height, 424, "no video")

    panel = build_panel(
        width=panel_width,
        height=video_strip.shape[0],
        idx=idx,
        total=total,
        joint_names=joint_names,
        state=state_seq[idx] if state_seq is not None else None,
        action=action_seq[idx] if action_seq is not None else None,
        paused=paused,
        speed=speed,
        topic=topic,
        loop=loop,
    )
    return cv2.hconcat([video_strip, panel])


# ═══════════════════════════════════════════════
# 3. RViz publish + 영상 imshow (같은 루프, 같은 프레임 인덱스)
# ═══════════════════════════════════════════════
WINDOW_NAME = "Piper replay + RViz (synchronized)"


def run(
    seq: np.ndarray,
    rate: float,
    loop: bool,
    topic: str,
    video_frames: dict[str, list[np.ndarray] | None],
    video_height: int,
    panel_width: int,
    joint_names: list[str],
    state_seq: np.ndarray | None,
    action_seq: np.ndarray | None,
):
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
            super().__init__("piper_replay_player_rviz")
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

    total = len(seq)
    if video_frames:
        known_lengths = [len(frames) for frames in video_frames.values() if frames is not None]
        if known_lengths:
            video_total = min(known_lengths)
            if video_total != total:
                warn(f"parquet frame({total})과 영상 frame({video_total})이 안 맞음 — 짧은 쪽 기준으로 재생")
            total = min(total, video_total)

    base_fps = 1.0 / rate

    print()
    info(f"토픽: {topic} (JointState) — robot_state_publisher가 이 토픽을 구독해야 RViz가 움직임")
    if video_frames:
        info(f"영상: {list(video_frames.keys())} — 같은 frame index로 동기화 표시")
    info(f"{total} frame, 기본 {base_fps:.2f}fps, {'반복' if loop else '1회'} 재생")
    info("컨트롤: space 정지/재생 | ,/. 또는 ←/→ 1프레임 | a/d 10프레임 | +/- 속도 | q/esc 종료\n")

    idx = 0
    paused = False
    speed = 1.0
    last_tick = time.monotonic()

    try:
        while rclpy.ok():
            node.publish_frame(seq[idx])
            rclpy.spin_once(node, timeout_sec=0.0)

            canvas = build_canvas(
                idx=idx,
                total=total,
                video_frames=video_frames,
                video_height=video_height,
                panel_width=panel_width,
                joint_names=joint_names,
                state_seq=state_seq,
                action_seq=action_seq,
                paused=paused,
                speed=speed,
                topic=topic,
                loop=loop,
            )
            cv2.imshow(WINDOW_NAME, canvas)

            wait_ms = 30 if paused else max(1, int(1000 / (base_fps * speed)))
            key_pressed = cv2.waitKey(wait_ms) & 0xFF

            if key_pressed in (ord("q"), 27):
                break
            elif key_pressed in (ord("+"), ord("=")):
                speed = min(4.0, speed + 0.25)
            elif key_pressed == ord("-"):
                speed = max(0.25, speed - 0.25)
            elif key_pressed == ord(" "):
                paused = not paused
            elif key_pressed in (81, ord(",")):
                idx = max(0, idx - 1)
                paused = True
            elif key_pressed in (83, ord(".")):
                idx = min(total - 1, idx + 1)
                paused = True
            elif key_pressed == ord("a"):
                idx = max(0, idx - 10)
                paused = True
            elif key_pressed == ord("d"):
                idx = min(total - 1, idx + 10)
                paused = True

            if not paused:
                now = time.monotonic()
                if now - last_tick >= 1.0 / max(1e-6, base_fps * speed):
                    last_tick = now
                    if idx < total - 1:
                        idx += 1
                    elif loop:
                        idx = 0
                    # loop=False and idx==total-1: 마지막 프레임에서 멈춰서 자세 유지
    except KeyboardInterrupt:
        info("종료")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


# ═══════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description="녹화된 episode를 RViz + 카메라 영상으로 동시 재생")
    p.add_argument("--dataset_root", required=True, help="LeRobotDataset 루트 경로")
    p.add_argument("--episode", type=int, default=0, help="episode 인덱스")
    p.add_argument("--column", default="action", help="RViz에 publish할 컬럼 (action / observation.state)")
    p.add_argument("--rate", type=float, default=None,
                   help="프레임 간 간격(초) — 기본은 meta/info.json의 fps로부터 자동 계산")
    p.add_argument("--loop", action="store_true", help="끝까지 재생 후 처음부터 반복")
    p.add_argument("--joint_state_topic", default="/joint_states",
                   help="publish할 JointState 토픽 (robot_state_publisher가 구독하는 토픽과 일치해야 함)")
    p.add_argument("--video_key", action="append", default=None,
                   help="표시할 video feature 이름, 반복 가능 (기본: 전부 자동 탐색)")
    p.add_argument("--video_height", type=int, default=300, help="영상 표시 높이(px)")
    p.add_argument("--panel_width", type=int, default=360, help="오른쪽 정보 패널 너비(px)")
    args = p.parse_args()

    root = pathlib.Path(args.dataset_root)
    if not root.exists():
        err(f"경로 없음: {root}")
        sys.exit(1)

    df = load_episode(root, args.episode)
    seq = extract_joint_sequence(df, args.column)
    range_check(seq, args.column)

    # 패널에는 항상 action/observation.state 둘 다 보여줌 (있는 쪽만)
    action_seq = try_extract_joint_sequence(df, "action")
    state_seq = try_extract_joint_sequence(df, "observation.state")
    joint_names = JOINT_NAMES + [GRIPPER_NAME]

    meta = load_meta(root)
    rate = args.rate if args.rate is not None else 1.0 / float(meta.fps or 30)
    video_frames = load_video_frames(meta, args.episode, args.video_key)

    run(
        seq, rate, args.loop, args.joint_state_topic, video_frames, args.video_height,
        args.panel_width, joint_names, state_seq, action_seq,
    )


if __name__ == "__main__":
    main()
