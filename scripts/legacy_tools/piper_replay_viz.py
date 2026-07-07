#!/usr/bin/env python3
"""
piper_replay_viz.py — 녹화된 episode(parquet)를 RViz에 EEF 궤적으로 시각화

사용법:
    # 터미널 1: RViz 먼저 실행
    ros2 launch piper_description display_piper.launch.py

    # 터미널 2: 이 스크립트 실행
    python piper_replay_viz.py --dataset_root /path/to/dataset --episode 0

옵션:
    --dataset_root   LeRobotDataset 루트 경로 (parquet들이 들어있는 폴더)
    --episode        episode 인덱스 (기본 0)
    --column         궤적으로 쓸 컬럼 이름 (기본 action, observation.state도 가능)
    --compare        action과 observation.state를 동시에 그려서 비교 (둘 다 그림)
    --loop           1회성이 아니라 계속 반복 퍼블리시 (기본 동작, Ctrl+C로 종료)
    --rate           퍼블리시 주기 초 (기본 1.0)

RViz에서 확인할 것:
    Add → By topic → /recorded_trajectory (Marker)  → action 궤적 (초록 선)
    Add → By topic → /recorded_trajectory_state (Marker) → state 궤적 (파란 선, --compare 시)
    Add → By topic → /recorded_waypoints (MarkerArray) → 시작(파랑)/끝(주황) 구체
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

# ───────────────────────────────────────────────
# 안전 범위 (raw SDK 정수 단위) — 사전 검사용
# ───────────────────────────────────────────────
SAFE_RANGE = {
    "x":        (100_000,  400_000),
    "y":       (-200_000,  200_000),
    "z":         (50_000,  350_000),
    "rx":      (-180_000,  180_000),
    "ry":      (-180_000,  180_000),
    "rz":      (-180_000,  180_000),
    "gripper":       (0,    70_000),
}
MAX_DELTA_PER_STEP = 30_000


class C:
    RESET = "\033[0m"; BOLD = "\033[1m"
    RED = "\033[91m"; GREEN = "\033[92m"; YELLOW = "\033[93m"; CYAN = "\033[96m"

def ok(m):   print(f"{C.GREEN}[OK]{C.RESET} {m}")
def warn(m): print(f"{C.YELLOW}[WARN]{C.RESET} {m}")
def err(m):  print(f"{C.RED}[ERROR]{C.RESET} {m}")
def info(m): print(f"{C.CYAN}[INFO]{C.RESET} {m}")


# ═══════════════════════════════════════════════
# 1. parquet 로드
# ═══════════════════════════════════════════════
def load_episode(dataset_root: pathlib.Path, episode: int) -> pd.DataFrame:
    """episode 인덱스에 해당하는 parquet 파일을 찾아 로드"""
    candidates = sorted(dataset_root.glob("data/**/*.parquet"))
    if not candidates:
        err(f"parquet 파일 없음: {dataset_root}/data/**/*.parquet")
        sys.exit(1)

    info(f"전체 {len(candidates)}개 parquet 발견")

    # 파일명에 episode 번호가 들어간 패턴 우선 탐색 (예: episode_000000.parquet)
    target = None
    for p in candidates:
        if f"{episode:06d}" in p.stem or f"episode_{episode}" in p.stem:
            target = p
            break

    if target is None:
        # 못 찾으면 정렬된 순서로 episode번째 파일 사용
        if episode < len(candidates):
            target = candidates[episode]
            warn(f"파일명으로 episode {episode}를 못 찾음 — 정렬 순서 기준 {target.name} 사용")
        else:
            err(f"episode {episode} 파일 없음 (전체 {len(candidates)}개)")
            sys.exit(1)

    info(f"로드: {target}")
    df = pd.read_parquet(target)
    info(f"{len(df)} frame 로드 완료, 컬럼: {list(df.columns)[:8]}...")
    return df


# ═══════════════════════════════════════════════
# 2. 컬럼 → EEF 좌표 array 추출
# ═══════════════════════════════════════════════
def extract_eef(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df.columns:
        err(f"'{column}' 컬럼 없음. 사용 가능한 컬럼: {list(df.columns)}")
        sys.exit(1)

    series = df[column]
    # LeRobotDataset은 보통 list[float] 형태로 저장됨
    arr = np.array(series.tolist())

    if arr.ndim != 2 or arr.shape[1] < 3:
        err(f"'{column}' 형태가 예상과 다름: shape={arr.shape}")
        sys.exit(1)

    return arr


# ═══════════════════════════════════════════════
# 3. 사전 안전 검사 (참고용 — 실제 arm에 보내는 게 아니므로 차단은 안 함)
# ═══════════════════════════════════════════════
def safety_report(arr: np.ndarray, label: str) -> bool:
    print()
    info(f"── {label} 안전 범위 점검 (참고용) ──────────")
    axis_names = list(SAFE_RANGE.keys())
    all_ok = True

    print(f"  {'축':>8}  {'최솟값':>12}  {'최댓값':>12}  {'안전범위':>25}  결과")
    print(f"  {'─'*72}")
    for i, (axis, (lo, hi)) in enumerate(SAFE_RANGE.items()):
        if i >= arr.shape[1]:
            break
        col = arr[:, i]
        vmin, vmax = col.min(), col.max()
        out = ((col < lo) | (col > hi)).sum()
        status = f"{C.GREEN}OK{C.RESET}" if out == 0 else f"{C.RED}범위초과({out}행){C.RESET}"
        print(f"  {axis:>8}  {vmin:>12.0f}  {vmax:>12.0f}  [{lo:>10},{hi:>10}]  {status}")
        if out:
            all_ok = False

    if len(arr) > 1:
        deltas = np.abs(np.diff(arr, axis=0)).max(axis=1)
        bad = np.where(deltas > MAX_DELTA_PER_STEP)[0]
        if len(bad) == 0:
            ok(f"모든 step delta < {MAX_DELTA_PER_STEP}")
        else:
            warn(f"{len(bad)}개 스텝에서 delta > {MAX_DELTA_PER_STEP} (frame {bad[:5].tolist()}...)")
            all_ok = False

    if all_ok:
        ok(f"{label} 궤적 정상 범위")
    else:
        warn(f"{label} 궤적 일부 이상 — RViz에서 빨간 선으로 표시됨")

    return all_ok


# ═══════════════════════════════════════════════
# 4. RViz 퍼블리시
# ═══════════════════════════════════════════════
def run_rviz(action_arr: np.ndarray,
             state_arr: np.ndarray | None,
             action_safe: bool,
             state_safe: bool,
             rate: float):
    try:
        import rclpy
        from rclpy.node import Node
        from visualization_msgs.msg import Marker, MarkerArray
        from geometry_msgs.msg import Point
        from std_msgs.msg import ColorRGBA
    except ImportError:
        err("rclpy 없음 — ROS2 환경에서 실행해야 함")
        err("source /opt/ros/humble/setup.bash 후 재실행")
        sys.exit(1)

    class ReplayVizNode(Node):
        def __init__(self):
            super().__init__("piper_replay_viz")
            self.pub_action = self.create_publisher(Marker, "/recorded_trajectory", 10)
            self.pub_state  = self.create_publisher(Marker, "/recorded_trajectory_state", 10)
            self.pub_points = self.create_publisher(MarkerArray, "/recorded_waypoints", 10)

        def _line_marker(self, arr, ns, mid, safe):
            m = Marker()
            m.header.frame_id = "base_link"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = ns
            m.id = mid
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.005
            if ns == "action":
                m.color = ColorRGBA(r=0.1, g=1.0, b=0.1, a=1.0) if safe \
                    else ColorRGBA(r=1.0, g=0.1, b=0.1, a=1.0)
            else:
                m.color = ColorRGBA(r=0.2, g=0.5, b=1.0, a=0.8)
            for a in arr:
                p = Point()
                p.x = float(a[0]) / 1_000_000.0
                p.y = float(a[1]) / 1_000_000.0
                p.z = float(a[2]) / 1_000_000.0
                m.points.append(p)
            return m

        def _waypoint_markers(self, arr):
            ma = MarkerArray()
            for idx, (label, color, a) in enumerate([
                ("start", ColorRGBA(r=0.0, g=0.5, b=1.0, a=1.0), arr[0]),
                ("end",   ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0), arr[-1]),
            ]):
                sm = Marker()
                sm.header.frame_id = "base_link"
                sm.header.stamp = self.get_clock().now().to_msg()
                sm.ns = label
                sm.id = idx
                sm.type = Marker.SPHERE
                sm.action = Marker.ADD
                sm.pose.position.x = float(a[0]) / 1_000_000.0
                sm.pose.position.y = float(a[1]) / 1_000_000.0
                sm.pose.position.z = float(a[2]) / 1_000_000.0
                sm.scale.x = sm.scale.y = sm.scale.z = 0.025
                sm.color = color
                ma.markers.append(sm)
            return ma

        def publish_all(self):
            self.pub_action.publish(self._line_marker(action_arr, "action", 0, action_safe))
            if state_arr is not None:
                self.pub_state.publish(self._line_marker(state_arr, "state", 1, state_safe))
            self.pub_points.publish(self._waypoint_markers(action_arr))

    rclpy.init()
    node = ReplayVizNode()

    print()
    info("RViz 설정:")
    info("  Add → By topic → /recorded_trajectory       (action 궤적, 초록/빨강)")
    if state_arr is not None:
        info("  Add → By topic → /recorded_trajectory_state  (state 궤적, 파랑)")
    info("  Add → By topic → /recorded_waypoints         (시작=파랑 구체, 끝=주황 구체)")
    info("  Fixed Frame은 'base_link'로 설정")
    info(f"\n{len(action_arr)} frame 궤적을 {rate}초 간격으로 반복 퍼블리시 — Ctrl+C로 종료\n")

    try:
        while rclpy.ok():
            node.publish_all()
            rclpy.spin_once(node, timeout_sec=rate)
            time.sleep(rate)
    except KeyboardInterrupt:
        info("종료")
    finally:
        node.destroy_node()
        rclpy.shutdown()


# ═══════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description="녹화된 episode를 RViz에 시각화")
    p.add_argument("--dataset_root", required=True, help="LeRobotDataset 루트 경로")
    p.add_argument("--episode", type=int, default=0, help="episode 인덱스")
    p.add_argument("--column", default="action", help="궤적으로 쓸 컬럼 (action / observation.state)")
    p.add_argument("--compare", action="store_true",
                   help="action과 observation.state를 동시에 그려서 비교")
    p.add_argument("--rate", type=float, default=1.0, help="퍼블리시 주기(초)")
    args = p.parse_args()

    root = pathlib.Path(args.dataset_root)
    if not root.exists():
        err(f"경로 없음: {root}")
        sys.exit(1)

    df = load_episode(root, args.episode)

    action_arr = extract_eef(df, "action" if args.compare else args.column)
    action_safe = safety_report(action_arr, "action")

    state_arr = None
    state_safe = True
    if args.compare:
        if "observation.state" in df.columns:
            state_arr = extract_eef(df, "observation.state")
            state_safe = safety_report(state_arr, "observation.state")
        else:
            warn("observation.state 컬럼 없음 — action만 표시")

    run_rviz(action_arr, state_arr, action_safe, state_safe, args.rate)


if __name__ == "__main__":
    main()
