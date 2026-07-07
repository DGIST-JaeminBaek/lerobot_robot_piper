#!/usr/bin/env python3
"""
piper_validate.py — UGRP PiPER 검증 파이프라인 통합 CLI

사용법:
    python piper_validate.py --step <단계> [옵션]

단계 목록:
    data_check      : 데이터셋 유효성 검사 (parquet, all-zero, 범위)
    calc_range      : 데이터셋에서 ACTION_MIN/MAX 자동 계산
    infer_dry       : SmolVLA 추론 (로봇 없이, action 저장)
    rviz_preview    : RViz로 추론 궤적 시각화
    replay_dry      : piper-replay dry-run (로봇 없이)
    replay_real     : piper-replay 실제 arm
    infer_real      : SmolVLA 실제 arm 추론
    full            : data_check → calc_range → infer_dry → rviz_preview → replay_dry 순서 자동 실행

예시:
    python piper_validate.py --step data_check --dataset_root /home/ugrp308/Group43/datasets/piper-smolvla
    python piper_validate.py --step infer_dry --pretrained_path outputs/piper-smolvla/checkpoints/last/pretrained_model
    python piper_validate.py --step rviz_preview --actions_file predicted_actions.json
    python piper_validate.py --step full --dataset_root /home/ugrp308/Group43/datasets/piper-smolvla --pretrained_path outputs/piper-smolvla/checkpoints/last/pretrained_model
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Optional

# ───────────────────────────────────────────────
# 컬러 출력 헬퍼
# ───────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"

def ok(msg):   print(f"{C.GREEN}[OK]{C.RESET} {msg}")
def warn(msg): print(f"{C.YELLOW}[WARN]{C.RESET} {msg}")
def err(msg):  print(f"{C.RED}[ERROR]{C.RESET} {msg}")
def info(msg): print(f"{C.CYAN}[INFO]{C.RESET} {msg}")
def header(msg):
    print(f"\n{C.BOLD}{'='*60}{C.RESET}")
    print(f"{C.BOLD}  {msg}{C.RESET}")
    print(f"{C.BOLD}{'='*60}{C.RESET}\n")

def confirm(prompt: str) -> bool:
    """사용자에게 y/n 확인"""
    ans = input(f"{C.YELLOW}[확인]{C.RESET} {prompt} (y/n): ").strip().lower()
    return ans == "y"

# ───────────────────────────────────────────────
# PiPER EEF 안전 범위 (raw SDK 정수 단위)
# EXPERIMENT.md 기준값
# ───────────────────────────────────────────────
SAFE_RANGE = {
    "x":       (100_000,  400_000),
    "y":      (-200_000,  200_000),
    "z":        (50_000,  350_000),
    "rx":     (-180_000,  180_000),
    "ry":     (-180_000,  180_000),
    "rz":     (-180_000,  180_000),
    "gripper":      (0,   70_000),
}
# 한 스텝 최대 이동량 (급격한 동작 감지)
MAX_DELTA_PER_STEP = 30_000


# ═══════════════════════════════════════════════
# STEP 1: data_check — 데이터셋 유효성 검사
# ═══════════════════════════════════════════════
def step_data_check(args):
    header("STEP 1: 데이터셋 유효성 검사")

    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        err("pandas / numpy 없음: pip install pandas numpy")
        sys.exit(1)

    root = pathlib.Path(args.dataset_root)
    if not root.exists():
        err(f"데이터셋 경로 없음: {root}")
        sys.exit(1)

    parquets = sorted(root.glob("data/**/*.parquet"))
    if not parquets:
        err(f"parquet 파일 없음: {root}/data/**/*.parquet")
        sys.exit(1)

    info(f"parquet 파일 {len(parquets)}개 발견")

    total_rows = 0
    total_zero = 0
    range_violations = []
    delta_violations = []

    for p in parquets:
        df = pd.read_parquet(p)
        total_rows += len(df)

        # ── all-zero 검사 ──────────────────────────
        if "observation.state" in df.columns:
            states = np.array(df["observation.state"].tolist())
            zero_mask = (states == 0).all(axis=1)
            n_zero = zero_mask.sum()
            total_zero += n_zero
            if n_zero > 0:
                warn(f"{p.name}: all-zero state {n_zero}행")

        # ── action 범위 검사 ───────────────────────
        if "action" in df.columns:
            actions = np.array(df["action"].tolist())
            axis_names = list(SAFE_RANGE.keys())

            for i, (axis, (lo, hi)) in enumerate(SAFE_RANGE.items()):
                if i >= actions.shape[1]:
                    break
                col = actions[:, i]
                out = ((col < lo) | (col > hi)).sum()
                if out > 0:
                    range_violations.append(
                        f"{p.name} / {axis}: {out}행이 범위 [{lo}, {hi}] 초과"
                    )

            # ── 급격한 delta 검사 ──────────────────
            deltas = np.abs(np.diff(actions, axis=0))
            big = (deltas > MAX_DELTA_PER_STEP).any(axis=1)
            n_big = big.sum()
            if n_big > 0:
                delta_violations.append(
                    f"{p.name}: {n_big}개 스텝에서 delta > {MAX_DELTA_PER_STEP}"
                )

        # ── frame_index 연속성 ─────────────────────
        if "frame_index" in df.columns:
            idx = df["frame_index"].values
            gaps = (idx[1:] - idx[:-1]) != 1
            if gaps.any():
                warn(f"{p.name}: frame_index 불연속 {gaps.sum()}곳")

        # ── timestamp 간격 ─────────────────────────
        if "timestamp" in df.columns:
            ts = df["timestamp"].values.astype(float)
            diffs = ts[1:] - ts[:-1]
            mean_dt = diffs.mean()
            std_dt  = diffs.std()
            if std_dt > mean_dt * 0.3:
                warn(f"{p.name}: timestamp 간격 불균일 (mean={mean_dt:.3f}s, std={std_dt:.3f}s)")

    # ── 결과 출력 ──────────────────────────────────
    print()
    info(f"총 {total_rows}행 검사 완료")

    if total_zero == 0:
        ok("all-zero state 없음")
    else:
        err(f"all-zero state 합계: {total_zero}행")

    if not range_violations:
        ok("action 범위 이상 없음")
    else:
        for v in range_violations:
            err(f"범위 초과: {v}")

    if not delta_violations:
        ok("급격한 delta 없음")
    else:
        for v in delta_violations:
            warn(f"delta 경고: {v}")

    passed = (total_zero == 0) and (not range_violations)
    if passed:
        ok("데이터셋 유효성 검사 통과")
    else:
        err("데이터셋 유효성 검사 실패 — 실제 arm 실험 전 데이터를 점검해")

    return passed


# ═══════════════════════════════════════════════
# STEP 2: calc_range — ACTION_MIN/MAX 자동 계산
# ═══════════════════════════════════════════════
def step_calc_range(args):
    header("STEP 2: ACTION_MIN/MAX 자동 계산")

    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        err("pandas / numpy 없음: pip install pandas numpy")
        sys.exit(1)

    root = pathlib.Path(args.dataset_root)
    parquets = sorted(root.glob("data/**/*.parquet"))
    if not parquets:
        err(f"parquet 파일 없음: {root}")
        sys.exit(1)

    all_actions = []
    for p in parquets:
        df = pd.read_parquet(p)
        if "action" in df.columns:
            all_actions.extend(df["action"].tolist())

    if not all_actions:
        err("action 컬럼 없음")
        sys.exit(1)

    actions_np = np.array(all_actions)
    margin = 0.05  # 5% 여유

    raw_min = actions_np.min(axis=0)
    raw_max = actions_np.max(axis=0)
    span = raw_max - raw_min
    action_min = (raw_min - span * margin).tolist()
    action_max = (raw_max + span * margin).tolist()

    result = {
        "ACTION_MIN": action_min,
        "ACTION_MAX": action_max,
        "raw_min":    raw_min.tolist(),
        "raw_max":    raw_max.tolist(),
        "margin":     margin,
        "n_episodes": len(parquets),
        "n_frames":   len(all_actions),
    }

    out_path = pathlib.Path(args.range_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    axis_names = list(SAFE_RANGE.keys())
    print()
    info(f"에피소드 {len(parquets)}개 / 프레임 {len(all_actions)}개 기반")
    for i, name in enumerate(axis_names):
        if i >= len(action_min):
            break
        print(f"  {name:>8}: [{action_min[i]:>10.0f}, {action_max[i]:>10.0f}]")

    ok(f"ACTION_MIN/MAX 저장 완료: {out_path}")
    info(f"smolvla_inference.py에 이 파일을 로드하도록 수정하거나,\n"
         f"  ACTION_MIN = {[round(v) for v in action_min]}\n"
         f"  ACTION_MAX = {[round(v) for v in action_max]}\n"
         f"  로 하드코딩을 교체해.")

    return True


# ═══════════════════════════════════════════════
# STEP 3: infer_dry — SmolVLA 추론 (로봇 없이)
# ═══════════════════════════════════════════════
def step_infer_dry(args):
    header("STEP 3: SmolVLA 추론 (use_devices=false)")

    if not args.pretrained_path:
        err("--pretrained_path 필요")
        sys.exit(1)

    cmd = [
        "smolvla-inference",
        f"--pretrained_path={args.pretrained_path}",
        "--use_devices=false",
        f"--max_steps={args.max_steps}",
        f"--task={args.task}",
    ]

    info(f"실행: {' '.join(cmd)}")
    info("추론 결과는 predicted_actions.json에 저장됩니다.")
    info("smolvla_inference.py에 --save_actions 옵션이 없으면 추가 필요.\n")

    # smolvla-inference가 --save_actions를 지원하지 않을 경우를 대비해
    # 직접 추론을 실행하고 stdout을 캡처하는 대신,
    # subprocess로 실행하고 종료 코드만 확인
    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            err(f"smolvla-inference 실패 (returncode={result.returncode})")
            info("smolvla_inference.py에 --save_actions=true 옵션이 있는지 확인해")
            return False
    except FileNotFoundError:
        err("smolvla-inference CLI 없음. pip install -e . 했는지 확인해")
        return False

    # predicted_actions.json 존재 확인
    if pathlib.Path("predicted_actions.json").exists():
        ok("predicted_actions.json 생성 확인")
        return True
    else:
        warn("predicted_actions.json 없음 — smolvla_inference.py에 save_actions 로직 추가 필요")
        warn("아래 코드를 smolvla_inference.py 추론 루프 끝에 추가:\n"
             "  with open('predicted_actions.json','w') as f:\n"
             "      json.dump({'actions': action_log}, f)")
        return False


# ═══════════════════════════════════════════════
# STEP 4: rviz_preview — 궤적 RViz 시각화 + 안전 검사
# ═══════════════════════════════════════════════
def step_rviz_preview(args):
    header("STEP 4: RViz 궤적 시각화 및 안전 검사")

    actions_file = pathlib.Path(args.actions_file)
    if not actions_file.exists():
        err(f"actions 파일 없음: {actions_file}")
        err("먼저 --step infer_dry 를 실행해")
        sys.exit(1)

    with open(actions_file) as f:
        data = json.load(f)
    actions = data.get("actions", [])

    if not actions:
        err("actions 리스트가 비어 있음")
        sys.exit(1)

    info(f"총 {len(actions)} steps 로드")

    # ── 사전 안전 검사 (RViz 띄우기 전) ──────────
    import numpy as np
    actions_np = np.array(actions)
    axis_names = list(SAFE_RANGE.keys())

    print()
    info("── 안전 범위 검사 ──────────────────────────")
    safety_ok = True

    for i, (axis, (lo, hi)) in enumerate(SAFE_RANGE.items()):
        if i >= actions_np.shape[1]:
            break
        col = actions_np[:, i]
        vmin, vmax = col.min(), col.max()
        out_lo = (col < lo).sum()
        out_hi = (col > hi).sum()
        status = "OK" if (out_lo == 0 and out_hi == 0) else "FAIL"
        color = C.GREEN if status == "OK" else C.RED
        print(f"  {color}{axis:>8}: [{vmin:>10.0f}, {vmax:>10.0f}]  안전범위:[{lo},{hi}]  {status}{C.RESET}")
        if status == "FAIL":
            safety_ok = False

    print()
    info("── 급격한 delta 검사 ───────────────────────")
    if len(actions_np) > 1:
        deltas = np.abs(np.diff(actions_np, axis=0))
        max_deltas = deltas.max(axis=1)
        bad_steps = np.where(max_deltas > MAX_DELTA_PER_STEP)[0]
        if len(bad_steps) == 0:
            ok(f"모든 스텝 delta < {MAX_DELTA_PER_STEP}")
        else:
            for s in bad_steps:
                warn(f"step {s}→{s+1}: max_delta={max_deltas[s]:.0f} (기준: {MAX_DELTA_PER_STEP})")
            safety_ok = False

    print()
    if safety_ok:
        ok("사전 안전 검사 통과 — RViz 시각화 시작")
    else:
        err("사전 안전 검사 실패 — RViz는 띄우지만 실제 arm 연결 금지")

    # ── RViz 퍼블리시 ─────────────────────────────
    try:
        import rclpy
        from rclpy.node import Node
        from visualization_msgs.msg import Marker, MarkerArray
        from geometry_msgs.msg import Point
        from std_msgs.msg import ColorRGBA
    except ImportError:
        err("rclpy 없음 — ROS2 환경에서 실행하고 있는지 확인해")
        err("source /opt/ros/humble/setup.bash")
        sys.exit(1)

    class PreviewNode(Node):
        def __init__(self):
            super().__init__("piper_trajectory_preview")
            self.pub_line   = self.create_publisher(Marker,      "/preview_trajectory", 10)
            self.pub_points = self.create_publisher(MarkerArray, "/preview_waypoints",  10)

        def publish_trajectory(self, acts, safe):
            # 궤적 선 (안전 여부에 따라 색상)
            m = Marker()
            m.header.frame_id = "base_link"
            m.header.stamp    = self.get_clock().now().to_msg()
            m.ns   = "predicted_traj"
            m.id   = 0
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.005
            # 안전하면 초록, 위험하면 빨강
            m.color = ColorRGBA(
                r=0.2 if safe else 1.0,
                g=1.0 if safe else 0.2,
                b=0.2,
                a=1.0
            )
            for a in acts:
                p = Point()
                p.x = a[0] / 1_000_000.0
                p.y = a[1] / 1_000_000.0
                p.z = a[2] / 1_000_000.0
                m.points.append(p)
            self.pub_line.publish(m)

            # 시작/끝 마커
            ma = MarkerArray()
            for idx, (label, color) in enumerate([
                ("start", ColorRGBA(r=0.0, g=0.5, b=1.0, a=1.0)),
                ("end",   ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0)),
            ]):
                a = acts[0] if label == "start" else acts[-1]
                sm = Marker()
                sm.header.frame_id = "base_link"
                sm.header.stamp    = self.get_clock().now().to_msg()
                sm.ns   = label
                sm.id   = idx
                sm.type = Marker.SPHERE
                sm.action = Marker.ADD
                sm.pose.position.x = a[0] / 1_000_000.0
                sm.pose.position.y = a[1] / 1_000_000.0
                sm.pose.position.z = a[2] / 1_000_000.0
                sm.scale.x = sm.scale.y = sm.scale.z = 0.02
                sm.color = color
                ma.markers.append(sm)
            self.pub_points.publish(ma)

            self.get_logger().info(
                f"궤적 퍼블리시 완료: {len(acts)} steps | "
                f"X[{min(a[0] for a in acts):.0f}~{max(a[0] for a in acts):.0f}] "
                f"Z[{min(a[2] for a in acts):.0f}~{max(a[2] for a in acts):.0f}]"
            )

    rclpy.init()
    node = PreviewNode()

    info("RViz에서 /preview_trajectory (Marker) 토픽을 추가해서 확인해")
    info("  파란 구체 = 시작점 / 주황 구체 = 끝점")
    color_hint = "초록 선 = 안전" if safety_ok else "빨간 선 = 위험"
    info(f"  {color_hint}")
    info("Ctrl+C로 종료\n")

    # 1초마다 퍼블리시 (RViz가 늦게 뜨더라도 볼 수 있게)
    try:
        while rclpy.ok():
            node.publish_trajectory(actions, safety_ok)
            rclpy.spin_once(node, timeout_sec=1.0)
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

    return safety_ok


# ═══════════════════════════════════════════════
# STEP 5: replay_dry — piper-replay dry-run
# ═══════════════════════════════════════════════
def step_replay_dry(args):
    header("STEP 5: piper-replay dry-run (use_devices=false)")

    cmd = [
        "piper-replay",
        f"--dataset_repo_id={args.dataset_repo_id}",
        f"--dataset_root={args.dataset_root}",
        f"--episode={args.episode}",
        "--use_devices=false",
        f"--start_frame={args.start_frame}",
        f"--max_steps={args.max_steps}",
        f"--replay_fps={args.replay_fps}",
    ]

    info(f"실행: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=False)
    except FileNotFoundError:
        err("piper-replay CLI 없음. pip install -e . 했는지 확인해")
        return False

    if result.returncode == 0:
        ok("piper-replay dry-run 통과")
        return True
    else:
        err(f"piper-replay dry-run 실패 (returncode={result.returncode})")
        return False


# ═══════════════════════════════════════════════
# STEP 6: replay_real — piper-replay 실제 arm
# ═══════════════════════════════════════════════
def step_replay_real(args):
    header("STEP 6: piper-replay 실제 arm")

    print(f"{C.YELLOW}{'─'*60}{C.RESET}")
    print(f"{C.YELLOW}  실제 로봇 arm이 움직입니다. 아래를 확인하세요.{C.RESET}")
    print(f"{C.YELLOW}{'─'*60}{C.RESET}")
    print("  □ arm 주변 50cm 이내 사람/물체 없음")
    print("  □ CAN 인터페이스 활성화 (can0 up)")
    print("  □ EEF non-zero 확인 완료")
    print("  □ dry-run 통과 완료")
    print(f"  □ 처음 실행 — max_steps={args.max_steps} (작게 시작)")
    print(f"{C.YELLOW}{'─'*60}{C.RESET}\n")

    if not confirm("위 항목 모두 확인했습니까? 실제 arm을 움직입니까?"):
        info("취소됨")
        return False

    cmd = [
        "piper-replay",
        f"--dataset_repo_id={args.dataset_repo_id}",
        f"--dataset_root={args.dataset_root}",
        f"--episode={args.episode}",
        "--use_devices=true",
        f"--can_interface={args.can_interface}",
        f"--start_frame={args.start_frame}",
        f"--max_steps={args.max_steps}",
        f"--replay_fps={args.replay_fps}",
    ]

    info(f"실행: {' '.join(cmd)}")
    info("이상 동작 시 즉시 Ctrl+C → 멈추지 않으면 sudo ip link set can0 down\n")

    try:
        result = subprocess.run(cmd, check=False)
    except FileNotFoundError:
        err("piper-replay CLI 없음")
        return False

    if result.returncode == 0:
        ok("piper-replay 실제 arm 완료")
        return True
    else:
        err(f"piper-replay 실패 (returncode={result.returncode})")
        return False


# ═══════════════════════════════════════════════
# STEP 7: infer_real — SmolVLA 실제 arm 추론
# ═══════════════════════════════════════════════
def step_infer_real(args):
    header("STEP 7: SmolVLA 실제 arm 추론")

    if not args.pretrained_path:
        err("--pretrained_path 필요")
        sys.exit(1)

    print(f"{C.RED}{'─'*60}{C.RESET}")
    print(f"{C.RED}  실제 로봇 arm에 모델 추론 결과를 전송합니다.{C.RESET}")
    print(f"{C.RED}{'─'*60}{C.RESET}")
    print("  □ arm 주변 50cm 이내 사람/물체 없음")
    print("  □ CAN 인터페이스 활성화")
    print("  □ EEF non-zero 확인 완료")
    print("  □ infer_dry에서 action 범위 정상 확인")
    print("  □ rviz_preview에서 궤적 확인 완료")
    print("  □ replay_dry / replay_real 통과")
    print(f"  □ max_steps={args.max_steps} (5에서 시작 권장)")
    print(f"{C.RED}{'─'*60}{C.RESET}\n")

    if not confirm("위 항목 모두 확인했습니까? 실제 arm에 추론을 전송합니까?"):
        info("취소됨")
        return False

    cmd = [
        "smolvla-inference",
        f"--pretrained_path={args.pretrained_path}",
        f"--can_interface={args.can_interface}",
        f"--top_serial={args.top_serial}",
        f"--wrist_serial={args.wrist_serial}",
        f"--max_steps={args.max_steps}",
        f"--task={args.task}",
    ]

    info(f"실행: {' '.join(cmd)}")
    info("이상 동작 시 즉시 Ctrl+C → sudo ip link set can0 down\n")

    try:
        result = subprocess.run(cmd, check=False)
    except FileNotFoundError:
        err("smolvla-inference CLI 없음")
        return False

    if result.returncode == 0:
        ok("SmolVLA 실제 arm 추론 완료")
        return True
    else:
        err(f"smolvla-inference 실패 (returncode={result.returncode})")
        return False


# ═══════════════════════════════════════════════
# STEP full — 자동 순서 실행
# ═══════════════════════════════════════════════
def step_full(args):
    header("FULL: 자동 순서 실행 (data_check → calc_range → infer_dry → rviz_preview → replay_dry)")

    results = {}

    # 1. 데이터 검사
    results["data_check"] = step_data_check(args)
    if not results["data_check"]:
        err("data_check 실패 — 중단")
        return

    # 2. ACTION_MIN/MAX 계산
    results["calc_range"] = step_calc_range(args)

    # 3. 추론 dry-run
    results["infer_dry"] = step_infer_dry(args)
    if not results["infer_dry"]:
        warn("infer_dry 실패 — rviz_preview 건너뜀")
    else:
        # 4. RViz 시각화
        results["rviz_preview"] = step_rviz_preview(args)

    # 5. replay dry-run
    results["replay_dry"] = step_replay_dry(args)

    # ── 결과 요약 ──
    header("FULL 결과 요약")
    all_passed = True
    for step, passed in results.items():
        color = C.GREEN if passed else C.RED
        mark  = "✓" if passed else "✗"
        print(f"  {color}{mark} {step}{C.RESET}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        ok("모든 단계 통과 — 실제 arm 실험 준비 완료")
        info("다음: python piper_validate.py --step replay_real ...")
        info("      python piper_validate.py --step infer_real ...")
    else:
        warn("일부 단계 실패 — 실제 arm 실험 전 수정 필요")


# ───────────────────────────────────────────────
# argparse 설정
# ───────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="UGRP PiPER 검증 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument("--step", required=True,
        choices=["data_check","calc_range","infer_dry","rviz_preview",
                 "replay_dry","replay_real","infer_real","full"],
        help="실행할 단계")

    # 데이터셋
    p.add_argument("--dataset_root",
        default="/home/ugrp308/Group43/datasets/piper-smolvla",
        help="LeRobotDataset 루트 경로")
    p.add_argument("--dataset_repo_id",
        default="local/piper-smolvla",
        help="LeRobot repo id")
    p.add_argument("--range_output",
        default="action_range.json",
        help="ACTION_MIN/MAX 저장 경로")

    # 추론
    p.add_argument("--pretrained_path",
        default="",
        help="SmolVLA 체크포인트 경로")
    p.add_argument("--task",
        default="pick the pan",
        help="태스크 설명 문자열")
    p.add_argument("--max_steps",
        type=int, default=5,
        help="최대 스텝 수 (실제 arm은 5에서 시작)")
    p.add_argument("--actions_file",
        default="predicted_actions.json",
        help="추론 결과 저장/로드 경로")

    # RViz / replay
    p.add_argument("--episode",
        type=int, default=0,
        help="replay할 episode 인덱스")
    p.add_argument("--start_frame",
        type=int, default=0,
        help="replay 시작 frame offset")
    p.add_argument("--replay_fps",
        type=int, default=5,
        help="replay 속도 (fps)")

    # 하드웨어
    p.add_argument("--can_interface",
        default="can0",
        help="CAN 인터페이스 이름")
    p.add_argument("--top_serial",
        default="327122074262",
        help="top 카메라 RealSense 시리얼")
    p.add_argument("--wrist_serial",
        default="243322071626",
        help="wrist 카메라 RealSense 시리얼")

    return p.parse_args()


# ───────────────────────────────────────────────
# 메인
# ───────────────────────────────────────────────
def main():
    args = parse_args()

    dispatch = {
        "data_check":   step_data_check,
        "calc_range":   step_calc_range,
        "infer_dry":    step_infer_dry,
        "rviz_preview": step_rviz_preview,
        "replay_dry":   step_replay_dry,
        "replay_real":  step_replay_real,
        "infer_real":   step_infer_real,
        "full":         step_full,
    }

    dispatch[args.step](args)


if __name__ == "__main__":
    main()
