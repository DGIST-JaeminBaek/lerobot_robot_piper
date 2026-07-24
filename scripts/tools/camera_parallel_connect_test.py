#!/usr/bin/env python3
"""RealSense 카메라만 켜서 병렬 connect()가 되는지 테스트 — 로봇 팔(CAN)은 전혀 안 건드림.

piper_follower.py의 PiperFollower.connect()가 이제 카메라 connect()를 병렬로
돌리도록 바뀌었는데, 예전에 이 방식이 실제 하드웨어에서 "read failed"/타임아웃을
낸 적이 있어서(USB 대역폭 경합으로 추정했었음, CPU 쿨링 문제였을 가능성도 있음)
로봇을 켜지 않고 카메라만으로 먼저 재현 여부를 확인하기 위한 스크립트.

사용법:
    conda activate ugrp
    cd /home/ugrp43/UGRP/lerobot_robot_piper
    python scripts/tools/camera_parallel_connect_test.py
"""

from __future__ import annotations

import pathlib
import time
from concurrent.futures import ThreadPoolExecutor

from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / "configs" / "recording.env"


def load_env(path: pathlib.Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def main() -> None:
    env = load_env(ENV_PATH)
    top_serial = env.get("TOP_CAM", "327122074262")
    wrist_serial = env.get("WRIST_CAM", "243322071626")
    width = int(env.get("CAM_WIDTH", "1280"))
    height = int(env.get("CAM_HEIGHT", "720"))
    fps = int(env.get("FPS", "30"))
    use_depth = (env.get("REALSENSE_USE_DEPTH", "false").lower() == "true")
    warmup_s = float(env.get("REALSENSE_WARMUP_S", "10.0"))

    print(f"top={top_serial} wrist={wrist_serial} {width}x{height}@{fps} use_depth={use_depth} warmup_s={warmup_s}")

    cams = {
        "top": RealSenseCamera(
            RealSenseCameraConfig(
                serial_number_or_name=top_serial, width=width, height=height, fps=fps,
                use_depth=use_depth, warmup_s=warmup_s,
            )
        ),
        "wrist": RealSenseCamera(
            RealSenseCameraConfig(
                serial_number_or_name=wrist_serial, width=width, height=height, fps=fps,
                use_depth=use_depth, warmup_s=warmup_s,
            )
        ),
    }

    print("\n=== 병렬 connect() 시도 ===")
    t0 = time.perf_counter()
    errors: dict[str, Exception] = {}
    with ThreadPoolExecutor(max_workers=len(cams), thread_name_prefix="cam_connect") as executor:
        futures = {name: executor.submit(cam.connect, warmup=True) for name, cam in cams.items()}
        for name, future in futures.items():
            try:
                future.result()
                print(f"  {name}: connected OK")
            except Exception as e:
                errors[name] = e
                print(f"  {name}: FAILED - {type(e).__name__}: {e}")
    dt = time.perf_counter() - t0
    print(f"총 소요 시간: {dt:.1f}s")

    if errors:
        print("\n병렬 connect 중 실패 발생 — piper_follower.py의 순차 연결로 되돌리는 걸 권장.")
        return

    print("\n=== 각 카메라에서 실제 프레임 읽기 확인 ===")
    for name, cam in cams.items():
        frame = cam.async_read()
        print(f"  {name}: frame shape={frame.shape} dtype={frame.dtype}")
        if getattr(cam, "use_depth", False):
            depth = cam.read_depth(timeout_ms=0)
            print(f"  {name}_depth: shape={depth.shape} dtype={depth.dtype}")

    print("\n=== disconnect ===")
    for name, cam in cams.items():
        cam.disconnect()
        print(f"  {name}: disconnected")

    print("\n결과: 병렬 connect 성공, 프레임 읽기도 정상 — 재현 안 됨.")


if __name__ == "__main__":
    main()
