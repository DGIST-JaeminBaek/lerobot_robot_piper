#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import pyrealsense2 as rs


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENV_FILE = REPO_ROOT / "configs" / "recording.env"


def load_env_file(path: Path) -> dict[str, str]:
    """recording.env를 읽되, 이미 설정된 셸 환경변수는 덮어쓰지 않는다."""
    loaded: dict[str, str] = {}
    if not path.is_file():
        return loaded
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key:
            loaded[key] = value
            os.environ.setdefault(key, value)
    return loaded


def list_devices() -> list[str]:
    # 연결된 RealSense serial 출력
    ctx = rs.context()
    serials = []
    for dev in ctx.query_devices():
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        serials.append(serial)
        print(f"{serial}  {name}")
    return serials


def start_pipeline(serial: str, width: int, height: int, fps: int, *, depth: bool = False) -> rs.pipeline:
    """컬러(선택적으로 depth) 스트림을 시작한다."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    if depth:
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    pipeline.start(config)
    return pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RealSense RGB view")
    parser.add_argument("--serial", default="", help="RealSense serial number")
    parser.add_argument("--serials", nargs="*", default=None, help="RealSense serial numbers for simultaneous view")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--configured", action="store_true", help="TOP_CAM/WRIST_CAM 설정 장치를 선택")
    parser.add_argument("--cycle", action="store_true", help="여러 장치를 한 창에서 A/D 또는 화살표로 전환")
    parser.add_argument("--depth", action="store_true", help="컬러 옆에 depth 컬러맵을 표시")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE, help="카메라 serial 설정 파일")
    args = parser.parse_args(argv)

    load_env_file(args.env_file)

    serials = list_devices()
    if args.list:
        return 0
    if not serials:
        print("No RealSense cameras found.", file=sys.stderr)
        return 1

    use_configured = args.configured or not args.serial and not args.serials and not args.list
    if use_configured:
        configured = [os.environ.get(key, "").strip() for key in ("TOP_CAM", "WRIST_CAM")]
        selected_serials = [serial for serial in configured if serial]
        if not selected_serials:
            print("TOP_CAM 또는 WRIST_CAM 환경변수가 설정되지 않았습니다.", file=sys.stderr)
            return 1
    elif args.serials:
        selected_serials = args.serials
    else:
        selected_serials = [args.serial or serials[0]]

    for serial in selected_serials:
        if serial not in serials:
            print(f"Serial not found: {serial}", file=sys.stderr)
            return 1

    cycle = args.cycle or (use_configured and len(selected_serials) > 1)
    if len(selected_serials) > 1:
        pipelines = {
            serial: start_pipeline(serial, args.width, args.height, args.fps, depth=args.depth)
            for serial in selected_serials
        }
        names = list(pipelines)
        current = 0
        colorizer = rs.colorizer()
        try:
            while True:
                active = names[current]
                if cycle:
                    targets = [active]
                else:
                    targets = names
                for serial in targets:
                    frames = pipelines[serial].wait_for_frames()
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue
                    image = np.asanyarray(color_frame.get_data())
                    if args.depth:
                        depth_frame = frames.get_depth_frame()
                        if depth_frame:
                            depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
                            image = np.hstack([image, depth_image])
                    cv2.putText(image, serial, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow("RealSense cycle (A/D switch, q quit)" if cycle else f"RealSense {serial}", image)
                key = cv2.waitKeyEx(1)
                if key in (ord("q"), ord("Q"), 27):
                    break
                if cycle and key in (83, 65363, 2555904, ord("d"), ord("D")):
                    current = (current + 1) % len(names)
                if cycle and key in (81, 65361, 2424832, ord("a"), ord("A")):
                    current = (current - 1) % len(names)
        finally:
            for pipeline in pipelines.values():
                pipeline.stop()
            cv2.destroyAllWindows()
        return 0

    selected = selected_serials[0]
    pipeline = start_pipeline(selected, args.width, args.height, args.fps, depth=args.depth)
    window = f"RealSense {selected}"
    colorizer = rs.colorizer()
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            image = np.asanyarray(color_frame.get_data())
            if args.depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    image = np.hstack([image, np.asanyarray(colorizer.colorize(depth_frame).get_data())])
            cv2.imshow(window, image)
            if cv2.waitKeyEx(1) in (ord("q"), ord("Q"), 27):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
