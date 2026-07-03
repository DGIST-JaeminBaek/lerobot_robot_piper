#!/usr/bin/env python3
import argparse
import sys

import cv2
import numpy as np
import pyrealsense2 as rs


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


def start_pipeline(serial: str, width: int, height: int, fps: int) -> rs.pipeline:
    # RGB stream 시작
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    pipeline.start(config)
    return pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="RealSense RGB view")
    parser.add_argument("--serial", default="", help="RealSense serial number")
    parser.add_argument("--serials", nargs="*", default=None, help="RealSense serial numbers for simultaneous view")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    serials = list_devices()
    if args.list:
        return 0
    if not serials:
        print("No RealSense cameras found.", file=sys.stderr)
        return 1

    if args.serials:
        # 여러 카메라 동시 stream 확인
        selected_serials = args.serials
        for serial in selected_serials:
            if serial not in serials:
                print(f"Serial not found: {serial}", file=sys.stderr)
                return 1

        pipelines = {serial: start_pipeline(serial, args.width, args.height, args.fps) for serial in selected_serials}
        try:
            while True:
                for serial, pipeline in pipelines.items():
                    frames = pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        image = np.asanyarray(color_frame.get_data())
                        cv2.imshow(f"RealSense {serial}", image)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
        finally:
            for pipeline in pipelines.values():
                pipeline.stop()
            cv2.destroyAllWindows()
        return 0

    selected = args.serial or serials[0]
    if selected not in serials:
        print(f"Serial not found: {selected}", file=sys.stderr)
        return 1

    pipeline = start_pipeline(selected, args.width, args.height, args.fps)
    window = f"RealSense {selected}"

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            image = np.asanyarray(color_frame.get_data())
            cv2.imshow(window, image)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
