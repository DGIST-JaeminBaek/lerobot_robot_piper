#!/usr/bin/env python3
"""RealSense camera cycle viewer.

Opens each configured RealSense camera by serial number (color stream only)
and lets you switch between them with the Left/Right arrow keys.

Env vars expected (matches your .env style):
    TOP_CAM=327122074262
    WRIST_CAM=243322071626

Add more by extending the CAMERAS dict below (name -> serial).
"""
import os
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs

VIEW_W, VIEW_H, FPS = 1280, 720, 30

# name -> serial number. Pulls from env vars, falls back to the values you gave.
CAMERAS = {
    "TOP": os.environ.get("TOP_CAM", "327122074262"),
    "WRIST": os.environ.get("WRIST_CAM", "243322071626"),
}

LEFT_KEYS = {81, 65361, 2424832, ord('a'), ord('A')}
RIGHT_KEYS = {83, 65363, 2555904, ord('d'), ord('D')}
QUIT_KEYS = {ord('q'), ord('Q'), 27}  # 27 = ESC


DEPTH_W, DEPTH_H = 1280, 720  # D435if depth maxes out at 1280x720


def ask_show_depth() -> bool:
    while True:
        ans = input("Depth map도 같이 볼까요? (y/n): ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("y 또는 n으로 입력해주세요.")


def start_pipeline(serial: str, show_depth: bool) -> rs.pipeline:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, VIEW_W, VIEW_H, rs.format.bgr8, FPS)
    if show_depth:
        config.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, FPS)
    pipeline.start(config)
    return pipeline


def main():
    show_depth = ask_show_depth()

    # --- Phase 1: connect to each camera by serial -------------------------
    print("Connecting to RealSense cameras by serial...")
    ctx = rs.context()
    connected_serials = {d.get_info(rs.camera_info.serial_number) for d in ctx.query_devices()}

    pipelines: dict[str, rs.pipeline] = {}
    aligners: dict[str, rs.align] = {}
    for name, serial in CAMERAS.items():
        if serial not in connected_serials:
            print(f"  {name} ({serial}): NOT FOUND, skipping")
            continue
        try:
            pipelines[name] = start_pipeline(serial, show_depth)
            if show_depth:
                aligners[name] = rs.align(rs.stream.color)  # depth -> color 픽셀 정합
            print(f"  {name} ({serial}): OK")
        except Exception as e:
            print(f"  {name} ({serial}): FAILED to start ({e})")

    if not pipelines:
        print("No configured RealSense cameras could be opened.")
        sys.exit(1)

    names = list(pipelines.keys())
    n = len(names)
    print(f"\n{n} camera(s) live. <- / -> (or A/D) to switch, 'q' to quit.\n")

    time.sleep(0.3)  # let auto-exposure settle a touch

    # --- Phase 2: single-camera display, cycle with arrow keys ------------
    current = 0
    win = "RealSense Cycle Viewer (<-/-> switch, q quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    colorizer = rs.colorizer()  # depth(z16) -> 보기 좋은 컬러맵(bgr8)

    while True:
        name = names[current]
        pipeline = pipelines[name]

        raw_frames = pipeline.wait_for_frames(timeout_ms=1000)

        if show_depth:
            align = aligners[name]
            frames = align.process(raw_frames)  # depth를 color 화각/픽셀에 맞춤
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
        else:
            color_frame = raw_frames.get_color_frame()
            depth_frame = None

        if color_frame:
            color_img = np.asanyarray(color_frame.get_data())
        else:
            color_img = np.zeros((VIEW_H, VIEW_W, 3), dtype=np.uint8)

        if show_depth:
            if depth_frame:
                depth_img = np.asanyarray(colorizer.colorize(depth_frame).get_data())
                depth_img = cv2.resize(depth_img, (color_img.shape[1], color_img.shape[0]))
            else:
                depth_img = np.zeros_like(color_img)
            frame = np.hstack([color_img, depth_img])  # color | depth 나란히
            suffix = "  color | depth"
        else:
            frame = color_img
            suffix = "  color"

        label = f"{name}  ({CAMERAS[name]})  [{current + 1}/{n}]{suffix}"
        cv2.putText(frame, label, (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow(win, frame)
        key = cv2.waitKeyEx(30)

        if key in QUIT_KEYS:
            break
        elif key in RIGHT_KEYS:
            current = (current + 1) % n
        elif key in LEFT_KEYS:
            current = (current - 1) % n

    # cleanup
    for pipeline in pipelines.values():
        pipeline.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
