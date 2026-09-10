from __future__ import annotations

"""투명 PNG를 원본 영상 프레임에 alpha composite하는 모듈.

선택된 방해 도형은 모든 프레임의 같은 좌표에 고정 삽입된다.
"""

import cv2
import numpy as np


def alpha_composite_bgra(frame_bgr: np.ndarray, overlay_bgra: np.ndarray, x: int, y: int) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = overlay_bgra.shape[:2]
    frame_h, frame_w = out.shape[:2]
    if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
        raise ValueError("Overlay lies outside the frame")
    roi = out[y : y + h, x : x + w].astype(np.float32)
    fg = overlay_bgra[:, :, :3].astype(np.float32)
    alpha = (overlay_bgra[:, :, 3:4].astype(np.float32) / 255.0)
    blended = fg * alpha + roi * (1.0 - alpha)
    out[y : y + h, x : x + w] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def write_augmented_video(input_video: str, output_video: str, overlay_bgra: np.ndarray, x: int, y: int) -> dict[str, int | float]:
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0 or fps <= 0:
        raise RuntimeError(f"Invalid video properties: width={width}, height={height}, fps={fps}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output writer: {output_video}")

    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(alpha_composite_bgra(frame, overlay_bgra, x, y))
        written += 1
    writer.release()
    cap.release()
    return {"width": width, "height": height, "fps": fps, "frames": written, "reported_frames": frame_count}


def overlay_mask(frame_bgr: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int], alpha: float) -> np.ndarray:
    out = frame_bgr.copy()
    color = np.zeros_like(out)
    color[:] = color_bgr
    selected = mask > 0
    out[selected] = cv2.addWeighted(out, 1.0 - alpha, color, alpha, 0)[selected]
    return out
