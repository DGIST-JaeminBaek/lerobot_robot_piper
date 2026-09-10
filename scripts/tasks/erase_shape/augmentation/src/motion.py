from __future__ import annotations

"""영상 전체에서 로봇팔 등 움직이는 영역을 찾아 unsafe mask로 만드는 모듈.

방해 도형이 로봇팔 동작 경로, 카메라 주변 움직임과 겹치지 않게 하기 위해 사용한다.
"""

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from .stabilization import align_to_reference


def _iter_indices(frame_count: int, stride: int) -> list[int]:
    stride = max(1, stride)
    return list(range(0, frame_count, stride))


def detect_motion_mask(
    video_path: str,
    board_mask: np.ndarray,
    sample_stride: int,
    pixel_threshold: int,
    ever_changed_threshold: int,
    changed_ratio_threshold: float,
    dilate_px: int,
    stabilize: bool,
) -> tuple[np.ndarray, dict[str, int | float]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = _iter_indices(frame_count, sample_stride)
    if not indices:
        raise RuntimeError("Video contains no readable frames")

    ref_gray: np.ndarray | None = None
    prev_gray: np.ndarray | None = None
    warp: np.ndarray | None = None
    change_counts = np.zeros(board_mask.shape, dtype=np.uint16)
    ever_changed = np.zeros(board_mask.shape, dtype=np.uint8)
    sampled = 0
    iterator = tqdm(indices, desc="Analyzing motion", unit="frame") if tqdm else indices

    for frame_idx in iterator:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if ref_gray is None:
            ref_gray = gray
            prev_gray = gray
            sampled += 1
            continue
        if stabilize:
            try:
                gray, warp = align_to_reference(gray, ref_gray, warp)
            except cv2.error:
                pass
        diff_prev = cv2.absdiff(gray, prev_gray if prev_gray is not None else ref_gray)
        diff_ref = cv2.absdiff(gray, ref_gray)
        changed = ((diff_prev >= pixel_threshold) | (diff_ref >= pixel_threshold)) & (board_mask > 0)
        ever = (diff_ref >= ever_changed_threshold) & (board_mask > 0)
        change_counts[changed] += 1
        ever_changed[ever] = 255
        prev_gray = gray
        sampled += 1

    cap.release()
    required_count = max(1, int(np.ceil(sampled * changed_ratio_threshold)))
    motion = np.where(change_counts >= required_count, 255, 0).astype(np.uint8)
    motion = cv2.bitwise_or(motion, ever_changed)
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        motion = cv2.morphologyEx(motion, cv2.MORPH_CLOSE, kernel)
        motion = cv2.dilate(motion, kernel)
    motion = cv2.bitwise_and(motion, board_mask)
    stats = {
        "video_frames": frame_count,
        "sampled_frames": sampled,
        "motion_unsafe_area": int(np.count_nonzero(motion)),
    }
    return motion, stats
