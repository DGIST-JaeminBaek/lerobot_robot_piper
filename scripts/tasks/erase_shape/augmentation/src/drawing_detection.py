from __future__ import annotations

"""첫 프레임의 기존 도형/마커선을 찾아 unsafe mask로 만드는 모듈.

방해 도형이 원래 지워야 할 목표 도형과 겹치지 않도록 보호하는 역할이다.
"""

import cv2
import numpy as np


def detect_existing_drawing_mask(first_frame: np.ndarray, board_mask: np.ndarray, gray_threshold: int, dilate_px: int, min_area: int = 20) -> np.ndarray:
    gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    dark = np.where((gray <= gray_threshold) & (board_mask > 0), 255, 0).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(dark, 8)
    cleaned = np.zeros_like(dark)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        cleaned = cv2.dilate(cleaned, kernel)
    return cv2.bitwise_and(cleaned, board_mask)
