from __future__ import annotations

"""화이트보드 ROI를 읽고, 저장하고, mask로 변환하는 유틸 모듈.

board_roi.json의 네 꼭짓점이 전체 증강 기준이 된다.
현재 top/bottom 분할도 이 ROI의 y 중앙선을 기준으로 계산한다.
"""

import json
from pathlib import Path

import cv2
import numpy as np


def load_roi(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    points = np.asarray(data.get("points"), dtype=np.int32)
    if points.shape != (4, 2):
        raise ValueError(f"ROI JSON must contain exactly four points: {path}")
    return points


def save_roi(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"points": points.astype(int).tolist()}
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def board_mask(shape: tuple[int, int], points: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points.astype(np.int32)], 255)
    return mask


def border_mask(mask: np.ndarray, margin_px: int) -> np.ndarray:
    if margin_px <= 0:
        return np.zeros_like(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin_px * 2 + 1, margin_px * 2 + 1))
    eroded = cv2.erode(mask, kernel)
    return cv2.bitwise_and(mask, cv2.bitwise_not(eroded))


def select_roi(first_frame: np.ndarray, save_path: Path | None = None) -> np.ndarray:
    points: list[tuple[int, int]] = []
    window = "Select whiteboard ROI: click 4 corners, Enter=save, R=reset, Esc=cancel"

    def draw() -> np.ndarray:
        preview = first_frame.copy()
        for idx, point in enumerate(points):
            cv2.circle(preview, point, 5, (0, 255, 255), -1)
            cv2.putText(preview, str(idx + 1), (point[0] + 6, point[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if len(points) > 1:
            cv2.polylines(preview, [np.asarray(points, dtype=np.int32)], len(points) == 4, (0, 255, 255), 2)
        return preview

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            cv2.imshow(window, draw())

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    cv2.imshow(window, draw())
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):
            if len(points) != 4:
                continue
            result = np.asarray(points, dtype=np.int32)
            if save_path is not None:
                save_roi(save_path, result)
            cv2.destroyWindow(window)
            return result
        if key in (ord("r"), ord("R")):
            points.clear()
            cv2.imshow(window, draw())
        if key == 27:
            cv2.destroyWindow(window)
            raise RuntimeError("ROI selection cancelled")
