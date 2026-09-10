from __future__ import annotations

"""motion mask 계산 전 프레임 정렬을 시도하는 보조 모듈.

카메라 흔들림이 있을 때 사용할 수 있지만, 현재 0802~0805 기본 실행에서는 stabilize 옵션을 켜지 않았다.
"""

import cv2
import numpy as np


def align_to_reference(gray: np.ndarray, reference_gray: np.ndarray, warp: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if warp is None:
        warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5)
    _cc, updated = cv2.findTransformECC(reference_gray, gray, warp, cv2.MOTION_EUCLIDEAN, criteria)
    aligned = cv2.warpAffine(gray, updated, (reference_gray.shape[1], reference_gray.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return aligned, updated
