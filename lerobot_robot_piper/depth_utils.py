"""RealSense raw depth(uint16) -> turbo 컬러맵 RGB(uint8) 변환.

jet 금지: 지각적으로 불균일해서 SigLIP 사전학습 색 분포와 어긋난다. turbo 사용.
정규화 범위(depth_min_m/depth_max_m)는 태스크마다(카메라-보드 거리 등) 재보정 필요 —
고정하지 않으면 프레임마다 같은 높이가 다른 색이 되어 학습이 안 됨.
"""

import cv2
import numpy as np
from numpy.typing import NDArray

DEPTH_MIN_M = 0.20
DEPTH_MAX_M = 0.80


def depth_to_colormap(
    depth_raw_u16: NDArray,
    depth_scale: float = 0.001,
    dmin: float = DEPTH_MIN_M,
    dmax: float = DEPTH_MAX_M,
) -> NDArray:
    """RealSense raw uint16 depth(H,W) -> turbo 컬러맵 RGB uint8(H,W,3).

    depth_scale: RealSense depth unit (D400 계열 기본 0.001 = 1mm). 실측 확인 필요.
    0 값(hole/결측)은 dmax로 처리(무한원 취급).
    """
    d = depth_raw_u16.astype(np.float32) * depth_scale
    d[d == 0.0] = dmax
    d = np.clip(d, dmin, dmax)
    norm = (d - dmin) / (dmax - dmin)
    u8 = (norm * 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
