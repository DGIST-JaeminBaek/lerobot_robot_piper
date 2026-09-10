#!/usr/bin/env python3
"""camera_overlay_view.py와 같은 뷰어. --box 없이 바로 6개 대표 위치를 보여준다.

candidate_assets/records_105(315개, circle/triangle/rectangle 각 105개)의
asset_index.csv(source_center_x/y, source_width/height)로 뽑았다:

  1. cy=300을 기준으로 위/아래 2단으로 나눈다 (y는 뚜렷한 이봉분포 —
     위 74~200 / 아래 410~590에 몰려있고 중간은 315개 중 1개뿐이라 실제로
     "중간 행"은 없다).
  2. 각 행 안에서 cx를 k-means(k=3)로 좌/중/우 3열로 나눈다 (x는 y와 달리
     3개 열로 고르게 갈린다 — top 36/56/64, bottom 38/56/65).
  3. 6개 클러스터 각각의 median (cx, cy, w, h)를 대표 박스로 썼다.

135개 학습 데이터셋(pick_up_the_eraser_0802_0804_0805_0813pm_135) 기준 위치는
따로 있다 — 우측 열(우상/우하) cx가 이 315개 세트보다 약 30px 안쪽이다. 그
데이터셋으로 학습한 체크포인트를 테스트할 땐 그쪽 좌표가 더 정확하다.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import camera_overlay_view as cov

OVERLAYS_315 = (
    cov.Overlay("top-left", cov.COLOR_NAMES["red"], 0.2062, 0.1018, 0.0891, 0.1653),
    cov.Overlay("top-mid", cov.COLOR_NAMES["orange"], 0.2798, 0.0743, 0.1000, 0.1778),
    cov.Overlay("top-right", cov.COLOR_NAMES["yellow"], 0.3898, 0.0972, 0.0926, 0.1729),
    cov.Overlay("bottom-left", cov.COLOR_NAMES["cyan"], 0.2204, 0.5903, 0.0840, 0.1632),
    cov.Overlay("bottom-mid", cov.COLOR_NAMES["magenta"], 0.2932, 0.5788, 0.0965, 0.1674),
    cov.Overlay("bottom-right", cov.COLOR_NAMES["blue"], 0.4047, 0.5576, 0.0930, 0.1736),
)

if __name__ == "__main__":
    cov.REFERENCE_OVERLAYS = OVERLAYS_315
    raise SystemExit(cov.main())
