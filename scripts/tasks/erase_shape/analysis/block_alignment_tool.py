#!/usr/bin/env python3
"""지우개 블럭을 학습 데이터와 같은 자리에 놓도록 실시간으로 안내한다.

정책은 학습 때 본 것과 같은 장면을 봐야 제대로 동작한다. 블럭 홈 위치가 어긋나면
집으러 가는 첫 동작부터 빗나가므로, 추론 전에 이 도구로 맞춰두는 것이 안전하다.

top 카메라를 열어 매 프레임 블럭을 찾고, 학습 데이터 255개(configs/erase_shape_315_manifest.json
기준 315개 중 0727 제외)에서 측정한 기준 위치와의 차이를 화살표와 수치로 보여준다.
로봇 팔은 전혀 건드리지 않는다.

기준값 출처(2026-08-18 갱신): erase_shape_315_manifest.json의 315개 원본 녹화 첫
프레임에서 블럭 무게중심을 측정. 0727(60개)은 도형 종류별로 블럭을 다시 놓아서
집단 자체가 갈린다 — rectangle 20개는 cy가 나머지보다 ~16~19px 위(다른 도형과
검증된 systematic offset, 무작위 노이즈 아님), circle/triangle 40개는 다른 세션과
비슷한 자리라 기준에서 제외했다(구 기준값도 같은 이유로 0727을 뺐었다). 나머지
255개(0802/0804/0805/0811/0812/0813/0813pm)의 std는 cx 1.7px / cy 2.3px.
이전 기준값(0802/0804/0805 120개, cx=428.9 cy=305.9)과 비교하면 cx +0.6px,
cy −0.3px로 사실상 같다 — 그동안 세팅이 드리프트하지 않았다는 뜻이기도 하다.

실행:
    python scripts/tasks/erase_shape/analysis/block_alignment_tool.py
    q 종료 / s 스냅샷 저장 / r 카메라 정렬 재측정 / c 그려진 도형 검사(--target-shape 필요)

    # 도형을 그릴 6개 대표 위치(상좌/상중/상우/하좌/하중/하우)도 같이 보려면
    # camera_overlay_view_315.py의 좌표를 재사용해 --shape-zones로 켠다:
    python scripts/tasks/erase_shape/analysis/block_alignment_tool.py --shape-zones 135

    # target 도형(circle/triangle/rectangle)별 실측 박스 크기로 그린다
    # 보려면(315/132 zone-set 지원, 135는 도형별 원본 데이터 없음) --target-shape를 준다.
    # 이 상태에서 c를 누르면 실제 채점 파이프라인(ink_metric.detect_shapes/classify)으로
    # 그려진 도형을 검사해 PASS/FAIL을 보여준다.
    python scripts/tasks/erase_shape/analysis/block_alignment_tool.py --shape-zones 315 --target-shape circle

    # 기본은 zone마다 다른 박스 크기(--size-source per-zone). 6개 zone 다 합친
    # 도형별 median 하나로 통일하려면(zone별 표본이 작아 세션 편차에 흔들릴 때):
    python scripts/tasks/erase_shape/analysis/block_alignment_tool.py --shape-zones 315 --target-shape circle --size-source global

    # 132 체크포인트(pick_up_the_eraser_0727_0812_0813am_132)를 평가할 땐 132 zone을
    # 쓴다 — 이 학습셋에는 315의 좌측 열에 해당하는 위치가 아예 없어서 zone이 4개다:
    python scripts/tasks/erase_shape/analysis/block_alignment_tool.py --shape-zones 132 --target-shape circle --size-source global
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import cv2
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
INFERENCE_DIR = REPO_ROOT / "scripts" / "piper" / "inference"
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))
LIB_DIR = REPO_ROOT / "scripts" / "tasks" / "erase_shape" / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))
import ink_metric as M  # noqa: E402  — c 키(그려진 도형 검사)에서만 쓴다

# 학습 데이터 255개(315개 중 0727 제외)에서 측정한 블럭 중심 (raw 1280x720 좌표).
# 2026-08-18: 이전 120개(0802/0804/0805) 기준값(428.9, 305.9)과 비교해 cx +0.6px,
# cy -0.3px — 드리프트 없음을 확인하고 표본만 315개 전체로 확장했다.
REFERENCE_CX = 429.5
REFERENCE_CY = 305.6
REFERENCE_SD = (1.7, 2.3)          # 학습 데이터의 표준편차 — 참고 표시용

# 카메라 정렬 기준. shape_position_analysis.py와 같은 값이다.
REFERENCE_EDGE_X = 751

# 블럭을 찾을 영역. 오른쪽 상한은 보드 경계 근처 브래킷을 피하려고 좁혀둔 값이다.
BOARD_X = (110, 680)

# 화면 px → 실제 mm 근사. 150개 에피소드에서 (도형 화면 위치) vs (지울 때 EEF 위치)를
# 회귀해 얻은 기울기다. 세로 R=0.943, 가로 R=0.818 — 방향과 크기 감을 잡는 용도이고
# 정밀 측정값은 아니다.
MM_PER_PX = (0.690, 0.625)         # (가로, 세로)

TOLERANCE_OK = 5.0                 # px. 이 안이면 맞춘 것으로 본다
TOLERANCE_WARN = 15.0

GREEN, AMBER, RED = (80, 190, 60), (30, 170, 235), (60, 60, 225)
WHITE, CYAN = (255, 255, 255), (255, 220, 60)

# 도형을 그릴 6개 대표 위치(상좌/상중/상우/하좌/하중/하우, raw 1280x720 좌표).
# camera_overlay_view_315.py와 같은 산출 방법(y=300 기준 위/아래 2단, 각 행에서
# cx k-means 3분할, 클러스터 median)이고 값도 그대로 가져왔다 — 135는
# pick_up_the_eraser_0802_0804_0805_0813pm_135 학습 데이터셋(135개) 기준,
# 315는 candidate_assets/records_105(315개 후보 풀) 기준. 우측 열(우상/우하)
# cx가 315쪽이 135쪽보다 30px가량 바깥이다 — 그 135개로 학습한 체크포인트를
# 테스트할 땐 135 쪽이 더 정확하다.
SHAPE_ZONES: dict[str, tuple[tuple[str, float, float, float, float], ...]] = {
    "135": (
        ("top-left", 319.4, 144.1, 118.0, 128.0),
        ("top-mid", 431.6, 129.3, 123.5, 129.5),
        ("top-right", 525.2, 129.9, 113.0, 129.0),
        ("bottom-left", 325.9, 475.5, 110.5, 119.0),
        ("bottom-mid", 448.8, 473.6, 124.0, 122.0),
        ("bottom-right", 548.1, 473.2, 118.0, 124.5),
    ),
    "315": (
        ("top-left", 321.0, 132.8, 114.0, 119.0),
        ("top-mid", 422.2, 117.5, 128.0, 128.0),
        ("top-right", 558.2, 132.2, 118.5, 124.5),
        ("bottom-left", 335.8, 483.8, 107.5, 117.5),
        ("bottom-mid", 437.0, 477.0, 123.5, 120.5),
        ("bottom-right", 577.5, 464.0, 119.0, 125.0),
    ),
    # pick_up_the_eraser_0727_0812_0813am_132 학습셋(=configs/erase_shape_pickup_prompt_
    # manifest.json 132개) 기준. **열이 2개뿐이라 4 zone이다** — 이 데이터에는 315의
    # 좌측 열(cx≈325~360)에 해당하는 위치가 아예 없고, 132의 "좌"가 315 기준으로는
    # 중앙 열(cx≈405~433)이다. 즉 132 체크포인트를 평가하면서 315의 좌측 zone에
    # 도형을 그리면 학습 분포 밖이다(2026-08-22 실측).
    # 산출: shape_position_analysis.py로 원본 첫 프레임에서 132개 전부 측정(132/132
    # 성공) -> cy=300으로 상/하 2행 -> 행마다 cx k-means(k=2) -> zone별 median.
    # 4 zone x 3도형 x 11개씩 정확히 균등하게 나눠졌다.
    # 주의: 위 "135"/"315" 값은 augmentation asset_index.csv에서 나온 것이라 측정
    # 파이프라인이 다르다. 절대 크기를 세 세트끼리 직접 비교하지 말 것.
    "132": (
        ("top-left", 419.4, 113.8, 134.0, 130.0),
        ("top-right", 571.3, 136.7, 123.0, 128.0),
        ("bottom-left", 432.0, 482.5, 123.0, 123.0),
        ("bottom-right", 589.3, 453.0, 124.0, 129.0),
    ),
}
ZONE_COLORS = (  # BGR, camera_overlay_view.py COLOR_NAMES와 맞춘 팔레트
    (0, 0, 255), (0, 140, 255), (0, 255, 255), (255, 255, 0), (255, 0, 255), (255, 0, 0),
)

# zone별 박스 크기(w,h)를 도형(target)별로 다시 잰 값. cx,cy는 SHAPE_ZONES["315"]와
# 그대로 같다(클러스터 중심은 안 바꾼다) — asset_index.csv(315개, shape 컬럼 있음)를
# camera_overlay_view_315.py와 같은 클러스터링(cy=300 기준 2행 x 행별 k-means k=3)으로
# zone에 배정한 뒤, 각 zone 안에서 도형별 median w/h만 다시 뽑았다(2026-08-20 실측).
# rectangle이 pooled와 제일 가깝고(대부분 ±0~8px), triangle은 전 zone에서 폭이 좁고
# (−5.5~−15.5px) 높이가 크며(대부분 +5~+20px), circle은 폭이 넓다(+1.5~+13px) — 무작위가
# 아니라 도형마다 일관된 패턴이다. 135 zone-set은 도형 라벨이 붙은 원본 데이터가 없어서
# 지원하지 않는다(--target-shape를 135와 같이 주면 경고만 찍고 pooled 박스를 그대로 씀).
SHAPE_SIZE_315: dict[str, dict[str, tuple[float, float]]] = {
    "circle": {
        "top-left": (127.0, 114.0), "top-mid": (136.0, 138.0), "top-right": (125.0, 122.0),
        "bottom-left": (109.0, 114.0), "bottom-mid": (128.0, 124.5), "bottom-right": (126.0, 124.5),
    },
    "triangle": {
        "top-left": (104.5, 139.0), "top-mid": (113.0, 129.0), "top-right": (103.0, 131.5),
        "bottom-left": (102.0, 129.0), "bottom-mid": (112.5, 120.5), "bottom-right": (105.5, 130.0),
    },
    "rectangle": {
        "top-left": (111.0, 113.0), "top-mid": (132.0, 120.0), "top-right": (117.5, 119.0),
        "bottom-left": (110.5, 116.0), "bottom-mid": (126.0, 119.0), "bottom-right": (119.0, 121.0),
    },
}
# 132 zone-set의 도형별 zone 크기. SHAPE_ZONES["132"]와 같은 측정치에서 뽑았다
# (zone당 도형별 11개씩).
SHAPE_SIZE_132: dict[str, dict[str, tuple[float, float]]] = {
    "circle": {
        "top-left": (135.0, 132.0), "top-right": (125.0, 115.0),
        "bottom-left": (120.0, 112.0), "bottom-right": (127.0, 122.0),
    },
    "triangle": {
        "top-left": (118.0, 131.0), "top-right": (121.0, 134.0),
        "bottom-left": (113.0, 125.0), "bottom-right": (115.0, 137.0),
    },
    "rectangle": {
        "top-left": (139.0, 125.0), "top-right": (129.0, 123.0),
        "bottom-left": (135.0, 125.0), "bottom-right": (135.0, 129.0),
    },
}
SHAPE_SIZE_PER_ZONE = {"315": SHAPE_SIZE_315, "132": SHAPE_SIZE_132}

# zone 구분 없이 도형별로 전체를 합쳐서 잰 median. zone별로 나누면 일부 조합 표본이
# 10~11개까지 줄어 세션 간 계통적 차이(예: 특정 세션에서만 유난히 작게/크게 그림)에
# median이 휘둘리는 걸 실측으로 확인했다(2026-08-21, 315의 top-left/circle: 0811
# 세션은 작게 0805 세션은 크게 그려서 median 대표성이 떨어짐). 자리마다 크기가 다를
# 이유가 실제로는 없으므로 이쪽이 표본이 커서 더 안정적이다(315는 도형당 105개,
# 132는 44개). --size-source global로 켠다(기본은 zone별 유지).
SHAPE_SIZE_GLOBAL: dict[str, dict[str, tuple[float, float]]] = {
    "315": {
        "circle": (127.0, 124.0),
        "triangle": (106.0, 129.0),
        "rectangle": (118.0, 119.0),
    },
    "132": {
        "circle": (127.0, 121.5),
        "triangle": (114.5, 131.0),
        "rectangle": (136.0, 125.5),
    },
}
CHECK_TOLERANCE_PX = 70.0           # c 키 검사: 검출된 도형 중심이 zone 중심에서 이 안이면 인정

# 직전 프로세스가 카메라를 닫은 직후엔 커널이 USB 장치를 완전히 회수하기까지 시간이
# 걸려서 곧바로 다시 열면 "Device or resource busy"로 실패한다(2026-08-21 실물:
# 이 도구를 GUI 토글 없이 터미널에서 연달아 재실행하다 재현, 결국 세그폴트까지 감).
# 이 프로젝트 다른 곳(ALIGN_LAUNCH_DELAY_MS=2500, CAMERA_POST_CONNECT_WAIT_S=2.0)에서
# 이미 "몇 초면 충분하다"가 실측으로 검증돼 있어서, 그 값을 그대로 재시도 간격으로 쓴다.
CAMERA_CONNECT_RETRIES = 4
CAMERA_CONNECT_RETRY_WAIT_S = 2.0


def connect_camera_with_retry(camera) -> None:
    """카메라가 아직 안 풀렸으면 몇 초 기다렸다 다시 연다. 다 실패하면 마지막 예외를 던진다."""
    for attempt in range(1, CAMERA_CONNECT_RETRIES + 1):
        try:
            camera.connect(warmup=True)
            return
        except ConnectionError as error:
            if attempt == CAMERA_CONNECT_RETRIES:
                raise
            print(f"[CAM] 연결 실패({attempt}/{CAMERA_CONNECT_RETRIES}) — 직전 프로세스가 아직 "
                  f"카메라를 안 놓았을 수 있음. {CAMERA_CONNECT_RETRY_WAIT_S:g}초 뒤 재시도: {error}",
                  file=sys.stderr)
            time.sleep(CAMERA_CONNECT_RETRY_WAIT_S)


def measure_edge_x(rgb: np.ndarray) -> int:
    """나무 테이블 경계의 x. 카메라가 미세하게 움직였는지 보는 기준선."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(float)
    profile = gray[300:500, :].mean(axis=0)
    return int(np.argmax(np.abs(np.diff(profile))[700:1100])) + 700


def detect_block(rgb: np.ndarray) -> dict | None:
    """나무 블럭 = 따뜻한 색(R이 B보다 큼)의 최대 덩어리.

    보드와 마커는 무채색이라 색상만으로 깔끔하게 갈린다. 높이 300px을 넘는 덩어리는
    블럭이 아니라 나무 테이블 자락이 들어온 것이므로 제외한다.
    """
    board = rgb[:, BOARD_X[0]:BOARD_X[1]]
    red, _, blue = (board[:, :, i].astype(int) for i in range(3))
    warm = ((red - blue) > 14).astype(np.uint8)
    warm = cv2.morphologyEx(warm, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    count, _, stats, centroids = cv2.connectedComponentsWithStats(warm, 8)
    candidates = [
        i for i in range(1, count)
        if stats[i, cv2.CC_STAT_AREA] > 2000 and stats[i, cv2.CC_STAT_HEIGHT] < 300
    ]
    if not candidates:
        return None
    best = max(candidates, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    return {
        "cx": float(centroids[best][0]) + BOARD_X[0],
        "cy": float(centroids[best][1]),
        "x": int(stats[best, 0]) + BOARD_X[0],
        "y": int(stats[best, 1]),
        "w": int(stats[best, 2]),
        "h": int(stats[best, 3]),
    }


def draw_shape_zones(canvas: np.ndarray, zones: tuple, target_shape: str | None = None) -> None:
    """도형을 그릴 대표 위치를 얇은 라벨 박스로 그린다. 블럭 정렬 표시보다
    먼저(아래층에) 그려서 시선이 블럭 쪽(초록/노랑/빨강)에 남게 한다.

    박스만 그린다. 예전에는 circle일 때 박스 안에 타원 가이드를 겹쳐 그렸는데,
    선이 많아 보드가 지저분해지고 실제로 따라 그리는 데 도움이 안 됐다
    (2026-08-25 사용자 판단 — 삼각형 가이드를 뺐던 것과 같은 이유).
    target_shape는 이제 박스 '크기'에만 영향을 준다.
    """
    for (label, cx, cy, w, h), color in zip(zones, ZONE_COLORS):
        x0, y0 = int(cx - w / 2), int(cy - h / 2)
        x1, y1 = int(cx + w / 2), int(cy + h / 2)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
        cv2.putText(canvas, label, (x0, max(0, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                   0.42, color, 1, cv2.LINE_AA)


def check_drawn_shape(frame_bgr: np.ndarray, board: tuple, dark_ratio: float, exclude,
                      target_shape: str, zones: tuple) -> tuple[bool, str]:
    """실제 채점 파이프라인(ink_metric.detect_shapes/classify)으로 그려진 도형을 검사한다.

    '이상적 도형과 얼마나 닮았나'를 새로 정의하는 대신, erase_run.py가 시도 시작 직전에
    쓰는 것과 같은 함수로 (a) target 종류로 분류되는지 (b) 6개 zone 중심 중 하나에
    충분히 가까운지만 본다 — 이 검사를 통과하면 실제 평가 게이트도 통과한다는 뜻이다.
    """
    # 화면 표시 문구는 영문만 쓴다 — draw_overlay와 같은 이유(cv2 기본 폰트는 한글 없음).
    try:
        shapes, _white = M.detect_shapes(frame_bgr, board, dark_ratio, exclude)
    except Exception as error:
        return False, f"detect failed: {error}"

    best_label, best_dist = None, None
    for label, (x, y, w, h) in shapes:
        base = label.split("#", 1)[0]
        if base != target_shape:
            continue
        scx, scy = x + w / 2, y + h / 2
        dist = min(float(np.hypot(scx - zcx, scy - zcy)) for _, zcx, zcy, _, _ in zones)
        if best_dist is None or dist < best_dist:
            best_label, best_dist = label, dist

    if best_label is None:
        return False, f"FAIL - no {target_shape} found"
    if best_dist > CHECK_TOLERANCE_PX:
        return False, f"FAIL - {best_label} found but {best_dist:.0f}px from nearest zone"
    return True, f"PASS - {best_label}, {best_dist:.0f}px from zone"


def draw_overlay(frame: np.ndarray, found: dict | None, ref: tuple[float, float],
                 edge_x: int, shift: float, zones: tuple = (), target_shape: str | None = None,
                 check_result: tuple[bool, str] | None = None) -> np.ndarray:
    canvas = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ref_x, ref_y = ref

    if zones:
        draw_shape_zones(canvas, zones, target_shape)

    # 목표 위치
    cv2.circle(canvas, (int(ref_x), int(ref_y)), int(TOLERANCE_OK), GREEN, 2, cv2.LINE_AA)
    cv2.circle(canvas, (int(ref_x), int(ref_y)), int(TOLERANCE_WARN), AMBER, 1, cv2.LINE_AA)
    cv2.drawMarker(canvas, (int(ref_x), int(ref_y)), GREEN, cv2.MARKER_CROSS, 26, 2, cv2.LINE_AA)

    # OpenCV 기본 폰트는 한글을 못 그리므로 오버레이 문구는 영문으로 둔다.
    lines: list[tuple[str, tuple]] = []
    if found is None:
        lines.append(("BLOCK NOT FOUND - place it on the board", RED))
    else:
        dx, dy = found["cx"] - ref_x, found["cy"] - ref_y
        dist = float(np.hypot(dx, dy))
        color = GREEN if dist <= TOLERANCE_OK else (AMBER if dist <= TOLERANCE_WARN else RED)

        cv2.rectangle(canvas, (found["x"], found["y"]),
                      (found["x"] + found["w"], found["y"] + found["h"]), color, 2, cv2.LINE_AA)
        cv2.drawMarker(canvas, (int(found["cx"]), int(found["cy"])), color,
                       cv2.MARKER_TILTED_CROSS, 22, 2, cv2.LINE_AA)

        if dist > TOLERANCE_OK:      # 목표로 되돌릴 방향
            cv2.arrowedLine(canvas, (int(found["cx"]), int(found["cy"])),
                            (int(ref_x), int(ref_y)), WHITE, 6, cv2.LINE_AA, tipLength=.28)
            cv2.arrowedLine(canvas, (int(found["cx"]), int(found["cy"])),
                            (int(ref_x), int(ref_y)), color, 3, cv2.LINE_AA, tipLength=.28)

        state = "OK" if dist <= TOLERANCE_OK else ("ADJUST" if dist <= TOLERANCE_WARN else "OFF")
        mm = np.hypot(dx * MM_PER_PX[0], dy * MM_PER_PX[1])
        lines.append((f"{state}   offset {dist:5.1f} px  (~{mm:4.1f} mm)", color))
        # 화면 좌표계: x가 크면 오른쪽, y가 크면 아래쪽. 어긋난 반대 방향으로 옮기면 된다.
        move_x = "LEFT " if dx > 0 else "RIGHT"
        move_y = "UP   " if dy > 0 else "DOWN "
        lines.append((f"move {move_x} {abs(dx) * MM_PER_PX[0]:4.1f} mm    "
                      f"move {move_y} {abs(dy) * MM_PER_PX[1]:4.1f} mm", WHITE))

    lines.append((f"camera edge x={edge_x} (ref {REFERENCE_EDGE_X}, shift {shift:+.0f}px)", CYAN))
    if check_result is not None:
        ok, msg = check_result
        # 블록 배치 상태(위 lines)와는 별개 줄 — 도형 그리기 자체를 검사한 결과다.
        lines.append((f"SHAPE CHECK: {msg}", GREEN if ok else RED))
    key_hint = "q quit   s snapshot   r re-measure camera"
    if target_shape:
        key_hint += "   c check drawn shape"
    lines.append((key_hint, (190, 190, 190)))

    panel = np.full((26 * len(lines) + 18, canvas.shape[1], 3), 28, np.uint8)
    for i, (text, color) in enumerate(lines):
        cv2.putText(panel, text, (14, 26 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, .62, color, 2, cv2.LINE_AA)

    # 블럭 주변 확대 인셋 — 미세 조정할 때 눈으로 보이게
    half = 130
    x0, y0 = int(ref_x) - half, int(ref_y) - half
    x1, y1 = x0 + 2 * half, y0 + 2 * half
    inset = canvas[max(0, y0):y1, max(0, x0):x1]
    if inset.size:
        inset = cv2.resize(inset, (360, 360), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(inset, (0, 0), (359, 359), (90, 90, 90), 2)
        cv2.putText(inset, "ZOOM x2.8", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, WHITE, 1, cv2.LINE_AA)
        canvas[10:370, canvas.shape[1] - 370:canvas.shape[1] - 10] = inset

    return cv2.vconcat([panel, canvas])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref-cx", type=float, default=REFERENCE_CX)
    parser.add_argument("--ref-cy", type=float, default=REFERENCE_CY)
    parser.add_argument("--env-file", type=pathlib.Path,
                        default=REPO_ROOT / "configs" / "recording.env")
    parser.add_argument("--snapshot-dir", type=pathlib.Path,
                        default=REPO_ROOT / "outputs" / "analysis" / "block_alignment")
    parser.add_argument("--shape-zones", choices=("none", *SHAPE_ZONES), default="none",
                        help="도형을 그릴 대표 위치를 같이 표시한다. 기본은 꺼짐(none) — "
                             "안 켜면 기존 동작과 동일. 135/315=6개(상·하 x 좌·중·우), "
                             "132=4개(상·하 x 좌·우, 이 학습셋엔 좌측 열이 없다). "
                             "평가할 체크포인트를 학습시킨 데이터셋에 맞춰 고를 것")
    parser.add_argument("--target-shape", choices=("circle", "triangle", "rectangle"), default=None,
                        help="박스 크기를 이 도형 실측값으로 바꾼다(박스만 그린다). "
                             "315 zone-set만 지원(135는 도형별 원본 데이터가 없음). "
                             "주면 c 키로 그려진 도형을 실제 채점 파이프라인으로 검사할 수 있다")
    parser.add_argument("--size-source", choices=("per-zone", "global"), default="per-zone",
                        help="--target-shape 박스 크기를 어디서 가져올지. per-zone(기본, 기존 동작)="
                             "zone마다 다른 median(표본 10~22개). global=6개 zone 다 합친 도형별 "
                             "median(표본 105개, zone별 세션 편차에 덜 흔들림) — 2026-08-21 실측: "
                             "일부 zone×도형 조합은 표본이 적어 특정 세션이 median을 크게 흔든다")
    parser.add_argument("--board", type=int, nargs=4, default=list(M.DEFAULT_BOARD),
                        help="c 키 검사에 쓸 보드 bbox (x y w h)")
    parser.add_argument(
        "--exclude", action="append", default=None,
        type=lambda s: tuple(int(v) for v in s.split(",")),
        metavar="X,Y,W,H", help="c 키 검사에서 뺄 구역. 기본값은 erase_check.py와 동일",
    )
    parser.add_argument("--dark-ratio", type=float, default=0.72, help="c 키 검사용")
    args = parser.parse_args(argv)
    args.exclude = M.resolve_exclude(args.exclude)
    zones = SHAPE_ZONES.get(args.shape_zones, ())
    if args.target_shape:
        if args.shape_zones in SHAPE_SIZE_GLOBAL and args.size_source == "global":
            gw, gh = SHAPE_SIZE_GLOBAL[args.shape_zones][args.target_shape]
            zones = tuple((label, cx, cy, gw, gh) for label, cx, cy, _, _ in zones)
        elif args.shape_zones in SHAPE_SIZE_PER_ZONE:
            sizes = SHAPE_SIZE_PER_ZONE[args.shape_zones][args.target_shape]
            zones = tuple(
                (label, cx, cy, *sizes[label]) for label, cx, cy, _, _ in zones
            )
        else:  # 135 — 도형 라벨이 붙은 원본 데이터가 없어 도형별 크기를 못 잰다
            print(f"[WARN] --target-shape는 {args.shape_zones} zone-set을 지원하지 "
                  "않습니다 (도형별 원본 데이터 없음) — pooled 박스 크기를 그대로 씁니다",
                  file=sys.stderr)

    from piper_human_approved_inference import load_env_file
    load_env_file(args.env_file.expanduser().resolve())

    from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

    serial = os.environ.get("TOP_CAM", "")
    if not serial:
        print("[ERROR] TOP_CAM이 비어 있습니다 — recording.env를 확인하세요", file=sys.stderr)
        return 2

    print(f"[CAM] TOP_CAM={serial} 연결 중…")
    camera = RealSenseCamera(RealSenseCameraConfig(
        serial_number_or_name=serial,
        width=int(os.environ.get("CAM_WIDTH", "1280")),
        height=int(os.environ.get("CAM_HEIGHT", "720")),
        fps=int(os.environ.get("FPS", "30")),
        use_depth=False,
        warmup_s=float(os.environ.get("REALSENSE_WARMUP_S", "3.0")),
    ))
    connect_camera_with_retry(camera)
    print(f"[REF] 목표 블럭 중심 cx={args.ref_cx:.1f} cy={args.ref_cy:.1f} "
          f"(학습 표준편차 {REFERENCE_SD[0]}/{REFERENCE_SD[1]}px)")

    window = "Block alignment  |  green = target,  arrow = move this way"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 860)

    edge_x, shift = None, 0.0
    check_result: tuple[bool, str] | None = None
    try:
        while True:
            frame = camera.async_read(timeout_ms=2000)
            if edge_x is None:
                edge_x = measure_edge_x(frame)
                shift = float(REFERENCE_EDGE_X - edge_x)
                print(f"[ALIGN] camera edge x={edge_x} → shift {shift:+.0f}px")
            if abs(shift) >= 0.5:
                frame = cv2.warpAffine(frame, np.float32([[1, 0, shift], [0, 1, 0]]),
                                       (frame.shape[1], frame.shape[0]),
                                       borderMode=cv2.BORDER_REPLICATE)

            found = detect_block(frame)
            canvas = draw_overlay(frame, found, (args.ref_cx, args.ref_cy), edge_x, shift, zones,
                                 args.target_shape, check_result)
            cv2.imshow(window, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                edge_x = None
            if key == ord("c"):
                if not args.target_shape:
                    print("[WARN] c 키는 --target-shape가 있어야 동작합니다", file=sys.stderr)
                else:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    check_result = check_drawn_shape(frame_bgr, tuple(args.board), args.dark_ratio,
                                                     args.exclude, args.target_shape, zones)
                    print(f"[CHECK] {check_result[1]}")
            if key == ord("s"):
                args.snapshot_dir.mkdir(parents=True, exist_ok=True)
                path = args.snapshot_dir / f"block_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                print(f"[SAVE] {path}")
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        camera.disconnect()
        print("[DONE] 카메라 해제")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
