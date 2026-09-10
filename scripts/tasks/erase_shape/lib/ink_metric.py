#!/usr/bin/env python3
"""top 카메라 영상에서 보드 위 도형별 잉크 잔량(erase completion)을 계산한다.

전제: 검은 마커 / 흰 보드. 도형은 매 에피소드 손으로 다시 그려지므로 위치·종류가
고정이 아니다 → 첫 프레임에서 도형을 자동 검출하고 원/삼각형/사각형으로 분류한 뒤,
각 도형 bbox 안의 잉크 픽셀 비율을 시간축으로 추적한다.

가림 처리 — 용도에 따라 추정기가 두 개다:
- track()/summarize(): 도형 영역이 가려진 프레임은 통째로 버리고 남은 프레임에 단조 감소
  envelope을 적용(지우기는 잉크를 늘릴 수 없다). 가장 견고해서 **성공 판정·QC용**.
- dense_progress(): 가려진 "픽셀"만 빼고 보이는 영역의 잉크 밀도를 잰다. 접촉 중에도
  관측이 되지만 분모가 흔들려 잡음이 크다. **진행도 곡선 전용**, 판정에 쓰지 말 것.

사용:
  python ink_metric.py <episode_dir> --dump-roi /tmp/roi.png   # 검출 결과 눈으로 확인
  python ink_metric.py '0802/0802/erase_the_circle_*' --csv out.csv
"""

import argparse
import csv
import glob
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# 2026-08 세팅(1280x720, 보드가 화면 좌측 세로) 기준 보드 "안쪽" 영역(테두리 프레임 제외).
# 카메라 옮기면 --board로 덮어쓸 것.
DEFAULT_BOARD = (228, 0, 515, 720)
MIN_SHAPE_AREA = 2000  # bbox 면적 하한 (먼지·자국 제거)
MAX_ASPECT = 4.0  # 보드 테두리·케이블 같은 길쭉한 것 제외
MAX_SPAN = 0.7  # 보드 한 변의 이 비율을 넘게 뻗으면 도형이 아님
MAX_SAT = 25  # 도형 판정용 상한 — contour bbox의 **평균** 채도. 나무 블록(평균 84)은
# 걸러지고 마커 도형(선+배경 평균이라 낮음)은 통과한다. 여기는 25로 충분하다.
#
# 픽셀 단위 잉크 계산에는 이 값을 쓰면 안 된다. 진한 마커 선은 픽셀 경계의 색수차로
# 채도가 26~49까지 올라가는데(실측 중앙값 26 — 25 바로 위), 지우개로 문질러 흐려진
# 잔여물은 채도가 낮아 전부 통과한다. 그래서 25로 자르면 **분모(지우기 전)만 절반으로
# 깎이고 분자(지운 후)는 그대로** 남아 "덜 지웠다"로 심하게 편향된다.
# 실측 2026-08-18: 실제 91.8% 지운 롤아웃이 77.8%로 채점됐다(BEFORE 550px 중 269px만
# 통과, AFTER는 80px 전부 통과). 시연 데이터에서도 마커 선 통과율이 27~91%로 들쭉날쭉했다.
#
# 45면 마커는 99~100% 살리면서(시연 6개 89~100%) 나무 블록은 12%만 통과한다 —
# 나무는 채도 p10=39/중앙값 84라 대부분 걸린다.
INK_MAX_SAT = 45
MERGE_GAP = 30  # 이 픽셀 이내로 붙어있는 조각은 한 도형으로 병합 (도형 간 간격은 100px 이상)
BORDER_MARGIN = 3  # 보드 오른쪽 경계에 이만큼 이내로 닿은 덩어리는 로봇 팔로 본다 (아래 참고)
OCCLUDER_KERNEL = 11  # 이 크기로 opening해서 남는 어두운 덩어리 = 팔·그리퍼 (얇은 잉크 획은 사라짐)
# 잉크 계산에서 팔을 뺄 때 쓰는 커널. OCCLUDER_KERNEL(11)을 그대로 쓰면 안 된다 —
# 그건 '굵은 팔뚝'을 잡으려고 큰 값인데, 도형 위를 지나는 팔 끝/케이블은 실측 약 6px
# 이라 11로 opening하면 통째로 사라져 blob이 비고 배제가 아예 안 걸린다(2026-08-18
# 실측: kernel 7~15 전부 blob=0). 5면 그 팔은 잡히고(잉크 0.01018 -> 0.00248,
# 실제 보드 0.0023과 일치) 시연 6개 첫 프레임의 진짜 잉크 손실은 0.0%였다.
ARM_KERNEL = 5
OCCL_JUMP = 0.15  # 비-흰색 비율이 이만큼 급증하면 가림 (판정용 track에서 사용)
MIN_VISIBLE = 0.70  # dense_progress: 도형 영역이 이만큼 안 보이면 그 프레임은 측정 불가
PAD = 12  # 도형 bbox 여유

# 보드에 눌어붙은 변색 테이프+카드 — 위치가 고정된 이물질이라 기본으로 늘 뺀다.
# 2026-08-18 실물에서 확인: 이 자리와 겹쳐 그려진 도형이 완전히 지워졌는데도
# rectangle로 오검출되어 '남음'으로 오판했다. 도형이 실제로 이 위치에 그려지면
# 그 프레임도 같이 안 잡힌다는 뜻이니, 카메라를 옮기거나 이 이물질을 실제로
# 떼어내면 여기를 비우거나 좌표를 다시 잴 것.
DEFAULT_EXCLUDE = [(370, 260, 150, 110)]
# 컨투어 bbox가 배제 사각형과 이만큼 겹치면 통째로 버린다 (detect_shapes 참고).
EXCLUDE_OVERLAP = 0.5

# 성공 판정 임계 (make_done_labels.py, erase_check.py가 공유)
SUCCESS_ERASED = 0.9  # target이 이만큼 지워지면 성공
MAX_DISTRACTOR = 0.10  # distractor를 이만큼 넘게 건드리면 실패.
# 노이즈 플로어 실측: 120 에피소드에서 평균 0.009 / p95 0.037 / 최대 0.097.
# 0.10은 플로어의 약 3배 — 이보다 조이면 정상 시연이 오탐으로 걸린다.


def detect_shapes(frame, board, dark_ratio, exclude=None):
    """첫 프레임에서 도형 bbox와 종류를 찾는다. -> [(label, (x,y,w,h)), ...]

    exclude: [(x,y,w,h), ...] 전역(원본 프레임) 좌표. 이 사각형과 EXCLUDE_OVERLAP
    이상 겹치는 **컨투어를 통째로 버린다**(마스크를 지우지 않는다 — 아래 참고).
    손으로 그린 도형(위치가 에피소드마다 바뀜)과 달리, 보드에 눌어붙은 테이프
    자국·스티커처럼 **위치가 고정된 이물질**을 배제할 때 쓴다.
    실제로 이런 이물질이 도형 자리와 겹쳐서 다 지운 도형을 '남음'으로 오판한 사례가
    있었다(2026-08-18, 보드에 붙은 변색 테이프+카드가 지워진 삼각형 자리를 rectangle로
    오검출). 위치가 안 바뀌는 게 확인되면 --exclude로 좌표를 넘긴다.
    """
    bx, by, bw, bh = board
    sub = frame[by : by + bh, bx : bx + bw]
    roi = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)[:, :, 1]
    white = float(np.percentile(roi, 90))
    ink = (roi < white * dark_ratio).astype(np.uint8)
    # 선이 끊겨 있어도 한 도형으로 묶이도록 닫기
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 배제 구역은 마스크를 0으로 지우는 게 아니라 **컨투어를 통째로 버리는** 방식이다.
    # 마스크를 지우면 배제 구역에 걸친 물체가 반으로 쪼개지고, 남은 조각이 원래
    # 물체와 다른 성질을 갖게 되어 필터를 통과해버린다. 실제로 그랬다
    # (2026-08-18 저녁): 지우개 나무 거치대는 유채색이라 채도 필터가 정상적으로
    # 걸러내는데, 배제 사각형이 거치대를 반으로 잘라서 남은 흰 조각이 무채색으로
    # 보였고 그게 가짜 'triangle'로 잡혔다 — 배제를 아예 끄면 안 나오던 도형이
    # 배제 때문에 생긴 셈이다.
    def not_excluded(c):
        x, y, w, h = cv2.boundingRect(c)
        area = float(w * h) or 1.0
        for ex, ey, ew, eh in exclude or []:
            # 전역 좌표 -> 보드 로컬
            ix0, iy0 = max(x, ex - bx), max(y, ey - by)
            ix1, iy1 = min(x + w, ex - bx + ew), min(y + h, ey - by + eh)
            if ix1 > ix0 and iy1 > iy0 and (ix1 - ix0) * (iy1 - iy0) / area >= EXCLUDE_OVERLAP:
                return False
        return True

    contours = [c for c in contours if not_excluded(c)]
    # 유채색 물체(나무 지우개 블록)는 병합 "전에" 버린다.
    # 나중에 버리면 블록이 옆 도형과 한 덩어리로 묶여 bbox와 잉크 측정이 오염된다.
    def achromatic(c):
        x, y, w, h = cv2.boundingRect(c)
        return float(sat[y : y + h, x : x + w].mean()) <= MAX_SAT

    # 로봇 팔·그리퍼는 검은 무채색이라 채도 필터에 안 걸린다. 대신 위치로 가른다 —
    # 팔은 화면 오른쪽 로봇 몸체와 이어져 있어서 그 덩어리가 **보드 오른쪽 경계에
    # 반드시 닿는다**. 손으로 그린 도형은 보드 안에 갇혀 있다.
    #
    # 보드 폭을 줄여서 팔을 잘라내는 방법은 못 쓴다. 실측(6개 데이터셋 첫 프레임)에서
    # 팔은 보드 로컬 x=405부터 파고드는데 실제 도형은 x+w=465까지 뻗어서 구간이 겹친다.
    # 폭으로 자르면 0813 같은 데이터의 진짜 도형이 잘린다.
    #
    # 병합 "전에" 버려야 한다. 팔과 도형이 MERGE_GAP(30px) 안으로 붙는 경우가 있어서
    # (0804: 팔 x=420 / 도형 x+w=413) 나중에 버리면 둘이 한 덩어리로 묶여버린다.
    def not_arm(c):
        x, y, w, h = cv2.boundingRect(c)
        return x + w < bw - BORDER_MARGIN

    contours = merge_nearby([c for c in contours if achromatic(c) and not_arm(c)])
    # 병합 결과가 경계에 닿았다면 놓친 팔 조각이 섞인 것이므로 한 번 더 거른다
    contours = [c for c in contours if not_arm(c)]
    shapes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < MIN_SHAPE_AREA:
            continue
        if max(w / h, h / w) > MAX_ASPECT or w > MAX_SPAN * bw or h > MAX_SPAN * bh:
            continue
        # PAD를 붙이되 프레임 밖으로 나가지 않게 자른다 (빈 crop 방지)
        fh, fw = frame.shape[:2]
        x0, y0 = max(x + bx - PAD, 0), max(y + by - PAD, 0)
        x1, y1 = min(x + bx + w + PAD, fw), min(y + by + h + PAD, fh)
        shapes.append((classify(c), (x0, y0, x1 - x0, y1 - y0)))
    shapes.sort(key=lambda s: s[1][1])  # 화면 위->아래
    return shapes, white


def merge_nearby(contours, gap=MERGE_GAP):
    """가까운 contour들을 한 도형으로 묶는다.

    손으로 그린 도형은 획이 끊겨서 하나의 도형이 여러 조각으로 잡힌다. 조각난 채로
    분류하면 삼각형이 원으로 오분류된다(hull이 도형 전체를 못 덮음). bbox가 gap 이내로
    가까운 조각들을 union-find로 묶은 뒤 점들을 합쳐서 하나의 contour로 취급한다.
    """
    boxes = [cv2.boundingRect(c) for c in contours]
    parent = list(range(len(contours)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, (x1, y1, w1, h1) in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            x2, y2, w2, h2 = boxes[j]
            dx = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
            dy = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
            if dx <= gap and dy <= gap:
                parent[find(i)] = find(j)

    groups = {}
    for i, c in enumerate(contours):
        groups.setdefault(find(i), []).append(c)
    return [np.vstack(g) for g in groups.values()]


def classify(contour):
    """convex hull 면적 / 최소외접사각형 면적(extent)으로 circle/triangle/rectangle 판정.

    손으로 그린 도형은 선이 떨려서 꼭짓점 개수(approxPolyDP)가 불안정하다. 반면
    extent는 이론값이 삼각형 0.50 / 원 0.785 / 사각형 1.0으로 잘 갈린다.
    """
    hull = cv2.convexHull(contour)
    rect_area = cv2.minAreaRect(contour)[1]
    rect_area = rect_area[0] * rect_area[1]
    if rect_area <= 0:
        return "unknown"
    extent = cv2.contourArea(hull) / rect_area
    if extent < 0.63:
        return "triangle"
    if extent > 0.88:
        return "rectangle"
    # 0.63~0.88은 원(0.785)과 "모서리가 둥근 손그림 사각형"이 겹치는 구간이다.
    # hull을 다각형 근사했을 때 꼭짓점이 적으면 사각형 쪽으로 본다.
    n = len(cv2.approxPolyDP(hull, 0.03 * cv2.arcLength(hull, True), True))
    return "rectangle" if n <= 4 else "circle"


def _video_codec(video):
    """PyAV로 코덱 이름만 싸게 확인한다(디코딩 안 함). 못 열면 None."""
    try:
        import av
    except ImportError:
        return None
    try:
        with av.open(str(video)) as c:
            return c.streams.video[0].codec_context.name
    except Exception:
        return None


# OpenCV가 이 환경에서 실제로 디코딩하는 코덱만 화이트리스트로 둔다.
# AV1을 cv2에 넘기면 안 된다 — 디코더가 없어서 실패할 뿐 아니라, 하드웨어 디코더를
# 탐색하는 과정에서 간헐적으로 프로세스가 통째로 죽는다(2026-08-18 실물: 평가 GUI가
# "Segmentation fault (core dumped)"로 종료). 로그도 프레임마다 av1 경고로 도배된다.
CV2_CODECS = {"h264", "hevc", "mpeg4", "mjpeg", "vp8", "vp9"}


def read_frames(video):
    """영상을 BGR 프레임으로 훑는다. 코덱을 먼저 보고 리더를 고른다.

    러너(piper_infer_runner)의 증강 녹화는 AV1(SVT-AV1)이고 이 환경 OpenCV는 AV1
    디코더가 없다. 예전엔 일단 cv2로 열어보고 실패하면 PyAV로 넘어갔는데, 그
    "일단 열어보는" 것 자체가 위험했다(위 CV2_CODECS 주석 참고). 그래서 지금은
    PyAV로 코덱 이름만 확인한 뒤 cv2가 확실히 읽는 코덱일 때만 cv2를 쓴다.
    기존 시연 데이터는 hevc라 그대로 cv2 경로를 탄다.
    """
    codec = _video_codec(video)
    if codec is None or codec not in CV2_CODECS:
        try:
            import av
        except ImportError:
            raise RuntimeError(
                f"영상을 읽을 수 없다: {video} (코덱={codec}). OpenCV가 못 읽는 "
                f"코덱인데 PyAV도 없다 — pip install av"
            ) from None
        with av.open(str(video)) as container:
            for f in container.decode(video=0):
                yield f.to_ndarray(format="bgr24")
        return

    cap = cv2.VideoCapture(str(video))
    try:
        ok, frame = cap.read()
        while ok:
            yield frame
            ok, frame = cap.read()
    finally:
        cap.release()


def _first_and_rest(video):
    """첫 프레임과, **첫 프레임을 포함한** 전체 이터레이터를 함께 준다.

    첫 프레임은 도형 검출(기준 잉크량)에 쓰고, 시계열에도 그대로 들어가야 한다 —
    빼면 t=0의 잉크량이 사라져 erased_frac의 분모가 한 프레임 밀린다.
    """
    it = read_frames(video)
    try:
        first = next(it)
    except StopIteration:
        raise RuntimeError(f"영상 첫 프레임 읽기 실패: {video}") from None
    return first, itertools.chain([first], it)


def track(video, board, dark_ratio, occl_jump=OCCL_JUMP, exclude=None):
    first, frames = _first_and_rest(video)

    detected, white = detect_shapes(first, board, dark_ratio, exclude)
    # 같은 종류가 여러 개 나올 수 있으므로 키를 유일하게 만든다 (circle#0, circle#1 ...)
    shapes = [(f"{lbl}#{i}", box) for i, (lbl, box) in enumerate(detected)]
    ink_thr = white * dark_ratio

    white_thr = white * 0.88  # 이보다 어두우면 보드 흰색이 아님(잉크·팔·지우개 블록)

    ink = {k: [] for k, _ in shapes}
    occ = {k: [] for k, _ in shapes}
    base_nonwhite = {}

    for frame in frames:
        for lbl, (x, y, w, h) in shapes:
            sub = frame[y : y + h, x : x + w]
            g = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
            # 잉크 계산에서 유채색(나무 지우개 블록)을 뺀다 — detect_shapes와
            # erase_check._ink가 쓰는 것과 같은 필터다. 안 걸면 지우개가 도형 위에
            # 놓인 프레임에서 잉크가 늘어난 것으로 잡혀 '다 지웠는데 실패'가 된다.
            # occ(가림) 판정에는 일부러 안 건다 — 가림 검출은 블록이 보여야 한다.
            sat = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)[:, :, 1]
            dark = g < ink_thr
            # 검은 무채색 덩어리(팔·그리퍼·케이블)는 채도 필터를 통과해버린다.
            # dense_progress가 쓰는 것과 같은 opening으로 뺀다 — 얇은 잉크 획은
            # 사라지고 굵은 덩어리만 남으므로 그 부분을 잉크에서 제외한다.
            # 실측 2026-08-18: --stop-on-release로 놓는 순간 녹화가 끝나면 마지막
            # 프레임에 팔이 보드를 가로질러 있고, 그게 잉크로 세어져 실제 92.2%
            # 지운 롤아웃이 65.4%로 채점됐다.
            blob = cv2.morphologyEx(
                dark.astype(np.uint8), cv2.MORPH_OPEN,
                np.ones((ARM_KERNEL, ARM_KERNEL), np.uint8),
            ).astype(bool)
            ink[lbl].append(float((dark & (sat <= INK_MAX_SAT) & ~blob).mean()))
            nw = float((g < white_thr).mean())
            base_nonwhite.setdefault(lbl, nw)
            occ[lbl].append(nw > base_nonwhite[lbl] + occl_jump)
    return shapes, ink, occ


def dense_progress(video, board, dark_ratio=0.72, min_visible=MIN_VISIBLE, exclude=None):
    """접촉 중에도 쓸 수 있는 조밀한 진행도 시계열. -> {label: (density, valid)}

    track()의 잔량 측정은 도형 영역이 조금이라도 가려지면 프레임을 통째로 버린다.
    에피소드 최종 판정에는 그게 맞다(가장 견고함). 하지만 팔이 들어와 있어도 도형
    면적의 2/3는 여전히 보이므로, 진행도 곡선용으로는 가려진 "픽셀"만 빼고 보이는
    영역의 잉크 밀도를 재는 편이 낫다 — 접촉 구간의 관측 가능 프레임이 0%에서
    약 58%로 올라간다.

    주의: 분모(가시 픽셀 수)가 프레임마다 달라져 잡음이 크다. 절대값 비교나 성공
    판정에는 쓰지 말고(그 용도로 쓰면 distractor 노이즈가 0.05 -> 0.20으로 뛴다)
    추세·진행도 용도로만 쓸 것.
    """
    first, frames = _first_and_rest(video)
    detected, white = detect_shapes(first, board, dark_ratio, exclude)
    shapes = [(f"{lbl}#{i}", box) for i, (lbl, box) in enumerate(detected)]
    ink_thr = white * dark_ratio
    kernel = np.ones((OCCLUDER_KERNEL, OCCLUDER_KERNEL), np.uint8)

    dens = {k: [] for k, _ in shapes}
    valid = {k: [] for k, _ in shapes}
    for frame in frames:
        for lbl, (x, y, w, h) in shapes:
            sub = frame[y : y + h, x : x + w]
            g = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
            sat = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)[:, :, 1]
            dark = g < ink_thr
            # 얇은 잉크 획은 opening으로 사라지고 팔·그리퍼 같은 굵은 덩어리만 남는다
            blob = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
            visible = ~(blob | (sat > INK_MAX_SAT))
            n_vis = int(visible.sum())
            dens[lbl].append(float((dark & visible).sum() / n_vis) if n_vis else 0.0)
            valid[lbl].append(float(visible.mean()) >= min_visible)
    return {k: (np.array(dens[k]), np.array(valid[k])) for k, _ in shapes}


def summarize(ink, occ):
    out = {}
    for lbl, vals in ink.items():
        v = np.asarray(vals, float)
        valid = ~np.asarray(occ[lbl])
        if valid.sum() < 2:
            out[lbl] = {"error": "유효 프레임 부족 — 도형이 거의 항상 가려짐"}
            continue
        # 가려진 프레임은 직전 유효값으로 채운 뒤 단조 감소 envelope
        filled, last = v.copy(), v[np.argmax(valid)]
        for i in range(len(filled)):
            if valid[i]:
                last = filled[i]
            else:
                filled[i] = last
        env = np.minimum.accumulate(filled)

        # 판정(erased_frac)에는 envelope이 아니라 "처음/마지막 유효 관측값"을 쓴다.
        # envelope의 running-min은 중간에 스쳐간 그림자 같은 일시적 저점을 그대로
        # 붙잡아서 distractor 노이즈를 키운다 (실측: 평균 0.044 -> 0.009, 분리 22배 -> 110배).
        # env는 진행도 곡선·완료 시점(frames_to_90pct)용으로만 남긴다.
        out[lbl] = {
            "env": env,
            "v_first": float(v[valid][0]),
            "v_last": float(v[valid][-1]),
            "occluded_frac": round(float(np.mean(occ[lbl])), 3),
        }
    return out


def aggregate(per_shape, label):
    """같은 종류 도형이 여러 개 검출된 경우(선 끊김으로 분할 등) 잉크를 합쳐서 하나로 본다."""
    parts = [v for k, v in per_shape.items() if k.split("#")[0] == label and "env" in v]
    if not parts:
        return None
    envs = [p["env"] for p in parts]
    n = min(len(e) for e in envs)
    env = np.sum([e[:n] for e in envs], axis=0)
    init = sum(p["v_first"] for p in parts)
    final = sum(p["v_last"] for p in parts)
    erased = 0.0 if init <= 0 else max(0.0, min(1.0, (init - final) / init))
    hit = np.nonzero(env <= env[0] * 0.1)[0]
    return {
        "n_blobs": len(envs),
        "ink_init": round(init, 5),
        "ink_final": round(final, 5),
        "erased_frac": round(erased, 4),
        "frames_to_90pct": int(hit[0]) if len(hit) else None,
        "n_frames": n,
    }


# ── 잔여 잉크의 양과 위치 ──────────────────────────────────────
# 설계 문서 §4.6-③. erased_frac은 "얼마나 지웠나"만 말해주고 "어디가 남았나"는
# 말해주지 않는다. 재시도를 몇 번 해도 같은 자리가 남는다면 그건 정책이 못 닿는
# 자리라는 뜻이고, 그건 리커버리 데이터를 어디에 모을지를 정하는 근거가 된다.
#
# 주의 — 이 값을 정책에 프롬프트로 넣을 통로는 아직 없다(§4.6-③ 표 참고).
# 학습 데이터에 방향 어휘가 없어서 "왼쪽 아래를 지워라"는 분포 밖이다.
# 그래서 이건 **진단·라벨링용**이지 제어 신호가 아니다.

# 3x3 격자 이름 — 사람이 로그를 읽고 바로 어디인지 알아야 해서 좌표 대신 말로 낸다
_ROW_NAMES = ("위", "중간", "아래")
_COL_NAMES = ("왼쪽", "가운데", "오른쪽")


def residual(frame, box, ink_thr, grid=3):
    """bbox 안에 남은 잉크의 양·무게중심·격자 분포. -> dict

    frame은 BGR, box는 (x, y, w, h), ink_thr은 set_reference가 잡은 절대 임계값.
    임계값을 인자로 받는 이유: 시도 전 프레임에서 정한 값을 그대로 써야
    전/후가 같은 자로 재진다(EraseChecker.reference["ink_thr"]와 같은 근거).
    """
    x, y, w, h = box
    sub = frame[y : y + h, x : x + w]
    g = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    # ★ 채도 필터를 여기서도 걸어야 한다. detect_shapes는 나무 지우개 블록(채도 ≈45)을
    #   걸러내는데 residual()은 처음에 그냥 어두운 픽셀을 셌다. 실물 검증에서 정책이
    #   지우개를 삼각형 위에 올려놓고 끝냈고, 그 프레임에서 잔여 잉크가 기준(3.4%)보다
    #   많은 15.2%로 나왔다 — 블록을 잉크로 센 것이다. 검은 마커는 채도 ≈10이라
    #   안 걸린다.
    chromatic = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)[:, :, 1] > INK_MAX_SAT
    mask = (g < ink_thr) & ~chromatic
    n = int(mask.sum())
    if n == 0:
        return {"ink_frac": 0.0, "centroid": None, "where": None, "grid": None,
                "spread_px": 0.0, "occluded_frac": round(float(chromatic.mean()), 4)}

    ys, xs = np.nonzero(mask)
    cx, cy = float(xs.mean()), float(ys.mean())

    # 격자 분포 — 어느 칸에 몰려 있는지. 셀별 "그 셀 픽셀 중 잉크 비율"이 아니라
    # "전체 잔여 잉크 중 이 셀이 차지하는 비율"이다. 어디가 남았는지를 묻고 있으므로.
    gy = np.clip((ys * grid // max(h, 1)), 0, grid - 1)
    gx = np.clip((xs * grid // max(w, 1)), 0, grid - 1)
    cells = np.zeros((grid, grid), float)
    np.add.at(cells, (gy, gx), 1.0)
    cells /= n

    # 이름표는 격자 최대칸이 아니라 **무게중심**에서 뽑는다.
    # 최대칸을 쓰면 속이 빈 도형(원·사각형 테두리)에서 가운데 칸이 0이 되고 테두리
    # 셀들이 거의 동점이라, argmax가 그중 아무 칸이나 집어 "위 가운데" 같은 엉뚱한
    # 이름이 나온다. 무게중심은 그런 경우 정직하게 가운데를 가리킨다(=고르게 남음).
    rx, ry = cx / max(w, 1), cy / max(h, 1)
    where = None
    if grid == 3:  # 이름표는 3x3에서만 의미가 있다
        band = lambda v: 0 if v < 1 / 3 else (1 if v < 2 / 3 else 2)  # noqa: E731
        where = f"{_ROW_NAMES[band(ry)]} {_COL_NAMES[band(rx)]}"

    return {
        "ink_frac": round(n / mask.size, 5),
        # bbox 기준 상대 좌표 (0~1) — bbox 크기가 도형마다 달라서 절대 픽셀은 못 비교한다
        "centroid": (round(rx, 3), round(ry, 3)),
        "where": where,
        # float()으로 눕힌다 — np.float64가 섞이면 json.dumps가 죽는다
        "grid": [[round(float(v), 3) for v in row] for row in cells],
        # 흩어져 남았나(얼룩) 한 곳에 뭉쳐 남았나(안 닿은 구석) — 처방이 다르다
        "spread_px": round(float(np.hypot(xs - cx, ys - cy).mean()), 1),
        # bbox 안에서 유채색(지우개 블록·팔)이 덮은 비율. 이게 크면 잔여 추정이
        # 그만큼 가려진 상태로 나온 것이므로 값을 곧이곧대로 믿으면 안 된다.
        "occluded_frac": round(float(chromatic.mean()), 4),
    }


def ink_in_boxes(frame, boxes, board, dark_ratio, exclude=None):
    """주어진 bbox들의 잉크 비율 합. park 프레임으로 최종 잔량을 잴 때 쓴다.

    임계값(보드 흰색 기준)은 이 프레임에서 다시 잡는다 — 조명이 미세하게 달라도
    detect_shapes와 같은 방식으로 매번 새로 잡아야 같은 자로 잰 것이 된다.
    """
    _, white = detect_shapes(frame, board, dark_ratio, exclude)
    thr = white * dark_ratio
    total = 0.0
    for x, y, w, h in boxes:
        sub = frame[y : y + h, x : x + w]
        if sub.size == 0:
            return None
        g = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
        sat = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)[:, :, 1]
        total += float(((g < thr) & (sat <= INK_MAX_SAT)).mean())
    return total


def imread(path):
    """cv2 의존을 이 모듈에 묶어두기 위한 얇은 래퍼."""
    return cv2.imread(str(path))


def top_video(ep_dir: Path) -> Path:
    hits = sorted(ep_dir.glob("videos/observation.images.top/**/*.mp4"))
    if not hits:
        raise FileNotFoundError(f"top 영상 없음: {ep_dir}")
    return hits[0]


def task_of(ep_dir: Path) -> str:
    """폴더명에서 target 도형을 뽑는다 (erase_the_circle_0802-1305 -> circle)."""
    for s in ("circle", "triangle", "rectangle"):
        if s in ep_dir.name:
            return s
    return "?"


def dump_roi(ep_dir, board, dark_ratio, path, exclude=None):
    # read_frames를 거쳐야 AV1 롤아웃에서도 확인 이미지를 뽑을 수 있다.
    # 카메라 점검(§4-4)은 롤아웃 영상으로도 해야 하므로 여기가 막히면 안 된다.
    frame, _ = _first_and_rest(top_video(ep_dir))
    shapes, _ = detect_shapes(frame, board, dark_ratio, exclude)
    bx, by, bw, bh = board
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 0, 0), 1)
    for lbl, (x, y, w, h) in shapes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, lbl, (x, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite(str(path), frame)
    print(f"검출 {len(shapes)}개 저장: {path}")


def resolve_exclude(exclude):
    """CLI --exclude 값을 실제로 쓸 목록으로 바꾼다.

    안 넘기면(None) DEFAULT_EXCLUDE. --exclude 0,0,0,0을 명시하면(면적 0)
    "기본 배제를 끄겠다"는 뜻으로 보고 빈 목록을 돌려준다 — 이물질을 실제로
    떼어냈거나 카메라를 옮겼을 때 쓴다.
    """
    if exclude is None:
        return DEFAULT_EXCLUDE
    return [z for z in exclude if z[2] > 0 and z[3] > 0]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("path", help="LeRobotDataset 에피소드 디렉터리 (glob 가능)")
    p.add_argument("--board", type=lambda s: tuple(int(v) for v in s.split(",")), default=DEFAULT_BOARD)
    p.add_argument("--dark-ratio", type=float, default=0.72, help="보드 흰색 대비 잉크 판정 임계 비율")
    p.add_argument("--occl-jump", type=float, default=OCCL_JUMP, help="비-흰색 비율이 이만큼 급증하면 가림 처리")
    p.add_argument("--csv", type=Path)
    p.add_argument("--dump-roi", type=Path, help="첫 에피소드 검출 결과를 그려서 저장 후 종료")
    p.add_argument(
        "--exclude", action="append", default=None,
        type=lambda s: tuple(int(v) for v in s.split(",")),
        metavar="X,Y,W,H",
        help="이 전역좌표 사각형은 도형 검출에서 뺀다 (여러 번 줄 수 있음). "
             f"기본값은 보드에 눌어붙은 테이프 자국 자리({DEFAULT_EXCLUDE[0]}) — "
             "떼어냈거나 카메라를 옮겼으면 --exclude 0,0,0,0으로 비울 것",
    )
    a = p.parse_args(argv)
    a.exclude = resolve_exclude(a.exclude)

    dirs = [Path(d) for d in sorted(glob.glob(a.path)) if Path(d).is_dir()]
    if not dirs:
        p.error(f"에피소드 디렉터리를 찾지 못함: {a.path}")

    if a.dump_roi:
        dump_roi(dirs[0], a.board, a.dark_ratio, a.dump_roi, a.exclude)
        return 0

    rows = []
    for d in dirs:
        target = task_of(d)
        shapes, ink, occ = track(top_video(d), a.board, a.dark_ratio, a.occl_jump, a.exclude)
        per_shape = summarize(ink, occ)
        labels = {k.split("#")[0] for k in per_shape}
        tgt = aggregate(per_shape, target) or {}
        others = [aggregate(per_shape, l) for l in labels if l != target]
        row = {
            "episode": d.name,
            "target": target,
            "n_shapes": len(shapes),
            "target_found": bool(tgt),
            "target_erased": tgt.get("erased_frac"),
            "target_frames_to_90pct": tgt.get("frames_to_90pct"),
            "n_frames": tgt.get("n_frames"),
            # distractor는 "안 지워져야" 정답 → 최대값이 낮을수록 좋다
            "max_distractor_erased": round(max((o["erased_frac"] for o in others if o), default=0.0), 4),
            "detected": ",".join(k for k, _ in shapes),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    ok = [r for r in rows if r["target_found"]]
    if ok:
        print(
            f"\n검출 성공 {len(ok)}/{len(rows)} | "
            f"target 평균 지움 {np.mean([r['target_erased'] for r in ok]):.3f} | "
            f"distractor 평균 지움 {np.mean([r['max_distractor_erased'] for r in ok]):.3f}"
        )
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"저장: {a.csv} ({len(rows)} episodes)")
    return 0


def _selftest():
    """합성 영상으로 검출·가림·envelope 검증."""
    import tempfile

    vid = Path(tempfile.mkdtemp()) / "t.mp4"
    board_bgr, ink_bgr, wood_bgr = (240, 240, 238), (60, 60, 55), (150, 190, 215)
    w = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 30, (400, 400))
    for i in range(60):
        f = np.full((400, 400, 3), board_bgr, np.uint8)
        cv2.circle(f, (100, 100), 40, ink_bgr, 3)  # target: 점점 지워짐
        if i >= 30:
            cv2.circle(f, (100, 100), 40, board_bgr, 8)
        cv2.rectangle(f, (250, 250), (330, 330), ink_bgr, 3)  # distractor: 그대로
        # distractor 2 — 도형이 3개일 때도 전부 잡히는지 (같은 종류 중복은 label#i로 갈린다)
        cv2.polylines(f, [np.array([[290, 70], [250, 140], [330, 140]])], True, ink_bgr, 3)
        cv2.rectangle(f, (60, 250), (150, 330), wood_bgr, -1)  # 나무 지우개 블록: 검출되면 안 됨
        # 로봇 팔 — 보드 오른쪽 경계에 닿은 검은 덩어리. 무채색이라 채도 필터엔 안 걸리고
        # 면적·종횡비도 도형 조건을 통과한다. BORDER_MARGIN 규칙으로만 걸러진다.
        f[150:280, 360:400] = 40
        if 20 <= i < 25:
            f[40:160, 40:160] = 40  # 팔이 target을 통째로 가림
        if 10 <= i < 15:
            f[40:160, 40:66] = 40  # 팔이 target 일부(약 20%)만 가림 (dense_progress는 살아야 함)
        w.write(f)
    w.release()

    board = (0, 0, 400, 400)
    shapes, ink, occ = track(vid, board, 0.72)
    labels = sorted(k.split("#")[0] for k, _ in shapes)
    # 팔이 섞였으면 4개가 되고, 팔이 도형과 병합됐으면 3개 미만이거나 분류가 깨진다
    assert labels == ["circle", "rectangle", "triangle"], f"도형 검출/분류 오류: {labels}"
    per_shape = summarize(ink, occ)
    assert occ["circle#0"][22] and not occ["circle#0"][5], "가림 판정 오류"
    c, r = aggregate(per_shape, "circle"), aggregate(per_shape, "rectangle")
    assert c["erased_frac"] > 0.9, c
    assert r["erased_frac"] < 0.1, r
    # 회귀 방지: 완료 프레임 인덱스는 영상 길이를 넘을 수 없다 (같은 라벨 키 충돌 버그)
    assert c["frames_to_90pct"] < c["n_frames"] == 60, c

    # dense_progress: 부분 가림(f10~14)에서는 살아있고 전체 가림(f20~24)에서만 죽어야 한다.
    # 판정용 track()은 부분 가림도 통째로 버리므로, 이 차이가 dense_progress의 존재 이유다.
    dp = dense_progress(vid, board, 0.72)
    dens, valid = dp["circle#0"]
    assert occ["circle#0"][12], "부분 가림인데 판정용 track이 가림으로 안 봄 (전제 붕괴)"
    assert valid[12], "부분 가림에서 dense_progress가 죽음 — 회수 실패"
    assert not valid[22], "전체 가림인데 dense_progress가 유효하다고 함"
    assert dens[:10].mean() > 5 * dens[-10:].mean(), "지워졌는데 밀도가 안 떨어짐"
    print("selftest OK", c, r)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        sys.exit(main())
