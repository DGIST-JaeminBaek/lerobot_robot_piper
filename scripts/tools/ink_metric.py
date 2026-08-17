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
MAX_SAT = 25  # 평균 채도 상한 — 나무 지우개 블록(≈45)은 걸러지고 검은 마커(≈10)는 통과
MERGE_GAP = 30  # 이 픽셀 이내로 붙어있는 조각은 한 도형으로 병합 (도형 간 간격은 100px 이상)
OCCLUDER_KERNEL = 11  # 이 크기로 opening해서 남는 어두운 덩어리 = 팔·그리퍼 (얇은 잉크 획은 사라짐)
OCCL_JUMP = 0.15  # 비-흰색 비율이 이만큼 급증하면 가림 (판정용 track에서 사용)
MIN_VISIBLE = 0.70  # dense_progress: 도형 영역이 이만큼 안 보이면 그 프레임은 측정 불가
PAD = 12  # 도형 bbox 여유

# 성공 판정 임계 (make_done_labels.py, erase_check.py가 공유)
SUCCESS_ERASED = 0.9  # target이 이만큼 지워지면 성공
MAX_DISTRACTOR = 0.10  # distractor를 이만큼 넘게 건드리면 실패.
# 노이즈 플로어 실측: 120 에피소드에서 평균 0.009 / p95 0.037 / 최대 0.097.
# 0.10은 플로어의 약 3배 — 이보다 조이면 정상 시연이 오탐으로 걸린다.


def detect_shapes(frame, board, dark_ratio):
    """첫 프레임에서 도형 bbox와 종류를 찾는다. -> [(label, (x,y,w,h)), ...]"""
    bx, by, bw, bh = board
    sub = frame[by : by + bh, bx : bx + bw]
    roi = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)[:, :, 1]
    white = float(np.percentile(roi, 90))
    ink = (roi < white * dark_ratio).astype(np.uint8)
    # 선이 끊겨 있어도 한 도형으로 묶이도록 닫기
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 유채색 물체(나무 지우개 블록)는 병합 "전에" 버린다.
    # 나중에 버리면 블록이 옆 도형과 한 덩어리로 묶여 bbox와 잉크 측정이 오염된다.
    def achromatic(c):
        x, y, w, h = cv2.boundingRect(c)
        return float(sat[y : y + h, x : x + w].mean()) <= MAX_SAT

    contours = merge_nearby([c for c in contours if achromatic(c)])
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


def track(video, board, dark_ratio, occl_jump=OCCL_JUMP):
    cap = cv2.VideoCapture(str(video))
    ok, first = cap.read()
    if not ok:
        raise RuntimeError(f"영상 첫 프레임 읽기 실패: {video}")

    detected, white = detect_shapes(first, board, dark_ratio)
    # 같은 종류가 여러 개 나올 수 있으므로 키를 유일하게 만든다 (circle#0, circle#1 ...)
    shapes = [(f"{lbl}#{i}", box) for i, (lbl, box) in enumerate(detected)]
    ink_thr = white * dark_ratio

    white_thr = white * 0.88  # 이보다 어두우면 보드 흰색이 아님(잉크·팔·지우개 블록)

    ink = {k: [] for k, _ in shapes}
    occ = {k: [] for k, _ in shapes}
    base_nonwhite = {}

    frame = first
    while ok:
        for lbl, (x, y, w, h) in shapes:
            g = cv2.cvtColor(frame[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)
            ink[lbl].append(float((g < ink_thr).mean()))
            nw = float((g < white_thr).mean())
            base_nonwhite.setdefault(lbl, nw)
            occ[lbl].append(nw > base_nonwhite[lbl] + occl_jump)
        ok, frame = cap.read()
    cap.release()
    return shapes, ink, occ


def dense_progress(video, board, dark_ratio=0.72, min_visible=MIN_VISIBLE):
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
    cap = cv2.VideoCapture(str(video))
    ok, first = cap.read()
    if not ok:
        raise RuntimeError(f"영상 첫 프레임 읽기 실패: {video}")
    detected, white = detect_shapes(first, board, dark_ratio)
    shapes = [(f"{lbl}#{i}", box) for i, (lbl, box) in enumerate(detected)]
    ink_thr = white * dark_ratio
    kernel = np.ones((OCCLUDER_KERNEL, OCCLUDER_KERNEL), np.uint8)

    dens = {k: [] for k, _ in shapes}
    valid = {k: [] for k, _ in shapes}
    frame = first
    while ok:
        for lbl, (x, y, w, h) in shapes:
            sub = frame[y : y + h, x : x + w]
            g = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
            sat = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)[:, :, 1]
            dark = g < ink_thr
            # 얇은 잉크 획은 opening으로 사라지고 팔·그리퍼 같은 굵은 덩어리만 남는다
            blob = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
            visible = ~(blob | (sat > MAX_SAT))
            n_vis = int(visible.sum())
            dens[lbl].append(float((dark & visible).sum() / n_vis) if n_vis else 0.0)
            valid[lbl].append(float(visible.mean()) >= min_visible)
        ok, frame = cap.read()
    cap.release()
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


def dump_roi(ep_dir, board, dark_ratio, path):
    cap = cv2.VideoCapture(str(top_video(ep_dir)))
    ok, frame = cap.read()
    cap.release()
    shapes, _ = detect_shapes(frame, board, dark_ratio)
    bx, by, bw, bh = board
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 0, 0), 1)
    for lbl, (x, y, w, h) in shapes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, lbl, (x, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite(str(path), frame)
    print(f"검출 {len(shapes)}개 저장: {path}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("path", help="LeRobotDataset 에피소드 디렉터리 (glob 가능)")
    p.add_argument("--board", type=lambda s: tuple(int(v) for v in s.split(",")), default=DEFAULT_BOARD)
    p.add_argument("--dark-ratio", type=float, default=0.72, help="보드 흰색 대비 잉크 판정 임계 비율")
    p.add_argument("--occl-jump", type=float, default=OCCL_JUMP, help="비-흰색 비율이 이만큼 급증하면 가림 처리")
    p.add_argument("--csv", type=Path)
    p.add_argument("--dump-roi", type=Path, help="첫 에피소드 검출 결과를 그려서 저장 후 종료")
    a = p.parse_args(argv)

    dirs = [Path(d) for d in sorted(glob.glob(a.path)) if Path(d).is_dir()]
    if not dirs:
        p.error(f"에피소드 디렉터리를 찾지 못함: {a.path}")

    if a.dump_roi:
        dump_roi(dirs[0], a.board, a.dark_ratio, a.dump_roi)
        return 0

    rows = []
    for d in dirs:
        target = task_of(d)
        shapes, ink, occ = track(top_video(d), a.board, a.dark_ratio, a.occl_jump)
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
        cv2.rectangle(f, (60, 250), (150, 330), wood_bgr, -1)  # 나무 지우개 블록: 검출되면 안 됨
        if 20 <= i < 25:
            f[40:160, 40:160] = 40  # 팔이 target을 통째로 가림
        if 10 <= i < 15:
            f[40:160, 40:66] = 40  # 팔이 target 일부(약 20%)만 가림 (dense_progress는 살아야 함)
        w.write(f)
    w.release()

    board = (0, 0, 400, 400)
    shapes, ink, occ = track(vid, board, 0.72)
    labels = sorted(k.split("#")[0] for k, _ in shapes)
    assert labels == ["circle", "rectangle"], f"도형 검출/분류 오류: {labels}"
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
