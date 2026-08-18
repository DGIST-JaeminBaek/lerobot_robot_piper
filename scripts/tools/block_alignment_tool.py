#!/usr/bin/env python3
"""지우개 블럭을 학습 데이터와 같은 자리에 놓도록 실시간으로 안내한다.

정책은 학습 때 본 것과 같은 장면을 봐야 제대로 동작한다. 블럭 홈 위치가 어긋나면
집으러 가는 첫 동작부터 빗나가므로, 추론 전에 이 도구로 맞춰두는 것이 안전하다.

top 카메라를 열어 매 프레임 블럭을 찾고, 학습 데이터 120개(0802/0804/0805)에서
측정한 기준 위치와의 차이를 화살표와 수치로 보여준다. 로봇 팔은 전혀 건드리지 않는다.

기준값 출처: 0802/0804/0805 녹화 120개의 첫 프레임에서 블럭 무게중심을 측정한 평균.
그 세 세션은 블럭을 보드에 표시된 마커 위에 놓아서 편차가 cx 1.5px / cy 1.8px 로
매우 작다. 0727은 검출이 불안정해 기준에서 제외했다.

실행:
    python scripts/tools/block_alignment_tool.py
    q 종료 / s 스냅샷 저장 / r 카메라 정렬 재측정

    # 도형을 그릴 6개 대표 위치(상좌/상중/상우/하좌/하중/하우)도 같이 보려면
    # camera_overlay_view_315.py의 좌표를 재사용해 --shape-zones로 켠다:
    python scripts/tools/block_alignment_tool.py --shape-zones 135
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import cv2
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# 학습 데이터 120개에서 측정한 블럭 중심 (raw 1280x720 좌표)
REFERENCE_CX = 428.9
REFERENCE_CY = 305.9
REFERENCE_SD = (1.5, 1.8)          # 학습 데이터의 표준편차 — 참고 표시용

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
}
ZONE_COLORS = (  # BGR, camera_overlay_view.py COLOR_NAMES와 맞춘 팔레트
    (0, 0, 255), (0, 140, 255), (0, 255, 255), (255, 255, 0), (255, 0, 255), (255, 0, 0),
)


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


def draw_shape_zones(canvas: np.ndarray, zones: tuple) -> None:
    """도형을 그릴 6개 대표 위치를 얇은 라벨 박스로 그린다. 블럭 정렬 표시보다
    먼저(아래층에) 그려서 시선이 블럭 쪽(초록/노랑/빨강)에 남게 한다."""
    for (label, cx, cy, w, h), color in zip(zones, ZONE_COLORS):
        x0, y0 = int(cx - w / 2), int(cy - h / 2)
        x1, y1 = int(cx + w / 2), int(cy + h / 2)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
        cv2.putText(canvas, label, (x0, max(0, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                   0.42, color, 1, cv2.LINE_AA)


def draw_overlay(frame: np.ndarray, found: dict | None, ref: tuple[float, float],
                 edge_x: int, shift: float, zones: tuple = ()) -> np.ndarray:
    canvas = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ref_x, ref_y = ref

    if zones:
        draw_shape_zones(canvas, zones)

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
    lines.append(("q quit   s snapshot   r re-measure camera", (190, 190, 190)))

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
                        help="도형을 그릴 6개 대표 위치(상좌/상중/상우/하좌/하중/하우)를 "
                             "같이 표시한다. 기본은 꺼짐(none) — 안 켜면 기존 동작과 동일. "
                             "135=학습 데이터셋 135개 기준, 315=candidate asset 315개 기준")
    args = parser.parse_args(argv)
    zones = SHAPE_ZONES.get(args.shape_zones, ())

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
    camera.connect(warmup=True)
    print(f"[REF] 목표 블럭 중심 cx={args.ref_cx:.1f} cy={args.ref_cy:.1f} "
          f"(학습 표준편차 {REFERENCE_SD[0]}/{REFERENCE_SD[1]}px)")

    window = "Block alignment  |  green = target,  arrow = move this way"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 860)

    edge_x, shift = None, 0.0
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
            cv2.imshow(window, draw_overlay(frame, found, (args.ref_cx, args.ref_cy), edge_x, shift, zones))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                edge_x = None
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
