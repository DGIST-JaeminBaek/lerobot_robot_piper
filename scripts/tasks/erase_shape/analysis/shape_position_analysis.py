#!/usr/bin/env python3
"""화이트보드 위 도형 위치 분포를 원본 녹화의 첫 프레임에서 뽑아 분석한다.

`erase the shape` 학습 데이터가 보드의 어느 영역을 얼마나 덮고 있는지 보기 위한
도구다. 도형별로 분포가 다르거나 특정 영역이 비어 있으면, 정책이 그 영역의 도형을
제대로 못 다루게 되고 추가 녹화 시 어디를 보강할지도 여기서 나온다.

첫 프레임을 쓰는 이유: 이 시점엔 팔이 홈 자세(접힌 상태)라 보드를 전혀 가리지 않는다.
QC manifest의 start_frame은 이미 팔이 움직이기 시작한 뒤일 수 있어 부적합하다.

검출은 ML 모델이 아니라 배경차분 기반 알고리즘이다 — 자세한 근거는 detect_shape()
주석 참고.

실행:
    python scripts/tasks/erase_shape/analysis/shape_position_analysis.py --help
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
import sys
from dataclasses import dataclass, asdict

import av
import cv2
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]

# 원본 녹화 해상도. 다른 해상도는 카메라 세팅이 달랐다는 뜻이라 건너뛴다.
RAW_SIZE = (1280, 720)

# 보드에서 도형을 찾을 x 범위. 오른쪽 상한을 680으로 둔 것은 보드 경계(x≈751)
# 근처의 검은 브래킷을 도형으로 잘못 잡는 오검출이 실제로 발생했기 때문이다.
BOARD_X = (110, 680)

# 카메라 정렬 기준. 0802/0804/0805에서 측정한 나무 테이블 경계 x (751.0 ± 0.45px).
# 0727은 750.3 ± 3.13px로 세션 내 편차가 있어 이 값에 맞춰 평행이동 보정한다.
REFERENCE_EDGE_X = 751

# 병합/가공 산출물 — raw 녹화가 아니므로 제외한다.
EXCLUDED_DIRS = {"erase_the_shape", "erase_the_shape_512"}

SHAPE_ORDER = ["circle", "triangle", "rectangle"]
# dataviz 기본 팔레트의 앞 3슬롯. 이 세 개는 all-pairs 검증을 통과하는 조합이라
# 산점도(overlay)에 쓸 수 있다. 4개째부터는 통과하지 못한다.
SHAPE_COLORS = {"circle": "#2a78d6", "triangle": "#eb6834", "rectangle": "#1baf7a"}


@dataclass
class Detection:
    session: str
    name: str
    shape: str
    ok: bool
    method: str = ""
    camera_shift: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0
    area: int = 0


# ═══════════════════════════════════════════════════════════════════
# 녹화 수집
# ═══════════════════════════════════════════════════════════════════
def find_recordings(records_root: pathlib.Path, sessions: list[str]) -> list[tuple[str, pathlib.Path]]:
    """(세션, 녹화 폴더) 목록. 0727은 도형별 하위폴더, 나머지는 평면 구조다."""
    found: list[tuple[str, pathlib.Path]] = []
    for session in sessions:
        session_dir = records_root / session
        if not session_dir.is_dir():
            print(f"[WARN] {session_dir} 없음 — 건너뜀")
            continue
        for entry in sorted(session_dir.iterdir()):
            if not entry.is_dir() or entry.name in EXCLUDED_DIRS:
                continue
            if (entry / "videos").is_dir():
                found.append((session, entry))          # 평면 구조
            else:
                for sub in sorted(entry.iterdir()):     # 도형별 하위폴더(0727)
                    if sub.is_dir() and (sub / "videos").is_dir():
                        found.append((session, sub))
    return found


def parse_shape(name: str) -> str:
    match = re.search(r"erase_the_(\w+?)_", name)
    return match.group(1) if match else "unknown"


def read_first_frame(recording: pathlib.Path) -> np.ndarray | None:
    videos = sorted((recording / "videos" / "observation.images.top").rglob("*.mp4"))
    if not videos:
        return None
    container = av.open(str(videos[0]))
    try:
        stream = container.streams.video[0]
        if (stream.width, stream.height) != RAW_SIZE:
            return None
        return next(container.decode(video=0)).to_ndarray(format="rgb24")
    finally:
        container.close()


# ═══════════════════════════════════════════════════════════════════
# 카메라 정렬
# ═══════════════════════════════════════════════════════════════════
def measure_edge_x(rgb: np.ndarray) -> int:
    """나무 테이블 경계의 x 좌표. 로봇도 도형도 없는 세로 밴드에서 잰다."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(float)
    profile = gray[300:500, :].mean(axis=0)
    gradient = np.abs(np.diff(profile))
    return int(np.argmax(gradient[700:1100])) + 700


def align(rgb: np.ndarray, shift: float) -> np.ndarray:
    """x축 평행이동으로 기준 프레임에 맞춘다.

    카메라가 회전하거나 높이가 바뀌었다면 평행이동만으로는 보정되지 않는다 —
    보정 후에도 특정 세션 분포가 어긋나면 그쪽을 의심할 것.
    """
    if abs(shift) < 0.5:
        return rgb
    matrix = np.float32([[1, 0, shift], [0, 1, 0]])
    return cv2.warpAffine(rgb, matrix, (rgb.shape[1], rgb.shape[0]), borderMode=cv2.BORDER_REPLICATE)


# ═══════════════════════════════════════════════════════════════════
# 도형 검출
# ═══════════════════════════════════════════════════════════════════
def _pick_outline(mask: np.ndarray) -> tuple | None:
    """연결성분 중 '속 빈 윤곽선'다운 최대 성분을 고른다."""
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if area < 300 or not (40 < w < 400) or not (40 < h < 400):
            continue
        if area / (w * h) > 0.6:          # 꽉 찬 덩어리는 도형이 아니다
            continue
        if best is None or area > best[0]:
            best = (area, centroids[i], stats[i])
    return best


def detect_shape(rgb: np.ndarray) -> tuple[dict | None, str]:
    """도형의 bbox와 무게중심을 찾는다. (결과, 사용한 방법)을 돌려준다.

    배경차분을 쓰는 이유는 그려진 선이 얇고 흐리기 때문이다. 전역 임계값 + MORPH_OPEN을
    먼저 시도했다가 얇은 선이 조각나 부서지는 것을 확인했다 — OPEN은 쓰면 안 되고,
    끊긴 선을 잇는 CLOSE를 써야 한다.

    "배경보다 어두운 쪽"만 남기는 극성 조건이 Canny보다 유리하다. 잉크는 항상 배경보다
    어둡지만 Canny는 밝아지는 경계(반사광, 흰 패치 테두리)도 똑같이 잡는다. 그래서
    Canny는 배경차분이 실패했을 때의 폴백으로만 쓴다.
    """
    board = rgb[:, BOARD_X[0]:BOARD_X[1]]
    gray = cv2.cvtColor(board, cv2.COLOR_RGB2GRAY)
    red, _, blue = (board[:, :, i].astype(int) for i in range(3))

    # 나무 블럭은 갈색이라 R이 B보다 뚜렷하게 크다. 주변까지 넉넉히 제외한다.
    warm = cv2.dilate(((red - blue) > 14).astype(np.uint8), np.ones((25, 25), np.uint8))

    background = cv2.medianBlur(gray, 51)      # 조명 불균일까지 배경으로 흡수
    ink = cv2.subtract(background, gray)
    mask = ((ink > 8) & (warm == 0)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    best = _pick_outline(mask)
    method = "bgsub"
    if best is None:
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 90)
        edges[warm > 0] = 0
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        best = _pick_outline((edges > 0).astype(np.uint8))
        method = "canny"

    if best is None:
        return None, ""

    area, (cx, cy), stat = best
    return {
        "cx": float(cx) + BOARD_X[0],
        "cy": float(cy),
        "x": int(stat[0]) + BOARD_X[0],
        "y": int(stat[1]),
        "w": int(stat[2]),
        "h": int(stat[3]),
        "area": int(area),
    }, method


# ═══════════════════════════════════════════════════════════════════
# 시각화
# ═══════════════════════════════════════════════════════════════════
def save_contact_sheets(frames: dict, results: list[Detection], out_dir: pathlib.Path) -> None:
    """도형별 대조 시트 — bbox가 실제 도형을 감쌌는지 육안으로 확인하는 용도."""
    for shape in SHAPE_ORDER:
        rows = [r for r in results if r.shape == shape]
        if not rows:
            continue
        cells = []
        for r in rows:
            image = cv2.cvtColor(frames[(r.session, r.name)], cv2.COLOR_RGB2BGR)
            if r.ok:
                cv2.rectangle(image, (r.x, r.y), (r.x + r.w, r.y + r.h), (0, 0, 255), 3)
                cv2.circle(image, (int(r.cx), int(r.cy)), 7, (255, 0, 0), -1)
            image = cv2.resize(image, (320, 180))
            label = f"{r.session} {r.name[11:][:22]}"
            cv2.putText(image, label, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (0, 255, 255) if r.ok else (0, 0, 255), 1)
            cells.append(image)
        per_row = 6
        grid = [cv2.hconcat(cells[i:i + per_row]) for i in range(0, len(cells), per_row)]
        width = max(row.shape[1] for row in grid)
        grid = [cv2.copyMakeBorder(row, 0, 0, 0, width - row.shape[1], cv2.BORDER_CONSTANT, value=(30, 30, 30))
                for row in grid]
        cv2.imwrite(str(out_dir / f"contact_{shape}.png"), cv2.vconcat(grid))


def save_charts(reference: np.ndarray, results: list[Detection], out_dir: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for family in ("Noto Sans CJK JP", "NanumGothic", "DejaVu Sans"):
        if any(f.name == family for f in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = family
            break
    plt.rcParams["axes.unicode_minus"] = False

    ok = [r for r in results if r.ok]
    surface, ink, muted = "#fcfcfb", "#0b0b0b", "#8a8983"

    # ── overlay: 실제 보드 사진 위에 위치 표시 ──────────────
    fig, ax = plt.subplots(figsize=(11, 6.6), facecolor=surface)
    ax.imshow(reference, alpha=0.55)
    for shape in SHAPE_ORDER:
        pts = [(r.cx, r.cy) for r in ok if r.shape == shape]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=64, color=SHAPE_COLORS[shape], alpha=0.75,
                   edgecolors="white", linewidths=1.1, label=f"{shape} (n={len(pts)})", zorder=3)
    ax.add_patch(plt.Rectangle((BOARD_X[0], 0), BOARD_X[1] - BOARD_X[0], RAW_SIZE[1],
                               fill=False, ec=muted, lw=1, ls=(0, (5, 4)), zorder=2))
    ax.set_xlim(0, RAW_SIZE[0]); ax.set_ylim(RAW_SIZE[1], 0)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92,
              edgecolor="#dedcd4", fontsize=10)
    ax.set_title(f"화이트보드 위 도형 위치 (원본 첫 프레임 기준, n={len(ok)})",
                 fontsize=13, color=ink, loc="left", pad=12, fontweight="semibold")
    fig.text(0.012, 0.02, "점선 = 검출 대상 영역.  배경은 정렬 기준 프레임 1장.",
             fontsize=9, color=muted)
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    fig.savefig(out_dir / "overlay.png", dpi=150, facecolor=surface)
    plt.close(fig)

    # ── distribution: 도형별 cx / cy ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=surface)
    rng = np.random.default_rng(0)
    for ax, (attr, title) in zip(axes, [("cx", "가로 위치  cx (px)"), ("cy", "세로 위치  cy (px)")]):
        ax.set_facecolor(surface)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#dedcd4")
        ax.grid(axis="y", color="#eceae2", lw=1)
        ax.set_axisbelow(True)
        for i, shape in enumerate(SHAPE_ORDER):
            values = np.array([getattr(r, attr) for r in ok if r.shape == shape])
            if not len(values):
                continue
            ax.scatter(i + rng.uniform(-0.17, 0.17, len(values)), values, s=28,
                       color=SHAPE_COLORS[shape], alpha=0.6, edgecolors=surface, linewidths=0.7, zorder=3)
            ax.plot([i - 0.3, i + 0.3], [values.mean()] * 2, color=SHAPE_COLORS[shape],
                    lw=2.4, zorder=4, solid_capstyle="round")
        ax.set_xticks(range(len(SHAPE_ORDER)))
        ax.set_xticklabels(SHAPE_ORDER, fontsize=10, color="#52514e")
        ax.tick_params(length=0, labelsize=9.5, colors="#52514e")
        ax.set_xlim(-0.55, len(SHAPE_ORDER) - 0.45)
        ax.set_title(title, fontsize=11.5, color=ink, loc="left", pad=10)
    fig.suptitle("도형별 위치 분포", fontsize=13, color=ink, x=0.012, ha="left",
                 y=0.98, fontweight="semibold")
    fig.text(0.012, 0.02, "가로선 = 평균.  점 하나 = 녹화 1개.  cy가 클수록 화면 아래쪽.",
             fontsize=9, color=muted)
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    fig.savefig(out_dir / "distribution.png", dpi=150, facecolor=surface)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sessions", default="0727,0802,0804,0805")
    parser.add_argument("--records-root", type=pathlib.Path, default=REPO_ROOT / "records")
    parser.add_argument("--output-dir", type=pathlib.Path,
                        default=REPO_ROOT / "outputs" / "analysis" / "shape_positions")
    parser.add_argument("--no-save-frames", dest="save_frames", action="store_false",
                        help="추출한 첫 프레임 개별 저장을 건너뛴다")
    args = parser.parse_args(argv)

    sessions = [s.strip() for s in args.sessions.split(",") if s.strip()]
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    recordings = find_recordings(args.records_root, sessions)
    print(f"[FIND] 녹화 {len(recordings)}개")
    if not recordings:
        print("[ERROR] 녹화를 찾지 못했습니다", file=sys.stderr)
        return 1

    results: list[Detection] = []
    frames: dict[tuple[str, str], np.ndarray] = {}
    reference: np.ndarray | None = None

    for index, (session, path) in enumerate(recordings, 1):
        shape = parse_shape(path.name)
        raw = read_first_frame(path)
        if raw is None:
            print(f"  [{index:3d}/{len(recordings)}] {path.name}: 프레임 없음/해상도 불일치 — 건너뜀")
            results.append(Detection(session, path.name, shape, ok=False))
            continue

        shift = float(REFERENCE_EDGE_X - measure_edge_x(raw))
        frame = align(raw, shift)
        frames[(session, path.name)] = frame
        if reference is None:
            reference = frame

        found, method = detect_shape(frame)
        if found is None:
            print(f"  [{index:3d}/{len(recordings)}] {path.name}: 검출 실패")
            results.append(Detection(session, path.name, shape, ok=False, camera_shift=shift))
        else:
            results.append(Detection(session, path.name, shape, ok=True, method=method,
                                     camera_shift=shift, **found))

        if args.save_frames:
            frame_dir = out_dir / "frames" / session
            frame_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(frame_dir / f"{path.name}.png"),
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    with (out_dir / "shape_positions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(r) for r in results)

    print(f"\n[DETECT] 성공 {sum(r.ok for r in results)}/{len(results)}")
    for shape in SHAPE_ORDER:
        rows = [r for r in results if r.shape == shape]
        if rows:
            print(f"  {shape:10} {sum(r.ok for r in rows):3d}/{len(rows):<3d}")
    fallback = [r.name for r in results if r.method == "canny"]
    if fallback:
        print(f"  (Canny 폴백 {len(fallback)}건: {fallback[:5]})")

    save_contact_sheets(frames, [r for r in results if (r.session, r.name) in frames], out_dir)
    if reference is not None:
        save_charts(reference, results, out_dir)
    print(f"[SAVE] {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
