#!/usr/bin/env python3
"""평가 결과(episodes.csv)를 **위치(zone) x 도형**으로 갈라 지운 비율 평균을 낸다.

왜 필요한가: 채점기(erase_eval.py)는 조건별 성공률과 평균만 낸다. 그런데 실물에서
잘 지우고 못 지우는 차이는 도형 종류보다 **보드 위 어느 자리인가**에서 훨씬 크게
갈린다(2026-08-21 132 topwrist 54회: 도형별 49~54%로 차이가 거의 없는데 위치별로는
32%~91%). 그 축이 채점 결과에는 안 남아 있어서 여기서 따로 붙인다.

위치는 CSV에 없으므로 각 에피소드의 `reference_frame.png`에서 target 도형을
`ink_metric.detect_shapes()`로 검출해 가장 가까운 zone에 배정한다(같은 파이프라인을
쓰므로 채점과 기준이 어긋나지 않는다). 실측 2026-08-21: 배정된 도형 중심이 zone
중심에서 중앙값 6px / 최대 18px 안이라 애매한 배정은 없었다.

배경 이미지는 **에피소드 reference 프레임들의 픽셀별 중앙값**이다. 도형이 자리마다
다르므로 중앙값을 취하면 도형만 사라지고 빈 보드가 남는다 — 따로 빈 보드를 찍어둘
필요가 없다.

사용:
    python scripts/tasks/erase_shape/analysis/ink_by_zone.py \\
        outputs/evaluation/erase_shape/0821_smolvla_pickup132_topwrist_present54/episodes.csv

산출물(--out-dir, 기본 outputs/analysis/erase_shape/ink_by_zone/<csv 폴더 이름>):
    erased_by_zone_<도형>.png  깨끗한 보드 위 6개 zone에 진행률 링(지운 비율 평균)
    ink_by_zone.csv            zone x 도형 지운비율 평균/SEM/성공수 표
    clean_board.png            합성한 빈 보드(재사용 가능)
"""

from __future__ import annotations

import argparse
import collections
import csv
import pathlib
import sys

import cv2
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
LIB_DIR = REPO_ROOT / "scripts" / "tasks" / "erase_shape" / "lib"
for path in (str(SCRIPT_DIR), str(LIB_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import ink_metric as M  # noqa: E402
import plot_ko  # noqa: E402
from block_alignment_tool import SHAPE_ZONES  # noqa: E402

SHAPES = ("circle", "triangle", "rectangle")
SHAPE_KO = {"circle": "원", "triangle": "삼각형", "rectangle": "사각형"}
SHAPE_COLORS = {"circle": "#e8564a", "triangle": "#2e9e5b", "rectangle": "#3b7dd8"}
ALL_COLOR = "#5b4b9e"   # 도형 전체 합산 그림 — 세 도형 색 어디에도 안 겹치게
# 보드만 남기는 가로 crop. zone cx가 321~577이라 여유를 두고 자른다.
CROP_X = (190, 760)
# 진행률 링 크기. zone 간 최소 간격이 101px(top-left↔top-mid)이라 반지름 46이 상한이다.
R_RING = 44
RING_W = 13


def resolve_episode(row: dict, rollout_root: pathlib.Path) -> pathlib.Path | None:
    """CSV의 path를 먼저 믿되, 아카이브로 옮겨졌으면 이름으로 다시 찾는다.

    episodes.csv의 path는 절대경로라 폴더를 옮기는 순간 깨진다(실측 2026-08-21:
    아카이브한 4개 세트 전부 깨져 있었다). 이름은 안 바뀌므로 그걸로 되짚는다.
    """
    direct = pathlib.Path(row.get("path", ""))
    if direct.is_dir():
        return direct
    name = row.get("episode", "")
    if not name:
        return None
    hits = sorted(rollout_root.glob(f"**/{name}"))
    return hits[0] if hits else None


def assign_zones(rows, zones, rollout_root, board, dark_ratio, exclude):
    """각 에피소드를 (zone, 도형, 지운비율)로 만든다. 못 쓰는 건 사유와 함께 건너뛴다."""
    out, skipped = [], []
    for row in rows:
        ep = resolve_episode(row, rollout_root)
        if ep is None:
            skipped.append((row.get("episode", "?"), "폴더 없음"))
            continue
        ref = ep / "reference_frame.png"
        if not ref.exists():
            skipped.append((ep.name, "reference_frame.png 없음"))
            continue
        frame = cv2.imread(str(ref))
        if frame is None:
            skipped.append((ep.name, "reference 읽기 실패"))
            continue
        target = row.get("target", "")
        shapes, _white = M.detect_shapes(frame, board, dark_ratio, exclude)
        cand = [bb for lbl, bb in shapes if lbl.split("#")[0] == target]
        if not cand:
            skipped.append((ep.name, f"기준 프레임에서 {target} 미검출"))
            continue
        try:
            erased = float(row["erased_target"])
            remaining = float(row["remaining_frac"])
        except (KeyError, TypeError, ValueError):
            # 8/20 재설계 이전 CSV는 스키마가 달라 이 컬럼들이 없다. 조용히 0으로
            # 때우면 안 된다 — 없는 데이터를 "0% 지움"으로 보고하게 된다.
            skipped.append((ep.name, "erased_target/remaining_frac 없음(구 스키마 CSV)"))
            continue
        x, y, w, h = cand[0]
        cx, cy = x + w / 2, y + h / 2
        zone = min(zones, key=lambda z: (cx - z[1]) ** 2 + (cy - z[2]) ** 2)
        out.append({
            "episode": ep.name, "dir": ep, "shape": target, "zone": zone[0],
            "dist_to_zone": float(np.hypot(cx - zone[1], cy - zone[2])),
            "erased": erased, "remaining": remaining,
            "success": str(row.get("success", "")).lower() == "true",
        })
    return out, skipped


def clean_board(records, crop_x) -> np.ndarray:
    """reference 프레임들의 픽셀별 중앙값 = 도형 없는 빈 보드."""
    frames = [cv2.imread(str(r["dir"] / "reference_frame.png")) for r in records]
    frames = [f for f in frames if f is not None]
    if not frames:
        raise RuntimeError("배경을 만들 reference 프레임이 없다")
    board = np.median(np.stack(frames), axis=0).astype(np.uint8)
    return board[:, crop_x[0]:crop_x[1]]


def draw_shape_figure(records, zones, board_img, crop_x, shape, title, out_png) -> None:
    """6개 zone 중심에 '지운 비율' 진행률 링을 그린다. shape=None이면 도형 전체 합산.

    한 그림에 세 도형을 부채꼴로 겹쳐 그리던 방식은 버렸다 — 이 과제는 도형의
    **테두리를 지우는 것**이라 면적 부채꼴이 뜻하는 바가 없고, zone 간격이 최소
    101px(top-left↔top-mid)이라 셋을 겹치면 서로 침범한다. zone마다 링 하나만 두고
    도형은 그림을 나눠서 본다.
    """
    font = plot_ko.use_korean()
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge

    cell = collections.defaultdict(list)
    for r in records:
        if shape is None or r["shape"] == shape:
            cell[r["zone"]].append(r["erased"])

    h, w = board_img.shape[:2]
    fig, ax = plt.subplots(figsize=(w / 72, h / 72), dpi=100)
    ax.imshow(cv2.cvtColor(board_img, cv2.COLOR_BGR2RGB))
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    outline = [pe.withStroke(linewidth=2.4, foreground="black")]
    color = SHAPE_COLORS[shape] if shape else ALL_COLOR

    for label, zx0, zy, _zw, _zh in zones:
        zx = zx0 - crop_x[0]
        vals = cell[label]
        if not vals:
            continue
        mean = float(np.mean(vals))
        # 바탕 링(100%) + 지운 만큼 채운 링. 12시에서 시계방향으로 찬다.
        ax.add_patch(Wedge((zx, zy), R_RING, 0, 360, width=RING_W,
                           facecolor="#ffffff", ec="#666666", lw=0.9,
                           alpha=0.75, zorder=3))
        if mean > 0:
            ax.add_patch(Wedge((zx, zy), R_RING, 90 - 360 * mean, 90, width=RING_W,
                               facecolor=color, ec="white", lw=0.8, alpha=0.95, zorder=4))
        ax.text(zx, zy, f"{mean:.0%}", ha="center", va="center", fontsize=15,
                fontweight="bold", color="white", zorder=6, path_effects=outline)
        above = label.startswith("top")   # 위/아래 줄 라벨이 서로 겹치지 않게
        ax.text(zx, zy - R_RING - 8 if above else zy + R_RING + 8,
                f"{label}  (n={len(vals)})",
                ha="center", va="bottom" if above else "top",
                fontsize=9.5, fontweight="bold", color="#111111", zorder=6,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#555555", alpha=0.9))

    ax.set_title(title, fontsize=12, pad=10, color=color, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    if font is None:
        print("[WARN] 한글 폰트를 못 찾았다 — 라벨이 □로 나올 수 있다", file=sys.stderr)


def write_table(records, zones, out_csv) -> None:
    cell = collections.defaultdict(list)
    for r in records:
        cell[(r["zone"], r["shape"])].append(r["erased"])

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["zone", "shape", "n", "erased_mean", "sem", "success_n"])
        for label, *_ in zones:
            for shape in SHAPES:
                vals = cell[(label, shape)]
                if not vals:
                    continue
                sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
                wins = sum(1 for r in records
                           if r["zone"] == label and r["shape"] == shape and r["success"])
                writer.writerow([label, shape, len(vals), f"{np.mean(vals):.4f}",
                                 f"{sem:.4f}", wins])

    print(f"\n[지운 비율 평균] 높을수록 잘 지운 것 (n={len(records)})\n")
    print(f"{'zone':14}" + "".join(f"{SHAPE_KO[s]:>12}" for s in SHAPES) + f"{'zone평균':>12}")
    for label, *_ in zones:
        line = f"{label:14}"
        for shape in SHAPES:
            vals = cell[(label, shape)]
            line += f"{np.mean(vals):>12.1%}" if vals else f"{'-':>12}"
        zone_vals = [v for s in SHAPES for v in cell[(label, s)]]
        print(line + (f"{np.mean(zone_vals):>12.1%}" if zone_vals else f"{'-':>12}"))
    line = f"{'도형평균':14}"
    for shape in SHAPES:
        vals = [v for lb, *_ in zones for v in cell[(lb, shape)]]
        line += f"{np.mean(vals):>12.1%}" if vals else f"{'-':>12}"
    everything = [r["remaining"] for r in records]
    print(line + f"{np.mean(everything):>12.1%}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=pathlib.Path, help="erase_eval.py가 만든 episodes.csv")
    parser.add_argument("--out-dir", type=pathlib.Path, default=None,
                        help="기본 outputs/analysis/erase_shape/ink_by_zone/<csv 폴더 이름>")
    parser.add_argument("--rollout-root", type=pathlib.Path,
                        default=REPO_ROOT / "records",
                        help="CSV의 path가 깨졌을 때 에피소드를 이름으로 찾을 위치")
    parser.add_argument("--shape-zones", choices=tuple(SHAPE_ZONES), default="315",
                        help="zone 좌표 기준 (block_alignment_tool.py와 같은 값)")
    parser.add_argument("--title", default=None, help="그림 제목 첫 줄")
    parser.add_argument("--board", type=int, nargs=4, default=list(M.DEFAULT_BOARD))
    parser.add_argument(
        "--exclude", action="append", default=None,
        type=lambda s: tuple(int(v) for v in s.split(",")), metavar="X,Y,W,H",
        help="도형 검출에서 뺄 구역. 기본값은 erase_check.py와 동일",
    )
    parser.add_argument("--dark-ratio", type=float, default=0.72)
    args = parser.parse_args(argv)
    args.exclude = M.resolve_exclude(args.exclude)

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    if not rows:
        print(f"[ERROR] 비어 있는 CSV: {args.csv}", file=sys.stderr)
        return 2

    zones = SHAPE_ZONES[args.shape_zones]
    records, skipped = assign_zones(rows, zones, args.rollout_root,
                                    tuple(args.board), args.dark_ratio, args.exclude)
    for name, why in skipped:
        print(f"[skip] {name}: {why}", file=sys.stderr)
    if not records:
        print("[ERROR] zone에 배정된 에피소드가 없다", file=sys.stderr)
        return 2

    out_dir = args.out_dir or (REPO_ROOT / "outputs" / "analysis" / "erase_shape"
                               / "ink_by_zone" / args.csv.parent.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    board_img = clean_board(records, CROP_X)
    cv2.imwrite(str(out_dir / "clean_board.png"), board_img)

    label_base = args.title or args.csv.parent.name
    made = []

    # 도형 전체 합산 — 위치 효과만 보고 싶을 때 이거 하나면 된다.
    all_png = out_dir / "erased_by_zone.png"
    draw_shape_figure(
        records, zones, board_img, CROP_X, None,
        f"위치별 지운 비율 평균 — 도형 전체 (n={len(records)}, "
        f"전체 평균 {np.mean([r['erased'] for r in records]):.0%})\n{label_base}"
        "   ·   링이 많이 찰수록 잘 지운 것",
        all_png)
    made.append(all_png.name)

    for shape in SHAPES:
        vals = [r["erased"] for r in records if r["shape"] == shape]
        if not vals:
            continue
        out_png = out_dir / f"erased_by_zone_{shape}.png"
        title = (f"{SHAPE_KO[shape]} — 위치별 지운 비율 평균 (n={len(vals)}, "
                 f"전체 평균 {np.mean(vals):.0%})\n{label_base}"
                 "   ·   링이 많이 찰수록 잘 지운 것")
        draw_shape_figure(records, zones, board_img, CROP_X, shape, title, out_png)
        made.append(out_png.name)
    write_table(records, zones, out_dir / "ink_by_zone.csv")

    dists = [r["dist_to_zone"] for r in records]
    print(f"\nzone 배정 거리: 중앙값 {np.median(dists):.0f}px / 최대 {max(dists):.0f}px "
          f"(멀면 배정이 의심스럽다)")
    print(f"→ {out_dir}")
    for name in made + ["ink_by_zone.csv", "clean_board.png"]:
        print(f"     {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
