#!/usr/bin/env python3
"""검증 A: top 카메라가 가려서 못 보는 구간에 wrist 카메라가 잉크를 볼 수 있는가?

top 카메라는 지우는 동안 팔·지우개가 도형을 덮어서 잉크 잔량을 관측할 수 없다
(ink_metric.py가 그 프레임을 '가림'으로 버린다). 손목 카메라는 지우개 바로 위에서
보드를 근접 촬영하므로 원리적으로는 그 구간에도 잉크가 보여야 한다 — 이걸 확인한다.

wrist는 카메라가 움직여서 고정 ROI가 불가능하므로, 프레임 전체에서
"어둡고(잉크) 채도 낮은(나무 블록 아님)" 픽셀 비율을 대리 지표로 쓴다.

출력: 시계열 비교 그림 + 손목 프레임에 잉크 마스크를 덧씌운 스트립.

사용:
  python wrist_ink_probe.py <episode_dir> -o /tmp/probe.png
"""

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 한글 라벨용 폰트 (macOS: AppleGothic, Linux 랩 PC: NanumGothic)
for _f in ("AppleGothic", "NanumGothic", "Malgun Gothic"):
    if any(_f == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False
import numpy as np

import ink_metric as M

WOOD_SAT = 40  # 이 이상이면 나무 지우개 블록 (마커는 ≈10)
DARK_RATIO = 0.72


def wrist_ink(video, crop=0.4, stride=1):
    """손목 영상의 프레임별 (잉크 비율, 나무 블록 비율).

    카메라가 그리퍼에 고정 장착돼 있어 집게와 케이블이 항상 화면 하단을 차지한다.
    이들이 '어두운 픽셀'로 잡혀 잉크 신호를 덮으므로 상단 crop 비율만 본다.
    """
    cap = cv2.VideoCapture(str(video))
    ink, wood, masks = [], [], {}
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % stride == 0:
            f = f[: int(f.shape[0] * crop)]
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            sat = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[:, :, 1]
            block = sat > WOOD_SAT
            # 보드 흰색 기준: 블록이 아닌 영역의 밝은 쪽 분위수
            board = g[~block]
            white = float(np.percentile(board, 90)) if board.size else 255.0
            m = (g < white * DARK_RATIO) & ~block
            ink.append(float(m.mean()))
            wood.append(float(block.mean()))
            masks[i] = m
        i += 1
    cap.release()
    return np.array(ink), np.array(wood), masks


def top_series(ep_dir):
    """top 카메라의 target 도형 잉크 envelope과 가림 플래그."""
    target = M.task_of(ep_dir)
    _, ink, occ = M.track(M.top_video(ep_dir), M.DEFAULT_BOARD, DARK_RATIO)
    per = M.summarize(ink, occ)
    keys = [k for k in per if k.split("#")[0] == target and "env" in per[k]]
    if not keys:
        raise RuntimeError(f"target({target}) 도형을 못 찾음")
    n = min(len(per[k]["env"]) for k in keys)
    env = np.sum([per[k]["env"][:n] for k in keys], axis=0)
    occluded = np.any([np.asarray(occ[k][:n]) for k in keys], axis=0)
    return env, occluded, target


def wrist_video(ep_dir: Path) -> Path:
    hits = sorted(ep_dir.glob("videos/observation.images.wrist/**/*.mp4"))
    if not hits:
        raise FileNotFoundError(f"wrist 영상 없음: {ep_dir}")
    return hits[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("episode")
    p.add_argument("-o", "--out", type=Path, required=True)
    p.add_argument("--crop", type=float, default=0.4, help="손목 화면 상단 이 비율만 사용 (그리퍼 제외)")
    a = p.parse_args()
    ep = Path(a.episode)

    env, occ, target = top_series(ep)
    wink, wood, masks = wrist_ink(wrist_video(ep), a.crop)
    n = min(len(env), len(wink))
    env, occ, wink, wood = env[:n], occ[:n], wink[:n], wood[:n]
    t = np.arange(n)

    # top이 가린 구간에서 wrist 신호가 살아있는지 = 이 검증의 핵심 질문
    occ_idx = np.nonzero(occ)[0]
    span = (occ_idx[0], occ_idx[-1]) if len(occ_idx) else (0, 0)
    inside = wink[occ] if occ.any() else np.array([0.0])
    outside = wink[~occ] if (~occ).any() else np.array([0.0])

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(3, 6, height_ratios=[2, 2, 1.6], hspace=0.45, wspace=0.15)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, env / max(env[0], 1e-9), color="tab:blue", lw=1.6, label="top: target 잉크 잔량 (정규화)")
    if occ.any():
        ax1.axvspan(*span, color="tab:red", alpha=0.12, label="top 가림 구간 (관측 불가)")
    ax1.set_ylabel("잉크 잔량")
    ax1.set_title(f"{ep.name}  —  target={target}", fontsize=11)
    ax1.legend(fontsize=8, loc="lower left")
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
    ax2.plot(t, wink, color="tab:green", lw=1.2, label="wrist: 잉크 픽셀 비율")
    ax2.plot(t, wood, color="tab:orange", lw=1.0, alpha=0.7, label="wrist: 지우개 블록 화면 점유율")
    if occ.any():
        ax2.axvspan(*span, color="tab:red", alpha=0.12)
    ax2.set_ylabel("화면 비율")
    ax2.set_xlabel("frame")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(alpha=0.3)

    # 손목 프레임 6장 + 잉크 마스크 오버레이
    picks = np.linspace(0, n - 1, 6).astype(int)
    cap = cv2.VideoCapture(str(wrist_video(ep)))
    frames = {}
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in picks:
            frames[i] = f
        i += 1
    cap.release()

    for j, k in enumerate(picks):
        ax = fig.add_subplot(gs[2, j])
        f = frames.get(k)
        if f is not None:
            vis = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).copy()
            m = masks.get(k)
            if m is not None:
                vis[: m.shape[0]][m] = [255, 0, 0]  # 잉크로 판정된 픽셀을 빨강으로
            cut = int(vis.shape[0] * a.crop)
            vis[cut : cut + 3] = [0, 128, 255]  # crop 경계선 (아래는 분석에서 제외)
            vis[cut + 3 :] = (vis[cut + 3 :] * 0.35).astype(vis.dtype)
            ax.imshow(vis)
        ax.set_title(f"f{k}\n{'가림' if occ[k] else '관측가능'}", fontsize=8)
        ax.axis("off")

    fig.suptitle(
        f"검증 A: top 가림 구간에서 wrist 잉크 신호  |  가림 중 평균 {inside.mean():.4f} "
        f"vs 비가림 {outside.mean():.4f}",
        fontsize=12,
    )
    fig.savefig(a.out, dpi=110, bbox_inches="tight")
    print(f"저장: {a.out}")
    print(f"  top 가림 구간: frame {span[0]}~{span[1]} ({occ.mean():.0%})")
    print(f"  wrist 잉크 비율  가림 중 {inside.mean():.4f} / 비가림 {outside.mean():.4f}")
    print(f"  wrist 블록 점유율 가림 중 {wood[occ].mean() if occ.any() else 0:.3f}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    main()
