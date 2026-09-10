#!/usr/bin/env python3
"""315개 원본 녹화의 **관절 상태 시계열**을 관절별로 그린다.

EEF(end-effector) 궤적 분석(`first_chunk_fk_analysis.py`)과 달리, 여기서는 순기구학을
거치지 않고 `observation.state`에 기록된 joint1~6 / gripper 값을 그대로 본다. EEF는 6개
관절의 합성이라 어느 관절이 흔들렸는지 가려지지만, 관절별로 보면 학습 데이터에서 실제로
움직이는 축과 세션마다 다른 축이 바로 드러난다.

데이터: `configs/erase_shape_315_manifest.json`의 315개 원본 녹화. QC로 확정된
`start_frame`(inclusive) ~ `end_frame`(exclusive) 구간만 쓴다 — 그 바깥은 팔이 홈에서
대기하거나 이미 종료한 구간이라 시계열에 평평한 꼬리만 붙는다.

단위: 데이터셋에 저장된 값은 정규화값(joint은 -100~100, gripper는 0~100)이다. 사람이
읽을 수 있게 Piper calibration raw 범위로 되돌려 joint은 도(deg), gripper는 열림
폭(mm)으로 환산한다. 환산식은 `first_chunk_fk_analysis.py`의
`normalized_joint_to_radians()`와 같은 것이다.

산출물(--out-dir, 기본 outputs/analysis/erase_shape/joint_state_timeseries):
    joint_timeseries_overlay.png     315개 에피소드를 실제 경과 시간축에 그대로 겹쳐 그림
    joint_timeseries_median_time.png 실제 경과 시간축 위의 도형별 중앙값 + IQR 밴드
    joint_timeseries_median.png      진행률(0~100%)로 정렬한 뒤 도형별 중앙값 + IQR 밴드
    joint_timeseries_stats.csv     관절별/도형별 가동 범위·시작·끝 요약

사용:
    python scripts/tasks/erase_shape/analysis/joint_state_timeseries.py
    python scripts/tasks/erase_shape/analysis/joint_state_timeseries.py --shape circle
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
LIB_DIR = REPO_ROOT / "scripts" / "tasks" / "erase_shape" / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

import plot_ko  # noqa: E402  (matplotlib.use("Agg")를 pyplot import 전에 건다)
from plot_ko import plt  # noqa: E402

FPS = 30.0
# 정책이 추론 1회에 내놓는 action 스텝 수. SmolVLA/pi0 학습 config 전부 chunk_size=50,
# n_action_steps=50이다(outputs/train/.../pretrained_model/config.json). 녹화·제어는 30Hz라
# chunk 1개 = 50스텝 = 약 1.67초에 해당한다 — 50은 fps가 아니라 chunk 길이다.
CHUNK_SIZE = 50
# 프레임 인덱스를 나눌 값과 축 라벨.
X_UNITS = {
    "sec": (FPS, "경과 시간 (s)"),
    "chunk": (float(CHUNK_SIZE), f"action chunk (1 chunk = {CHUNK_SIZE}스텝 ≈ {CHUNK_SIZE / FPS:.2f}s)"),
}
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]

# Piper plugin / RViz player와 동일한 MotorCalibration 값.
# joint raw unit은 0.001도, gripper raw unit은 0.001mm다.
CALIBRATION_RAW = {
    "joint1": (-150_000.0, 150_000.0),
    "joint2": (0.0, 180_000.0),
    "joint3": (-170_000.0, 0.0),
    "joint4": (-100_000.0, 100_000.0),
    "joint5": (-65_000.0, 65_000.0),
    "joint6": (-100_000.0, 130_000.0),
    "gripper": (0.0, 68_000.0),
}
# piper_follower.py: joint1~6은 RANGE_M100_100, gripper만 RANGE_0_100이다.
NORM_SPAN = {name: (-100.0, 100.0) for name in JOINT_NAMES}
NORM_SPAN["gripper"] = (0.0, 100.0)

UNIT = {name: "deg" for name in JOINT_NAMES}
UNIT["gripper"] = "mm"

SHAPE_ORDER = ["circle", "triangle", "rectangle"]
# shape_position_analysis.py와 같은 팔레트. dataviz all-pairs 검증을 통과하는 3색 조합이다.
SHAPE_COLORS = {"circle": "#2a78d6", "triangle": "#eb6834", "rectangle": "#1baf7a"}

INK = "#1b1b1b"
MUTED = "#6b6b6b"
GRID = "#d9d9d6"
SURFACE = "#fcfcfb"


def denormalize(name: str, values: np.ndarray) -> np.ndarray:
    """정규화값을 joint은 도, gripper는 mm로 되돌린다."""
    raw_min, raw_max = CALIBRATION_RAW[name]
    norm_min, norm_max = NORM_SPAN[name]
    ratio = (values - norm_min) / (norm_max - norm_min)
    return (ratio * (raw_max - raw_min) + raw_min) / 1000.0


def shape_of(source: str) -> str:
    lowered = source.lower()
    for shape in SHAPE_ORDER:
        if shape in lowered:
            return shape
    raise ValueError(f"Cannot infer shape from {source!r}")


def resolve_dataset(source: str) -> pathlib.Path:
    """매니페스트 경로를 실제 폴더로 푼다.

    매니페스트의 `records/local/...` 24개는 그 뒤 날짜 폴더로 정리되면서 지금은
    `records/0813/...`에 있다. 매니페스트는 사람이 검토한 QC 기록이라 건드리지 않고,
    여기서 폴더 이름으로 다시 찾는다.
    """
    direct = REPO_ROOT / source
    if direct.is_dir():
        return direct
    name = pathlib.PurePosixPath(source).name
    hits = sorted(
        set(REPO_ROOT.joinpath("records").glob(f"*/{name}"))
        | set(REPO_ROOT.joinpath("records").glob(f"*/*/{name}"))
    )
    if len(hits) != 1:
        raise FileNotFoundError(f"{source}: {len(hits)} candidates {hits}")
    return hits[0]


def load_episodes(manifest_path: pathlib.Path, shape_filter: str | None) -> list[dict]:
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))["episodes"]
    episodes = []
    for entry in entries:
        if not entry.get("enabled", True):
            continue
        source = entry["source_dataset"]
        shape = shape_of(source)
        if shape_filter and shape != shape_filter:
            continue
        root = resolve_dataset(source)
        frame = pd.read_parquet(root / "data" / "chunk-000" / "file-000.parquet",
                                columns=["observation.state"])
        state = np.stack(frame["observation.state"].to_numpy())
        start = int(entry["start_frame"])
        end = min(int(entry["end_frame"]), len(state))
        # observation.state는 [pos×7, effort×7, vel×6] = 20차원. 앞 7개만 관절 위치다.
        positions = state[start:end, :7].astype(np.float64)
        series = {name: denormalize(name, positions[:, i]) for i, name in enumerate(JOINT_NAMES)}
        episodes.append({"name": pathlib.PurePosixPath(source).name, "shape": shape, "series": series})
    return episodes


def style_axis(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)


def make_grid(title: str, subtitle: str):
    fig, axes = plt.subplots(4, 2, figsize=(14, 12.5), facecolor=SURFACE)
    fig.suptitle(title, x=0.012, y=0.982, ha="left", fontsize=15, color=INK, weight="bold")
    fig.text(0.012, 0.955, subtitle, ha="left", fontsize=9.5, color=MUTED)
    return fig, axes.ravel()


def finish(fig, axes, handles, labels, note: str, out_path: pathlib.Path) -> None:
    legend_ax = axes[7]
    legend_ax.axis("off")
    legend_ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=10,
                     labelcolor=INK, handlelength=1.6)
    legend_ax.text(0, 0.62, note, ha="left", va="top", fontsize=9, color=MUTED,
                   transform=legend_ax.transAxes, linespacing=1.55)
    fig.tight_layout(rect=[0, 0.005, 1, 0.945])
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def mark_chunk_boundaries(ax, x_unit: str, longest_x: float) -> None:
    """chunk 축일 때 추론 경계(정수 chunk)에 눈금선을 그린다.

    정책은 chunk 경계에서만 새로 추론한다 — 경계 사이 구간은 한 번의 추론으로
    열린 루프로 실행되는 구간이라, 궤적을 이 격자에 대고 봐야 어느 chunk에서
    무슨 동작이 일어나는지 셀 수 있다.
    """
    if x_unit != "chunk":
        return
    for boundary in range(1, int(np.ceil(longest_x)) + 1):
        ax.axvline(boundary, color=GRID, linewidth=0.8, alpha=0.9, zorder=0)


def plot_overlay(episodes: list[dict], out_path: pathlib.Path, x_unit: str = "sec") -> None:
    divisor, xlabel = X_UNITS[x_unit]
    unit_note = ("에피소드 1개 = 선 1개, 도형별 색."
                 if x_unit == "sec" else
                 "세로 눈금선 = 추론 경계(정수 chunk). 에피소드 1개 = 선 1개, 도형별 색.")
    fig, axes = make_grid(
        "관절별 상태 시계열 — 315개 원본 녹화 전체",
        f"observation.state의 joint*.pos, QC 구간(start_frame~end_frame)만. {unit_note}",
    )
    longest_x = max(len(e["series"]["joint1"]) for e in episodes) / divisor
    for index, name in enumerate(JOINT_NAMES):
        ax = axes[index]
        style_axis(ax)
        mark_chunk_boundaries(ax, x_unit, longest_x)
        for episode in episodes:
            values = episode["series"][name]
            ax.plot(np.arange(len(values)) / divisor, values,
                    color=SHAPE_COLORS[episode["shape"]], linewidth=0.6, alpha=0.10)
        ax.set_title(f"{name}  ({UNIT[name]})", fontsize=11, color=INK, loc="left", pad=6)
        ax.set_xlabel(xlabel, fontsize=8.5, color=MUTED)

    durations = np.array([len(e["series"]["joint1"]) / divisor for e in episodes])
    handles = [plt.Line2D([], [], color=SHAPE_COLORS[s], linewidth=2.5) for s in SHAPE_ORDER]
    counts = {s: sum(e["shape"] == s for e in episodes) for s in SHAPE_ORDER}
    unit_label = "s" if x_unit == "sec" else " chunk"
    note = (f"에피소드 {len(episodes)}개\n"
            f"길이 중앙값 {np.median(durations):.1f}{unit_label} "
            f"(최소 {durations.min():.1f} / 최대 {durations.max():.1f})\n\n"
            "선이 오른쪽으로 갈수록 얇아지는 것은\n에피소드마다 길이가 달라 남은 개수가\n줄기 때문이다.")
    finish(fig, axes, handles, [f"{s} ({counts[s]}개)" for s in SHAPE_ORDER], note, out_path)


def plot_median(episodes: list[dict], out_path: pathlib.Path) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, 101)
    resampled = {name: {shape: [] for shape in SHAPE_ORDER} for name in JOINT_NAMES}
    for episode in episodes:
        for name in JOINT_NAMES:
            values = episode["series"][name]
            progress = np.linspace(0.0, 1.0, len(values))
            resampled[name][episode["shape"]].append(np.interp(grid, progress, values))

    fig, axes = make_grid(
        "관절별 상태 시계열 — 진행률 정렬 중앙값",
        "에피소드마다 길이가 달라 0~100% 진행률로 리샘플링한 뒤, 도형별 중앙값(선)과 25~75% 구간(밴드).",
    )
    for index, name in enumerate(JOINT_NAMES):
        ax = axes[index]
        style_axis(ax)
        for shape in SHAPE_ORDER:
            stack = np.array(resampled[name][shape])
            if not len(stack):
                continue
            low, mid, high = np.percentile(stack, [25, 50, 75], axis=0)
            ax.fill_between(grid * 100, low, high, color=SHAPE_COLORS[shape], alpha=0.16, linewidth=0)
            ax.plot(grid * 100, mid, color=SHAPE_COLORS[shape], linewidth=2.0)
        ax.set_title(f"{name}  ({UNIT[name]})", fontsize=11, color=INK, loc="left", pad=6)
        ax.set_xlabel("에피소드 진행률 (%)", fontsize=8.5, color=MUTED)
        ax.set_xlim(0, 100)

    handles = [plt.Line2D([], [], color=SHAPE_COLORS[s], linewidth=2.5) for s in SHAPE_ORDER]
    counts = {s: sum(e["shape"] == s for e in episodes) for s in SHAPE_ORDER}
    note = ("밴드가 넓은 구간 = 에피소드마다 자세가\n갈리는 구간(도형 위치가 다르니 당연),\n"
            "좁은 구간 = 모든 에피소드가 같은 자세를\n지나가는 구간(집기·복귀 동작).")
    finish(fig, axes, handles, [f"{s} ({counts[s]}개)" for s in SHAPE_ORDER], note, out_path)
    return resampled


# 실제 시간축에서 중앙값을 낼 때, 시각 t를 넘긴 에피소드만 그 시점 통계에 기여한다.
# 끝으로 갈수록 표본이 "오래 걸린 에피소드"로만 남는 생존 편향이라, 남은 비율에 따라
# 선을 나눠 그린다 — 실측(2026-08-31): t=15s 257개, t=20s 64개, t=25s 8개.
SOLID_COVERAGE = 0.50   # 이 비율 이상 남아 있는 구간만 실선
FADED_COVERAGE = 0.10   # 이 아래로 떨어지면 아예 그리지 않는다


def plot_median_time(episodes: list[dict], out_path: pathlib.Path, x_unit: str = "sec") -> None:
    """진행률로 늘리지 않고 실제 경과 시간축(또는 chunk 축) 그대로 도형별 중앙값을 낸다."""
    divisor, xlabel = X_UNITS[x_unit]
    longest = max(len(e["series"]["joint1"]) for e in episodes)
    times = np.arange(longest) / divisor

    padded = {name: {} for name in JOINT_NAMES}
    survivors = {}
    for shape in SHAPE_ORDER:
        picked = [e for e in episodes if e["shape"] == shape]
        for name in JOINT_NAMES:
            stack = np.full((len(picked), longest), np.nan)
            for row, episode in enumerate(picked):
                values = episode["series"][name]
                stack[row, :len(values)] = values
            padded[name][shape] = stack
        # 관절과 무관하게 길이만으로 정해지므로 joint1 것을 그대로 쓴다.
        survivors[shape] = np.count_nonzero(~np.isnan(padded["joint1"][shape]), axis=0)

    if x_unit == "sec":
        title = "관절별 상태 시계열 — 실제 시간축 중앙값"
        subtitle = ("QC 구간(start_frame~end_frame) 시작을 0s로 맞춘 실제 경과 시간. "
                    "도형별 중앙값(선)과 25~75% 구간(밴드).")
    else:
        title = "관절별 상태 시계열 — action chunk 축 중앙값"
        subtitle = (f"QC 구간 시작을 chunk 0으로 맞췄다(1 chunk = {CHUNK_SIZE}스텝 ≈ {CHUNK_SIZE / FPS:.2f}s, "
                    "30Hz 녹화). 세로 눈금선 = 추론 경계.")
    fig, axes = make_grid(title, subtitle)
    for index, name in enumerate(JOINT_NAMES):
        ax = axes[index]
        style_axis(ax)
        mark_chunk_boundaries(ax, x_unit, times[-1])
        for shape in SHAPE_ORDER:
            stack = padded[name][shape]
            alive = survivors[shape]
            total = stack.shape[0]
            keep = alive >= max(2, int(total * FADED_COVERAGE))
            solid = alive >= total * SOLID_COVERAGE
            with np.errstate(all="ignore"):
                low, mid, high = np.nanpercentile(stack[:, keep], [25, 50, 75], axis=0)
            t_keep = times[keep]
            color = SHAPE_COLORS[shape]
            ax.fill_between(t_keep, low, high, color=color, alpha=0.16, linewidth=0)
            # 실선과 점선이 끊겨 보이지 않게 경계 한 점을 양쪽에 겹쳐 넣는다.
            edge = int(solid.sum())
            ax.plot(t_keep[:edge], mid[:edge], color=color, linewidth=2.0)
            ax.plot(t_keep[max(edge - 1, 0):], mid[max(edge - 1, 0):],
                    color=color, linewidth=1.3, linestyle=(0, (3, 2)), alpha=0.75)
        ax.set_title(f"{name}  ({UNIT[name]})", fontsize=11, color=INK, loc="left", pad=6)
        ax.set_xlabel(xlabel, fontsize=8.5, color=MUTED)
        ax.set_xlim(0, times[-1])

    # 8번째 칸은 범례 대신 생존 곡선 — 어느 시점부터 표본이 무너지는지 직접 보여준다.
    ax = axes[7]
    style_axis(ax)
    mark_chunk_boundaries(ax, x_unit, times[-1])
    for shape in SHAPE_ORDER:
        ax.plot(times, survivors[shape], color=SHAPE_COLORS[shape], linewidth=2.0)
    total_all = len(episodes) / len(SHAPE_ORDER)
    ax.axhline(total_all * SOLID_COVERAGE, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(times[-1] * 0.015, total_all * SOLID_COVERAGE + 2, "실선/점선 경계 (표본 50%)",
            ha="left", va="bottom", fontsize=8, color=MUTED)
    ax.set_title("각 시점에 아직 진행 중인 에피소드 수", fontsize=11, color=INK, loc="left", pad=6)
    ax.set_xlabel(xlabel, fontsize=8.5, color=MUTED)
    ax.set_xlim(0, times[-1])

    handles = [plt.Line2D([], [], color=SHAPE_COLORS[s], linewidth=2.5) for s in SHAPE_ORDER]
    counts = {s: sum(e["shape"] == s for e in episodes) for s in SHAPE_ORDER}
    labels = [f"{s} ({counts[s]}개)" for s in SHAPE_ORDER]
    handles.append(plt.Line2D([], [], color=MUTED, linewidth=1.3, linestyle=(0, (3, 2))))
    labels.append("표본 50% 미만 (참고용)")
    # 생존 곡선은 왼쪽이 높고 오른쪽이 낮다 — 범례는 비는 오른쪽 위 대신 곡선과
    # 겹치지 않는 오른쪽 중단에 둔다.
    ax.legend(handles, labels, loc="center right", frameon=False, fontsize=9,
              labelcolor=INK, handlelength=1.6)
    fig.tight_layout(rect=[0, 0.005, 1, 0.945])
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def write_stats(episodes: list[dict], out_path: pathlib.Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["joint", "shape", "n", "unit", "start_median", "end_median",
                         "min", "max", "range_median", "range_p95"])
        for name in JOINT_NAMES:
            for shape in [*SHAPE_ORDER, "all"]:
                picked = [e for e in episodes if shape in ("all", e["shape"])]
                if not picked:
                    continue
                series = [e["series"][name] for e in picked]
                spans = np.array([s.max() - s.min() for s in series])
                writer.writerow([
                    name, shape, len(picked), UNIT[name],
                    f"{np.median([s[0] for s in series]):.2f}",
                    f"{np.median([s[-1] for s in series]):.2f}",
                    f"{min(s.min() for s in series):.2f}",
                    f"{max(s.max() for s in series):.2f}",
                    f"{np.median(spans):.2f}",
                    f"{np.percentile(spans, 95):.2f}",
                ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=pathlib.Path,
                        default=REPO_ROOT / "configs" / "erase_shape_315_manifest.json")
    parser.add_argument("--shape", choices=SHAPE_ORDER, default=None,
                        help="한 도형만 그린다. 기본은 315개 전체.")
    parser.add_argument("--x-unit", choices=["sec", "chunk", "both"], default="both",
                        help="시간축 그림의 x축 단위. chunk는 정책 chunk_size(=50스텝) 단위다.")
    parser.add_argument("--out-dir", type=pathlib.Path,
                        default=REPO_ROOT / "outputs" / "analysis" / "erase_shape" / "joint_state_timeseries")
    args = parser.parse_args()

    if plot_ko.use_korean() is None:
        print("[WARN] 한글 폰트를 못 찾았다 — 라벨이 두부(□)로 나온다")

    episodes = load_episodes(args.manifest, args.shape)
    if not episodes:
        raise SystemExit("No episodes matched")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    units = ["sec", "chunk"] if args.x_unit == "both" else [args.x_unit]
    for unit in units:
        suffix = "" if unit == "sec" else "_chunk"
        plot_overlay(episodes, args.out_dir / f"joint_timeseries_overlay{suffix}.png", unit)
        plot_median_time(episodes, args.out_dir / f"joint_timeseries_median_time{suffix}.png", unit)
    # 진행률(0~100%) 축은 무차원이라 chunk 단위가 의미가 없다 — 한 장만 만든다.
    plot_median(episodes, args.out_dir / "joint_timeseries_median.png")
    write_stats(episodes, args.out_dir / "joint_timeseries_stats.csv")
    print(f"[SAVE] {args.out_dir}  (episodes={len(episodes)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
