#!/usr/bin/env python3
"""policy_conditioning_probe.py가 남긴 npz들을 비교해 **조건부 붕괴**를 판정한다.

지표가 두 개다. 둘 다 **추론 seed를 고정한 상태**에서만 의미가 있다 — pi0/SmolVLA는
둘 다 flow matching이라 초기 noise가 같으면 ODE 적분이 결정론적이고, 그때 출력 차이는
오직 관측 o의 차이에서만 온다. seed를 안 고정하면 관측을 무시하는 정책도 marginal
분포에서 샘플이 흩어져 아래 분산비가 1에 가깝게 나와 버린다.

(용어 주의: flow matching은 같은 o에 대한 다봉 분포를 평균내는 mode averaging은 하지
않는다. 여기서 재는 것은 그게 아니라 **조건 입력 o를 아예 안 쓰는** conditioning
collapse다. 둘은 다른 실패 양상이다.)

### 지표 1 — 진행률 지점 p, 관절 j에 대해:

    spread(정책) = std_over_episodes( 예측 chunk의 첫 스텝 )
    spread_ratio = spread(정책) / spread(정답)

정답은 도형이 보드 어디 있느냐에 따라 에피소드마다 벌어진다. 예측이 그만큼 안
벌어지면(비율 << 1) 그 정책은 관측을 무시하고 평균 궤적을 재생하고 있다는 뜻이다.
MAE는 평균만 맞혀도 낮아질 수 있어 이 구분을 못 한다 — 그래서 둘을 같이 본다.

### 왜 chunk의 "변위"를 보는가 (2026-09-01 수정)

처음에는 chunk 첫 스텝의 **절대값**으로 쟀는데 SmolVLA가 집기 구간에서도 상관 0.997이
나왔다. 집기 구간은 도형 위치와 무관해서 시각 정보 없이도 맞힐 수 있는 곳이라 이건
지표가 틀렸다는 신호였다. 원인: `observation.state`에 현재 관절 위치가 그대로 들어가므로
정책은 이미지를 안 봐도 "지금 위치를 그대로 출력"하면 첫 스텝을 맞힌다. 에피소드마다
현재 위치가 다르니 상관계수가 자동으로 1에 가까워진다 — state 복사를 조건부 능력으로
착각한 것이다.

그래서 chunk 안의 **변위** `chunk[k] - chunk[0]`을 쓴다. 공통의 "지금 어디에 있나"
성분이 빠지고 "앞으로 어디로 갈 계획인가"만 남는다. 실측(SmolVLA, joint1, 진행률 70%):
절대값 상관은 k=1에서 0.998이지만 변위 상관은 0.609다. k를 늘리면 정답 변위 자체가
커져(k=49에서 std 14~19도) 조건부 신호가 뚜렷해진다.

### 지표 2 — 에피소드 간 상관계수

    corr = pearson_over_episodes( 예측, 정답 )

분산비는 `std(정답)`을 분모로 쓰는데, 정답의 흩어짐에는 도형 위치가 만든 변동뿐 아니라
사람 시연자의 자연스러운 편차도 섞여 있다. 그래서 "분산비 1.0 = 정답"이라고 읽으면 안
되고 정책끼리 비교하는 용도로만 써야 한다. 상관계수는 크기와 무관하게 "예측이 정답을
에피소드 단위로 따라가느냐"만 보므로 이 문제에서 자유롭다 — 관측을 무시하면 0,
제대로 쓰면 1에 가깝다. **판정은 상관계수를 우선으로 한다.**

정규화값을 도(gripper는 mm)로 되돌려 비교하므로 관절 간 크기도 직접 읽을 수 있다
(joint_state_timeseries.py와 같은 환산).

사용:
    python scripts/tasks/erase_shape/analysis/compare_policy_conditioning.py \\
        --probes outputs/analysis/erase_shape/policy_conditioning/probe_{pi0,smolvla}_315.npz
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
ANALYSIS_DIR = REPO_ROOT / "scripts" / "tasks" / "erase_shape" / "analysis"
LIB_DIR = REPO_ROOT / "scripts" / "tasks" / "erase_shape" / "lib"
for path in (str(ANALYSIS_DIR), str(LIB_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import plot_ko  # noqa: E402
from plot_ko import plt  # noqa: E402
from joint_state_timeseries import (  # noqa: E402
    GRID, INK, JOINT_NAMES, MUTED, SURFACE, UNIT, denormalize, style_axis,
)

# 정책별 색. dataviz all-pairs 검증을 통과한 3색 조합에서 앞 두 개를 쓴다.
POLICY_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
TRUTH_COLOR = "#5a5a5a"


def load(paths: list[pathlib.Path]) -> list[dict]:
    runs = []
    for path in paths:
        data = np.load(path, allow_pickle=True)
        label = str(data["label"])
        pred = data["pred"].astype(np.float64)
        truth = data["truth"].astype(np.float64)
        # 저장은 정규화값 — 관절은 도, gripper는 mm로 되돌린다.
        for index, name in enumerate(JOINT_NAMES):
            pred[..., index] = denormalize(name, pred[..., index])
            truth[..., index] = denormalize(name, truth[..., index])
        runs.append({
            "label": label, "pred": pred, "truth": truth,
            "probes": data["probes"], "episodes": data["episodes"],
            "latency_s": data["latency_s"],
        })
    base = runs[0]
    for run in runs[1:]:
        if not np.array_equal(run["episodes"], base["episodes"]):
            raise SystemExit("npz들의 에피소드 집합이 다르다 — 같은 --episodes로 다시 돌릴 것")
        if not np.allclose(run["probes"], base["probes"]):
            raise SystemExit("npz들의 진행률 지점이 다르다")
    return runs


def metrics(run: dict, horizon: int) -> dict:
    """chunk 변위(= chunk[horizon] - chunk[0])로 조건부 반응을 잰다.

    절대값이 아니라 변위를 쓰는 이유는 모듈 docstring 참고 — 절대값은 정책이
    현재 관절 위치를 그대로 복사하기만 해도 상관계수가 1에 가까워진다.
    """
    pred, truth = run["pred"], run["truth"]
    step = min(horizon, pred.shape[2] - 1)
    pred_first = pred[:, :, step, :] - pred[:, :, 0, :]
    truth_first = truth[:, :, step, :] - truth[:, :, 0, :]
    valid = ~np.isnan(pred_first[..., 0]) & ~np.isnan(truth_first[..., 0])
    spread_pred = np.full((pred.shape[1], 7), np.nan)
    spread_truth = np.full((pred.shape[1], 7), np.nan)
    mae = np.full((pred.shape[1], 7), np.nan)
    corr = np.full((pred.shape[1], 7), np.nan)
    for column in range(pred.shape[1]):
        rows = valid[:, column]
        if rows.sum() < 3:
            continue
        p_col, t_col = pred_first[rows, column], truth_first[rows, column]
        spread_pred[column] = p_col.std(axis=0)
        spread_truth[column] = t_col.std(axis=0)
        mae[column] = np.abs(p_col - t_col).mean(axis=0)
        for joint in range(7):
            # 한쪽이 상수면 상관계수가 정의되지 않는다 — 그 경우가 곧 "관측 무시"다.
            if p_col[:, joint].std() < 1e-9 or t_col[:, joint].std() < 1e-9:
                corr[column, joint] = 0.0
            else:
                corr[column, joint] = np.corrcoef(p_col[:, joint], t_col[:, joint])[0, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = spread_pred / spread_truth
    return {"spread_pred": spread_pred, "spread_truth": spread_truth,
            "ratio": ratio, "mae": mae, "corr": corr}


PLOTS = {
    "corr": {
        "title": "조건부 정보를 쓰고 있는가 — 예측 vs 정답의 에피소드 간 상관계수",
        "subtitle": "chunk 변위(chunk[49]-chunk[0])로 계산 — 현재 위치 복사 성분을 뺀 값. 1이면 도형 위치에 맞춰 갈 곳을 정한다는 뜻, 0이면 관측과 무관. 추론 seed 고정.",
        "ylabel": "상관계수",
        "reference": 1.0,
        "ylim": (-0.35, 1.05),
    },
    "ratio": {
        "title": "예측 변위의 에피소드 간 분산 / 정답 변위의 분산",
        "subtitle": "보조 지표. 정답 분산에는 시연자 편차도 섞여 있어 '1.0=정답'으로 읽지 말고 정책 간 비교로만 볼 것.",
        "ylabel": "분산비",
        "reference": 1.0,
        "ylim": None,
    },
}


def plot(runs: list[dict], stats: list[dict], out_path: pathlib.Path, key: str) -> None:
    spec = PLOTS[key]
    probes = runs[0]["probes"] * 100
    fig, axes = plt.subplots(4, 2, figsize=(14, 12.5), facecolor=SURFACE)
    fig.suptitle(spec["title"], x=0.012, y=0.982, ha="left", fontsize=15,
                 color=INK, weight="bold")
    fig.text(0.012, 0.955, spec["subtitle"], ha="left", fontsize=9.5, color=MUTED)
    axes = axes.ravel()
    for index, name in enumerate(JOINT_NAMES):
        ax = axes[index]
        style_axis(ax)
        ax.axhline(spec["reference"], color=TRUTH_COLOR, linewidth=1.0,
                   linestyle=(0, (4, 3)), zorder=1)
        if key == "corr":
            ax.axhline(0.0, color=GRID, linewidth=1.0, zorder=1)
        for run, stat, color in zip(runs, stats, POLICY_COLORS):
            ax.plot(probes, stat[key][:, index], color=color, linewidth=2.0,
                    marker="o", markersize=5)
        if spec["ylim"]:
            ax.set_ylim(*spec["ylim"])
        else:
            ax.set_ylim(0, max(1.35, np.nanmax([s[key][:, index] for s in stats]) * 1.1))
        ax.set_title(f"{name}  ({UNIT[name]})", fontsize=11, color=INK, loc="left", pad=6)
        ax.set_xlabel("에피소드 진행률 (%)", fontsize=8.5, color=MUTED)
        ax.set_ylabel(spec["ylabel"], fontsize=8.5, color=MUTED)

    legend_ax = axes[7]
    legend_ax.axis("off")
    handles = [plt.Line2D([], [], color=c, linewidth=2.5, marker="o", markersize=5)
               for c in POLICY_COLORS[:len(runs)]]
    handles.append(plt.Line2D([], [], color=TRUTH_COLOR, linewidth=1.0, linestyle=(0, (4, 3))))
    labels = [run["label"] for run in runs] + ["기준선 1.0"]
    legend_ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=10,
                     labelcolor=INK, handlelength=1.8)
    note = ("집기 구간(진행률 ~35%)은 모든 에피소드가\n같은 동작이라 정답 분산 자체가 작다 —\n"
            "여기 비율은 해석하지 말 것.\n\n"
            "판단은 지우기 구간(55~85%)에서 한다.")
    legend_ax.text(0, 0.66, note, ha="left", va="top", fontsize=9, color=MUTED,
                   transform=legend_ax.transAxes, linespacing=1.55)
    fig.tight_layout(rect=[0, 0.005, 1, 0.945])
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def write_csv(runs: list[dict], stats: list[dict], out_path: pathlib.Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["policy", "progress_pct", "joint", "unit", "corr",
                         "spread_pred", "spread_truth", "spread_ratio", "mae"])
        for run, stat in zip(runs, stats):
            for column, fraction in enumerate(run["probes"]):
                for index, name in enumerate(JOINT_NAMES):
                    writer.writerow([
                        run["label"], f"{fraction * 100:.0f}", name, UNIT[name],
                        f"{stat['corr'][column, index]:.3f}",
                        f"{stat['spread_pred'][column, index]:.3f}",
                        f"{stat['spread_truth'][column, index]:.3f}",
                        f"{stat['ratio'][column, index]:.3f}",
                        f"{stat['mae'][column, index]:.3f}",
                    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probes", type=pathlib.Path, nargs="+", required=True)
    parser.add_argument("--horizon", type=int, default=49,
                        help="chunk 안에서 변위를 잴 스텝 (기본 49 = chunk 끝)")
    parser.add_argument("--out-dir", type=pathlib.Path,
                        default=REPO_ROOT / "outputs" / "analysis" / "erase_shape" / "policy_conditioning")
    args = parser.parse_args()
    plot_ko.use_korean()

    runs = load(args.probes)
    stats = [metrics(run, args.horizon) for run in runs]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot(runs, stats, args.out_dir / "conditioning_corr.png", "corr")
    plot(runs, stats, args.out_dir / "conditioning_ratio.png", "ratio")
    write_csv(runs, stats, args.out_dir / "conditioning_metrics.csv")

    # 지우기 구간(진행률 50% 이상) 요약 — 판정은 여기서 한다.
    print(f"(chunk 변위 기준, horizon={args.horizon})")
    print(f"{'정책':22s} {'구간':>12s} {'상관계수':>9s} {'분산비':>8s} {'MAE joint1':>12s} {'추론(ms)':>10s}")
    for run, stat in zip(runs, stats):
        for name, mask in (("집기 0~35%", run["probes"] < 0.5),
                           ("지우기 55~85%", run["probes"] >= 0.5)):
            corr = np.nanmedian(stat["corr"][mask][:, :6])
            ratio = np.nanmedian(stat["ratio"][mask][:, :6])
            mae1 = np.nanmean(stat["mae"][mask][:, 0])
            print(f"{run['label']:22s} {name:>12s} {corr:9.3f} {ratio:8.3f} {mae1:11.2f}° "
                  f"{np.median(run['latency_s']) * 1000:9.0f}")
    print(f"\n[SAVE] {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
