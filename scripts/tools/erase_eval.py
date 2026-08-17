#!/usr/bin/env python3
"""erase_run 로그들을 모아 §8 성능 표를 만든다 (통계 프로토콜은 설계 문서 §8.1).

입력은 `erase_run.py --out`이 남긴 JSON들. 조건별로 디렉터리를 나눠 두거나
파일명에 조건을 넣어두고 그걸 라벨로 준다:

    python erase_eval.py \
        --condition baseline:runs/baseline/*.json \
        --condition gate:runs/gate/*.json \
        --condition gate_hil:runs/gate_hil/*.json \
        --out outputs/analysis/eval --plot

왜 별도 도구인가 — 성공률 점추정만 찍는 건 한 줄이면 되지만, n=20에서 "65%"는
사실상 정보가 없다(95% 구간이 ±20%p를 넘는다). 여기서 붙이는 것:

  - Wilson score 구간 (Wald는 n이 작거나 p가 0/1 근처면 깨진다)
  - 조건 간 Fisher exact (카이제곱 근사 조건을 n이 못 맞춘다)
  - 스텝 수는 Mann-Whitney U + bootstrap 구간 (분포가 비대칭이라 t-test 부적합)
  - 효과 크기: Cohen's h (비율), Cliff's delta (순서)

의존성은 numpy와 scipy만. scipy가 없으면 검정만 건너뛰고 나머지는 낸다.
"""

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    from scipy import stats as _st
except ImportError:  # 검정 없이도 표는 나와야 한다
    _st = None


# ── 통계 ────────────────────────────────────────────────
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """이항 비율의 Wilson score 신뢰구간.

    Wald(p ± z·sqrt(p(1-p)/n))를 안 쓰는 이유: k=n이면 폭이 0이 되고(20/20을
    [100%, 100%]로 보고하게 된다), p가 0에 가까우면 하한이 음수로 샌다.
    Wilson은 둘 다 정상이다.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def cohens_h(p1: float, p2: float) -> float:
    """두 비율의 효과 크기. 0.2/0.5/0.8 = small/medium/large (관례)."""
    phi = lambda p: 2 * math.asin(math.sqrt(max(0.0, min(1.0, p))))  # noqa: E731
    return phi(p1) - phi(p2)


def cliffs_delta(a, b) -> float:
    """순서형 효과 크기. -1~+1. |d| 0.147/0.33/0.474 = small/medium/large.

    a의 값이 b보다 큰 쌍의 비율에서 작은 쌍의 비율을 뺀다. 평균 차이와 달리
    분포 모양을 가정하지 않아서, 스텝 수처럼 비대칭인 값에 쓸 수 있다.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    diff = a[:, None] - b[None, :]
    return float((diff > 0).mean() - (diff < 0).mean())


def bootstrap_ci(vals, stat=np.median, n_boot=10000, seed=0, alpha=0.05):
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.choice(vals, size=(n_boot, len(vals)), replace=True)
    boots = stat(draws, axis=1)
    return tuple(float(v) for v in np.percentile(boots, [100 * alpha / 2,
                                                         100 * (1 - alpha / 2)]))


# ── 로그 읽기 ────────────────────────────────────────────
def load_run(path: Path) -> dict | None:
    """erase_run 로그 1개 = 시도 리스트. 한 '트라이얼'로 눕힌다.

    한 트라이얼의 성공은 **마지막 시도의 성공**이다(재시도해서 성공했으면 성공).
    스텝 수는 재시도 전부의 합이다 — 재시도 비용을 숨기면 게이트가 공짜로 보인다.
    """
    try:
        attempts = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  [건너뜀] {path}: {exc}", file=sys.stderr)
        return None
    if not attempts:
        return None
    last = attempts[-1]
    return {
        "file": str(path),
        "success": bool(last.get("success")),
        "attempts": len(attempts),
        "steps": int(sum(a.get("steps", 0) for a in attempts)),
        "interventions": int(sum(a.get("interventions", 0) for a in attempts)),
        "target_erased": last.get("target_erased"),
        "max_distractor": last.get("max_distractor_erased"),
        "target_found": bool(last.get("target_found", True)),
        "residual_where": (last.get("residual") or {}).get("where"),
    }


def summarize_condition(name: str, runs: list[dict]) -> dict:
    n = len(runs)
    k = sum(r["success"] for r in runs)
    lo, hi = wilson(k, n)
    steps = [r["steps"] for r in runs]
    total_steps = sum(steps)
    total_iv = sum(r["interventions"] for r in runs)
    erased = [r["target_erased"] for r in runs if r["target_erased"] is not None]
    return {
        "조건": name,
        "n": n,
        "성공": k,
        "성공률": k / n if n else 0.0,
        "성공률 95% CI": (lo, hi),
        "평균 시도": float(np.mean([r["attempts"] for r in runs])) if n else 0.0,
        "스텝 중앙값": float(np.median(steps)) if n else 0.0,
        "스텝 CI": bootstrap_ci(steps),
        "개입률": (total_iv / total_steps) if total_steps else 0.0,
        "erased 평균": float(np.mean(erased)) if erased else float("nan"),
        # 실패한 트라이얼이 '거의 다 지움'인지 '전혀 못 지움'인지 — 이진 판정
        # 연구가 못 내는 표다(§8). 평균만 내면 이 구분이 지워진다.
        "실패 시 erased": [r["target_erased"] for r in runs
                          if not r["success"] and r["target_erased"] is not None],
        "_runs": runs,
    }


def compare(a: dict, b: dict) -> dict:
    """b(기준) 대비 a. -> 표에 붙일 검정 결과."""
    out = {"대비": f"{a['조건']} vs {b['조건']}"}
    ka, na = a["성공"], a["n"]
    kb, nb = b["성공"], b["n"]
    out["성공률 차이(%p)"] = 100 * (a["성공률"] - b["성공률"])
    out["Cohen's h"] = cohens_h(a["성공률"], b["성공률"])
    sa = [r["steps"] for r in a["_runs"]]
    sb = [r["steps"] for r in b["_runs"]]
    out["Cliff's delta(스텝)"] = cliffs_delta(sa, sb)
    if _st is not None:
        # Fisher: n이 작아 카이제곱 근사가 성립하지 않는다
        out["Fisher p"] = float(
            _st.fisher_exact([[ka, na - ka], [kb, nb - kb]]).pvalue
        )
        if sa and sb:
            out["Mann-Whitney p"] = float(
                _st.mannwhitneyu(sa, sb, alternative="two-sided").pvalue
            )
    return out


def sensitivity(conditions: dict, thresholds) -> list[dict]:
    """성공 임계를 훑으며 결론이 뒤집히는 지점을 찾는다.

    주의: 이건 **부록**이다. 주 결과는 사전 등록한 0.9/0.10으로 낸다(§8.1).
    여기서 좋아 보이는 값을 골라 주 결과로 쓰면 그건 평가가 아니라 튜닝이다.
    """
    rows = []
    for th in thresholds:
        row = {"임계": round(float(th), 3)}
        for name, c in conditions.items():
            vals = [r["target_erased"] for r in c["_runs"] if r["target_erased"] is not None]
            k = sum(1 for v in vals if v >= th)
            row[name] = f"{k}/{len(vals)} ({k/len(vals):.0%})" if vals else "-"
        rows.append(row)
    return rows


# ── 출력 ────────────────────────────────────────────────
def print_table(conds: list[dict]):
    print("\n" + "═" * 92)
    print(f"{'조건':<14}{'n':>4}{'성공률':>10}{'95% CI':>18}"
          f"{'평균시도':>9}{'스텝(중앙)':>12}{'개입률':>9}{'erased':>9}")
    print("─" * 92)
    for c in conds:
        lo, hi = c["성공률 95% CI"]
        ci = f"[{lo:.0%}, {hi:.0%}]"
        steps = f"{c['스텝 중앙값']:.0f}"
        print(f"{c['조건']:<14}{c['n']:>4}{c['성공률']:>9.0%}{ci:>18}"
              f"{c['평균 시도']:>9.2f}{steps:>12}"
              f"{c['개입률']:>8.1%}{c['erased 평균']:>9.3f}")
    print("═" * 92)
    print("  성공률 CI는 Wilson score. 구간이 겹치는 두 조건은 이 n으로 구분 못 한다.")


def plot(conds: list[dict], path: Path, note: str | None = None):
    from plot_ko import plt, use_korean
    use_korean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    names = [c["조건"] for c in conds]
    rates = [100 * c["성공률"] for c in conds]
    errs = np.array([[100 * (c["성공률"] - c["성공률 95% CI"][0]) for c in conds],
                     [100 * (c["성공률 95% CI"][1] - c["성공률"]) for c in conds]])
    ax1.bar(names, rates, color="#2e8b57", width=0.55)
    ax1.errorbar(names, rates, yerr=errs, fmt="none", ecolor="black", capsize=6)
    for i, c in enumerate(conds):
        ax1.text(i, rates[i] + 3, f"{c['성공']}/{c['n']}", ha="center", fontsize=10)
    ax1.set_ylabel("성공률 (%)")
    ax1.set_ylim(0, 108)
    ax1.set_title("조건별 성공률 (막대) + Wilson 95% 구간\n"
                  "구간이 겹치면 n이 부족하다는 뜻이다")

    # 실패 트라이얼의 잔량 분포 — 이진 판정 연구가 못 내는 그림
    for c in conds:
        vals = c["실패 시 erased"]
        if vals:
            xs = np.sort(vals)
            ax2.step(xs, np.arange(1, len(xs) + 1) / len(xs), where="post",
                     label=f"{c['조건']} (n={len(xs)})", lw=2)
    ax2.axvline(0.9, color="crimson", ls="--", label="성공 임계 0.9")
    ax2.set_xlabel("실패한 트라이얼의 target_erased")
    ax2.set_ylabel("누적 비율 (ECDF)")
    ax2.set_xlim(0, 1)
    ax2.set_title("실패가 '거의 다 지움'인가 '전혀 못 지움'인가\n"
                  "평균만 내면 지워지는 구분")
    ax2.legend(fontsize=9)

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if note:
        # 그림은 슬라이드·문서로 혼자 돌아다닌다. 어떤 데이터로 그린 건지(세션,
        # 날짜, 합성 여부)를 그림 안에 박아두지 않으면 나중에 구분이 안 된다.
        fig.text(0.5, 0.5, note, fontsize=34, color="red", alpha=0.16,
                 ha="center", va="center", rotation=18, zorder=10)
        fig.text(0.01, 0.01, note, fontsize=9, color="crimson")
    fig.savefig(path, dpi=130)
    print(f"[INFO] 그림 저장: {path}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--condition", action="append", required=True, metavar="NAME:GLOB",
                   help="조건 이름과 로그 glob. 여러 번 줄 수 있다")
    p.add_argument("--baseline", default=None,
                   help="비교 기준 조건 이름 (기본: 첫 번째)")
    p.add_argument("--out", type=Path, default=Path("outputs/analysis/eval"))
    p.add_argument("--plot", action="store_true")
    p.add_argument("--note", default=None,
                   help="그림에 박을 표기 (예: '합성 데이터 — 도구 검증용', 세션 ID)")
    p.add_argument("--sensitivity", action="store_true",
                   help="성공 임계 민감도 표 (부록용 — 주 결과로 쓰지 말 것)")
    a = p.parse_args(argv)

    conditions = {}
    for spec in a.condition:
        if ":" not in spec:
            p.error(f"--condition은 NAME:GLOB 형식이어야 한다: {spec}")
        name, pattern = spec.split(":", 1)
        runs = [r for f in sorted(glob.glob(pattern)) if (r := load_run(Path(f)))]
        if not runs:
            print(f"[WARN] 조건 '{name}': 로그를 못 찾음 ({pattern})", file=sys.stderr)
            continue
        conditions[name] = summarize_condition(name, runs)
        print(f"[INFO] {name}: 트라이얼 {len(runs)}개")

    if not conditions:
        print("읽을 로그가 없다.", file=sys.stderr)
        return 1

    conds = list(conditions.values())
    print_table(conds)

    base_name = a.baseline or conds[0]["조건"]
    if base_name in conditions and len(conds) > 1:
        print(f"\n── {base_name} 대비 " + "─" * 40)
        for c in conds:
            if c["조건"] == base_name:
                continue
            for k, v in compare(c, conditions[base_name]).items():
                print(f"  {k}: {v if isinstance(v, str) else f'{v:+.4f}'}")
            print()

    if a.sensitivity:
        print("── 임계 민감도 (부록) " + "─" * 30)
        for row in sensitivity(conditions, np.arange(0.7, 1.001, 0.05)):
            print("  " + "  ".join(f"{k}={v}" for k, v in row.items()))

    a.out.mkdir(parents=True, exist_ok=True)
    payload = {name: {k: v for k, v in c.items() if not k.startswith("_")}
               for name, c in conditions.items()}
    (a.out / "summary.json").write_text(json.dumps(payload, ensure_ascii=False,
                                                   indent=2, default=float))
    print(f"\n[INFO] 요약 저장: {a.out / 'summary.json'}")
    if a.plot:
        plot(conds, a.out / "eval.png", a.note)
    return 0


def _selftest():
    assert wilson(20, 20)[1] == 1.0 and wilson(20, 20)[0] < 1.0, wilson(20, 20)
    # Wald였다면 폭이 0이 됐을 자리
    assert wilson(20, 20)[0] > 0.8, wilson(20, 20)
    assert wilson(0, 20)[0] == 0.0, wilson(0, 20)
    assert abs(cohens_h(0.5, 0.5)) < 1e-9
    assert cliffs_delta([5, 6, 7], [1, 2, 3]) == 1.0
    assert cliffs_delta([1, 2, 3], [5, 6, 7]) == -1.0
    lo, hi = bootstrap_ci([1, 2, 3, 4, 5])
    assert lo <= 3 <= hi, (lo, hi)

    runs = [{"success": i < 13, "attempts": 1, "steps": 900 + i,
             "interventions": 0, "target_erased": 0.95 if i < 13 else 0.4,
             "max_distractor": 0.0, "target_found": True, "residual_where": None,
             "file": ""} for i in range(20)]
    c = summarize_condition("t", runs)
    assert c["성공"] == 13 and abs(c["성공률"] - 0.65) < 1e-9
    assert len(c["실패 시 erased"]) == 7
    print("selftest OK")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    if "--selftest" in sys.argv:
        _selftest()
    else:
        sys.exit(main())
