#!/usr/bin/env python3
"""episodes.csv → 조건별 성능 표 (Q1 논문에 그대로 들어가는 형태).

    python scripts/tools/eval_kit/summarize.py
    python scripts/tools/eval_kit/summarize.py --csv outputs/analysis/episodes.csv --markdown

기본은 **에피소드당 마지막 시도**(is_final=1)만 센다 — 재시도까지 포함한 최종 결과가
"이 방법의 성공률"이다. 시도 단위로 보고 싶으면 --all-attempts.

내는 것:
  - 성공률 + Wilson 95% 신뢰구간. n=20에서 점추정만 쓰면 정보가 거의 없다.
  - target_erased 평균/중앙값 — 0.85 실패와 0.30 실패를 구분해준다.
  - 위반: 도형 훼손률, 훼손 위반 건수, 타임아웃 건수.
  - 게이트 불일치 건수 — 이 값이 0이 아니면 판정 분리가 실제로 뭔가를 잡은 것이다.

조건 간 유의성 검정(Fisher/McNemar)과 효과 크기는 기존 erase_stats.py 쪽에 이미
있다. 여기서는 표만 낸다 — 두 도구가 같은 걸 두 번 구현하지 않게.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

DEFAULT_CSV = Path("outputs/analysis/episodes.csv")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """이항 비율의 Wilson score 구간. Wald는 k=n이면 폭이 0이 되어 못 쓴다."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _f(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r.get("condition") or "(무조건)"].append(r)

    out = []
    for cond, rs in sorted(groups.items()):
        n = len(rs)
        k = sum(1 for r in rs if r.get("judge_success") == "1")
        lo, hi = wilson(k, n)
        te = [v for v in (_f(r, "target_erased") for r in rs) if v is not None]
        dmg = [v for v in (_f(r, "shape_damage") for r in rs) if v is not None]
        steps = [v for v in (_f(r, "steps") for r in rs) if v is not None]
        out.append({
            "condition": cond,
            "n": n,
            "success": k,
            "rate": k / n if n else 0.0,
            "ci": (lo, hi),
            "te_mean": st.mean(te) if te else float("nan"),
            "te_sd": st.stdev(te) if len(te) > 1 else 0.0,
            "te_median": st.median(te) if te else float("nan"),
            "dmg_mean": st.mean(dmg) if dmg else float("nan"),
            "dmg_max": max(dmg) if dmg else float("nan"),
            "violations": sum(1 for r in rs if r.get("damage_violation") == "1"),
            "timeouts": sum(1 for r in rs if r.get("timeout") == "1"),
            "steps_median": st.median(steps) if steps else float("nan"),
            "disagree": sum(1 for r in rs if r.get("disagree") == "1"),
            "patterns": len({r.get("pattern_id") for r in rs if r.get("pattern_id")}),
        })
    return out


HEAD = ["조건", "n", "성공률 (95% CI)", "target_erased", "훼손 평균/최대",
        "위반", "타임아웃", "스텝 중앙", "게이트 불일치"]


def _cells(s: dict) -> list[str]:
    return [
        s["condition"],
        str(s["n"]),
        f"{s['rate']:.1%} [{s['ci'][0]:.1%}, {s['ci'][1]:.1%}]",
        f"{s['te_mean']:.3f} ± {s['te_sd']:.3f}",
        f"{s['dmg_mean']:.3f} / {s['dmg_max']:.3f}",
        str(s["violations"]),
        str(s["timeouts"]),
        f"{s['steps_median']:.0f}" if s["steps_median"] == s["steps_median"] else "-",
        str(s["disagree"]),
    ]


def print_table(stats: list[dict], markdown=False) -> None:
    rows = [HEAD] + [_cells(s) for s in stats]
    if markdown:
        print("| " + " | ".join(rows[0]) + " |")
        print("|" + "---|" * len(rows[0]))
        for r in rows[1:]:
            print("| " + " | ".join(r) + " |")
        return
    widths = [max(len(r[i]) for r in rows) for i in range(len(HEAD))]
    for i, r in enumerate(rows):
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))
        if i == 0:
            print("  ".join("-" * w for w in widths))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--csv", default=str(DEFAULT_CSV))
    p.add_argument("--all-attempts", action="store_true",
                   help="에피소드 마지막 시도만이 아니라 모든 시도를 센다")
    p.add_argument("--markdown", action="store_true")
    a = p.parse_args(argv)

    path = Path(a.csv)
    if not path.exists():
        print(f"[ERROR] CSV가 없다: {path}  (먼저 adjudicate.py를 돌릴 것)")
        return 1
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not a.all_attempts:
        rows = [r for r in rows if r.get("is_final") == "1"]
    if not rows:
        print("[ERROR] 셀 행이 없다")
        return 1

    stats = summarize(rows)
    print_table(stats, a.markdown)

    versions = {r.get("config_hash") for r in rows if r.get("config_hash")}
    if len(versions) > 1:
        print(f"\n[WARN] ★ 서로 다른 판정 설정이 섞여 있다: {sorted(versions)}")
        print("       임계값이 다른 숫자를 한 표에 넣으면 안 된다. "
              "`adjudicate.py --force`로 전부 다시 판정할 것.")
    unpaired = [s for s in stats if s["patterns"] == 0]
    if unpaired:
        print("\n[WARN] pattern_id가 비어 있는 조건이 있다 — 페어링 통계(McNemar)를 못 쓴다.")
        print("       실행마다 stamp.py --pattern-id 를 붙일 것.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
