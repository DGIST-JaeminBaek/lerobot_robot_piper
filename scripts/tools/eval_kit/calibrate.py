#!/usr/bin/env python3
"""사람 라벨로 판정 임계값 theta를 캘리브레이션한다.

이걸 안 하면 theta는 "그냥 0.9로 뒀다"가 되고, 리뷰어가 정확히 그 지점을 친다.
논문에는 한 문단이면 끝나지만 데이터가 없으면 못 쓴다:
"자동 판정기는 3명의 라벨과 95% 일치했다(kappa=0.87), theta는 ROC의 Youden J
최대점으로 정하고 이후 모든 실험에서 동결했다."

절차:

  1) 라벨 시트 만들기 — 판정된 실행들에서 프레임을 무작위로 뽑는다
     python scripts/tools/eval_kit/calibrate.py --make-sheet 40 \
         --runs 'records/hil/*' --out outputs/analysis/labels.csv

  2) 사람이 채운다. 시트에 프레임 경로가 있으니 이미지를 보고
     rater_A / rater_B / rater_C 칸에 1(다 지웠다) 또는 0을 적는다.
     ★ 서로 안 보이게 따로 채울 것. 같이 보면 kappa가 의미를 잃는다.
     ★ target_erased 칸은 시트에서 지우고 주는 걸 권한다 — 숫자를 보면 사람이 끌려간다.

  3) 분석
     python scripts/tools/eval_kit/calibrate.py --analyze outputs/analysis/labels.csv

     → 라벨러 간 일치도(kappa), ROC/Youden 최적 theta, 현재 theta의 FP/FN.
       결과를 보고 judge_config.json의 theta를 한 번만 고치고 동결한다.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import random
from itertools import combinations
from pathlib import Path

SHEET_COLUMNS = ["run_id", "attempt", "frame", "reference", "target",
                 "target_erased", "rater_A", "rater_B", "rater_C"]


def make_sheet(run_globs: list[str], n: int, out: Path, seed: int = 0) -> int:
    """판정 완료된 실행들에서 (프레임, 자동측정값) 샘플을 뽑아 라벨 시트를 만든다."""
    items = []
    for pattern in run_globs:
        for raw in sorted(glob.glob(pattern)):
            d = Path(raw)
            vpath = d / "eval_kit" / "verdict.json"
            if not vpath.exists():
                continue
            v = json.loads(vpath.read_text())
            for att in v["attempts"]:
                items.append({
                    "run_id": v["run_id"],
                    "attempt": att["attempt"],
                    "frame": str(d / att["frame"]),
                    "reference": str(d / "00_reference.png"),
                    "target": v["target"],
                    "target_erased": att.get("target_erased", ""),
                    "rater_A": "", "rater_B": "", "rater_C": "",
                })
    if not items:
        print("[ERROR] 뽑을 게 없다 — 먼저 adjudicate.py를 돌릴 것")
        return 0

    random.Random(seed).shuffle(items)  # seed 고정 — 시트를 다시 만들어도 같은 표본
    items = items[:n]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        w.writeheader()
        w.writerows(items)
    print(f"[OK] {len(items)}행 시트: {out}")
    print("     이미지를 보고 rater_* 칸에 1(다 지웠다)/0을 채운 뒤 --analyze로 넘길 것.")
    return len(items)


# ── 통계 ────────────────────────────────────────────────────
def cohen_kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def roc_youden(scores: list[float], labels: list[int]) -> tuple[float, float, float, float]:
    """모든 후보 임계값을 훑어 Youden J(=TPR-FPR) 최대점을 찾는다.
    반환: (theta, J, TPR, FPR)"""
    best = (float("nan"), -2.0, 0.0, 0.0)
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return best
    for thr in sorted(set(scores)):
        tp = sum(1 for s, y in zip(scores, labels) if s >= thr and y == 1)
        fp = sum(1 for s, y in zip(scores, labels) if s >= thr and y == 0)
        tpr, fpr = tp / pos, fp / neg
        j = tpr - fpr
        if j > best[1]:
            best = (thr, j, tpr, fpr)
    return best


def analyze(path: Path, cfg_path: Path | None = None) -> int:
    with path.open(newline="") as f:
        rows = [r for r in csv.DictReader(f)]

    raters = [c for c in rows[0] if c.startswith("rater_")] if rows else []
    filled = {r: [] for r in raters}
    for row in rows:
        for r in raters:
            v = (row.get(r) or "").strip()
            if v in ("0", "1"):
                filled[r].append(v)
    active = [r for r in raters if len(filled[r]) == len(rows) and len(rows) > 0]
    if not active:
        print("[ERROR] 끝까지 채워진 라벨러 칸이 없다. 빈 칸이 없어야 분석이 된다.")
        return 1

    print(f"표본 {len(rows)}건 / 라벨러 {len(active)}명: {', '.join(active)}\n")

    labels = {r: [int(row[r]) for row in rows] for r in active}
    if len(active) >= 2:
        print("── 라벨러 간 일치도 (Cohen's kappa) ──")
        for a, b in combinations(active, 2):
            k = cohen_kappa(labels[a], labels[b])
            flag = "" if k >= 0.7 else "   ★ 0.7 미만 — 판정 기준을 말로 먼저 합의할 것"
            print(f"  {a} vs {b}: kappa={k:.3f}{flag}")
        print()

    # 다수결이 정답. 짝수 명이면 동점은 버린다(애매한 표본은 임계값을 왜곡한다).
    # 점수와 정답은 반드시 같은 행에서 함께 뽑는다 — 따로 모으면 어긋난다.
    scores, gold, dropped, missing = [], [], 0, 0
    for i, row in enumerate(rows):
        votes = [labels[r][i] for r in active]
        if votes.count(1) * 2 == len(votes):
            dropped += 1
            continue
        try:
            score = float(row["target_erased"])
        except (KeyError, TypeError, ValueError):
            missing += 1
            continue
        scores.append(score)
        gold.append(1 if votes.count(1) * 2 > len(votes) else 0)
    if dropped:
        print(f"[INFO] 동점 {dropped}건 제외")
    if missing:
        print(f"[INFO] target_erased 없는 {missing}건 제외")
    if dropped or missing:
        print()
    if not scores:
        print("[ERROR] target_erased 값이 없다 — 시트를 make-sheet로 다시 만들 것")
        return 1

    theta, j, tpr, fpr = roc_youden(scores, gold)
    print("── ROC / Youden ──")
    print(f"  최적 theta = {theta:.4f}   J={j:.3f}  (TPR={tpr:.2f}, FPR={fpr:.2f})")

    cfg = json.loads(Path(cfg_path or Path(__file__).with_name("judge_config.json")).read_text())
    cur = cfg["theta"]
    tp = sum(1 for s, y in zip(scores, gold) if s >= cur and y == 1)
    fp = sum(1 for s, y in zip(scores, gold) if s >= cur and y == 0)
    fn = sum(1 for s, y in zip(scores, gold) if s < cur and y == 1)
    tn = sum(1 for s, y in zip(scores, gold) if s < cur and y == 0)
    acc = (tp + tn) / len(gold)
    print(f"\n── 현재 theta={cur} 로 재면 ──")
    print(f"  일치율 {acc:.1%}   TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"  (FP = 사람은 덜 지웠다는데 자동이 성공이라 한 건 — 성능을 부풀리는 방향)")
    if abs(theta - cur) > 0.02:
        print(f"\n  ★ 최적점이 현재값과 {abs(theta - cur):.3f} 떨어져 있다. "
              f"judge_config.json의 theta를 {theta:.3f}으로 바꾸고,")
        print("    바꿨으면 judge_version을 올린 뒤 adjudicate.py --force로 전부 재판정할 것.")
    else:
        print("\n  현재 theta가 최적점과 사실상 같다 — 그대로 두고 이 결과를 논문에 싣는다.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--make-sheet", type=int, metavar="N", help="라벨 시트를 N행 만든다")
    p.add_argument("--runs", nargs="*", default=["records/hil/*"], help="실행 폴더 glob")
    p.add_argument("--out", default="outputs/analysis/labels.csv")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--analyze", metavar="CSV", help="채워진 시트를 분석")
    p.add_argument("--config", default=None)
    a = p.parse_args(argv)

    if a.make_sheet:
        return 0 if make_sheet(a.runs, a.make_sheet, Path(a.out), a.seed) else 1
    if a.analyze:
        return analyze(Path(a.analyze), Path(a.config) if a.config else None)
    p.error("--make-sheet 또는 --analyze 중 하나가 필요")


if __name__ == "__main__":
    raise SystemExit(main())
