#!/usr/bin/env python3
"""effort_audit.py — 여러 에피소드 폴더를 한 번에 훑어 effort 분포를 감사한다.

check_effort.py는 데이터셋 1개의 첫 parquet 1개만 본다. 우리 records/는
1에피소드 = 1데이터셋 폴더가 수십 개라, 안전 임계값 결정과 오염 감사에는
전체 분포가 필요하다.

  python scripts/tools/effort_audit.py records/local
  python scripts/tools/effort_audit.py records --limit 8.0
  python scripts/tools/effort_audit.py --selftest

내는 것:
  1) 관절별 effort 분위수 -> SAFETY_EFFORT_LIMIT 근거
  2) 현재 limit이 과거 데이터에서 몇 번 걸렸을지 (컷오프 오발 위험)
  3) smooth_start의 옛 버그(앞 N프레임 effort를 parking 위치값으로 덮어씀)
     오염 여부 — 2026-07-27 upstream 커밋 이전 녹화분이 대상
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

SMOOTH_N = 100  # SMOOTH_START_FRAMES_DEFAULT


def load_episode(root: pathlib.Path):
    """(effort (L,7), 관절 이름들) 반환. effort 없으면 None."""
    import pyarrow.parquet as pq  # selftest는 pyarrow 없이도 돌게 지연 import

    info = json.loads((root / "meta" / "info.json").read_text())
    names = info["features"]["observation.state"]["names"]
    idx = [i for i, n in enumerate(names) if n.endswith(".effort")]
    if not idx:
        return None
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        return None
    chunks = []
    for f in files:
        t = pq.read_table(f)
        state = np.stack(t.column("observation.state").to_pylist())
        order = np.argsort(t.column("frame_index").to_numpy(zero_copy_only=False))
        chunks.append(state[order][:, idx])
    return np.concatenate(chunks), [names[i] for i in idx]


def is_contaminated(eff: np.ndarray) -> bool:
    """앞 SMOOTH_N 프레임이 선형 램프면 smooth_start에 덮어써진 것.
    선형 구간은 2차 차분이 0 — 실제 토크 측정값에서는 나올 수 없다."""
    if len(eff) < SMOOTH_N + 20:
        return False
    return bool(np.abs(np.diff(eff[:SMOOTH_N], n=2, axis=0)).max() < 1e-4)


def audit(roots: list[pathlib.Path], limit: float) -> int:
    eps = []
    for root in roots:
        got = load_episode(root)
        if got is None:
            print(f"  (건너뜀, effort 없음) {root.name}")
            continue
        eff, jnames = got
        eps.append((root.name, eff, jnames))
    if not eps:
        print("effort가 있는 에피소드를 못 찾음")
        return 1

    jnames = eps[0][2]
    allv = np.concatenate([np.abs(e) for _, e, _ in eps])

    print(f"\n=== 관절별 |effort| 분위수 (에피소드 {len(eps)}개, {len(allv)} 프레임) ===")
    print(f"  {'관절':18s} {'p50':>7s} {'p95':>7s} {'p99':>7s} {'max':>7s}")
    for j, n in enumerate(jnames):
        c = allv[:, j]
        print(f"  {n:18s} {np.percentile(c,50):7.3f} {np.percentile(c,95):7.3f} "
              f"{np.percentile(c,99):7.3f} {c.max():7.3f}")

    peak = allv.max()
    print(f"\n  전체 |effort| 최댓값 {peak:.3f} N·m -> SAFETY_EFFORT_LIMIT 권장 "
          f"{peak*1.5:.1f} (관측 최댓값 x1.5)")

    # 현재 limit이 과거 데이터에서 걸렸을 횟수 = 컷오프 오발 위험
    tripped = [(name, int((np.abs(e) > limit).any(axis=1).sum())) for name, e, _ in eps]
    bad = [(n, c) for n, c in tripped if c]
    print(f"\n=== 현재 SAFETY_EFFORT_LIMIT={limit} 오발 위험 ===")
    if bad:
        print(f"  ❌ {len(bad)}/{len(eps)} 에피소드에서 걸렸을 것 — 정상 시연이 잘린다")
        for n, c in sorted(bad, key=lambda x: -x[1])[:10]:
            print(f"     {n:40s} {c:5d} 프레임")
    else:
        margin = limit / peak if peak else float("inf")
        print(f"  ✅ 걸린 에피소드 없음 (최댓값 대비 여유 {margin:.2f}x)")
        if margin < 1.3:
            print("  ⚠️  여유가 30% 미만 — 조금만 세게 눌러도 걸린다")

    # smooth_start 오염
    dirty = [n for n, e, _ in eps if is_contaminated(e)]
    print(f"\n=== smooth_start effort 오염 (upstream 2026-07-27 이전 버그) ===")
    if dirty:
        print(f"  ❌ {len(dirty)}/{len(eps)} 에피소드의 앞 {SMOOTH_N}프레임 effort가 덮어써짐")
        for n in dirty[:10]:
            print(f"     {n}")
        print(f"  → 접촉 분석 시 앞 {SMOOTH_N}프레임을 버리고 쓸 것. pos/action은 멀쩡함")
    else:
        print(f"  ✅ 오염 없음")

    return 1 if (bad or dirty) else 0


def find_roots(paths: list[str]) -> list[pathlib.Path]:
    out = []
    for p in paths:
        base = pathlib.Path(p)
        if (base / "meta" / "info.json").exists():
            out.append(base)
        else:
            out += sorted(f.parent.parent for f in base.rglob("meta/info.json"))
    return out


def _selftest() -> int:
    ramp = np.linspace(-100.0, 2.0, SMOOTH_N)[:, None].repeat(7, 1)
    real = np.random.RandomState(0).randn(200, 7) * 0.5 + 2.0
    assert is_contaminated(np.vstack([ramp, real])), "선형 램프를 못 잡음"
    assert not is_contaminated(real.repeat(2, 0)), "정상 데이터를 오염으로 오판"
    assert not is_contaminated(real[:50]), "짧은 에피소드는 판정 생략해야 함"
    print("selftest ok")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="데이터셋 root 또는 그것들을 담은 상위 폴더")
    ap.add_argument("--limit", type=float, default=8.0, help="검사할 SAFETY_EFFORT_LIMIT")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    if not a.paths:
        ap.error("경로를 하나 이상 지정하거나 --selftest")
    roots = find_roots(a.paths)
    print(f"데이터셋 {len(roots)}개 발견")
    sys.exit(audit(roots, a.limit))
