#!/usr/bin/env python3
"""시연·롤아웃에서 "팔이 도형을 비켜주는 구간"을 찾아 통계를 낸다.

왜 이걸 재는가 — `docs/erase_run_design.md` §4.2는 판정을 **시도 경계**에서만 할 수
있다고 전제한다. 근거는 "에피소드의 51%가 가려짐"이었다. 그런데 그건 **총량**이지
**연속성**이 아니다. 팔이 지우는 도중에도 주기적으로 물러나 보드를 드러낸다면,
그 창(window)에서 판정할 수 있고 다음이 전부 달라진다:

  - 시도 끝까지 안 기다리고 **중간에** 조기 종료할 수 있다 (증상 A의 직접 해결)
  - 판정 때문에 park를 강제할 이유가 없어진다 (설계 문서가 인정한 유일한 비용)
  - 사람이 볼 실시간 진행도 %를 **판정용 metric 그대로** 띄울 수 있다
    (dense_progress의 노이즈 22배를 안 떠안고)

그래서 재는 값은 "얼마나 가려졌나"가 아니라 **"안 가려진 구간이 판정에 쓸 만큼
길게, 자주 오는가"** 다.

용어:
  가림 창(occluded run)   — occ=True가 연속된 구간
  노출 창(exposed run)    — occ=False가 연속된 구간
  중간 노출(mid-task)     — 접촉 시작 후 ~ 완료(90% 지워짐) 전에 생긴 노출 창.
                            시작 전 접근 구간과 완료 후 유휴 구간은 제외한다 —
                            그 둘은 이미 알려진 구조라(27% / 22%) 새 정보가 없다.

사용:
    python retreat_analysis.py 'records/0802/erase_the_*'  --csv out.csv --plot out.png
"""

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ink_metric as IM  # noqa: E402

# 판정 한 번에 필요한 최소 노출 길이. 30fps 기준 9프레임 = 0.3초.
# 근거: 실물 판정은 프레임 한 장이면 되지만, 팔이 빠져나가는 동안의 모션블러와
# 가림 판정 경계(OCCL_JUMP)의 히스테리시스를 감안해 여유를 둔다. 1초(30프레임)로
# 잡으면 실제로 쓸 수 있는 창을 상당수 놓친다 — 아래 --min-exposed로 민감도를 본다.
MIN_EXPOSED = 9


def runs(mask: np.ndarray):
    """불리언 배열에서 True가 연속된 구간을 [(start, end_exclusive), ...]로."""
    if len(mask) == 0:
        return []
    edges = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(len(mask))
    return list(zip(starts, ends))


def completion_frame(ink: np.ndarray, occ: np.ndarray, frac=0.9):
    """잉크가 frac만큼 줄어든 첫 프레임. 없으면 None.

    envelope(단조 감소)을 쓰는 이유는 ink_metric.summarize()와 같다 — 지우기는
    잉크를 늘릴 수 없으므로, 가림에 의한 일시적 상승을 진행으로 오독하지 않는다.
    여기서는 '완료 시점'을 찾는 용도라 envelope이 맞다(판정값이 아니다).
    """
    valid = ~occ
    if valid.sum() < 2:
        return None
    filled, last = ink.copy(), ink[np.argmax(valid)]
    for i in range(len(filled)):
        if valid[i]:
            last = filled[i]
        else:
            filled[i] = last
    env = np.minimum.accumulate(filled)
    base = env[0]
    if base <= 0:
        return None
    hit = np.flatnonzero(env <= base * (1.0 - frac))
    return int(hit[0]) if len(hit) else None


def analyze_episode(ep_dir: Path, board, dark_ratio, min_exposed, fps=30.0):
    """-> dict 또는 None(대상 도형을 못 찾음)."""
    target = IM.task_of(ep_dir)
    shapes, ink_d, occ_d = IM.track(IM.top_video(ep_dir), board, dark_ratio)
    keys = [k for k, _ in shapes if k.split("#")[0] == target]
    if not keys:
        return None
    # 같은 종류가 여러 개면 가장 잉크가 많은 것을 대상으로 본다 (조각난 검출 대비)
    key = max(keys, key=lambda k: np.asarray(ink_d[k], float)[0])

    ink = np.asarray(ink_d[key], float)
    occ = np.asarray(occ_d[key], bool)
    n = len(ink)

    occ_runs = runs(occ)
    if not occ_runs:
        return None  # 팔이 이 도형을 한 번도 안 가렸다 = 안 지웠다
    contact_start = occ_runs[0][0]
    done_at = completion_frame(ink, occ, 0.9)
    contact_end = done_at if done_at is not None else n

    mid = []
    for s, e in runs(~occ):
        if s <= contact_start or s >= contact_end:
            continue  # 접근 전 / 완료 후는 이미 아는 구조라 제외
        mid.append((int(s), int(e), int(e - s)))

    usable = [r for r in mid if r[2] >= min_exposed]
    gaps = [usable[i + 1][0] - usable[i][1] for i in range(len(usable) - 1)]

    # ★ 각 노출 창에서 "그 시점에 몇 % 지워져 있는가"를 실제로 잰다.
    #
    # 여기가 이 분석의 핵심이다. "창이 완료 시점보다 이른가"를 재려고 하면
    # 동어반복에 빠진다 — 완료 시점은 envelope으로 찾는데 envelope은 가려지지
    # 않은 프레임에서만 갱신되므로, 완료가 처음 관측되는 프레임은 **정의상**
    # 노출 창 안이다. 그래서 그 비율은 항상 1.0 근처로 나오고 아무 정보가 없다.
    #
    # 대신 창마다 erased_frac을 직접 재면 질문이 제대로 선다: 중간값(0.3, 0.6 …)이
    # 실제로 찍히면 시도 도중 판정이 가능하다는 뜻이고, 전부 0 아니면 1로만
    # 찍히면 창이 있어도 "다 지웠나?"에만 답할 수 있다는 뜻이다.
    base = float(ink[: max(contact_start, 1)].min()) if contact_start > 0 else float(ink[0])
    window_erased = []
    for s, e, _ in usable:
        seen = float(np.median(ink[s:e]))  # 창 안에서는 중앙값 — 진입/이탈 프레임의 잔상 배제
        frac = 0.0 if base <= 0 else float(np.clip((base - seen) / base, 0.0, 1.0))
        window_erased.append({"start": s, "end": e, "erased": round(frac, 3),
                              "at_s": round(s / fps, 2)})

    return {
        "episode": ep_dir.name,
        "target": target,
        "shape_key": key,
        "frames": n,
        "occluded_frac": round(float(occ.mean()), 3),
        # runs()가 numpy 인덱스를 돌려주므로 int64가 섞인다 — json이 못 직렬화한다
        "contact_start": int(contact_start),
        "completed_at": done_at,
        "contact_frames": int(contact_end - contact_start),
        "mid_exposed_runs": len(mid),
        "usable_runs": len(usable),
        "usable_frames_total": int(sum(r[2] for r in usable)),
        "usable_run_len_median": float(np.median([r[2] for r in usable])) if usable else 0.0,
        "usable_run_len_max": int(max((r[2] for r in usable), default=0)),
        # 판정 기회가 얼마나 자주 오는가 — 조기 종료의 시간 해상도를 정한다
        "gap_between_usable_median_s": round(float(np.median(gaps)) / fps, 2) if gaps else None,
        "first_usable_at_s": round(usable[0][0] / fps, 2) if usable else None,
        "window_erased": window_erased,
        # 조기 종료가 실제로 가능한가: 완료(0.9) 전에 "부분적으로 지워짐"이 관측되는 창
        "partial_windows": sum(1 for w in window_erased if 0.05 < w["erased"] < 0.9),
        "first_window_erased": window_erased[0]["erased"] if window_erased else None,
        "_usable": usable,
        "_ink": ink,
        "_occ": occ,
    }


def summarize_all(rows, fps=30.0):
    n = len(rows)
    have = [r for r in rows if r["usable_runs"] > 0]
    med = lambda vals: float(np.median(vals)) if vals else 0.0  # noqa: E731
    return {
        "에피소드 수": n,
        "중간 노출 창이 1개 이상인 에피소드": f"{len(have)}/{n} ({100*len(have)/n:.0f}%)",
        "에피소드당 쓸 만한 창 (중앙값)": med([r["usable_runs"] for r in rows]),
        "창 길이 중앙값 (프레임)": med([r["usable_run_len_median"] for r in have]),
        "창 사이 간격 중앙값 (초)": med([r["gap_between_usable_median_s"] for r in have
                                        if r["gap_between_usable_median_s"] is not None]),
        "첫 창까지 (초, 중앙값)": med([r["first_usable_at_s"] for r in have
                                     if r["first_usable_at_s"] is not None]),
        "가림 비율 평균": round(float(np.mean([r["occluded_frac"] for r in rows])), 3),
        # 조기 종료 가능성의 직접 근거
        "부분 지워짐(0.05~0.9)이 보이는 창을 가진 에피소드":
            f"{sum(1 for r in rows if r['partial_windows'] > 0)}/{n}",
        "첫 창의 erased 중앙값": round(med([r["first_window_erased"] for r in rows
                                          if r["first_window_erased"] is not None]), 3),
    }


def plot(rows, path: Path, fps=30.0, max_eps=30):
    from plot_ko import plt, use_korean
    use_korean()

    rows = rows[:max_eps]
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 2]}
    )

    # 위: 에피소드별 타임라인. 회색=가림, 색칠=노출 창(색 = 그 시점의 erased 값)
    cmap = plt.get_cmap("RdYlGn")
    for i, r in enumerate(rows):
        occ = r["_occ"]
        t = np.arange(len(occ)) / fps
        ax1.fill_between(t, i - 0.42, i + 0.42, where=occ, color="0.82", linewidth=0)
        for w in r["window_erased"]:
            ax1.fill_between([w["start"] / fps, w["end"] / fps], i - 0.42, i + 0.42,
                             color=cmap(w["erased"]), linewidth=0)
            # 부분 지워짐일 때만 숫자를 적는다 — 1.00이 대부분이라 다 적으면 안 보인다
            if w["erased"] < 0.9:
                ax1.text((w["start"] + w["end"]) / 2 / fps, i, f"{w['erased']:.2f}",
                         ha="center", va="center", fontsize=6.5, color="black")
    ax1.set_yticks([])
    ax1.set_xlabel("시간 (초)")
    ax1.set_ylabel(f"에피소드 (n={len(rows)})")
    ax1.set_title(
        "팔이 도형을 가린 구간(회색) vs 팔이 물러나 보드가 드러난 창(색)\n"
        "창의 색 = 그 순간 실제로 지워져 있던 비율 (빨강 0.0 → 초록 1.0)"
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=ax1, label="erased_frac", pad=0.01, fraction=0.025)

    # 아래: 노출 창에서 관측된 erased 값의 분포 — 이게 핵심 결과다.
    # 0과 1에만 몰려 있으면 "도중 판정"이라는 게 원리적으로 불가능하다는 뜻이다.
    vals = [w["erased"] for r in rows for w in r["window_erased"]]
    ax2.hist(vals, bins=np.linspace(0, 1, 21), color="#2e8b57", edgecolor="white")
    ax2.set_xlabel("노출 창에서 관측된 erased_frac")
    ax2.set_ylabel("창 개수")
    ax2.axvspan(0.05, 0.9, color="orange", alpha=0.15,
                label="'부분적으로 지워짐' 구간 — 조기 판정이 의미를 갖는 영역")
    ax2.set_title(f"노출 창 {len(vals)}개 중 부분 구간에 들어온 것: "
                  f"{sum(1 for v in vals if 0.05 < v < 0.9)}개")
    ax2.legend(loc="upper left", fontsize=9)

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"[INFO] 그림 저장: {path}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="에피소드 디렉터리 (glob 가능)")
    p.add_argument("--board", type=lambda s: tuple(int(v) for v in s.split(",")),
                   default=IM.DEFAULT_BOARD)
    p.add_argument("--dark-ratio", type=float, default=0.72)
    p.add_argument("--min-exposed", type=int, default=MIN_EXPOSED,
                   help=f"판정 1회에 필요한 최소 연속 노출 프레임 (기본 {MIN_EXPOSED})")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--limit", type=int, default=0, help="앞에서 N개만")
    p.add_argument("--csv", type=Path)
    p.add_argument("--plot", type=Path)
    p.add_argument("--json", type=Path)
    a = p.parse_args(argv)

    dirs = [Path(d) for d in sorted(glob.glob(a.path)) if Path(d).is_dir()]
    if a.limit:
        dirs = dirs[: a.limit]
    if not dirs:
        print(f"에피소드를 못 찾음: {a.path}", file=sys.stderr)
        return 2

    rows, skipped = [], []
    for i, d in enumerate(dirs, 1):
        try:
            r = analyze_episode(d, a.board, a.dark_ratio, a.min_exposed, a.fps)
        except Exception as exc:  # 한 에피소드가 깨져도 전체를 멈추지 않는다
            skipped.append((d.name, str(exc)))
            continue
        if r is None:
            skipped.append((d.name, "대상 도형 미검출 / 접촉 없음"))
            continue
        rows.append(r)
        print(f"[{i}/{len(dirs)}] {d.name}: 창 {r['usable_runs']}개 "
              f"(중앙 {r['usable_run_len_median']:.0f}프레임), 가림 {r['occluded_frac']:.0%}")

    if not rows:
        print("분석 가능한 에피소드가 없다.", file=sys.stderr)
        return 1

    summary = summarize_all(rows, a.fps)
    print("\n── 요약 " + "─" * 40)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    if skipped:
        print(f"  제외: {len(skipped)}개")
        for name, why in skipped[:5]:
            print(f"    - {name}: {why}")

    if a.csv:
        cols = [k for k in rows[0] if not k.startswith("_")]
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in cols})
        print(f"[INFO] CSV 저장: {a.csv}")
    if a.json:
        a.json.write_text(json.dumps(
            {"summary": summary,
             "min_exposed": a.min_exposed,
             "episodes": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
             "skipped": skipped},
            ensure_ascii=False, indent=2))
        print(f"[INFO] JSON 저장: {a.json}")
    if a.plot:
        plot(rows, a.plot, a.fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
