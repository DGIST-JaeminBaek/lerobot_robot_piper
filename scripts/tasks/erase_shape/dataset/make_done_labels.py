#!/usr/bin/env python3
"""잉크 metric으로 에피소드별 done 라벨 경계와 QC 제외 목록을 만든다.

VLAC(arXiv 2509.15937)의 라벨링을 따르되 시간 비율 대신 물리적 잉크 잔량을 쓴다:
확실히 진행 중인 구간만 done=0, 확실히 끝난 구간만 done=1, 애매한 구간은 라벨을 비운다.
억지로 0/1을 매기면 경계에서 모델이 흔들린다.

  erased < 0.5            -> done = 0
  0.5 <= erased < 0.9     -> -1 (라벨 없음)
  erased >= 0.9           -> done = 1
  팔이 도형을 가린 프레임  -> -1 (관측 불가이므로 알 수 없음)

마지막 규칙이 중요하다. 지우는 동안 top 카메라는 도형을 못 본다(접촉 구간 중 관측
가능한 프레임이 중앙값 1%). 그 구간에 done=0을 박으면 "이미 다 지웠는데 아직 아니다"를
가르치게 된다.

데이터셋 원본은 건드리지 않고 sidecar JSON만 쓴다. 학습 코드에서 load_labels()로 읽어
프레임 배열로 펼쳐 쓴다.

사용:
  python make_done_labels.py '0802/*/erase_the_*' -o done_labels.json
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_DIR / "lib"))
import ink_metric as M

DONE_HI = 0.9  # 이 이상 지워졌으면 확실히 완료
DONE_LO = 0.5  # 이 미만이면 확실히 진행 중
MIN_TARGET_ERASED = M.SUCCESS_ERASED  # QC: 시연자가 target을 이만큼도 못 지웠으면 제외
MAX_DISTRACTOR = M.MAX_DISTRACTOR  # QC: distractor를 이만큼 건드렸으면 제외


def analyze(ep_dir: Path, board, dark_ratio, occl_jump):
    target = M.task_of(ep_dir)
    shapes, ink, occ = M.track(M.top_video(ep_dir), board, dark_ratio, occl_jump)
    per = M.summarize(ink, occ)
    labels = {k.split("#")[0] for k in per}

    tgt = M.aggregate(per, target)
    if not tgt or tgt["ink_init"] <= 0:
        return {"episode": ep_dir.name, "status": "exclude", "reason": "target 도형 미검출/오분류"}

    others = [M.aggregate(per, l) for l in labels if l != target]
    max_distractor = max((o["erased_frac"] for o in others if o), default=0.0)

    # target 도형의 잉크 envelope -> 진행도 곡선
    ks = [k for k in per if k.split("#")[0] == target and "env" in per[k]]
    n = min(len(per[k]["env"]) for k in ks)
    env = np.sum([per[k]["env"][:n] for k in ks], axis=0)
    prog = 1 - env / env[0]
    occluded = np.any([np.asarray(occ[k][:n]) for k in ks], axis=0)

    def first(cond):
        hit = np.nonzero(cond)[0]
        return int(hit[0]) if len(hit) else None

    t_lo, t_hi = first(prog >= DONE_LO), first(prog >= DONE_HI)

    reason = None
    if tgt["erased_frac"] < MIN_TARGET_ERASED:
        reason = f"target 미완 {tgt['erased_frac']:.2f}"
    elif max_distractor > MAX_DISTRACTOR:
        reason = f"distractor 침범 {max_distractor:.2f}"
    elif t_hi is None:
        reason = "완료 시점 특정 불가"

    return {
        "episode": ep_dir.name,
        "path": str(ep_dir),
        "target": target,
        "n_frames": n,
        "erased_frac": tgt["erased_frac"],
        "max_distractor_erased": round(max_distractor, 4),
        "t_ambiguous": t_lo,  # done=0 구간의 끝
        "t_done": t_hi,  # done=1 구간의 시작
        "idle_frames": (n - t_hi) if t_hi is not None else 0,
        "occluded": [[int(a), int(b)] for a, b in occluded_spans(occluded)],
        "status": "exclude" if reason else "ok",
        "reason": reason,
    }


def occluded_spans(mask):
    """가림 플래그 배열을 [start, end) 구간 목록으로 압축 (JSON 크기 절약)."""
    spans, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(mask)))
    return spans


def load_labels(entry):
    """sidecar 한 항목을 프레임별 done 배열로 펼친다. 0 / 1 / -1(라벨 없음)."""
    n = entry["n_frames"]
    out = np.zeros(n, dtype=np.int8)
    t_lo, t_hi = entry.get("t_ambiguous"), entry.get("t_done")
    if t_lo is not None:
        out[t_lo:] = -1
    if t_hi is not None:
        out[t_hi:] = 1
    for a, b in entry.get("occluded", []):
        out[a:b] = -1  # 관측 불가 구간은 알 수 없음
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("path", help="에피소드 디렉터리 (glob)")
    p.add_argument("-o", "--out", type=Path, required=True)
    p.add_argument("--board", type=lambda s: tuple(int(v) for v in s.split(",")), default=M.DEFAULT_BOARD)
    p.add_argument("--dark-ratio", type=float, default=0.72)
    p.add_argument("--occl-jump", type=float, default=M.OCCL_JUMP)
    a = p.parse_args(argv)

    dirs = [Path(d) for d in sorted(glob.glob(a.path)) if Path(d).is_dir()]
    if not dirs:
        p.error(f"에피소드를 찾지 못함: {a.path}")

    entries = []
    for d in dirs:
        e = analyze(d, a.board, a.dark_ratio, a.occl_jump)
        entries.append(e)
        if e["status"] == "exclude":
            print(f"  [제외] {e['episode']}: {e['reason']}")

    ok = [e for e in entries if e["status"] == "ok"]
    idle = sum(e["idle_frames"] for e in ok)
    done1 = sum(int((load_labels(e) == 1).sum()) for e in ok)
    done0 = sum(int((load_labels(e) == 0).sum()) for e in ok)
    unl = sum(int((load_labels(e) == -1).sum()) for e in ok)

    a.out.write_text(json.dumps({"episodes": entries}, ensure_ascii=False, indent=1))
    print(f"\n저장: {a.out}")
    print(f"  전체 {len(entries)} / 사용 {len(ok)} / 제외 {len(entries) - len(ok)}")
    print(f"  완료 후 유휴 프레임 총 {idle:,} ({idle / 30:.0f}초 분량)")
    print(f"  done=1 {done1:,} | done=0 {done0:,} | 라벨없음 {unl:,}")
    return 0


def _selftest():
    e = {
        "n_frames": 10,
        "t_ambiguous": 4,
        "t_done": 7,
        "occluded": [[2, 4]],
    }
    got = load_labels(e).tolist()
    assert got == [0, 0, -1, -1, -1, -1, -1, 1, 1, 1], got
    # 가림 구간이 done 구간을 덮으면 라벨없음이 이긴다 (관측 못 했으므로)
    e2 = {"n_frames": 5, "t_ambiguous": 1, "t_done": 2, "occluded": [[3, 5]]}
    assert load_labels(e2).tolist() == [0, -1, 1, -1, -1]
    # 끝까지 완료 못 한 에피소드
    e3 = {"n_frames": 4, "t_ambiguous": 2, "t_done": None, "occluded": []}
    assert load_labels(e3).tolist() == [0, 0, -1, -1]
    assert occluded_spans([0, 1, 1, 0, 1]) == [(1, 3), (4, 5)]
    print("selftest OK")


if __name__ == "__main__":
    sys.path.insert(0, str(TASK_DIR / "evaluation"))
    if "--selftest" in sys.argv:
        _selftest()
    else:
        sys.exit(main())
