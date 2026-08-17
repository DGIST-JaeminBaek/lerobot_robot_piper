#!/usr/bin/env python3
"""추론 시 "다 지웠나?" 판정 — check-and-retry 루프용.

핵심: 잉크 metric은 팔이 도형을 가리는 동안엔 못 쓰지만, 팔이 park로 빠진 상태에서는
완벽하다(target 1.00 vs distractor 0.05, 20배 분리). 정책은 한 시도가 끝나면 반드시
팔을 뺀다 → **매 시도 끝마다 정답을 알 수 있다.**

    ref = checker.set_reference(frame_before)   # 시도 전 (팔 park, 도형 다 보임)
    ...정책 실행...
    r = checker.check(frame_after)              # 시도 후 (팔 park)
    if not r["success"]: 재시도

이 모듈은 판정만 한다. 로봇을 움직이지 않는다 — 재시도 여부는 호출하는 쪽이 정한다.

오프라인 검증:
  python erase_check.py --validate '0802/*/erase_the_*'   # 첫/마지막 프레임만으로 판정
"""

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

import ink_metric as M

SUCCESS_ERASED = M.SUCCESS_ERASED
MAX_DISTRACTOR = M.MAX_DISTRACTOR


class EraseChecker:
    """시도 전/후 프레임 한 장씩으로 성공 여부를 판정한다."""

    def __init__(self, board=M.DEFAULT_BOARD, dark_ratio=0.72):
        self.board = board
        self.dark_ratio = dark_ratio
        self.reference = None

    def set_reference(self, frame):
        """시도 전 프레임에서 도형을 검출하고 각 도형의 잉크량을 기록한다."""
        detected, white = M.detect_shapes(frame, self.board, self.dark_ratio)
        if not detected:
            raise RuntimeError("기준 프레임에서 도형을 하나도 못 찾음 — 보드/조명 확인")
        ink_thr = white * self.dark_ratio
        shapes = [(f"{lbl}#{i}", box) for i, (lbl, box) in enumerate(detected)]
        self.reference = {
            "ink_thr": ink_thr,
            "shapes": shapes,
            "ink": {k: self._ink(frame, box, ink_thr) for k, box in shapes},
        }
        return self.reference

    @staticmethod
    def _ink(frame, box, ink_thr):
        x, y, w, h = box
        g = cv2.cvtColor(frame[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)
        return float((g < ink_thr).mean())

    def check(self, frame, target):
        """시도 후 프레임으로 판정. target은 'circle'/'triangle'/'rectangle'."""
        if self.reference is None:
            raise RuntimeError("set_reference()를 먼저 호출할 것")
        ref, thr = self.reference, self.reference["ink_thr"]

        erased = {}
        for k, box in ref["shapes"]:
            before = ref["ink"][k]
            after = self._ink(frame, box, thr)
            erased[k] = 0.0 if before <= 0 else max(0.0, min(1.0, (before - after) / before))

        def by_label(label):
            vals = [v for k, v in erased.items() if k.split("#")[0] == label]
            return vals

        tgt = by_label(target)
        others = [v for k, v in erased.items() if k.split("#")[0] != target]
        # 같은 종류가 여러 개면 잉크량 가중이 아니라 최솟값(가장 안 지워진 것) 기준 — 보수적
        target_erased = min(tgt) if tgt else None
        max_distractor = max(others) if others else 0.0

        if target_erased is None:
            return {
                "success": False,
                "target_found": False,
                "reason": f"기준 프레임에 {target} 도형이 없음",
                "detected": [k for k, _ in ref["shapes"]],
            }
        success = target_erased >= SUCCESS_ERASED and max_distractor <= MAX_DISTRACTOR

        # 실패했을 때 "얼마나, 어디가" 남았는지. 성공했으면 안 잰다 — 남은 게 없다.
        # 이 값은 진단·리커버리 데이터 수집용이지 정책에 넣는 신호가 아니다
        # (docs/erase_run_design.md §4.6-③: 전달할 통로가 없다).
        residual = None
        if not success and target_erased < SUCCESS_ERASED:
            worst = min(
                (kv for kv in erased.items() if kv[0].split("#")[0] == target),
                key=lambda kv: kv[1],
            )[0]
            box = dict(ref["shapes"])[worst]
            residual = M.residual(frame, box, thr)
            residual["shape"] = worst

        return {
            "success": bool(success),
            "target_found": True,
            "target_erased": round(target_erased, 4),
            "max_distractor_erased": round(max_distractor, 4),
            # 사람이 바로 읽는 값 — "몇 % 남았나". erased의 여집합이다.
            "remaining_frac": round(1.0 - target_erased, 4),
            "residual": residual,
            "reason": None
            if success
            else ("target 덜 지워짐" if target_erased < SUCCESS_ERASED else "distractor 침범"),
            "detected": [k for k, _ in ref["shapes"]],
        }


def retry_loop(checker, frame_grabber, run_attempt, target, max_attempts=3):
    """check-and-retry 골격. 로봇 제어는 run_attempt 콜백이 담당한다.

    frame_grabber() -> BGR 프레임 (팔이 park에 있을 때 호출되어야 한다)
    run_attempt()   -> 정책 1회 실행 (호출하는 쪽이 구현; 이 모듈은 로봇을 안 건드린다)
    """
    checker.set_reference(frame_grabber())
    history = []
    for i in range(1, max_attempts + 1):
        run_attempt()
        r = checker.check(frame_grabber(), target)
        r["attempt"] = i
        history.append(r)
        if r["success"]:
            break
    return history


def first_last_frames(video):
    cap = cv2.VideoCapture(str(video))
    ok, first = cap.read()
    if not ok:
        raise RuntimeError(f"영상 읽기 실패: {video}")
    last = first
    while True:
        ok, f = cap.read()
        if not ok:
            break
        last = f
    cap.release()
    return first, last


def validate(pattern, board, dark_ratio):
    """전체 영상 기반 metric과 '첫/마지막 프레임 2장'만 쓴 판정이 일치하는지 확인."""
    dirs = [Path(d) for d in sorted(glob.glob(pattern)) if Path(d).is_dir()]
    agree = disagree = nofind = 0
    rows = []
    for d in dirs:
        target = M.task_of(d)
        first, last = first_last_frames(M.top_video(d))
        c = EraseChecker(board, dark_ratio)
        try:
            c.set_reference(first)
            r = c.check(last, target)
        except RuntimeError as e:
            nofind += 1
            print(f"  [검출실패] {d.name}: {e}")
            continue
        if not r["target_found"]:
            nofind += 1
            print(f"  [검출실패] {d.name}: {r['reason']}")
            continue

        # 기준 정답: 전체 영상 기반 metric
        shapes, ink, occ = M.track(M.top_video(d), board, dark_ratio)
        per = M.summarize(ink, occ)
        tgt = M.aggregate(per, target)
        others = [M.aggregate(per, l) for l in {k.split("#")[0] for k in per} if l != target]
        full_ok = bool(
            tgt
            and tgt["erased_frac"] >= SUCCESS_ERASED
            and max((o["erased_frac"] for o in others if o), default=0.0) <= MAX_DISTRACTOR
        )
        if full_ok == r["success"]:
            agree += 1
        else:
            disagree += 1
            print(
                f"  [불일치] {d.name}: 2프레임 판정={r['success']} (te={r['target_erased']}, "
                f"de={r['max_distractor_erased']}) vs 전체영상={full_ok}"
            )
        rows.append((d.name, r["success"], full_ok))

    total = agree + disagree
    print(f"\n검출 실패 {nofind} / 판정 비교 {total}건")
    if total:
        print(f"일치 {agree} ({agree / total:.1%}) / 불일치 {disagree}")
    return rows


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--validate", metavar="GLOB", help="에피소드들의 첫/마지막 프레임으로 오프라인 검증")
    p.add_argument("--before", type=Path, help="시도 전 이미지")
    p.add_argument("--after", type=Path, help="시도 후 이미지")
    p.add_argument("--target", default=None, help="circle / triangle / rectangle")
    p.add_argument("--board", type=lambda s: tuple(int(v) for v in s.split(",")), default=M.DEFAULT_BOARD)
    p.add_argument("--dark-ratio", type=float, default=0.72)
    a = p.parse_args(argv)

    if a.validate:
        validate(a.validate, a.board, a.dark_ratio)
        return 0
    if not (a.before and a.after and a.target):
        p.error("--validate 또는 (--before --after --target) 중 하나가 필요")
    c = EraseChecker(a.board, a.dark_ratio)
    c.set_reference(cv2.imread(str(a.before)))
    print(c.check(cv2.imread(str(a.after)), a.target))
    return 0


def _selftest():
    board_bgr, ink_bgr = (240, 240, 238), (60, 60, 55)

    def scene(erase_circle=False, erase_rect=False):
        f = np.full((400, 400, 3), board_bgr, np.uint8)
        if not erase_circle:
            cv2.circle(f, (100, 100), 40, ink_bgr, 3)
        if not erase_rect:
            cv2.rectangle(f, (250, 250), (330, 330), ink_bgr, 3)
        return f

    board = (0, 0, 400, 400)
    c = EraseChecker(board)
    c.set_reference(scene())

    r = c.check(scene(erase_circle=True), "circle")
    assert r["success"] and r["target_erased"] > 0.9, r

    r = c.check(scene(), "circle")  # 아무것도 안 지움
    assert not r["success"] and r["reason"] == "target 덜 지워짐", r
    assert r["remaining_frac"] == 1.0, r
    # 잔여 위치: 원이 bbox를 꽉 채우므로 무게중심은 가운데여야 한다
    assert r["residual"] and r["residual"]["where"] == "중간 가운데", r["residual"]

    # 잔여가 한쪽에 몰려 있으면 그쪽으로 잡히는가 — 아래 절반만 지운 원
    half = scene()
    cv2.rectangle(half, (50, 100), (150, 160), board_bgr, -1)  # 원의 아래쪽을 덮는다
    r = c.check(half, "circle")
    assert not r["success"], r
    assert r["residual"]["centroid"][1] < 0.5, r["residual"]  # 무게중심이 위쪽

    r = c.check(scene(erase_circle=True, erase_rect=True), "circle")  # distractor까지 지움
    assert not r["success"] and r["reason"] == "distractor 침범", r

    r = c.check(scene(erase_circle=True), "triangle")  # 없는 도형을 target으로
    assert not r["success"] and not r["target_found"], r

    # retry_loop: 2번째 시도에서 성공하면 3번째는 안 돈다
    calls = {"n": 0}
    frames = [scene(), scene(), scene(erase_circle=True)]

    def grab():
        return frames[min(calls["n"], len(frames) - 1)]

    def attempt():
        calls["n"] += 1

    h = retry_loop(EraseChecker(board), grab, attempt, "circle", max_attempts=3)
    assert len(h) == 2 and h[-1]["success"], h
    print("selftest OK")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    if "--selftest" in sys.argv:
        _selftest()
    else:
        sys.exit(main())
