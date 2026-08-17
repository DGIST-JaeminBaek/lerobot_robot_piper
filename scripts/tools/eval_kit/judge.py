#!/usr/bin/env python3
"""독립 판정기 — 저장된 판정 프레임 2장으로 성공/위반을 다시 계산한다.

**왜 erase_check.py를 안 쓰는가.** erase_check는 게이트가 "이제 그만해도 되나"를
정하는 데 쓰는 코드다. 그 코드로 성능도 판정하면 "게이트가 성공이라 믿을 때 성공"이
되어 동어반복이 된다 — 게이트가 과대추정하는 방향으로 편향돼 있으면 성능이 그만큼
공짜로 부풀고, 그걸 검출할 방법이 없다.

그래서 이 파일은 scripts/tools의 어떤 모듈도 import하지 않는다. cv2/numpy만 쓴다.
도형 위치 찾기부터 잉크 측정, 임계값까지 전부 여기 안에서 독립적으로 정의되고,
임계값은 judge_config.json에 동결돼 있다. 두 경로가 어긋나면 CSV의 `disagree`
칼럼에 그대로 드러난다 — 그게 이 분리의 목적이다.

측정 정의(게이트와 개념은 같다. 같은 물리량을 재야 비교가 의미 있다):
    ink(box)      = box 안에서 (어둡고 && 무채색)인 픽셀 비율
                    무채색 조건이 없으면 나무 지우개 블록이 잉크로 잡혀서
                    다 지운 시도가 erased=0으로 뒤집힌다.
    erased(shape) = clip((ink_before - ink_after) / ink_before, 0, 1)
    target_erased = target 라벨 인스턴스들의 erased 중 **최솟값** (보수적)
    shape_damage  = target이 아닌 도형들의 erased 중 **최댓값** (= 부작용)
    success       = target_erased >= theta  AND  shape_damage <= max_damage
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

CONFIG_PATH = Path(__file__).with_name("judge_config.json")


def load_config(path: Path | str | None = None) -> dict:
    cfg = json.loads(Path(path or CONFIG_PATH).read_text())
    cfg["board"] = tuple(int(v) for v in cfg["board"])
    return cfg


def config_hash(cfg: dict) -> str:
    """설정 지문. CSV에 박아서 '어떤 임계값으로 낸 숫자인지'를 사후에 증명한다."""
    import hashlib

    payload = {k: v for k, v in sorted(cfg.items()) if not k.startswith("_")}
    blob = json.dumps(payload, sort_keys=True, default=list).encode()
    return hashlib.sha1(blob).hexdigest()[:10]


# ── 도형 찾기 (기준 프레임에서 한 번만) ───────────────────────
def _classify(contour) -> str:
    """convex hull 면적 / 최소외접사각형 면적으로 circle/triangle/rectangle."""
    hull = cv2.convexHull(contour)
    (rw, rh) = cv2.minAreaRect(contour)[1]
    rect_area = rw * rh
    if rect_area <= 0:
        return "unknown"
    extent = cv2.contourArea(hull) / rect_area
    if extent < 0.63:
        return "triangle"
    if extent > 0.88:
        return "rectangle"
    n = len(cv2.approxPolyDP(hull, 0.03 * cv2.arcLength(hull, True), True))
    return "rectangle" if n <= 4 else "circle"


def _merge_nearby(contours, gap):
    """끊긴 획을 한 도형으로 묶는다 (bbox 거리 union-find)."""
    boxes = [cv2.boundingRect(c) for c in contours]
    parent = list(range(len(contours)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, (x1, y1, w1, h1) in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            x2, y2, w2, h2 = boxes[j]
            dx = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
            dy = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
            if dx <= gap and dy <= gap:
                parent[find(i)] = find(j)

    groups: dict[int, list] = {}
    for i, c in enumerate(contours):
        groups.setdefault(find(i), []).append(c)
    return [np.vstack(g) for g in groups.values()]


def locate_shapes(frame, cfg) -> tuple[list[tuple[str, tuple]], float]:
    """기준 프레임에서 도형 [(label, (x,y,w,h)), ...]와 흰색 기준값을 찾는다."""
    bx, by, bw, bh = cfg["board"]
    sub = frame[by : by + bh, bx : bx + bw]
    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)[:, :, 1]
    white = float(np.percentile(gray, 90))
    ink = (gray < white * cfg["dark_ratio"]).astype(np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def achromatic(c):
        x, y, w, h = cv2.boundingRect(c)
        return float(sat[y : y + h, x : x + w].mean()) <= cfg["max_sat"]

    contours = _merge_nearby([c for c in contours if achromatic(c)], cfg["merge_gap"])

    fh, fw = frame.shape[:2]
    pad = cfg["pad"]
    shapes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < cfg["min_shape_area"]:
            continue
        if max(w / h, h / w) > cfg["max_aspect"] or w > cfg["max_span"] * bw or h > cfg["max_span"] * bh:
            continue
        x0, y0 = max(x + bx - pad, 0), max(y + by - pad, 0)
        x1, y1 = min(x + bx + w + pad, fw), min(y + by + h + pad, fh)
        shapes.append((_classify(c), (x0, y0, x1 - x0, y1 - y0)))
    shapes.sort(key=lambda s: s[1][1])  # 화면 위 -> 아래
    return shapes, white


def ink_frac(frame, box, ink_thr, max_sat) -> float:
    """box 안 '어둡고 무채색'인 픽셀 비율."""
    x, y, w, h = box
    sub = frame[y : y + h, x : x + w]
    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)[:, :, 1]
    return float(((gray < ink_thr) & (sat <= max_sat)).mean())


# ── 판정 ────────────────────────────────────────────────────
class Judge:
    """기준 프레임 1장 + 시도 후 프레임 N장 → 시도별 판정."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.reference = None

    def set_reference(self, frame) -> dict:
        shapes, white = locate_shapes(frame, self.cfg)
        if not shapes:
            raise RuntimeError("기준 프레임에서 도형을 하나도 못 찾음 — 보드/조명 확인")
        ink_thr = white * self.cfg["dark_ratio"]
        keyed = [(f"{lbl}#{i}", box) for i, (lbl, box) in enumerate(shapes)]
        self.reference = {
            "ink_thr": ink_thr,
            "white": white,
            "shapes": keyed,
            "ink": {k: ink_frac(frame, box, ink_thr, self.cfg["max_sat"]) for k, box in keyed},
        }
        return self.reference

    def judge(self, frame, target: str) -> dict:
        if self.reference is None:
            raise RuntimeError("set_reference()를 먼저 호출할 것")
        ref = self.reference
        thr, max_sat = ref["ink_thr"], self.cfg["max_sat"]

        ink_after, erased = {}, {}
        for k, box in ref["shapes"]:
            before = ref["ink"][k]
            after = ink_frac(frame, box, thr, max_sat)
            ink_after[k] = after
            erased[k] = 0.0 if before <= 0 else float(np.clip((before - after) / before, 0.0, 1.0))

        tgt = [v for k, v in erased.items() if k.split("#")[0] == target]
        others = {k: v for k, v in erased.items() if k.split("#")[0] != target}
        target_erased = min(tgt) if tgt else None
        shape_damage = max(others.values()) if others else 0.0
        worst_distractor = max(others, key=others.get) if others else None

        if target_erased is None:
            return {
                "target_found": False,
                "success": False,
                "reason": f"기준 프레임에 {target} 도형이 없음",
                "detected": [k for k, _ in ref["shapes"]],
                "target_erased": None,
                "shape_damage": round(shape_damage, 4),
                "damage_violation": bool(shape_damage > self.cfg["max_damage"]),
                # 임계값을 나중에 다시 쓸어보려면(θ sweep) 원시값이 있어야 한다.
                "ink_before": {k: round(v, 6) for k, v in ref["ink"].items()},
                "ink_after": {k: round(v, 6) for k, v in ink_after.items()},
                "erased": {k: round(v, 4) for k, v in erased.items()},
            }

        target_ok = target_erased >= self.cfg["theta"]
        damage_violation = shape_damage > self.cfg["max_damage"]
        success = bool(target_ok and not damage_violation)
        reason = None
        if not success:
            reason = "target 덜 지워짐" if not target_ok else "distractor 침범"
            if not target_ok and damage_violation:
                reason = "target 덜 지워짐 + distractor 침범"

        return {
            "target_found": True,
            "success": success,
            "reason": reason,
            "target_erased": round(target_erased, 4),
            "remaining_frac": round(1.0 - target_erased, 4),
            "shape_damage": round(shape_damage, 4),
            "worst_distractor": worst_distractor,
            "damage_violation": bool(damage_violation),
            "target_incomplete": bool(not target_ok),
            "detected": [k for k, _ in ref["shapes"]],
            "ink_before": {k: round(v, 6) for k, v in ref["ink"].items()},
            "ink_after": {k: round(v, 6) for k, v in ink_after.items()},
            "erased": {k: round(v, 4) for k, v in erased.items()},
        }


def judge_pair(before_path, after_path, target, cfg=None) -> dict:
    """이미지 경로 2장으로 바로 판정 (CLI·캘리브레이션용)."""
    before = cv2.imread(str(before_path))
    after = cv2.imread(str(after_path))
    if before is None or after is None:
        raise FileNotFoundError(f"이미지를 못 읽었다: {before_path} / {after_path}")
    j = Judge(cfg)
    j.set_reference(before)
    return j.judge(after, target)


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="독립 판정기 — 이미지 2장으로 판정")
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.add_argument("--target", required=True, choices=["circle", "triangle", "rectangle"])
    p.add_argument("--config", default=None)
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    print(json.dumps(judge_pair(a.before, a.after, a.target, cfg), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
