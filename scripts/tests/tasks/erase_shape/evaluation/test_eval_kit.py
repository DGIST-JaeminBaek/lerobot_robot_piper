#!/usr/bin/env python3
"""eval_kit 셀프테스트 — 로봇도 저장된 실행도 필요 없다 (합성 이미지로 돈다).

    python -m pytest scripts/tests/tasks/erase_shape/evaluation/test_eval_kit.py
"""

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

SCRIPTS_DIR = next(parent for parent in Path(__file__).resolve().parents if (parent / "0__launch_gui.sh").is_file())
KIT = SCRIPTS_DIR / "tasks" / "erase_shape" / "evaluation" / "eval_kit"
sys.path.insert(0, str(KIT))

import adjudicate as AD  # noqa: E402
import calibrate as CAL  # noqa: E402
import summarize as SUM  # noqa: E402
from judge import Judge, config_hash, load_config  # noqa: E402

BOARD_BGR, INK_BGR = (240, 240, 238), (60, 60, 55)


def scene(erase_circle=False, erase_rect=False):
    f = np.full((400, 400, 3), BOARD_BGR, np.uint8)
    if not erase_circle:
        cv2.circle(f, (100, 100), 40, INK_BGR, 3)
    if not erase_rect:
        cv2.rectangle(f, (250, 250), (330, 330), INK_BGR, 3)
    return f


def make_test_cfg():
    cfg = load_config()
    cfg["board"] = (0, 0, 400, 400)
    return cfg


def test_judge():
    cfg = make_test_cfg()
    j = Judge(cfg)
    ref = j.set_reference(scene())
    assert len(ref["shapes"]) == 2, ref["shapes"]

    r = j.judge(scene(erase_circle=True), "circle")
    assert r["success"] and r["target_erased"] > 0.9, r
    assert r["shape_damage"] <= cfg["max_damage"], r

    r = j.judge(scene(), "circle")  # 아무것도 안 지움
    assert not r["success"] and r["target_incomplete"] and r["remaining_frac"] == 1.0, r

    # distractor까지 지움 → 위반
    r = j.judge(scene(erase_circle=True, erase_rect=True), "circle")
    assert not r["success"] and r["damage_violation"], r
    assert r["shape_damage"] > 0.9 and r["worst_distractor"].startswith("rectangle"), r

    # 없는 도형을 target으로
    r = j.judge(scene(erase_circle=True), "triangle")
    assert not r["success"] and not r["target_found"], r

    # 나무 지우개 블록(유채색)이 도형 위에 놓여도 잉크가 늘지 않아야 한다.
    # 채도 필터가 빠지면 여기서 erased가 0으로 뒤집힌다 — 실물에서 났던 실패다.
    with_block = scene(erase_circle=True)
    cv2.rectangle(with_block, (70, 70), (130, 130), (60, 130, 190), -1)  # 갈색 블록
    r = j.judge(with_block, "circle")
    assert r["success"], r

    # 임계값을 바꾸면 설정 지문도 바뀌어야 한다 (CSV에서 섞임을 잡는 근거)
    other = dict(cfg, theta=0.5)
    assert config_hash(cfg) != config_hash(other)
    print("  judge OK")


def _fake_run(root: Path, name, target="circle", erase=True, gate_success=None):
    d = root / name
    d.mkdir(parents=True)
    cv2.imwrite(str(d / "00_reference.png"), scene())
    cv2.imwrite(str(d / "01_after.png"), scene(erase_circle=erase))
    (d / "meta.json").write_text(json.dumps({
        "started": f"2026-08-17 12:00:{name[-2:]}", "target": target,
        "max_steps": 940, "mode": "demo", "hil": False,
        "aggregate_fn": "weighted_average", "policy_path": "ckpt", "task": "t",
    }))
    (d / "log.json").write_text(json.dumps([{
        "attempt": 1, "success": erase if gate_success is None else gate_success,
        "target_erased": 1.0 if erase else 0.0, "steps": 940,
        "interventions": 0, "measured_fps": 29.4, "status": "finished", "aborted": False,
    }]))
    return d


def test_adjudicate():
    cfg = make_test_cfg()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ok = _fake_run(root, "run01", erase=True)
        bad = _fake_run(root, "run02", erase=False)
        # 게이트는 성공이라 했는데 실제로는 안 지워진 실행 — 분리가 잡아야 하는 것
        lie = _fake_run(root, "run03", erase=False, gate_success=True)

        rows = []  # 사이드카(eval_kit.json) 없이도 돌아야 한다
        for d, cond in ((ok, "gate"), (bad, "gate"), (lie, "baseline")):
            v = AD.adjudicate_run(d, cfg)
            rows.extend(v.pop("_rows"))
            assert (d / "eval_kit" / "verdict.json").exists()

        by_run = {r["run_id"]: r for r in rows}
        assert by_run["run01"]["judge_success"] == 1
        assert by_run["run02"]["judge_success"] == 0
        assert by_run["run03"]["judge_success"] == 0
        assert by_run["run03"]["disagree"] == 1, "게이트 불일치를 못 잡았다"
        assert by_run["run01"]["disagree"] == 0
        assert by_run["run01"]["timeout"] == 1  # steps == max_steps

        csv_path = root / "episodes.csv"
        AD.upsert(csv_path, rows)
        again = AD.read_rows(csv_path)
        assert len(again) == 3, again

        # 재판정해도 행이 늘지 않는다
        AD.upsert(csv_path, rows)
        assert len(AD.read_rows(csv_path)) == 3

        # 사이드카가 없으면 condition이 비고, 전부 한 그룹으로 묶인다
        stats = SUM.summarize(again)
        assert len(stats) == 1 and stats[0]["n"] == 3, stats
        assert stats[0]["disagree"] == 1, stats
        print("  adjudicate/CSV OK")


def test_sidecar_and_summary():
    import stamp as ST

    cfg = make_test_cfg()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        d = _fake_run(root, "run01", erase=True)
        ST.stamp(d, condition="gate", pattern_id="P03", operator="tester")
        # 두 번 찍어도 기존 값이 빈 값으로 덮이지 않아야 한다
        ST.stamp(d, condition=None, notes="두번째")
        side = json.loads((d / "eval_kit.json").read_text())
        assert side["condition"] == "gate" and side["pattern_id"] == "P03"
        assert side["notes"] == "두번째"

        rows = AD.adjudicate_run(d, cfg).pop("_rows")
        assert rows[0]["condition"] == "gate" and rows[0]["pattern_id"] == "P03"

        stats = SUM.summarize(rows)
        assert stats[0]["condition"] == "gate" and stats[0]["n"] == 1
        lo, hi = stats[0]["ci"]
        assert 0.0 <= lo <= hi <= 1.0 and hi < 1.0, (lo, hi)  # Wald와 달리 폭이 0이 아니다
        print("  sidecar/summary OK")


def test_calibration_math():
    # 완전 분리된 점수: 임계값이 그 사이 어딘가로 잡혀야 한다
    scores = [0.1, 0.3, 0.5, 0.95, 0.97, 0.99]
    gold = [0, 0, 0, 1, 1, 1]
    theta, j, tpr, fpr = CAL.roc_youden(scores, gold)
    assert j == 1.0 and tpr == 1.0 and fpr == 0.0, (theta, j, tpr, fpr)
    assert 0.5 < theta <= 0.95, theta

    assert CAL.cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert abs(CAL.cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1]) + 1.0) < 1e-9
    print("  calibration math OK")


def test_wilson():
    lo, hi = SUM.wilson(20, 20)
    assert hi == 1.0 and lo < 1.0, (lo, hi)  # Wald면 [1.0, 1.0]이 된다
    lo, hi = SUM.wilson(0, 20)
    assert lo == 0.0 and hi > 0.0, (lo, hi)
    lo, hi = SUM.wilson(17, 20)
    assert lo < 0.85 < hi, (lo, hi)
    print("  wilson OK")


if __name__ == "__main__":
    test_judge()
    test_adjudicate()
    test_sidecar_and_summary()
    test_calibration_math()
    test_wilson()
    print("eval_kit selftest OK")
