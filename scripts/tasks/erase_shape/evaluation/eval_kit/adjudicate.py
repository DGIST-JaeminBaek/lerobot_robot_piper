#!/usr/bin/env python3
"""저장된 실행 폴더를 오프라인으로 재판정하고 에피소드 1행 CSV로 쌓는다.

    # 아직 판정 안 된 실행만 골라 처리 (기본)
    python scripts/tasks/erase_shape/evaluation/eval_kit/adjudicate.py records/hil/*

    # 전부 다시 판정 (임계값을 바꿨을 때)
    python scripts/tasks/erase_shape/evaluation/eval_kit/adjudicate.py records/hil/* --force

로봇도, 카메라도, 정책도 필요 없다. 입력은 erase_run.py가 이미 남기는 것들뿐이다:

    <run_dir>/meta.json        실행 조건
    <run_dir>/log.json         게이트가 내린 판정 + 러너 요약
    <run_dir>/00_reference.png 시도 전 기준 프레임
    <run_dir>/NN_after.png     시도 N 직후 프레임
    <run_dir>/eval_kit.json    (선택) 재현성 사이드카 — stamp.py가 남긴다

산출물:
    <run_dir>/eval_kit/verdict.json   이 실행의 독립 판정 전문
    outputs/analysis/episodes.csv     에피소드당 1행 (조건별 집계는 summarize.py)

기존 파일은 하나도 안 건드린다. 폴더를 통째로 지우면 원래 상태로 돌아간다.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2  # noqa: E402

from judge import Judge, config_hash, load_config  # noqa: E402

DEFAULT_CSV = Path("outputs/analysis/episodes.csv")

# CSV 스키마. 순서를 바꾸지 말고 뒤에만 추가할 것 — 기존 CSV와 이어붙는다.
COLUMNS = [
    # ── 신원 ──
    "run_id", "attempt", "is_final", "run_dir",
    # ── 재현성 (사이드카 + meta.json) ──
    "condition", "pattern_id", "operator", "commit", "dirty",
    "started", "policy_path", "dataset_root", "task", "target",
    "mode", "hil", "aggregate_fn", "max_relative_target", "top_crop", "wrist_crop",
    "judge_version", "config_hash", "adjudicated_at",
    # ── 성과 ──
    "judge_success", "target_erased", "remaining_frac",
    # ── 위반·부작용 ──
    "shape_damage", "damage_violation", "target_incomplete", "worst_distractor",
    # ── 실행 특성 ──
    "steps", "max_steps", "timeout", "interventions", "measured_fps",
    "runner_status", "aborted",
    # ── 게이트 대조 (동어반복 방지의 증거) ──
    "gate_success", "gate_target_erased", "disagree",
    # ── 자유 메모 ──
    "notes",
]


def _load_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def adjudicate_run(run_dir: Path, cfg: dict) -> dict:
    """실행 폴더 하나를 재판정. 반환: verdict dict (rows 포함)."""
    import time

    meta = _load_json(run_dir / "meta.json", {}) or {}
    gate_log = _load_json(run_dir / "log.json", []) or []
    side = _load_json(run_dir / "eval_kit.json", {}) or {}

    target = side.get("target") or meta.get("target")
    if not target:
        raise ValueError(f"{run_dir}: target을 모른다 (meta.json 확인)")

    ref_path = run_dir / "00_reference.png"
    if not ref_path.exists():
        raise FileNotFoundError(f"{run_dir}: 기준 프레임(00_reference.png)이 없다")
    ref_frame = cv2.imread(str(ref_path))
    if ref_frame is None:
        raise RuntimeError(f"{run_dir}: 기준 프레임을 못 읽었다")

    judge = Judge(cfg)
    judge.set_reference(ref_frame)

    after_paths = sorted(run_dir.glob("[0-9][0-9]_after.png"))
    if not after_paths:
        raise FileNotFoundError(f"{run_dir}: 시도 후 프레임(NN_after.png)이 없다")

    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    chash = config_hash(cfg)
    max_steps = meta.get("max_steps")
    rows, verdicts = [], []

    for path in after_paths:
        attempt = int(path.name.split("_")[0])
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"{path}: 프레임을 못 읽었다")
        v = judge.judge(frame, target)
        v["attempt"] = attempt
        v["frame"] = path.name
        verdicts.append(v)

        gate = next((g for g in gate_log if g.get("attempt") == attempt), {})
        steps = gate.get("steps")
        rows.append({
            "run_id": run_dir.name,
            "attempt": attempt,
            "is_final": int(path is after_paths[-1]),
            "run_dir": str(run_dir),

            "condition": side.get("condition", ""),
            "pattern_id": side.get("pattern_id", ""),
            "operator": side.get("operator", ""),
            "commit": side.get("commit", ""),
            "dirty": side.get("dirty", ""),
            "started": meta.get("started", ""),
            "policy_path": meta.get("policy_path", ""),
            "dataset_root": meta.get("dataset_root", ""),
            "task": meta.get("task", ""),
            "target": target,
            "mode": meta.get("mode", ""),
            "hil": int(bool(meta.get("hil"))),
            "aggregate_fn": meta.get("aggregate_fn") or "",
            "max_relative_target": meta.get("max_relative_target") if meta.get("max_relative_target") is not None else "",
            "top_crop": meta.get("top_crop", ""),
            "wrist_crop": meta.get("wrist_crop", ""),
            "judge_version": cfg.get("judge_version", ""),
            "config_hash": chash,
            "adjudicated_at": stamp,

            "judge_success": int(bool(v["success"])),
            "target_erased": v.get("target_erased", ""),
            "remaining_frac": v.get("remaining_frac", ""),

            "shape_damage": v.get("shape_damage", ""),
            "damage_violation": int(bool(v.get("damage_violation"))),
            "target_incomplete": int(bool(v.get("target_incomplete"))),
            "worst_distractor": v.get("worst_distractor") or "",

            "steps": steps if steps is not None else "",
            "max_steps": max_steps if max_steps is not None else "",
            # 스텝 예산을 다 쓰고 끝났나. 성공률과 별도로 봐야 하는 값이다 —
            # 빠르게 실패하는 정책과 시간이 모자란 정책은 다른 문제다.
            "timeout": int(steps is not None and max_steps is not None and steps >= max_steps),
            "interventions": gate.get("interventions", ""),
            "measured_fps": gate.get("measured_fps", ""),
            "runner_status": gate.get("status", ""),
            "aborted": int(bool(gate.get("aborted"))),

            "gate_success": int(bool(gate.get("success"))) if gate else "",
            "gate_target_erased": gate.get("target_erased", ""),
            # ★ 이 칼럼이 분리의 존재 이유다. 1이 쌓이면 게이트나 판정기 중
            #   하나가 틀린 것이고, 어느 쪽인지는 저장된 프레임으로 가린다.
            "disagree": int(bool(gate) and bool(gate.get("success")) != bool(v["success"])),

            "notes": side.get("notes", ""),
        })

    verdict = {
        "run_id": run_dir.name,
        "adjudicated_at": stamp,
        "judge_version": cfg.get("judge_version"),
        "config_hash": chash,
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "target": target,
        "reference_shapes": [
            {"key": k, "box": list(box), "ink": round(judge.reference["ink"][k], 6)}
            for k, box in judge.reference["shapes"]
        ],
        "attempts": verdicts,
        "final_success": bool(verdicts[-1]["success"]),
    }
    out_dir = run_dir / "eval_kit"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2))
    verdict["_rows"] = rows
    return verdict


# ── CSV 누적 ────────────────────────────────────────────────
def read_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_rows(csv_path: Path, rows: list[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = csv_path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})
    tmp.replace(csv_path)  # 중간에 죽어도 기존 CSV가 안 깨진다


def upsert(csv_path: Path, new_rows: list[dict]) -> tuple[int, int]:
    """(run_id, attempt) 키로 갈아끼운다. 재판정해도 행이 중복되지 않는다."""
    existing = read_rows(csv_path)
    keys = {(r["run_id"], str(r["attempt"])) for r in new_rows}
    kept = [r for r in existing if (r.get("run_id"), str(r.get("attempt"))) not in keys]
    merged = kept + new_rows
    merged.sort(key=lambda r: (str(r.get("started", "")), str(r.get("run_id")), int(r.get("attempt", 0))))
    write_rows(csv_path, merged)
    return len(new_rows), len(existing) - len(kept)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("run_dirs", nargs="+", help="records/hil/<시각> 폴더들 (glob 가능)")
    p.add_argument("--csv", default=str(DEFAULT_CSV), help=f"누적 CSV (기본 {DEFAULT_CSV})")
    p.add_argument("--config", default=None, help="judge_config.json 경로 (임계값 sweep용)")
    p.add_argument("--force", action="store_true", help="이미 판정된 폴더도 다시 판정")
    p.add_argument("--no-csv", action="store_true", help="verdict.json만 쓰고 CSV는 안 건드림")
    a = p.parse_args(argv)

    cfg = load_config(a.config)
    csv_path = Path(a.csv)
    all_rows, done, skipped, failed, incomplete = [], 0, 0, 0, 0

    for raw in a.run_dirs:
        d = Path(raw)
        if not d.is_dir():
            continue
        if not (d / "meta.json").exists():
            continue  # records/hil/README.md 같은 것들
        if not a.force and (d / "eval_kit" / "verdict.json").exists():
            skipped += 1
            continue
        try:
            v = adjudicate_run(d, cfg)
        except FileNotFoundError as e:
            # 중단된 실행(프레임이 안 남은 폴더)은 오류가 아니다 — 판정할 게 없을 뿐.
            # 이걸 실패로 세면 종료코드가 1이 되어 파이프라인에 못 넣는다.
            print(f"[미완] {d.name}: {e}")
            incomplete += 1
            continue
        except Exception as e:  # 한 폴더가 깨져도 나머지는 처리한다
            print(f"[실패] {d.name}: {e}")
            failed += 1
            continue
        all_rows.extend(v.pop("_rows"))
        done += 1
        last = v["attempts"][-1]
        print(f"[OK] {d.name}  success={v['final_success']}  "
              f"target_erased={last.get('target_erased')}  "
              f"damage={last.get('shape_damage')}  "
              f"({len(v['attempts'])}시도)")

    if all_rows and not a.no_csv:
        added, replaced = upsert(csv_path, all_rows)
        print(f"\n[CSV] {csv_path} — {added}행 기록 (갱신 {replaced}행)")
    print(f"[요약] 판정 {done} / 이미판정 {skipped} / 미완 {incomplete} / 실패 {failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
