#!/usr/bin/env python3
"""315개 녹화 전체의 시작/종료 프레임을 처음부터 다시 계산한다.

기존 QC JSON(`configs/erase_shape_frame_ranges.json`,
`configs/0802_qc_review_local.json`, `configs/qc_review_local.json`)은 읽지도
쓰지도 않는다 — 이 스크립트는 `qc_core.inspect()`만으로 완전히 새로 계산해
별도 파일에 저장한다. 계산 로직 자체는 바꾸지 않았다(`episode_segmentation`을
그대로 통해서 쓴다); 이 스크립트가 하는 새로운 일은 8개 세션 폴더를 순회하는
것뿐이다.

`records/0727`만 도형별 하위폴더(`erase_the_circle/` 등) 구조라 그 밑의 각
하위폴더를 별도 배치로 취급한다 — `qc_core.compare_group()`의 이상치 판정이
배치 단위(길이 z-score, 시작 자세 이탈)라서, 배치 경계를 기존에 `qc_studio.py
--folder`를 폴더별로 돌리던 것과 동일하게 유지해야 라벨 재현성이 보장된다.

실행:
    python qc_recompute_all.py --sessions 0805 --output /tmp/x.json
    python qc_recompute_all.py   # 8개 세션 전체
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(TASK_DIR / "dataset"))
from episode_segmentation import DEFAULT_END_EVENT, END_EVENTS, END_MARGIN, gripper_plateau
from qc_core import Report, compare_group, inspect, read_episode


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RECORDS_ROOT = REPO_ROOT / "records"
DEFAULT_OUTPUT = REPO_ROOT / "configs/qc_review_all_315.json"

DEFAULT_SESSIONS = [
    "0727",
    "0802_joint4_corrected",
    "0804_joint4_corrected",
    "0805",
    "0811",
    "0812",
    "0813",
    "local",
]

# 병합/가공 산출물 — raw 녹화가 아니므로 제외한다 (shape_position_analysis.py와 동일).
EXCLUDED_DIRS = {"erase_the_shape", "erase_the_shape_512"}


def find_batches(records_root: Path, session: str) -> list[Path]:
    """이 세션에서 qc_core.compare_group()의 배치 단위가 될 폴더 목록.

    0727은 도형별 하위폴더가 각각 배치(과거 qc_studio.py --folder 실행 단위와
    동일), 나머지는 세션 폴더 자신이 배치."""
    session_dir = records_root / session
    if not session_dir.is_dir():
        print(f"[WARN] {session_dir} 없음 — 건너뜀")
        return []
    subdirs = [p for p in sorted(session_dir.iterdir()) if p.is_dir() and p.name not in EXCLUDED_DIRS]
    if any((p / "videos").is_dir() for p in subdirs):
        return [session_dir]  # 평면 구조: 세션 폴더 자체가 배치
    return [p for p in subdirs if any((c / "videos").is_dir() for c in p.iterdir() if c.is_dir())]


def parse_shape(name: str) -> str:
    match = re.search(r"erase_the_(\w+?)_", name)
    return match.group(1) if match else "unknown"


def gripper_trace(path: Path, start: int, end: int, margin: int) -> dict:
    """웹 리뷰에서 종료 구간 프레임과 나란히 보여줄 그리퍼 명령/실측 값.

    `qc_extract_review_frames.py`가 뽑는 종료 프레임 창(`[max(start,end-margin), end)`)과
    정확히 같은 인덱스 범위를 써야 이미지와 숫자가 프레임 단위로 정렬된다."""
    frame, _ = read_episode(path)
    action = np.stack(frame["action"].to_numpy())
    state = np.stack(frame["observation.state"].to_numpy())
    window = list(range(max(start, end - margin), end))
    return {
        "frame_index": window,
        "action": [round(float(action[i, 6]), 2) for i in window],
        "state": [round(float(state[i, 6]), 2) for i in window],
        "plateau": round(float(gripper_plateau(action)), 2),
    }


def report_to_episode(report: Report, session: str, margin: int) -> dict:
    return {
        "source_dataset": report.path.resolve().relative_to(REPO_ROOT).as_posix(),
        "session": session,
        "shape": parse_shape(report.name),
        "total_frames": report.total_frames,
        "fps": report.fps,
        "cameras": [c.removeprefix("observation.images.") for c in report.cameras],
        "broken": report.broken,
        "missing": list(report.missing),
        "qc_level": report.level,
        "qc_notes": report.reasons + report.notes,
        "auto_start_frame": report.start,
        "auto_end_frame": report.end,
        "motion_onset": report.motion_onset,
        "gripper_release": report.gripper_release,
        "gripper_release_done": report.gripper_release_done,
        "erased_ratio": None if report.erased is None else round(report.erased, 3),
        "n_shapes": report.n_shapes,
        "n_jumps": report.n_jumps,
        "max_jump": round(report.max_jump, 2),
        "start_dev": report.start_dev,
        "length_z": report.length_z,
        "gripper_end": gripper_trace(report.path, report.start, report.end, margin)
        if not report.broken and report.start is not None and report.end is not None
        else {},
    }


def recompute(records_root: Path, sessions: list[str], margin: int, **inspect_kwargs) -> list[dict]:
    episodes: list[dict] = []
    for session in sessions:
        for batch_dir in find_batches(records_root, session):
            reports = [inspect(path, **inspect_kwargs) for path in sorted(p for p in batch_dir.iterdir() if p.is_dir())]
            compare_group(reports)
            episodes.extend(report_to_episode(r, session, margin) for r in reports)
    return episodes


def regression_check(episodes: list[dict], reference_json: Path) -> None:
    """기존 라벨(qc_review_local.json)의 0805 항목 하나와 재계산 결과를 대조한다.

    참조 JSON의 source_dataset은 생성 당시 `records/local/...`였지만 그 폴더는
    이후 다른 내용으로 재사용됐다 (local은 스테이징 폴더라 내용이 바뀐다) —
    그래서 전체 경로가 아니라 폴더 이름(basename)으로 매칭한다."""
    if not reference_json.is_file():
        print(f"[SKIP] 회귀 검증용 참조 파일 없음: {reference_json}")
        return
    reference = json.loads(reference_json.read_text())
    target_name = "erase_the_circle_0805-183247"
    old = next((e for e in reference["episodes"] if Path(e["source_dataset"]).name == target_name), None)
    new = next((e for e in episodes if Path(e["source_dataset"]).name == target_name), None)
    if old is None or new is None:
        print(f"[SKIP] 회귀 검증용 녹화({target_name})를 찾을 수 없음")
        return
    ok = old["auto_start_frame"] == new["auto_start_frame"] and old["auto_end_frame"] == new["auto_end_frame"]
    status = "OK" if ok else "MISMATCH"
    print(
        f"[{status}] 회귀 검증 {target_name}: "
        f"기존 auto=({old['auto_start_frame']},{old['auto_end_frame']}) "
        f"신규 auto=({new['auto_start_frame']},{new['auto_end_frame']})"
    )


def print_summary(episodes: list[dict]) -> None:
    by_session: dict[str, int] = {}
    by_shape: dict[str, int] = {}
    broken = 0
    for ep in episodes:
        by_session[ep["session"]] = by_session.get(ep["session"], 0) + 1
        by_shape[ep["shape"]] = by_shape.get(ep["shape"], 0) + 1
        broken += ep["broken"]
    print(f"[SUMMARY] 전체 {len(episodes)}개, broken {broken}개")
    print(f"[SUMMARY] 세션별: {by_session}")
    print(f"[SUMMARY] 도형별: {by_shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records-root", type=Path, default=DEFAULT_RECORDS_ROOT)
    parser.add_argument("--sessions", type=str, default=",".join(DEFAULT_SESSIONS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-margin", type=int, default=22)
    parser.add_argument("--end-margin", type=str, default="", help="비우면 end-event별 기본값 사용")
    parser.add_argument("--round-start", type=int, default=10)
    parser.add_argument("--end-event", choices=END_EVENTS, default=DEFAULT_END_EVENT)
    parser.add_argument(
        "--margin",
        type=int,
        default=20,
        help="웹 리뷰용 종료 구간 그리퍼 궤적 창 크기 — qc_extract_review_frames.py --margin과 같아야 프레임/숫자가 정렬됨 (default: 20)",
    )
    parser.add_argument(
        "--reference-json",
        type=Path,
        default=REPO_ROOT / "configs/qc_review_local.json",
        help="회귀 검증에 쓸 기존 라벨 파일 (읽기 전용, 건드리지 않음)",
    )
    parser.add_argument("--no-regression-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sessions = [s.strip() for s in args.sessions.split(",") if s.strip()]
    end_margin = int(args.end_margin) if args.end_margin.strip() else None

    episodes = recompute(
        args.records_root,
        sessions,
        args.margin,
        start_margin=args.start_margin,
        end_margin=end_margin,
        round_start=args.round_start,
        end_event=args.end_event,
    )

    payload = {
        "format_version": 1,
        "generated_by": "scripts/tasks/erase_shape/qc/qc_recompute_all.py",
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "range_semantics": "start_frame inclusive, end_frame exclusive",
        "end_event": args.end_event,
        "start_margin": args.start_margin,
        "end_margin": end_margin,
        "round_start": args.round_start,
        "margin": args.margin,
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"[WRITE] {args.output} ({len(episodes)}개 에피소드)")

    print_summary(episodes)
    if not args.no_regression_check:
        regression_check(episodes, args.reference_json)


if __name__ == "__main__":
    main()
