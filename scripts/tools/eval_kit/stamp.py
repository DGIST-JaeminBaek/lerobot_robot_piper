#!/usr/bin/env python3
"""재현성 사이드카를 실행 폴더에 남긴다 — erase_run.py는 안 건드린다.

erase_run.py의 meta.json에는 정책·데이터셋·크롭 같은 실행 조건이 이미 다 있다.
없는 건 **사람과 물리 세계 쪽 정보**다:

    commit / dirty  그 실행이 어떤 코드로 돌았나
    condition       baseline / gate / ... (통계에서 조건을 가르는 키)
    pattern_id      보드에 그린 낙서 패턴 번호 — 페어링 통계의 전제
    operator        누가 돌렸나
    notes           그날의 이상 징후

이걸 <run_dir>/eval_kit.json으로 떨어뜨리고, adjudicate.py가 읽어 CSV에 넣는다.

    # 방금 끝난 실행에 도장 찍기 (가장 최근 폴더를 자동으로 고른다)
    python scripts/tools/eval_kit/stamp.py --latest records/hil \
        --condition gate --pattern-id P03 --operator seongil

    # 폴더를 직접 지정
    python scripts/tools/eval_kit/stamp.py records/hil/20260817-122743 --condition gate --pattern-id P03
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def git_state(repo: Path | None = None) -> dict:
    def run(*args):
        return subprocess.run(
            args, cwd=str(repo) if repo else None,
            capture_output=True, text=True, check=True,
        ).stdout.strip()

    try:
        return {
            "commit": run("git", "rev-parse", "--short", "HEAD"),
            "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": int(bool(run("git", "status", "--porcelain"))),
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": "", "branch": "", "dirty": ""}


def latest_run_dir(root: Path) -> Path:
    dirs = [d for d in root.iterdir() if d.is_dir() and (d / "meta.json").exists()]
    if not dirs:
        raise FileNotFoundError(f"{root} 아래에 실행 폴더가 없다")
    return max(dirs, key=lambda d: d.stat().st_mtime)


def stamp(run_dir: Path, **fields) -> Path:
    payload = {"stamped_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    payload.update(git_state(Path.cwd()))
    # 빈 값으로 기존 사이드카를 덮지 않는다 — 두 번 도장 찍어도 정보가 줄지 않게.
    existing = {}
    path = run_dir / "eval_kit.json"
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            existing = {}
    payload = {**existing, **payload, **{k: v for k, v in fields.items() if v not in (None, "")}}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("run_dir", nargs="?", help="records/hil/<시각>")
    p.add_argument("--latest", metavar="ROOT", help="이 폴더에서 가장 최근 실행을 고른다")
    p.add_argument("--condition", help="baseline / gate / ... (통계 조건 키)")
    p.add_argument("--pattern-id", help="보드 낙서 패턴 번호 (페어링용)")
    p.add_argument("--operator")
    p.add_argument("--notes")
    a = p.parse_args(argv)

    if a.latest:
        run_dir = latest_run_dir(Path(a.latest))
    elif a.run_dir:
        run_dir = Path(a.run_dir)
    else:
        p.error("run_dir 또는 --latest 중 하나가 필요")
    if not (run_dir / "meta.json").exists():
        p.error(f"{run_dir}는 erase_run 실행 폴더가 아니다 (meta.json 없음)")

    path = stamp(run_dir, condition=a.condition, pattern_id=a.pattern_id,
                 operator=a.operator, notes=a.notes)
    print(f"[STAMP] {path}")
    print(path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
