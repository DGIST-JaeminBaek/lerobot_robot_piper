#!/usr/bin/env python3
"""erase_eval_ui의 세션 흐름을 창을 띄운 채 스크립트로 돌려본다.

mainloop 대신 update()로 이벤트를 돌리므로 사람 없이 실행된다. 확인하는 것:
  시작 → 채점 → 사람이 실패 유형 수정 → 확인 → CSV 1행
  버리기는 CSV에 안 남는다
  창을 다시 열면 trial 번호를 이어받는다

기존 시연 에피소드가 필요하다 (로봇·카메라는 불필요):
    python scripts/tools/test_erase_eval_ui.py '<데이터셋>/erase_the_rectangle_*'
"""

from __future__ import annotations

import argparse
import glob
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import erase_eval as E  # noqa: E402
import erase_eval_ui as U  # noqa: E402


def run(episodes: list[Path], out_dir: Path) -> None:
    args = argparse.Namespace(
        model="testmodel", condition="testcond", target=None, trials=2, cutoff=60.0,
        rollout_cmd=None, watch_dir=None, dry_run=episodes, out_dir=out_dir,
        board=list(E.M.DEFAULT_BOARD), dark_ratio=0.72, fps=30.0,
    )
    app = U.EvalSession(args)
    app.withdraw()

    def pump(seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            app.update()
            time.sleep(0.05)

    assert app.trial == 1 and app.phase == "idle"

    app.on_start()
    assert app.phase == "running", "dry-run 롤아웃이 안 돎"
    deadline = time.time() + 180
    while app.row is None and time.time() < deadline:
        app.update()
        time.sleep(0.1)
    assert app.row is not None, "채점 결과가 안 옴"
    assert app.phase == "review", app.phase
    print("headline:", app.headline.cget("text"))
    print(app.detail.cget("text"))

    # 사람이 자동 판정을 뒤집는 경로 — 원본이 메모에 남아야 나중에 일치도를 볼 수 있다.
    app.note.insert("1.0", "테스트 메모")
    auto = app.row.get("failure_mode") or "success"
    app.failure.set("premature_release")
    app.on_confirm()
    assert app.trial == 2 and app.row is None and app.phase == "idle"

    rows = E._read_csv(out_dir / "episodes.csv")
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["failure_mode"] == "premature_release", row
    assert row["success"] is False, "사람이 실패 유형을 붙였는데 성공으로 남음"
    assert f"auto={auto}" in row["note"] and "테스트 메모" in row["note"], row["note"]
    assert (out_dir / "summary.md").exists() and (out_dir / "summary.json").exists()

    # 버리기: 기록하지 않는다
    if len(episodes) > 1:
        app.on_start()
        pump(1.0)
        app.on_discard()
        assert app.phase == "idle" and app.row is None
        assert len(E._read_csv(out_dir / "episodes.csv")) == 1, "버린 시도가 기록됨"

    app.destroy()

    # 재시작 시 같은 model/condition의 시행 번호를 이어받는다
    app2 = U.EvalSession(args)
    app2.withdraw()
    assert app2.trial == 2, app2.trial
    app2.destroy()
    print("UI 세션 흐름 OK")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("episodes", nargs="+", help="기존 에피소드 폴더 (glob 가능)")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args(argv)

    paths = [Path(q) for pat in args.episodes for q in sorted(glob.glob(pat)) if Path(q).is_dir()]
    if not paths:
        p.error("에피소드 폴더를 못 찾음")
    out = args.out_dir or Path(tempfile.mkdtemp()) / "session"
    run(paths[:2], out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
