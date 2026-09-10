#!/usr/bin/env python3
"""erase_eval.py — 지우기 롤아웃의 자동 채점기 (사람 개입 없이 CSV/JSON/MD 산출).

측정은 새로 만들지 않는다. erase_check.py(실시간 게이트)의 `EraseChecker`를 그대로
재사용한다 — reference(시도 전) / judge(시도 후, park) 프레임 두 장으로 target의
잉크 잔여 비율만 잰다:

  erase_check.EraseChecker   -> reference/judge 프레임 2장으로 erased_target 계산
  data/*.parquet 의 action[:,6]  -> 그리퍼 개폐 = 파지/놓기 시점(진단용, 판정과 무관)
  ------------------------------------------------------------------
  erase_eval.py               -> 종료 사유, 조건별 집계·요약(CSV/MD/JSON)

`reference_frame.png`/`judge_frame.png`는 erase_run.py가 실시간 게이트와 같은
순간에 찍어 롤아웃 폴더에 넣어준다 — 그래서 이 오프라인 채점기와 실행 중 게이트가
항상 같은 숫자를 본다. 그 두 파일이 없는 과거 데이터는 영상의 첫/마지막 프레임으로
대체한다(erase_check.first_last_frames와 동일한 fallback).

2026-08-20: 접근/접촉/개시/완료/선택성 5단계 점수, distractor 침범 추적, 진행도
곡선(t50/t90/AUC/정체·회복)은 전부 폐지했다 — VLA가 release나 max-steps까지 스스로
제어하게 두고, 채점은 첫/마지막 잉크 비율만 본다. 옛 설계와 근거는 git 이력
(`tmp/eval_backup_before_endstep_ink_redesign_*`)과
docs/tasks/erase_shape/runtime/erase_run_design.md에 남아 있다.

사용:
  # 하드웨어 없이 자기검증 (합성 이미지)
  python scripts/tasks/erase_shape/evaluation/erase_eval.py --selftest

  # 기존 시연 데이터로 채점기 자체를 검증 (지움 성공이어야 함)
  python scripts/tasks/erase_shape/evaluation/erase_eval.py '0802/0804/erase_the_rectangle_*' --model demo --condition teleop

  # 실제 롤아웃 채점
  python scripts/tasks/erase_shape/evaluation/erase_eval.py 'rollouts/smolvla_async/*' --model smolvla --condition async \\
      --out-dir outputs/evaluation/erase_shape/0818_smolvla_async

산출물(--out-dir):
  episodes.csv   에피소드 1행씩. 이어붙이므로 여러 번 돌려도 누적된다.
  summary.json   조건별 집계 (신뢰구간 포함). AI/스크립트 입력용.
  summary.md     사람이 읽는 요약. 그대로 보고서에 붙일 수 있다.
"""

from __future__ import annotations

import argparse
import csv
import os
import glob
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_DIR / "lib"))
import ink_metric as M  # noqa: E402  (cv2 의존은 여기 한 곳으로 모은다)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import erase_check as EC  # noqa: E402

# pyarrow는 반드시 **모듈 임포트 시점(=메인 스레드)**에 잡아둔다. 예전에는 read_gripper()
# 안에서 지연 임포트했는데, 그러면 pyarrow를 '처음' 임포트하는 게 채점 워커 스레드가
# 된다(erase_eval_ui.py의 _begin_scoring()이 스레드에서 score_episode를 부른다).
# 메인이 아닌 스레드에서 pyarrow를 최초 임포트하면 libarrow가 세그폴트를 내면서
# 프로세스를 통째로 죽인다 — 파이썬 예외가 아니라 SIGSEGV라 try/except로도 못 잡는다.
# 실측(2026-08-21, pyarrow 25.0.0): 워커 스레드에서 임포트 10/10 크래시,
# 메인 스레드에서 미리 임포트해두면 0/10. 실물에서도 평가 GUI가 시도를 몇 번 돌리다
# "Segmentation fault (core dumped)"로 죽는 것으로 4번 재현됐고, 크래시 주소가
# (libarrow +0x13b4c89) 4번 모두 동일했다.
# None이면 pyarrow가 아예 없는 환경 — read_gripper()가 경고하고 종료 사유 판정만 끈다.
try:
    import pyarrow.parquet as _pq  # noqa: E402
except ImportError:  # pragma: no cover - 프로젝트 환경에는 항상 있다
    _pq = None


# ═══════════════════════════════════════════════════════════════════
# 채점 상수 — EraseChecker(ink_metric.SUCCESS_ERASED/MAX_DISTRACTOR)와 값을 공유한다
# ═══════════════════════════════════════════════════════════════════
SUCCESS_ERASED = M.SUCCESS_ERASED  # 0.90 — target 완료 임계
MAX_DISTRACTOR = M.MAX_DISTRACTOR  # 0.10 — distractor 허용 상한(노이즈 플로어 3배)


# 그리퍼(action[:,6], 0~100). 실측: 대기 ~0, 벌림 60~100, 지우개 파지 ~20.
GRIP_OPEN = 45.0  # 이 이상이면 벌린 상태
GRIP_HOLD_MAX = 35.0  # 이 이하로 닫혀 있으면 뭔가 쥔 상태로 본다
GRIP_HOLD_MIN = 8.0  # 완전히 닫힌(빈손) 것과 구분
HOLD_FRAMES = 15  # 0.5초 이상 유지돼야 파지로 인정
RELEASE_JUMP = 15.0  # 파지 수준에서 이만큼 벌어지면 놓은 것


# ═══════════════════════════════════════════════════════════════════
# 그리퍼 위상 — 파지 / 놓기 (진단용. 판정에는 안 쓴다)
# ═══════════════════════════════════════════════════════════════════
def gripper_phases(grip: np.ndarray) -> dict:
    """그리퍼 명령 시계열에서 파지 시작·놓기 프레임을 찾는다.

    실물에서 "놓쳤다"와 "놓았다"는 눈으로 구분하기 어렵지만 명령값으로는 갈린다 —
    명령이 벌어졌으면 의도적으로 놓은 것이고, 명령은 닫힌 채인데 물체가 없으면
    놓친 것이다. 여기서는 명령만 보므로 '놓았다'만 판정한다.
    """
    hold = (grip >= GRIP_HOLD_MIN) & (grip <= GRIP_HOLD_MAX)
    span = _longest_run(hold)
    # "처음" 구간이 아니라 "가장 긴" 구간을 파지로 본다. 그리퍼가 벌어지고 닫히는
    # 램프가 파지 범위를 스쳐 지나가는데, 첫 구간을 쓰면 그 램프를 파지로 오인하고
    # 곧바로 이어지는 개방을 "놓기"로 잘못 읽는다 (실데이터에서 확인).
    if span is None or span[1] - span[0] < HOLD_FRAMES:
        return {"hold_start": None, "release": None, "hold_level": None}

    start, end = span
    level = float(np.median(grip[start:end]))
    opened = np.nonzero(grip[end:] > level + RELEASE_JUMP)[0]
    release = int(end + opened[0]) if len(opened) else None
    return {"hold_start": int(start), "release": release, "hold_level": level}


def _longest_run(mask: np.ndarray) -> tuple[int, int] | None:
    """가장 긴 True 구간 [start, end). 없으면 None."""
    best = cur = None
    for i, v in enumerate(mask):
        if v:
            cur = (cur[0], i + 1) if cur else (i, i + 1)
            if best is None or cur[1] - cur[0] > best[1] - best[0]:
                best = cur
        else:
            cur = None
    return best


# ═══════════════════════════════════════════════════════════════════
# 에피소드 1개 채점
# ═══════════════════════════════════════════════════════════════════
def _load_ref_and_judge(ep_dir: Path, exclude) -> tuple[np.ndarray, np.ndarray, str]:
    """reference/judge 프레임 한 장씩을 구한다. -> (ref_frame, judge_frame, source)

    erase_run.py가 만든 reference_frame.png/judge_frame.png가 있으면 그걸 그대로
    쓴다(실시간 게이트와 같은 사진 — source="park_frame"). 없으면(과거 데이터·
    erase_run.py 없이 만든 롤아웃) 영상의 첫/마지막 프레임으로 대신한다
    (source="video_fallback").
    """
    ref = M.imread(ep_dir / "reference_frame.png")
    judge = M.imread(ep_dir / "judge_frame.png")
    if ref is not None and judge is not None:
        return ref, judge, "park_frame"

    video = M.top_video(ep_dir)
    first, last = EC.first_last_frames(video)
    return (ref if ref is not None else first), (judge if judge is not None else last), "video_fallback"


def score_episode(ep_dir: Path, target: str | None, board, dark_ratio: float, fps: float,
                   exclude=None) -> dict:
    target = target or M.task_of(ep_dir)

    row: dict = {"episode": ep_dir.name, "path": str(ep_dir), "target": target}

    try:
        ref_frame, judge_frame, frame_source = _load_ref_and_judge(ep_dir, exclude)
    except Exception as exc:
        row.update(valid=False, note=f"프레임 로드 실패: {type(exc).__name__}: {exc}")
        return row
    row["frame_source"] = frame_source

    checker = EC.EraseChecker(board, dark_ratio)
    try:
        ref = checker.set_reference(ref_frame, exclude=exclude)
    except RuntimeError as exc:
        row.update(valid=False, note=str(exc))
        return row
    row["shapes_found"] = "|".join(sorted({k.split("#")[0] for k, _ in ref["shapes"]}))

    # 방해 도형(타겟이 아닌 도형)이 기준 프레임에 몇 개나, **어디에** 있었나.
    #
    # 개수를 안 적으면 선택성을 읽을 수 없다 — check()의 max_distractor_erased는 방해
    # 도형이 아예 없을 때도 0.0을 내므로 "있었는데 안 건드렸다"와 "처음부터 없었다"가
    # 구분되지 않는다. 증강 전 315 데이터셋은 보드에 도형이 하나뿐이라 전부 후자다.
    #
    # 위치를 안 적으면 선택성의 의미를 못 읽는다 — "프롬프트를 이해했다"와 "항상 왼쪽
    # 것을 지운다"가 똑같은 숫자로 나온다. 타겟이 우연히 계속 한쪽에 있었으면 위치
    # 편향이 선택성 100%로 보이는데, 라벨만 남기면 사후에 캐낼 방법이 없다.
    # detect_shapes는 원래부터 bbox를 주고 있었고 여기서 버리고 있었을 뿐이다.
    board_mid_x = board[0] + board[2] / 2.0

    def _center(box):
        x, y, w, h = box
        return int(x + w / 2), int(y + h / 2)

    def _side(box):  # 보드 가로 중앙 기준 좌/우
        return "L" if _center(box)[0] < board_mid_x else "R"

    # 검출 순서를 유지한다 — 라벨/위치/좌우 세 컬럼이 같은 순서로 짝지어야 한다.
    tgt_boxes = [b for k, b in ref["shapes"] if k.split("#")[0] == target]
    dis = [(k.split("#")[0], b) for k, b in ref["shapes"] if k.split("#")[0] != target]

    row["target_pos"] = "|".join("{},{}".format(*_center(b)) for b in tgt_boxes)
    row["target_side"] = "|".join(_side(b) for b in tgt_boxes)
    row["distractors"] = "|".join(lbl for lbl, _ in dis)
    row["distractor_pos"] = "|".join("{},{}".format(*_center(b)) for _, b in dis)
    row["distractor_side"] = "|".join(_side(b) for _, b in dis)
    row["n_distractors"] = len(dis)
    distractors = [lbl for lbl, _ in dis]

    result = checker.check(judge_frame, target)
    if not result["target_found"]:
        row.update(valid=False, note=result["reason"])
        return row

    grip = read_gripper(ep_dir)
    ph = gripper_phases(grip) if grip is not None else {"hold_start": None, "release": None}

    row.update(
        valid=True,
        note="",
        success=result["success"],
        erased_target=result["target_erased"],
        # 방해 도형 중 가장 많이 지워진 비율. success는 이미 이 값이 MAX_DISTRACTOR
        # 이하일 것을 요구한다(erase_check.check) — 여기서는 그 근거를 눈에 보이게 남긴다.
        erased_distractor=result["max_distractor_erased"] if distractors else None,
        selective_ok=(result["max_distractor_erased"] <= MAX_DISTRACTOR) if distractors else None,
        remaining_frac=result["remaining_frac"],
        grasp_s=round(ph["hold_start"] / fps, 2) if ph["hold_start"] is not None else None,
        release_s=round(ph["release"] / fps, 2) if ph["release"] is not None else None,
        failure_mode="",  # 사람이 GUI에서 직접 태그(자동 추측 폐지)
    )
    row["termination"] = termination_reason(row, ph, grip)
    return row


def termination_reason(row: dict, ph: dict, grip) -> str:
    """왜 끝났나. 성공 판정과 분리한다 — 로봇이 '끝났다'고 놓은 것과 실제로
    다 지운 것은 다른 사건이고, 섞으면 착각 종료가 정상 종료로 기록된다."""
    if grip is None:
        return "unknown(no_log)"
    if ph.get("release") is not None:
        return "release"  # 로봇이 스스로 지우개를 놓음
    return "cutoff"  # max-steps까지 다 씀 (스스로 안 끝냄)


def read_gripper(ep_dir: Path) -> np.ndarray | None:
    """action[:,6] = 그리퍼 명령. parquet이 없거나 pyarrow가 없으면 None."""
    hits = sorted(ep_dir.glob("data/**/*.parquet"))
    if not hits:
        return None
    # 조용히 None을 내면 안 된다. 그리퍼를 못 읽으면 종료 사유가 전부 unknown(no_log)이
    # 되고 premature_release 판정이 통째로 죽는데, 예전엔 그 이유가 어디에도 안 남아서
    # "환경을 잘못 잡았다"와 "로그가 원래 없다"를 구분할 수 없었다. 실제로 pyarrow 없는
    # 인터프리터로 30개를 채점하고 나서야 알아챘다.
    if _pq is None:
        _warn_once("pyarrow가 없어 그리퍼 로그를 못 읽는다 — 종료 사유 판정이 비활성된다. "
                   "프로젝트 환경(conda activate ugrp)에서 돌릴 것.")
        return None
    try:
        table = _pq.read_table(hits[0], columns=["action"])
        return np.asarray(table["action"].to_pylist(), dtype=np.float32)[:, 6]
    except Exception as e:
        _warn_once(f"그리퍼 로그 읽기 실패({type(e).__name__}: {e}) — 종료 사유 판정 비활성")
        return None


_WARNED: set[str] = set()


def _warn_once(msg: str) -> None:
    """에피소드마다 같은 경고를 수십 번 쏟지 않게 한 번만 낸다."""
    if msg not in _WARNED:
        _WARNED.add(msg)
        print(f"[경고] {msg}", file=sys.stderr)


# ═══════════════════════════════════════════════════════════════════
# 집계
# ═══════════════════════════════════════════════════════════════════
def default_out_dir(model: str, condition: str) -> Path:
    """결과를 task별 generated output 아래에 날짜·조건별로 저장한다."""
    stamp = datetime.now().strftime("%m%d")
    name = "_".join(x for x in (stamp, model, condition) if x)
    return _repo_root() / "outputs" / "evaluation" / "erase_shape" / name


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists() or (parent / "scripts").is_dir():
            return parent
    return Path.cwd()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """성공률의 Wilson score 95% 신뢰구간. n이 작을 때 정규근사보다 정직하다.

    시행이 10~50회뿐이면 성공률 차이는 대부분 구간이 겹친다. 그래서 주장은
    연속 지표(erased_target)로 하고, 성공률은 구간과 함께 보고만 한다.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


def mean_sem(values: list[float]) -> dict:
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if len(v) == 0:
        return {"n": 0, "mean": None, "sem": None}
    sem = float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0
    return {"n": len(v), "mean": round(float(v.mean()), 4), "sem": round(sem, 4)}


def summarize_rows(rows: list[dict]) -> dict:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        if r.get("valid"):
            groups.setdefault((r.get("model", ""), r.get("condition", "")), []).append(r)

    out = {}
    for (model, cond), rs in sorted(groups.items()):
        n = len(rs)
        k = sum(1 for r in rs if r["success"])
        modes: dict[str, int] = {}
        for r in rs:
            if r["failure_mode"]:
                modes[r["failure_mode"]] = modes.get(r["failure_mode"], 0) + 1
        # 방해 도형이 있던 시행만 따로 센다. 없던 시행을 섞으면 선택성이 공짜로 100%가 된다.
        sel = [r for r in rs if r.get("n_distractors")]
        n_sel = len(sel)
        k_sel = sum(1 for r in sel if r.get("selective_ok"))
        out[f"{model}/{cond}"] = {
            "n_episodes": n,
            "success_rate": round(k / n, 4),
            "success_ci95": wilson(k, n),
            "erased_target": mean_sem([r["erased_target"] for r in rs]),
            "n_with_distractor": n_sel,
            "erased_distractor": mean_sem([r.get("erased_distractor") for r in sel]),
            "selective_rate": round(k_sel / n_sel, 4) if n_sel else None,
            "selective_ci95": wilson(k_sel, n_sel) if n_sel else None,
            "failure_modes": dict(sorted(modes.items(), key=lambda kv: -kv[1])),
            "termination": _counts([r["termination"] for r in rs]),
        }
    return out


def _counts(values) -> dict:
    out: dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def render_markdown(summary: dict, rows: list[dict]) -> str:
    n_bad = sum(1 for r in rows if not r.get("valid"))
    lines = [
        "# 지우기 롤아웃 평가 요약",
        "",
        f"- 채점 에피소드 {len(rows) - n_bad}개" + (f" (채점 실패 {n_bad}개)" if n_bad else ""),
        f"- 성공 기준: target erased ≥ {SUCCESS_ERASED} **이면서** 방해 도형 erased ≤ {MAX_DISTRACTOR}",
        "- 선택성 열은 방해 도형이 실제로 보드에 있던 시행에서만 잰다. 증강 전 단일 도형 "
        "데이터는 잴 대상이 없어 `—`로 나온다.",
        "- 성공률 구간은 Wilson 95%. 시행이 적으면 구간이 넓으니 연속 지표를 함께 볼 것.",
        "- 실패 유형은 자동 추측 없이 GUI에서 사람이 직접 태그한 값이다.",
        "",
        "## 조건별",
        "",
        "| 조건 | n | 성공률 (95% CI) | erased(target) | 방해도형 n | erased(방해) | 선택성 (95% CI) |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for key, s in summary.items():
        lo, hi = s["success_ci95"]
        if s.get("n_with_distractor"):
            slo, shi = s["selective_ci95"]
            sel_txt = f"{s['selective_rate']:.2f} ({slo:.2f}–{shi:.2f})"
        else:
            sel_txt = "—"
        lines.append(
            f"| {key} | {s['n_episodes']} | {s['success_rate']:.2f} ({lo:.2f}–{hi:.2f}) "
            f"| {_fmt(s['erased_target'])} | {s.get('n_with_distractor', 0)} "
            f"| {_fmt(s['erased_distractor'])} | {sel_txt} |"
        )
    lines += ["", "## 실패 유형 (사람 태그)", ""]
    for key, s in summary.items():
        modes = ", ".join(f"{k} {v}" for k, v in s["failure_modes"].items()) or "없음"
        term = ", ".join(f"{k} {v}" for k, v in s["termination"].items())
        lines += [f"- **{key}** — 실패: {modes} / 종료: {term}"]
    if n_bad:
        lines += ["", "## 채점 실패 (사람이 확인 필요)", ""]
        lines += [f"- `{r['episode']}` — {r.get('note', '')}" for r in rows if not r.get("valid")]
    return "\n".join(lines) + "\n"


def _fmt(ms: dict) -> str:
    return "—" if ms["mean"] is None else f"{ms['mean']:.3g} ± {ms['sem']:.2g}"


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════
FIELDS = [
    "episode", "model", "condition", "trial", "target", "valid", "scored_by", "note",
    "success", "erased_target", "erased_distractor", "selective_ok", "remaining_frac",
    "grasp_s", "release_s", "termination", "failure_mode",
    "frame_source", "shapes_found",
    "target_side", "target_pos", "n_distractors", "distractors",
    "distractor_side", "distractor_pos", "path",
]


def append_rows(csv_path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    """episodes.csv에 이어쓴다. 기존 헤더에 없는 컬럼이 생겼으면 파일을 먼저 확장한다.

    컬럼을 추가한 뒤 옛 파일에 그냥 append하면 18칸 헤더 아래에 22칸 행이 붙어 값이
    통째로 한 칸씩 밀린다(2026-08-25 선택성 컬럼을 넣으면서 실제로 재현했다 — path 칸에
    빈 문자열이 들어갔다). 세션을 며칠에 걸쳐 이어서 채점하는 일이 흔하고, 밀린 CSV는
    나중에 집계할 때까지 아무도 못 알아채므로 조용히 넘어가면 안 된다.

    기존 컬럼은 하나도 버리지 않는다 — 2026-08-20 재설계 이전 40컬럼 파일처럼 지금
    FIELDS에 없는 컬럼이 남아 있고, 그건 그 시점의 판정 근거라 지우면 재현이 끊긴다.
    그래서 '덮어쓰기'가 아니라 '기존 헤더 + 빠진 컬럼'의 합집합으로 넓힌다.
    """
    fields = list(fields or FIELDS)  # reach_eval_ui 등 다른 컬럼 집합도 이 헬퍼를 쓴다
    rows = list(rows)
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        return

    with csv_path.open(encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        header = list(rd.fieldnames or [])
        old = list(rd)

    missing = [c for c in fields if c not in header]
    if not missing:
        # 헤더가 이미 충분하다(순서가 FIELDS와 달라도 헤더 순서를 그대로 따른다).
        with csv_path.open("a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=header, extrasaction="ignore").writerows(rows)
        return

    merged = header + missing
    tmp = csv_path.with_name(csv_path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=merged, extrasaction="ignore")
        w.writeheader()
        w.writerows(old)      # 옛 행은 새 컬럼이 빈칸으로 남는다 — 잴 수 없었던 값이라 맞다
        w.writerows(rows)
    os.replace(tmp, csv_path)  # 원자적 교체 — 도중에 죽어도 원본이 통째로 남는다
    print(f"[CSV] 컬럼 추가({', '.join(missing)}) — {csv_path.name}를 {len(merged)}컬럼으로 확장")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("episodes", nargs="*", help="에피소드 디렉터리 (glob 가능)")
    p.add_argument("--model", default="", help="비교 축 1 (smolvla / pi0 …)")
    p.add_argument("--condition", default="", help="비교 축 2 (async / sync …)")
    p.add_argument("--target", default=None, help="지울 도형. 생략하면 폴더명에서 추론")
    p.add_argument("--board", type=int, nargs=4, default=M.DEFAULT_BOARD, metavar=("X", "Y", "W", "H"))
    p.add_argument(
        "--exclude", action="append", default=None,
        type=lambda s: tuple(int(v) for v in s.split(",")),
        metavar="X,Y,W,H",
        help="이 전역좌표 사각형은 도형 검출에서 뺀다 (ink_metric.py --exclude와 동일, "
             f"여러 번 줄 수 있음). 기본값은 보드 테이프 자국 자리({M.DEFAULT_EXCLUDE[0]}) — "
             "떼어냈으면 --exclude 0,0,0,0으로 비울 것",
    )
    p.add_argument("--dark-ratio", type=float, default=0.72)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="생략하면 화면 출력만. --save를 주면 outputs/evaluation/erase_shape/<MMDD>_<model>_<condition>")
    p.add_argument("--save", action="store_true", help="기본 폴더에 저장")
    p.add_argument("--selftest", action="store_true", help="합성 이미지로 채점 로직 검증")
    p.add_argument("--summarize", type=Path, default=None,
                   help="이미 있는 episodes.csv로 요약만 다시 만든다 (손 채점 시트도 그대로 됨)")
    args = p.parse_args(argv)
    args.exclude = M.resolve_exclude(args.exclude)

    if args.selftest:
        _selftest()
        return 0

    if args.summarize:
        rows = _read_csv(args.summarize)
        summary = summarize_rows(rows)
        text = render_markdown(summary, rows)
        print(text)
        out = args.out_dir or args.summarize.parent
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.md").write_text(text, encoding="utf-8")
        (out / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"→ {out}/summary.md, summary.json")
        return 0

    paths = sorted({Path(q) for pat in args.episodes for q in glob.glob(pat)})
    paths = [q for q in paths if q.is_dir()]
    if not paths:
        p.error("채점할 에피소드를 못 찾음")

    rows = []
    for i, ep in enumerate(paths, 1):
        try:
            row = score_episode(ep, args.target, tuple(args.board), args.dark_ratio, args.fps,
                                 exclude=args.exclude)
        except Exception as exc:  # 한 에피소드가 깨져도 나머지는 채점한다
            row = {"episode": ep.name, "path": str(ep), "valid": False, "note": f"{type(exc).__name__}: {exc}"}
        row.update(model=args.model, condition=args.condition, trial=i, scored_by="auto")
        rows.append(row)
        flag = "OK " if row.get("valid") else "SKIP"
        print(
            f"[{flag}] {ep.name}: "
            + (
                f"erased={row['erased_target']:.3f} "
                + (f"방해={row['erased_distractor']:.3f} " if row.get("n_distractors") else "")
                + f"success={row['success']} "
                f"{row['termination']} ({row['frame_source']})"
                if row.get("valid")
                else row.get("note", "")
            )
        )

    summary = summarize_rows(rows)
    print()
    print(render_markdown(summary, rows))

    if args.save and args.out_dir is None:
        args.out_dir = default_out_dir(args.model, args.condition)
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = args.out_dir / "episodes.csv"
        append_rows(csv_path, rows)
        # 집계는 CSV 전체를 다시 읽어 만든다 — 여러 번 나눠 돌려도 항상 최신 전체 기준.
        all_rows = _read_csv(csv_path)
        summary = summarize_rows(all_rows)
        (args.out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (args.out_dir / "summary.md").write_text(render_markdown(summary, all_rows), encoding="utf-8")
        print(f"→ {csv_path} ({len(all_rows)}행), summary.json, summary.md")
    return 0


def _read_csv(path: Path) -> list[dict]:
    numeric = {"erased_target", "erased_distractor", "remaining_frac", "grasp_s", "release_s"}
    truthy = ("True", "true", "1")
    rows = []
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            r["valid"] = r["valid"] in truthy
            r["success"] = r["success"] in truthy
            # selective_ok는 3상태다: True / False / 빈칸(방해 도형이 없어 잴 수 없었음).
            # 빈칸을 False로 접으면 단일 도형 시행이 전부 "선택 실패"로 집계된다.
            sv = r.get("selective_ok", "")
            r["selective_ok"] = None if sv in ("", "None") else sv in truthy
            nd = r.get("n_distractors", "")
            r["n_distractors"] = 0 if nd in ("", "None") else int(nd)
            for k in numeric:
                v = r.get(k, "")
                r[k] = None if v in ("", "None") else float(v)
            rows.append(r)
    return rows


# ═══════════════════════════════════════════════════════════════════
# 자기검증 — 하드웨어·실데이터 없이 채점 로직을 확인한다
# ═══════════════════════════════════════════════════════════════════
def _selftest():
    import tempfile

    import cv2

    tmp = Path(tempfile.mkdtemp())
    board_bgr, ink_bgr = (240, 240, 238), (60, 60, 55)

    def scene(erase_circle=False, rect=False, erase_rect=False):
        f = np.full((400, 400, 3), board_bgr, np.uint8)
        if not erase_circle:
            cv2.circle(f, (100, 100), 40, ink_bgr, 3)
        if rect and not erase_rect:
            cv2.rectangle(f, (250, 250), (350, 350), ink_bgr, 3)
        return f

    ep = tmp / "ep1"
    ep.mkdir()
    cv2.imwrite(str(ep / "reference_frame.png"), scene())
    cv2.imwrite(str(ep / "judge_frame.png"), scene(erase_circle=True))

    board = (0, 0, 400, 400)
    row = score_episode(ep, "circle", board, 0.72, 30.0)
    assert row["valid"], row
    assert row["frame_source"] == "park_frame", row
    assert row["success"] and row["erased_target"] > 0.9, row

    # 못 지운 경우
    ep2 = tmp / "ep2"
    ep2.mkdir()
    cv2.imwrite(str(ep2 / "reference_frame.png"), scene())
    cv2.imwrite(str(ep2 / "judge_frame.png"), scene())
    row2 = score_episode(ep2, "circle", board, 0.72, 30.0)
    assert row2["valid"] and not row2["success"] and row2["erased_target"] < 0.1, row2

    # ── 선택성: 보드에 방해 도형이 같이 있을 때 ────────────────────────
    # 단일 도형 시행은 잴 대상이 없다. 0.0이 아니라 None이어야 한다 — 0.0으로 적으면
    # 집계에서 "방해 도형을 안 건드렸다"로 세어져 선택성이 공짜로 100%가 된다.
    assert row["n_distractors"] == 0 and row["erased_distractor"] is None, row
    assert row["selective_ok"] is None, row

    def _two_shape_ep(name, **kw):
        d = tmp / name
        d.mkdir()
        cv2.imwrite(str(d / "reference_frame.png"), scene(rect=True))
        cv2.imwrite(str(d / "judge_frame.png"), scene(rect=True, **kw))
        return score_episode(d, "circle", board, 0.72, 30.0)

    # 타겟만 지움 -> 성공 + 선택 OK
    r_ok = _two_shape_ep("ep_sel_ok", erase_circle=True)
    assert r_ok["valid"] and r_ok["n_distractors"] == 1, r_ok
    assert r_ok["distractors"] == "rectangle", r_ok
    # 배치 — 원은 (100,100) 왼쪽, 사각형은 (250..350) 오른쪽. 400폭 보드의 중앙은 200.
    assert r_ok["target_side"] == "L" and r_ok["distractor_side"] == "R", r_ok
    assert r_ok["target_pos"] == "100,100", r_ok
    assert r_ok["distractor_pos"] == "300,300", r_ok
    # 단일 도형 시행은 방해 쪽이 빈칸이지만 타겟 위치는 남아야 한다
    assert row["target_side"] == "L" and row["distractor_side"] == "", row
    assert r_ok["erased_distractor"] < MAX_DISTRACTOR, r_ok
    assert r_ok["selective_ok"] and r_ok["success"], r_ok

    # 둘 다 지움 -> 타겟은 다 지웠지만 방해 도형까지 지웠으므로 실패
    r_ng = _two_shape_ep("ep_sel_ng", erase_circle=True, erase_rect=True)
    assert r_ng["valid"] and r_ng["erased_target"] > 0.9, r_ng
    assert r_ng["erased_distractor"] > MAX_DISTRACTOR, r_ng
    assert r_ng["selective_ok"] is False and r_ng["success"] is False, r_ng

    # 집계: 방해 도형이 없던 시행은 선택성 분모에서 빠진다
    agg = summarize_rows([
        dict(r, valid=True, model="m", condition="c", failure_mode="", termination="release")
        for r in (row, r_ok, r_ng)
    ])["m/c"]
    assert agg["n_episodes"] == 3 and agg["n_with_distractor"] == 2, agg
    assert agg["selective_rate"] == 0.5, agg

    # reference_frame.png/judge_frame.png가 없으면 영상 첫/마지막 프레임으로 대체된다
    ep3 = tmp / "ep3" / "erase_the_circle_x"
    ep3.mkdir(parents=True)
    video_dir = ep3 / "videos" / "observation.images.top" / "chunk-000"
    video_dir.mkdir(parents=True)
    video_path = video_dir / "episode_000000.mp4"
    w = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (400, 400))
    for i in range(10):
        w.write(scene(erase_circle=(i >= 5)))
    w.release()
    row3 = score_episode(ep3, "circle", board, 0.72, 30.0)
    assert row3["valid"] and row3["frame_source"] == "video_fallback", row3
    assert row3["success"], row3

    # 그리퍼 위상
    g = np.concatenate([np.zeros(30), np.full(30, 90.0), np.full(60, 20.0), np.full(30, 70.0)])
    ph = gripper_phases(g)
    assert ph["hold_start"] == 60, ph
    assert ph["release"] == 120, ph
    assert gripper_phases(np.zeros(100))["hold_start"] is None
    # 개폐 램프가 파지 범위를 스쳐도 그걸 파지로 오인하면 안 된다 (실데이터 회귀).
    ramp = np.concatenate(
        [np.zeros(20), np.linspace(0, 90, 40), np.full(20, 90.0), np.full(80, 20.0), np.full(20, 70.0)]
    )
    ph = gripper_phases(ramp)
    assert ph["hold_start"] == 80, ph  # 램프(20~35 구간)가 아니라 진짜 파지 구간
    assert ph["release"] == 160, ph

    # 종료 사유
    assert termination_reason({}, {"release": 50}, np.zeros(10)) == "release"
    assert termination_reason({}, {"release": None}, np.zeros(10)) == "cutoff"
    assert termination_reason({}, {"release": None}, None) == "unknown(no_log)"

    # 이어쓰기 호환 — 컬럼이 늘기 전 헤더로 만들어진 파일에 이어 붙일 수 있어야 한다
    csv_path = tmp / "episodes.csv"
    legacy = ["episode", "target", "valid", "success", "erased_target", "path"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=legacy)
        w.writeheader()
        w.writerow({"episode": "old1", "target": "circle", "valid": "True",
                    "success": "True", "erased_target": "0.95", "path": "/tmp/old1"})
    append_rows(csv_path, [dict(r_ng, episode="new1", path="/tmp/new1")])
    with csv_path.open(encoding="utf-8", newline="") as fh:
        got = list(csv.DictReader(fh))
    # 옛 행의 값이 밀리지 않았고, 옛 컬럼도 그대로 남아 있어야 한다
    assert got[0]["path"] == "/tmp/old1" and got[0]["erased_target"] == "0.95", got[0]
    assert got[0]["selective_ok"] == "" and got[0]["n_distractors"] == "", got[0]
    assert got[1]["path"] == "/tmp/new1" and got[1]["selective_ok"] == "False", got[1]
    assert all(c in got[1] for c in FIELDS), sorted(set(FIELDS) - set(got[1]))
    # 두 번째 이어쓰기는 확장 없이 그대로 붙는다
    append_rows(csv_path, [dict(r_ok, episode="new2", path="/tmp/new2")])
    with csv_path.open(encoding="utf-8", newline="") as fh:
        got2 = list(csv.DictReader(fh))
    assert len(got2) == 3 and got2[2]["path"] == "/tmp/new2", got2[2]
    assert got2[2]["selective_ok"] == "True", got2[2]

    # Wilson 구간 — 10회 중 8회 성공이면 구간이 넓어야 한다(주장 못 함을 보이는 근거)
    lo, hi = wilson(8, 10)
    assert lo < 0.55 and hi > 0.93, (lo, hi)
    assert wilson(0, 0) == (0.0, 0.0)

    print("selftest OK — 잉크 판정(park_frame/video_fallback) / 선택성 / CSV 이어쓰기 호환 / "
          "그리퍼 위상 / 종료사유 / CI 전부 통과")


if __name__ == "__main__":
    sys.exit(main())
