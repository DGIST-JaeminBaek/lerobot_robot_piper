#!/usr/bin/env python3
"""erase_eval.py — 지우기 롤아웃의 자동 채점기 (사람 개입 없이 CSV/JSON/MD 산출).

측정은 새로 만들지 않는다. ink_metric.py가 이미 하는 일(도형 검출, 가림 처리,
잉크 잔량 추적)을 그대로 쓰고, 그 위에 **평가 프로토콜**만 얹는다:

  ink_metric.track/summarize/aggregate  -> 판정용 잉크 잔량 (가려진 프레임 폐기)
  ink_metric.dense_progress             -> 진행도 곡선 (접촉 중에도 관측)
  data/*.parquet 의 action[:,6]         -> 그리퍼 개폐 = 파지/놓기 시점
  ------------------------------------------------------------------
  erase_eval.py                         -> 단계 점수, 종료 사유, 실패 유형,
                                           t@50/t@90, 정체·회복, 조건별 집계

근거가 되는 선행연구와 각 수치를 그렇게 정한 이유는 docs/policy/evaluation_protocol.md
에 전부 적어뒀다. 이 파일은 그 문서의 실행 가능한 형태다.

사용:
  # 하드웨어 없이 자기검증 (합성 영상)
  python erase_eval.py --selftest

  # 기존 시연 데이터로 채점기 자체를 검증 (지움 성공이어야 함)
  python erase_eval.py '0802/0804/erase_the_rectangle_*' --model demo --condition teleop

  # 실제 롤아웃 채점
  python erase_eval.py 'rollouts/smolvla_async/*' --model smolvla --condition async \\
      --out-dir outputs/eval/0818

산출물(--out-dir):
  episodes.csv   에피소드 1행씩. 이어붙이므로 여러 번 돌려도 누적된다.
  summary.json   조건별 집계 (신뢰구간 포함). AI/스크립트 입력용.
  summary.md     사람이 읽는 요약. 그대로 보고서에 붙일 수 있다.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ink_metric as M  # noqa: E402  (cv2 의존은 여기 한 곳으로 모은다)


# ═══════════════════════════════════════════════════════════════════
# 채점 상수 — 값의 근거는 docs/policy/evaluation_protocol.md §3
# ═══════════════════════════════════════════════════════════════════
SUCCESS_ERASED = M.SUCCESS_ERASED  # 0.90 — target 완료 임계
MAX_DISTRACTOR = M.MAX_DISTRACTOR  # 0.10 — distractor 허용 상한(노이즈 플로어 3배)
START_ERASED = 0.15  # "지우기를 개시했다"로 볼 최소 진행
CONTACT_FRAMES = 15  # 0.5초(30fps) 연속 가림 = 접촉 확립
STALL_FRAMES = 90  # 3초간 진행 없음 = 정체.
# 실측으로 정했다(0804 시연 30개). 1초면 에피소드당 9.6건이 잡히는데, 그건 왕복
# 스트로크 사이의 정상적인 되돌림까지 세는 것이라 신호가 아니다. 진행도에 단조증가
# envelope을 씌우고 3초 창을 쓰면 사람 시연 기준 2~4건으로 내려간다 —
# 정책 값은 이 baseline과 비교해서 읽어야 한다(절대값 아님). docs 참고.
STALL_EPS = 0.02  # 이 미만의 진행은 노이즈로 본다 (정규화 진행도 기준)

# 단계 배점 (합 10). "여기서 실패하면 뒤가 전부 불가능한가"로 가중.
STAGE_POINTS = {"approach": 1, "contact": 2, "start": 2, "complete": 3, "selective": 2}

# 그리퍼(action[:,6], 0~100). 실측: 대기 ~0, 벌림 60~100, 지우개 파지 ~20.
GRIP_OPEN = 45.0  # 이 이상이면 벌린 상태
GRIP_HOLD_MAX = 35.0  # 이 이하로 닫혀 있으면 뭔가 쥔 상태로 본다
GRIP_HOLD_MIN = 8.0  # 완전히 닫힌(빈손) 것과 구분
HOLD_FRAMES = 15  # 0.5초 이상 유지돼야 파지로 인정
RELEASE_JUMP = 15.0  # 파지 수준에서 이만큼 벌어지면 놓은 것


# ═══════════════════════════════════════════════════════════════════
# 그리퍼 위상 — 파지 / 놓기
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


def _first_run(mask: np.ndarray, n: int) -> int | None:
    """mask가 n프레임 연속 True가 되는 첫 시작 인덱스."""
    if len(mask) < n:
        return None
    run = 0
    for i, v in enumerate(mask):
        run = run + 1 if v else 0
        if run >= n:
            return i - n + 1
    return None


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
# 진행도 곡선 → 시간 지표 / 정체·회복
# ═══════════════════════════════════════════════════════════════════
def target_envelope(per_shape: dict, target: str) -> np.ndarray | None:
    """target 도형들의 단조감소 envelope을 합쳐 하나로 만든다."""
    envs = [v["env"] for k, v in per_shape.items() if k.split("#")[0] == target and "env" in v]
    if not envs:
        return None
    n = min(len(e) for e in envs)
    return np.sum([e[:n] for e in envs], axis=0)


def frames_to(env: np.ndarray, fraction: float) -> int | None:
    """잉크가 초기값 대비 fraction만큼 줄어든 첫 프레임. (0.9 = 90% 지워짐)"""
    if env is None or len(env) == 0 or env[0] <= 0:
        return None
    hit = np.nonzero(env <= env[0] * (1.0 - fraction))[0]
    return int(hit[0]) if len(hit) else None


def reach(progress: np.ndarray, valid: np.ndarray, level: float, window: int = 5) -> int | None:
    """진행도가 level에 처음 도달한 프레임. 잡음에 안 흔들리게 window만큼 유지되어야 한다."""
    idx = np.nonzero(valid & (progress >= level))[0]
    for i in idx:
        seg = progress[i : i + window][valid[i : i + window]]
        if len(seg) and float(seg.min()) >= level - STALL_EPS:
            return int(i)
    return None


def progress_curve(dense: dict, target: str) -> tuple[np.ndarray, np.ndarray] | None:
    """dense_progress를 target 기준 0~1 진행도로 바꾼다. -> (progress, valid)"""
    parts = [(d, v) for k, (d, v) in dense.items() if k.split("#")[0] == target]
    if not parts:
        return None
    n = min(len(d) for d, _ in parts)
    dens = np.sum([d[:n] for d, _ in parts], axis=0)
    valid = np.all([v[:n] for _, v in parts], axis=0)
    if not valid.any():
        return None
    base = float(dens[valid][0])
    if base <= 0:
        return None
    return np.clip(1.0 - dens / base, 0.0, 1.0), valid


def stall_and_recovery(progress: np.ndarray, valid: np.ndarray) -> dict:
    """정체 구간과 그로부터의 회복을 센다.

    SO-101 논문(Recovery Rate = 성공 회복 / 회복 기회)의 정의를 우리 태스크에서
    자동 계산 가능하게 조작적으로 옮긴 것. 원 논문은 사람이 영상을 보고 회복
    기회를 세지만, 지우기는 "진행도가 멈췄다"가 곧 실패 신호라 로그로 잡힌다.

    회복 기회 = 진행도가 STALL_FRAMES 동안 STALL_EPS 미만으로 움직인 구간
    회복 성공 = 그 구간 이후 다시 STALL_EPS 이상 진행이 재개됨

    창은 관측 프레임이 아니라 **실제 경과 프레임** 기준이다. 관측이 없는(가려진)
    구간은 직전 값으로 채운다 — 안 그러면 팔이 오래 가리고 있던 시간이 압축돼
    "3초"가 실제로는 6초가 된다.
    또 진행도에 단조증가 envelope을 씌운다. 지우기는 잉크를 되돌릴 수 없는데
    dense 밀도는 분모(가시 픽셀)가 흔들려 오르내리므로, 그대로 두면 그 출렁임이
    전부 가짜 정체로 잡힌다(실측: 에피소드당 9.6건 → 3건).
    """
    if not valid.any() or len(progress) < STALL_FRAMES + 1:
        return {"stall_events": 0, "recovered": 0, "recovery_rate": None, "ended_stalled": False}

    p = progress.copy()
    last = p[np.argmax(valid)]
    for i in range(len(p)):
        if valid[i]:
            last = p[i]
        else:
            p[i] = last
    p = np.maximum.accumulate(p)

    stalls, i = [], 0
    while i + STALL_FRAMES < len(p):
        if p[i + STALL_FRAMES] - p[i] < STALL_EPS:
            j = i
            while j + STALL_FRAMES < len(p) and p[j + STALL_FRAMES] - p[j] < STALL_EPS:
                j += 1
            stalls.append((i, j + STALL_FRAMES))
            i = j + 1
        else:
            i += 1

    # 마지막 정체가 에피소드 끝까지 이어지면 그건 "종료"지 "회복 기회"가 아니다.
    if stalls and stalls[-1][1] >= len(p) - 1:
        ended_stalled = True
        stalls = stalls[:-1]
    else:
        ended_stalled = False

    recovered = sum(1 for _, end in stalls if end < len(p) - 1 and p[-1] - p[end] >= STALL_EPS)
    n = len(stalls)
    return {
        "stall_events": n,
        "recovered": recovered,
        "recovery_rate": round(recovered / n, 3) if n else None,
        "ended_stalled": ended_stalled,
    }


# ═══════════════════════════════════════════════════════════════════
# 에피소드 1개 채점
# ═══════════════════════════════════════════════════════════════════
def score_episode(ep_dir: Path, target: str | None, board, dark_ratio: float, fps: float,
                   exclude=None) -> dict:
    video = M.top_video(ep_dir)
    target = target or M.task_of(ep_dir)

    shapes, ink, occ = M.track(video, board, dark_ratio, exclude=exclude)
    per_shape = M.summarize(ink, occ)
    labels = sorted({k.split("#")[0] for k, _ in shapes})

    row: dict = {
        "episode": ep_dir.name,
        "path": str(ep_dir),
        "target": target,
        "shapes_found": "|".join(labels),
        "n_frames": len(next(iter(ink.values()))) if ink else 0,
    }

    tgt = M.aggregate(per_shape, target)
    if tgt is None:
        row.update(valid=False, note=f"target '{target}' 미검출 (검출: {labels})")
        return row

    # distractor = target이 아닌 모든 도형. 가장 많이 건드린 것을 대표값으로.
    d_erased = [
        M.aggregate(per_shape, lb)["erased_frac"]
        for lb in labels
        if lb != target and M.aggregate(per_shape, lb) is not None
    ]
    distractor = max(d_erased) if d_erased else 0.0

    env = target_envelope(per_shape, target)
    dense = M.dense_progress(video, board, dark_ratio, exclude=exclude)
    curve = progress_curve(dense, target)
    if curve is not None:
        progress, valid = curve
        auc = float(progress[valid].mean())
        sr = stall_and_recovery(progress, valid)
        # 시간 지표는 dense 곡선에서 뽑는다. 판정용 envelope은 팔이 도형을 가리는
        # 동안 갱신이 멈춰서, 실제로는 진작 지워졌어도 팔이 빠지는 순간에야 값이
        # 떨어진다 — 그대로 쓰면 t@50과 t@90이 거의 같아지고 "언제 지웠나"가 아니라
        # "언제 확인됐나"를 재게 된다.
        f50, f90 = reach(progress, valid, 0.5), reach(progress, valid, SUCCESS_ERASED)
    else:
        auc, sr = None, {"stall_events": None, "recovered": None, "recovery_rate": None}
        f50, f90 = frames_to(env, 0.5), frames_to(env, SUCCESS_ERASED)

    # 접촉: target이 CONTACT_FRAMES 이상 연속으로 가려진 시점
    occ_t = np.any([occ[k] for k, _ in shapes if k.split("#")[0] == target], axis=0)
    contact_at = _first_run(occ_t, CONTACT_FRAMES)

    grip = read_gripper(ep_dir)
    ph = gripper_phases(grip) if grip is not None else {"hold_start": None, "release": None}

    # ── 단계 점수 ────────────────────────────────────────
    stages = {
        # 그리퍼 로그가 없으면 접촉이 있었다는 사실로 접근을 인정한다(영상만으로 채점 가능).
        "approach": ph["hold_start"] is not None if grip is not None else contact_at is not None,
        "contact": contact_at is not None,
        "start": tgt["erased_frac"] >= START_ERASED,
        "complete": tgt["erased_frac"] >= SUCCESS_ERASED,
        "selective": distractor <= MAX_DISTRACTOR,
    }
    score = sum(STAGE_POINTS[k] for k, v in stages.items() if v)

    row.update(
        valid=True,
        note="",
        erased_target=tgt["erased_frac"],
        erased_distractor=round(float(distractor), 4),
        selective_erase=round(tgt["erased_frac"] - float(distractor), 4),
        ink_init=tgt["ink_init"],
        ink_final=tgt["ink_final"],
        occluded_frac=next(
            (v["occluded_frac"] for k, v in per_shape.items()
             if k.split("#")[0] == target and "occluded_frac" in v),
            None,
        ),
        success=stages["complete"] and stages["selective"],
        score=score,
        task_progress=round(100.0 * score / sum(STAGE_POINTS.values()), 1),
        **{f"stage_{k}": int(v) for k, v in stages.items()},
        t_contact_s=round(contact_at / fps, 2) if contact_at is not None else None,
        t50_s=round(f50 / fps, 2) if f50 is not None else None,
        t90_s=round(f90 / fps, 2) if f90 is not None else None,
        auc_progress=round(auc, 4) if auc is not None else None,
        duration_s=round(row["n_frames"] / fps, 2),
        grasp_s=round(ph["hold_start"] / fps, 2) if ph["hold_start"] is not None else None,
        release_s=round(ph["release"] / fps, 2) if ph["release"] is not None else None,
        **sr,
    )
    row["termination"] = termination_reason(row, ph, grip)
    row["failure_mode"] = failure_mode(row, stages, ph)
    return row


def termination_reason(row: dict, ph: dict, grip) -> str:
    """왜 끝났나. 성공 판정과 분리한다 — 로봇이 '끝났다'고 놓은 것과 실제로
    다 지운 것은 다른 사건이고, 섞으면 착각 종료가 정상 종료로 기록된다."""
    if grip is None:
        return "unknown(no_log)"
    if ph.get("release") is not None:
        return "release"  # 로봇이 스스로 지우개를 놓음
    if row.get("ended_stalled"):
        return "stall"
    return "cutoff"


def failure_mode(row: dict, stages: dict, ph: dict) -> str:
    """실패 1건당 주된 원인 하나(SO-101 논문의 single primary failure mode 규칙)."""
    if row["success"]:
        return ""
    if not stages["selective"]:
        return "selectivity_failure"  # distractor를 건드림
    if not stages["approach"]:
        return "grasp_failure"  # 지우개를 못 잡음
    if not stages["contact"]:
        return "approach_failure"  # 잡았지만 보드에 못 닿음
    if not stages["start"]:
        return "contact_ineffective"  # 닿았지만 안 지워짐 (누르는 힘/자세)
    if ph.get("release") is not None:
        return "premature_release"  # 덜 지웠는데 다 지웠다고 판단하고 놓음
    if (row.get("stall_events") or 0) >= 2:
        return "repetition_loop"  # 같은 곳만 반복, 진행 없음
    return "incomplete_erase"


def read_gripper(ep_dir: Path) -> np.ndarray | None:
    """action[:,6] = 그리퍼 명령. parquet이 없거나 pyarrow가 없으면 None."""
    hits = sorted(ep_dir.glob("data/**/*.parquet"))
    if not hits:
        return None
    # 조용히 None을 내면 안 된다. 그리퍼를 못 읽으면 종료 사유가 전부 unknown(no_log)이
    # 되고 premature_release 판정이 통째로 죽는데, 예전엔 그 이유가 어디에도 안 남아서
    # "환경을 잘못 잡았다"와 "로그가 원래 없다"를 구분할 수 없었다. 실제로 pyarrow 없는
    # 인터프리터로 30개를 채점하고 나서야 알아챘다.
    try:
        import pyarrow.parquet as pq
    except ImportError:
        _warn_once("pyarrow가 없어 그리퍼 로그를 못 읽는다 — 종료 사유·premature_release "
                   "판정이 전부 비활성된다. 프로젝트 환경(conda activate ugrp)에서 돌릴 것.")
        return None
    try:
        table = pq.read_table(hits[0], columns=["action"])
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
    """결과는 리포 최상단 evaluation/ 아래에 날짜·조건별 폴더로 모은다.

    outputs/ 밑에 두면 학습 산출물에 섞여서 나중에 못 찾는다. 실험 결과는
    사람이 자주 열어보는 것이라 눈에 띄는 곳에 둔다.
    """
    stamp = datetime.now().strftime("%m%d")
    name = "_".join(x for x in (stamp, model, condition) if x)
    return _repo_root() / "evaluation" / name


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists() or (parent / "scripts").is_dir():
            return parent
    return Path.cwd()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """성공률의 Wilson score 95% 신뢰구간. n이 작을 때 정규근사보다 정직하다.

    시행이 10~50회뿐이면 성공률 차이는 대부분 구간이 겹친다. 그래서 주장은
    연속 지표(erased_target 등)로 하고, 성공률은 구간과 함께 보고만 한다.
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
        opp = sum(r["stall_events"] or 0 for r in rs)
        rec = sum(r["recovered"] or 0 for r in rs)
        out[f"{model}/{cond}"] = {
            "n_episodes": n,
            "success_rate": round(k / n, 4),
            "success_ci95": wilson(k, n),
            "task_progress": mean_sem([r["task_progress"] for r in rs]),
            "erased_target": mean_sem([r["erased_target"] for r in rs]),
            "erased_distractor": mean_sem([r["erased_distractor"] for r in rs]),
            "selective_erase": mean_sem([r["selective_erase"] for r in rs]),
            "t90_s": mean_sem([r["t90_s"] for r in rs]),
            "auc_progress": mean_sem([r["auc_progress"] for r in rs]),
            "duration_s": mean_sem([r["duration_s"] for r in rs]),
            "recovery_rate": round(rec / opp, 3) if opp else None,
            "recovery_opportunities": opp,
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
        f"- 성공 기준: target erased ≥ {SUCCESS_ERASED}, distractor ≤ {MAX_DISTRACTOR}",
        "- 성공률 구간은 Wilson 95%. 시행이 적으면 구간이 넓으니 연속 지표를 함께 볼 것.",
        "",
        "## 조건별",
        "",
        "| 조건 | n | 성공률 (95% CI) | 진행률 | erased(target) | erased(distr.) | t@90 (s) | 회복률 |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for key, s in summary.items():
        lo, hi = s["success_ci95"]
        lines.append(
            f"| {key} | {s['n_episodes']} | {s['success_rate']:.2f} ({lo:.2f}–{hi:.2f}) "
            f"| {_fmt(s['task_progress'])} | {_fmt(s['erased_target'])} "
            f"| {_fmt(s['erased_distractor'])} | {_fmt(s['t90_s'])} "
            f"| {s['recovery_rate'] if s['recovery_rate'] is not None else '—'} |"
        )
    lines += ["", "## 실패 유형", ""]
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
    "success", "score", "task_progress",
    "stage_approach", "stage_contact", "stage_start", "stage_complete", "stage_selective",
    "erased_target", "erased_distractor", "selective_erase", "ink_init", "ink_final",
    "t_contact_s", "t50_s", "t90_s", "auc_progress", "duration_s",
    "grasp_s", "release_s", "termination", "failure_mode",
    "stall_events", "recovered", "recovery_rate", "ended_stalled",
    "occluded_frac", "shapes_found", "n_frames", "path",
]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("episodes", nargs="*", help="에피소드 디렉터리 (glob 가능)")
    p.add_argument("--model", default="", help="비교 축 1 (smolvla / pi0 …)")
    p.add_argument("--condition", default="", help="비교 축 2 (async / sync …)")
    p.add_argument("--target", default=None, help="지울 도형. 생략하면 폴더명에서 추론")
    p.add_argument("--board", type=int, nargs=4, default=M.DEFAULT_BOARD, metavar=("X", "Y", "W", "H"))
    p.add_argument(
        "--exclude", action="append", default=[],
        type=lambda s: tuple(int(v) for v in s.split(",")),
        metavar="X,Y,W,H",
        help="이 전역좌표 사각형은 도형 검출에서 뺀다 (ink_metric.py --exclude와 동일, "
             "여러 번 줄 수 있음). 보드에 눌어붙은 테이프 자국처럼 위치가 고정된 이물질용",
    )
    p.add_argument("--dark-ratio", type=float, default=0.72)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="생략하면 화면 출력만. --save를 주면 evaluation/<MMDD>_<model>_<condition>")
    p.add_argument("--save", action="store_true", help="기본 폴더에 저장")
    p.add_argument("--selftest", action="store_true", help="합성 영상으로 채점 로직 검증")
    p.add_argument("--summarize", type=Path, default=None,
                   help="이미 있는 episodes.csv로 요약만 다시 만든다 (손 채점 시트도 그대로 됨)")
    args = p.parse_args(argv)

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
                f"erased={row['erased_target']:.3f} distr={row['erased_distractor']:.3f} "
                f"score={row['score']}/10 {row['termination']} {row['failure_mode']}"
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
        new = not csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerows(rows)
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
    numeric = {
        "score", "task_progress", "erased_target", "erased_distractor",
        "selective_erase", "t50_s", "t90_s", "auc_progress", "duration_s",
        "stall_events", "recovered",
    }
    truthy = ("True", "true", "1")
    rows = []
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            r["valid"] = r["valid"] in truthy
            r["success"] = r["success"] in truthy
            r["ended_stalled"] = r.get("ended_stalled") in truthy
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

    def make(path, erase_to, stall=False, distractor_hit=False):
        """target(원)을 erase_to 비율만큼 지우는 합성 영상."""
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (400, 400))
        board, ink = (240, 240, 238), (60, 60, 55)
        for i in range(120):
            f = np.full((400, 400, 3), board, np.uint8)
            cv2.circle(f, (100, 100), 40, ink, 3)
            cv2.rectangle(f, (250, 250), (330, 330), ink, 3)
            # 진행: 30프레임부터 각도를 늘려가며 지운다. stall이면 중간에 60프레임 멈춤.
            t = max(0, i - 30)
            if stall and i > 60:
                t = 30
            frac = min(erase_to, t / 60.0)
            if frac > 0:
                cv2.ellipse(f, (100, 100), (40, 40), 0, 0, 360 * frac, board, 8)
            if distractor_hit and i > 80:
                cv2.line(f, (250, 250), (330, 330), board, 10)
            w.write(f)
        w.release()

    board = (0, 0, 400, 400)

    ok = tmp / "ok.mp4"
    make(ok, 1.0)
    shapes, ink, occ = M.track(ok, board, 0.72)
    ps = M.summarize(ink, occ)
    env = target_envelope(ps, "circle")
    assert env is not None and frames_to(env, 0.9) is not None, "완료 프레임을 못 찾음"
    assert M.aggregate(ps, "circle")["erased_frac"] > 0.9

    # 부분 지움 → complete 단계 탈락, incomplete_erase
    half = tmp / "half.mp4"
    make(half, 0.5)
    shapes, ink, occ = M.track(half, board, 0.72)
    ps = M.summarize(ink, occ)
    e = M.aggregate(ps, "circle")["erased_frac"]
    assert START_ERASED <= e < SUCCESS_ERASED, f"부분 지움이 {e}로 나옴"

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

    # 정체·회복: 멈췄다가 다시 진행하면 회복 1건
    prog = np.concatenate([np.linspace(0, 0.3, 30), np.full(120, 0.3), np.linspace(0.3, 0.9, 50)])
    sr = stall_and_recovery(prog, np.ones(len(prog), bool))
    assert sr["stall_events"] >= 1 and sr["recovered"] >= 1, sr
    # 짧은 숨고르기(1초)는 정체로 세면 안 된다
    short = np.concatenate([np.linspace(0, 0.3, 30), np.full(45, 0.3), np.linspace(0.3, 0.9, 50)])
    assert stall_and_recovery(short, np.ones(len(short), bool))["stall_events"] == 0
    # 멈춘 채로 끝나면 회복 기회로 세지 않는다
    prog2 = np.concatenate([np.linspace(0, 0.3, 30), np.full(150, 0.3)])
    assert stall_and_recovery(prog2, np.ones(len(prog2), bool))["ended_stalled"], "종료 정체 미탐지"

    # 실패 유형 분기
    base = {"success": False, "stall_events": 0, "ended_stalled": False}
    st = {"approach": True, "contact": True, "start": True, "complete": False, "selective": True}
    assert failure_mode(base, st, {"release": 50}) == "premature_release"
    assert failure_mode({**base, "stall_events": 3}, st, {"release": None}) == "repetition_loop"
    assert failure_mode(base, {**st, "selective": False}, {}) == "selectivity_failure"
    assert failure_mode(base, {**st, "contact": False}, {}) == "approach_failure"
    assert failure_mode({"success": True}, st, {}) == ""

    # Wilson 구간 — 10회 중 8회 성공이면 구간이 넓어야 한다(주장 못 함을 보이는 근거)
    lo, hi = wilson(8, 10)
    assert lo < 0.55 and hi > 0.93, (lo, hi)
    assert wilson(0, 0) == (0.0, 0.0)

    print("selftest OK — 잉크 판정 / 그리퍼 위상 / 정체·회복 / 실패분기 / CI 전부 통과")


if __name__ == "__main__":
    sys.exit(main())
