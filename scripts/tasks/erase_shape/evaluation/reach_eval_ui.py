#!/usr/bin/env python3
"""reach_eval_ui.py — "맞는 도형으로 갔나"만 보는 평가 창.

erase_eval_ui.py의 축소판이다. 저 도구는 "끝까지 다 지웠나"를 재는데, 그건 지우기
실력과 도형 선택 능력이 한 숫자에 섞인다 — 프롬프트를 제대로 알아듣고 옳은 도형으로
갔는데 마무리를 못 해서 실패로 찍히는 시도가 반복됐다. 방해 도형을 넣은 뒤로는
**어느 도형을 골랐나**만 따로 보고 싶어서 이 창을 나눴다.

erase_eval_ui.py 대비 뺀 것:
  - 영상·데이터셋 녹화        (--mode demo로 러너에서 애초에 안 만든다)
  - 끝까지 지웠는지 판정      (성공/실패, erased_target 임계)
  - 실패 유형 7종 분류
  - 잉크 잔여량·residual 분석
  - 그리퍼 위상 / 종료 사유

남긴 것:
  - 대상 도형 지정과 정렬 도구 연동 (도형을 매번 같은 자리·같은 크기로 그리게)
  - 보드에 뭐가 어디 있었나 — 좌/우 배치까지 (위치 편향을 사후에 가려내려면 필수)
  - 사람이 직접 고르는 최종 판정

**자동 추천은 참고값이다.** 기준 프레임과 최종 프레임의 도형별 잉크 변화를 보고
어느 쪽을 건드렸는지 추천만 하고, 라디오 버튼을 미리 골라줄 뿐 기록은 사람이 누른
값으로 남는다. 잉크 변화는 '문질렀나'를 재는 것이지 '갔나'를 재는 게 아니라서,
옳은 도형 위까지 갔지만 안 문지른 시도는 '접근 없음'으로 나온다 — 그래서 사람이
덮어쓸 수 있어야 한다. 사람이 바꾼 경우 auto_verdict/agree 컬럼에 원래 추천이 남아
나중에 자동 판정의 일치도를 따로 볼 수 있다.

⚠️ **[중단]은 안전장치가 아니다.** 롤아웃 프로세스를 종료할 뿐이라 팔이 즉시 서지
않는다. 위험할 때는 하드웨어 비상정지를 먼저 누른다.

사용:
    # 하드웨어 없이 UI 점검 — 기존 시도 폴더를 롤아웃 결과인 척 순서대로 먹인다
    python scripts/tasks/erase_shape/evaluation/reach_eval_ui.py \\
        --target circle --dry-run 'records/0822/hil/20260822-15*'

    # 실제 세션 (러너는 --mode demo — 영상을 안 남긴다)
    python scripts/tasks/erase_shape/evaluation/reach_eval_ui.py \\
        --target circle --model smolvla --condition distractor \\
        --watch-dir records/hil \\
        --rollout-cmd "python scripts/tasks/erase_shape/runtime/erase_run.py ... --mode demo --confirm"

키: Enter 기록하고 다음 · space 시작/중단 · n 이번 시도 버리기 · 1~4 판정 · x 무효
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import json
import os
import queue
import shlex
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import font as tkfont
from tkinter import messagebox, ttk

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import erase_eval as E  # noqa: E402
import erase_check as EC  # noqa: E402

M = E.M
ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "analysis"

# 이 이상 잉크가 줄면 "그 도형을 건드렸다"로 본다.
#
# 근거: ink_metric의 방해 도형 허용 상한 MAX_DISTRACTOR=0.10이 노이즈 플로어(≈0.033)의
# 3배로 잡혀 있다. 여기서 보려는 건 "다 지웠나"가 아니라 "건드렸나"라서 그보다 훨씬
# 낮게 잡아야 한다 — 다만 노이즈 위여야 조명 흔들림에 안 속는다. 0.05는 노이즈의
# 1.5배이면서 완주 임계(0.90)와는 한참 떨어져 있다.
TOUCH = 0.05

VERDICTS = ["target", "distractor", "none", "unclear"]
VERDICT_LABELS = {
    "target": "1 맞는 도형으로 갔다",
    "distractor": "2 엉뚱한 도형으로 갔다",
    "none": "3 어느 도형에도 안 갔다 (못 잡음 / 보드에 못 닿음 포함)",
    "unclear": "4 판단 불가 (둘 다 건드림 / 애매함)",
}
DETAIL_ROWS = [
    ("layout", "보드 배치"),
    ("auto", "자동 추천"),
    ("ink", "잉크 변화"),
    ("frame_source", "판정 프레임"),
]

FIELDS = [
    "episode", "model", "condition", "trial", "target",
    "valid", "verdict", "auto_verdict", "agree", "note",
    "target_side", "target_pos",
    "n_distractors", "distractors", "distractor_side", "distractor_pos",
    "shapes_found", "d_target", "d_distractor", "frame_source",
    "elapsed_s", "path",
]


def _font(base: str, size: int, bold: bool = False) -> tkfont.Font:
    """명명 폰트를 크기만 바꿔 복제한다.

    conda ugrp 환경의 기본 Tk는 Xft 없이 빌드되어 한글 글리프가 없다 — 이 상태로
    font=를 명시하면 폴백이 깨져 한글이 □로 나온다. 실행은 scripts/14__reach_session.sh
    로 해서 시스템 Tcl/Tk를 LD_PRELOAD로 얹어야 한다(13__eval_session.sh와 같은 이유).
    """
    f = tkfont.nametofont(base).copy()
    f.configure(size=size, weight="bold" if bold else "normal")
    return f


# ═══════════════════════════════════════════════════════════════════
# 판정 — "어느 도형을 건드렸나"
# ═══════════════════════════════════════════════════════════════════
def load_frames(ep_dir: Path):
    """기준/최종 프레임 한 장씩. -> (ref, after, source)

    --mode demo는 영상을 안 만들고 PNG 두 장만 남긴다(erase_run.py save_frame:
    00_reference.png / 01_after.png). 영상을 남기는 모드로 돌린 폴더도 그대로
    읽을 수 있게 erase_run.py가 롤아웃 폴더에 쓰는 이름도 같이 본다.
    """
    for ref_name, aft_glob, src in (
        ("00_reference.png", "*_after.png", "park_frame"),
        ("reference_frame.png", "judge_frame.png", "park_frame"),
    ):
        ref = M.imread(ep_dir / ref_name)
        hits = sorted(ep_dir.glob(aft_glob)) if "*" in aft_glob else [ep_dir / aft_glob]
        aft = M.imread(hits[-1]) if hits else None
        if ref is not None and aft is not None:
            return ref, aft, src
    raise FileNotFoundError(
        "기준/최종 프레임을 못 찾았습니다 — 00_reference.png + *_after.png 가 있어야 합니다 "
        "(러너를 --no-save-frames로 돌렸는지 확인)"
    )


def judge_reach(ep_dir: Path, target: str, board, dark_ratio: float, exclude=None) -> dict:
    """어느 도형을 건드렸는지 추천한다. 최종 판정은 사람이 한다.

    성공/실패를 내지 않는다 — 이 도구는 "다 지웠나"를 안 본다. 내는 건 도형별 잉크
    변화량과 거기서 나온 추천값뿐이다.
    """
    row: dict = {"episode": ep_dir.name, "path": str(ep_dir), "target": target,
                 "valid": True, "note": ""}
    try:
        ref_frame, after_frame, source = load_frames(ep_dir)
    except Exception as exc:
        row.update(valid=False, auto_verdict="unclear",
                   note=f"프레임 로드 실패: {type(exc).__name__}: {exc}")
        return row
    row["frame_source"] = source

    checker = EC.EraseChecker(board, dark_ratio)
    try:
        ref = checker.set_reference(ref_frame, exclude=exclude)
    except RuntimeError as exc:
        row.update(valid=False, auto_verdict="unclear", note=str(exc))
        return row

    # 배치 — 좌/우가 없으면 "프롬프트를 이해했다"와 "항상 한쪽으로 간다"를 구분할 수
    # 없다. 타겟이 우연히 계속 같은 쪽에 있으면 위치 편향이 정답률로 위장된다.
    board_mid_x = board[0] + board[2] / 2.0

    def center(box):
        x, y, w, h = box
        return int(x + w / 2), int(y + h / 2)

    def side(box):
        return "L" if center(box)[0] < board_mid_x else "R"

    # 검출 순서를 유지한다 — 라벨/좌표/좌우 세 컬럼이 같은 순서로 짝지어야 한다.
    tgt = [b for k, b in ref["shapes"] if k.split("#")[0] == target]
    dis = [(k.split("#")[0], b) for k, b in ref["shapes"] if k.split("#")[0] != target]
    row["shapes_found"] = "|".join(sorted({k.split("#")[0] for k, _ in ref["shapes"]}))
    row["target_pos"] = "|".join("{},{}".format(*center(b)) for b in tgt)
    row["target_side"] = "|".join(side(b) for b in tgt)
    row["distractors"] = "|".join(lbl for lbl, _ in dis)
    row["distractor_pos"] = "|".join("{},{}".format(*center(b)) for _, b in dis)
    row["distractor_side"] = "|".join(side(b) for _, b in dis)
    row["n_distractors"] = len(dis)

    if not tgt:
        row.update(valid=False, auto_verdict="unclear",
                   note=f"기준 프레임에 {target} 도형이 없음 — 보드/조명 확인")
        return row

    result = checker.check(after_frame, target)
    d_t = result["target_erased"]
    d_d = result["max_distractor_erased"] if dis else None
    row["d_target"] = d_t
    row["d_distractor"] = d_d

    touched_t = d_t >= TOUCH
    touched_d = d_d is not None and d_d >= TOUCH
    if touched_t and not touched_d:
        row["auto_verdict"] = "target"
    elif touched_d and not touched_t:
        row["auto_verdict"] = "distractor"
    elif touched_t and touched_d:
        # 둘 다 문질렀다. 어느 쪽을 '먼저 골랐나'는 두 장의 사진으로는 못 가른다
        # — 지나가며 스친 것과 골라서 간 것이 같은 흔적을 남긴다. 사람이 본다.
        row["auto_verdict"] = "unclear"
        row["note"] = "둘 다 건드림 — 사람이 판단"
    else:
        # 아무것도 안 문질렀다. 도형 위까지 갔지만 안 닿았을 수도 있으므로
        # '안 갔다'로 단정하지 않고 추천만 한다.
        row["auto_verdict"] = "none"
        row["note"] = "잉크 변화 없음 — 접근만 했는지 사람이 확인"
    return row


# ═══════════════════════════════════════════════════════════════════
# 롤아웃 실행
# ═══════════════════════════════════════════════════════════════════
class Rollout:
    """롤아웃 1회를 돌리고, 끝나면 볼 시도 폴더를 알려준다."""

    # SIGINT를 받은 러너가 정리를 끝낼 때까지 기다리는 시간. erase_eval_ui는 영상
    # 인코딩(실측 40~75초)까지 기다리느라 180초였지만, 여기는 --mode demo라 인코딩이
    # 없다 — 남는 건 파킹 이동(최대 ~15초)뿐이라 60초면 충분하다.
    GRACEFUL_STOP_S = 60.0

    def __init__(self, cmd: str | None, watch_dir: Path | None, dry_queue: list[Path] | None):
        self.cmd = cmd
        self.watch_dir = watch_dir
        self.dry_queue = list(dry_queue or [])
        self.proc: subprocess.Popen | None = None
        self.log: queue.Queue[str] = queue.Queue()
        self._before: set[Path] = set()
        self._dry_pick: Path | None = None
        self._dry_until = 0.0
        self._kill_deadline: float | None = None

    @property
    def dry_run(self) -> bool:
        return self.cmd is None

    def start(self) -> bool:
        # 이전 시도의 중단 상태가 남아 있으면 stop()의 중복 방지 가드에 걸려
        # 이번 시도는 [중단]을 눌러도 SIGINT가 안 나간다.
        self._kill_deadline = None
        if self.dry_run:
            if not self.dry_queue:
                self.log.put("[DRY] 더 먹일 시도 폴더가 없습니다")
                return False
            self._dry_pick = self.dry_queue.pop(0)
            self._dry_until = time.monotonic() + 2.0
            self.log.put(f"[DRY] {self._dry_pick.name} 를 롤아웃 결과로 사용합니다 (2초)")
            return True

        self._before = self._snapshot()
        self.log.put(f"$ {self.cmd}")
        # 자식이 또 자식을 띄우므로 프로세스 그룹째 잡는다 — 안 그러면 중단해도 손자가 남는다.
        self.proc = subprocess.Popen(
            shlex.split(self.cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, start_new_session=True,
        )
        threading.Thread(target=self._pump, daemon=True).start()
        return True

    def _pump(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            self.log.put(line.rstrip())

    def running(self) -> bool:
        if self.dry_run:
            return self._dry_pick is not None and time.monotonic() < self._dry_until
        return self.proc is not None and self.proc.poll() is None

    def stop(self, reason: str) -> None:
        """SIGINT만 보내고 즉시 돌아온다 — 여기서 기다리면 GUI가 그만큼 얼어붙는다."""
        if self._kill_deadline is not None:
            return  # 이미 중단 요청이 나갔다. SIGINT를 또 보내면 파킹 정리를 끊는다.
        self.log.put(f"[STOP] {reason}")
        if self.dry_run:
            self._dry_until = 0.0
            return
        if self.proc and self.proc.poll() is None:
            with contextlib.suppress(Exception):
                os.killpg(os.getpgid(self.proc.pid), signal.SIGINT)
            self._kill_deadline = time.monotonic() + self.GRACEFUL_STOP_S
            self.log.put(f"[STOP] 파킹 정리를 최대 {self.GRACEFUL_STOP_S:g}초까지 기다립니다")

    def enforce_stop_deadline(self) -> None:
        """정리가 GRACEFUL_STOP_S를 넘기면 그때만 강제 종료한다(_tick에서 매번 호출)."""
        if self._kill_deadline is None or self.proc is None:
            return
        if self.proc.poll() is not None:
            self._kill_deadline = None
            return
        if time.monotonic() >= self._kill_deadline:
            self.log.put("[STOP] 정리가 끝나지 않아 강제 종료합니다")
            with contextlib.suppress(Exception):
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            self._kill_deadline = None

    def result_dir(self) -> Path | None:
        if self.dry_run:
            return self._dry_pick
        new = sorted(self._snapshot() - self._before, key=lambda p: p.stat().st_mtime)
        return new[-1] if new else None

    def _snapshot(self) -> set[Path]:
        if not self.watch_dir or not self.watch_dir.is_dir():
            return set()
        return {p for p in self.watch_dir.iterdir() if p.is_dir()}


# ═══════════════════════════════════════════════════════════════════
# 창
# ═══════════════════════════════════════════════════════════════════
class ReachSession(tk.Tk):
    POLL_MS = 200
    # 파킹 이후 후처리에 허용하는 시간. 영상 인코딩이 없으니 erase_eval_ui(300초)보다
    # 짧게 잡는다 — 여기서 오래 걸리면 그건 정상이 아니라 걸린 것이다.
    POST_PARK_TIMEOUT_S = 90.0
    # RealSense는 프로세스가 죽어도 USB 레벨에서 바로 안 풀린다. 그 전에 다른
    # 프로세스가 같은 장치를 열면 librealsense가 세그폴트로 죽는다(2026-08-18 실물).
    ALIGN_LAUNCH_DELAY_MS = 2500

    def __init__(self, args):
        super().__init__()
        self.title("도형 선택 평가 세션 (도달 판정만)")
        self.geometry("1400x900")
        self.args = args
        self.rollout = Rollout(args.rollout_cmd, args.watch_dir, args.dry_run)
        self.trial = self._next_trial()
        self.row: dict | None = None
        # idle → running → scoring → review → idle. 버튼 상태를 읽어서 분기하면 안 된다:
        # ttk 위젯의 ["state"]는 문자열이 아니라 Tcl 객체라 == "normal" 비교가 항상 거짓이다.
        self.phase = "idle"
        self.score_q: queue.Queue[dict] = queue.Queue()
        self.started_at: float | None = None
        self.elapsed_s: float | None = None
        self.parked_at: float | None = None
        self.align_proc: subprocess.Popen | None = None
        self._aligning = False
        self._align_after_id: str | None = None
        self._rollout_was_busy = False
        self.aborted = False

        self.verdict = tk.StringVar(value="")
        self.valid = tk.BooleanVar(value=True)
        self._build()
        self._bind_keys()
        self.after(self.POLL_MS, self._tick)

    # ------------------------------------------------------------- 레이아웃
    def _build(self) -> None:
        top = ttk.Frame(self, padding=(10, 8))
        top.pack(fill="x")
        mode = "DRY-RUN (로봇 없음)" if self.rollout.dry_run else "실행"
        ttk.Label(top, text=f"{self.args.model or '?'} / {self.args.condition or '?'} · "
                            f"타겟 {self.args.target} · {mode}",
                  font=_font("TkDefaultFont", 12, True)).pack(side="left")
        self.trial_label = ttk.Label(top, text="", font=_font("TkDefaultFont", 12))
        self.trial_label.pack(side="left", padx=16)
        self.clock = ttk.Label(top, text="", font=_font("TkDefaultFont", 12))
        self.clock.pack(side="right")

        warn = ttk.Frame(self, padding=(10, 0))
        warn.pack(fill="x")
        ttk.Label(warn, text="⚠ [중단]은 안전장치가 아닙니다. 위험하면 하드웨어 비상정지를 먼저 누르세요.",
                  foreground="#b00").pack(side="left")

        bar = ttk.Frame(self, padding=(10, 8))
        bar.pack(fill="x")
        self.btn_start = ttk.Button(bar, text="시작 (space)", command=self.on_start)
        self.btn_start.pack(side="left")
        self.btn_stop = ttk.Button(bar, text="중단 (space)", command=self.on_stop, state="disabled")
        self.btn_stop.pack(side="left", padx=6)
        ttk.Button(bar, text="이번 시도 버리기 (n)", command=self.on_discard).pack(side="left", padx=6)
        self.btn_align = ttk.Button(bar, text="정렬 도구 켜기", command=self.on_toggle_align)
        self.btn_align.pack(side="left", padx=6)
        ttk.Button(bar, text="세션 종료", command=self.on_close).pack(side="right")

        body = ttk.Frame(self, padding=(10, 4))
        body.pack(fill="both", expand=True)

        left = ttk.LabelFrame(body, text="이번 시도", padding=10)
        left.pack(side="left", fill="both", expand=True)
        self.headline = ttk.Label(left, text="아직 시도 없음", font=_font("TkDefaultFont", 20, True))
        self.headline.pack(anchor="w")
        # 여러 줄을 라벨 하나에 담지 않는다 — 이 환경의 Tk는 한글 줄 간격을 기본 폰트의
        # linespace로 잡아서 한글이 섞인 줄이 위아래로 겹친다(2026-08-18 실물 캡처).
        self.detail = ttk.Frame(left)
        self.detail.pack(anchor="w", pady=(8, 10), fill="x")
        self._detail_vals: dict[str, ttk.Label] = {}
        for i, (key, label) in enumerate(DETAIL_ROWS):
            ttk.Label(self.detail, text=label, font=_font("TkDefaultFont", 11)).grid(
                row=i, column=0, sticky="w", padx=(0, 16))
            val = ttk.Label(self.detail, text="", font=_font("TkFixedFont", 11))
            val.grid(row=i, column=1, sticky="w")
            self._detail_vals[key] = val
        self.detail_note = ttk.Label(self.detail, text="", font=_font("TkDefaultFont", 11),
                                     wraplength=520, justify="left")
        self.detail_note.grid(row=len(DETAIL_ROWS), column=0, columnspan=2, sticky="w", pady=(6, 0))

        ttk.Label(left, text="판정 — 맞는 도형으로 갔나? (사람이 최종 결정)",
                  font=_font("TkDefaultFont", 11, True)).pack(anchor="w")
        for v in VERDICTS:
            ttk.Radiobutton(left, text=VERDICT_LABELS[v], value=v,
                            variable=self.verdict).pack(anchor="w")
        ttk.Label(left, text="자동 추천은 잉크 변화 기반입니다 — 도형 위까지 갔어도 안 "
                             "문질렀으면 '안 갔다'로 추천됩니다. 눈으로 본 것을 우선하세요.",
                  wraplength=520, justify="left", foreground="#555").pack(anchor="w", pady=(4, 0))

        ttk.Separator(left).pack(fill="x", pady=10)
        ttk.Checkbutton(left, text="이 시도는 유효함 (끄면 집계에서 빠짐)",
                        variable=self.valid).pack(anchor="w")
        ttk.Label(left, text="메모 — 사람이 건드림 / 마커 번짐 / 조명 변화 등").pack(anchor="w", pady=(8, 2))
        self.note = tk.Text(left, height=3, width=52)
        self.note.pack(fill="x")
        self.btn_confirm = ttk.Button(left, text="기록하고 다음 (Enter)",
                                      command=self.on_confirm, state="disabled")
        self.btn_confirm.pack(anchor="e", pady=8)

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))
        ttk.Label(right, text="롤아웃 로그", font=_font("TkDefaultFont", 11, True)).pack(anchor="w")
        self.logbox = tk.Text(right, height=16, width=60, font=_font("TkFixedFont", 10))
        self.logbox.pack(fill="both", expand=True)
        ttk.Label(right, text="이번 세션 기록", font=_font("TkDefaultFont", 11, True)).pack(
            anchor="w", pady=(8, 2))
        self.tree = ttk.Treeview(right, columns=("verdict", "side", "auto"),
                                 show="headings", height=10)
        for col, cap, w in (("verdict", "판정", 110), ("side", "타겟 위치", 80),
                            ("auto", "자동 추천", 110)):
            self.tree.heading(col, text=cap)
            self.tree.column(col, width=w, anchor="center")
        self.tree.pack(fill="both", expand=True)

        self.status = ttk.Label(self, text="", padding=(10, 6))
        self.status.pack(fill="x")
        self._refresh_header()

    def _bind_keys(self) -> None:
        self.bind("<Return>", lambda _: self.on_confirm())
        self.bind("<space>", lambda _: self.on_stop() if self.phase == "running" else self.on_start())
        self.bind("n", lambda _: self.on_discard())
        self.bind("x", lambda _: self.valid.set(not self.valid.get()))
        for i, v in enumerate(VERDICTS, start=1):
            self.bind(str(i), lambda _, m=v: self.verdict.set(m))

    # ------------------------------------------------------------- 동작
    def _set_phase(self, phase: str) -> None:
        self.phase = phase
        align_active = self.align_proc is not None or self._aligning
        # 러너가 아직 살아 있으면(중단 후 파킹 중) 카메라를 못 넘겨받는다.
        can_start = phase == "idle" and not align_active and not self.rollout.running()
        self.btn_start.configure(state="normal" if can_start else "disabled")
        self.btn_stop.configure(state="normal" if phase == "running" else "disabled")
        self.btn_confirm.configure(state="normal" if phase == "review" else "disabled")
        # 두 프로세스가 top 카메라를 동시에 열면 세그폴트난다(2026-08-18). 러너는
        # [DISCONNECT] 시점에 카메라를 이미 놓으므로 그 뒤로는 열어준다.
        camera_free = phase in ("idle", "review") or self.parked_at is not None
        self.btn_align.configure(
            text="정렬 도구 끄기" if align_active else "정렬 도구 켜기",
            state="normal" if camera_free else "disabled",
        )

    def on_start(self) -> None:
        if self.phase != "idle":
            return
        if not self.rollout.start():
            self._log("시작할 수 없습니다")
            return
        self.aborted = False
        self.parked_at = None
        self.started_at = time.monotonic()
        self._set_phase("running")
        self.headline.configure(text="롤아웃 진행 중…")
        self._clear_detail()
        self._set_status(f"시도 {self.trial} 실행 중 — 어느 도형으로 가는지 보세요")

    def on_stop(self) -> None:
        if self.phase != "running":
            return
        self.aborted = True
        self.rollout.stop("사람이 중단")
        self._set_status("중단했습니다 — 판정이 끝나면 기록하거나 [이번 시도 버리기]로 넘기세요")

    def on_discard(self) -> None:
        if self.phase == "running":
            self.rollout.stop("시도 버리기")
        self.row = None
        self._set_phase("idle")
        self._reset_card("버림 — 기록하지 않음")

    def on_confirm(self) -> None:
        if self.row is None:
            return
        if not self.verdict.get():
            self._set_status("판정을 먼저 고르세요 (1~4)")
            return
        row = dict(self.row)
        auto = row.get("auto_verdict") or ""
        row["verdict"] = self.verdict.get()
        row["agree"] = (row["verdict"] == auto)
        row["valid"] = bool(self.valid.get())
        row["elapsed_s"] = round(self.elapsed_s, 1) if self.elapsed_s is not None else None
        note = self.note.get("1.0", "end").strip()
        base = self.row.get("note") or ""
        if base and base not in note:
            note = (base + " / " + note).strip(" /")
        if self.aborted:
            note = (note + " [사람이 중단]").strip()
        row["note"] = note

        E.append_rows(self._csv_path(), [row], fields=FIELDS)
        self._write_summary()
        self.tree.insert("", "end", values=(
            VERDICT_LABELS[row["verdict"]].split(" ", 1)[1][:12],
            row.get("target_side", "-") or "-",
            (VERDICT_LABELS.get(auto, "—").split(" ", 1)[-1][:12] if auto else "—"),
        ))
        self.trial += 1
        self.row = None
        self._set_phase("idle")
        self._reset_card("기록했습니다 — [시작]으로 다음 시도")
        self._refresh_header()

    # ------------------------------------------------------------- 정렬 도구
    def _build_align_argv(self) -> list[str]:
        """block_alignment_tool.py 인자를 이 GUI의 args에서 그대로 만들어 붙인다.

        --target을 넘겨서 도형별 박스 크기 + 가이드를 그리게 한다(315 zone-set만 지원).
        --size-source global은 6개 zone을 합친 도형별 median(표본 105개) — zone별로
        나누면 일부 조합 표본이 10~11개까지 줄어 세션 편차에 흔들린다(2026-08-21 실측).
        """
        argv = [sys.executable, str(ANALYSIS_DIR / "block_alignment_tool.py"),
                "--shape-zones", "315", "--size-source", "global"]
        if self.args.target:
            argv += ["--target-shape", self.args.target]
        argv += ["--board", *map(str, self.args.board)]
        for x, y, w, h in self.args.exclude:
            argv += ["--exclude", f"{x},{y},{w},{h}"]
        argv += ["--dark-ratio", str(self.args.dark_ratio)]
        return argv

    def on_toggle_align(self) -> None:
        if self.align_proc is not None:
            if self.align_proc.poll() is None:
                with contextlib.suppress(Exception):
                    self.align_proc.terminate()
            self._log("[ALIGN] 정렬 도구를 끕니다")
            # align_proc은 여기서 None으로 만들지 않는다 — _tick()의 poll 루프가 종료를
            # 감지해서 카메라 해제 대기까지 처리하게 둔다(검증된 경로 재사용).
        elif self._aligning:
            if self._align_after_id is not None:
                self.after_cancel(self._align_after_id)
                self._align_after_id = None
            self._aligning = False
            self._log("[ALIGN] 실행 대기를 취소했습니다")
            self._set_phase(self.phase)
        else:
            self._start_alignment_check()

    def _start_alignment_check(self) -> None:
        if self.align_proc is not None or self._aligning:
            return
        self._aligning = True
        self._set_phase(self.phase)
        self._log(f"[ALIGN] {self.ALIGN_LAUNCH_DELAY_MS/1000:.1f}초 뒤 정렬 도구를 띄웁니다")
        self._set_status("카메라 정리 대기 중…")
        self._align_after_id = self.after(self.ALIGN_LAUNCH_DELAY_MS, self._launch_alignment_tool)

    def _launch_alignment_tool(self) -> None:
        self._aligning = False
        self._align_after_id = None
        argv = shlex.split(self.args.align_cmd) if self.args.align_cmd else self._build_align_argv()
        try:
            self.align_proc = subprocess.Popen(argv, start_new_session=True)
        except Exception as error:
            self._log(f"[WARN] 정렬 도구 실행 실패: {error}")
            self.align_proc = None
            self._set_phase(self.phase)
            return
        self._set_phase(self.phase)
        self._log("[ALIGN] 정렬 도구를 띄웠습니다 — 끄면 [시작]이 열립니다")
        self._set_status("정렬 확인 중 — 타겟과 방해 도형을 모두 그려두세요")

    def _finish_alignment_wait(self) -> None:
        self._aligning = False
        self._set_phase(self.phase)
        self._log("[ALIGN] 카메라 해제 완료 — [시작]으로 다음 시도")
        self._set_status("정렬 확인 완료 — [시작]으로 다음 시도")

    def on_close(self) -> None:
        if self._align_after_id is not None:
            self.after_cancel(self._align_after_id)
        if self.align_proc is not None and self.align_proc.poll() is None:
            with contextlib.suppress(Exception):
                self.align_proc.terminate()
        if self.rollout.running():
            if not messagebox.askyesno(
                "세션 종료",
                "롤아웃이 돌고 있습니다. 중단하고 종료할까요?\n\n"
                "창은 바로 닫히지만 러너는 백그라운드에서 파킹을 끝내고 스스로 "
                "종료합니다. 그동안 카메라를 쥐고 있으니 다른 카메라 도구를 바로 "
                "띄우지 마세요.",
            ):
                return
            self.rollout.stop("세션 종료")
        self.destroy()

    # ------------------------------------------------------------- 루프
    def _tick(self) -> None:
        while True:
            try:
                line = self.rollout.log.get_nowait()
            except queue.Empty:
                break
            self._log(line)
            if "[DISCONNECT]" in line and self.parked_at is None:
                # 러너가 robot.disconnect(park=...) 직전에 찍는 로그다. 여기부터
                # 로봇 단계는 끝났고 카메라도 풀렸다 — 컷오프 예산을 후처리용으로 바꾸고
                # 정렬 도구를 열어준다(다음 시도 도형을 미리 그릴 수 있게).
                self.parked_at = time.monotonic()
                self._log(f"[CUTOFF] 로봇 단계 종료 — 후처리에 {self.POST_PARK_TIMEOUT_S:g}초까지")
                self._set_phase(self.phase)

        # 정렬 도구가 닫혔으면 카메라가 풀릴 시간을 준 뒤 [시작] 잠금을 푼다. 즉시
        # 풀면 다음 롤아웃이 아직 안 풀린 RealSense를 열어 세그폴트가 난다(2026-08-18).
        if self.align_proc is not None and self.align_proc.poll() is not None:
            self.align_proc = None
            self._aligning = True
            self._set_phase(self.phase)
            self._log(f"[ALIGN] 정렬 도구 종료 — 카메라 해제 대기 "
                      f"({self.ALIGN_LAUNCH_DELAY_MS/1000:.1f}초)")
            self.after(self.ALIGN_LAUNCH_DELAY_MS, self._finish_alignment_wait)

        # 중단 요청이 나갔으면 정리가 끝났는지 매 tick 확인한다. phase와 무관하게
        # 돌려야 한다 — [이번 시도 버리기]는 phase를 idle로 되돌리기 때문이다.
        self.rollout.enforce_stop_deadline()
        busy = self.rollout.running()
        if busy != self._rollout_was_busy:
            self._rollout_was_busy = busy
            self._set_phase(self.phase)

        if self.phase == "running":
            elapsed = time.monotonic() - (self.started_at or time.monotonic())
            self.clock.configure(text=f"{elapsed:5.1f}s")
            if self.parked_at is None:
                if self.args.cutoff and elapsed >= self.args.cutoff:
                    self._log(f"[CUTOFF] {self.args.cutoff:g}초 초과 — 중단합니다")
                    self.rollout.stop("컷오프")
            elif time.monotonic() - self.parked_at >= self.POST_PARK_TIMEOUT_S:
                self._log(f"[CUTOFF] 후처리가 {self.POST_PARK_TIMEOUT_S:g}초를 넘었습니다 — 중단")
                self.rollout.stop("후처리 컷오프")
            if not self.rollout.running():
                self.elapsed_s = elapsed
                self._begin_judging()
        elif self.phase == "scoring":
            try:
                self._judged(self.score_q.get_nowait())
            except queue.Empty:
                pass

        self.after(self.POLL_MS, self._tick)

    def _begin_judging(self) -> None:
        ep = self.rollout.result_dir()
        if ep is None:
            self._set_phase("idle")
            self._reset_card("새 시도 폴더를 못 찾았습니다 — --watch-dir 를 확인하세요")
            return
        self._set_phase("scoring")
        self.headline.configure(text="판정 중…")
        self._set_status(f"{ep.name} 확인 중")

        def work():
            try:
                row = judge_reach(ep, self.args.target, tuple(self.args.board),
                                  self.args.dark_ratio, exclude=self.args.exclude)
            except Exception as exc:
                row = {"episode": ep.name, "path": str(ep), "target": self.args.target,
                       "valid": False, "auto_verdict": "unclear",
                       "note": f"{type(exc).__name__}: {exc}"}
            # 결과는 큐로만 넘긴다. 워커 스레드에서 위젯이나 after()를 건드리면
            # tkinter가 "main thread is not in main loop"로 죽는다.
            self.score_q.put(row)

        threading.Thread(target=work, daemon=True).start()

    def _judged(self, row: dict) -> None:
        row.update(model=self.args.model, condition=self.args.condition, trial=self.trial)
        self.row = row
        auto = row.get("auto_verdict") or ""

        if row.get("valid") is False and not row.get("shapes_found"):
            self.headline.configure(text="판정 실패 — 사람이 직접 고르세요")
            self._clear_detail()
            self.detail_note.configure(text=row.get("note", ""))
            self.valid.set(False)
        else:
            parts = [f"{row['target']}@{row.get('target_side') or '?'} (타겟)"]
            parts += [f"{l}@{s}" for l, s in
                      zip((row.get("distractors") or "").split("|"),
                          (row.get("distractor_side") or "").split("|")) if l]
            d_t, d_d = row.get("d_target"), row.get("d_distractor")
            self.headline.configure(text=f"추천: {VERDICT_LABELS.get(auto, '—')}")
            self._set_detail({
                "layout": "  ".join(parts),
                "auto": VERDICT_LABELS.get(auto, "—"),
                "ink": (f"타겟 {d_t:.3f}" if d_t is not None else "타겟 —")
                       + (f"   방해 {d_d:.3f}" if d_d is not None else "   방해 없음")
                       + f"   (기준 {TOUCH:.2f})",
                "frame_source": str(row.get("frame_source", "—")),
            })
            self.detail_note.configure(text=row.get("note", ""))
            self.valid.set(True)

        # 추천을 미리 골라둔다 — 사람이 그대로 두면 동의, 바꾸면 agree=False로 남는다.
        self.verdict.set(auto)
        self._set_phase("review")
        self.btn_confirm.focus_set()
        self._set_status("본 대로 판정을 고르고 Enter — 추천이 틀렸으면 바꾸세요 (1~4)")

    # ------------------------------------------------------------- 보조
    def _csv_path(self) -> Path:
        self.args.out_dir.mkdir(parents=True, exist_ok=True)
        return self.args.out_dir / "trials.csv"

    def _write_summary(self) -> None:
        rows = read_trials(self._csv_path())
        (self.args.out_dir / "summary.md").write_text(
            render_markdown(rows), encoding="utf-8")
        (self.args.out_dir / "summary.json").write_text(
            json.dumps(summarize(rows), ensure_ascii=False, indent=2), encoding="utf-8")

    def _next_trial(self) -> int:
        path = self.args.out_dir / "trials.csv"
        if not path.exists():
            return 1
        same = [r for r in read_trials(path)
                if r.get("model") == self.args.model and r.get("condition") == self.args.condition]
        return len(same) + 1

    def _set_detail(self, values: dict[str, str]) -> None:
        for key, lab in self._detail_vals.items():
            lab.configure(text=values.get(key, ""))

    def _clear_detail(self) -> None:
        for lab in self._detail_vals.values():
            lab.configure(text="")
        self.detail_note.configure(text="")

    def _reset_card(self, message: str) -> None:
        self.headline.configure(text=message)
        self._clear_detail()
        self.note.delete("1.0", "end")
        self.verdict.set("")
        self.valid.set(True)
        self.clock.configure(text="")

    def _refresh_header(self) -> None:
        target = f"/{self.args.trials}" if self.args.trials else ""
        self.trial_label.configure(text=f"시도 {self.trial}{target}")

    def _log(self, line: str) -> None:
        self.logbox.insert("end", f"{datetime.now():%H:%M:%S}  {line}\n")
        self.logbox.see("end")

    def _set_status(self, text: str) -> None:
        self.status.configure(text=text)


# ═══════════════════════════════════════════════════════════════════
# 집계
# ═══════════════════════════════════════════════════════════════════
def read_trials(path: Path) -> list[dict]:
    truthy = ("True", "true", "1")
    rows = []
    with path.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            r["valid"] = r.get("valid", "") in truthy
            r["agree"] = r.get("agree", "") in truthy
            for k in ("d_target", "d_distractor", "elapsed_s"):
                v = r.get(k, "")
                r[k] = None if v in ("", "None") else float(v)
            rows.append(r)
    return rows


def summarize(rows: list[dict]) -> dict:
    """조건별 요약. 좌/우로도 쪼갠다 — 위치 편향을 여기서 본다."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("valid"):
            groups.setdefault(f"{r.get('model','')}/{r.get('condition','')}", []).append(r)

    out = {}
    for key, rs in sorted(groups.items()):
        n = len(rs)
        k = sum(1 for r in rs if r["verdict"] == "target")
        by_side = {}
        for s in ("L", "R"):
            sub = [r for r in rs if r.get("target_side") == s]
            if sub:
                ks = sum(1 for r in sub if r["verdict"] == "target")
                by_side[s] = {"n": len(sub), "rate": round(ks / len(sub), 4),
                              "ci95": E.wilson(ks, len(sub))}
        scored = [r for r in rs if r.get("auto_verdict")]
        out[key] = {
            "n_trials": n,
            "correct_rate": round(k / n, 4) if n else None,
            "correct_ci95": E.wilson(k, n),
            "verdicts": _counts(r["verdict"] for r in rs),
            "by_target_side": by_side,
            "auto_agreement": (round(sum(1 for r in scored if r["agree"]) / len(scored), 4)
                               if scored else None),
        }
    return out


def _counts(values) -> dict:
    out: dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def render_markdown(rows: list[dict]) -> str:
    summary = summarize(rows)
    n_bad = sum(1 for r in rows if not r.get("valid"))
    lines = [
        "# 도형 선택(도달) 평가 요약",
        "",
        f"- 기록 {len(rows) - n_bad}회" + (f" (무효 {n_bad}회)" if n_bad else ""),
        "- 판정은 **사람이 눈으로 본 것**이다. 자동 추천은 잉크 변화 기반 참고값이라",
        "  도형 위까지 갔지만 안 문지른 시도를 '안 갔다'로 추천한다.",
        "- 다 지웠는지는 이 도구가 보지 않는다 — 어느 도형을 골랐는지만 본다.",
        "- 좌/우 표는 위치 편향 진단용이다. 한쪽만 높으면 프롬프트가 아니라 자리를",
        "  따라간 것이므로, 타겟을 좌우 절반씩 배치해 다시 볼 것.",
        "",
        "## 조건별",
        "",
        "| 조건 | n | 정답률 (95% CI) | 타겟L | 타겟R | 자동 일치도 |",
        "|---|---:|---|---|---|---:|",
    ]
    for key, s in summary.items():
        lo, hi = s["correct_ci95"]

        def _side(t):
            d = s["by_target_side"].get(t)
            return "—" if not d else f"{d['rate']:.2f} (n={d['n']})"

        agree = "—" if s["auto_agreement"] is None else f"{s['auto_agreement']:.2f}"
        lines.append(f"| {key} | {s['n_trials']} | {s['correct_rate']:.2f} ({lo:.2f}–{hi:.2f}) "
                     f"| {_side('L')} | {_side('R')} | {agree} |")
    lines += ["", "## 판정 분포", ""]
    for key, s in summary.items():
        got = ", ".join(f"{VERDICT_LABELS.get(k, k).split(' ', 1)[-1]} {v}"
                        for k, v in s["verdicts"].items())
        lines += [f"- **{key}** — {got}"]
    return "\n".join(lines) + "\n"


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", default=None, choices=("circle", "triangle", "rectangle"),
                   help="지울 대상 도형. 롤아웃 명령의 --target/--task와 같아야 한다")
    p.add_argument("--model", default="")
    p.add_argument("--condition", default="")
    p.add_argument("--trials", type=int, default=0, help="목표 시행 수 (표시용)")
    p.add_argument("--cutoff", type=float, default=45.0,
                   help="0이면 컷오프 없음. 도달만 보므로 완주 평가(120초)보다 짧게 잡는다")
    p.add_argument("--rollout-cmd", default=None,
                   help="롤아웃 1회를 도는 명령. --mode demo 로 줘야 영상이 안 남는다")
    p.add_argument("--watch-dir", type=Path, default=None,
                   help="러너가 시도 폴더를 만드는 곳 (--mode demo 기본은 records/hil)")
    p.add_argument("--dry-run", nargs="*", default=None, metavar="GLOB",
                   help="로봇 없이 UI 점검. 기존 시도 폴더를 순서대로 먹인다")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="기본값 outputs/evaluation/erase_shape_reach/<MMDD>_<model>_<condition>")
    p.add_argument("--align-cmd", default=None,
                   help="[정렬 도구 켜기] 명령 직접 지정(escape hatch). 안 주면 "
                        "--target/--board/--exclude/--dark-ratio로 자동 구성한다")
    p.add_argument("--board", type=int, nargs=4, default=list(M.DEFAULT_BOARD))
    p.add_argument("--exclude", action="append", default=None,
                   type=lambda s: tuple(int(v) for v in s.split(",")), metavar="X,Y,W,H",
                   help=f"기본값은 보드 테이프 자국 자리({M.DEFAULT_EXCLUDE[0]}) — "
                        "떼어냈으면 --exclude 0,0,0,0으로 비울 것")
    p.add_argument("--dark-ratio", type=float, default=0.72)
    p.add_argument("--selftest", action="store_true", help="합성 이미지로 판정 로직 검증")
    args = p.parse_args(argv)

    if args.selftest:
        _selftest()
        return 0
    # --selftest는 타겟이 필요 없으므로 argparse의 required 대신 여기서 본다.
    if not args.target:
        p.error("--target 은 필수입니다 (circle / triangle / rectangle)")

    args.exclude = M.resolve_exclude(args.exclude)
    if args.out_dir is None:
        stamp = datetime.now().strftime("%m%d")
        name = "_".join(x for x in (stamp, args.model, args.condition) if x)
        args.out_dir = E._repo_root() / "outputs" / "evaluation" / "erase_shape_reach" / name

    if args.dry_run is not None:
        args.rollout_cmd = None
        args.dry_run = [Path(q) for pat in args.dry_run for q in sorted(glob.glob(pat))
                        if Path(q).is_dir()]
        if not args.dry_run:
            p.error("--dry-run 에 맞는 시도 폴더가 없습니다")
    elif not args.rollout_cmd or not args.watch_dir:
        p.error("--rollout-cmd 와 --watch-dir 를 함께 주거나, --dry-run 으로 실행하세요")

    app = ReachSession(args)
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
    return 0


def _selftest():
    import tempfile

    import cv2

    tmp = Path(tempfile.mkdtemp())
    board_bgr, ink_bgr = (240, 240, 238), (60, 60, 55)
    board = (0, 0, 400, 400)

    def scene(circle=True, rect=True):
        f = np.full((400, 400, 3), board_bgr, np.uint8)
        if circle:
            cv2.circle(f, (100, 100), 40, ink_bgr, 3)
        if rect:
            cv2.rectangle(f, (250, 250), (350, 350), ink_bgr, 3)
        return f

    def ep(name, **after):
        d = tmp / name
        d.mkdir()
        cv2.imwrite(str(d / "00_reference.png"), scene())
        cv2.imwrite(str(d / "01_after.png"), scene(**after))
        return judge_reach(d, "circle", board, 0.72)

    # 타겟만 지움 -> target 추천
    r = ep("t", circle=False)
    assert r["auto_verdict"] == "target", r
    # 배치 — 원(100,100)은 왼쪽, 사각형(300,300)은 오른쪽. 400폭 보드의 중앙은 200.
    assert r["target_side"] == "L" and r["distractor_side"] == "R", r
    assert r["target_pos"] == "100,100" and r["distractor_pos"] == "300,300", r
    assert r["distractors"] == "rectangle" and r["n_distractors"] == 1, r

    # 방해만 지움 -> distractor 추천
    assert ep("d", rect=False)["auto_verdict"] == "distractor", "방해 추천 실패"
    # 둘 다 -> 사람 판단
    assert ep("b", circle=False, rect=False)["auto_verdict"] == "unclear", "둘 다 추천 실패"
    # 아무것도 -> none (접근만 했는지는 두 장으로 못 가린다)
    r_none = ep("n")
    assert r_none["auto_verdict"] == "none" and r_none["d_target"] < TOUCH, r_none

    # 프레임이 없으면 판정 실패로 떨어지되 창은 살아 있어야 한다
    (tmp / "empty").mkdir()
    r_bad = judge_reach(tmp / "empty", "circle", board, 0.72)
    assert r_bad["valid"] is False and r_bad["auto_verdict"] == "unclear", r_bad

    # CSV 왕복 — 컬럼이 안 밀리고 집계까지 되는지
    path = tmp / "trials.csv"
    E.append_rows(path, [
        dict(r, model="m", condition="c", verdict="target", agree=True, valid=True),
        dict(ep("t2", circle=False), model="m", condition="c",
             verdict="distractor", agree=False, valid=True),
    ], fields=FIELDS)
    rows = read_trials(path)
    assert len(rows) == 2 and rows[0]["path"].endswith("/t"), rows[0]
    s = summarize(rows)["m/c"]
    assert s["n_trials"] == 2 and s["correct_rate"] == 0.5, s
    assert s["by_target_side"]["L"]["n"] == 2, s          # 둘 다 타겟이 왼쪽
    assert s["auto_agreement"] == 0.5, s
    assert "정답률" in render_markdown(rows)

    print("selftest OK — 추천 4종 / 배치 / 프레임 없음 / CSV 왕복·집계 전부 통과")


if __name__ == "__main__":
    sys.exit(main())
