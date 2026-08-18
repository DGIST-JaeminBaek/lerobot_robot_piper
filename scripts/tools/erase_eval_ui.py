#!/usr/bin/env python3
"""erase_eval_ui.py — 평가 세션 운영 창. 롤아웃 실행 → 자동 채점 → 확인 → 다음.

채점 로직은 erase_eval.py 그대로다. 이 창이 더하는 건 **사람이 옆에 붙어 있을 때만
할 수 있는 일** 세 가지다:

  1. 즉시 중단 / 재개 / 건너뛰기 — 위험하거나 실험이 망가진 순간
  2. 메모 — "사람이 건드림", "마커 번짐" 같은 실험 사고 기록
  3. 시도 직후 점수 확인 — 자동 실패 유형이 틀렸으면 그 자리에서 고침

⚠️ **[중단] 버튼은 안전장치가 아니다.** 롤아웃 프로세스를 종료할 뿐이라 팔이 즉시
서지 않을 수 있다. 진짜 위험할 때는 하드웨어 비상정지를 먼저 누르고, 그 다음 이
버튼으로 세션을 정리한다. 창에도 같은 문구를 띄운다.

사용:
    # 하드웨어 없이 UI 점검 — 기존 에피소드를 롤아웃 결과인 척 순서대로 먹인다
    python scripts/tools/erase_eval_ui.py --dry-run '0802/0804/erase_the_rectangle_0804-18*'

    # 실제 세션
    python scripts/tools/erase_eval_ui.py \\
        --watch-dir /home/ugrp308/Group43/rollouts \\
        --rollout-cmd "python scripts/tools/erase_run.py --policy_path ... --confirm" \\
        --model smolvla --condition async --target rectangle \\
        --out-dir outputs/eval/0818

키: Enter 확인하고 다음 · space 시작/중단 · n 건너뛰기 · 1~7 실패 유형 · x 무효 처리
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import glob
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

sys.path.insert(0, str(Path(__file__).parent))
import erase_eval as E  # noqa: E402

FAILURE_MODES = [
    "",  # 성공
    "selectivity_failure",
    "grasp_failure",
    "approach_failure",
    "contact_ineffective",
    "premature_release",
    "repetition_loop",
    "incomplete_erase",
]
FAILURE_LABELS = {
    "": "성공 (실패 아님)",
    "selectivity_failure": "1 distractor를 건드림",
    "grasp_failure": "2 지우개를 못 잡음",
    "approach_failure": "3 잡았지만 보드에 못 닿음",
    "contact_ineffective": "4 닿았지만 안 지워짐",
    "premature_release": "5 덜 지웠는데 놓음",
    "repetition_loop": "6 같은 곳만 반복",
    "incomplete_erase": "7 그 외 미완료",
}
DETAIL_ROWS = [
    ("erased_target", "지움(target)"),
    ("erased_distractor", "지움(distractor)"),
    ("t", "t@50 / t@90"),
    ("grip", "파지 / 놓기"),
    ("termination", "종료 사유"),
    ("stall", "정체 / 회복"),
    ("occluded", "가림 비율"),
]
STAGE_LABELS = {
    "approach": "1 접근(파지)",
    "contact": "2 접촉",
    "start": "3 지우기 개시",
    "complete": "4 실질 완료",
    "selective": "5 선택성 유지",
}


def _font(base: str, size: int, bold: bool = False) -> tkfont.Font:
    """명명 폰트(TkDefaultFont/TkFixedFont)를 크기만 바꿔 복제한다.

    ("TkDefaultFont", 12, "bold")처럼 튜플의 **가족 이름 자리**에 명명 폰트를 넣으면
    안 된다. Tk는 "TkDefaultFont"를 실제 폰트 가족으로 찾다가 실패하고 한글 글리프가
    없는 폰트로 떨어져서, 해당 라벨이 통째로 □로 나온다. 폰트를 지정하지 않은 위젯만
    멀쩡했던 게 그 증거였다 (2026-08-18 실물 캡처에서 발견).

    이 환경의 Tk는 Xft 없이 X 코어 폰트 20개만 본다 — Noto CJK가 목록에 아예 없어서
    폴백이 곧 한글 손실이다. 그래서 "한글 되는 가족을 직접 지정"하는 해법도 못 쓴다.
    명명 폰트를 복제하면 Tk가 잡아둔 폴백 사슬이 그대로 유지된다.
    """
    f = tkfont.nametofont(base).copy()
    f.configure(size=size, weight="bold" if bold else "normal")
    return f


# ═══════════════════════════════════════════════════════════════════
# 롤아웃 실행 — 실제 subprocess / dry-run 둘 다 같은 인터페이스
# ═══════════════════════════════════════════════════════════════════
class Rollout:
    """롤아웃 1회를 돌리고, 끝나면 채점할 에피소드 디렉터리를 알려준다."""

    def __init__(self, cmd: str | None, watch_dir: Path | None, dry_queue: list[Path] | None):
        self.cmd = cmd
        self.watch_dir = watch_dir
        self.dry_queue = list(dry_queue or [])
        self.proc: subprocess.Popen | None = None
        self.log: queue.Queue[str] = queue.Queue()
        self._before: set[Path] = set()
        self._dry_pick: Path | None = None
        self._dry_until = 0.0

    @property
    def dry_run(self) -> bool:
        return self.cmd is None

    def start(self) -> bool:
        if self.dry_run:
            if not self.dry_queue:
                self.log.put("[DRY] 더 먹일 에피소드가 없습니다")
                return False
            self._dry_pick = self.dry_queue.pop(0)
            self._dry_until = time.monotonic() + 3.0  # 롤아웃이 도는 척
            self.log.put(f"[DRY] {self._dry_pick.name} 를 롤아웃 결과로 사용합니다 (3초)")
            return True

        self._before = self._snapshot()
        self.log.put(f"$ {self.cmd}")
        # 자식이 또 자식을 띄우므로 프로세스 그룹째 잡는다 — 안 그러면 중단해도 손자가 남는다.
        self.proc = subprocess.Popen(
            shlex.split(self.cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
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
        self.log.put(f"[STOP] {reason}")
        if self.dry_run:
            self._dry_until = 0.0
            return
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGINT)  # 러너의 park 정리를 태운다
                self.proc.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except Exception:
                    pass

    def result_dir(self) -> Path | None:
        """이번 롤아웃이 만든 에피소드 디렉터리."""
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
class EvalSession(tk.Tk):
    POLL_MS = 200
    # 파킹 이후 후처리(파킹 이동 + 영상 인코딩)에 허용하는 시간. 실측으로
    # 663프레임 인코딩이 수십 초 걸리므로 넉넉히 잡되, 무한 대기는 막는다.
    POST_PARK_TIMEOUT_S = 300.0

    def __init__(self, args):
        super().__init__()
        self.title("지우기 평가 세션")
        self.geometry("1080x760")
        self.args = args
        self.rollout = Rollout(args.rollout_cmd, args.watch_dir, args.dry_run)
        self.trial = self._next_trial()
        self.row: dict | None = None  # 확인 대기 중인 채점 결과
        # idle → running → scoring → review → idle. 버튼 상태를 읽어서 분기하면 안 된다:
        # ttk 위젯의 ["state"]는 문자열이 아니라 Tcl 객체라 == "normal" 비교가 항상 거짓이다.
        self.phase = "idle"
        self.score_q: queue.Queue[dict] = queue.Queue()
        self.started_at: float | None = None
        # [DISCONNECT] 로그를 본 시각. None이면 아직 로봇이 도는 중 (_tick 참고)
        self.parked_at: float | None = None
        # 시도 사이에 띄우는 정렬 확인 도구(block_alignment_tool). 이게 살아 있는
        # 동안은 top 카메라를 그 도구가 잡고 있으므로 다음 롤아웃을 시작하면 안 된다.
        self.align_proc: subprocess.Popen | None = None
        self._aligning = False  # 정렬 도구 실행 대기(카메라 해제 시간) 중인지
        self.aborted = False

        self.failure = tk.StringVar(value="")
        self.valid = tk.BooleanVar(value=True)
        self._build()
        self._bind_keys()
        self.after(self.POLL_MS, self._tick)

    # ------------------------------------------------------------- 레이아웃
    def _build(self) -> None:
        top = ttk.Frame(self, padding=(10, 8))
        top.pack(fill="x")
        mode = "DRY-RUN (로봇 없음)" if self.rollout.dry_run else "실행"
        ttk.Label(
            top,
            text=f"{self.args.model or '?'} / {self.args.condition or '?'} · {mode}",
            font=_font("TkDefaultFont", 12, True),
        ).pack(side="left")
        self.trial_label = ttk.Label(top, text="", font=_font("TkDefaultFont", 12))
        self.trial_label.pack(side="left", padx=16)
        self.clock = ttk.Label(top, text="", font=_font("TkDefaultFont", 12))
        self.clock.pack(side="right")

        warn = ttk.Frame(self, padding=(10, 0))
        warn.pack(fill="x")
        ttk.Label(
            warn,
            text="⚠ [중단]은 안전장치가 아닙니다. 위험하면 하드웨어 비상정지를 먼저 누르세요.",
            foreground="#b00",
        ).pack(side="left")

        bar = ttk.Frame(self, padding=(10, 8))
        bar.pack(fill="x")
        self.btn_start = ttk.Button(bar, text="시작 (space)", command=self.on_start)
        self.btn_start.pack(side="left")
        self.btn_stop = ttk.Button(bar, text="중단 (space)", command=self.on_stop, state="disabled")
        self.btn_stop.pack(side="left", padx=6)
        ttk.Button(bar, text="이번 시도 버리기 (n)", command=self.on_discard).pack(side="left", padx=6)
        ttk.Button(bar, text="세션 종료", command=self.on_close).pack(side="right")

        body = ttk.Frame(self, padding=(10, 4))
        body.pack(fill="both", expand=True)

        # 왼쪽: 점수판
        left = ttk.LabelFrame(body, text="이번 시도 결과", padding=10)
        left.pack(side="left", fill="both", expand=True)
        self.headline = ttk.Label(left, text="아직 시도 없음", font=_font("TkDefaultFont", 20, True))
        self.headline.pack(anchor="w")
        # 여러 줄을 라벨 하나에 담지 않는다. 이 환경의 Tk는 한글을 폴백 폰트로 그리는데
        # 줄 간격은 기본 폰트의 linespace(13px)로 잡혀서, 한글이 섞인 줄이 위아래로
        # 겹쳐 읽을 수 없게 된다 (2026-08-18 실물 캡처에서 발견). 한 줄짜리 라벨은
        # 각자 자기 높이를 요구하므로 겹치지 않는다 → 이름/값 2열 그리드로 쪼갠다.
        # 공백으로 열을 맞추던 것도 같이 없어져서 폭 계산에 기대지 않게 된다.
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
                                     wraplength=480, justify="left")
        self.detail_note.grid(row=len(DETAIL_ROWS), column=0, columnspan=2, sticky="w", pady=(6, 0))

        ttk.Label(left, text="단계", font=_font("TkDefaultFont", 11, True)).pack(anchor="w")
        self.stage_labels: dict[str, ttk.Label] = {}
        for key, caption in STAGE_LABELS.items():
            lab = ttk.Label(left, text=f"· {caption}")
            lab.pack(anchor="w")
            self.stage_labels[key] = lab

        ttk.Separator(left).pack(fill="x", pady=10)
        ttk.Label(left, text="실패 유형 (자동 판정 — 틀렸으면 고치세요)",
                  font=_font("TkDefaultFont", 11, True)).pack(anchor="w")
        for mode in FAILURE_MODES:
            ttk.Radiobutton(left, text=FAILURE_LABELS[mode], value=mode,
                            variable=self.failure).pack(anchor="w")

        ttk.Separator(left).pack(fill="x", pady=10)
        ttk.Checkbutton(left, text="이 에피소드는 유효함 (끄면 집계에서 빠짐)",
                        variable=self.valid).pack(anchor="w")
        ttk.Label(left, text="메모 — 사람이 건드림 / 마커 번짐 / 조명 변화 등").pack(anchor="w", pady=(8, 2))
        self.note = tk.Text(left, height=3, width=52)
        self.note.pack(fill="x")
        self.btn_confirm = ttk.Button(left, text="확인하고 다음 (Enter)",
                                      command=self.on_confirm, state="disabled")
        self.btn_confirm.pack(anchor="e", pady=8)

        # 오른쪽: 로그 + 지금까지 기록
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))
        ttk.Label(right, text="롤아웃 로그", font=_font("TkDefaultFont", 11, True)).pack(anchor="w")
        self.logbox = tk.Text(right, height=16, width=60, font=_font("TkFixedFont", 10))
        self.logbox.pack(fill="both", expand=True)
        ttk.Label(right, text="이번 세션 기록", font=_font("TkDefaultFont", 11, True)).pack(anchor="w", pady=(8, 2))
        self.tree = ttk.Treeview(right, columns=("score", "erased", "term", "fail"),
                                 show="headings", height=10)
        for col, cap, w in (("score", "점수", 60), ("erased", "지움", 70),
                            ("term", "종료", 90), ("fail", "실패유형", 150)):
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
        for i, mode in enumerate(FAILURE_MODES[1:], start=1):
            self.bind(str(i), lambda _, m=mode: self.failure.set(m))

    # ------------------------------------------------------------- 동작
    def _set_phase(self, phase: str) -> None:
        self.phase = phase
        can_start = phase == "idle" and self.align_proc is None and not getattr(self, "_aligning", False)
        self.btn_start.configure(state="normal" if can_start else "disabled")
        self.btn_stop.configure(state="normal" if phase == "running" else "disabled")
        self.btn_confirm.configure(state="normal" if phase == "review" else "disabled")

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
        self._set_status(f"시도 {self.trial} 실행 중")

    def on_stop(self) -> None:
        if self.phase != "running":
            return
        self.aborted = True
        self.rollout.stop("사람이 중단")
        self._set_status("중단했습니다 — 채점이 끝나면 확인하거나 [이번 시도 버리기]로 넘기세요")

    def on_discard(self) -> None:
        """채점하지 않고 넘긴다. 실험 사고로 시도 자체가 무의미할 때."""
        if self.phase == "running":
            self.rollout.stop("시도 버리기")
        self.row = None
        self._set_phase("idle")
        self._reset_card("버림 — 기록하지 않음")

    def on_confirm(self) -> None:
        if self.row is None:
            return
        row = dict(self.row)
        row["scored_by"] = "auto"
        row["failure_mode"] = self.failure.get()
        row["valid"] = bool(self.valid.get())
        note = self.note.get("1.0", "end").strip()
        # 자동 판정을 사람이 바꿨으면 원본을 메모에 남긴다 — 나중에 일치도를 볼 수 있다.
        if row["failure_mode"] != (self.row.get("failure_mode") or ""):
            note = (note + f" [auto={self.row.get('failure_mode') or 'success'}]").strip()
        if self.aborted:
            note = (note + " [사람이 중단]").strip()
        row["note"] = note
        # 사람이 실패 유형을 붙였으면 성공일 수 없다.
        if row["failure_mode"]:
            row["success"] = False

        self._append(row)
        self.tree.insert(
            "", "end",
            values=(f"{row.get('score', '-')}/10",
                    f"{row.get('erased_target', 0):.2f}" if row.get("erased_target") is not None else "-",
                    row.get("termination", "-"),
                    row["failure_mode"] or ("성공" if row.get("success") else "-")),
        )
        self.trial += 1
        self.row = None
        self._set_phase("idle")
        self._reset_card("기록했습니다 — [시작]으로 다음 시도")
        self._refresh_header()
        self._start_alignment_check()

    # 롤아웃 프로세스가 카메라를 닫고 실제로 나간 뒤에도, RealSense 장치가 USB
    # 레벨에서 완전히 풀리기까지 시간이 더 걸린다. 이 프로젝트에 이미 근거가
    # 있다(recording.env REALSENSE_WARMUP_S=3.0, CAMERA_POST_CONNECT_WAIT_S=2.0).
    # 그 전에 다른 프로세스가 같은 장치를 열면 librealsense가 세그폴트로 죽는다 —
    # 실물에서 확인함(2026-08-18, 정렬 도구를 확인 직후 즉시 띄우게 했더니 GUI가
    # "Segmentation fault (core dumped)"로 죽었다). 그래서 즉시 실행하지 않고
    # ALIGN_LAUNCH_DELAY_MS만큼 늦춘다.
    ALIGN_LAUNCH_DELAY_MS = 2500

    def _start_alignment_check(self) -> None:
        """다음 시도 전에 도형·지우개 위치를 눈으로 확인하는 도구를 띄운다.

        top 카메라를 이 도구가 잡으므로 롤아웃과 동시에 뜨면 안 된다. 그래서
        시도가 끝나고 기록까지 마친 뒤에만 예약하고, 실제 실행은 카메라가 완전히
        풀릴 시간(ALIGN_LAUNCH_DELAY_MS)을 준 뒤에 한다. [시작]은 그 대기
        구간부터 이미 잠근다 — 사용자가 그 사이 다음 시도를 눌러 롤아웃과
        카메라를 다시 경합시키면 안 되기 때문이다.
        """
        if not self.args.align_cmd or self.align_proc is not None:
            return
        self._aligning = True     # 대기 중에도 [시작]을 잠그기 위한 플래그
        self._set_phase("idle")
        self._log(f"[ALIGN] {self.ALIGN_LAUNCH_DELAY_MS/1000:.1f}초 뒤 정렬 확인 도구를 띄웁니다 "
                  f"— 카메라가 풀릴 시간을 준다")
        self._set_status("카메라 정리 대기 중…")
        self.after(self.ALIGN_LAUNCH_DELAY_MS, self._launch_alignment_tool)

    def _launch_alignment_tool(self) -> None:
        self._aligning = False
        try:
            self.align_proc = subprocess.Popen(
                shlex.split(self.args.align_cmd), start_new_session=True
            )
        except Exception as error:
            self._log(f"[WARN] 정렬 확인 도구 실행 실패: {error}")
            self.align_proc = None
            self._set_phase("idle")
            return
        self._set_phase("idle")   # 잠금 유지(align_proc가 안 None이면 계속 잠김)
        self._log("[ALIGN] 정렬 확인 도구를 띄웠습니다 — 닫으면 다음 시도를 시작할 수 있습니다")
        self._set_status("정렬 확인 중 — 도구 창을 닫으면 [시작]이 열립니다")

    def _finish_alignment_wait(self) -> None:
        self._aligning = False
        self._set_phase(self.phase)
        self._log("[ALIGN] 카메라 해제 완료 — [시작]으로 다음 시도")
        self._set_status("정렬 확인 완료 — [시작]으로 다음 시도")

    def on_close(self) -> None:
        if self.align_proc is not None and self.align_proc.poll() is None:
            with contextlib.suppress(Exception):
                self.align_proc.terminate()
        if self.rollout.running():
            if not messagebox.askyesno("세션 종료", "롤아웃이 돌고 있습니다. 중단하고 종료할까요?"):
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
                # 로봇 단계는 끝났고 남은 건 파킹 이동과 영상 인코딩 같은 후처리뿐이다.
                #
                # 이 시점 이후로 컷오프를 그대로 걸면 안 된다. 컷오프는 os.killpg로
                # 프로세스 그룹 전체에 SIGINT를 보내는데, 인코딩 워커도 같은 그룹이라
                # 인코딩 도중 죽으면 영상이 임시 폴더에 갇힌 채 videos/로 옮겨지지
                # 않아 채점이 'top 영상 없음'으로 실패한다. 파킹 이동 자체를 끊는 것도
                # 위험하다(팔이 중간 자세에서 힘을 잃는다).
                #
                # 그렇다고 아예 무제한으로 두면 인코딩이 걸렸을 때 세션이 영영 멈춘다.
                # 그래서 후처리 전용 예산으로 갈아탄다(POST_PARK_TIMEOUT_S).
                self.parked_at = time.monotonic()
                self._log(f"[CUTOFF] 로봇 단계 종료 — 후처리에는 "
                          f"{self.POST_PARK_TIMEOUT_S:g}초까지 기다립니다")

        # 정렬 확인 도구가 닫혔으면 카메라가 풀릴 시간을 준 뒤 [시작] 잠금을 푼다.
        # 즉시 풀면 안 된다 — 대칭적인 문제다: 롤아웃 종료 직후 정렬 도구가
        # 카메라를 열어서 세그폴트가 났던 것과 똑같이, 정렬 도구 종료 직후
        # 다음 롤아웃이 카메라를 열어도 같은 이유(RealSense가 USB 레벨에서
        # 아직 안 풀림)로 죽는다. 실물에서 실제로 겪었다(2026-08-18): 첫 번째
        # 지연 수정 이후에도 세그폴트가 재발했는데, 이번엔 정렬 도구를 닫고
        # 바로 다음 롤아웃을 시작한 시점이었다.
        if self.align_proc is not None and self.align_proc.poll() is not None:
            self.align_proc = None
            self._aligning = True  # 해제 대기 동안에도 [시작]을 잠근다
            self._set_phase(self.phase)
            self._log(f"[ALIGN] 정렬 확인 도구 종료 — 카메라 해제 대기 중 "
                      f"({self.ALIGN_LAUNCH_DELAY_MS/1000:.1f}초)")
            self._set_status("카메라 정리 대기 중…")
            self.after(self.ALIGN_LAUNCH_DELAY_MS, self._finish_alignment_wait)

        if self.phase == "running":
            elapsed = time.monotonic() - (self.started_at or time.monotonic())
            self.clock.configure(text=f"{elapsed:5.1f}s")
            if self.parked_at is None:
                # 로봇이 아직 도는 중 — 원래 컷오프로 끊는다
                if self.args.cutoff and elapsed >= self.args.cutoff:
                    self._log(f"[CUTOFF] {self.args.cutoff:g}초 초과 — 중단합니다")
                    self.rollout.stop("컷오프")
            elif time.monotonic() - self.parked_at >= self.POST_PARK_TIMEOUT_S:
                self._log(f"[CUTOFF] 후처리가 {self.POST_PARK_TIMEOUT_S:g}초를 넘었습니다 "
                          f"— 중단합니다 (녹화가 손상될 수 있음)")
                self.rollout.stop("후처리 컷오프")
            if not self.rollout.running():
                self._begin_scoring()
        elif self.phase == "scoring":
            try:
                self._scored(self.score_q.get_nowait())
            except queue.Empty:
                pass

        self.after(self.POLL_MS, self._tick)

    def _begin_scoring(self) -> None:
        ep = self.rollout.result_dir()
        if ep is None:
            self._set_phase("idle")
            self._reset_card("새 에피소드 폴더를 못 찾았습니다 — 녹화가 안 됐을 수 있습니다")
            return
        self._set_phase("scoring")
        self.headline.configure(text="채점 중…")
        self._set_status(f"{ep.name} 채점 중 — 영상 길이에 따라 10~30초")

        def work():
            try:
                row = E.score_episode(ep, self.args.target, tuple(self.args.board),
                                      self.args.dark_ratio, self.args.fps,
                                      exclude=self.args.exclude)
            except Exception as exc:
                row = {"episode": ep.name, "path": str(ep), "valid": False,
                       "note": f"{type(exc).__name__}: {exc}"}
            # 결과는 큐로만 넘긴다. 워커 스레드에서 위젯이나 after()를 건드리면
            # tkinter가 "main thread is not in main loop"로 죽는다.
            self.score_q.put(row)

        threading.Thread(target=work, daemon=True).start()

    def _scored(self, row: dict) -> None:
        row.update(model=self.args.model, condition=self.args.condition, trial=self.trial)
        self.row = row

        if not row.get("valid"):
            self.headline.configure(text="채점 실패")
            self._clear_detail()
            self.detail_note.configure(text=row.get("note", ""))
            self.valid.set(False)
            self.failure.set("")
        else:
            ok = row["success"]
            self.headline.configure(
                text=f"{'성공' if ok else '실패'}   {row['score']:g}/10   진행률 {row['task_progress']:.0f}%"
            )
            self._set_detail({
                "erased_target": f"{row['erased_target']:.3f}",
                "erased_distractor": f"{row['erased_distractor']:.3f}",
                "t": f"{_s(row['t50_s'])} / {_s(row['t90_s'])}",
                "grip": f"{_s(row['grasp_s'])} / {_s(row['release_s'])}",
                "termination": str(row["termination"]),
                "stall": f"{row['stall_events']} / {row['recovered']}"
                         f"   (사람 시연 기준 정체 2.2건)",
                "occluded": str(row["occluded_frac"]),
            })
            self.valid.set(True)
            self.failure.set(row.get("failure_mode") or "")

        for key, lab in self.stage_labels.items():
            got = row.get(f"stage_{key}")
            mark = "[O]" if got else ("[X]" if got is not None else "[ ]")
            lab.configure(text=f"{mark} {STAGE_LABELS[key]}",
                          foreground="#070" if got else ("#b00" if got is not None else "#666"))

        self._set_phase("review")
        self.btn_confirm.focus_set()
        self._set_status("확인 후 Enter — 실패 유형이 틀렸으면 먼저 고치세요")

    # ------------------------------------------------------------- 보조
    def _append(self, row: dict) -> None:
        self.args.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.args.out_dir / "episodes.csv"
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=E.FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)
        # 요약은 매번 다시 만든다 — 세션 도중에 죽어도 지금까지 것이 남는다.
        rows = E._read_csv(path)
        summary = E.summarize_rows(rows)
        (self.args.out_dir / "summary.md").write_text(
            E.render_markdown(summary, rows), encoding="utf-8"
        )
        import json

        (self.args.out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _next_trial(self) -> int:
        path = self.args.out_dir / "episodes.csv"
        if not path.exists():
            return 1
        same = [
            r for r in E._read_csv(path)
            if r.get("model") == self.args.model and r.get("condition") == self.args.condition
        ]
        return len(same) + 1

    def _set_detail(self, values: dict[str, str]) -> None:
        for key, lab in self._detail_vals.items():
            lab.configure(text=values.get(key, ""))
        self.detail_note.configure(text="")

    def _clear_detail(self) -> None:
        for lab in self._detail_vals.values():
            lab.configure(text="")
        self.detail_note.configure(text="")

    def _reset_card(self, message: str) -> None:
        self.headline.configure(text=message)
        self._clear_detail()
        self.note.delete("1.0", "end")
        self.failure.set("")
        self.valid.set(True)
        self.clock.configure(text="")
        for key, lab in self.stage_labels.items():
            lab.configure(text=f"· {STAGE_LABELS[key]}", foreground="#666")

    def _refresh_header(self) -> None:
        target = f"/{self.args.trials}" if self.args.trials else ""
        self.trial_label.configure(text=f"시도 {self.trial}{target}")

    def _log(self, line: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.logbox.insert("end", f"{stamp}  {line}\n")
        self.logbox.see("end")

    def _set_status(self, text: str) -> None:
        self.status.configure(text=text)


def _s(v) -> str:
    return "—" if v is None else f"{v:.1f}s"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="")
    p.add_argument("--condition", default="")
    p.add_argument("--target", default=None, help="지울 도형. 생략하면 폴더명에서 추론")
    p.add_argument("--trials", type=int, default=0, help="목표 시행 수 (표시용)")
    p.add_argument("--cutoff", type=float, default=60.0, help="0이면 컷오프 없음")
    p.add_argument("--rollout-cmd", default=None, help="롤아웃 1회를 도는 명령")
    p.add_argument("--watch-dir", type=Path, default=None, help="새 에피소드가 생기는 폴더")
    p.add_argument("--dry-run", nargs="*", default=None, metavar="GLOB",
                   help="로봇 없이 UI 점검. 기존 에피소드를 롤아웃 결과처럼 순서대로 먹인다")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="기본값 evaluation/<MMDD>_<model>_<condition> (리포 최상단)")
    p.add_argument("--align-cmd", default=None,
                   help="시도 확인 직후 띄울 정렬 확인 명령. 이 도구가 top 카메라를 "
                        "잡으므로 창을 닫을 때까지 [시작]이 잠긴다. 예: "
                        "'python scripts/tools/block_alignment_tool.py --shape-zones 135'")
    p.add_argument("--board", type=int, nargs=4, default=list(E.M.DEFAULT_BOARD))
    p.add_argument(
        "--exclude", action="append", default=None,
        type=lambda s: tuple(int(v) for v in s.split(",")),
        metavar="X,Y,W,H",
        help=f"기본값은 보드 테이프 자국 자리({E.M.DEFAULT_EXCLUDE[0]}) — "
             "떼어냈으면 --exclude 0,0,0,0으로 비울 것",
    )
    p.add_argument("--dark-ratio", type=float, default=0.72)
    p.add_argument("--fps", type=float, default=30.0)
    args = p.parse_args(argv)
    args.exclude = E.M.resolve_exclude(args.exclude)

    if args.out_dir is None:
        args.out_dir = E.default_out_dir(args.model, args.condition)

    if args.dry_run is not None:
        args.rollout_cmd = None
        args.dry_run = [Path(q) for pat in args.dry_run for q in sorted(glob.glob(pat))
                        if Path(q).is_dir()]
        if not args.dry_run:
            p.error("--dry-run 에 맞는 에피소드 폴더가 없습니다")
    elif not args.rollout_cmd or not args.watch_dir:
        p.error("--rollout-cmd 와 --watch-dir 를 함께 주거나, --dry-run 으로 실행하세요")

    app = EvalSession(args)
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
