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
    python scripts/tasks/erase_shape/evaluation/erase_eval_ui.py --dry-run '0802/0804/erase_the_rectangle_0804-18*'

    # 실제 세션
    python scripts/tasks/erase_shape/evaluation/erase_eval_ui.py \\
        --watch-dir /home/ugrp308/Group43/rollouts \\
        --rollout-cmd "python scripts/tasks/erase_shape/runtime/erase_run.py --policy_path ... --confirm" \\
        --model smolvla --condition async --target rectangle \\
        --out-dir outputs/evaluation/erase_shape/0818_smolvla_async

키: Enter 확인하고 다음 · space 시작/중단 · n 건너뛰기 · 1~7 실패 유형 · x 무효 처리
"""

from __future__ import annotations

import argparse
import contextlib
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

ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "analysis"

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
    # 방해 도형이 보드에 있을 때만 채운다. 없을 때 0.000을 찍으면 "안 건드렸다"로
    # 읽히지만 실제로는 잴 대상이 없었던 것이라 정반대의 오해를 만든다.
    ("erased_distractor", "지움(방해)"),
    # 어느 쪽에 뭐가 있었나. 선택성이 프롬프트 이해인지 위치 편향인지는 이게 있어야 갈린다.
    ("layout", "보드 배치"),
    ("grip", "파지 / 놓기"),
    ("termination", "종료 사유"),
    ("frame_source", "판정 프레임"),
]


def _font(base: str, size: int, bold: bool = False) -> tkfont.Font:
    """명명 폰트(TkDefaultFont/TkFixedFont)를 크기만 바꿔 복제한다.

    conda ugrp 환경의 기본 Tk는 Xft/fontconfig 없이 빌드되어 한글 글리프가 있는
    글꼴이 아예 없다 — 이 상태로는 font=를 명시(명명 폰트를 복제해 크기만 바꾸는
    것도 포함)하는 순간 폴백이 깨져 한글 라벨이 통째로 □로 나온다 (2026-08-18
    실물 캡처에서 발견). 진짜 고정은 이 파일이 아니라 실행 방법 쪽 —
    teleop_ui.py와 마찬가지로 GUI 프로세스에 시스템 Tcl/Tk(Xft 있음, Noto Sans
    CJK KR 인식)를 LD_PRELOAD로 얹어서 실행해야 한다(scripts/13__eval_session.sh,
    docs/operations.md §7 "Teleop UI 한글 글꼴" 참고). 시스템 Tk 위에서는 이렇게
    명명 폰트를 복제해도 fontconfig가 한글 대체 글꼴을 정상적으로 물어준다.
    """
    f = tkfont.nametofont(base).copy()
    f.configure(size=size, weight="bold" if bold else "normal")
    return f


# ═══════════════════════════════════════════════════════════════════
# 롤아웃 실행 — 실제 subprocess / dry-run 둘 다 같은 인터페이스
# ═══════════════════════════════════════════════════════════════════
class Rollout:
    """롤아웃 1회를 돌리고, 끝나면 채점할 에피소드 디렉터리를 알려준다."""

    # SIGINT를 받은 러너가 정리를 끝낼 때까지 기다려주는 시간. 정리는 파킹 이동
    # (최대 ~15초)과 영상 인코딩(실측 2026-08-21: 848~1187프레임에 40~75초)으로
    # 이뤄진다. 예전에는 SIGINT 직후 10초만 기다리고 SIGKILL을 보냈는데, 그러면
    # 사람이 [중단]을 누를 때마다 거의 항상 인코딩 도중에 죽어서 videos/도 data/도
    # 없는 껍데기 폴더만 남았다(실물 2026-08-21: images/ PNG 1068장만 남고
    # meta/info.json은 total_episodes=0 — 그 시도는 통째로 못 쓰게 됐다).
    # on_stop()의 안내 문구("채점이 끝나면 확인하거나…")가 원래 의도였으니,
    # 정리를 끝까지 기다렸다가 채점으로 넘긴다.
    GRACEFUL_STOP_S = 180.0

    def __init__(self, cmd: str | None, watch_dir: Path | None, dry_queue: list[Path] | None):
        self.cmd = cmd
        self.watch_dir = watch_dir
        self.dry_queue = list(dry_queue or [])
        self.proc: subprocess.Popen | None = None
        self.log: queue.Queue[str] = queue.Queue()
        self._before: set[Path] = set()
        self._dry_pick: Path | None = None
        self._dry_until = 0.0
        # SIGINT를 보낸 시각 + GRACEFUL_STOP_S. None이면 중단 요청이 없는 상태.
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
        """SIGINT만 보내고 즉시 돌아온다 — 여기서 기다리면 GUI가 그만큼 얼어붙는다.

        실제 종료 여부는 enforce_stop_deadline()이 _tick에서 폴링으로 확인한다.
        """
        if self._kill_deadline is not None:
            return  # 이미 중단 요청이 나갔다. SIGINT를 또 보내면 정리를 끊는다.
        self.log.put(f"[STOP] {reason}")
        if self.dry_run:
            self._dry_until = 0.0
            return
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGINT)  # 러너의 park 정리를 태운다
            except Exception:
                pass
            self._kill_deadline = time.monotonic() + self.GRACEFUL_STOP_S
            self.log.put(f"[STOP] 정리(파킹·영상 인코딩)를 최대 "
                         f"{self.GRACEFUL_STOP_S:g}초까지 기다립니다")

    def enforce_stop_deadline(self) -> None:
        """정리가 GRACEFUL_STOP_S를 넘기면 그때만 강제 종료한다(_tick에서 매번 호출)."""
        if self._kill_deadline is None or self.proc is None:
            return
        if self.proc.poll() is not None:      # 스스로 정리를 마치고 끝났다
            self._kill_deadline = None
            return
        if time.monotonic() >= self._kill_deadline:
            self.log.put("[STOP] 정리가 끝나지 않아 강제 종료합니다 — 녹화가 손상될 수 있습니다")
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except Exception:
                pass
            self._kill_deadline = None

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
        self.geometry("1400x900")
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
        self._align_after_id: str | None = None  # 대기 중 예약된 after() id — 토글로 취소용
        # 러너 생존 여부가 바뀌는 순간을 잡아 버튼 상태를 다시 계산하기 위한 값 (_tick 참고)
        self._rollout_was_busy = False
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
        self.btn_align = ttk.Button(bar, text="정렬 도구 켜기", command=self.on_toggle_align)
        self.btn_align.pack(side="left", padx=6)
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

        ttk.Label(left, text="실패 유형 (해당하면 직접 고르세요)",
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
        self.tree = ttk.Treeview(right, columns=("erased", "term", "fail"),
                                 show="headings", height=10)
        for col, cap, w in (("erased", "지움", 70),
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
        align_active = self.align_proc is not None or getattr(self, "_aligning", False)
        # 러너가 아직 살아 있으면(중단 후 파킹·인코딩 중) 카메라를 못 넘겨받는다.
        can_start = phase == "idle" and not align_active and not self.rollout.running()
        self.btn_start.configure(state="normal" if can_start else "disabled")
        self.btn_stop.configure(state="normal" if phase == "running" else "disabled")
        self.btn_confirm.configure(state="normal" if phase == "review" else "disabled")
        # 두 프로세스가 top 카메라를 동시에 열면 세그폴트난다(2026-08-18). 그래서
        # 롤아웃이 카메라를 쥐고 있는 동안은 정렬 도구를 막는다. 다만 러너는
        # [DISCONNECT](=parked_at) 시점에 로봇과 카메라를 이미 놓는다 — 그 뒤로 남은
        # 영상 인코딩(실측 40~75초) 동안 카메라는 놀고 있으므로, 그 시간에 다음 시도
        # 도형을 미리 그려둘 수 있게 열어준다.
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
            values=(f"{row.get('erased_target', 0):.2f}" if row.get("erased_target") is not None else "-",
                    row.get("termination", "-"),
                    row["failure_mode"] or ("성공" if row.get("success") else "-")),
        )
        self.trial += 1
        self.row = None
        self._set_phase("idle")
        self._reset_card("기록했습니다 — [시작]으로 다음 시도")
        self._refresh_header()

    # 롤아웃 프로세스가 카메라를 닫고 실제로 나간 뒤에도, RealSense 장치가 USB
    # 레벨에서 완전히 풀리기까지 시간이 더 걸린다. 이 프로젝트에 이미 근거가
    # 있다(recording.env REALSENSE_WARMUP_S=3.0, CAMERA_POST_CONNECT_WAIT_S=2.0).
    # 그 전에 다른 프로세스가 같은 장치를 열면 librealsense가 세그폴트로 죽는다 —
    # 실물에서 확인함(2026-08-18, 정렬 도구를 확인 직후 즉시 띄우게 했더니 GUI가
    # "Segmentation fault (core dumped)"로 죽었다). 그래서 즉시 실행하지 않고
    # ALIGN_LAUNCH_DELAY_MS만큼 늦춘다.
    ALIGN_LAUNCH_DELAY_MS = 2500

    def _build_align_argv(self) -> list[str]:
        """block_alignment_tool.py 인자를 이 GUI의 args에서 그대로 만들어 붙인다.

        --align-cmd로 사용자가 전체 명령을 직접 줬으면 그쪽을 우선한다(escape hatch) —
        여기는 그게 없을 때만 쓰는 기본 자동 구성이다. --target을 그대로 넘겨서 정렬
        도구가 target 도형별 박스 크기 + 가이드를 그리게 한다(315 zone-set만 지원).
        박스 크기는 --size-source global(6개 zone 합친 도형별 median, 표본 105개) —
        zone별로 나누면 일부 조합 표본이 10~11개까지 줄어 세션 편차에 흔들린다
        (2026-08-21 실측: top-left/circle이 0811/0805 세션 편차로 대표성이 떨어짐).
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
            # align_proc은 여기서 None으로 만들지 않는다 — _tick()의 기존 poll 루프가
            # 종료를 감지해서 카메라 해제 대기(ALIGN_LAUNCH_DELAY_MS)까지 그대로
            # 처리하게 둔다(이미 검증된 경로 재사용, on_close()와 같은 패턴).
        elif self._aligning:
            if self._align_after_id is not None:
                self.after_cancel(self._align_after_id)
                self._align_after_id = None
            self._aligning = False
            self._log("[ALIGN] 실행 대기를 취소했습니다")
            self._set_phase(self.phase)
            self._set_status("정렬 도구 대기 취소")
        else:
            self._start_alignment_check()

    def _start_alignment_check(self) -> None:
        """정렬·도형 확인 도구를 띄운다(토글 버튼에서 호출).

        top 카메라를 이 도구가 잡으므로 롤아웃과 동시에 뜨면 안 된다. 그래서 실제
        실행은 카메라가 완전히 풀릴 시간(ALIGN_LAUNCH_DELAY_MS)을 준 뒤에 한다.
        [시작]은 그 대기 구간부터 이미 잠근다 — 사용자가 그 사이 다음 시도를 눌러
        롤아웃과 카메라를 다시 경합시키면 안 되기 때문이다.
        """
        if self.align_proc is not None or self._aligning:
            return
        self._aligning = True     # 대기 중에도 [시작]을 잠그기 위한 플래그
        self._set_phase(self.phase)
        self._log(f"[ALIGN] {self.ALIGN_LAUNCH_DELAY_MS/1000:.1f}초 뒤 정렬 확인 도구를 띄웁니다 "
                  f"— 카메라가 풀릴 시간을 준다")
        self._set_status("카메라 정리 대기 중…")
        self._align_after_id = self.after(self.ALIGN_LAUNCH_DELAY_MS, self._launch_alignment_tool)

    def _launch_alignment_tool(self) -> None:
        self._aligning = False
        self._align_after_id = None
        argv = shlex.split(self.args.align_cmd) if self.args.align_cmd else self._build_align_argv()
        try:
            self.align_proc = subprocess.Popen(argv, start_new_session=True)
        except Exception as error:
            self._log(f"[WARN] 정렬 확인 도구 실행 실패: {error}")
            self.align_proc = None
            self._set_phase(self.phase)
            return
        self._set_phase(self.phase)   # 잠금 유지(align_proc가 안 None이면 계속 잠김)
        self._log("[ALIGN] 정렬 확인 도구를 띄웠습니다 — [정렬 도구 끄기]나 창을 닫으면 풀립니다")
        self._set_status("정렬 확인 중 — 도구 창을 닫으면 [시작]이 열립니다")

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
                "창은 바로 닫히지만 러너는 백그라운드에서 파킹과 영상 저장을 끝내고 "
                "스스로 종료합니다(최대 수십 초). 그동안 카메라를 쥐고 있으니 다른 "
                "카메라 도구를 바로 띄우지 마세요.",
            ):
                return
            # 여기서 기다렸다 죽이면 저장이 깨진다 — SIGINT만 보내고 창을 닫는다.
            # 러너는 start_new_session이라 창이 닫혀도 정리를 마저 끝낸다.
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
                # 카메라가 풀렸으니 정렬 도구를 열어준다(_set_phase의 camera_free 참고).
                # 남은 영상 인코딩 동안 다음 시도 도형을 미리 그릴 수 있다.
                self._set_phase(self.phase)
                self._log("[ALIGN] 카메라가 풀렸습니다 — 인코딩 동안 정렬 도구를 쓸 수 있습니다")

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

        # 중단 요청이 나갔으면 정리가 끝났는지(또는 너무 오래 걸리는지) 매 tick 확인한다.
        # stop() 자신이 기다리면 그동안 GUI가 멈추므로 여기서 폴링한다. phase와 무관하게
        # 돌려야 한다 — [이번 시도 버리기]는 phase를 idle로 되돌리기 때문이다.
        self.rollout.enforce_stop_deadline()
        # 중단·버리기 뒤에도 러너는 파킹·인코딩을 끝낼 때까지 카메라를 쥐고 있다.
        # 그 사이 [시작]을 누르면 다음 롤아웃이 같은 RealSense를 열어 세그폴트가 난다
        # (정렬 도구에서 이미 겪은 것과 같은 원인). 살아 있는 동안은 [시작]을 잠그고,
        # 끝나는 순간 버튼 상태를 다시 계산한다.
        busy = self.rollout.running()
        if busy != self._rollout_was_busy:
            self._rollout_was_busy = busy
            self._set_phase(self.phase)

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
            head = f"{'성공' if ok else '실패'}   지움 {row['erased_target']:.0%}"
            detail = {
                "erased_target": f"{row['erased_target']:.3f}",
                "grip": f"{_s(row['grasp_s'])} / {_s(row['release_s'])}",
                "termination": str(row["termination"]),
                "frame_source": str(row["frame_source"]),
            }
            if row.get("target_side"):
                parts = [f"{row['target']}@{row['target_side']} (타겟)"]
                parts += [f"{l}@{s}" for l, s in zip(row["distractors"].split("|"),
                                                     row["distractor_side"].split("|")) if l]
                detail["layout"] = "  ".join(parts)
            if row.get("n_distractors"):
                # 실패 사유가 "덜 지웠다"인지 "엉뚱한 걸 지웠다"인지 헤드라인에서 갈린다.
                # 지움 95%인데 실패로 뜨는 이유를 사람이 즉시 알 수 있어야 한다.
                sel_ok = row["selective_ok"]
                detail["erased_distractor"] = (
                    f"{row['erased_distractor']:.3f}  [{row['distractors']}]  "
                    f"{'OK' if sel_ok else 'NG'}"
                )
                if not sel_ok:
                    head += "   ⚠ 방해 도형을 지움"
            self.headline.configure(text=head)
            self._set_detail(detail)
            self.valid.set(True)
            self.failure.set(row.get("failure_mode") or "")

        self._set_phase("review")
        self.btn_confirm.focus_set()
        self._set_status("확인 후 Enter — 실패했으면 실패 유형을 직접 고르세요")

    # ------------------------------------------------------------- 보조
    def _append(self, row: dict) -> None:
        self.args.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.args.out_dir / "episodes.csv"
        # 컬럼이 늘어난 뒤 옛 세션을 이어서 채점하면 값이 밀린다 — 확장까지 처리하는
        # 공용 헬퍼를 쓴다(erase_eval.append_rows).
        E.append_rows(path, [row])
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
                   help="기본값 outputs/evaluation/erase_shape/<MMDD>_<model>_<condition>")
    p.add_argument("--align-cmd", default=None,
                   help="[정렬 도구 켜기] 버튼으로 띄울 명령을 직접 지정(escape hatch). "
                        "안 주면 이 GUI의 --target/--board/--exclude/--dark-ratio로 "
                        "block_alignment_tool.py --shape-zones 315를 자동 구성한다. "
                        "이 도구가 top 카메라를 잡으므로 켜져 있는 동안 [시작]이 잠긴다")
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
