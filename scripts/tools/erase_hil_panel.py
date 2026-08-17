#!/usr/bin/env python3
"""HIL 실행 중 화면에 띄우는 상태 패널 (tkinter, 항상 위).

왜 필요한가 — 상태 표시는 원래 `erase_status.LiveStatus`가 터미널에 그린다.
그런데 실행 주체가 터미널을 안 보고 있으면(예: 에이전트가 대신 띄웠거나,
nohup으로 돌렸거나, 사용자가 리더암 앞에 서 있어서 모니터의 터미널 창을
못 보는 경우) 그 표시가 사람에게 도달하지 않는다.

HIL 첫 검증은 "space를 누른 순간 팔로워가 튀는가"를 **눈으로** 보는 작업이라
깜깜이로 하면 안 된다. 그래서 X 화면에 작은 창을 띄운다.

키 조작은 `hil_clutch.KeyToggle`의 pynput 전역 리스너가 그대로 담당한다 —
이 창에 포커스가 없어도 space/q가 먹는다. 이 패널은 **표시 전용**이고,
버튼은 그 전역 토글을 대신 누르는 편의 수단이다.

    from erase_hil_panel import HilPanel
    panel = HilPanel(max_attempts=3, max_steps=940, toggle=key_toggle)
    panel.start()
    ...
    panel.on_step(payload)          # LiveStatus와 같은 인터페이스
    panel.set_measurement(result)
    panel.close()

tkinter가 없거나 DISPLAY가 없으면 조용히 비활성화된다 — 패널 때문에 실물
실행이 죽으면 안 된다.
"""

import multiprocessing as mp
import os
import queue

BG = "#1b1d22"
FG = "#e6e6e6"
DIM = "#8b8f98"
GREEN = "#3fbf6f"
AMBER = "#e0a52b"
RED = "#e2564d"
BLUE = "#4c9be8"


def _panel_process(state_q, cmd_q, max_attempts, max_steps, title):
    """패널 프로세스의 진입점. **여기가 그 프로세스의 메인 스레드다.**

    스레드가 아니라 프로세스인 이유: Tk 인터프리터는 만든 스레드에서만
    해제할 수 있는데, 보조 스레드에서 만들면 위젯 콜백 클로저가 root를
    계속 참조해서 실제 해제가 파이썬 종료 시점(=메인 스레드)으로 밀리고
    `Tcl_AsyncDelete: async handler deleted by the wrong thread`로 코어를
    덤프한다. 스레드 안에서 destroy/quit/del을 아무리 불러도 안 없어졌다.
    프로세스로 두면 Tk가 자기 메인 스레드를 갖게 되어 이 문제가 사라지고,
    최악의 경우 부모가 terminate()로 확실히 정리할 수 있다.
    """
    import tkinter as tk

    root = tk.Tk()
    root.title(title)
    root.configure(bg=BG)
    root.geometry("520x300")
    # 리더암 앞에 서서 봐야 하므로 다른 창에 가리면 안 된다
    root.attributes("-topmost", True)

    font = ("DejaVu Sans", 11)
    big = ("DejaVu Sans", 26, "bold")
    mid = ("DejaVu Sans", 15, "bold")

    def label(text, fg=FG, fnt=font, **kw):
        w = tk.Label(root, text=text, bg=BG, fg=fg, font=fnt, anchor="w", **kw)
        w.pack(fill="x", padx=14, pady=1)
        return w

    l_head = label("시도 -/-  준비", DIM)
    l_prog = label("[진행] " + "░" * 24 + "   0/0", BLUE, mid)
    l_fps = label("", DIM)
    tk.Frame(root, bg="#33363d", height=1).pack(fill="x", padx=14, pady=6)
    l_meas_cap = label("[측정] 아직 판정 전 — 시도가 끝나고 park에서 잰다", DIM)
    l_meas = label("", GREEN, big)
    l_where = label("", DIM)
    tk.Frame(root, bg="#33363d", height=1).pack(fill="x", padx=14, pady=6)
    l_align = label("", DIM)
    l_hil = label("[HIL] 정책 주행", DIM, mid)
    l_hil_sub = label("space = 개입 전환   q = 시도 중단  (창 포커스 불필요)", DIM)

    btns = tk.Frame(root, bg=BG)
    btns.pack(fill="x", padx=14, pady=8)

    # 버튼은 부모에게 명령만 보내고, 부모가 러너의 KeyToggle을 직접 뒤집는다.
    #
    # 한때 pynput Controller로 space를 합성해서 전역 리스너에 태우려 했는데
    # 안 된다 — 같은 프로세스에 Listener와 Controller가 함께 있으면 주입한
    # 이벤트가 두 번 전달돼서 한 번 클릭에 토글이 ON→OFF로 되돌아온다
    # (실측: "개입 ON" / "개입 OFF"가 클릭 한 번에 연속으로 찍혔다).
    tk.Button(btns, text="개입 전환 (space)", command=lambda: cmd_q.put("toggle"),
              bg="#2b2f36", fg=FG, relief="flat", padx=10, pady=4).pack(side="left")
    tk.Button(btns, text="시도 중단 (q)", command=lambda: cmd_q.put("abort"),
              bg="#3a2626", fg=FG, relief="flat", padx=10, pady=4).pack(side="left", padx=8)

    state = {"attempt": 0, "step": 0, "fps": 0.0, "phase": "준비",
             "intervening": False, "intervention_steps": 0,
             "erased": None, "where": None, "starved": 0, "leader_dev": None}

    def pump():
        closing = False
        try:
            while True:
                kw = state_q.get_nowait()
                if kw.pop("_close", False):
                    closing = True
                state.update(kw)
        except queue.Empty:
            pass

        s = state
        l_head.config(text=f"시도 {s['attempt']}/{max_attempts}   {s['phase']}")
        frac = s["step"] / max_steps if max_steps else 0.0
        n = int(round(min(max(frac, 0.0), 1.0) * 24))
        l_prog.config(text=f"[진행] {'█'*n}{'░'*(24-n)}  {s['step']}/{max_steps}  ({frac:.0%})")
        l_fps.config(text=f"{s['fps']:.1f} fps    큐 대기 {s['starved']}회")

        if s["erased"] is None:
            l_meas_cap.config(text="[측정] 아직 판정 전 — 시도가 끝나고 park에서 잰다")
            l_meas.config(text="")
            l_where.config(text="")
        else:
            e = s["erased"]
            col = GREEN if e >= 0.9 else (AMBER if e >= 0.5 else RED)
            l_meas_cap.config(text="[측정] park 프레임에서 실제로 잰 값")
            l_meas.config(text=f"지워짐 {e:.1%}   남음 {1-e:.1%}", fg=col)
            l_where.config(text=f"잔여 위치: {s['where']}" if s["where"] else "")

        # 개입 전 정렬 안내. 개입 중에는 의미가 없어 숨긴다(클러치라 델타만 쓴다).
        dev = s.get("leader_dev")
        if dev and not s["intervening"]:
            worst = max(dev, key=lambda k: abs(dev[k]))
            parts = " ".join(f"{k[-1]}:{v:+.0f}" for k, v in dev.items())
            col = GREEN if abs(dev[worst]) <= 10 else (AMBER if abs(dev[worst]) <= 30 else RED)
            l_align.config(text=f"리더-팔로워 편차  {parts}   (최대 {worst} {dev[worst]:+.1f})",
                           fg=col)
        else:
            l_align.config(text="")

        if s["intervening"]:
            l_hil.config(text="[HIL] ●  사람 개입 중 — 리더암이 몬다", fg=AMBER)
        else:
            l_hil.config(text="[HIL] 정책 주행", fg=DIM)
        l_hil_sub.config(text=f"개입 누적 {s['intervention_steps']}스텝   "
                              f"space = 개입 전환,  q = 시도 중단")

        if closing:
            root.after(1200, root.destroy)  # 마지막 판정값을 잠깐 보여주고 닫는다
        else:
            root.after(100, pump)

    root.after(100, pump)
    root.mainloop()


class HilPanel:
    """별도 **프로세스**로 도는 tkinter 상태 창.

    제어 루프는 위젯을 직접 안 건드리고 큐로 상태 dict만 보낸다.
    버튼 입력은 반대 방향 큐(cmd_q)로 와서 poll()에서 토글에 반영된다.
    """

    def __init__(self, max_attempts: int, max_steps: int, toggle=None,
                 title="지우기 게이트 — HIL"):
        self.max_attempts = max_attempts
        self.max_steps = max_steps
        self._toggle = toggle      # 테스트에서 직접 주입할 때만 쓴다
        self._runner = None
        self.title = title
        self._proc = None
        self._state_q = None
        self._cmd_q = None

    def available(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def start(self) -> "HilPanel":
        if not os.environ.get("DISPLAY"):
            print("[PANEL] DISPLAY가 없어 상태 창을 띄우지 않는다 (터미널 표시는 그대로).")
            return self
        try:
            import tkinter  # noqa: F401
        except ImportError:
            print("[PANEL] tkinter가 없어 상태 창을 띄우지 않는다.")
            return self
        # spawn을 쓴다. 기본 fork는 이미 CUDA/torch가 올라온 프로세스를 복제하게
        # 되는데, 자식이 CUDA를 안 건드려도 fork 후 상태가 불안정한 걸로 알려져 있다.
        # 자식이 하는 일은 tkinter뿐이라 spawn의 재import 비용이 문제되지 않는다.
        ctx = mp.get_context("spawn")
        self._state_q = ctx.Queue()
        self._cmd_q = ctx.Queue()
        self._proc = ctx.Process(
            target=_panel_process,
            args=(self._state_q, self._cmd_q, self.max_attempts, self.max_steps,
                  self.title),
            daemon=True,
        )
        self._proc.start()
        return self

    # ── 제어 루프 쪽 인터페이스 (LiveStatus와 같은 이름) ──
    def start_attempt(self, i: int):
        self._push(attempt=i, step=0, phase="추론 중")

    def on_step(self, payload: dict):
        kw = {"step": payload.get("step", 0) + 1,
              "intervening": bool(payload.get("intervention"))}
        if payload.get("leader_dev"):
            kw["leader_dev"] = payload["leader_dev"]
        self._push(**kw)
        self.poll()

    def set_phase(self, phase: str):
        self._push(phase=phase)

    def set_measurement(self, result: dict):
        if result.get("target_found"):
            self._push(erased=result.get("target_erased"),
                       where=(result.get("residual") or {}).get("where"))

    def set_fps(self, fps: float, starved: int = 0, intervention_steps: int = 0):
        self._push(fps=fps, starved=starved, intervention_steps=intervention_steps)

    def attach_runner(self, run):
        """러너가 뜬 직후 호출 — 개입 토글을 건네받는다.

        토글은 러너 루프가 --hil일 때 만들기 때문에 패널 시작 시점에는 없다.
        스레드라 잠깐 늦게 생길 수 있어 여기서는 러너만 잡아두고, poll()에서
        그때그때 읽는다.
        """
        self._runner = run

    @property
    def toggle(self):
        if self._toggle is not None:
            return self._toggle
        return getattr(self._runner, "hil_toggle", None)

    def poll(self):
        """패널 버튼 입력을 러너의 개입 토글에 반영한다.

        키보드(space/q)는 pynput 전역 리스너가 토글을 직접 건드리므로 이 경로를
        안 탄다. 이건 버튼 전용 — 양손이 리더암에 있을 때를 위한 보조 수단이다.
        """
        if self._cmd_q is None:
            return
        try:
            while True:
                cmd = self._cmd_q.get_nowait()
                t = self.toggle
                if t is None:
                    print("[PANEL] 아직 개입 토글이 없다 (--hil 없이 실행 중?)")
                    continue
                if cmd == "toggle":
                    t.active = not t.active
                    print(f"[HIL] {'개입 ON — 리더암 조작' if t.active else '개입 OFF — 정책 반환'} (패널 버튼)")
                elif cmd == "abort":
                    t.abort = True
                    print("[HIL] 시도 중단 요청 (패널 버튼)")
        except queue.Empty:
            pass

    def close(self, timeout: float = 5.0):
        """창을 닫고 프로세스가 끝나기를 기다린다. 안 끝나면 확실히 죽인다."""
        if self._proc is None:
            return
        self._push(_close=True)
        self._proc.join(timeout=timeout)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=2.0)
        self._proc = None

    # ── 내부 ──────────────────────────────────────────
    def _push(self, **kw):
        if self.available():
            try:
                self._state_q.put_nowait(kw)
            except Exception:
                pass  # 표시가 밀리는 건 실행을 멈출 이유가 아니다


def _selftest():
    """DISPLAY 없이도 죽지 않는지 — 실물 실행이 패널 때문에 멈추면 안 된다."""
    saved = os.environ.pop("DISPLAY", None)
    try:
        p = HilPanel(3, 940).start()
        assert not p.available()
        p.start_attempt(1)          # 전부 무해해야 한다
        p.on_step({"step": 1})
        p.set_measurement({"target_found": True, "target_erased": 0.5})
        p.close()
    finally:
        if saved is not None:
            os.environ["DISPLAY"] = saved
    print("selftest OK")


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        _selftest()
    else:
        # 눈으로 확인용 — 가짜 진행을 흘려보낸다
        import time

        class _T:
            active = False
            abort = False

        t = _T()
        p = HilPanel(3, 940, toggle=t).start()
        print("패널이 화면에 떴는지 확인하세요. 버튼도 눌러보세요.")
        p.start_attempt(1)
        for i in range(0, 940, 7):
            p.on_step({"step": i, "intervention": t.active})
            p.set_fps(29.4, intervention_steps=42 if t.active else 0)
            p.poll()
            if i == 700:
                p.set_measurement({"target_found": True, "target_erased": 0.257,
                                   "residual": {"where": "중간 오른쪽"}})
            time.sleep(0.05)
            if t.abort:
                break
        p.close()   # 스레드 종료까지 기다린다
