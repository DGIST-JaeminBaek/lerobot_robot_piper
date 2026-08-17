#!/usr/bin/env python3
"""추론 중 진행 상황을 터미널에 실시간으로 띄운다 (erase_run 전용 표시 계층).

★ 이 파일이 지키는 규칙 하나 — **측정한 값과 안 한 값을 섞지 않는다.**

`outputs/analysis/retreat/`의 실측(시연 29개, retreat_analysis.py):
팔이 물러나 보드가 드러나는 창은 모든 에피소드에 있지만, 그 창에서 관측되는
erased_frac은 0.00 아니면 1.00으로 갈린다(43개 창 중 부분값은 4개). 즉 **지우는
도중에는 잉크 잔량을 알 방법이 없다.** 팔이 가리고 있고, 팔이 비켜줄 때는 이미
다 지운 뒤다.

그래서 화면의 퍼센트는 두 종류이고 반드시 구분해서 표시한다:

  [측정]  지워짐 85.1%   ← 시도 경계 park 프레임에서 실제로 잰 값. 판정 근거.
  [진행]  step 512/940   ← 시도가 얼마나 진행됐나. 잉크와 무관한 시간축 진행률.

"지금 몇 % 지워졌나"를 실시간으로 보여주는 건 원리적으로 불가능하고, 억지로
dense_progress()를 띄우면 distractor 노이즈가 22배라 사람을 오도한다. 진행바를
잉크처럼 보이게 칠하지 않는 이유다.
"""

import os
import shutil
import sys
import time

# ANSI. 파이프로 넘기거나 로그로 남길 때는 전부 빈 문자열이 된다.
_TTY = sys.stdout.isatty() and os.environ.get("TERM") not in (None, "", "dumb")


def _c(code: str) -> str:
    return code if _TTY else ""


DIM, BOLD, RESET = _c("\033[2m"), _c("\033[1m"), _c("\033[0m")
GREEN, YELLOW, RED, CYAN = _c("\033[32m"), _c("\033[33m"), _c("\033[31m"), _c("\033[36m")
CLEAR_LINE = _c("\033[2K\r")
UP = lambda n: _c(f"\033[{n}A")  # noqa: E731


def bar(frac: float, width: int = 24, fill="█", empty="░") -> str:
    frac = max(0.0, min(1.0, frac))
    n = int(round(frac * width))
    return fill * n + empty * (width - n)


def pct_color(frac: float) -> str:
    """지워진 비율에 따라 색. 임계 0.9를 눈으로 바로 알 수 있게."""
    if frac >= 0.9:
        return GREEN
    if frac >= 0.5:
        return YELLOW
    return RED


class LiveStatus:
    """시도 진행 상황을 몇 줄짜리 블록으로 갱신한다.

    TTY가 아니면 커서 이동을 안 하고 일정 간격으로 한 줄씩 찍는다 — nohup이나
    파일 리다이렉트에서 ANSI 쓰레기가 쌓이는 걸 막는다.
    """

    def __init__(self, max_attempts: int, max_steps: int, hil: bool,
                 min_interval: float = 0.1, stream=None):
        self.max_attempts = max_attempts
        self.max_steps = max_steps
        self.hil = hil
        self.min_interval = min_interval
        self.out = stream or sys.stdout

        self.attempt = 0
        self.step = 0
        self.fps = 0.0
        self.lag = 0
        self.intervening = False
        self.intervention_steps = 0
        self.phase = "준비"
        # 마지막으로 **실제로 측정한** 판정값. 측정 전에는 None이고, None인 동안
        # 화면에 퍼센트를 띄우지 않는다 — 0%로 표시하면 "하나도 못 지웠다"로 읽힌다.
        self.measured_erased = None
        self.measured_residual = None

        self._last_draw = 0.0
        self._lines_drawn = 0
        self._t0 = time.perf_counter()

    # ── 상태 갱신 ──────────────────────────────────────
    def start_attempt(self, i: int):
        self.attempt, self.step, self.phase = i, 0, "추론 중"
        self._t0 = time.perf_counter()
        self.draw(force=True)

    def on_step(self, payload: dict):
        self.step = payload.get("step", self.step) + 1
        infer_ms = payload.get("infer_ms")
        if infer_ms:
            # 33ms(30fps 주기)를 넘으면 chunk가 도착할 때 이미 과거다 — 지연 스텝 수
            self.lag = int(infer_ms // (1000.0 / 30.0))
        elapsed = time.perf_counter() - self._t0
        if elapsed > 0:
            self.fps = self.step / elapsed
        if payload.get("intervention"):
            self.intervening = True
            self.intervention_steps += 1
        else:
            self.intervening = False
        self.draw()

    def set_phase(self, phase: str):
        self.phase = phase
        self.draw(force=True)

    def set_measurement(self, result: dict):
        """erase_check.check() 결과를 받는다 — 여기서만 퍼센트가 갱신된다."""
        if result.get("target_found"):
            self.measured_erased = result.get("target_erased")
            self.measured_residual = result.get("residual")
        self.draw(force=True)

    # ── 렌더 ──────────────────────────────────────────
    def _compose(self) -> list[str]:
        width = shutil.get_terminal_size((100, 24)).columns
        lines = []

        head = (f"{BOLD}시도 {self.attempt}/{self.max_attempts}{RESET}  "
                f"{DIM}{self.phase}{RESET}")
        lines.append(head)

        # 진행바 — 시간축이다. 잉크가 아니다. 라벨로 못 박는다.
        p = self.step / self.max_steps if self.max_steps else 0.0
        lines.append(
            f"  {DIM}[진행]{RESET} {CYAN}{bar(p)}{RESET} "
            f"{self.step:4d}/{self.max_steps} ({p:4.0%})   "
            f"{self.fps:4.1f} fps  {DIM}추론지연 {self.lag}스텝{RESET}"
        )

        # 측정값 — 없으면 없다고 쓴다
        if self.measured_erased is None:
            lines.append(f"  {DIM}[측정] 아직 판정 전 — 시도가 끝나고 park에서 잰다{RESET}")
        else:
            e = self.measured_erased
            col = pct_color(e)
            line = (f"  {DIM}[측정]{RESET} {col}{bar(e)}{RESET} "
                    f"지워짐 {col}{BOLD}{e:5.1%}{RESET}  남음 {1 - e:5.1%}")
            if self.measured_residual and self.measured_residual.get("where"):
                r = self.measured_residual
                line += f"   {DIM}잔여 위치: {r['where']}{RESET}"
            lines.append(line)

        if self.hil:
            if self.intervening:
                state = f"{YELLOW}{BOLD}사람 개입 중{RESET}"
            else:
                state = f"{DIM}정책 주행{RESET}"
            lines.append(f"  {DIM}[HIL]{RESET} {state}  "
                         f"{DIM}개입 누적 {self.intervention_steps}스텝  "
                         f"(space=개입 전환, q=시도 중단){RESET}")

        return [ln[:width + (len(ln) - len(_strip(ln)))] for ln in lines]

    def draw(self, force=False):
        now = time.perf_counter()
        if not force and now - self._last_draw < self.min_interval:
            return
        self._last_draw = now
        lines = self._compose()

        if not _TTY:
            # 커서 이동 없이 — 너무 자주 찍으면 로그가 못 쓰게 되므로 1초에 한 번
            if force or now - getattr(self, "_last_plain", 0) > 1.0:
                self._last_plain = now
                print(" | ".join(_strip(ln).strip() for ln in lines), file=self.out,
                      flush=True)
            return

        if self._lines_drawn:
            self.out.write(UP(self._lines_drawn))
        for ln in lines:
            self.out.write(CLEAR_LINE + ln + "\n")
        self.out.flush()
        self._lines_drawn = len(lines)

    def finish(self):
        """블록을 남겨두고 커서를 아래로 — 다음 출력이 덮어쓰지 않게."""
        self._lines_drawn = 0
        if _TTY:
            self.out.write("\n")
            self.out.flush()


def _strip(s: str) -> str:
    """ANSI 제거 — 폭 계산과 비-TTY 출력용."""
    out, i = [], 0
    while i < len(s):
        if s[i] == "\033":
            while i < len(s) and s[i] not in "mK":
                i += 1
            i += 1
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def format_residual(res: dict | None) -> str:
    """판정 후 사람이 읽을 잔여 요약 한 줄."""
    if not res:
        return "잔여 없음"
    parts = [f"도형 {res.get('shape', '?')}"]
    if res.get("where"):
        parts.append(f"주로 {res['where']}")
    if res.get("centroid"):
        cx, cy = res["centroid"]
        parts.append(f"무게중심 ({cx:.2f}, {cy:.2f})")
    parts.append(f"잔여 잉크 {res.get('ink_frac', 0):.3%}")
    parts.append(f"퍼짐 {res.get('spread_px', 0):.0f}px")
    return " · ".join(parts)


def format_grid(res: dict | None) -> list[str]:
    """3x3 잔여 분포를 눈으로 보는 작은 표. 어느 구석이 안 닦였는지 바로 보인다."""
    if not res or not res.get("grid"):
        return []
    blocks = " ░▒▓█"
    rows = []
    for r, row in enumerate(res["grid"]):
        cells = "".join(blocks[min(int(v * len(blocks) * 2.5), len(blocks) - 1)] * 2
                        for v in row)
        rows.append(f"      {cells}  {DIM}{_ROWN[r]}{RESET}")
    return rows


_ROWN = ("위", "중간", "아래")


def _selftest():
    import io

    s = LiveStatus(max_attempts=3, max_steps=100, hil=True, stream=io.StringIO())
    s.start_attempt(1)
    for i in range(0, 100, 10):
        s.on_step({"step": i, "infer_ms": 45.0, "intervention": i > 60})
    # 측정 전에는 퍼센트를 띄우지 않는다 — 0%로 보이면 '못 지웠다'로 오독된다
    assert any("아직 판정 전" in _strip(l) for l in s._compose()), s._compose()

    s.set_measurement({"target_found": True, "target_erased": 0.8513,
                       "residual": {"shape": "triangle#0", "where": "아래 왼쪽",
                                    "centroid": (0.2, 0.8), "ink_frac": 0.012,
                                    "spread_px": 14.0,
                                    "grid": [[0, 0, 0], [0, .1, 0], [.7, .2, 0]]}})
    body = "\n".join(_strip(l) for l in s._compose())
    assert "85.1%" in body and "14.9%" in body, body
    assert "아래 왼쪽" in body, body
    assert "개입" in body, body

    # 진행바는 시간축이라 측정값(85.1%)과 독립이어야 한다 — 마지막 step 90 -> 91
    assert "91/100" in body and " 91%" in body, body

    assert "아래 왼쪽" in format_residual({"shape": "t#0", "where": "아래 왼쪽",
                                          "centroid": (0.2, 0.8), "ink_frac": 0.01,
                                          "spread_px": 3})
    assert len(format_grid({"grid": [[0, 0, 0], [0, 1, 0], [0, 0, 0]]})) == 3
    assert format_grid(None) == []

    # 비-TTY에서 ANSI가 새지 않는지
    assert "\033" not in _strip("\033[1mabc\033[0m")
    print("selftest OK")


if __name__ == "__main__":
    _selftest()
