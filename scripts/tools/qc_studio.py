#!/usr/bin/env python3
"""Review a folder of recordings and confirm each episode's trim range.

    python scripts/tools/qc_studio.py                          # UI on records/local
    python scripts/tools/qc_studio.py --folder records/0727/erase_the_circle
    python scripts/tools/qc_studio.py --report                 # no window, print the table

The window shows one episode at a time: the proposed start and end frame as
actual video stills, the traffic-light verdict from ``qc_core``, and sliders to
move either boundary. Confirming writes to a review file of its own -- the
recordings and any existing manifest are only ever read.

The end boundary can hang off either end of the gripper release -- the frame it
starts opening, or the frame it finishes and the eraser is down. The radio above
the trace switches between them and re-proposes every episode not yet confirmed;
``--end-event`` sets the one the window opens with.

Half-built recordings (the retry path leaves a folder without ``videos/``) are
collected into their own dialog and can be moved to a quarantine folder after
a confirmation. Nothing is deleted.

Keys: Enter confirm & next · ←/→ previous/next episode · a/d nudge start ·
j/l nudge end · r reset to auto · x exclude.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from autofill_frame_ranges import load_action
from episode_segmentation import DEFAULT_END_EVENT, END_EVENTS, END_MARGIN
from qc_core import GREEN, RED, YELLOW, Report, inspect_folder


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FOLDER = REPO_ROOT / "records/local"
DEFAULT_QUARANTINE = REPO_ROOT / "records/_quarantine"
PREVIEW_WIDTH = 430
TRACE_HEIGHT = 210

LEVEL_COLOR = {GREEN: "#2e7d32", YELLOW: "#e08800", RED: "#c62828"}
LEVEL_ROW = {GREEN: "#e8f5e9", YELLOW: "#fff8e1", RED: "#ffebee"}
LEVEL_LABEL = {GREEN: "● 정상", YELLOW: "▲ 확인 필요", RED: "■ 사용 불가"}
END_EVENT_LABEL = {
    "release_start": "그리퍼가 벌어지기 시작할 때",
    "release_done": "지우개를 놓고 다 벌어졌을 때",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", type=Path, default=DEFAULT_FOLDER, help=f"(default: {DEFAULT_FOLDER})")
    parser.add_argument("--output", type=Path, default=None, help="Review JSON (default: configs/qc_review_<folder>.json)")
    parser.add_argument("--quarantine", type=Path, default=DEFAULT_QUARANTINE)
    parser.add_argument("--report", action="store_true", help="Print the table and exit, no window.")
    parser.add_argument("--start-margin", type=int, default=22)
    parser.add_argument("--end-margin", type=int, default=None,
                        help=f"Frames dropped before the end event (default: per event, {END_MARGIN}).")
    parser.add_argument("--end-event", choices=END_EVENTS, default=DEFAULT_END_EVENT,
                        help=f"Which end of the gripper release closes the episode (default: {DEFAULT_END_EVENT}).")
    parser.add_argument("--round-start", type=int, default=10)
    return parser.parse_args()


def print_report(reports: list[Report], end_event: str = DEFAULT_END_EVENT) -> None:
    counts = {level: sum(r.level == level for r in reports) for level in (GREEN, YELLOW, RED)}
    print(f"종료 기준: {end_event} — {END_EVENT_LABEL[end_event]}\n")
    print(f"{'status':<7}{'episode':<32}{'frames':>7}{'range':>14}{'kept':>8}{'erased':>8}  notes")
    print("-" * 110)
    for report in reports:
        span = f"[{report.start}, {report.end})" if report.start is not None else "-"
        kept = f"{report.kept_seconds:.1f}s" if report.kept_frames else "-"
        erased = "-" if report.erased is None else f"{report.erased * 100:.0f}%"
        print(f"{report.level.upper():<7}{report.name:<32}{report.total_frames:>7}{span:>14}{kept:>8}"
              f"{erased:>8}  {'; '.join(report.reasons + report.notes)}")
    print(f"\n{len(reports)} folders — {counts[GREEN]} green, {counts[YELLOW]} yellow, {counts[RED]} red "
          f"({sum(r.broken for r in reports)} broken)")


class Studio(tk.Tk):
    def __init__(self, reports: list[Report], output: Path, quarantine: Path,
                 end_event: str = DEFAULT_END_EVENT, end_margin: int | None = None):
        super().__init__()
        self.title("Piper QC Studio")
        self.geometry("1330x950")

        self.reports = [r for r in reports if not r.broken]
        self.broken = [r for r in reports if r.broken]
        self.output = output
        self.quarantine = quarantine
        self.end_margin_override = end_margin
        self.index = 0
        self.captures: dict[str, cv2.VideoCapture] = {}
        self.photos: dict[str, ImageTk.PhotoImage] = {}
        self.actions: dict[str, np.ndarray | None] = {}
        self.end_event = tk.StringVar(value=end_event)
        # Derived rather than taken from ``report.end`` so the window is correct
        # even if the reports were inspected under the other end event.
        self.state: dict[str, dict] = {
            r.name: {"start": r.start, "end": self.auto_end(r), "confirmed": False, "excluded": r.level == RED}
            for r in self.reports
        }

        self._build()
        if self.reports:
            self.show(0)
        self._bind_keys()

    # ------------------------------------------------------------ end event

    def release_of(self, report: Report, event: str | None = None) -> int | None:
        event = event or self.end_event.get()
        return report.gripper_release if event == "release_start" else report.gripper_release_done

    def auto_end(self, report: Report) -> int:
        """The proposed end for the currently selected event.

        Recomputed from the two release frames the report already carries, so
        switching the radio costs nothing -- no parquet or video is re-read.
        """
        event = self.end_event.get()
        release = self.release_of(report, event)
        if release is None:
            return report.end  # qc_core already fell back to the last sustained motion
        margin = self.end_margin_override if self.end_margin_override is not None else END_MARGIN[event]
        return max(0, min(release - margin, report.total_frames))

    def on_end_event(self) -> None:
        """Re-propose the end of every episode the operator has not confirmed yet."""
        touched = 0
        for report in self.reports:
            state = self.state[report.name]
            if state["confirmed"]:
                continue
            state["end"] = self.auto_end(report)
            touched += 1
        self.status.configure(text=f"종료 기준: {END_EVENT_LABEL[self.end_event.get()]} — "
                                   f"확인 전 {touched}개 에피소드의 종료 프레임을 다시 계산했습니다.")
        self.show(self.index)

    # ---------------------------------------------------------------- layout

    def _build(self) -> None:
        toolbar = ttk.Frame(self, padding=(10, 8))
        toolbar.pack(fill="x")
        counts = {level: sum(r.level == level for r in self.reports) for level in (GREEN, YELLOW, RED)}
        summary = f"정상 {counts[GREEN]}   확인 필요 {counts[YELLOW]}   사용 불가 {counts[RED]}"
        ttk.Label(toolbar, text=summary, font=("TkDefaultFont", 11, "bold")).pack(side="left")
        ttk.Button(toolbar, text="저장", command=self.save).pack(side="right")
        if self.broken:
            ttk.Button(toolbar, text=f"녹화 실패 폴더 {len(self.broken)}개 정리…",
                       command=self.open_broken).pack(side="right", padx=8)

        body = ttk.Frame(self, padding=(10, 0))
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body)
        left.pack(side="left", fill="y")
        self.tree = ttk.Treeview(left, columns=("status", "kept"), show="tree headings", height=30, selectmode="browse")
        self.tree.heading("#0", text="에피소드")
        self.tree.heading("status", text="상태")
        self.tree.heading("kept", text="길이")
        self.tree.column("#0", width=250)
        self.tree.column("status", width=90, anchor="center")
        self.tree.column("kept", width=60, anchor="e")
        for level, color in LEVEL_ROW.items():
            self.tree.tag_configure(level, background=color)
        self.tree.tag_configure("confirmed", foreground="#1565c0")
        self.tree.pack(side="left", fill="y")
        ttk.Scrollbar(left, orient="vertical", command=self.tree.yview).pack(side="left", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self.on_select)
        self.refresh_tree()

        right = ttk.Frame(body, padding=(12, 0))
        right.pack(side="left", fill="both", expand=True)

        self.title_label = ttk.Label(right, text="", font=("TkDefaultFont", 13, "bold"))
        self.title_label.pack(anchor="w")
        self.verdict = ttk.Label(right, text="", font=("TkDefaultFont", 10), wraplength=680, justify="left")
        self.verdict.pack(anchor="w", pady=(2, 8))

        frames = ttk.Frame(right)
        frames.pack(fill="x")
        self.previews: dict[str, ttk.Label] = {}
        self.frame_labels: dict[str, ttk.Label] = {}
        self.sliders: dict[str, ttk.Scale] = {}
        for column, (key, caption) in enumerate((("start", "시작 프레임"), ("end", "종료 프레임"))):
            box = ttk.LabelFrame(frames, text=caption, padding=6)
            box.grid(row=0, column=column, padx=(0, 10), sticky="n")
            preview = ttk.Label(box)
            preview.pack()
            self.previews[key] = preview
            self.frame_labels[key] = ttk.Label(box, text="", font=("TkDefaultFont", 11, "bold"))
            self.frame_labels[key].pack(pady=(6, 2))
            slider = ttk.Scale(box, from_=0, to=100, orient="horizontal", length=PREVIEW_WIDTH,
                               command=lambda value, k=key: self.on_slide(k, value))
            slider.pack()
            self.sliders[key] = slider
            nudges = ttk.Frame(box)
            nudges.pack(pady=4)
            for delta in (-10, -1, +1, +10):
                ttk.Button(nudges, text=f"{delta:+d}", width=4,
                           command=lambda d=delta, k=key: self.nudge(k, d)).pack(side="left", padx=2)

        chooser = ttk.LabelFrame(right, text="종료 기준", padding=(8, 4))
        chooser.pack(fill="x", pady=(10, 0))
        for event in END_EVENTS:
            ttk.Radiobutton(chooser, text=END_EVENT_LABEL[event], value=event,
                            variable=self.end_event, command=self.on_end_event).pack(side="left", padx=(0, 18))
        ttk.Label(chooser, text="확인 완료한 에피소드는 그대로 둡니다", foreground="#777").pack(side="left")

        trace_box = ttk.LabelFrame(right, text="관절 움직임 (위) · 그리퍼 (아래)  —  클릭하면 가까운 경계가 이동합니다",
                                   padding=4)
        trace_box.pack(fill="both", expand=True, pady=(10, 0))
        self.trace = tk.Canvas(trace_box, height=TRACE_HEIGHT, background="white", highlightthickness=0)
        self.trace.pack(fill="both", expand=True)
        self.trace.bind("<Button-1>", self.on_trace_click)
        self.trace.bind("<B1-Motion>", self.on_trace_click)
        self.trace.bind("<Configure>", lambda _: self.draw_trace())

        actions = ttk.Frame(right, padding=(0, 12))
        actions.pack(fill="x")
        self.exclude_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(actions, text="이 에피소드 제외 (x)", variable=self.exclude_var,
                        command=self.on_exclude).pack(side="left")
        ttk.Button(actions, text="자동값 복원 (r)", command=self.reset_auto).pack(side="left", padx=10)
        ttk.Button(actions, text="확인하고 다음 (Enter)", command=self.confirm_next).pack(side="right")

        self.status = ttk.Label(self, text=f"검토 결과 저장 위치: {self.output}", padding=(10, 6),
                                relief="sunken", anchor="w")
        self.status.pack(fill="x", side="bottom")

    def _bind_keys(self) -> None:
        self.bind("<Return>", lambda _: self.confirm_next())
        self.bind("<Right>", lambda _: self.show(self.index + 1))
        self.bind("<Left>", lambda _: self.show(self.index - 1))
        self.bind("a", lambda _: self.nudge("start", -1))
        self.bind("d", lambda _: self.nudge("start", +1))
        self.bind("j", lambda _: self.nudge("end", -1))
        self.bind("l", lambda _: self.nudge("end", +1))
        self.bind("r", lambda _: self.reset_auto())
        self.bind("x", lambda _: (self.exclude_var.set(not self.exclude_var.get()), self.on_exclude()))

    # ------------------------------------------------------------- rendering

    def refresh_tree(self) -> None:
        selected = self.tree.selection()
        self.tree.delete(*self.tree.get_children())
        for report in self.reports:
            state = self.state[report.name]
            mark = "✓ " if state["confirmed"] else ""
            if state["excluded"]:
                mark = "✗ "
            tags = [report.level] + (["confirmed"] if state["confirmed"] else [])
            self.tree.insert("", "end", iid=report.name, text=f"{mark}{report.name}",
                             values=(LEVEL_LABEL[report.level], f"{report.kept_seconds:.0f}s"), tags=tags)
        if selected:
            self.tree.selection_set(selected)

    def capture(self, report: Report) -> cv2.VideoCapture | None:
        if report.name not in self.captures:
            videos = sorted(report.path.glob("videos/*/chunk-*/file-*.mp4"))
            top = [v for v in videos if "top" in v.parts[-3]] or videos
            self.captures[report.name] = cv2.VideoCapture(str(top[0])) if top else None
        return self.captures[report.name]

    def render_frame(self, key: str, report: Report, frame_index: int) -> None:
        self.frame_labels[key].configure(text=f"{frame_index}  ({frame_index / report.fps:.2f}s)")
        capture = self.capture(report)
        if capture is None or not capture.isOpened():
            self.previews[key].configure(image="", text="영상 없음")
            return
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(frame_index, report.total_frames - 1)))
        ok, frame = capture.read()
        if not ok:
            self.previews[key].configure(image="", text="프레임 읽기 실패")
            return
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = image.resize((PREVIEW_WIDTH, round(PREVIEW_WIDTH * image.height / image.width)))
        self.photos[key] = ImageTk.PhotoImage(image)
        self.previews[key].configure(image=self.photos[key], text="")

    def action_of(self, report: Report) -> np.ndarray | None:
        """The commanded trajectory, loaded once per episode for the trace plot."""
        if report.name not in self.actions:
            try:
                self.actions[report.name] = load_action(report.path)
            except Exception:  # noqa: BLE001 - a missing trace must not take the window down
                self.actions[report.name] = None
        return self.actions[report.name]

    def draw_trace(self) -> None:
        canvas = self.trace
        canvas.delete("all")
        if not self.reports:
            return
        report = self.reports[self.index]
        action = self.action_of(report)
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if action is None or width < 50 or height < 50:
            return

        total = len(action)
        pad = 4
        lane = (height - 3 * pad) / 2
        step = np.abs(np.diff(action[:, :6], axis=0)).max(axis=1)
        gripper = action[:, 6]

        def polyline(values: np.ndarray, top: float, ceiling: float, color: str) -> None:
            scale = ceiling if ceiling > 0 else 1.0
            points = []
            for x in range(width):
                lo = int(x / width * len(values))
                hi = max(lo + 1, int((x + 1) / width * len(values)))
                peak = float(values[lo:hi].max())
                points.extend((x, top + lane - min(peak / scale, 1.0) * lane))
            canvas.create_line(*points, fill=color, width=1)

        polyline(step, pad, 8.0, "#444444")
        polyline(gripper, 2 * pad + lane, max(float(gripper.max()), 1.0), "#444444")
        canvas.create_text(4, pad + 8, anchor="w", text="관절 변화량", fill="#999", font=("TkDefaultFont", 9))
        canvas.create_text(4, 2 * pad + lane + 8, anchor="w", text="그리퍼", fill="#999", font=("TkDefaultFont", 9))

        def vline(frame_index: int, color: str, dash: tuple[int, int] | None, label: str, row: int = 0) -> None:
            # ``row`` stacks the captions: the two release markers can sit four
            # frames apart, and side by side their labels are unreadable.
            x = frame_index / total * width
            canvas.create_line(x, 0, x, height, fill=color, width=2, dash=dash)
            canvas.create_text(min(x + 4, width - 4), 4 + row * 13, anchor="nw", text=label, fill=color,
                               font=("TkDefaultFont", 9, "bold"))

        if report.motion_onset is not None:
            vline(report.motion_onset, "#1f77b4", (3, 3), "움직임", row=1)
        # Both ends of the release are drawn whichever one is selected, so the gap
        # between "starts opening" and "eraser down" is visible while nudging.
        if report.gripper_release is not None:
            vline(report.gripper_release, "#ff7f0e", (3, 3), "벌리기 시작", row=1)
        if report.gripper_release_done is not None and report.gripper_release_done != report.gripper_release:
            vline(report.gripper_release_done, "#9467bd", (3, 3), "놓기 완료", row=2)
        state = self.state[report.name]
        vline(state["start"], "#2e7d32", None, "시작")
        vline(state["end"], "#2e7d32", None, "종료")

    def on_trace_click(self, event) -> None:
        if not self.reports:
            return
        report = self.reports[self.index]
        width = max(self.trace.winfo_width(), 1)
        frame_index = int(max(0, min(event.x / width, 1.0)) * (report.total_frames - 1))
        state = self.state[report.name]
        key = "start" if abs(frame_index - state["start"]) <= abs(frame_index - state["end"]) else "end"
        state[key] = frame_index
        self.sliders[key].set(frame_index)
        self.render_frame(key, report, frame_index)
        self.draw_trace()

    def show(self, index: int) -> None:
        if not self.reports:
            return
        self.index = max(0, min(index, len(self.reports) - 1))
        report = self.reports[self.index]
        state = self.state[report.name]

        self.title_label.configure(text=f"{report.name}   ({self.index + 1}/{len(self.reports)})")
        notes = "; ".join(report.reasons) or "특이사항 없음"
        gap = (report.gripper_release_done - report.gripper_release
               if report.gripper_release is not None and report.gripper_release_done is not None else None)
        self.verdict.configure(text=f"{LEVEL_LABEL[report.level]}  —  {notes}\n"
                                    f"전체 {report.total_frames}프레임 · 움직임 시작 {report.motion_onset} · "
                                    f"벌리기 시작 {report.gripper_release} · "
                                    f"놓기 완료 {report.gripper_release_done}"
                                    + (f" (+{gap}프레임)" if gap else "")
                                    + f" · 지움 {'판정 불가' if report.erased is None else f'{report.erased * 100:.0f}%'}"
                                    + (f"   ({'; '.join(report.notes)})" if report.notes else ""),
                               foreground=LEVEL_COLOR[report.level])
        self.exclude_var.set(state["excluded"])

        for key in ("start", "end"):
            self.sliders[key].configure(to=report.total_frames - 1)
            self.sliders[key].set(state[key])
            self.render_frame(key, report, state[key])
        self.draw_trace()

        self.tree.selection_set(report.name)
        self.tree.see(report.name)

    # --------------------------------------------------------------- actions

    def on_select(self, _event) -> None:
        selection = self.tree.selection()
        if selection:
            index = next((i for i, r in enumerate(self.reports) if r.name == selection[0]), None)
            if index is not None and index != self.index:
                self.show(index)

    def on_slide(self, key: str, value: str) -> None:
        if not self.reports:
            return
        report = self.reports[self.index]
        frame_index = int(float(value))
        if self.state[report.name][key] == frame_index:
            return
        self.state[report.name][key] = frame_index
        self.render_frame(key, report, frame_index)
        self.draw_trace()

    def nudge(self, key: str, delta: int) -> None:
        report = self.reports[self.index]
        value = max(0, min(self.state[report.name][key] + delta, report.total_frames - 1))
        self.state[report.name][key] = value
        self.sliders[key].set(value)
        self.render_frame(key, report, value)
        self.draw_trace()

    def reset_auto(self) -> None:
        report = self.reports[self.index]
        self.state[report.name].update(start=report.start, end=self.auto_end(report))
        self.show(self.index)

    def on_exclude(self) -> None:
        report = self.reports[self.index]
        self.state[report.name]["excluded"] = self.exclude_var.get()
        self.refresh_tree()

    def confirm_next(self) -> None:
        report = self.reports[self.index]
        state = self.state[report.name]
        if state["end"] <= state["start"]:
            messagebox.showwarning("범위 오류", "종료 프레임이 시작 프레임보다 뒤여야 합니다.")
            return
        state["confirmed"] = True
        self.refresh_tree()
        remaining = [i for i, r in enumerate(self.reports) if not self.state[r.name]["confirmed"]]
        self.status.configure(text=f"확인 완료 {len(self.reports) - len(remaining)}/{len(self.reports)}   ·   "
                                   f"저장 위치: {self.output}")
        self.show(remaining[0] if remaining else self.index + 1)

    def save(self) -> None:
        episodes = []
        for report in self.reports:
            state = self.state[report.name]
            episodes.append({
                "source_dataset": report.path.resolve().relative_to(REPO_ROOT).as_posix(),
                "total_frames": report.total_frames,
                "enabled": not state["excluded"],
                "start_frame": state["start"],
                "end_frame": state["end"],
                "confirmed": state["confirmed"],
                "qc_level": report.level,
                "qc_notes": report.reasons + report.notes,
                "erased_ratio": None if report.erased is None else round(report.erased, 3),
                "auto_start_frame": report.start,
                "auto_end_frame": self.auto_end(report),
                "gripper_release": report.gripper_release,
                "gripper_release_done": report.gripper_release_done,
            })
        document = {
            "format_version": 1,
            "generated_by": "scripts/tools/qc_studio.py",
            "reviewed_at": datetime.now().isoformat(timespec="seconds"),
            "range_semantics": "start_frame is inclusive; end_frame is exclusive",
            "end_event": self.end_event.get(),
            "episodes": episodes,
        }
        if self.output.exists():
            existing = json.loads(self.output.read_text())
            if existing.get("generated_by") != "scripts/tools/qc_studio.py":
                if not messagebox.askyesno(
                    "덮어쓰기 확인",
                    f"{self.output}\n\n이 파일은 QC Studio가 만든 파일이 아닙니다 "
                    f"(직접 라벨링한 파일일 수 있습니다).\n덮어쓸까요?",
                ):
                    return
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
        confirmed = sum(e["confirmed"] for e in episodes)
        enabled = sum(e["enabled"] for e in episodes)
        self.status.configure(text=f"저장 완료: {self.output}  ({enabled}개 사용, {confirmed}개 확인됨)")
        messagebox.showinfo("저장 완료", f"{self.output}\n\n사용 {enabled}개 · 확인 완료 {confirmed}개")

    def open_broken(self) -> None:
        window = tk.Toplevel(self)
        window.title("녹화 실패 폴더")
        window.geometry("640x420")
        ttk.Label(window, padding=10, wraplength=600, justify="left",
                  text="녹화가 중단되어 하위 폴더가 다 만들어지지 않은 폴더입니다. "
                       "선택한 폴더를 예비 폴더로 옮깁니다 — 삭제하지 않으므로 되돌릴 수 있습니다.").pack(anchor="w")

        checks: dict[str, tk.BooleanVar] = {}
        listing = ttk.Frame(window, padding=(10, 0))
        listing.pack(fill="both", expand=True)
        for report in self.broken:
            var = tk.BooleanVar(value=True)
            checks[report.name] = var
            ttk.Checkbutton(listing, variable=var,
                            text=f"{report.name}  —  {'; '.join(report.reasons)}").pack(anchor="w")

        def move() -> None:
            chosen = [r for r in self.broken if checks[r.name].get()]
            if not chosen:
                return
            if not messagebox.askyesno("이동 확인", f"{len(chosen)}개 폴더를 아래로 옮깁니다.\n\n"
                                                    f"{self.quarantine}\n\n진행할까요?", parent=window):
                return
            self.quarantine.mkdir(parents=True, exist_ok=True)
            moved = []
            for report in chosen:
                target = self.quarantine / report.name
                if target.exists():
                    target = self.quarantine / f"{report.name}_{datetime.now():%H%M%S}"
                shutil.move(str(report.path), str(target))
                moved.append(report)
            self.broken = [r for r in self.broken if r not in moved]
            self.status.configure(text=f"{len(moved)}개 폴더를 {self.quarantine} 로 옮겼습니다.")
            window.destroy()

        buttons = ttk.Frame(window, padding=10)
        buttons.pack(fill="x", side="bottom")
        ttk.Button(buttons, text="취소", command=window.destroy).pack(side="right")
        ttk.Button(buttons, text="예비 폴더로 이동", command=move).pack(side="right", padx=8)


def main() -> int:
    args = parse_args()
    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(f"folder not found: {folder}")

    reports = inspect_folder(folder, start_margin=args.start_margin, end_margin=args.end_margin,
                             round_start=args.round_start, end_event=args.end_event)
    if not reports:
        raise SystemExit(f"no recordings under {folder}")

    if args.report:
        print_report(reports, args.end_event)
        return 0

    output = args.output or REPO_ROOT / f"configs/qc_review_{folder.name}.json"
    studio = Studio(reports, output.expanduser().resolve(), args.quarantine.expanduser().resolve(),
                    end_event=args.end_event, end_margin=args.end_margin)
    studio.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
