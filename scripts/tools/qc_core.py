#!/usr/bin/env python3
"""Quality checks and cut suggestions for a folder of recordings.

Pure logic, no UI -- ``qc_studio.py`` draws this, and ``--report`` prints it.

Three things are decided here:

*Is the folder even a recording?* A successful ``record`` leaves ``data/``,
``meta/`` and ``videos/``. The retry path that shows up in the terminal after
an occasional successful episode leaves the directory half-built -- usually
``meta/`` alone, no video at all. Those are reported as ``BROKEN`` and are
candidates for quarantine; nothing is ever deleted from here.

*Where does the episode start and end?* Delegated to ``episode_segmentation``.

*Is the episode worth training on?* The question that decides this is whether
the target shape actually came off the board, and that is settled by looking at
the video rather than at the arm. The first frame and the clearest of the last
ten (by then the arm has parked and the board is unobstructed) are compared:
each drawn shape is a connected blob of dark pixels, and a shape counts as
erased when its ink is gone in the after-frame.

Which shape is the target is never assumed -- whichever one disappeared is the
one that was meant to go. That keeps this working when the task moves off the
circle. How long the wipe took, how many strokes it needed and how hard the
eraser pressed are deliberately *not* judged: a slow, scrubbing, effortful
episode that ends with a clean board is a good episode.

An earlier version of this file scored contact through ``joint5.effort``. It was
checked against the video and dropped: episodes wiped clean registered as little
as 2 contact frames against a batch median of 44, so the measure produced false
alarms on successful episodes and is not used for anything now.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from episode_segmentation import (
    DEFAULT_END_EVENT,
    gripper_plateau,
    gripper_release,
    gripper_release_done,
    motion_onset,
    suggest_range,
)


REQUIRED_SUBDIRS = ("data", "meta", "videos")

FPS_TOL = 0.15          # allowed deviation from 1/fps between consecutive frames
JUMP_THRESH = 15.0      # per-frame joint change treated as a teleop discontinuity
MIN_KEPT_SECONDS = 5.0  # a usable episode is at least this long after trimming
LENGTH_Z = 2.0          # length outliers are reported, but do not change the verdict

# Erase verification, tuned against the video. Across 21 recordings every shape
# that a human calls erased scores 86-100%, while the episodes already set aside
# as failures score -0.1%, 7%, 54% and 56% -- so the two bands are far apart and
# the thresholds sit in the empty space between them.
INK_LEVEL = 120         # a pixel this dark on the whiteboard is ink
SHAPE_MIN_AREA = 200    # smaller blobs are noise
SHAPE_MAX_AREA = 5000   # larger ones are the eraser holder or the arm, not a drawn shape
TAIL_FRAMES = 10        # after-frame is picked from this many frames at the end
BOARD_COLUMNS = (200, 800)  # the whiteboard, used only to score how occluded a frame is
ERASED_OK = 0.80        # target shape gone
ERASED_PARTIAL = 0.40   # below this nothing meaningful came off

RED, YELLOW, GREEN = "red", "yellow", "green"
LEVEL_ORDER = {RED: 0, YELLOW: 1, GREEN: 2}


@dataclass
class Report:
    """One recording folder, whether or not it holds a usable episode."""

    name: str
    path: Path
    level: str = GREEN
    reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    broken: bool = False
    missing: tuple[str, ...] = ()

    total_frames: int = 0
    fps: int = 30
    start: int | None = None
    end: int | None = None
    motion_onset: int | None = None
    gripper_release: int | None = None
    gripper_release_done: int | None = None
    erased: float | None = None
    n_shapes: int = 0
    n_jumps: int = 0
    max_jump: float = 0.0
    start_pose: np.ndarray | None = None
    start_dev: float = 0.0
    length_z: float = 0.0
    cameras: tuple[str, ...] = ()

    def flag(self, level: str, reason: str) -> None:
        self.reasons.append(reason)
        if LEVEL_ORDER[level] < LEVEL_ORDER[self.level]:
            self.level = level

    @property
    def kept_frames(self) -> int:
        if self.start is None or self.end is None:
            return 0
        return self.end - self.start

    @property
    def kept_seconds(self) -> float:
        return self.kept_frames / self.fps if self.fps else 0.0


def scan(root: Path) -> list[Path]:
    """Every immediate subdirectory of a recording folder, complete or not."""
    return sorted(path for path in root.iterdir() if path.is_dir())


def board_frames(video: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """The board before the episode and after it, in grayscale.

    The after-frame is the least occluded of the last few frames: by then the
    arm has parked, but picking the brightest one guards against a run that ends
    with the gripper still over the board.
    """
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return None
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, first = capture.read()
        if not ok:
            return None

        best = None
        for index in range(max(0, total - TAIL_FRAMES), total):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = capture.read()
            if not ok:
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clearness = float(gray[:, slice(*BOARD_COLUMNS)].mean())
            if best is None or clearness > best[0]:
                best = (clearness, gray)
        if best is None:
            return None
        return cv2.cvtColor(first, cv2.COLOR_BGR2GRAY), best[1]
    finally:
        capture.release()


def erase_ratios(video: Path) -> list[float]:
    """How much of each drawn shape is gone by the end, best first."""
    pair = board_frames(video)
    if pair is None:
        return []
    before, after = pair

    ink = (before < INK_LEVEL).astype(np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)

    ratios = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if not SHAPE_MIN_AREA <= area <= SHAPE_MAX_AREA:
            continue
        mask = labels == index
        was = int((before[mask] < INK_LEVEL).sum())
        still = int((after[mask] < INK_LEVEL).sum())
        ratios.append(1.0 - still / max(was, 1))
    return sorted(ratios, reverse=True)


def read_episode(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    info = json.loads((root / "meta/info.json").read_text())
    files = sorted(root.glob("data/chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError("no episode parquet")
    frame = pd.concat([pd.read_parquet(path) for path in files]).sort_values("frame_index")
    return frame, info


def inspect(
    path: Path,
    start_margin: int = 22,
    end_margin: int | None = None,
    round_start: int = 10,
    end_event: str = DEFAULT_END_EVENT,
) -> Report:
    """Everything that can be judged from one recording folder on its own."""
    report = Report(name=path.name, path=path)

    missing = tuple(sub for sub in REQUIRED_SUBDIRS if not (path / sub).is_dir())
    if missing:
        report.broken = True
        report.missing = missing
        report.flag(RED, f"녹화 미완료 — {', '.join(missing)}/ 폴더 없음")
        return report

    try:
        frame, info = read_episode(path)
    except Exception as error:  # noqa: BLE001 - any read failure is a red flag, not a crash
        report.broken = True
        report.flag(RED, f"읽을 수 없음: {error}")
        return report

    report.fps = info.get("fps", 30)
    report.total_frames = len(frame)
    report.cameras = tuple(sorted(p.name for p in (path / "videos").iterdir() if p.is_dir()))
    if not report.cameras:
        report.broken = True
        report.flag(RED, "카메라 영상이 하나도 없음")

    action = np.stack(frame["action"].to_numpy())
    state = np.stack(frame["observation.state"].to_numpy())

    if info.get("total_frames") not in (None, len(frame)):
        report.flag(RED, f"메타는 {info['total_frames']}프레임인데 데이터는 {len(frame)}프레임")
    if np.isnan(action).any() or np.isnan(state).any():
        report.flag(RED, "action/state에 NaN 있음")

    timestamps = frame["timestamp"].to_numpy()
    if len(timestamps) > 1:
        expected = 1.0 / report.fps
        bad = np.abs(np.diff(timestamps) - expected) > FPS_TOL * expected
        if bad.any():
            report.flag(RED, f"프레임 간격 이상 {int(bad.sum())}곳")

    step = np.abs(np.diff(action[:, :6], axis=0)).max(axis=1)
    report.n_jumps = int((step > JUMP_THRESH).sum())
    report.max_jump = float(step.max()) if len(step) else 0.0
    if report.n_jumps:
        report.flag(YELLOW, f"텔레옵 끊김 {report.n_jumps}회 (최대 {report.max_jump:.1f}도/프레임)")

    report.motion_onset = motion_onset(action)
    report.gripper_release = gripper_release(action)
    report.gripper_release_done = gripper_release_done(action, report.gripper_release)
    start, end, warnings = suggest_range(action, start_margin, end_margin, round_start, end_event)
    report.start, report.end = start, end
    for warning in warnings:
        report.flag(RED, warning)
    if report.gripper_release is None:
        report.flag(RED, "그리퍼가 지우개를 놓지 않음 — 종료 프레임은 추정값")
    if report.kept_seconds < MIN_KEPT_SECONDS:
        report.flag(RED, f"잘라내면 {report.kept_seconds:.1f}초밖에 안 남음")

    videos = sorted(path.glob("videos/*top*/chunk-*/file-*.mp4")) or sorted(path.glob("videos/*/chunk-*/file-*.mp4"))
    ratios = erase_ratios(videos[0]) if videos else []
    report.n_shapes = len(ratios)
    report.erased = ratios[0] if ratios else None
    if report.erased is None:
        report.flag(YELLOW, "보드를 읽을 수 없음 — 지워졌는지 직접 확인 필요")
    elif report.erased < ERASED_PARTIAL:
        report.flag(RED, f"대상 도형이 보드에 그대로 남음 ({report.erased * 100:.0f}% 지워짐)")
    elif report.erased < ERASED_OK:
        report.flag(YELLOW, f"도형이 일부만 지워짐 ({report.erased * 100:.0f}%)")

    report.start_pose = state[0, :7]
    return report


def compare_group(reports: list[Report]) -> None:
    """Second pass: flags that only make sense against the rest of the batch."""
    usable = [r for r in reports if not r.broken and r.total_frames]
    if len(usable) < 4:
        return

    lengths = np.array([r.kept_frames for r in usable], dtype=float)
    mean, sd = lengths.mean(), lengths.std()
    poses = np.stack([r.start_pose for r in usable])
    pose_median = np.median(poses, axis=0)
    deviations = np.linalg.norm(poses - pose_median, axis=1)
    pose_limit = float(np.median(deviations) + 2 * deviations.std())

    for report, deviation in zip(usable, deviations):
        report.start_dev = round(float(deviation), 2)
        report.length_z = round(float((report.kept_frames - mean) / sd), 2) if sd else 0.0

        # Notes, not verdicts: an episode that took its time or started from a
        # slightly different pose is still a good episode if the board came clean.
        if abs(report.length_z) > LENGTH_Z:
            report.notes.append(f"길이가 평소와 다름 ({report.kept_seconds:.1f}초, z={report.length_z:+.1f})")
        if deviation > pose_limit:
            report.notes.append(f"시작 자세가 중앙값에서 {deviation:.1f} 벗어남")


def inspect_folder(root: Path, **kwargs: Any) -> list[Report]:
    reports = [inspect(path, **kwargs) for path in scan(root)]
    compare_group(reports)
    return reports
