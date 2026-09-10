#!/usr/bin/env python3
"""Where a teleoperated episode really starts and ends.

Shared by ``autofill_frame_ranges.py`` (fills the trim manifest) and available
to the QC report, so both answer "is this frame idle?" the same way.

Two events carry the episode boundaries, and both are visible in the seven
commanded values of ``action`` alone -- no video decoding:

``motion_onset``
    The commanded joint step ``max_j |action_j[t+1] - action_j[t]|`` sits at the
    encoder-noise floor (< 0.1 deg/frame) while the operator lines up, then
    crosses 0.5 deg/frame and stays there. The sustain requirement matters: the
    idle stretch contains isolated single-frame spikes above 0.5, so a bare
    threshold reports onset within the first frames of an episode that in fact
    waits another 50.

``gripper_release``
    While erasing, the gripper command rests on a flat plateau (~17). It opens
    twice per episode -- once to take the eraser, once to put it down -- so the
    release is the *last* upward crossing of ``plateau + 1`` that stays high.

That crossing is where the gripper *begins* to open, which is not the same frame
as the one where the eraser is actually down: the opening ramp runs 14 frames on
average and up to 44 in the batch, and in the slow episodes the first move off
the plateau is a partial loosening while the operator is still lowering the
eraser onto its stand. ``gripper_release_done`` is the top of that ramp -- the
gripper is at its widest, the eraser is free, and the arm retreats from the next
frame on. Which of the two ends the episode is the caller's choice:

``release_start`` (default)
    Where the 60 hand-labelled episodes sit, so it stays the default and existing
    manifests keep reproducing. Its margin of 6 is fitted against those labels
    (MAE 2.8 frames) and drops the first flicker off the plateau.

``release_done``
    Ends the cut *on* the fully-open frame, so the placement is inside the
    training data. Its margin is 0 by design and deliberately not fitted: fitting
    it against the existing labels returns 19, which lands the cut back before
    the gripper has opened at all -- the same footage as ``release_start``, which
    is the one thing this option exists to avoid. Re-fit it only once a labelling
    session has actually marked the placement.
"""

from __future__ import annotations

import numpy as np


MOTION_STEP_DEG = 0.5
MOTION_HOLD_FRAMES = 3
GRIPPER_MARGIN = 1.0
GRIPPER_HOLD_FRAMES = 5
PLATEAU_WINDOW = (0.3, 0.7)
RELEASE_RAMP_FRAMES = 90  # the opening ramp tops out well inside this many frames

END_EVENTS = ("release_start", "release_done")
END_MARGIN = {"release_start": 6, "release_done": 0}
DEFAULT_END_EVENT = "release_start"


def motion_onset(
    action: np.ndarray,
    threshold: float = MOTION_STEP_DEG,
    hold: int = MOTION_HOLD_FRAMES,
) -> int:
    """First frame of sustained arm motion, or 0 if the arm never settles first."""
    step = np.abs(np.diff(action[:, :6], axis=0)).max(axis=1)
    moving = step > threshold
    for frame in range(len(moving) - hold):
        if moving[frame : frame + hold].all():
            return frame
    return 0


def motion_offset(
    action: np.ndarray,
    threshold: float = MOTION_STEP_DEG,
    hold: int = MOTION_HOLD_FRAMES,
) -> int:
    """Last frame of sustained arm motion, or ``len(action)`` if it never stops."""
    step = np.abs(np.diff(action[:, :6], axis=0)).max(axis=1)
    moving = step > threshold
    for frame in range(len(moving) - hold, 0, -1):
        if moving[frame : frame + hold].all():
            return frame + hold
    return len(action)


def gripper_plateau(action: np.ndarray) -> float:
    """The gripper command value held while the eraser is in hand."""
    gripper = action[:, 6]
    low, high = (int(len(gripper) * bound) for bound in PLATEAU_WINDOW)
    return float(np.median(gripper[low:high]))


def gripper_release(action: np.ndarray) -> int | None:
    """Frame where the gripper leaves its holding plateau to put the eraser down."""
    gripper = action[:, 6]
    threshold = gripper_plateau(action) + GRIPPER_MARGIN

    release = None
    for frame in range(len(gripper) - GRIPPER_HOLD_FRAMES - 1):
        opens = gripper[frame] <= threshold and (
            gripper[frame + 1 : frame + 1 + GRIPPER_HOLD_FRAMES] > threshold
        ).all()
        if opens:
            release = frame + 1
    return release


def gripper_release_done(
    action: np.ndarray,
    release: int | None = None,
    ramp: int = RELEASE_RAMP_FRAMES,
) -> int | None:
    """Frame where the gripper reaches full open — the eraser is down and free."""
    if release is None:
        release = gripper_release(action)
    if release is None:
        return None
    opening = action[release : release + ramp, 6]
    return release + int(np.argmax(opening >= opening.max()))


def release_frame(action: np.ndarray, event: str = DEFAULT_END_EVENT) -> int | None:
    """Whichever end of the release the caller asked for."""
    if event not in END_EVENTS:
        raise ValueError(f"end event must be one of {END_EVENTS}, got {event!r}")
    release = gripper_release(action)
    return release if event == "release_start" else gripper_release_done(action, release)


def suggest_range(
    action: np.ndarray,
    start_margin: int = 22,
    end_margin: int | None = None,
    round_start: int = 10,
    end_event: str = DEFAULT_END_EVENT,
) -> tuple[int, int, list[str]]:
    """Trim range for one episode, plus anything that needs a human eye."""
    total = len(action)
    warnings: list[str] = []
    if end_margin is None:
        end_margin = END_MARGIN[end_event]

    onset = motion_onset(action)
    start = onset - start_margin
    if round_start > 0:
        start -= start % round_start
    start = max(0, start)

    release = release_frame(action, end_event)
    if release is None:
        warnings.append("그리퍼 릴리즈를 못 찾음 — 마지막 움직임으로 대체")
        end = min(motion_offset(action), total)
    else:
        end = release - end_margin

    if end <= start:
        warnings.append(f"범위가 잘못됨 [{start}, {end})")
    return start, min(end, total), warnings
