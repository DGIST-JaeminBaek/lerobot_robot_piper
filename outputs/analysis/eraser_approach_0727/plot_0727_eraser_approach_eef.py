#!/usr/bin/env python3
"""Plot the 2026-07-27 eraser-approach demonstrations in EEF space.

The recording stores both state and action in LeRobot's normalized motor
coordinates, not Cartesian coordinates.  This offline tool deliberately uses
the recorded ``action`` (the target actually sent to the follower), converts
its first six values to radians using the same calibration ranges used by the
inference FK analysis, then calls the SDK's pure-DH FK implementation.

"Pick approach" means the full demonstration prefix: from recording frame 0
(the common home/start pose) through the first firmly-grasped plateau. It
includes opening the gripper and travelling to the eraser. Firm grasp is the
first post-close interval where both commanded and measured gripper positions
have settled, rather than the instant of first contact. The baseline line is the
time-normalized mean EEF trajectory over the complete 0727 folder batch.

No CAN interface or robot class is imported or opened.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]  # outputs/analysis/eraser_approach_0727/ 밑으로 옮겨진 파일 기준
sys.path.insert(0, str(REPO_ROOT))
from synthetic.kinematics.piper_fk import forward_kinematics_eef_batch  # noqa: E402

MOTOR_RANGES = np.array(
    [
        (-150_000.0, 150_000.0),
        (0.0, 180_000.0),
        (-170_000.0, 0.0),
        (-100_000.0, 100_000.0),
        (-65_000.0, 65_000.0),
        (-100_000.0, 130_000.0),
    ],
    dtype=np.float64,
)

# block_alignment_tool.py에서 검증한 학습 장면의 블럭 중심과 화면 이동 스케일.
# 이는 EEF→camera 전체 보정은 아니므로, 아래 topview 결과는 'block-anchored
# plan-view proxy'로만 쓴다. 높이(Z)는 topview 평면에 투영하지 않는다.
BLOCK_REFERENCE_PX = np.array([428.9, 305.9])
MM_PER_PX = np.array([0.690, 0.625])
SESSION_IDS = (
    "0728", "0802", "0802_joint4_corrected", "0804", "0804_joint4_corrected",
    "0805", "0811", "0812", "0813",
)
SESSION_COLORS = {
    "0728": "#009fe3", "0802": "#e45756", "0804": "#54a24b", "0805": "#b279a2",
    "0811": "#f58518", "0812": "#72b7b2", "0813": "#777777",
    # joint4 영점 오프셋을 보정한 버전(records/0802_joint4_corrected 등) — 원본과
    # 나란히 비교할 수 있게 같은 계열의 밝은 색을 쓴다.
    "0802_joint4_corrected": "#f2989a", "0804_joint4_corrected": "#9fd39f",
}
GRASP_SETTLE_STEP = 0.31
GRASP_SETTLE_FRAMES = 8
GRASP_MIN_CLOSE = 15.0
GRASP_SEARCH_FRAMES = 180


def normalized_action_to_rad(action: np.ndarray) -> np.ndarray:
    """LeRobot normalized absolute action targets -> Piper FK radians."""
    raw_millidegree = ((action[:, :6] + 100.0) / 200.0) * (MOTOR_RANGES[:, 1] - MOTOR_RANGES[:, 0]) + MOTOR_RANGES[:, 0]
    return np.deg2rad(raw_millidegree / 1000.0)


def contiguous_runs(indices: np.ndarray) -> list[tuple[int, int]]:
    if len(indices) == 0:
        return []
    split = np.flatnonzero(np.diff(indices) > 1) + 1
    return [(int(group[0]), int(group[-1])) for group in np.split(indices, split)]


def approach_bounds(
    gripper_action: np.ndarray,
    gripper_state: np.ndarray,
    threshold: float = 0.25,
) -> tuple[int, int, bool]:
    """Return [recording-start, firmly-grasped plateau] and whether it was found."""
    if gripper_action.shape != gripper_state.shape:
        raise ValueError("action/state gripper sequences must have the same shape")
    delta = np.diff(gripper_action)
    opening = contiguous_runs(np.flatnonzero(delta > threshold))
    closing = contiguous_runs(np.flatnonzero(delta < -threshold))
    if not opening:
        raise ValueError("no initial gripper-opening ramp")

    open_first, open_last = opening[0]

    # First closing after the opening is the grasp.  Tiny encoder corrections
    # are rejected by requiring a three-frame run.
    close_candidates = [(first, last) for first, last in closing if first > open_last and last - first >= 2]
    if not close_candidates:
        raise ValueError("no sustained gripper-closing ramp after opening")
    close_start = close_candidates[0][0] + 1
    if close_start < 10:
        raise ValueError(f"pick window too short: [0, {close_start})")

    # 접촉 직후에도 그리퍼는 계속 닫힌다. 우리가 원하는 것은 접촉이 아니라 '꽉 문'
    # 시점이므로 첫 닫힘에서 최소 15만큼 닫힌 뒤 action/state 모두 8프레임 동안
    # 멈추는 최초 plateau를 찾는다. 이 조건은 고정 action-state offset과 무관하다.
    search_end = min(close_start + GRASP_SEARCH_FRAMES, len(gripper_action))
    for frame in range(close_start + 1, search_end - GRASP_SETTLE_FRAMES + 1):
        closed_enough = gripper_action[close_start] - gripper_action[frame] >= GRASP_MIN_CLOSE
        command_settled = np.abs(np.diff(gripper_action[frame : frame + GRASP_SETTLE_FRAMES])).max() <= GRASP_SETTLE_STEP
        measured_settled = np.abs(np.diff(gripper_state[frame : frame + GRASP_SETTLE_FRAMES])).max() <= GRASP_SETTLE_STEP
        if closed_enough and command_settled and measured_settled:
            return 0, frame + 1, True

    # 접촉 후보가 없으면 데이터를 조용히 버리지 않고 이전의 보수적 경계로 후퇴한다.
    return 0, close_start, False


def resample_xyz(xyz: np.ndarray, samples: int) -> np.ndarray:
    source = np.linspace(0.0, 1.0, len(xyz))
    target = np.linspace(0.0, 1.0, samples)
    return np.column_stack([np.interp(target, source, xyz[:, axis]) for axis in range(3)])


def medoid_index(trajectories: np.ndarray) -> int:
    # Time-normalized RMS distance makes the representative a real recorded path.
    pairwise = np.sqrt(np.mean((trajectories[:, None] - trajectories[None, :]) ** 2, axis=(2, 3)))
    return int(np.argmin(pairwise.mean(axis=1)))


def session_recordings(session: str) -> list[Path]:
    """Raw recordings for one requested calendar session; 0813 may legitimately be empty."""
    root = REPO_ROOT / "records" / session
    if not root.is_dir():
        return []
    candidates = (root.glob("*/*0727-*") if session == "0727" else root.glob("*"))
    return sorted(path for path in candidates if (path / "data").is_dir())


def baseline_recordings_0727_folder() -> list[Path]:
    """All 60 recordings stored in the three original 0727 task folders.

    Some circle recordings retain a 0726 timestamp in their filename, but they
    belong to the same collection batch and must not be discarded from the
    requested folder-level baseline.
    """
    root = REPO_ROOT / "records/0727"
    return sorted(path for path in root.glob("erase_the_*/*") if (path / "data").is_dir())


def load_action_trajectories(episodes: list[Path], samples: int) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Action-FK trajectories from recording start through first grasp."""
    trajectories: list[np.ndarray] = []
    reports: list[dict[str, object]] = []
    for episode in episodes:
        try:
            data = pd.concat([pd.read_parquet(p) for p in sorted(episode.glob("data/chunk-*/file-*.parquet"))])
            action = np.stack(data["action"].to_numpy()).astype(np.float64)
            state = np.stack(data["observation.state"].to_numpy()).astype(np.float64)
            start, end, contact_detected = approach_bounds(action[:, 6], state[:, 6])
            xyz = forward_kinematics_eef_batch(normalized_action_to_rad(action[start:end]))[:, :3]
            trajectories.append(resample_xyz(xyz, samples))
            reports.append({"episode": str(episode), "included": True, "approach_start_frame": start, "firm_grasp_frame": end - 1, "firm_grasp_detected": contact_detected, "approach_frames": end - start})
        except Exception as error:  # report unusable episodes instead of silently excluding them
            reports.append({"episode": str(episode), "included": False, "reason": str(error)})
    return np.stack(trajectories) if trajectories else np.empty((0, samples, 3)), reports


def plot(output: Path, trajectories: np.ndarray, names: list[str], voxel_mm: float) -> tuple[int, int]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    medoid = medoid_index(trajectories)  # topview 배경으로 쓸 실제 녹화 선택용
    mean_trajectory = trajectories.mean(axis=0)
    points = trajectories.reshape(-1, 3)
    origin = np.floor(points.min(axis=0) / voxel_mm) * voxel_mm
    cells = np.floor((points - origin) / voxel_mm).astype(int)
    unique, counts = np.unique(cells, axis=0, return_counts=True)
    centers = origin + (unique + 0.5) * voxel_mm

    figure = plt.figure(figsize=(13, 10), layout="constrained")
    axis = figure.add_subplot(111, projection="3d")
    density = counts / counts.max()
    colors = plt.cm.magma(Normalize(0, 1)(density))
    colors[:, 3] = 0.06 + 0.40 * density
    axis.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=30 + 160 * density, c=colors, marker="o", linewidths=0, label="trajectory density (8 mm voxels)")
    axis.plot(*mean_trajectory.T, color="#111827", linewidth=3.5, linestyle="--", label=f"baseline: mean of {len(trajectories)} recordings")
    axis.scatter(*trajectories[:, 0, :].mean(axis=0), color="#2e7d32", s=80, marker="o", label="mean ready-to-reach point")
    axis.scatter(*trajectories[:, -1, :].mean(axis=0), color="#c62828", s=100, marker="X", label="mean grasp point")
    axis.set(title="2026-07-27 eraser approach: EEF trajectory distribution", xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)")
    # 이 데이터에서는 Y 폭이 작고 X 이동·Z(높이) 하강이 핵심이다. Y축을
    # 바라보는 측면도를 기본으로 해서 이미지에서도 Z 높이가 곧바로 읽힌다.
    axis.view_init(elev=0, azim=-90)
    axis.set_box_aspect(np.ptp(points, axis=0))
    axis.legend(loc="upper left", fontsize=8)
    figure.savefig(output, dpi=220)
    plt.close(figure)
    return medoid, len(unique)


def save_topview_overlay(output: Path, video: Path, trajectories: np.ndarray, medoid: int) -> None:
    """Overlay a block-anchored top-down EEF plan view on a real recording frame.

    The previous block-alignment tool supplies the fixed eraser-home pixel and
    a measured px/mm scale.  It does *not* supply a calibrated camera matrix,
    so each trajectory is translated so its grasp endpoint coincides with that
    home.  This makes the distribution inspectable on the real image without
    pretending that Z-height can be seen from a top view.
    """
    import cv2

    capture = cv2.VideoCapture(str(video))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"cannot decode top video: {video}")

    # A top camera sees the horizontal EEF plane.  X maps left/right; the sign
    # convention for Y follows the block tool's image-coordinate convention.
    endpoints = trajectories[:, -1, :2]
    delta_mm = trajectories[:, :, :2] - endpoints[:, None, :]
    projected = np.empty_like(delta_mm)
    projected[..., 0] = BLOCK_REFERENCE_PX[0] - delta_mm[..., 0] / MM_PER_PX[0]
    projected[..., 1] = BLOCK_REFERENCE_PX[1] + delta_mm[..., 1] / MM_PER_PX[1]

    overlay = frame.copy()
    for path in projected:
        cv2.polylines(overlay, [np.rint(path).astype(np.int32)], False, (180, 80, 210), 1, cv2.LINE_AA)
    frame = cv2.addWeighted(overlay, 0.36, frame, 0.64, 0)
    standard = np.rint(projected.mean(axis=0)).astype(np.int32)
    cv2.polylines(frame, [standard], False, (230, 180, 0), 4, cv2.LINE_AA)
    cv2.circle(frame, tuple(np.rint(projected[:, 0].mean(axis=0)).astype(int)), 8, (50, 160, 50), -1, cv2.LINE_AA)
    cv2.drawMarker(frame, tuple(np.rint(BLOCK_REFERENCE_PX).astype(int)), (30, 30, 220), cv2.MARKER_TILTED_CROSS, 22, 3, cv2.LINE_AA)
    cv2.putText(frame, "0727 approach distribution: block-anchored top-view proxy", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 4, cv2.LINE_AA)
    cv2.putText(frame, "purple=all demos  yellow=standard  green=start  red=block/grasp", (24, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(frame, "Z(height) is intentionally not projected onto this top view", (24, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.imwrite(str(output), frame)


def save_interactive_html(
    output: Path,
    trajectories: np.ndarray,
    names: list[str],
    voxel_mm: float,
    session_averages: dict[str, np.ndarray],
    session_counts: dict[str, int],
) -> int:
    """Save a browser-viewable Plotly scene without adding a Python dependency.

    Plotly itself is loaded from its official CDN by the browser.  Keeping the
    trajectory data inline makes this one HTML file portable over SCP/SFTP.
    """
    medoid = medoid_index(trajectories)
    points = trajectories.reshape(-1, 3)
    origin = np.floor(points.min(axis=0) / voxel_mm) * voxel_mm
    cells = np.floor((points - origin) / voxel_mm).astype(int)
    unique, counts = np.unique(cells, axis=0, return_counts=True)
    centers = origin + (unique + 0.5) * voxel_mm
    density = counts / counts.max()
    payload = {
        "centers": centers.round(4).tolist(),
        "density": density.round(5).tolist(),
        "standard": trajectories.mean(axis=0).round(4).tolist(),
        "standardName": f"mean of {len(trajectories)} recordings",
        "ready": trajectories[:, 0, :].mean(axis=0).round(4).tolist(),
        "grasp": trajectories[:, -1, :].mean(axis=0).round(4).tolist(),
        "sessions": [
            {"id": session, "n": session_counts[session], "xyz": path.round(4).tolist(),
             "color": SESSION_COLORS[session]}
            for session, path in session_averages.items()
        ],
    }
    html = """<!doctype html>
<html lang=\"ko\"><meta charset=\"utf-8\"><title>0727 Eraser Approach EEF</title>
<style>html,body{width:100%;height:100%;margin:0;background:#fff;font-family:sans-serif}.note{height:64px;box-sizing:border-box;padding:7px 16px;color:#333}.controls label{margin-right:14px;white-space:nowrap}.plots{height:calc(100% - 64px);display:flex}.plot{width:50%;height:100%}</style>
<div class=\"note\"><div>왼쪽: X–Z 검수 화면 (드래그=이동, 휠=확대, 더블클릭=초기화). Z는 높이입니다.</div><div class=\"controls\" id=\"controls\"></div></div>
<div class=\"plots\"><div id=\"xz\" class=\"plot\"></div><div id=\"plot3d\" class=\"plot\"></div></div>
<script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
<script>
const data = __PAYLOAD__;
const xyz = i => data.centers.map(p => p[i]);
const sx = data.standard.map(p => p[0]), sy = data.standard.map(p => p[1]), sz = data.standard.map(p => p[2]);
const fog2d = {type:'scattergl', mode:'markers', name:'trajectory density (8 mm cells)', x:xyz(0), y:xyz(2), marker:{size:data.density.map(v=>4+16*v), color:data.density, colorscale:'Magma', opacity:0.34, showscale:true, colorbar:{title:'relative<br>density'}}, hovertemplate:'X=%{x:.1f} mm<br>Z(height)=%{y:.1f} mm<extra></extra>'};
const standard2d = {type:'scatter', mode:'lines', name:'baseline: 0727 '+data.standardName, x:sx, y:sz, line:{color:'#111827', width:4, dash:'dash'}};
const ready2d = {type:'scatter', mode:'markers', name:'mean ready-to-reach point', x:[data.ready[0]], y:[data.ready[2]], marker:{size:11,color:'#2e7d32'}};
const grasp2d = {type:'scatter', mode:'markers', name:'mean grasp point', x:[data.grasp[0]], y:[data.grasp[2]], marker:{size:13,color:'#c62828',symbol:'x'}};
const average2d = data.sessions.map((s,i) => ({type:'scattergl', mode:'lines', name:s.id+' mean action EEF (n='+s.n+')', x:s.xyz.map(p=>p[0]), y:s.xyz.map(p=>p[2]), line:{color:s.color,width:4}, visible:i===0}));
const average3d = data.sessions.map((s,i) => ({type:'scatter3d', mode:'lines', name:s.id+' mean action EEF (n='+s.n+')', x:s.xyz.map(p=>p[0]), y:s.xyz.map(p=>p[1]), z:s.xyz.map(p=>p[2]), line:{color:s.color,width:7}, visible:i===0}));
Plotly.newPlot('xz', [fog2d, standard2d, ready2d, grasp2d, ...average2d], {title:'검수 기본: X–Z 단면 (Z = 높이)', xaxis:{title:'X (mm)',zeroline:false}, yaxis:{title:'Z / height (mm)',zeroline:false,scaleanchor:'x',scaleratio:1}, dragmode:'pan', margin:{l:70,r:10,b:60,t:50}, legend:{x:0,y:1}}, {responsive:true, scrollZoom:true});
Plotly.newPlot('plot3d', [
 {type:'scatter3d', mode:'markers', name:'trajectory density (8 mm voxels)',
  x:xyz(0), y:xyz(1), z:xyz(2), marker:{size:data.density.map(v=>2+8*v), color:data.density, colorscale:'Magma', opacity:0.20, showscale:true, colorbar:{title:'relative<br>density'}}},
 {type:'scatter3d', mode:'lines', name:'baseline: 0727 '+data.standardName,
  x:sx, y:sy, z:sz, line:{color:'#111827', width:8, dash:'dash'}},
 {type:'scatter3d', mode:'markers', name:'mean ready-to-reach point',
  x:[data.ready[0]], y:[data.ready[1]], z:[data.ready[2]], marker:{size:7,color:'#2e7d32'}},
 {type:'scatter3d', mode:'markers', name:'mean grasp point',
  x:[data.grasp[0]], y:[data.grasp[1]], z:[data.grasp[2]], marker:{size:8,color:'#c62828',symbol:'x'}}, ...average3d
], {title:'보조: 3D 분포', scene:{aspectmode:'data',camera:{eye:{x:0,y:-2.5,z:0},up:{x:0,y:0,z:1}},xaxis:{title:'X (mm)'},yaxis:{title:'Y (mm)'},zaxis:{title:'Z / height (mm)'}}, margin:{l:0,r:0,b:0,t:48}, showlegend:false}, {responsive:true});
const controls=document.getElementById('controls');
const baseline=document.createElement('input'); baseline.type='checkbox'; baseline.checked=true; baseline.id='baseline'; baseline.onchange=()=>{Plotly.restyle('xz',{visible:baseline.checked},[1]); Plotly.restyle('plot3d',{visible:baseline.checked},[1]);}; const baselineLabel=document.createElement('label'); baselineLabel.htmlFor='baseline'; baselineLabel.style.color='#111827'; baselineLabel.append(baseline,' baseline: 0727 평균 (n=60)'); controls.append(baselineLabel);
data.sessions.forEach((s,i)=>{const box=document.createElement('input'); box.type='checkbox'; box.checked=i===0; box.id='session-'+s.id; box.onchange=()=>{const visible=box.checked; Plotly.restyle('xz',{visible},[4+i]); Plotly.restyle('plot3d',{visible},[4+i]);}; const label=document.createElement('label'); label.htmlFor=box.id; label.style.color=s.color; label.append(box,' '+s.id+' 평균 action EEF (n='+s.n+')'); controls.append(label);});
</script></html>""".replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
    output.write_text(html, encoding="utf-8")
    return medoid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "records/0727")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/analysis/eraser_approach_0727")
    parser.add_argument("--samples", type=int, default=160, help="time-normalized samples per trajectory")
    parser.add_argument("--voxel-mm", type=float, default=8.0, help="density-cloud voxel size")
    args = parser.parse_args()

    episodes = baseline_recordings_0727_folder()
    if not episodes:
        raise SystemExit(f"No timestamped 0727 recordings below {args.root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stack, reports = load_action_trajectories(episodes, args.samples)
    if len(stack) < 2:
        raise SystemExit("Fewer than two usable approach trajectories")
    names = [Path(str(row["episode"])).name for row in reports if row["included"]]
    session_averages: dict[str, np.ndarray] = {}
    session_counts: dict[str, int] = {}
    session_contact_counts: dict[str, int] = {}
    for session in SESSION_IDS:
        other_stack, other_reports = load_action_trajectories(session_recordings(session), args.samples)
        if len(other_stack):
            session_averages[session] = other_stack.mean(axis=0)
        else:
            session_averages[session] = np.empty((0, 3))
        session_counts[session] = len(other_stack)
        session_contact_counts[session] = sum(bool(row.get("firm_grasp_detected")) for row in other_reports if row.get("included"))
    image = args.output_dir / "eef_approach_distribution.png"
    medoid, occupied_voxels = plot(image, stack, names, args.voxel_mm)
    representative = next(Path(str(row["episode"])) for row in reports if row.get("included") and Path(str(row["episode"])).name == names[medoid])
    save_topview_overlay(
        args.output_dir / "topview_block_anchored_approach_overlay.png",
        next((representative / "videos" / "observation.images.top").rglob("*.mp4")),
        stack,
        medoid,
    )
    interactive = args.output_dir / "eef_approach_distribution_interactive.html"
    if save_interactive_html(interactive, stack, names, args.voxel_mm, session_averages, session_counts) != medoid:
        raise RuntimeError("static and interactive standard trajectory disagree")
    (args.output_dir / "approach_segments.json").write_text(json.dumps({"definition": "recording frame 0 to first firmly-grasped gripper plateau after the initial closing ramp; fallback is first closing frame", "eef_source": "recorded action (follower target), not observation.state", "firm_grasp_signal": {"search_window_after_first_close_frames": GRASP_SEARCH_FRAMES, "minimum_action_close": GRASP_MIN_CLOSE, "command_and_state_settled_step": GRASP_SETTLE_STEP, "hold_frames": GRASP_SETTLE_FRAMES}, "baseline": "time-normalized mean EEF trajectory over all included recordings", "resampled_points_per_episode": args.samples, "topview_background_episode": names[medoid], "episodes": reports}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "standard_trajectory_eef_mm.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["progress", "x_mm", "y_mm", "z_mm"])
        for progress, xyz in zip(np.linspace(0, 1, args.samples), stack.mean(axis=0)):
            writer.writerow([progress, *xyz])
    (args.output_dir / "session_average_action_eef.json").write_text(json.dumps({"definition": "recording frame 0 to stable action-state gripper stall; EEF is FK(action)", "samples": args.samples, "counts": session_counts, "contact_detected_counts": session_contact_counts, "trajectories": {session: path.tolist() for session, path in session_averages.items()}}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # 0805처럼 사람이 wrist 영상을 검토할 때 바로 seek할 수 있도록 stable-grasp
    # 후보 시점만 별도 표로 남긴다. action 기반 EEF와 contact 판정은 분리한다.
    _, review_reports = load_action_trajectories(session_recordings("0805"), args.samples)
    with (args.output_dir / "0805_grasp_contact_review.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["episode", "firm_grasp_detected", "firm_grasp_frame", "firm_grasp_seconds_30fps", "review_start_seconds", "review_end_seconds"])
        writer.writeheader()
        for row in review_reports:
            if not row.get("included"):
                continue
            frame = int(row["firm_grasp_frame"])
            seconds = frame / 30.0
            writer.writerow({"episode": Path(str(row["episode"])).name, "firm_grasp_detected": row["firm_grasp_detected"], "firm_grasp_frame": frame, "firm_grasp_seconds_30fps": f"{seconds:.3f}", "review_start_seconds": f"{max(0.0, seconds - 2.0):.3f}", "review_end_seconds": f"{seconds + 2.0:.3f}"})
    print(f"included {len(stack)}/{len(episodes)} recordings; baseline=mean trajectory; occupied density voxels={occupied_voxels}")
    print(f"session action-trajectory counts: {session_counts}")
    print(f"session contact detections: {session_contact_counts}")
    print(image)
    print(interactive)


if __name__ == "__main__":
    main()
