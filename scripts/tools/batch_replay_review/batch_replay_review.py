from __future__ import annotations

"""records/0802_joint4_corrected, 0804_joint4_corrected 같은 "녹화 폴더 묶음"을
하나씩 순회하며 리플레이로 눈검사하는 도구.

왜 새로 만들었나 — scripts/tools/piper_replay_player.py는 "데이터셋 1개 안의
에피소드"를 ↑/↓로 넘긴다. 그런데 이 프로젝트의 원본 녹화는 녹화 1건이 곧
데이터셋 1개(에피소드 1개)라서(CLAUDE.md 참고) 기존 플레이어로는 폴더 경계를
못 넘는다. 93개를 손으로 하나씩 실행하는 대신 폴더 목록을 순회하게 했다.

렌더링(영상 디코딩, 패널 그리기)은 piper_replay_player의 함수를 그대로 import해
쓴다 — 표시 형식이 기존 GUI의 Play 버튼과 어긋나면 비교가 안 되기 때문.

판정 결과는 CSV에 계속 적어두고 다음 실행 때 이어서 볼 수 있다(--resume).
93개를 한 번에 다 보기는 어려우므로 중단/재개가 사실상 필수다.

기본은 화면 재생만 하고 하드웨어를 건드리지 않는다. x 키로 현재 녹화를 실제
follower 팔에 재생할 수도 있는데, 이건 --real-robot과 확인 토큰이 둘 다 있어야
열리고 창 안에서도 x를 두 번 눌러야 나간다(팔이 실제로 움직이는 경로라 오조작
한 번으로 나가면 안 된다).

사용 예:
    python scripts/tools/batch_replay_review/batch_replay_review.py            # 기본 두 폴더
    python scripts/tools/batch_replay_review/batch_replay_review.py --resume   # 미판정만
    python scripts/tools/batch_replay_review/batch_replay_review.py --summary  # 진행률만 출력
    python scripts/tools/batch_replay_review/batch_replay_review.py \
        --real-robot --real-robot-confirm I_UNDERSTAND_REAL_ROBOT              # x 키로 실물 재생
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

# piper_replay_player는 패키지가 아니라 scripts/tools 밑의 단독 스크립트라
# sys.path에 그 디렉터리를 직접 넣어야 import된다.
TOOLS_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = TOOLS_DIR.parent.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import piper_replay_player as rp  # noqa: E402


DEFAULT_GROUPS = [
    "records/0802_joint4_corrected",
    "records/0804_joint4_corrected",
]

DEFAULT_REVIEW_CSV = "outputs/analysis/replay_review/joint4_corrected_review.csv"

# 1/2/3 키로 찍는 판정. 색은 패널 배지에 그대로 쓰임(BGR).
VERDICTS: dict[str, tuple[str, tuple[int, int, int]]] = {
    "1": ("OK", (110, 220, 130)),
    "2": ("SUSPECT", (80, 200, 250)),
    "3": ("BAD", (90, 90, 250)),
}
VERDICT_COLORS = {name: color for name, color in VERDICTS.values()}
VERDICT_COLORS["-"] = (150, 150, 150)

# 이 판정을 찍으면 비고 창이 자동으로 뜬다 — "왜 버렸는지"를 그 자리에서
# 안 적으면 나중에 CSV만 보고는 재현이 안 된다.
NOTE_ON_VERDICT = {"BAD", "SUSPECT"}


# --------------------------------------------------------------------------
# 녹화 폴더 수집 / 판정 저장
# --------------------------------------------------------------------------


def discover_recordings(groups: list[str]) -> list[Path]:
    """각 그룹 폴더 밑에서 meta/info.json을 가진 디렉터리를 녹화로 인정.

    .rar 같은 압축 파일이 섞여 있으므로(records/0802_joint4_corrected 참고)
    디렉터리 여부만으로 거르면 안 되고 메타 존재를 확인해야 한다."""
    found: list[Path] = []
    for group in groups:
        group_path = Path(group)
        if not group_path.is_absolute():
            group_path = REPO_ROOT / group_path
        if not group_path.is_dir():
            print(f"[WARN] group not found, skipping: {group_path}")
            continue
        entries = sorted(p for p in group_path.iterdir() if (p / "meta" / "info.json").is_file())
        if not entries:
            print(f"[WARN] no LeRobot recordings under: {group_path}")
        found.extend(entries)
    return found


def load_reviews(csv_path: Path) -> dict[str, dict[str, str]]:
    if not csv_path.is_file():
        return {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return {row["recording"]: row for row in csv.DictReader(f) if row.get("recording")}


def save_reviews(csv_path: Path, reviews: dict[str, dict[str, str]], order: list[Path]) -> None:
    """매 판정마다 전체를 다시 쓴다 — 93행짜리라 비용이 무의미하고,
    중간에 강제 종료돼도 결과가 남는 편이 훨씬 중요하다."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["recording", "group", "verdict", "frames", "note", "reviewed_at"]
    keys = [p.name for p in order]
    keys += [k for k in reviews if k not in set(keys)]  # 목록에 없는 과거 기록도 보존
    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key in keys:
            row = reviews.get(key)
            if row:
                writer.writerow({k: row.get(k, "") for k in fields})
    tmp.replace(csv_path)


def print_summary(recordings: list[Path], reviews: dict[str, dict[str, str]]) -> None:
    counts: dict[str, int] = {}
    for rec in recordings:
        verdict = reviews.get(rec.name, {}).get("verdict", "")
        counts[verdict or "(unreviewed)"] = counts.get(verdict or "(unreviewed)", 0) + 1
    print(f"\n총 {len(recordings)}건")
    for name in ["OK", "SUSPECT", "BAD", "(unreviewed)"]:
        if name in counts:
            print(f"  {name:<12} {counts[name]}")
    bad = [r.name for r in recordings if reviews.get(r.name, {}).get("verdict") in ("SUSPECT", "BAD")]
    if bad:
        print("\n재확인 필요:")
        for name in bad:
            print(f"  [{reviews[name]['verdict']}] {name}")


# --------------------------------------------------------------------------
# 한 녹화 로드 (기존 플레이어 헬퍼 재사용)
# --------------------------------------------------------------------------


def load_recording(root: Path, view: str, data_key: str, action_key: str) -> dict[str, Any] | None:
    """영상 전체를 메모리에 디코딩해 올린다. 실패하면 None을 돌려 호출부가 건너뛴다.

    _load_video_frames가 통째로 디코딩하는 방식이라(AV1 seek 깨짐 회피) 녹화당
    수백 MB를 쓸 수 있다 — 그래서 다음 녹화로 넘어갈 때 이전 것을 붙들지 않는다."""
    try:
        info = rp._load_info(root)
    except FileNotFoundError as e:
        print(f"[WARN] {root.name}: {e}")
        return None

    try:
        data_path = rp._data_path(root, info, None, 0)
        df = pd.read_parquet(data_path)
        if "episode_index" in df.columns:
            df = df[df["episode_index"] == 0].reset_index(drop=True)
    except Exception as e:
        print(f"[WARN] {root.name}: parquet 읽기 실패 ({e})")
        return None
    if df.empty:
        print(f"[WARN] {root.name}: 데이터가 비어 있음")
        return None

    video_keys = rp._video_keys(info, None, view)
    video_frames: dict[str, list[np.ndarray] | None] = {}
    for key in video_keys:
        path = rp._video_path(root, info, None, 0, key)
        is_depth = rp._is_depth_key(info, key)
        video_frames[key] = rp._load_video_frames(
            path, is_depth=is_depth, depth_params=rp._depth_params(info, key) if is_depth else None
        )

    lengths = [len(f) for f in video_frames.values() if f is not None]
    total_frames = min(len(df), *lengths) if lengths else len(df)

    return {
        "info": info,
        "df": df,
        "video_keys": video_keys,
        "video_frames": video_frames,
        "total_frames": total_frames,
        "fps": float(info.get("fps") or 30),
        "joint_names": rp._feature_names(info, data_key),
        "action_key": action_key if action_key in df.columns else None,
    }


# --------------------------------------------------------------------------
# 화면 구성
# --------------------------------------------------------------------------


def draw_header(
    panel: np.ndarray,
    rec: Path,
    index: int,
    total: int,
    verdict: str,
    frame_pos: int,
    total_frames: int,
    paused: bool,
) -> None:
    """패널 상단을 덮어써서 "지금 무슨 녹화를 보고 있는지"를 항상 띄운다.

    _panel()이 찍어둔 "Piper replay player" 타이틀 자리를 그대로 재사용 —
    90개를 넘기다 보면 어느 파일인지 놓치는 게 제일 흔한 실수라 가장 위에 크게 둔다.
    이 띠가 _panel()의 frame/timestamp 줄까지 덮으므로 아래에서 다시 그려준다.

    표시 문자열은 전부 ASCII다 — cv2.putText는 한글을 '?'로 그린다."""
    w = panel.shape[1]
    cv2.rectangle(panel, (0, 0), (w, 92), (44, 48, 56), -1)

    # 이름이 패널 폭을 넘으면 앞을 잘라 뒤쪽(타임스탬프)을 남긴다 — 같은 태스크
    # 이름이 반복되므로 구분에 쓰이는 건 뒤쪽 시각이다.
    name = rec.name
    scale = 0.52
    while cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > w - 24 and len(name) > 8:
        name = name[1:]
    if name != rec.name:
        name = "…" + name[1:]

    rp._draw_text(panel, name, (12, 24), scale=scale, color=(255, 255, 255), thickness=1)
    rp._draw_text(panel, rec.parent.name, (12, 44), scale=0.44, color=(170, 200, 240))
    rp._draw_text(panel, f"[{index + 1}/{total}]", (12, 64), scale=0.46, color=(200, 200, 200))

    badge = verdict or "-"
    color = VERDICT_COLORS.get(badge, (150, 150, 150))
    (bw, _), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(panel, (w - bw - 26, 48), (w - 10, 70), color, -1)
    rp._draw_text(panel, badge, (w - bw - 18, 64), scale=0.5, color=(20, 20, 20), thickness=2)

    # 헤더 띠가 가린 _panel()의 진행 정보를 복원
    rp._draw_text(panel, f"frame {frame_pos + 1}/{total_frames}  {'PAUSED' if paused else 'PLAYING'}",
                  (12, 85), scale=0.45, color=(190, 220, 255))


def draw_controls(panel: np.ndarray, robot_enabled: bool, note: str = "") -> None:
    """_panel()의 기본 컨트롤 안내는 에피소드 이동 기준이라 여기선 맞지 않는다.
    ASCII만 쓴다 — cv2.putText는 한글을 못 그린다."""
    h = panel.shape[0]
    cv2.rectangle(panel, (0, h - 172), (panel.shape[1], h), (28, 30, 34), -1)

    if note:
        # 한글 비고는 cv2가 '?'로 그리므로 있다는 사실만 표시하고 내용은 터미널에서 본다
        ascii_only = note.encode("ascii", "ignore").decode()
        shown = ascii_only if ascii_only.strip() else "(see terminal / csv)"
        rp._draw_text(panel, f"note: {shown[:44]}", (12, h - 146), scale=0.43, color=(120, 210, 240))

    y = h - 118
    robot_line = ("x x: REPLAY ON REAL ROBOT", (110, 160, 255)) if robot_enabled else \
                 ("x: real robot (disabled)", (120, 120, 120))
    for line, col in [
        ("ENTER / n / ]: next rec    p / [: prev rec", (255, 255, 255)),
        ("1:OK  2:SUSPECT(+note)  3:BAD(+note)  -> auto next", (200, 230, 205)),
        robot_line,
        ("space pause   r restart   , . a d seek   +/- speed", (185, 185, 185)),
        ("m: note (opens input window)  f: path  q/esc: quit", (185, 185, 185)),
    ]:
        rp._draw_text(panel, line, (12, y), scale=0.43, color=col)
        y += 18


# --------------------------------------------------------------------------
# 실물 로봇 재생
# --------------------------------------------------------------------------

REAL_ROBOT_TOKEN = "I_UNDERSTAND_REAL_ROBOT"


def run_on_robot(rec: Path, args: argparse.Namespace) -> int:
    """현재 녹화를 follower 팔에 실제로 재생한다.

    커맨드를 직접 조립하지 않고 replay_one_on_robot.sh를 부른다 — 안전 인자
    (max_relative_target, effort 컷오프, use_action_offset=false, torque 해제 방식)가
    run_common.sh 한 곳에서만 정의되게 두려는 것. 여기서 따로 조립하면 그쪽이 바뀔 때
    이 도구만 옛 설정으로 팔을 움직이게 된다.

    데이터셋 경로는 REVIEW_ 접두어로 넘긴다 — DATASET_ROOT로 넘기면 래퍼가 읽는
    configs/recording.env에 같은 이름이 있어서 덮여버린다(replay_one_on_robot.sh 주석 참고)."""
    script = Path(args.replay_script)
    if not script.is_absolute():
        script = REPO_ROOT / script
    if not script.is_file():
        print(f"[ERROR] replay 스크립트를 찾을 수 없습니다: {script}")
        return 1

    env = dict(os.environ)
    env.update({
        "REVIEW_FOLLOWER_PORT": args.follower_port,
        "REVIEW_DATASET_ROOT": str(rec),
    })
    if args.dry_run:
        env["DRY_RUN"] = "true"

    print("=" * 72)
    print(f"[ROBOT] 실물 재생: {rec}")
    print(f"[ROBOT] port={args.follower_port}  script={script}" + ("  (DRY_RUN)" if args.dry_run else ""))
    print("[ROBOT] 중단하려면 이 터미널에서 Ctrl+C — E-STOP에 손을 두세요.")
    print("=" * 72, flush=True)  # 자식 출력과 순서가 섞이지 않도록 먼저 내보낸다
    try:
        proc = subprocess.run([str(script)], env=env, cwd=str(REPO_ROOT))
        rc = proc.returncode
    except KeyboardInterrupt:
        # replay 자식이 Ctrl+C로 죽는 건 정상 종료 경로다(사용자가 멈춘 것).
        print("\n[ROBOT] 사용자 중단(Ctrl+C).")
        rc = 130
    except Exception as e:
        print(f"[ROBOT] 실행 실패: {e}")
        rc = 1
    print(f"[ROBOT] 종료 코드 {rc}\n")
    return rc


def robot_gate_reason(args: argparse.Namespace) -> str | None:
    """실물 재생이 막혀 있으면 그 이유를, 열려 있으면 None을 돌려준다.

    팔이 실제로 움직이는 경로라 플래그 두 개를 모두 요구한다 — 저장소의 다른
    실물 도구(piper_infer_runner.py)와 같은 관례."""
    if not args.real_robot:
        return "--real-robot 없음"
    if args.real_robot_confirm != REAL_ROBOT_TOKEN:
        return f"--real-robot-confirm {REAL_ROBOT_TOKEN} 없음"
    return None


def ask_note_dialog(rec: Path, current: str) -> str | None:
    """비고 입력 창. 저장할 문자열을 돌려주고, 취소면 None.

    tkinter를 쓰는 이유 — OpenCV 창은 텍스트 입력 위젯이 없고 cv2.putText가 한글을
    '?'로 그린다. tkinter는 한글 입력/표시가 되고 저장소 GUI(teleop_ui)도 tkinter라
    의존성이 늘지 않는다. 창을 못 띄우는 환경(SSH 등)에서는 None 대신 예외를 내서
    호출부가 터미널 입력으로 넘어가게 한다."""
    import tkinter as tk
    from tkinter import simpledialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)  # OpenCV 창 뒤에 숨는 걸 막는다
    try:
        return simpledialog.askstring(
            "비고 입력",
            f"{rec.name}\n\n비고를 입력하세요 (비우면 삭제):",
            initialvalue=current,
            parent=root,
        )
    finally:
        root.destroy()


def edit_note(rec: Path, reviews: dict[str, dict[str, str]], csv_path: Path,
              recordings: list[Path], total_frames: int) -> None:
    """비고 창을 띄우고 결과를 CSV에 반영한다. m 키와 BAD 판정이 함께 쓴다."""
    current = reviews.get(rec.name, {}).get("note", "")
    try:
        text = ask_note_dialog(rec, current)
    except Exception as e:
        # 창을 못 띄우는 환경(SSH 등)이면 터미널 입력으로 물러난다
        print(f"[WARN] 비고 창을 열 수 없어 터미널로 받습니다 ({e})")
        print(f"    비고 입력 ({rec.name}) — 그냥 Enter면 취소:")
        try:
            text = input("    > ")
        except (EOFError, KeyboardInterrupt):
            text = None
            print()

    if text is None:  # 창의 Cancel / Esc
        print("    비고 입력 취소")
        return

    row = reviews.get(rec.name) or {
        "recording": rec.name, "group": rec.parent.name,
        "verdict": "", "frames": str(total_frames), "note": "", "reviewed_at": "",
    }
    row["note"] = text.strip()  # 빈 값으로 확인하면 삭제
    row["frames"] = str(total_frames)
    row["reviewed_at"] = datetime.now().isoformat(timespec="seconds")
    reviews[rec.name] = row
    save_reviews(csv_path, reviews, recordings)
    print(f"    비고 저장: {row['note'] or '(삭제됨)'}")


def countdown_abort(window: str, rec: Path, seconds: float) -> bool:
    """실물 재생 직전 카운트다운. 키가 눌리면 True(=이번 건 건너뛰기)를 돌려준다.

    --auto-robot은 사람이 확인키를 안 누르므로, 팔이 움직이기 전에 멈출 틈을
    반드시 줘야 한다."""
    end = time.monotonic() + seconds
    while True:
        left = end - time.monotonic()
        if left <= 0:
            return False
        canvas = np.full((260, 900, 3), (20, 20, 140), dtype=np.uint8)
        rp._draw_text(canvas, f"REAL ROBOT IN {left:.1f}s", (24, 70), scale=1.0,
                      color=(255, 255, 255), thickness=2)
        rp._draw_text(canvas, rec.name, (24, 130), scale=0.7, color=(255, 255, 255), thickness=2)
        rp._draw_text(canvas, "press any key to SKIP this one", (24, 180), scale=0.6,
                      color=(200, 220, 255))
        rp._draw_text(canvas, "keep hand on E-STOP", (24, 220), scale=0.6, color=(200, 220, 255))
        cv2.imshow(window, canvas)
        if cv2.waitKey(30) & 0xFF != 255:
            return True


def loading_screen(window: str, rec: Path, index: int, total: int) -> None:
    """영상 통째 디코딩이 몇 초 걸려서 그 동안 창이 멈춘 것처럼 보인다 —
    무슨 파일을 읽는 중인지 띄워준다."""
    canvas = np.full((260, 900, 3), (28, 30, 34), dtype=np.uint8)
    rp._draw_text(canvas, f"loading [{index + 1}/{total}]", (24, 60), scale=0.7, color=(190, 220, 255), thickness=2)
    rp._draw_text(canvas, rec.name, (24, 110), scale=0.7, color=(255, 255, 255), thickness=2)
    rp._draw_text(canvas, str(rec.parent.name), (24, 150), scale=0.5, color=(170, 200, 240))
    cv2.imshow(window, canvas)
    cv2.waitKey(1)


# --------------------------------------------------------------------------
# 메인 루프
# --------------------------------------------------------------------------


def review(args: argparse.Namespace) -> int:
    recordings = discover_recordings(args.groups)
    if not recordings:
        print("[ERROR] 검사할 녹화를 찾지 못했습니다.")
        return 1

    csv_path = Path(args.review_csv)
    if not csv_path.is_absolute():
        csv_path = REPO_ROOT / csv_path
    reviews = load_reviews(csv_path)

    if args.summary:
        print_summary(recordings, reviews)
        return 0

    todo = recordings
    if args.resume:
        # 기본은 OK만 건너뛴다 — SUSPECT/BAD는 "다시 봐야 할 것"으로 찍은 값이라
        # 건너뛰면 재확인 자체가 불가능해진다.
        skip = {v.strip().upper() for v in args.skip_verdicts.split(",") if v.strip()}
        todo = [r for r in recordings if reviews.get(r.name, {}).get("verdict", "").upper() not in skip]
        skipped = len(recordings) - len(todo)
        print(f"[INFO] resume: 전체 {len(recordings)}건 중 {skipped}건 건너뜀"
              f"(verdict in {sorted(skip)}) → {len(todo)}건 확인")
        if not todo:
            print("[INFO] 건너뛸 것 외에 남은 녹화가 없습니다.")
            print_summary(recordings, reviews)
            return 0

    reason = robot_gate_reason(args)
    if reason:
        print(f"[INFO] 실물 로봇 재생 OFF ({reason}) — 영상/관절값 재생만 합니다.")
    else:
        mode = "녹화마다 자동 실행" if args.auto_robot else "x 를 두 번 누르면 실행"
        print(f"[WARN] 실물 로봇 재생 ON (port={args.follower_port}, {mode})"
              + (" [DRY_RUN]" if args.dry_run else "")
              + " — follower 팔이 실제로 움직입니다. E-STOP 준비하세요.")

    window = args.window_name
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    index = 0
    speed = args.speed
    quit_all = False

    while 0 <= index < len(todo) and not quit_all:
        rec = todo[index]
        loading_screen(window, rec, index, len(todo))
        loaded = load_recording(rec, args.view, args.data_key, args.action_key)
        if loaded is None:
            # 못 읽는 녹화도 결과에 남겨야 나중에 "안 본 것"과 구분된다.
            reviews[rec.name] = {
                "recording": rec.name,
                "group": rec.parent.name,
                "verdict": "LOAD_FAIL",
                "frames": "0",
                "note": "load failed",
                "reviewed_at": datetime.now().isoformat(timespec="seconds"),
            }
            save_reviews(csv_path, reviews, recordings)
            index += 1
            continue

        df = loaded["df"]
        total_frames = loaded["total_frames"]
        fps = float(args.fps or loaded["fps"])

        # 터미널에도 "지금 무슨 파일을 보고 있는지"를 절대경로로 남긴다 —
        # 화면에는 이름만 뜨므로, 문제를 찾았을 때 바로 복사해 쓸 경로가 필요하다.
        print(f"\n[{index + 1}/{len(todo)}] {rec.name}  (frames={total_frames}, fps={fps:g})")
        print(f"    {rec}")

        frame_pos = 0
        paused = args.start_paused
        last_tick = time.monotonic()
        step = 0  # 이 녹화를 벗어날 때 이동량(+1 다음 / -1 이전 / 0 제자리)
        running = True
        arm_confirm_until = 0.0  # x 두 번 누르기 확인 창(실물 재생)
        notice, notice_until = "", 0.0

        # --auto-robot: 녹화를 열자마자 실물에 재생한다. 매번 x를 두 번 누르지
        # 않아도 되지만, 그만큼 "모르는 새 팔이 움직이는" 상태이므로 카운트다운
        # 동안 아무 키나 누르면 이번 건은 건너뛴다.
        if args.auto_robot and robot_gate_reason(args) is None:
            if countdown_abort(window, rec, args.auto_robot_delay):
                print("[ROBOT] 사용자가 이번 녹화의 실물 재생을 건너뛰었습니다.")
            else:
                cv2.destroyWindow(window)
                cv2.waitKey(1)
                run_on_robot(rec, args)
                cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        while running:
            frames = []
            for key in loaded["video_keys"]:
                decoded = loaded["video_frames"].get(key)
                if decoded is None:
                    frame = np.full((240, 424, 3), (20, 20, 20), dtype=np.uint8)
                    rp._draw_text(frame, f"video not found: {key}", (18, 38), color=(80, 80, 255))
                elif frame_pos >= len(decoded):
                    frame = np.full((240, 424, 3), (20, 20, 20), dtype=np.uint8)
                    rp._draw_text(frame, f"missing frame: {key}", (18, 38), color=(80, 170, 255))
                else:
                    frame = decoded[frame_pos].copy()
                rp._draw_text(frame, key, (12, 24), color=(40, 230, 255), thickness=2)
                frames.append(rp._resize_to_height(frame, args.video_height))

            video_strip = cv2.vconcat(frames)
            row_pos = min(frame_pos, len(df) - 1)
            panel = rp._panel(
                df.iloc[row_pos],
                frame_pos,
                total_frames,
                loaded["joint_names"],
                args.data_key,
                loaded["action_key"],
                args.panel_width,
                video_strip.shape[0],
                paused,
            )
            draw_header(panel, rec, index, len(todo), reviews.get(rec.name, {}).get("verdict", ""),
                        frame_pos, total_frames, paused)
            draw_controls(panel, robot_enabled=robot_gate_reason(args) is None)
            # 배속은 헤더 띠 안에 — 아래는 관절 표라 겹치면 값이 가려진다
            rp._draw_text(panel, f"{speed:.2f}x", (panel.shape[1] - 66, 85), scale=0.45, color=(190, 220, 255))

            # 녹화 이름은 영상 위에도 한 번 더 — 영상만 크게 띄워놓고 볼 때를 대비.
            rp._draw_text(video_strip, rec.name, (12, video_strip.shape[0] - 14), scale=0.55,
                          color=(255, 255, 255), thickness=2)

            canvas = cv2.hconcat([video_strip, panel])
            if time.monotonic() < notice_until:
                # 실물 재생 확인처럼 놓치면 안 되는 알림은 영상 한가운데 크게
                (tw, th), _ = cv2.getTextSize(notice, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
                cx, cy = (canvas.shape[1] - tw) // 2, canvas.shape[0] // 2
                cv2.rectangle(canvas, (cx - 18, cy - th - 16), (cx + tw + 18, cy + 16), (20, 20, 160), -1)
                cv2.putText(canvas, notice, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow(window, canvas)
            key = cv2.waitKey(max(1, int(1000 / (fps * speed))) if not paused else 30) & 0xFF

            verdict_key = chr(key) if 32 <= key < 127 else ""

            if verdict_key in VERDICTS:
                verdict = VERDICTS[verdict_key][0]
                reviews[rec.name] = {
                    "recording": rec.name,
                    "group": rec.parent.name,
                    "verdict": verdict,
                    "frames": str(total_frames),
                    "note": reviews.get(rec.name, {}).get("note", ""),
                    "reviewed_at": datetime.now().isoformat(timespec="seconds"),
                }
                save_reviews(csv_path, reviews, recordings)
                print(f"    -> {verdict}")
                # BAD처럼 "왜"가 중요한 판정은 넘어가기 전에 비고를 받는다
                if verdict in NOTE_ON_VERDICT and not args.no_note_prompt:
                    paused = True
                    edit_note(rec, reviews, csv_path, recordings, total_frames)
                if not args.no_auto_advance:
                    step = 1
                    running = False
            elif key in (ord("q"), 27):
                quit_all = True
                running = False
            elif key in (ord("n"), ord("]"), 13, 10):  # n, ], Enter
                step = 1
                running = False
            elif key in (ord("p"), ord("[")):
                step = -1
                running = False
            elif key == ord("f"):
                print(f"    {rec}")
            elif key == ord("m"):
                paused = True  # 입력하는 동안 영상이 흘러가면 위치를 잃는다
                edit_note(rec, reviews, csv_path, recordings, total_frames)
            elif key in (ord("x"), ord("X")):
                reason = robot_gate_reason(args)
                if reason:
                    print(f"[ROBOT] 실물 재생이 꺼져 있습니다 ({reason}).")
                    notice, notice_until = f"REAL ROBOT DISABLED ({reason})", time.monotonic() + 2.5
                elif time.monotonic() < arm_confirm_until:
                    # 두 번째 x — 실행. 창을 닫고 나서 돌린다: 팔이 움직이는 동안
                    # OpenCV 창이 응답 없이 떠 있으면 E-STOP 판단을 방해한다.
                    arm_confirm_until = 0.0
                    cv2.destroyWindow(window)
                    cv2.waitKey(1)
                    run_on_robot(rec, args)
                    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                    paused = True
                else:
                    arm_confirm_until = time.monotonic() + 3.0
                    notice, notice_until = "PRESS X AGAIN -> MOVE REAL ROBOT", arm_confirm_until
                    print("[ROBOT] 3초 안에 x를 한 번 더 누르면 실물 팔이 움직입니다.")
            elif key in (ord("+"), ord("=")):
                speed = min(4.0, speed + 0.25)
            elif key == ord("-"):
                speed = max(0.25, speed - 0.25)
            elif key == ord(" "):
                paused = not paused
            elif key == ord("r"):
                frame_pos, paused = 0, False
            elif key in (81, ord(",")):
                frame_pos, paused = max(0, frame_pos - 1), True
            elif key in (83, ord(".")):
                frame_pos, paused = min(total_frames - 1, frame_pos + 1), True
            elif key == ord("a"):
                frame_pos, paused = max(0, frame_pos - 10), True
            elif key == ord("d"):
                frame_pos, paused = min(total_frames - 1, frame_pos + 10), True

            if not paused and running:
                now = time.monotonic()
                if now - last_tick >= 1.0 / max(1e-6, fps * speed):
                    frame_pos += 1
                    last_tick = now
                if frame_pos >= total_frames:
                    # 끝나면 멈춰서 판정을 기다린다 — 자동으로 넘어가면
                    # 판정 없이 지나가버려서 무엇을 봤는지 남지 않는다.
                    frame_pos = total_frames - 1
                    paused = True

        # 다음 녹화로 넘어가기 전에 디코딩 프레임을 놓아준다(수백 MB).
        loaded.clear()
        index = max(0, index + step) if step else index
        if step > 0 and index >= len(todo):
            break

    cv2.destroyAllWindows()
    print(f"\n판정 저장: {csv_path}")
    print_summary(recordings, reviews)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="녹화 폴더 묶음을 리플레이로 순회 검사하고 판정을 CSV에 남긴다.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS,
                   help="검사할 녹화 묶음 폴더들 (repo 상대경로 가능)")
    p.add_argument("--review-csv", default=DEFAULT_REVIEW_CSV, help="판정 결과 CSV 경로")
    p.add_argument("--resume", action="store_true",
                   help="CSV에 이미 판정된 녹화를 건너뛴다 (기본: OK만 건너뜀)")
    p.add_argument("--skip-verdicts", default="OK",
                   help="--resume 시 건너뛸 verdict 목록(쉼표 구분). "
                        "예: OK,BAD / 전부 건너뛰려면 OK,SUSPECT,BAD,LOAD_FAIL")
    p.add_argument("--summary", action="store_true", help="재생 없이 진행 상황만 출력하고 종료")
    p.add_argument("--view", default="both", choices=["both", "rgb", "depth"],
                   help="표시할 카메라 종류 (piper_replay_player와 동일)")
    p.add_argument("--data-key", default="observation.state")
    p.add_argument("--action-key", default="action")
    p.add_argument("--fps", type=float, default=None, help="재생 FPS 강제 지정")
    p.add_argument("--speed", type=float, default=1.0, help="배속")
    p.add_argument("--start-paused", action="store_true", help="일시정지 상태로 시작")
    p.add_argument("--video-height", type=int, default=360, help="카메라 한 개당 표시 높이")
    p.add_argument("--panel-width", type=int, default=470)
    p.add_argument("--no-auto-advance", action="store_true",
                   help="판정해도 자동으로 다음 녹화로 넘어가지 않음")
    p.add_argument("--no-note-prompt", action="store_true",
                   help=f"{'/'.join(sorted(NOTE_ON_VERDICT))} 판정 시 비고 창을 자동으로 띄우지 않음")
    p.add_argument("--window-name", default="Batch replay review (joint4_corrected)")

    robot = p.add_argument_group("실물 로봇 재생 (x 키)")
    robot.add_argument("--real-robot", action="store_true",
                       help="x 키로 현재 녹화를 실제 follower 팔에 재생하도록 허용")
    robot.add_argument("--real-robot-confirm", default="",
                       help=f"실물 재생을 열려면 {REAL_ROBOT_TOKEN} 을 그대로 넘겨야 함")
    robot.add_argument("--follower-port", default=os.environ.get("FOLLOWER_PORT", "can_follower"),
                       help="follower CAN 인터페이스 이름")
    robot.add_argument("--replay-script",
                       default="scripts/tools/batch_replay_review/replay_one_on_robot.sh",
                       help="실물 재생에 쓸 래퍼 (안전 인자는 run_common.sh에서 가져온다)")
    robot.add_argument("--auto-robot", action="store_true",
                       help="녹화를 열 때마다 x를 누르지 않아도 자동으로 실물 재생 "
                            "(카운트다운 중 아무 키나 누르면 그 건은 건너뜀)")
    robot.add_argument("--auto-robot-delay", type=float, default=3.0,
                       help="--auto-robot 카운트다운 시간(초)")
    robot.add_argument("--dry-run", action="store_true",
                       help="실물 재생 시 커맨드만 출력하고 실행하지 않음 (DRY_RUN=true)")
    return p.parse_args()


def main() -> None:
    raise SystemExit(review(parse_args()))


if __name__ == "__main__":
    main()
