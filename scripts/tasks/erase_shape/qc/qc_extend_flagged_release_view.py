#!/usr/bin/env python3
"""SUSPECT/BAD로 판정된 녹화만, 종료 직후 15프레임을 리뷰 화면에 덧붙인다.

일회성 스크립트다 — `qc_recompute_all.py`/`qc_extract_review_frames.py`는
건드리지 않는다(전체 315개 재생성용 도구는 그대로 유지).

배경: 사람 리뷰 노트 81개 중 대부분이 "떨어지는게 (조금만) 더 보였으면"이었다.
확인해보니 그리퍼 열림(ramp) 자체는 문제가 아니었고(이 81개 평균 9.6프레임,
오히려 전체 평균보다 짧음), 지금 종료 기준(`release_done` = 그리퍼가 완전히
열리는 순간)에는 팔이 아직 거치대 위에 떠 있어서 지우개가 실제로 놓였는지
카메라에 가려 안 보이는 게 진짜 원인이었다. 팔이 물러나 지우개가 보이기까지는
예시 하나로 확인한 바 70프레임 이상 걸리기도 해서 자동 검출은 안 하고, 우선
15프레임만 붙여서 사람이 다시 보고 판단하게 한다.

학습용 `auto_end_frame`(QC 컷 경계)은 바꾸지 않는다 — 리뷰 화면에 보여주는
프레임만 늘어난다.

실행:
    python qc_extend_flagged_release_view.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import av
import cv2

REPO_ROOT = Path(__file__).resolve().parents[4]
QC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(QC_DIR))

from qc_core import read_episode  # noqa: E402
import numpy as np  # noqa: E402

DEFAULT_QC_JSON = REPO_ROOT / "configs/qc_review_all_315_release_done.json"
DEFAULT_VERDICTS_CSV = REPO_ROOT / "outputs/analysis/qc_review_315_release_done/verdicts.csv"
DEFAULT_FRAMES_DIR = REPO_ROOT / "outputs/analysis/qc_review_315_release_done/frames"

TARGET_VERDICTS = {"SUSPECT", "BAD"}
BASE_MARGIN = 20  # 기존 종료 창 크기 — qc_extract_review_frames.py 기본값과 동일, offset 이어붙이기 기준


def load_flagged(verdicts_csv: Path) -> set[str]:
    import csv

    with verdicts_csv.open(encoding="utf-8") as f:
        return {row["source_dataset"] for row in csv.DictReader(f) if row.get("verdict") in TARGET_VERDICTS}


def find_video(source_dataset: str, camera: str) -> Path | None:
    recording = REPO_ROOT / source_dataset
    matches = sorted(recording.glob(f"videos/observation.images.{camera}/chunk-*/file-*.mp4"))
    return matches[0] if matches else None


def extend_frames(video_path: Path, end: int, after_margin: int, out_dir: Path, camera: str, quality: int, resize: tuple[int, int] | None) -> int:
    """[end, end+after_margin) 프레임만 뽑아 기존 end_NN 시퀀스 뒤에 이어 붙인다."""
    wanted = set(range(end, end + after_margin))
    stop_at = end + after_margin
    saved = 0
    container = av.open(str(video_path))
    try:
        for idx, frame in enumerate(container.decode(video=0)):
            if idx >= stop_at:
                break
            if idx not in wanted:
                continue
            img = frame.to_ndarray(format="bgr24")
            if resize:
                img = cv2.resize(img, resize)
            offset = BASE_MARGIN + (idx - end)  # 기존 0..19 뒤에 20..(20+after_margin-1)로 이어짐
            cv2.imwrite(str(out_dir / f"{camera}_end_{offset:02d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            saved += 1
    finally:
        container.close()
    return saved


def extend_gripper_trace(path: Path, end: int, after_margin: int) -> dict:
    frame, _ = read_episode(path)
    total = len(frame)
    action = np.stack(frame["action"].to_numpy())
    state = np.stack(frame["observation.state"].to_numpy())
    window = list(range(end, min(end + after_margin, total)))
    return {
        "frame_index": window,
        "action": [round(float(action[i, 6]), 2) for i in window],
        "state": [round(float(state[i, 6]), 2) for i in window],
    }


def episode_out_dir(out_root: Path, episode: dict) -> Path:
    name = Path(episode["source_dataset"]).name
    return out_root / f"{episode['session']}__{name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--qc-json", type=Path, default=DEFAULT_QC_JSON)
    parser.add_argument("--verdicts-csv", type=Path, default=DEFAULT_VERDICTS_CSV)
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR)
    parser.add_argument("--after-margin", type=int, default=15)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--resize", type=str, default="640x360")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resize = None
    if args.resize:
        w, h = args.resize.lower().split("x")
        resize = (int(w), int(h))

    flagged = load_flagged(args.verdicts_csv)
    print(f"[INFO] SUSPECT/BAD {len(flagged)}개 대상")

    payload = json.loads(args.qc_json.read_text())
    episodes = payload["episodes"]
    by_key = {e["source_dataset"]: e for e in episodes}

    done, skipped = 0, []
    for key in sorted(flagged):
        episode = by_key.get(key)
        if episode is None:
            skipped.append((key, "qc-json에 없음"))
            continue
        end = episode["auto_end_frame"]
        if end is None:
            skipped.append((key, "auto_end_frame 없음"))
            continue

        out_dir = episode_out_dir(args.frames_dir, episode)
        saved_imgs = 0
        for camera in episode["cameras"]:
            video = find_video(key, camera)
            if video is None:
                skipped.append((key, f"video 없음: {camera}"))
                continue
            saved_imgs += extend_frames(video, end, args.after_margin, out_dir, camera, args.jpeg_quality, resize)

        extra_trace = extend_gripper_trace(REPO_ROOT / key, end, args.after_margin)
        gripper_end = episode.setdefault("gripper_end", {"frame_index": [], "action": [], "state": []})
        gripper_end["frame_index"] = gripper_end.get("frame_index", []) + extra_trace["frame_index"]
        gripper_end["action"] = gripper_end.get("action", []) + extra_trace["action"]
        gripper_end["state"] = gripper_end.get("state", []) + extra_trace["state"]

        done += 1
        print(f"[OK] {key}: 이미지 {saved_imgs}장, 궤적 {len(extra_trace['frame_index'])}프레임 추가")

    args.qc_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"[WRITE] {args.qc_json} (81개 중 {done}개 갱신, 스킵 {len(skipped)}개)")
    for key, reason in skipped:
        print(f"  - SKIP {key}: {reason}")


if __name__ == "__main__":
    main()
