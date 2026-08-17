#!/usr/bin/env python3
"""QC 재계산 결과(`qc_recompute_all.py` 출력)를 읽어 사람 리뷰용 프레임을 뽑는다.

각 녹화마다 시작 프레임 직후 20장, 종료 프레임 직전 20장만 저장한다. 전체
프레임을 메모리에 올리지 않는다(`batch_replay_review.py`처럼 비디오 전체를
리스트로 들고 있는 방식은 315개×카메라 규모에서 부적절) — PyAV로 순차
디코딩하면서 필요한 인덱스가 나올 때만 JPEG로 쓰고 나머지는 버린다.

`end_frame` 이후(팔이 원위치로 복귀하는 구간)는 디코딩할 필요가 없으므로 그
지점에서 바로 멈춘다. 이 저장소 비디오는 seek(`cv2.VideoCapture.set`)가 코덱에
따라 깨질 수 있다는 경고가 프로젝트 관례로 남아 있어(`piper_replay_player.py`),
seek 대신 순차 디코딩을 그대로 쓴다 — HEVC라 어차피 수백~천 프레임 디코딩에
1초 남짓이라 비용 문제도 아니다.

실행:
    python scripts/tools/qc_extract_review_frames.py \
        --qc-json configs/qc_review_all_315.json \
        --out-dir outputs/analysis/qc_review_315/frames
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import av
import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QC_JSON = REPO_ROOT / "configs/qc_review_all_315.json"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs/analysis/qc_review_315/frames"


def find_video(source_dataset: str, camera: str) -> Path | None:
    """source_dataset은 qc_recompute_all.py가 REPO_ROOT 기준으로 저장한 상대경로다."""
    recording = REPO_ROOT / source_dataset
    matches = sorted(recording.glob(f"videos/observation.images.{camera}/chunk-*/file-*.mp4"))
    return matches[0] if matches else None


def extract_video(video_path: Path, start: int, end: int, margin: int, out_dir: Path, camera: str, quality: int, resize: tuple[int, int] | None) -> int:
    wanted_start = set(range(max(0, start), min(start + margin, end)))
    wanted_end = set(range(max(start, end - margin), end))
    if not wanted_start and not wanted_end:
        return 0
    stop_at = end
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    container = av.open(str(video_path))
    try:
        for idx, frame in enumerate(container.decode(video=0)):
            if idx >= stop_at:
                break
            in_start = idx in wanted_start
            in_end = idx in wanted_end
            if not in_start and not in_end:
                continue
            img = frame.to_ndarray(format="bgr24")
            if resize:
                img = cv2.resize(img, resize)
            if in_start:
                offset = idx - start
                cv2.imwrite(str(out_dir / f"{camera}_start_{offset:02d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
                saved += 1
            if in_end:
                offset = idx - (end - margin)
                cv2.imwrite(str(out_dir / f"{camera}_end_{offset:02d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
                saved += 1
    finally:
        container.close()
    return saved


def expected_count(cameras: list[str], margin: int) -> int:
    return len(cameras) * 2 * margin


def episode_out_dir(out_root: Path, episode: dict) -> Path:
    name = Path(episode["source_dataset"]).name
    return out_root / f"{episode['session']}__{name}"


def extract_episode(episode: dict, out_root: Path, margin: int, quality: int, resize: tuple[int, int] | None, force: bool) -> tuple[str, int, list[str]]:
    key = episode["source_dataset"]
    errors: list[str] = []
    out_dir = episode_out_dir(out_root, episode)
    if not force and out_dir.is_dir():
        existing = len(list(out_dir.glob("*.jpg")))
        if existing >= expected_count(episode["cameras"], margin):
            return key, existing, errors

    start, end = episode["auto_start_frame"], episode["auto_end_frame"]
    if start is None or end is None or end <= start:
        return key, 0, [f"invalid range start={start} end={end}"]

    total = 0
    for camera in episode["cameras"]:
        video = find_video(key, camera)
        if video is None:
            errors.append(f"video not found for camera={camera}")
            continue
        try:
            total += extract_video(video, start, end, margin, out_dir, camera, quality, resize)
        except Exception as error:  # noqa: BLE001 - one bad video shouldn't kill the batch
            errors.append(f"{camera}: {error}")
    return key, total, errors


def parse_resize(value: str) -> tuple[int, int] | None:
    if not value:
        return None
    w, h = value.lower().split("x")
    return int(w), int(h)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--qc-json", type=Path, default=DEFAULT_QC_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--margin", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--resize", type=str, default="640x360", help="WxH, 빈 문자열이면 원본 해상도 유지")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import json

    payload = json.loads(args.qc_json.read_text())
    episodes = [e for e in payload["episodes"] if not e["broken"]]
    resize = parse_resize(args.resize)

    print(f"[START] {len(episodes)}개 녹화, margin={args.margin}, workers={args.workers}")
    done, failed = 0, []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(extract_episode, ep, args.out_dir, args.margin, args.jpeg_quality, resize, args.force): ep["source_dataset"]
            for ep in episodes
        }
        for future in as_completed(futures):
            key, saved, errors = future.result()
            done += 1
            if errors:
                failed.append((key, errors))
                print(f"[{done}/{len(episodes)}] {key}: {saved}장, 오류={errors}")
            elif done % 20 == 0 or done == len(episodes):
                print(f"[{done}/{len(episodes)}] ...")

    print(f"[DONE] {done}개 처리, 오류 {len(failed)}건")
    for key, errors in failed:
        print(f"  - {key}: {errors}")


if __name__ == "__main__":
    main()
