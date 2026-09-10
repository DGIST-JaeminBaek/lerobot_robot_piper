from __future__ import annotations

"""0727 원본 영상에서 top 카메라 첫 프레임을 PNG로 추출하는 자산 준비용 스크립트.

현재 0802~0805 최종 증강 영상을 만드는 스크립트는 아니다.
투명 PNG asset을 만들기 전, 0727 episode의 첫 프레임을 모으기 위해 사용했다.
"""

import argparse
import csv
from pathlib import Path

import cv2


CLASS_DIRS = {
    "circle": "erase_the_circle",
    "triangle": "erase_the_triangle",
    "rectangle": "erase_the_rectangle",
}


def find_top_videos(records_root: Path, date_dir: str | None, class_dir: str) -> list[Path]:
    root = records_root / class_dir if date_dir is None else records_root / date_dir / class_dir
    if not root.is_dir():
        raise FileNotFoundError(f"Class folder does not exist: {root}")
    videos = sorted(root.glob("*/videos/observation.images.top/chunk-000/file-000.mp4"))
    return videos


def extract_first_frame(video_path: Path, output_path: Path) -> tuple[int, int, float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    ok, frame = cap.read()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read first frame: {video_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Could not write PNG: {output_path}")
    return width, height, fps, frame_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract t=0 top-camera frames from 0727 erase-the-shape episodes.")
    parser.add_argument("--records-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--date-dir", default="0727")
    parser.add_argument(
        "--direct-layout",
        action="store_true",
        help="Use records-root/erase_the_<shape>/... instead of records-root/date-dir/erase_the_<shape>/...",
    )
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "source_frames" / "0727")
    parser.add_argument("--limit", type=int, default=0, help="Optional per-class limit; 0 means all")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest_rows: list[dict[str, str | int | float]] = []
    total = 0
    date_dir = None if args.direct_layout else args.date_dir
    for shape_name, class_dir in CLASS_DIRS.items():
        videos = find_top_videos(args.records_root, date_dir, class_dir)
        if args.limit > 0:
            videos = videos[: args.limit]
        print(f"{class_dir}: {len(videos)} top videos")
        for idx, video in enumerate(videos, start=1):
            episode_name = video.parents[3].name
            output_path = args.output_root / shape_name / f"{idx:02d}_{episode_name}.png"
            if output_path.exists() and not args.overwrite:
                print(f"  skip existing {output_path.name}")
                continue
            width, height, fps, frame_count = extract_first_frame(video, output_path)
            manifest_rows.append(
                {
                    "shape": shape_name,
                    "class_dir": class_dir,
                    "episode": episode_name,
                    "source_video": str(video),
                    "output_png": str(output_path),
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "frames": frame_count,
                }
            )
            total += 1
            print(f"  wrote {output_path}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["shape", "class_dir", "episode", "source_video", "output_png", "width", "height", "fps", "frames"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Done. Extracted {total} PNG files.")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
