from __future__ import annotations

"""0802~0805 데이터의 목표 도형 위치를 먼저 측정하는 스크립트.

주의:
- 현재 top 카메라 영상은 사람이 보는 화이트보드 기준에서 90도 돌아간 상태이다.
- 따라서 목표 도형을 이미지 x축 left/right로 나누면 잘못된 결과가 나온다.
- 현재 기준은 board ROI의 y 중앙선이다.
  target center_y가 ROI y 중앙선보다 작으면 top, 크거나 같으면 bottom으로 기록한다.
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.roi import board_mask, load_roi


TARGETS = ("circle", "rectangle", "triangle")


@dataclass(frozen=True)
class TargetDetection:
    x: int
    y: int
    w: int
    h: int
    center_x: float
    center_y: float
    area: float
    fill_ratio: float
    vertices: int
    score: float


def infer_target_from_episode(name: str) -> str | None:
    lower = name.lower()
    for target in TARGETS:
        if f"erase_the_{target}" in lower:
            return target
    return None


def collect_top_videos(records_root: Path, dates: list[str]) -> list[tuple[str, str, Path]]:
    # 0802~0805는 0727과 달리 날짜 폴더 바로 아래에 episode 폴더가 있다.
    rows: list[tuple[str, str, Path]] = []
    for date in dates:
        date_root = records_root / date
        if not date_root.is_dir():
            raise FileNotFoundError(f"Date folder not found: {date_root}")
        for episode_dir in sorted(date_root.glob("erase_the_*")):
            if not episode_dir.is_dir():
                continue
            target = infer_target_from_episode(episode_dir.name)
            if target is None:
                continue
            video = episode_dir / "videos" / "observation.images.top" / "chunk-000" / "file-000.mp4"
            if video.exists():
                rows.append((date, target, video))
    return rows


def read_first_frame(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read first frame: {video_path}")
    return frame


def target_score(contour: np.ndarray, target: str, x: int, y: int, w: int, h: int, fill_ratio: float) -> tuple[float, int]:
    # 완벽한 도형 인식기가 아니라, 첫 프레임에서 목표 도형 후보를 고르기 위한 휴리스틱 점수이다.
    perimeter = cv2.arcLength(contour, True)
    approx_eps = 0.06 if target == "triangle" else 0.04
    approx = cv2.approxPolyDP(contour, approx_eps * perimeter, True) if perimeter > 0 else contour
    vertices = len(approx)
    area = cv2.contourArea(contour)
    circularity = 4.0 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0
    aspect = w / float(h)

    thin_outline = max(0.0, 0.22 - fill_ratio) * 4.0
    size_bonus = min(w, h) / 260.0

    if target == "circle":
        shape_bonus = max(0.0, 1.0 - abs(aspect - 1.0)) * 2.0 + circularity * 2.0
    elif target == "rectangle":
        shape_bonus = max(0.0, 1.0 - abs(aspect - 1.0)) * 2.0 + (1.5 if 4 <= vertices <= 8 else 0.0)
    else:
        shape_bonus = (2.5 if 3 <= vertices <= 5 else 0.0) + max(0.0, 1.0 - abs(aspect - 0.75))
    return shape_bonus + thin_outline + size_bonus, vertices


def detect_target(frame_bgr: np.ndarray, roi_points: np.ndarray, target: str, args: argparse.Namespace) -> TargetDetection:
    height, width = frame_bgr.shape[:2]
    bmask = board_mask((height, width), roi_points)
    erode_px = args.board_erode
    if erode_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1))
        inner_board = cv2.erode(bmask, kernel)
    else:
        inner_board = bmask

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # 130보다 높이면 보드 그림자와 배경 음영이 붙어 contour가 커지는 경우가 있었다.
    dark = np.where((gray <= args.dark_threshold) & (inner_board > 0), 255, 0).astype(np.uint8)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    contours, _hierarchy = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[TargetDetection] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < args.min_size or h < args.min_size or w > args.max_size or h > args.max_size:
            continue
        aspect = w / float(h)
        if not (args.min_aspect <= aspect <= args.max_aspect):
            continue
        roi = dark[y : y + h, x : x + w]
        fill_ratio = cv2.countNonZero(roi) / float(w * h)
        if not (args.min_fill <= fill_ratio <= args.max_fill):
            continue
        area = cv2.contourArea(contour)
        score, vertices = target_score(contour, target, x, y, w, h, fill_ratio)
        candidates.append(TargetDetection(x, y, w, h, x + w / 2.0, y + h / 2.0, area, fill_ratio, vertices, score))

    if not candidates:
        raise RuntimeError("No target-like component found")
    return max(candidates, key=lambda item: item.score)


def draw_debug(frame: np.ndarray, roi_points: np.ndarray, detection: TargetDetection, side: str, out_path: Path) -> None:
    preview = frame.copy()
    cv2.polylines(preview, [roi_points.astype(np.int32)], True, (0, 255, 255), 2)
    x, y, w, h = detection.x, detection.y, detection.w, detection.h
    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.circle(preview, (int(detection.center_x), int(detection.center_y)), 5, (255, 0, 255), -1)
    cv2.putText(preview, side, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), preview)


def summarize(rows: list[dict[str, str | int | float]]) -> dict[str, object]:
    summary: dict[str, object] = {"total": len(rows), "by_target": {}, "by_side": {}, "by_date": {}}
    for row in rows:
        for key in ("target", "side", "date"):
            bucket = summary[f"by_{key}"]  # type: ignore[index]
            value = str(row[key])
            bucket[value] = bucket.get(value, 0) + 1
    xs = [float(row["target_center_x"]) for row in rows]
    if xs:
        summary["center_x"] = {
            "min": min(xs),
            "mean": sum(xs) / len(xs),
            "max": max(xs),
        }
    ys = [float(row["target_center_y"]) for row in rows]
    if ys:
        summary["center_y"] = {
            "min": min(ys),
            "mean": sum(ys) / len(ys),
            "max": max(ys),
        }
    return summary


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Measure target shape positions for 0802/0804/0805 before augmentation.")
    parser.add_argument("--records-root", type=Path, default=project_root.parent)
    parser.add_argument("--dates", nargs="+", default=["0802", "0804", "0805"])
    parser.add_argument("--roi-json", type=Path, default=project_root / "board_roi.json")
    parser.add_argument("--output-dir", type=Path, default=project_root / "outputs" / "target_position_analysis_0802_0805_board_y")
    parser.add_argument("--board-erode", type=int, default=30)
    parser.add_argument("--dark-threshold", type=int, default=130)
    parser.add_argument("--min-size", type=int, default=35)
    parser.add_argument("--max-size", type=int, default=280)
    parser.add_argument("--min-aspect", type=float, default=0.30)
    parser.add_argument("--max-aspect", type=float, default=2.60)
    parser.add_argument("--min-fill", type=float, default=0.006)
    parser.add_argument("--max-fill", type=float, default=0.35)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    roi_points = load_roi(args.roi_json)
    board_min_x = float(np.min(roi_points[:, 0]))
    board_max_x = float(np.max(roi_points[:, 0]))
    board_mid_x = (board_min_x + board_max_x) / 2.0
    board_min_y = float(np.min(roi_points[:, 1]))
    board_max_y = float(np.max(roi_points[:, 1]))
    board_mid_y = (board_min_y + board_max_y) / 2.0

    videos = collect_top_videos(args.records_root, args.dates)
    rows: list[dict[str, str | int | float]] = []
    failures: list[dict[str, str]] = []

    for date, target, video in videos:
        episode = video.parents[3].name
        try:
            frame = read_first_frame(video)
            detection = detect_target(frame, roi_points, target, args)
            # 보드가 90도 돌아간 영상이므로 x축 left/right가 아니라 y축 top/bottom으로 나눈다.
            side = "top" if detection.center_y < board_mid_y else "bottom"
            row: dict[str, str | int | float] = {
                "date": date,
                "target": target,
                "episode": episode,
                "video": str(video),
                "target_center_x": round(detection.center_x, 3),
                "target_center_y": round(detection.center_y, 3),
                "target_bbox_x": detection.x,
                "target_bbox_y": detection.y,
                "target_bbox_w": detection.w,
                "target_bbox_h": detection.h,
                "side": side,
                "board_min_x": round(board_min_x, 3),
                "board_mid_x": round(board_mid_x, 3),
                "board_max_x": round(board_max_x, 3),
                "board_min_y": round(board_min_y, 3),
                "board_mid_y": round(board_mid_y, 3),
                "board_max_y": round(board_max_y, 3),
                "score": round(detection.score, 6),
                "fill_ratio": round(detection.fill_ratio, 6),
                "vertices": detection.vertices,
            }
            rows.append(row)
            debug_path = args.output_dir / "debug" / date / target / f"{episode}.png"
            draw_debug(frame, roi_points, detection, side, debug_path)
        except Exception as exc:
            failures.append({"date": date, "target": target, "episode": episode, "video": str(video), "error": str(exc)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "target_positions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "date",
            "target",
            "episode",
            "video",
            "target_center_x",
            "target_center_y",
            "target_bbox_x",
            "target_bbox_y",
            "target_bbox_w",
            "target_bbox_h",
            "side",
            "board_min_x",
            "board_mid_x",
            "board_max_x",
            "board_min_y",
            "board_mid_y",
            "board_max_y",
            "score",
            "fill_ratio",
            "vertices",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    fail_path = args.output_dir / "target_position_failures.json"
    with fail_path.open("w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2, ensure_ascii=False)

    summary = summarize(rows)
    summary["failures"] = len(failures)
    summary["failure_path"] = str(fail_path)
    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Videos found: {len(videos)}")
    print(f"Detected: {len(rows)}")
    print(f"Failures: {len(failures)}")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
