#!/usr/bin/env python3
"""매니페스트의 도형 위치를 보드 사진 위에 **실제 도형 모양**으로 겹쳐 그린다.

기존 `shape_position_analysis.py`는 `--sessions`로 날짜 폴더 전체를 훑어서 학습셋
매니페스트(315개 / 132개)와 대상이 정확히 일치하지 않는다. 이 도구는 매니페스트에
적힌 에피소드만 측정하므로 "그 학습셋이 실제로 덮은 영역"을 그대로 보여준다.

점(원)이 아니라 도형 종류대로 그리는 이유: 색만으로 구분하면 범례를 계속 대조해야
하지만, 모양이 다르면 어느 자리에 어떤 도형이 있었는지 바로 읽힌다.

기본은 **고정 크기 글리프**다. 검출된 bbox 크기까지 반영하면(--true-size) 315개가
겹쳐 선이 뒤엉켜 위치 분포가 안 보인다. 여기서 보려는 것은 "보드의 어디를 덮었나"
이므로 크기는 버리고 중심만 쓴다. 도형별 실제 크기가 필요하면 같이 저장되는
positions_*.csv의 w/h를 볼 것(원 129x126, 삼각형 110x132, 사각형 121x120 px).

검출은 `shape_position_analysis.detect_shape()`를 그대로 재사용한다 — 그쪽 분석과
기준이 어긋나지 않게 하려는 것이고, 카메라 정렬(나무 테이블 경계 x 기준 평행이동)도
같은 함수를 쓴다.

측정 결과는 CSV로 캐시한다. 첫 프레임 디코딩이 병목이라(에피소드당 ~0.3초) 그림만
다시 그릴 때는 `--from-csv`로 건너뛴다.

사용:
    python scripts/tasks/erase_shape/analysis/shape_position_render.py \\
        --manifest configs/erase_shape_315_manifest.json --label 315
    python scripts/tasks/erase_shape/analysis/shape_position_render.py \\
        --manifest configs/erase_shape_pickup_prompt_manifest.json --label 132
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from shape_position_analysis import (  # noqa: E402
    REFERENCE_EDGE_X, align, detect_shape, measure_edge_x, parse_shape, read_first_frame,
)

SHAPE_ORDER = ("circle", "triangle", "rectangle")
# shape_position_analysis.py와 같은 팔레트(BGR로 뒤집어 씀).
# dataviz all-pairs 검증을 통과하는 3색 조합이다.
SHAPE_RGB = {"circle": (42, 120, 214), "triangle": (235, 104, 52), "rectangle": (27, 175, 122)}

# cv2.putText는 한글을 못 그린다(전부 '?'로 나온다) — 배너만 PIL로 그린다.
FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
)

BANNER_H = 78
INK = (24, 24, 24)
MUTED = (110, 110, 110)
BANNER_BG = (246, 246, 244)


def resolve_dataset(source: str) -> pathlib.Path:
    """매니페스트 경로를 실제 폴더로 푼다 (records/local/... 은 지금 records/0813/...)."""
    direct = REPO_ROOT / source
    if direct.is_dir():
        return direct
    name = pathlib.PurePosixPath(source).name
    hits = sorted(set(REPO_ROOT.joinpath("records").glob(f"*/{name}"))
                  | set(REPO_ROOT.joinpath("records").glob(f"*/*/{name}")))
    if len(hits) != 1:
        raise FileNotFoundError(f"{source}: {len(hits)} candidates")
    return hits[0]


def measure(manifest_path: pathlib.Path, csv_path: pathlib.Path,
            background_path: pathlib.Path, background_frames: int) -> list[dict]:
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))["episodes"]
    rows: list[dict] = []
    # 깨끗한 빈 보드는 첫 프레임들의 **픽셀별 중앙값**으로 만든다. 도형이 에피소드마다
    # 다른 자리에 있으므로 중앙값을 취하면 도형만 지워지고 보드·블록·로봇은 남는다
    # (ink_by_zone.py가 쓰는 것과 같은 방법). 첫 프레임을 쓰는 이유는 그 시점엔 팔이
    # 홈 자세라 보드를 안 가리기 때문이다.
    # 전 프레임을 다 쌓으면 315장 x 720x1280x3 = 약 870MB라, 고르게 표집해 상한을 둔다.
    stride = max(1, len(entries) // max(1, background_frames))
    samples: list[np.ndarray] = []
    for index, entry in enumerate(entries, 1):
        source = entry["source_dataset"]
        recording = resolve_dataset(source)
        shape = parse_shape(recording.name)
        raw = read_first_frame(recording)
        if raw is None:
            print(f"  [{index:3d}/{len(entries)}] {recording.name}: 프레임 없음 — 건너뜀")
            continue
        shift = float(REFERENCE_EDGE_X - measure_edge_x(raw))
        frame = align(raw, shift)
        if (index - 1) % stride == 0 and len(samples) < background_frames:
            samples.append(frame)
        found, method = detect_shape(frame)
        if found is None:
            print(f"  [{index:3d}/{len(entries)}] {recording.name}: 검출 실패")
            continue
        rows.append({"name": recording.name, "shape": shape, "method": method,
                     "camera_shift": f"{shift:.2f}", **found})
        if index % 50 == 0 or index == len(entries):
            print(f"  [{index}/{len(entries)}] 검출 {len(rows)}개", flush=True)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if samples:
        clean = np.median(np.stack(samples), axis=0).astype(np.uint8)
        background_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(background_path), cv2.cvtColor(clean, cv2.COLOR_RGB2BGR))
        print(f"[BG] 빈 보드 합성: {len(samples)}장의 픽셀별 중앙값")
    return rows


def shape_points(shape: str, cx: float, cy: float, half: float) -> np.ndarray:
    """중심 (cx, cy)에 고정 크기로 그릴 도형 외곽 점들.

    검출된 bbox 크기는 쓰지 않는다 — 315개를 실제 크기로 겹치면 선이 뒤엉켜
    위치 분포가 안 보인다. 여기서 볼 것은 "보드의 어디를 덮었나"이므로 중심만 쓴다.
    실제 크기는 positions_*.csv의 w/h에 남아 있다.
    """
    if shape == "circle":
        angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        return np.stack([cx + half * np.cos(angles),
                         cy + half * np.sin(angles)], axis=1).astype(np.int32)
    if shape == "triangle":
        # 녹화 영상은 사람이 보는 보드 기준에서 90도 돌아가 있어, 데이터의 삼각형은
        # 이미지 좌표에서 꼭짓점이 왼쪽을 향한다(asset PNG로 직접 확인).
        return np.array([[cx - half, cy], [cx + half * 0.85, cy - half],
                         [cx + half * 0.85, cy + half]], dtype=np.int32)
    return np.array([[cx - half, cy - half], [cx + half, cy - half],
                     [cx + half, cy + half], [cx - half, cy + half]], dtype=np.int32)


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if pathlib.Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def render_banner(width: int, label: str, total: int, counts: dict[str, int]) -> np.ndarray:
    """제목 + 도형별 색 범례. RGB로 그린 뒤 BGR로 돌려준다."""
    image = Image.new("RGB", (width, BANNER_H), BANNER_BG[::-1])
    draw = ImageDraw.Draw(image)
    draw.text((18, 12), f"{label}개 데이터셋 — 도형 위치 분포 (검출 {total}개)",
              font=load_font(26), fill=INK[::-1])
    legend_font = load_font(17)
    x = 18
    for shape in SHAPE_ORDER:
        draw.rectangle([x, 52, x + 24, 66], fill=SHAPE_RGB[shape])
        text = f"{shape}: {counts[shape]}"
        draw.text((x + 32, 50), text, font=legend_font, fill=MUTED[::-1])
        x += 32 + int(draw.textlength(text, font=legend_font)) + 30
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def render(rows: list[dict], background: np.ndarray, label: str,
           out_path: pathlib.Path, alpha: float, marker: int) -> None:
    canvas = background.copy()
    for row in rows:
        shape = row["shape"]
        points = shape_points(shape, float(row["cx"]), float(row["cy"]), float(marker))
        # 도형마다 따로 합성한다 — 겹칠수록 진해져 밀집도가 그대로 보인다. 레이어를
        # 하나로 모아 한 번에 합성하면 겹친 곳도 같은 농도가 되어 그 정보가 사라진다.
        patch = canvas.copy()
        cv2.fillPoly(patch, [points], SHAPE_RGB[shape][::-1], cv2.LINE_AA)
        cv2.polylines(patch, [points], True, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.addWeighted(patch, alpha, canvas, 1.0 - alpha, 0, canvas)

    counts = {shape: sum(r["shape"] == shape for r in rows) for shape in SHAPE_ORDER}
    banner = render_banner(canvas.shape[1], label, len(rows), counts)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack([banner, canvas]))
    print(f"[SAVE] {out_path}  (도형 {len(rows)}개)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--label", required=True, help="산출물 이름에 쓸 데이터셋 규모 (예: 315)")
    parser.add_argument("--out-dir", type=pathlib.Path,
                        default=REPO_ROOT / "outputs" / "analysis" / "erase_shape" / "shape_position_render")
    parser.add_argument("--from-csv", action="store_true",
                        help="측정을 건너뛰고 캐시된 CSV로 그림만 다시 그린다")
    parser.add_argument("--background-frames", type=int, default=120,
                        help="빈 보드 합성에 쓸 첫 프레임 수(중앙값). 많을수록 도형 잔상이 준다")
    parser.add_argument("--alpha", type=float, default=0.42,
                        help="도형 1개의 불투명도. 낮을수록 겹침 농도가 잘 보인다")
    parser.add_argument("--marker", type=int, default=17, help="고정 글리프의 반지름(px)")
    args = parser.parse_args()

    csv_path = args.out_dir / f"positions_{args.label}.csv"
    background_path = args.out_dir / f"background_{args.label}.png"
    if args.from_csv:
        if not csv_path.is_file():
            raise SystemExit(f"{csv_path} 없음 — --from-csv 없이 먼저 측정할 것")
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    else:
        rows = measure(args.manifest, csv_path, background_path, args.background_frames)
    if not rows:
        raise SystemExit("검출된 도형이 없다")

    background = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
    if background is None:
        raise SystemExit(f"{background_path} 없음")
    render(rows, background, args.label,
           args.out_dir / f"shape_positions_{args.label}.png",
           args.alpha, args.marker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
