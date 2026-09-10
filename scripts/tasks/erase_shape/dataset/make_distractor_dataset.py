#!/usr/bin/env python3
"""top 영상만 '방해 도형이 추가된 증강본'으로 갈아끼운 학습 데이터셋을 만든다.

증강 자료는 원본 top 영상 위에 **타겟이 아닌 다른 도형 하나를 합성해 넣은 것**이다
(예: circle을 지우는 에피소드의 보드에 rectangle을 하나 더 그려 넣음). 관측 중
top 카메라 화면만 달라지고 관절 상태·행동·wrist 영상은 원본 그대로다.

목적: 지금 학습셋은 보드에 도형이 하나뿐이라 프롬프트의 도형 단어가 시각 입력과
100% 중복이다 — 언어를 접지할 압력이 없다. 방해 도형을 넣으면 "어느 걸 지울지"가
프롬프트로만 결정되므로 선택성을 학습할 수 있다.

**무엇을 바꾸고 무엇을 그대로 두는가**
  바꾸는 것 : videos/observation.images.top/*.mp4 (증강 영상에서 다시 인코딩)
  그대로    : data/*.parquet(상태·행동·task_index), wrist 영상, meta 전부
              -> 하드링크라 용량을 새로 먹지 않는다.

**정합성을 어떻게 지키는가**
  기존 데이터셋은 에피소드를 파일 몇 개에 이어붙여 저장하고, 각 에피소드의 구간을
  meta/episodes의 from/to_timestamp로 가리킨다. 그래서 증강 영상도 **같은 순서,
  같은 프레임 수, 같은 파일 배치**로 다시 써야 타임스탬프가 안 깨진다. 프레임 수가
  하나라도 어긋나면 관측과 행동이 밀리므로, 쓰기 전에 전수 검증하고 어긋나면 중단한다.

사용:
    python scripts/tasks/erase_shape/dataset/make_distractor_dataset.py \\
        --src records/outputs/.../pick_up_the_eraser_315_shape_prompt \\
        --aug-root <증강 압축을 푼 폴더> \\
        --manifest configs/erase_shape_315_manifest.json \\
        --dst records/outputs/.../pick_up_the_eraser_315_distractor \\
        [--vcodec h264] [--dry-run]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_index(aug_root: Path, manifest: Path) -> pd.DataFrame:
    """증강 영상 <-> 매니페스트 대응표. 어긋나면 예외."""
    man = [e for e in json.loads(manifest.read_text())["episodes"] if e.get("enabled", True)]
    by_name = {Path(e["source_dataset"]).name: e for e in man}
    rows, bad = [], []
    for mp4 in sorted(aug_root.glob("*/*/*/*/augmented.mp4")):
        name = mp4.parent.name
        target = mp4.parent.parent.parent.name.replace("erase_", "")
        distractor = mp4.parent.parent.name
        entry = by_name.get(name)
        if entry is None:
            bad.append(f"{name}: 매니페스트에 없음")
            continue
        shape = re.search(r"erase_the_(\w+?)_\d{4}-", entry["source_dataset"]).group(1)
        with av.open(str(mp4)) as c:
            s = c.streams.video[0]
            n, w, h = s.frames, s.codec_context.width, s.codec_context.height
        if target != shape:
            bad.append(f"{name}: 타겟 불일치 {target} vs {shape}")
        if distractor == shape:
            bad.append(f"{name}: 방해도형이 타겟과 같음")
        if n < entry["end_frame"]:
            bad.append(f"{name}: 프레임 부족 {n} < {entry['end_frame']}")
        if (w, h) != (1280, 720):
            bad.append(f"{name}: 해상도 {w}x{h}")
        rows.append(dict(episode=name, target=shape, distractor=distractor, path=str(mp4),
                         start=entry["start_frame"], end=entry["end_frame"]))
    if bad:
        raise SystemExit("증강 자료 검증 실패:\n  " + "\n  ".join(bad[:20]))
    if len(rows) != len(man):
        raise SystemExit(f"개수 불일치: 증강 {len(rows)} vs 매니페스트 {len(man)}")
    return pd.DataFrame(rows)


def episode_meta(src: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(src / "meta/episodes/**/*.parquet"), recursive=True))
    return pd.concat([pd.read_parquet(f) for f in files]).sort_values("episode_index")


def aug_frames(path: str, start: int, end: int, crop: tuple[int, int, int], size: int):
    """증강 영상에서 [start, end) 구간을 잘라 학습 데이터셋과 같은 형태로 내준다.

    증강본은 원본 카메라 해상도(1280x720) 그대로지만, 학습 데이터셋의 top은
    prepare_erase_shape_dataset.py가 정사각 크롭 후 리사이즈한 512x512다. 이 단계를
    빼면 프레임 크기가 안 맞아 데이터셋이 깨진다(features는 512x512라고 적혀 있는데
    실제 영상은 1280x720이 된다). 크롭·보간 방식을 그 스크립트와 똑같이 맞춘다.
    """
    cx, cy, csize = crop
    got = 0
    with av.open(path) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i < start:
                continue
            if i >= end:
                break
            img = f.to_ndarray(format="rgb24")
            h, w = img.shape[:2]
            if cx + csize > w or cy + csize > h:
                raise RuntimeError(f"{path}: 크롭({cx},{cy},{csize})이 {w}x{h}를 벗어난다")
            sub = img[cy:cy + csize, cx:cx + csize]
            got += 1
            yield cv2.resize(sub, (size, size),
                             interpolation=cv2.INTER_AREA if csize >= size else cv2.INTER_LINEAR)
    if got != end - start:
        raise RuntimeError(f"{path}: {end-start}프레임 필요한데 {got}개만 나왔다")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True, help="기반 데이터셋(도형별 프롬프트 권장)")
    p.add_argument("--aug-root", type=Path, required=True, help="증강 압축을 푼 최상위 폴더")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--vcodec", default=os.environ.get("VCODEC", "h264"))
    p.add_argument("--top-crop", default="280,0,720", metavar="X,Y,SIZE",
                   help="증강본(원본 해상도)에 적용할 정사각 크롭. 기존 학습 데이터셋을 "
                        "만들 때 쓴 값과 같아야 한다(315 기준 280,0,720)")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    crop = tuple(int(v) for v in a.top_crop.split(","))
    if len(crop) != 3:
        raise SystemExit("--top-crop 은 X,Y,SIZE 형식")

    src, dst = a.src.resolve(), a.dst.resolve()
    if dst.exists():
        raise SystemExit(f"이미 있다: {dst}")

    idx = build_index(a.aug_root.resolve(), a.manifest).set_index("episode")
    ep = episode_meta(src)
    info = json.loads((src / "meta/info.json").read_text())
    fps = int(info["fps"])
    key = "observation.images.top"
    # 출력 크기는 기존 데이터셋이 선언한 값을 그대로 따른다 — 여기서 임의로 정하면
    # features(512x512)와 실제 영상이 어긋나 데이터셋이 깨진다.
    fh, fw, _ = info["features"][key]["shape"]
    if fh != fw:
        raise SystemExit(f"정사각이 아닌 top feature: {fh}x{fw}")
    out_size = int(fh)

    # 컬럼명에 슬래시가 있어 itertuples로는 못 읽는다 — 직접 인덱싱한다
    ci = ep[f"videos/{key}/chunk_index"].to_numpy()
    fi = ep[f"videos/{key}/file_index"].to_numpy()
    lens = ep["length"].to_numpy()
    order = ep["episode_index"].to_numpy()

    # 에피소드 순서 -> 원본 녹화 이름 (매니페스트 순서와 같다)
    man = [e for e in json.loads(a.manifest.read_text())["episodes"] if e.get("enabled", True)]
    names = [Path(e["source_dataset"]).name for e in man]
    if len(names) != len(order):
        raise SystemExit("에피소드 수와 매니페스트 수가 다르다")
    if not (lens == np.array([idx.loc[n, "end"] - idx.loc[n, "start"] for n in names])).all():
        raise SystemExit("에피소드 길이와 매니페스트 구간이 다르다 — 중단")

    groups: dict[tuple[int, int], list[int]] = {}
    for i in range(len(order)):
        groups.setdefault((int(ci[i]), int(fi[i])), []).append(i)

    print(f"원본 : {src.relative_to(REPO_ROOT)}")
    print(f"증강 : {a.aug_root}")
    print(f"대상 : {dst.relative_to(REPO_ROOT)}")
    print(f"에피소드 {len(order)}개 / 총 {int(lens.sum())}프레임 / 영상파일 {len(groups)}개 / 코덱 {a.vcodec}")
    print(f"top 변환: 크롭{crop} -> {out_size}x{out_size} (기존 데이터셋 features 기준)")
    print("방해도형 조합:")
    print(idx.groupby(["target", "distractor"]).size().to_string())
    if a.dry_run:
        print("\n(dry-run — 아무것도 쓰지 않았다)")
        return 0

    # 1) top 영상을 제외한 모든 것 하드링크
    n_link = 0
    for f in src.rglob("*"):
        if f.is_dir() or f"videos/{key}/" in str(f).replace("\\", "/"):
            continue
        link_or_copy(f, dst / f.relative_to(src))
        n_link += 1
    print(f"\n원본 파일 {n_link}개 링크(영상 top 제외)")

    # 2) top 영상을 증강본으로 다시 인코딩 — 순서·프레임 수·파일 배치 그대로
    for (c_i, f_i), members in sorted(groups.items()):
        out = dst / f"videos/{key}/chunk-{c_i:03d}/file-{f_i:03d}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        cont = av.open(str(out), mode="w")
        st = cont.add_stream(a.vcodec, rate=fps)
        st.width, st.height, st.pix_fmt = out_size, out_size, "yuv420p"
        st.options = {"g": "2", **({"crf": "30"} if a.vcodec in ("h264", "hevc", "libsvtav1")
                                   else {"rc": "constqp", "qp": "30"})}
        written = 0
        for i in members:
            name = names[i]
            row = idx.loc[name]
            for frame in aug_frames(row["path"], int(row["start"]), int(row["end"]),
                                    crop, out_size):
                cont.mux(st.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
                written += 1
        cont.mux(st.encode())
        cont.close()
        want = int(sum(lens[i] for i in members))
        if written != want:
            raise SystemExit(f"{out}: {want}프레임 필요한데 {written}개 썼다")
        print(f"  {out.relative_to(dst)}  에피소드 {len(members)}개 · {written}프레임")

    print(f"\n완료: {dst}")
    print("주의: meta/stats.json의 이미지 통계는 원본 것이 그대로다 "
          "(SmolVLA는 VISUAL 정규화가 IDENTITY라 학습에 쓰이지 않는다).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
