#!/usr/bin/env python3
"""LeRobotDataset 전 프레임을 여러 프로세스로 나눠 디코딩해보고 실패를 리포트한다.

빌드 스크립트(prepare_erase_shape_dataset.py)의 validate_output()은 프레임 수·task·
state/action 정합만 확인하고 실제 비디오 디코딩 가능 여부는 확인하지 않는다. GOP이 큰
데이터셋에서는 디코딩이 seek 뒤 되감기를 필요로 해서, 드물게 프레임 하나가 깨져 있어도
빌드 단계에서는 안 걸리고 학습 도중에만 터진다(FrameTimestampError 등). 이 스크립트로
학습 시작 전에 미리 전 프레임을 확인한다.

사용법:
    python scripts/tools/decode_check_parallel.py <repo_id> <root> [--workers 14]
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


def check_chunk(args: tuple[str, str, int, int, int]) -> list[tuple[int, str, str]]:
    repo_id, root, worker_id, start, end = args
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, root=root, video_backend="pyav")
    failures = []
    for i in range(start, end):
        try:
            ds[i]
        except Exception as exc:  # noqa: BLE001
            failures.append((i, type(exc).__name__, str(exc)[:200]))
        done = i - start + 1
        if done % 500 == 0:
            print(f"[worker {worker_id}] {done}/{end - start} 완료", flush=True)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", help="예: local/erase_the_shape_150")
    parser.add_argument("root", help="데이터셋 루트 경로")
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(args.repo_id, root=args.root, video_backend="pyav")
    total = len(ds)
    print(f"총 프레임 수: {total}", flush=True)
    del ds

    chunk_size = (total + args.workers - 1) // args.workers
    chunks = [
        (args.repo_id, args.root, w, i, min(i + chunk_size, total))
        for w, i in enumerate(range(0, total, chunk_size))
    ]
    print(f"{len(chunks)}개 워커로 분할, 워커당 약 {chunk_size}프레임", flush=True)

    start_time = time.time()
    all_failures: list[tuple[int, str, str]] = []
    done_chunks = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(check_chunk, c): c for c in chunks}
        for future in as_completed(futures):
            worker_id = futures[future][2]
            failures = future.result()
            all_failures.extend(failures)
            done_chunks += 1
            elapsed = time.time() - start_time
            print(
                f"[청크 {done_chunks}/{len(chunks)} 완료] worker={worker_id}, "
                f"실패 {len(failures)}건, 경과 {elapsed:.0f}s",
                flush=True,
            )

    print(f"\nTOTAL FAIL {len(all_failures)}", flush=True)
    for f in all_failures[:20]:
        print(f, flush=True)
    print(f"총 소요시간: {time.time() - start_time:.0f}s", flush=True)
    return 1 if all_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
