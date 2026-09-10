"""effort 검증 CLI가 공유하는 dataset schema·parquet 읽기 함수.

출력 정책과 합격 기준은 각 CLI에 남긴다. 이 모듈은 데이터를 읽고 같은 기준으로
smooth-start 오염만 판정한다.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np

SMOOTH_START_FRAMES = 100


@dataclass(frozen=True)
class EffortSchema:
    names: list[str]
    position_indices: list[int]
    effort_indices: list[int]
    velocity_indices: list[int]


def load_info(root: pathlib.Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))


def effort_schema(info: dict) -> EffortSchema | None:
    feature = info.get("features", {}).get("observation.state")
    if feature is None:
        return None
    names = list(feature.get("names", []))
    return EffortSchema(
        names=names,
        position_indices=[index for index, name in enumerate(names) if name.endswith(".pos")],
        effort_indices=[index for index, name in enumerate(names) if name.endswith(".effort")],
        velocity_indices=[index for index, name in enumerate(names) if name.endswith(".vel")],
    )


def parquet_files(root: pathlib.Path) -> list[pathlib.Path]:
    return sorted((root / "data").rglob("*.parquet"))


def read_state_table(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """parquet 하나에서 observation.state, episode/frame index를 읽는다."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    state = np.stack(table.column("observation.state").to_pylist())
    episode_index = table.column("episode_index").to_numpy(zero_copy_only=False)
    frame_index = table.column("frame_index").to_numpy(zero_copy_only=False)
    return state, episode_index, frame_index


def load_episode_effort(root: pathlib.Path) -> tuple[np.ndarray, list[str]] | None:
    """dataset 전체 parquet의 effort를 frame 순서로 합쳐 반환한다."""
    schema = effort_schema(load_info(root))
    if schema is None or not schema.effort_indices:
        return None
    chunks: list[np.ndarray] = []
    for path in parquet_files(root):
        state, _, frame_index = read_state_table(path)
        chunks.append(state[np.argsort(frame_index)][:, schema.effort_indices])
    if not chunks:
        return None
    return np.concatenate(chunks), [schema.names[index] for index in schema.effort_indices]


def is_smooth_start_contaminated(effort: np.ndarray, *, frames: int = SMOOTH_START_FRAMES) -> bool:
    """초반 effort가 선형 램프인지 판정한다."""
    if len(effort) < frames + 20:
        return False
    return bool(np.abs(np.diff(effort[:frames], n=2, axis=0)).max() < 1e-4)
