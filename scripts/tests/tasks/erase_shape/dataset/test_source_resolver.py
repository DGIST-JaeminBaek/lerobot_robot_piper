"""Frame-range manifest가 항상 같은 원본 dataset을 찾는지 검증한다."""

from __future__ import annotations

import pathlib
import sys

import pytest


SCRIPTS_DIR = next(
    parent
    for parent in pathlib.Path(__file__).resolve().parents
    if (parent / "0__launch_gui.sh").is_file()
)
DATASET_DIR = SCRIPTS_DIR / "tasks" / "erase_shape" / "dataset"
sys.path.insert(0, str(DATASET_DIR))

from source_resolver import resolve_source  # noqa: E402


def dataset(root: pathlib.Path, relative: str) -> pathlib.Path:
    path = root / relative
    (path / "meta").mkdir(parents=True)
    (path / "meta" / "info.json").write_text("{}")
    return path


def test_resolver_uses_exact_manifest_path(tmp_path):
    expected = dataset(tmp_path, "records/local/group/episode")
    assert resolve_source("records/local/group/episode", repo_root=tmp_path) == expected.resolve()


def test_resolver_finds_a_relocated_dataset(tmp_path):
    expected = dataset(tmp_path, "records/0727/group/episode")
    assert resolve_source("records/local/group/episode", repo_root=tmp_path) == expected.resolve()


def test_resolver_rejects_ambiguous_dataset_name(tmp_path):
    dataset(tmp_path, "records/0801/group/episode")
    dataset(tmp_path, "records/0802/group/episode")
    with pytest.raises(RuntimeError, match="ambiguous"):
        resolve_source("records/local/group/episode", repo_root=tmp_path)
