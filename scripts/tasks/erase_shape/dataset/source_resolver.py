"""Frame-range manifest의 source dataset을 안전하게 실제 경로로 해석한다."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]


def resolve_source(source: str, *, repo_root: Path = REPO_ROOT) -> Path:
    """Resolve a manifest source path without silently picking a duplicate.

    기존 녹화가 ``records/local``에서 날짜별 폴더로 이동한 경우도 찾는다.
    이름과 그룹이 같은 후보가 둘 이상이면 호출자가 원본을 명시하도록 실패한다.
    """
    root = repo_root.resolve()
    path = Path(source)
    direct = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if (direct / "meta/info.json").is_file():
        return direct

    relocated = (root / "records/0727" / path.parent.name / path.name).resolve()
    if (relocated / "meta/info.json").is_file():
        return relocated

    group_name = path.parent.name
    candidates = sorted(
        candidate.resolve()
        for candidate in (root / "records").rglob(path.name)
        if candidate.is_dir()
        and (not group_name or candidate.parent.name == group_name)
        and (candidate / "meta/info.json").is_file()
    )
    if not candidates:
        raise FileNotFoundError(f"Source dataset does not exist: {source}")
    if len(candidates) > 1:
        formatted = "\n".join(f"- {candidate}" for candidate in candidates)
        raise RuntimeError(f"Source dataset is ambiguous: {source}\n{formatted}")
    return candidates[0]
