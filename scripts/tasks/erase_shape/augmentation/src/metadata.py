from __future__ import annotations

"""각 episode의 증강 조건을 placement_metadata.json으로 저장하는 모듈.

검수나 재현이 필요할 때 asset, 위치, target side, placement side, mask 통계 등을 확인한다.
"""

import json
from pathlib import Path
from typing import Any


def save_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
