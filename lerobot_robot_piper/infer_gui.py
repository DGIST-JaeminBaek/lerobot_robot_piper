"""`python -m lerobot_robot_piper.infer_gui` 진입점.

실제 구현은 scripts/tools/piper_infer_gui.py에 있다. 그 파일은 같은 디렉터리의
piper_offline_chunk_rollout / piper_human_approved_inference / piper_infer_preview를
sys.path 기반으로 import하므로, 여기서는 경로만 맞춰 주고 main()을 호출한다.
"""

from __future__ import annotations

import pathlib
import sys

_TOOLS_DIR = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "tools"


def main() -> int:
    if str(_TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(_TOOLS_DIR))
    from piper_infer_gui import main as gui_main

    return gui_main()


if __name__ == "__main__":
    raise SystemExit(main())
