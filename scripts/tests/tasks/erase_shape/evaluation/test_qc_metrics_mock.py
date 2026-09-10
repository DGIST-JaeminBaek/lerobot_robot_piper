"""qc_core의 clip_saturation / warmup_frames 계약 테스트 (데이터셋/하드웨어 불필요).

두 지표 모두 action과 observation.state의 앞 6개 관절만 본다. USE_EFFORT=true면
observation.state가 pos(7)+effort(7)+vel(6)=20차원이 되므로, effort/vel 칸이 붙어도
결과가 변하지 않아야 한다 — 그 계약을 고정한다.

실행: python scripts/tests/tasks/erase_shape/evaluation/test_qc_metrics_mock.py
"""

import pathlib
import sys

import numpy as np

# qc_core는 Report를 dataclass로 선언하므로 sys.modules에 정상 등록되는 경로로
# import해야 한다 (importlib.util 수동 로드는 dataclass 처리에서 깨진다).
SCRIPTS_DIR = next(parent for parent in pathlib.Path(__file__).resolve().parents if (parent / "0__launch_gui.sh").is_file())
sys.path.insert(0, str(SCRIPTS_DIR / "tasks" / "erase_shape" / "qc"))

import qc_core as qc  # noqa: E402


def _pair(n=10):
    return np.zeros((n, 7)), np.zeros((n, 20))


def test_clip_counts_only_saturated_frames():
    action, state = _pair()
    action[:4, 0] = qc.CLIP_LIMIT          # 4/10이 상한에 붙음
    assert qc.clip_saturation(action, state) == 40.0


def test_clip_ignores_frames_below_the_near_band():
    action, state = _pair()
    action[:4, 0] = qc.CLIP_LIMIT * qc.CLIP_NEAR - 0.01
    assert qc.clip_saturation(action, state) == 0.0


def test_clip_ignores_effort_and_velocity_columns():
    action, state = _pair()
    state[:, 7:] = 999.0                   # effort/vel 칸에 큰 값이 들어와도
    assert qc.clip_saturation(action, state) == 0.0


def test_clip_handles_empty_episode():
    assert qc.clip_saturation(np.zeros((0, 7)), np.zeros((0, 20))) == 0.0


def test_warmup_counts_leading_copied_frames():
    action, state = _pair()
    action[3:, 0] = 1.0
    assert qc.warmup_frames(action, state) == 3


def test_warmup_returns_full_length_when_never_commanded():
    action, state = _pair()
    assert qc.warmup_frames(action, state) == len(action)


def test_warmup_is_zero_when_moving_from_the_first_frame():
    action, state = _pair()
    action[:, 0] = 1.0
    assert qc.warmup_frames(action, state) == 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("\nall passed")
