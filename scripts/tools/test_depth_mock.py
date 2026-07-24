"""observation depth(turbo 컬러맵) 로직 mock 테스트 (하드웨어 불필요).

실행: PYTHONPATH=. python scripts/tools/test_depth_mock.py
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lerobot_robot_piper.piper_follower import PiperFollower


class FakeCamera:
    def __init__(self, use_depth: bool, height: int = 4, width: int = 4):
        self.use_depth = use_depth
        self.height = height
        self.width = width
        self.is_connected = True
        self.frame_lock = threading.Lock()
        self.latest_depth_frame = (
            np.full((height, width), 500, dtype=np.uint16) if use_depth else None  # 0.5m
        )

    def async_read(self):
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)


class FakeBus:
    is_connected = True
    motors = {"joint1": None}

    def get_action(self):
        return {"joint1": 0.0}


def make_follower(use_depth_observation: bool, cameras: dict, depth_raw_dir: str = "") -> PiperFollower:
    follower = PiperFollower.__new__(PiperFollower)
    follower.id = "test_follower"
    follower.bus = FakeBus()
    follower.cameras = cameras
    follower._camera_executor = ThreadPoolExecutor(max_workers=2)
    follower.config = SimpleNamespace(
        use_effort=False,
        use_depth_observation=use_depth_observation,
        depth_scale=0.001,
        depth_min_m=0.20,
        depth_max_m=0.80,
        depth_raw_dir=depth_raw_dir,
    )
    return follower


def test_depth_off_by_default():
    f = make_follower(use_depth_observation=False, cameras={"top": FakeCamera(use_depth=True)})
    assert f._depth_cam_keys == []
    assert "top_depth" not in f.observation_features
    obs = f.get_observation()
    assert "top_depth" not in obs


def test_depth_on_only_for_depth_enabled_cams():
    cameras = {"top": FakeCamera(use_depth=True), "wrist": FakeCamera(use_depth=False)}
    f = make_follower(use_depth_observation=True, cameras=cameras)
    assert f._depth_cam_keys == ["top"]
    assert "top_depth" in f.observation_features
    assert "wrist_depth" not in f.observation_features

    obs = f.get_observation()
    assert "top" in obs and "wrist" in obs  # 기존 color 키 유지
    assert obs["top_depth"].shape == (4, 4, 3)
    assert obs["top_depth"].dtype.name == "uint8"
    assert "wrist_depth" not in obs


def test_depth_raw_dir_saves_npy_sidecar(tmp_path):
    # Evo-Depth IDEM 보조 supervision 대비: 컬러맵(8bit)과 별개로 원본 uint16을
    # <depth_raw_dir>/<cam>/*.npy로 저장하는지, 그리고 컬러맵 저장은 계속 되는지 확인.
    cameras = {"top": FakeCamera(use_depth=True)}
    f = make_follower(use_depth_observation=True, cameras=cameras, depth_raw_dir=str(tmp_path))
    obs = f.get_observation()
    assert "top_depth" in obs  # 컬러맵은 그대로 저장됨

    saved = list((tmp_path / "top").glob("*.npy"))
    assert len(saved) == 1
    raw = np.load(saved[0])
    assert raw.dtype == np.uint16
    assert (raw == 500).all()  # FakeCamera가 채운 원본 그대로(변환 전)

    f.get_observation()
    assert len(list((tmp_path / "top").glob("*.npy"))) == 2  # 호출마다 누적


def test_depth_raw_dir_empty_skips_save(tmp_path):
    cameras = {"top": FakeCamera(use_depth=True)}
    f = make_follower(use_depth_observation=True, cameras=cameras, depth_raw_dir="")
    f.get_observation()
    assert not (tmp_path / "top").exists()


if __name__ == "__main__":
    import tempfile

    test_depth_off_by_default()
    test_depth_on_only_for_depth_enabled_cams()
    with tempfile.TemporaryDirectory() as d:
        test_depth_raw_dir_saves_npy_sidecar(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_depth_raw_dir_empty_skips_save(Path(d))
    print("OK: depth mock 테스트 통과")
