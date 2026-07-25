"""effort/vel/depth가 LeRobotDataset 스키마에 실제로 어떻게 들어가는지 검증 (하드웨어 불필요).

test_effort_mock.py / test_depth_mock.py는 PiperFollower가 내보내는 observation dict까지만
확인한다. 이 테스트는 그 뒤 단계 — lerobot_record.record()가 쓰는 실제 함수
(aggregate_pipeline_dataset_features / combine_feature_dicts / build_dataset_frame)를
그대로 태워서, 녹화된 파케이가 갖게 될 컬럼 구조를 확정한다.

핵심 결론(이 테스트가 고정하는 계약):
  - effort/vel은 별도 컬럼이 아니라 observation.state 벡터 안에 pos 뒤로 이어 붙는다.
  - 어느 인덱스가 무엇인지는 features["observation.state"]["names"]에 남으므로
    meta/info.json만 보면 나중에 슬라이싱해서 뽑아 쓸 수 있다.
  - depth는 observation.images.<cam>_depth 라는 별도 video 스트림이 된다.
lerobot 버전을 올렸을 때 이 계약이 깨지면 여기서 먼저 실패한다.

실행: PYTHONPATH=. python scripts/tools/test_dataset_features_mock.py
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.processor import make_default_processors

from lerobot_robot_piper.piper_follower import PiperFollower

MOTORS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]


class FakeCamera:
    def __init__(self, use_depth: bool, height: int = 4, width: int = 4):
        self.use_depth = use_depth
        self.height = height
        self.width = width
        self.is_connected = True
        self.frame_lock = threading.Lock()
        self.latest_depth_frame = np.full((height, width), 500, dtype=np.uint16) if use_depth else None

    def async_read(self):
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)


class FakeBus:
    is_connected = True
    motors = dict.fromkeys(MOTORS)

    def get_action(self):
        return {m: float(i) for i, m in enumerate(MOTORS)}

    def get_effort(self):
        return {m: 10.0 + i for i, m in enumerate(MOTORS)}

    def get_velocity(self):
        return {m: 100.0 + i for i, m in enumerate(MOTORS) if m != "gripper"}


def make_follower(use_effort: bool, use_depth: bool) -> PiperFollower:
    f = PiperFollower.__new__(PiperFollower)
    f.id = "test_follower"
    f.bus = FakeBus()
    f.cameras = {"top": FakeCamera(use_depth=use_depth), "wrist": FakeCamera(use_depth=False)}
    f._camera_executor = ThreadPoolExecutor(max_workers=2)
    f.config = SimpleNamespace(
        use_effort=use_effort,
        use_depth_observation=use_depth,
        depth_scale=0.001,
        depth_min_m=0.20,
        depth_max_m=0.80,
        depth_raw_dir="",
    )
    return f


def dataset_features(robot: PiperFollower) -> dict:
    """lerobot/scripts/lerobot_record.py의 record()가 dataset_features를 만드는 것과 동일한 경로."""
    teleop_action_processor, _, robot_observation_processor = make_default_processors()
    return combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )


def test_effort_off_keeps_state_pos_only():
    feats = dataset_features(make_follower(use_effort=False, use_depth=False))
    assert feats["observation.state"]["names"] == [f"{m}.pos" for m in MOTORS]
    assert feats["observation.state"]["shape"] == (7,)
    assert feats["action"]["names"] == [f"{m}.pos" for m in MOTORS]


def test_effort_on_appends_to_observation_state_with_names():
    robot = make_follower(use_effort=True, use_depth=False)
    feats = dataset_features(robot)

    expected = (
        [f"{m}.pos" for m in MOTORS]
        + [f"{m}.effort" for m in MOTORS]
        + [f"{m}.vel" for m in MOTORS if m != "gripper"]
    )
    # effort는 observation.effort 같은 별도 컬럼이 아님 — state 벡터에 이어 붙는다.
    assert "observation.effort" not in feats
    assert feats["observation.state"]["names"] == expected
    assert feats["observation.state"]["shape"] == (7 + 7 + 6,)
    # action은 pos만 — effort가 action 쪽으로 새면 정책 출력 차원이 망가진다.
    assert feats["action"]["names"] == [f"{m}.pos" for m in MOTORS]

    # 실제 프레임에서 값이 이름 순서대로 들어가는지 (나중에 인덱스로 슬라이싱 가능한지)
    frame = build_dataset_frame(feats, robot.get_observation(), prefix="observation")
    state = frame["observation.state"]
    assert state.dtype == np.float32
    assert state.shape == (20,)
    assert state[:7].tolist() == [float(i) for i in range(7)]  # pos
    assert state[7:14].tolist() == [10.0 + i for i in range(7)]  # effort
    assert state[14:].tolist() == [100.0 + i for i in range(6)]  # vel
    # names로 되짚어서 뽑는 실제 사용 패턴
    names = feats["observation.state"]["names"]
    assert state[names.index("joint3.effort")] == 12.0


def test_depth_on_adds_separate_video_stream():
    robot = make_follower(use_effort=False, use_depth=True)
    feats = dataset_features(robot)

    assert feats["observation.images.top_depth"]["dtype"] == "video"
    assert feats["observation.images.top_depth"]["shape"] == (4, 4, 3)
    assert "observation.images.wrist_depth" not in feats  # use_depth 꺼진 카메라는 제외
    assert "observation.images.top" in feats  # color는 그대로

    frame = build_dataset_frame(feats, robot.get_observation(), prefix="observation")
    assert frame["observation.images.top_depth"].shape == (4, 4, 3)
    assert frame["observation.images.top_depth"].dtype == np.uint8


if __name__ == "__main__":
    test_effort_off_keeps_state_pos_only()
    test_effort_on_appends_to_observation_state_with_names()
    test_depth_on_adds_separate_video_stream()
    print("OK: dataset feature 스키마 테스트 통과 (effort -> observation.state, depth -> 별도 video)")
