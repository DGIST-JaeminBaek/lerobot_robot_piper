"""observation.effort 로직 mock 테스트 (하드웨어 불필요).

PiperFollower.__init__을 건너뛰고 bus/config만 mock으로 채워서
_effort_ft / observation_features / get_observation()의 effort 분기를 검증한다.
실행: python scripts/tools/test_effort_mock.py
"""

from types import SimpleNamespace

from lerobot_robot_piper.piper_follower import PiperFollower


class FakeBus:
    is_connected = True
    motors = {"joint1": None, "joint2": None, "joint3": None,
              "joint4": None, "joint5": None, "joint6": None, "gripper": None}

    def get_action(self):
        return {m: 0.0 for m in self.motors}

    def get_effort(self):
        return {m: 1.23 for m in self.motors}

    def get_velocity(self):
        return {m: 4.56 for m in self.motors if m != "gripper"}


def make_follower(use_effort: bool) -> PiperFollower:
    follower = PiperFollower.__new__(PiperFollower)
    follower.id = "test_follower"
    follower.bus = FakeBus()
    follower.cameras = {}
    follower.config = SimpleNamespace(use_effort=use_effort, use_depth_observation=False)
    return follower


def test_effort_off_by_default():
    f = make_follower(use_effort=False)
    assert f._effort_ft == {}
    assert f._velocity_ft == {}
    assert "joint1.effort" not in f.observation_features
    assert "joint1.vel" not in f.observation_features
    obs = f.get_observation()
    assert not any(k.endswith(".effort") or k.endswith(".vel") for k in obs)


def test_effort_on_adds_effort_and_velocity_fields():
    # NEXT(외력 추정)는 effort와 같은 타임스탬프의 velocity가 필요 —
    # 별도 플래그 없이 use_effort에 velocity도 같이 켜지는지 확인.
    f = make_follower(use_effort=True)
    assert set(f._effort_ft) == {f"{m}.effort" for m in FakeBus.motors}
    assert set(f._velocity_ft) == {f"{m}.vel" for m in FakeBus.motors if m != "gripper"}
    assert set(f.observation_features) >= set(f._effort_ft) | set(f._velocity_ft)
    obs = f.get_observation()
    assert obs["joint1.effort"] == 1.23
    assert obs["gripper.effort"] == 1.23
    assert obs["joint1.vel"] == 4.56
    assert "gripper.vel" not in obs  # SDK에 그리퍼 motor_speed 없음
    assert "joint1.pos" in obs  # 기존 pos 필드는 그대로 유지


if __name__ == "__main__":
    test_effort_off_by_default()
    test_effort_on_adds_effort_and_velocity_fields()
    print("OK: effort/velocity mock 테스트 통과")
