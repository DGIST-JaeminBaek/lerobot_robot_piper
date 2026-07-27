"""effort 기반 실시간 안전 컷오프 mock 테스트 (하드웨어 불필요).

리플레이/정책 출력이 관절 명령으로 바뀌어 로봇에 나가는 지점(PiperFollower.send_action
-> bus.set_action 직전)에서, effort가 임계값을 넘으면 명령을 보류하는지 확인한다.
use_effort(데이터셋 로깅용)가 꺼져 있어도 안전 체크는 독립 동작해야 한다.
실행: PYTHONPATH=. python scripts/tools/test_safety_mock.py
"""

from types import SimpleNamespace

try:
    from lerobot_robot_piper.motors.piper_motors_bus import PiperMotorsBus
    from lerobot_robot_piper.piper_follower import PiperFollower
except ModuleNotFoundError as e:
    if "depth_utils" not in str(e):
        raise
    # jmbaek의 depth 백포트가 적용된 lerobot clone에서만 import된다
    # (stock pip lerobot 0.4.4에는 lerobot.datasets.depth_utils가 없음).
    # 랩 PC에서는 정상 실행되고, 그 외 환경에서는 조용히 건너뛴다.
    import sys
    print("SKIP: 패치된 lerobot(depth 백포트)이 필요합니다 — docs/depth/README.md 참고")
    sys.exit(0)


def test_is_overloaded():
    bus = PiperMotorsBus.__new__(PiperMotorsBus)
    assert bus.is_overloaded({"joint1": 3.0, "joint2": -2.0}, limit=8.0) is False
    assert bus.is_overloaded({"joint1": 3.0, "joint2": -9.0}, limit=8.0) is True


class FakeBus:
    is_connected = True
    motors = {"joint1": None, "joint2": None}
    set_action_calls = 0

    def __init__(self, effort: dict[str, float]):
        self._effort = effort
        self.set_action_calls = 0

    def get_action(self):
        return {m: 0.0 for m in self.motors}

    def get_effort(self):
        return self._effort

    def is_overloaded(self, effort, limit):
        return any(abs(v) > limit for v in effort.values())

    def sync_read(self, name, motors=None):
        return {m: 0.0 for m in self.motors}

    def set_action(self, action, is_conv=True):
        self.set_action_calls += 1
        return action


def make_follower(safety_enabled: bool, effort: dict[str, float], limit: float = 8.0) -> PiperFollower:
    follower = PiperFollower.__new__(PiperFollower)
    follower.id = "test_follower"
    follower.bus = FakeBus(effort)
    follower.cameras = {}
    follower.config = SimpleNamespace(
        use_action_offset=False,
        max_relative_target=None,
        safety_enabled=safety_enabled,
        safety_effort_limit=limit,
    )
    return follower


def test_normal_effort_sends_action():
    f = make_follower(safety_enabled=True, effort={"joint1": 1.0, "joint2": 1.0})
    result = f.send_action({"joint1.pos": 10.0, "joint2.pos": -5.0})
    assert f.bus.set_action_calls == 1
    assert result == {"joint1.pos": 10.0, "joint2.pos": -5.0}


def test_overload_holds_last_position_instead_of_sending():
    f = make_follower(safety_enabled=True, effort={"joint1": 1.0, "joint2": 9.0})
    result = f.send_action({"joint1.pos": 10.0, "joint2.pos": -5.0})
    assert f.bus.set_action_calls == 0  # 명령이 로봇으로 안 나감
    assert result == {"joint1.pos": 0.0, "joint2.pos": 0.0}  # 현재 위치 유지


def test_safety_disabled_ignores_overload():
    f = make_follower(safety_enabled=False, effort={"joint1": 99.0, "joint2": 99.0})
    result = f.send_action({"joint1.pos": 10.0, "joint2.pos": -5.0})
    assert f.bus.set_action_calls == 1
    assert result == {"joint1.pos": 10.0, "joint2.pos": -5.0}


if __name__ == "__main__":
    test_is_overloaded()
    test_normal_effort_sends_action()
    test_overload_holds_last_position_instead_of_sending()
    test_safety_disabled_ignores_overload()
    print("OK: safety cutoff mock 테스트 통과")
