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
    # 트립 순간의 실측 자세 — 얼어붙힐 목표로 쓰인다
    present = {"joint1": 3.0, "joint2": -4.0}
    set_action_calls = 0

    def __init__(self, effort: dict[str, float]):
        self._effort = effort
        self.set_action_calls = 0
        self.parking_calls = 0
        self.effort_reads = 0
        self.ramp_calls = []
        self.sent = []

    def parking(self):
        self.parking_calls += 1

    def ramp_to(self, target, ramp_s=2.0, step_s=0.05):
        # park 복귀는 최고속 점프가 아니라 잘게 쪼갠 이동이어야 한다 — 외력으로
        # 트립된 직후엔 사람 손이 팔에 닿아 있을 수 있으므로.
        self.ramp_calls.append((dict(target), ramp_s))

    def get_action(self):
        return dict(self.present)

    def get_effort(self):
        self.effort_reads += 1
        return self._effort

    def is_overloaded(self, effort, limit):
        return any(abs(v) > limit for v in effort.values())

    def sync_read(self, name, motors=None):
        return {m: 0.0 for m in self.motors}

    def set_action(self, action, is_conv=True):
        self.set_action_calls += 1
        self.sent.append(dict(action))
        return action


def make_follower(
    safety_enabled: bool,
    effort: dict[str, float],
    limit: float = 8.0,
    on_overload: str = "hold",
    park_ramp_s: float = 0.0,
    hold_resend: bool = True,
) -> PiperFollower:
    follower = PiperFollower.__new__(PiperFollower)
    follower.id = "test_follower"
    follower.bus = FakeBus(effort)
    follower.cameras = {}
    follower._safety_tripped = False
    follower._safety_park_thread = None
    follower._safety_park_active = False
    follower._safety_hold_pos = {}
    follower.config = SimpleNamespace(
        use_action_offset=False,
        max_relative_target=None,
        safety_enabled=safety_enabled,
        safety_effort_limit=limit,
        safety_on_overload=on_overload,
        safety_park_ramp_s=park_ramp_s,
        safety_hold_resend=hold_resend,
    )
    return follower


def test_normal_effort_sends_action():
    f = make_follower(safety_enabled=True, effort={"joint1": 1.0, "joint2": 1.0})
    result = f.send_action({"joint1.pos": 10.0, "joint2.pos": -5.0})
    assert f.bus.set_action_calls == 1
    assert result == {"joint1.pos": 10.0, "joint2.pos": -5.0}


def test_overload_freezes_at_trip_pose_and_keeps_streaming():
    """리더 목표는 무시하되 트립 자세를 계속 재전송해야 한다.

    아무것도 안 보내면 Piper가 목표 스트림이 끊긴 걸로 보고 유지 토크를 놓아서
    팔이 늘어진다("에포트 커지면 힘이 풀린다"의 원인) — 그 회귀를 막는 테스트.
    """
    f = make_follower(safety_enabled=True, effort={"joint1": 1.0, "joint2": 9.0})
    frozen = dict(f.bus.present)
    result = f.send_action({"joint1.pos": 10.0, "joint2.pos": -5.0})
    assert f.bus.sent == [frozen]  # 리더 목표(10, -5)가 아니라 트립 자세가 나감
    assert result == {"joint1.pos": 3.0, "joint2.pos": -4.0}

    # 매 스텝 계속 재전송되어야 함 (한 번만 보내고 끊기면 똑같이 늘어진다)
    f.send_action({"joint1.pos": 50.0, "joint2.pos": -50.0})
    f.send_action({"joint1.pos": 60.0, "joint2.pos": -60.0})
    assert f.bus.sent == [frozen, frozen, frozen]


def test_hold_resend_off_sends_nothing():
    # 늘어짐 원인 구분용 스위치 — 끄면 옛 동작(명령 완전 중단)
    f = make_follower(safety_enabled=True, effort={"joint1": 1.0, "joint2": 9.0},
                      hold_resend=False)
    f.send_action({"joint1.pos": 10.0, "joint2.pos": -5.0})
    assert f.bus.sent == []


def test_safety_disabled_ignores_overload():
    f = make_follower(safety_enabled=False, effort={"joint1": 99.0, "joint2": 99.0})
    result = f.send_action({"joint1.pos": 10.0, "joint2.pos": -5.0})
    assert f.bus.set_action_calls == 1
    assert result == {"joint1.pos": 10.0, "joint2.pos": -5.0}


def test_overload_park_moves_to_parking_once():
    f = make_follower(
        safety_enabled=True, effort={"joint1": 1.0, "joint2": 9.0}, on_overload="park"
    )
    f.send_action({"joint1.pos": 10.0, "joint2.pos": -5.0})
    f._safety_park_thread.join(timeout=5.0)
    assert f.bus.parking_calls == 1
    assert f.safety_tripped is True
    assert f._safety_park_active is False  # 스레드가 끝나면 루프에 목표를 넘긴다

    # 트립 후에는 래치 — effort를 다시 읽지 않고, 리더 목표도 무시한다.
    # 다만 파킹 자세는 계속 재전송해야 한다(안 그러면 늘어짐).
    reads_before = f.bus.effort_reads
    f.bus.sent.clear()
    f.send_action({"joint1.pos": 20.0, "joint2.pos": -20.0})
    assert f.bus.effort_reads == reads_before
    assert f.bus.parking_calls == 1
    assert f.bus.sent == [dict(f.bus.present)]


def test_no_send_while_parking_thread_owns_bus():
    """parking 이동 중에는 제어 루프가 전송하면 안 된다 — 둘이 같이 쏘면 서로
    목표를 덮어써서 팔이 떤다. (스레드 타이밍에 안 흔들리게 상태를 직접 세팅)"""
    f = make_follower(safety_enabled=True, effort={"joint1": 1.0, "joint2": 1.0},
                      on_overload="park")
    f._safety_tripped = True
    f._safety_park_active = True
    f._safety_hold_pos = dict(f.bus.present)
    f.send_action({"joint1.pos": 10.0, "joint2.pos": -5.0})
    assert f.bus.sent == []


def test_overload_park_ramps_when_ramp_s_set():
    f = make_follower(
        safety_enabled=True, effort={"joint1": 1.0, "joint2": 9.0},
        on_overload="park", park_ramp_s=4.0,
    )
    f.send_action({"joint1.pos": 10.0, "joint2.pos": -5.0})
    f._safety_park_thread.join(timeout=5.0)
    assert len(f.bus.ramp_calls) == 1
    target, ramp_s = f.bus.ramp_calls[0]
    assert ramp_s == 4.0
    assert "gripper" not in target  # 잡고 있는 물체/손이 끼지 않도록 gripper는 제외
    assert f.bus.parking_calls == 1  # ramp 후 최종 정렬


def test_overload_hold_does_not_park():
    f = make_follower(
        safety_enabled=True, effort={"joint1": 1.0, "joint2": 9.0}, on_overload="hold"
    )
    f.send_action({"joint1.pos": 10.0, "joint2.pos": -5.0})
    assert f.bus.parking_calls == 0
    assert f.bus.sent == [dict(f.bus.present)]  # 트립 자세 유지 (늘어지지 않게)
    assert f.safety_tripped is True


def test_reset_safety_allows_commands_again():
    f = make_follower(
        safety_enabled=True, effort={"joint1": 1.0, "joint2": 9.0}, on_overload="hold"
    )
    f.send_action({"joint1.pos": 10.0, "joint2.pos": -5.0})
    f.bus._effort = {"joint1": 1.0, "joint2": 1.0}  # 원인 제거
    f.reset_safety()
    f.bus.sent.clear()
    f.send_action({"joint1.pos": 10.0, "joint2.pos": -5.0})
    assert f.bus.sent == [{"joint1": 10.0, "joint2": -5.0}]  # 다시 리더 목표를 따른다


if __name__ == "__main__":
    test_is_overloaded()
    test_normal_effort_sends_action()
    test_overload_freezes_at_trip_pose_and_keeps_streaming()
    test_hold_resend_off_sends_nothing()
    test_safety_disabled_ignores_overload()
    test_overload_park_moves_to_parking_once()
    test_no_send_while_parking_thread_owns_bus()
    test_overload_park_ramps_when_ramp_s_set()
    test_overload_hold_does_not_park()
    test_reset_safety_allows_commands_again()
    print("OK: safety cutoff mock 테스트 통과")
