"""torque 해제 방식(release_torque_safely) mock 테스트 (하드웨어 불필요).

실기 측정(2026-07-28) 결과 torque를 풀 때 실제로 떨어지는 건 손목(joint5)뿐이고
(joint1~4/6은 0.00도) 그 낙차가 24.4도였다 — 그게 "쿵" 하고 놓이는 느낌의 정체.
그래서 lower 모드는 팔을 옮기지 않고 손목만 미리 내린 뒤 해제한다. 이 테스트는
각 모드가 실제로 어떤 이동을 하는지(또는 안 하는지), lower가 손목 외의 관절은
건드리지 않는지, 그리퍼가 제대로 실능(0x00)되는지를 확인한다.

그리퍼는 팔 모터와 별개 노드(0x159)라 DisablePiper()로는 안 풀린다 — 실기에서
"팔은 풀렸는데 그리퍼가 계속 물고 있다"로 나타났던 문제.

실행: PYTHONPATH=. python scripts/tools/test_release_mock.py
"""

import sys
from types import SimpleNamespace

try:
    from lerobot.motors import MotorNormMode
    from lerobot_robot_piper.motors.piper_motors_bus import PiperMotorsBus
    from lerobot_robot_piper.motors.tables import WRIST_RELEASE_REST_DEG
except ModuleNotFoundError as e:  # pragma: no cover - 랩 PC 외 환경
    print(f"SKIP: piper_sdk/lerobot import 실패 — {e}")
    sys.exit(0)


JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


class FakePiper:
    def __init__(self, disable_after: int = 1):
        self.disable_calls = 0
        self.gripper_calls = []  # (angle_raw, effort, code)
        # 실능 명령 몇 번째에 실제로 풀릴지 (1이면 첫 시도에 성공)
        self.disable_after = disable_after
        self.gripper_enabled = True
        self.mode_ctrl_calls = 0

    def DisablePiper(self):
        self.disable_calls += 1

    def ModeCtrl(self, ctrl_mode, move_mode, speed, is_mit):
        # 그리퍼 각도 명령이 먹으려면 팔이 CAN 제어 모드여야 한다 — cycle_gripper가
        # 이걸 먼저 보내는지 확인용
        self.mode_ctrl_calls += 1

    def GripperCtrl(self, angle, effort, code, set_zero):
        self.gripper_calls.append((angle, effort, code))
        if code in (0x00, 0x02):
            disables = sum(1 for c in self.gripper_calls if c[2] in (0x00, 0x02))
            if disables >= self.disable_after:
                self.gripper_enabled = False
        else:
            self.gripper_enabled = True

    def GetArmGripperMsgs(self):
        status = (1 << 6) if self.gripper_enabled else 0
        return SimpleNamespace(gripper_state=SimpleNamespace(status_code=status))


def make_bus(start: dict[str, float]) -> PiperMotorsBus:
    """__init__(piper_sdk 연결)을 건너뛰고 필요한 부분만 채운 bus."""
    bus = PiperMotorsBus.__new__(PiperMotorsBus)
    bus.id = "test_follower"
    bus.piper = FakePiper()
    bus._pose = dict(start)
    bus.sent = []
    # 손목 각도 <-> 정규화값 변환에 calibration 범위가 필요 (piper_follower.py의
    # joint5: -65000~65000 = -65~65도)
    # _unnormalize()가 norm_mode를 보므로 motors도 필요
    bus.motors = {
        "gripper": SimpleNamespace(norm_mode=MotorNormMode.RANGE_0_100, model="AGILEX-S"),
    }
    bus.calibration = {
        "joint5": SimpleNamespace(range_min=-65000, range_max=65000),
        "gripper": SimpleNamespace(range_min=0, range_max=68000, drive_mode=0),
    }

    def get_action():
        return dict(bus._pose)

    def set_action(action, is_conv=True):
        bus.sent.append(dict(action))
        bus._pose = dict(action)
        return dict(action)

    bus.get_action = get_action
    bus.set_action = set_action
    return bus


START = {j: 0.0 for j in JOINTS} | {"gripper": 42.0}


def test_in_place_does_not_move():
    bus = make_bus(START)
    bus.release_torque_safely(mode="in_place", gripper_cycle=False)
    assert bus.sent == []
    assert bus.piper.disable_calls == 1


def test_park_moves_to_parking_then_releases():
    bus = make_bus(START)
    parking_calls = []
    bus.parking = lambda: parking_calls.append(1)
    bus.release_torque_safely(mode="park", gripper_cycle=False)
    assert parking_calls == [1]
    assert bus.piper.disable_calls == 1


def test_lower_drops_only_wrist_and_keeps_gripper():
    bus = make_bus(START)  # 손목 0도에서 시작 -> 정지각(24.4도)까지 내려가야 함
    bus.release_torque_safely(mode="lower", ramp_s=0.2, settle_s=0.0, gripper_cycle=False)

    # 한 번에 점프하지 않고 여러 스텝으로 나눠서 내려간다
    assert len(bus.sent) > 1
    final = bus.sent[-1]
    # joint5(손목)만 움직이고 나머지 팔 관절은 그 자리 그대로여야 한다
    for joint in JOINTS:
        if joint == "joint5":
            continue
        assert abs(final[joint] - START[joint]) < 1e-9, joint
    # 24.4도(절대) -> 정규화: (24400 - (-65000)) / 130000 * 200 - 100
    expected = (WRIST_RELEASE_REST_DEG * 1000.0 + 65000) / 130000 * 200 - 100
    assert abs(final["joint5"] - expected) < 1e-6
    # gripper는 전 구간에서 손도 대지 않음 (잡고 있는 물체/손이 끼지 않도록)
    assert all(abs(step["gripper"] - START["gripper"]) < 1e-9 for step in bus.sent)
    assert bus.piper.disable_calls == 1


def test_lower_uses_custom_wrist_rest():
    bus = make_bus(START)
    bus.release_torque_safely(mode="lower", wrist_rest_deg=10.0, ramp_s=0.2, settle_s=0.0,
                              gripper_cycle=False)
    assert abs(bus.sent[-1]["joint5"] - ((10000 + 65000) / 130000 * 200 - 100)) < 1e-6


def test_lower_never_lifts_wrist_back_up():
    """손목이 이미 정지각보다 아래면 그대로 둔다 — 도로 들어올리면 놓을 때 다시 떨어진다.

    실기에서 손목 30도(이미 정지)에 상대 델타를 더 줬다가 해제하니 29.9도로 튕겨
    올라온 케이스의 회귀 테스트.
    """
    start = dict(START)
    start["joint5"] = (40000 + 65000) / 130000 * 200 - 100  # 40도 = 정지각보다 아래
    bus = make_bus(start)
    bus.release_torque_safely(mode="lower", wrist_rest_deg=24.4, ramp_s=0.2, settle_s=0.0,
                              gripper_cycle=False)
    assert all(abs(step["joint5"] - start["joint5"]) < 1e-9 for step in bus.sent)


def test_measure_wrist_rest_reports_angle_and_drop():
    bus = make_bus(START)  # 0도에서 시작
    # torque를 푸는 순간 손목이 24.4도로 떨어지는 상황을 흉내
    def disable():
        bus._pose["joint5"] = (24400 + 65000) / 130000 * 200 - 100
    bus.piper.DisablePiper = disable
    rest, drop = bus.measure_wrist_rest(settle_s=0.0)
    assert abs(rest - 24.4) < 0.01, rest
    assert abs(drop - 24.4) < 0.01, drop


def test_release_always_disables_gripper():
    """그리퍼는 팔 모터와 별개 노드라 DisablePiper()만으로는 안 풀린다 —
    해제 경로에서 반드시 code=0x00(실능)을 보내야 한다."""
    bus = make_bus(START)
    bus.release_torque_safely(mode="in_place", gripper_cycle=False)
    assert bus.piper.disable_calls == 1
    codes = [c for _, _, c in bus.piper.gripper_calls]
    assert 0x00 in codes, codes
    # 실능은 effort 0으로 (각도 명령으로 움직이지 않게)
    disable_call = next(c for c in bus.piper.gripper_calls if c[2] == 0x00)
    assert disable_call[1] == 0
    assert bus.is_gripper_enabled() is False


def test_disable_gripper_retries_until_status_confirms():
    """한 번 쏘고 끝내면 실기에서 안 풀렸다 — 상태코드로 확인될 때까지 재시도해야 한다."""
    bus = make_bus(START)
    bus.piper.disable_after = 3  # 세 번째 실능 명령에서야 풀리는 상황
    assert bus.disable_gripper(wait_s=0.0) is True
    disable_codes = [c[2] for c in bus.piper.gripper_calls if c[2] in (0x00, 0x02)]
    assert disable_codes == [0x00, 0x02, 0x00]  # 0x00 / 0x02 번갈아


def test_disable_gripper_reports_failure():
    bus = make_bus(START)
    bus.piper.disable_after = 99  # 끝까지 안 풀리는 상황
    assert bus.disable_gripper(retries=3, wait_s=0.0) is False


def test_gripper_cycle_opens_then_closes_to_parking():
    bus = make_bus(START)
    bus.release_torque_safely(mode="in_place", gripper_cycle=True,
                              gripper_open=100.0, gripper_wait_s=0.0)
    # 사용(0x03) 명령 두 번 = 열기 -> 닫기, 그 다음 실능(0x00)
    # 팔이 CAN 제어 모드가 아니면 그리퍼 각도 명령이 무시된다(실기 확인)
    assert bus.piper.mode_ctrl_calls >= 1
    enable_calls = [c for c in bus.piper.gripper_calls if c[2] == 0x03]
    assert len(enable_calls) == 2, bus.piper.gripper_calls
    assert enable_calls[0][0] == 68000   # 정규화 100 -> raw 68mm (완전 열림)
    assert enable_calls[1][0] == 0       # 파킹 위치(닫힘)
    assert bus.piper.gripper_calls[-1][2] == 0x00  # 마지막은 실능


def test_unknown_mode_falls_back_to_in_place():
    bus = make_bus(START)
    bus.release_torque_safely(mode="nonsense", gripper_cycle=False)
    assert bus.sent == []
    assert bus.piper.disable_calls == 1


if __name__ == "__main__":
    test_in_place_does_not_move()
    test_park_moves_to_parking_then_releases()
    test_lower_drops_only_wrist_and_keeps_gripper()
    test_lower_uses_custom_wrist_rest()
    test_lower_never_lifts_wrist_back_up()
    test_measure_wrist_rest_reports_angle_and_drop()
    test_release_always_disables_gripper()
    test_disable_gripper_retries_until_status_confirms()
    test_disable_gripper_reports_failure()
    test_gripper_cycle_opens_then_closes_to_parking()
    test_unknown_mode_falls_back_to_in_place()
    print("OK: torque 해제 모드 mock 테스트 통과")


# ── park_lower: 파킹 자세로 간 뒤 손목까지 내리고 해제 ────
# 예전 기본값 "lower"는 팔이 있던 자리에 그대로 늘어져서, 추론이 끝난 위치
# (보드 앞 등)에 팔이 남았다. 보관 자세로 돌아가야 한다.
class _OrderRecordingBus:
    """release_torque_safely의 동작 순서만 확인하는 스텁."""

    move_mode = 1
    move_speed_rate = 30

    def __init__(self):
        self.id = "test"
        self.calls: list[str] = []
        self.piper = self

    def parking(self):
        self.calls.append("parking")

    def wrist_rest_target(self, rest_deg):
        self.calls.append("wrist_rest_target")
        return {"joint5": 0.0}

    def ramp_to(self, target, ramp_s):
        self.calls.append("ramp_to")

    def cycle_gripper(self, **kwargs):
        self.calls.append("cycle_gripper")

    def DisablePiper(self):
        self.calls.append("DisablePiper")

    def disable_gripper(self):
        self.calls.append("disable_gripper")

    release_torque_safely = PiperMotorsBus.release_torque_safely


def _release(mode):
    bus = _OrderRecordingBus()
    bus.release_torque_safely(mode=mode, ramp_s=0.0, settle_s=0.0)
    return bus.calls


def test_park_lower_parks_before_lowering_the_wrist():
    """순서가 뒤집히면 wrist_rest_target이 파킹 전 자세를 기준으로 잡는다."""
    calls = _release("park_lower")
    assert calls.index("parking") < calls.index("wrist_rest_target")
    assert calls.index("wrist_rest_target") < calls.index("ramp_to")
    assert calls.index("ramp_to") < calls.index("DisablePiper")


def test_park_lower_is_the_config_default():
    from lerobot_robot_piper.config_piper import PiperFollowerConfig

    assert PiperFollowerConfig.park_release_mode == "park_lower"


def test_park_alone_does_not_lower_the_wrist():
    assert "ramp_to" not in _release("park")
    assert "parking" in _release("park")


def test_lower_alone_does_not_park():
    calls = _release("lower")
    assert "parking" not in calls
    assert "ramp_to" in calls


def test_in_place_moves_nothing():
    calls = _release("in_place")
    assert "parking" not in calls and "ramp_to" not in calls
    assert "DisablePiper" in calls


def test_gripper_is_cycled_before_torque_release():
    """팔이 늘어진 뒤 그리퍼를 여닫으면 반력으로 팔이 흔들린다."""
    calls = _release("park_lower")
    assert calls.index("cycle_gripper") < calls.index("DisablePiper")


# ── park=False일 때 파킹 모드 강등 ────────────────────────
# park_lower를 추가하면서 생긴 구멍: 조기 종료(park=False)인데도 파킹 이동이
# 일어나면, 사람이 "여기서 멈춰"라고 한 의도와 반대로 팔이 움직인다.
def _downgrade(release_mode, park):
    """piper_follower.disconnect의 강등 규칙과 같은 수식."""
    if not park:
        return {"park": "in_place", "park_lower": "lower"}.get(release_mode, release_mode)
    return release_mode


def test_park_lower_is_downgraded_when_not_parking():
    assert _downgrade("park_lower", park=False) == "lower"
    assert _downgrade("park", park=False) == "in_place"


def test_modes_without_parking_are_untouched():
    assert _downgrade("lower", park=False) == "lower"
    assert _downgrade("in_place", park=False) == "in_place"


def test_nothing_is_downgraded_when_parking():
    for mode in ("park_lower", "park", "lower", "in_place"):
        assert _downgrade(mode, park=True) == mode


def test_follower_downgrade_matches_this_rule():
    import inspect

    from lerobot_robot_piper.piper_follower import PiperFollower

    source = inspect.getsource(PiperFollower.disconnect)
    assert '"park": "in_place", "park_lower": "lower"' in source
