"""torque 해제 방식(release_torque_safely) mock 테스트 (하드웨어 불필요).

실기 측정(2026-07-28) 결과 torque를 풀 때 실제로 떨어지는 건 손목(joint5)뿐이고
(joint1~4/6은 0.00도) 그 낙차가 24.4도였다 — 그게 "쿵" 하고 놓이는 느낌의 정체.
그래서 lower 모드는 팔을 옮기지 않고 손목만 미리 내린 뒤 해제한다. 이 테스트는
각 모드가 실제로 어떤 이동을 하는지(또는 안 하는지), lower가 손목 외의 관절은
건드리지 않는지, gripper를 절대 건드리지 않는지를 확인한다.

실행: PYTHONPATH=. python scripts/tools/test_release_mock.py
"""

import sys
from types import SimpleNamespace

try:
    from lerobot_robot_piper.motors.piper_motors_bus import PiperMotorsBus
    from lerobot_robot_piper.motors.tables import WRIST_RELEASE_DROP_DEG
except ModuleNotFoundError as e:  # pragma: no cover - 랩 PC 외 환경
    print(f"SKIP: piper_sdk/lerobot import 실패 — {e}")
    sys.exit(0)


JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


class FakePiper:
    def __init__(self):
        self.disable_calls = 0

    def DisablePiper(self):
        self.disable_calls += 1


def make_bus(start: dict[str, float]) -> PiperMotorsBus:
    """__init__(piper_sdk 연결)을 건너뛰고 필요한 부분만 채운 bus."""
    bus = PiperMotorsBus.__new__(PiperMotorsBus)
    bus.id = "test_follower"
    bus.piper = FakePiper()
    bus._pose = dict(start)
    bus.sent = []
    # 손목 각도 <-> 정규화값 변환에 calibration 범위가 필요 (piper_follower.py의
    # joint5: -65000~65000 = -65~65도)
    bus.calibration = {
        "joint5": SimpleNamespace(range_min=-65000, range_max=65000),
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
    bus.release_torque_safely(mode="in_place")
    assert bus.sent == []
    assert bus.piper.disable_calls == 1


def test_park_moves_to_parking_then_releases():
    bus = make_bus(START)
    parking_calls = []
    bus.parking = lambda: parking_calls.append(1)
    bus.release_torque_safely(mode="park")
    assert parking_calls == [1]
    assert bus.piper.disable_calls == 1


def test_lower_drops_only_wrist_and_keeps_gripper():
    bus = make_bus(START)
    bus.release_torque_safely(mode="lower", ramp_s=0.2, settle_s=0.0)

    # 한 번에 점프하지 않고 여러 스텝으로 나눠서 내려간다
    assert len(bus.sent) > 1
    final = bus.sent[-1]
    # joint5(손목)만 움직이고 나머지 팔 관절은 그 자리 그대로여야 한다
    for joint in JOINTS:
        if joint == "joint5":
            continue
        assert abs(final[joint] - START[joint]) < 1e-9, joint
    # 24.4도 -> 정규화 델타: 24.4*1000 / (65000-(-65000)) * 200
    expected = START["joint5"] + WRIST_RELEASE_DROP_DEG * 1000.0 / 130000 * 200
    assert abs(final["joint5"] - expected) < 1e-6
    # gripper는 전 구간에서 손도 대지 않음 (잡고 있는 물체/손이 끼지 않도록)
    assert all(abs(step["gripper"] - START["gripper"]) < 1e-9 for step in bus.sent)
    assert bus.piper.disable_calls == 1


def test_lower_uses_custom_wrist_drop():
    bus = make_bus(START)
    bus.release_torque_safely(mode="lower", wrist_drop_deg=10.0, ramp_s=0.2, settle_s=0.0)
    assert abs(bus.sent[-1]["joint5"] - (10.0 * 1000.0 / 130000 * 200)) < 1e-6


def test_measure_wrist_drop_reports_degrees():
    bus = make_bus(START)
    # torque를 푸는 순간 손목이 정규화 37.538(=24.4도)만큼 떨어지는 상황을 흉내
    def disable():
        bus._pose["joint5"] = 37.538461538
    bus.piper.DisablePiper = disable
    drop = bus.measure_wrist_drop(settle_s=0.0)
    assert abs(drop - 24.4) < 0.01, drop


def test_unknown_mode_falls_back_to_in_place():
    bus = make_bus(START)
    bus.release_torque_safely(mode="nonsense")
    assert bus.sent == []
    assert bus.piper.disable_calls == 1


if __name__ == "__main__":
    test_in_place_does_not_move()
    test_park_moves_to_parking_then_releases()
    test_lower_drops_only_wrist_and_keeps_gripper()
    test_lower_uses_custom_wrist_drop()
    test_measure_wrist_drop_reports_degrees()
    test_unknown_mode_falls_back_to_in_place()
    print("OK: torque 해제 모드 mock 테스트 통과")
