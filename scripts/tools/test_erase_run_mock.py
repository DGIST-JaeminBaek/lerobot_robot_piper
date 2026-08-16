#!/usr/bin/env python3
"""erase_run.py 로직 검증 — 하드웨어 없이 mock으로.

리포 컨벤션(test_*_mock.py)을 따른다. 실행:
    python test_erase_run_mock.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from erase_run import ACTION_NAMES, Clutch, deviation, run_attempt  # noqa: E402

ZERO = {k: 0.0 for k in ACTION_NAMES}


class MockRobot:
    """팔로워: 보낸 action이 곧 다음 관측이 된다(즉시 추종 가정)."""

    def __init__(self, state=None):
        self.state = dict(state or ZERO)
        self.sent = []
        self.parked = 0

    def get_observation(self):
        return dict(self.state)

    def send_action(self, action):
        self.sent.append(dict(action))
        self.state.update(action)
        return action

    def parking(self):
        self.parked += 1
        self.state = dict(ZERO)


class MockLeader:
    """리더암: 스텝별 joint1 절대 자세를 미리 정해둔 대로 낸다."""

    def __init__(self, positions):
        self.positions = list(positions)
        self.i = 0

    def get_action(self):
        v = self.positions[min(self.i, len(self.positions) - 1)]
        self.i += 1
        return {**ZERO, "joint1.pos": float(v)}


class MockToggle:
    """스텝별 개입 on/off 스케줄. KeyToggle과 같은 인터페이스(.active/.abort)."""

    def __init__(self, schedule):
        self.schedule = list(schedule)
        self.i = 0
        self.abort = False

    @property
    def active(self):
        v = self.schedule[min(self.i, len(self.schedule) - 1)]
        self.i += 1
        return v


def const_policy(_obs):
    return dict(ZERO)


def drift_policy(obs):
    """정책이 팔로워를 서서히 몰고 간다 — 개입 시점의 어긋남을 만들기 위해."""
    return {**obs, "joint1.pos": obs["joint1.pos"] + 1.0}


# ── 기본 루프 ────────────────────────────────────────────────
def test_no_hil():
    """HIL 없으면 개입 플래그가 전부 False이고, 끝에 park가 강제된다."""
    r = MockRobot()
    log = run_attempt(r, const_policy, r.parking, fps=1000, max_steps=10)
    assert len(log) == 10
    assert not any(s["intervention"] for s in log)
    assert r.parked == 1, "시도 종료 시 park를 강제해야 한다"


def test_abort_stops_attempt():
    """q(abort)를 누르면 max_steps 전에 시도가 끝나되 park는 여전히 강제된다."""
    r = MockRobot()
    t = MockToggle([False])
    t.abort = True
    log = run_attempt(r, const_policy, r.parking, fps=1000, max_steps=100,
                      toggle=t, leader=MockLeader([0]), clutch=Clutch())
    assert len(log) == 1, f"첫 스텝에서 멈춰야 한다: {len(log)}"
    assert r.parked == 1


# ── 클러치(델타 인계) ────────────────────────────────────────
def test_clutch_no_jump_on_engage():
    """★ PiperLeader는 팔로워를 추종할 수 없어 인계 시 반드시 어긋나 있다.
    델타 방식이면 어긋남이 아무리 커도 첫 목표 = 팔로워 현재 자세여야 한다."""
    c = Clutch()
    follower = {**ZERO, "joint1.pos": 10.0}
    leader = {**ZERO, "joint1.pos": 70.0}   # 60이나 벌어져 있음
    assert deviation(leader, follower) == 60.0

    c.engage(leader, follower)
    first = c.target(leader)
    assert first["joint1.pos"] == 10.0, f"인계 첫 목표는 팔로워 현재 자세여야 한다: {first}"


def test_clutch_tracks_leader_delta():
    """개입 중에는 리더의 변화량만큼만 팔로워가 움직인다."""
    c = Clutch()
    c.engage({**ZERO, "joint1.pos": 70.0}, {**ZERO, "joint1.pos": 10.0})
    for lead, expect in [(75.0, 15.0), (80.0, 20.0), (65.0, 5.0)]:
        got = c.target({**ZERO, "joint1.pos": lead})["joint1.pos"]
        assert abs(got - expect) < 1e-9, f"lead={lead} → {got}, 기대 {expect}"


def test_clutch_gain():
    """gain<1이면 리더 움직임이 축소돼 반영된다(정밀 보정용)."""
    c = Clutch(gain=0.5)
    c.engage({**ZERO, "joint1.pos": 0.0}, {**ZERO, "joint1.pos": 0.0})
    assert c.target({**ZERO, "joint1.pos": 10.0})["joint1.pos"] == 5.0


def test_clutch_reengage_uses_new_reference():
    """개입을 껐다 켜면 기준점이 다시 잡혀야 한다 — 안 그러면 두 번째 인계에서 튄다."""
    c = Clutch()
    c.engage({**ZERO, "joint1.pos": 0.0}, {**ZERO, "joint1.pos": 0.0})
    c.release()
    c.engage({**ZERO, "joint1.pos": 50.0}, {**ZERO, "joint1.pos": 30.0})
    assert c.target({**ZERO, "joint1.pos": 50.0})["joint1.pos"] == 30.0


# ── 루프 통합 ────────────────────────────────────────────────
def test_intervention_logged_and_no_jump_in_loop():
    """루프 안에서: 개입 구간이 정확히 기록되고, 인계 스텝에서 팔로워가 튀지 않는다."""
    r = MockRobot()
    # 정책이 팔로워를 스텝당 +1로 몰고 간다. 리더는 계속 70에 있다(사람이 안 잡고 있음).
    sched = [False, False, True, True, True, False, False]
    log = run_attempt(r, drift_policy, r.parking, fps=1000, max_steps=7,
                      toggle=MockToggle(sched), leader=MockLeader([70] * 7),
                      clutch=Clutch())

    flags = [s["intervention"] for s in log]
    assert flags == sched, flags

    # 인계 스텝(index 2)에서 명령이 직전 팔로워 자세에서 튀지 않아야 한다.
    j = [s["action"]["joint1.pos"] for s in log]
    assert abs(j[2] - j[1]) < 1e-6, f"인계 순간 점프 발생: {j[1]} → {j[2]}"

    # 인계 시점의 어긋남이 기록으로 남아야 한다(나중에 자동 감지 설계 근거)
    assert "engage_deviation" in log[2], log[2]
    assert log[2]["engage_deviation"] > 60, log[2]


def test_release_returns_control_to_policy():
    """개입을 끄면 다시 정책이 몬다."""
    r = MockRobot()
    sched = [True, True, False, False]
    log = run_attempt(r, drift_policy, r.parking, fps=1000, max_steps=4,
                      toggle=MockToggle(sched), leader=MockLeader([0, 0, 0, 0]),
                      clutch=Clutch())
    assert [s["intervention"] for s in log] == sched
    # 정책 복귀 후에는 다시 +1씩 증가
    j = [s["action"]["joint1.pos"] for s in log]
    assert j[3] > j[2], f"정책 반환 후 정책이 움직여야 한다: {j}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ok  {name}")
    print("전부 통과")
