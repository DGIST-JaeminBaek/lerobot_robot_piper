"""replay_safe의 정렬 로직 점검 (하드웨어 없이). python test_replay_safe.py"""

import sys
import types

import numpy as np

# 하드웨어 의존 import를 모킹 — 정렬 로직만 검사한다
for name, attr in (("lerobot_robot_piper.piper_follower", "PiperFollower"),
                   ("lerobot_robot_piper.config_piper", "PiperFollowerConfig")):
    mod = types.ModuleType(name)
    setattr(mod, attr, object)
    sys.modules[name] = mod

import replay_safe as R  # noqa: E402

NAMES = [f"j{i}" for i in range(7)]


class FakeArm:
    """send_action의 클램프(max_relative_target=5.0)와 팔의 실제 속도 한계를 흉내낸다.
    속도 한계가 클램프보다 빡빡한 게 핵심 — 리플레이가 기어가던 이유가 이것."""

    def __init__(self, speed=2.0):
        self.pos = np.zeros(7)
        self.speed = speed

    def get_observation(self):
        return {n: v for n, v in zip(NAMES, self.pos)}

    def send_action(self, action):
        goal = np.array([action[n] for n in NAMES])
        goal = self.pos + np.clip(goal - self.pos, -5.0, 5.0)
        self.pos = self.pos + np.clip(goal - self.pos, -self.speed, self.speed)


class Args:
    align_step, align_tol, align_timeout, fps = 1.0, 1.0, 30.0, 1000


def test_first_unclipped():
    """파킹 자세(±100에 잘린 값) 구간을 건너뛰는지."""
    park = [[-100.0, 100, -35, 0, 0, 0, 0]] * 12
    real = [[-98.0, 97, -35, 0, 0, 0, 0]] * 50
    A = np.array(park + real)
    assert R.first_unclipped(A) == 12
    assert R.first_unclipped(A[12:]) == 0          # 이미 실측 구간이면 0
    assert R.first_unclipped(np.array(park)) == 12  # 전 구간 클리핑이면 길이 그대로


def test_preflight():
    """preflight는 팔을 움직이지 않고 판정만 한다. 통과/불통과가 갈리는지."""
    t = np.linspace(0, 3, 400)[:, None]
    act = np.hstack([30 * np.sin(t)] + [np.full_like(t, v) for v in (-40, 60, -20, 10, 0, 5)])

    class A:
        abort_err, preflight_gap = 8.0, 60.0

    R.preflight(act, act[0].copy(), A)                    # 정렬된 상태 → 통과

    far = act[0].copy()
    far[0] += 200
    try:                                                   # 너무 멀면 사람이 먼저 확인
        R.preflight(act, far, A)
        raise AssertionError("먼 시작 자세를 통과시켰다")
    except SystemExit as e:
        assert e.code == 1

    clipped = np.vstack([np.full(7, 100.0), act])          # 잘린 목표면 불통과
    try:
        R.preflight(clipped, clipped[0].copy(), A)
        raise AssertionError("클리핑된 시작 목표를 통과시켰다")
    except SystemExit as e:
        assert e.code == 1


def main():
    target = np.array([40.0, -30, 20, 10, -5, 0, 50])

    arm = FakeArm()
    assert R.align(arm, target, NAMES, Args), "정렬 실패"
    assert np.abs(target - arm.pos).max() <= Args.align_tol, arm.pos

    # 타임아웃은 반드시 False를 돌려줘야 한다 (실패를 성공으로 넘기면 팔이 엉뚱한 데서 재생 시작)
    class NoTime(Args):
        align_timeout = 0.0

    assert not R.align(FakeArm(), target, NAMES, NoTime), "타임아웃이 성공으로 처리됨"

    # 이미 도달한 상태면 명령 없이 즉시 True
    done = FakeArm()
    done.pos = target.copy()
    assert R.align(done, target, NAMES, Args)

    test_first_unclipped()
    test_preflight()

    print("ok")


if __name__ == "__main__":
    main()
