"""fix_action_offset 핵심 로직 자체 점검: python test_fix_action_offset.py"""

import numpy as np

from fix_action_offset import leading_trim, measure_lead, rebuild_action

FPS, WARMUP, LEAD = 30, 45, 2


def make_episode():
    """8/4 데이터 모사: 앞 WARMUP 프레임은 정지 + action==state,
    이후 실제 궤적. action은 상수 offset만큼 어긋나 있다."""
    t = np.linspace(0, 3, 240)[:, None]
    traj = np.hstack([10 * np.sin(t), 5 * t])
    state = np.vstack([np.repeat(traj[:1], WARMUP, axis=0), traj])

    offset = np.array([7.5, -3.0])  # warmup 때 잠긴 엉뚱한 offset
    action = np.roll(state, -LEAD, axis=0) - offset
    action[-LEAD:] = action[-LEAD - 1]
    action[:WARMUP] = state[:WARMUP]  # warmup 구간은 state 복사본
    return state, action


def main():
    state, action = make_episode()

    n = leading_trim(state, action, still_thresh=0.3)
    # warmup 전체 + 궤적이 임계를 넘기까지의 몇 프레임까지 잘린다
    assert WARMUP <= n <= WARMUP + 5, f"trim {n} not near {WARMUP}"

    assert measure_lead(state[WARMUP:], action[WARMUP:]) == LEAD  # 트리밍 이후 구간에서 추정

    s = state[n:]
    abs_a = rebuild_action(s, LEAD, "absolute")
    # 재구성된 절대 action은 팔이 실제 도달한 자세 — offset 오염 없음
    assert np.allclose(abs_a[:-LEAD], s[LEAD:]), "absolute action mismatch"
    assert abs_a.shape == s.shape

    delta_a = rebuild_action(s, LEAD, "delta")
    assert np.allclose(delta_a, abs_a - s), "delta != absolute - state"
    # 상수 offset이 걸린 원본에서도 delta는 동일 — 좌표계 어긋남이 소거된다
    assert np.abs(delta_a[0]).max() > 0, "delta가 전부 0이면 궤적이 죽은 것"

    # 무동작 에피소드는 전부 잘려야 한다 (짧아져서 호출부에서 드롭됨)
    flat = np.zeros((100, 2))
    assert leading_trim(flat, flat.copy(), 0.3) == len(flat)

    print("ok")


if __name__ == "__main__":
    main()
