"""smooth_start_frames의 parking 보정 대상 판정 테스트 (데이터셋/하드웨어 불필요).

USE_EFFORT=true로 녹화하면 observation.state가 pos(7)+effort(7)+vel(6) = 20차원이 된다.
초반 프레임 보정은 *자세(.pos)* 만 parking에서 시작하도록 덮어써야 하고, effort/vel은
측정값이라 손대면 안 된다. 그 계약을 고정한다.

실행: PYTHONPATH=. python scripts/tools/test_smooth_start_mock.py
"""

import importlib.util
import pathlib

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "smooth_start_frames", pathlib.Path(__file__).with_name("smooth_start_frames.py")
)
ssf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ssf)

MOTORS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
# tables.INITIALIZE_POSITION과 같은 모양의 값 (joint2/joint3가 0이 아닌 게 핵심 —
# 예전 버그에서 effort 칸에 -100/100이 써지던 그 값들)
PARKING = {"joint1": 0.0, "joint2": -100.0, "joint3": 100.0, "joint4": 0.0,
           "joint5": 0.0, "joint6": -13.04, "gripper": 0.0}

POS_ONLY = [f"{m}.pos" for m in MOTORS]
WITH_EFFORT = (
    POS_ONLY
    + [f"{m}.effort" for m in MOTORS]
    + [f"{m}.vel" for m in MOTORS if m != "gripper"]
)


def test_pos_only_dataset_masks_everything():
    vec, mask = ssf._parking_vector(POS_ONLY, PARKING)
    assert mask.all()  # USE_EFFORT=false면 기존 동작과 완전히 동일해야 함
    assert np.allclose(vec, [PARKING[m] for m in MOTORS])  # float32 반올림 허용


def test_effort_and_vel_excluded_from_parking_vector():
    vec, mask = ssf._parking_vector(WITH_EFFORT, PARKING)
    assert mask[:7].all()  # pos만 보정 대상
    assert not mask[7:].any()  # effort/vel은 제외
    assert np.allclose(vec[:7], [PARKING[m] for m in MOTORS])


def test_unknown_motor_still_raises():
    # "조용히 틀린 parking으로 덮어쓰지 않는다"는 기존 방침이 유지되는지.
    try:
        ssf._parking_vector(["joint9.pos"], PARKING)
    except KeyError:
        return
    raise AssertionError("모르는 관절 이름인데 KeyError가 안 났음")


def test_interpolate_leaves_effort_columns_untouched():
    vec, mask = ssf._parking_vector(WITH_EFFORT, PARKING)
    L, dim = 10, len(WITH_EFFORT)
    values = np.arange(L * dim, dtype=np.float32).reshape(L, dim)

    out = ssf._interpolate_start(values, vec, num_frames=4, mask=mask)

    # effort/vel 컬럼은 한 프레임도 안 바뀜 (예전엔 여기가 -100/100으로 덮여 있었다)
    assert np.array_equal(out[:, 7:], values[:, 7:])
    # pos 컬럼은 0번 프레임이 정확히 parking, 4번(타깃)은 원본 유지
    assert np.allclose(out[0, :7], [PARKING[m] for m in MOTORS], atol=1e-4)
    assert np.array_equal(out[4:, :7], values[4:, :7])
    # 중간 프레임은 실제로 덮어써짐. joint1은 parking=0 + 값이 선형 arange라 보간값이
    # 원본과 우연히 같아질 수 있어서, parking이 0이 아닌 joint2(col 1)로 확인한다.
    assert out[1, 1] != values[1, 1]
    assert np.isclose(out[1, 1], PARKING["joint2"] * 0.75 + values[4, 1] * 0.25)


def test_interpolate_without_mask_is_unchanged_behaviour():
    vec, _ = ssf._parking_vector(POS_ONLY, PARKING)
    values = np.arange(10 * 7, dtype=np.float32).reshape(10, 7)
    a = ssf._interpolate_start(values, vec, num_frames=4)
    b = ssf._interpolate_start(values, vec, num_frames=4, mask=np.ones(7, dtype=bool))
    assert np.array_equal(a, b)


if __name__ == "__main__":
    test_pos_only_dataset_masks_everything()
    test_effort_and_vel_excluded_from_parking_vector()
    test_unknown_motor_still_raises()
    test_interpolate_leaves_effort_columns_untouched()
    test_interpolate_without_mask_is_unchanged_behaviour()
    print("OK: smooth start parking 마스크 테스트 통과 (effort/vel 보존)")
