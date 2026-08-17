#!/usr/bin/env python3
"""hil_clutch.ClutchMixer + InferenceRunner HIL 배선 검증 — 하드웨어 없이.

클러치 자체(점프 0, 델타 추종, gain, 재인계)는 test_erase_run_mock.py가 이미
덮는다. 여기서는 통합으로 **새로 생긴 표면**만 본다:
  - ClutchMixer의 상태 전이(engage/release를 한 곳에 모은 것)
  - InferenceRunner가 HIL 설정을 받아 넘기는 배선
  - 실물 전송이 닫혀 있으면 개입이 열리지 않는다는 안전 조건

실행: python -m pytest scripts/tools/test_hil_clutch_mock.py -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import piper_infer_runner as runner  # noqa: E402
from hil_clutch import ACTION_NAMES, Clutch, ClutchMixer  # noqa: E402

ZERO = {k: 0.0 for k in ACTION_NAMES}


class FakeToggle:
    def __init__(self, active=False):
        self.active = active
        self.abort = False


class FakeLeader:
    """리더 자세를 밖에서 직접 세팅한다."""

    def __init__(self, pose=None):
        self.pose = dict(pose or ZERO)
        self.reads = 0

    def get_action(self):
        self.reads += 1
        return dict(self.pose)


def make_mixer(active=False, leader_pose=None, gain=1.0):
    toggle = FakeToggle(active)
    leader = FakeLeader(leader_pose)
    return ClutchMixer(toggle, leader, Clutch(gain)), toggle, leader


# ── ClutchMixer 상태 전이 ────────────────────────────────────
def test_inactive_passes_policy_action_through():
    mixer, _, leader = make_mixer(active=False)
    policy = dict(ZERO, **{"joint1.pos": 5.0})
    out, intervened, dev = mixer.step(policy, ZERO)
    assert out == policy and intervened is False and dev is None
    assert leader.reads == 0, "개입이 꺼져 있으면 리더를 읽지도 않아야 한다"


def test_engage_reports_deviation_once_not_every_step():
    # 팔로워는 0, 리더는 60 어긋난 상태에서 인계
    mixer, toggle, _ = make_mixer(active=True, leader_pose=dict(ZERO, **{"joint1.pos": 60.0}))
    _, _, dev_first = mixer.step(dict(ZERO), dict(ZERO))
    assert dev_first == 60.0, "인계 시점 편차가 기록돼야 한다"
    _, _, dev_second = mixer.step(dict(ZERO), dict(ZERO))
    assert dev_second is None, "이미 인계 중이면 새 인계로 보고하지 않는다"


def test_engage_has_no_jump_even_when_far_apart():
    """★ 핵심: 60만큼 어긋나 있어도 인계 첫 스텝 목표 = 팔로워 현재 자세."""
    follower = dict(ZERO, **{"joint1.pos": 10.0})
    mixer, _, _ = make_mixer(active=True, leader_pose=dict(ZERO, **{"joint1.pos": 70.0}))
    out, intervened, _ = mixer.step(dict(ZERO), follower)
    assert intervened is True
    assert out["joint1.pos"] == 10.0, "점프가 있으면 안 된다"


def test_release_then_reengage_uses_new_reference():
    follower = dict(ZERO, **{"joint1.pos": 10.0})
    mixer, toggle, leader = make_mixer(active=True, leader_pose=dict(ZERO, **{"joint1.pos": 0.0}))
    mixer.step(dict(ZERO), follower)

    # 사람이 리더를 20 움직임 -> 팔로워도 20 따라감
    leader.pose["joint1.pos"] = 20.0
    out, _, _ = mixer.step(dict(ZERO), follower)
    assert out["joint1.pos"] == 30.0

    # 반환
    toggle.active = False
    policy = dict(ZERO, **{"joint1.pos": 99.0})
    out, intervened, _ = mixer.step(policy, dict(ZERO, **{"joint1.pos": 30.0}))
    assert intervened is False and out == policy

    # 다시 인계 — 기준점이 새로 잡혀 또 점프가 없어야 한다
    toggle.active = True
    new_follower = dict(ZERO, **{"joint1.pos": 45.0})
    out, _, dev = mixer.step(dict(ZERO), new_follower)
    assert out["joint1.pos"] == 45.0
    assert dev == 25.0, "재인계 편차 = |리더20 - 팔로워45|"


def test_intervened_steps_are_counted():
    mixer, toggle, _ = make_mixer(active=True)
    for _ in range(3):
        mixer.step(dict(ZERO), dict(ZERO))
    toggle.active = False
    mixer.step(dict(ZERO), dict(ZERO))
    assert mixer.intervened_steps == 3


def test_gain_scales_leader_delta():
    follower = dict(ZERO, **{"joint1.pos": 10.0})
    mixer, _, leader = make_mixer(active=True, leader_pose=dict(ZERO), gain=0.5)
    mixer.step(dict(ZERO), follower)
    leader.pose["joint1.pos"] = 40.0
    out, _, _ = mixer.step(dict(ZERO), follower)
    assert out["joint1.pos"] == 10.0 + 0.5 * 40.0


def test_missing_leader_or_toggle_is_passthrough():
    mixer = ClutchMixer(None, None, Clutch())
    policy = dict(ZERO, **{"joint1.pos": 7.0})
    out, intervened, dev = mixer.step(policy, dict(ZERO))
    assert out == policy and intervened is False and dev is None


# ── InferenceRunner 배선 ─────────────────────────────────────
def test_hil_is_off_by_default():
    settings = runner.RunSettings(dataset_root="d", policy_path="p")
    assert settings.hil is False, "기본 꺼짐이어야 기존 동작과 같다"
    assert settings.clutch_gain == 1.0


def test_cli_flags_map_to_settings():
    args = runner.parse_args(
        [
            "--dataset-root", "d", "--policy-path", "p",
            "--hil", "--leader-port", "can_leader", "--clutch-gain", "0.4",
        ]
    )
    settings = runner.settings_from_args(args)
    assert settings.hil is True
    assert settings.leader_port == "can_leader"
    assert settings.clutch_gain == 0.4


def test_hil_requires_real_robot_transmission():
    """실물 전송이 안 열렸으면 개입도 열리면 안 된다 (CAN만 점유하고 하는 일이 없음)."""
    settings = runner.RunSettings(
        dataset_root="d", policy_path="p", hil=True, source="dataset"
    )
    assert settings.real_robot_enabled() is False

    opened = runner.RunSettings(
        dataset_root="d", policy_path="p", hil=True,
        source="robot", apply_to_robot=True,
        real_robot_confirm=runner.REAL_ROBOT_CONFIRM,
    )
    assert opened.real_robot_enabled() is True


def test_runner_starts_with_zero_intervention_counters():
    run = runner.InferenceRunner(runner.RunSettings(dataset_root="d", policy_path="p"))
    assert run.intervention_steps == 0
    assert run.engage_deviations == []
    assert run.hil_aborted is False


# ── 개입 중 빈 큐 회귀 테스트 ──────────────────────────────
# 실물 2차 HIL에서 개입 첫 스텝에 러너가 status=error로 죽었다. 원인은
# "개입 중에는 추론을 기다리지 않는다"는 최적화를 넣으면서, 그 앞의
# pipeline.next_action()이 빈 큐로 호출되는 걸 안 막은 것.


def test_empty_pipeline_during_intervention_must_not_raise():
    """개입 중 큐가 비면 next_action()을 부르면 안 된다.

    러너 루프 전체를 띄우지 않고 그 분기 조건만 재현한다 — 실제 코드가
    'pending==0 그리고 개입 중'이면 next_action을 건너뛰는지가 핵심이다.
    """
    import numpy as np
    from action_smoothing import SmoothingConfig, SmoothingPipeline

    pipeline = SmoothingPipeline(SmoothingConfig())
    pipeline.reset(np.zeros(7, np.float32))
    assert pipeline.pending_steps == 0

    # 고치기 전 동작: 빈 큐에서 next_action()은 예외를 던진다 (이게 원인이었다)
    try:
        pipeline.next_action()
        raised = False
    except RuntimeError:
        raised = True
    assert raised, "빈 큐에서 next_action()이 조용히 통과하면 이 테스트의 전제가 깨진다"

    # 고친 동작: 개입 중이면 현재 자세를 자리표시자로 쓴다
    measured = np.array([1, 2, 3, 4, 5, 6, 7], np.float32)
    intervening_now = True
    if pipeline.pending_steps == 0 and intervening_now:
        action = measured.astype(np.float32).copy()
    else:
        action = pipeline.next_action()
    assert np.array_equal(action, measured)


def test_runner_source_guards_next_action_when_intervening():
    """러너 소스에 실제로 그 가드가 들어있는지 — 리팩터링으로 사라지면 잡는다."""
    import pathlib as _p

    src = (_p.Path(__file__).parent / "piper_infer_runner.py").read_text()
    assert "if pipeline.pending_steps == 0 and intervening_now:" in src
    assert "if pipeline.pending_steps == 0 and not intervening_now:" in src
