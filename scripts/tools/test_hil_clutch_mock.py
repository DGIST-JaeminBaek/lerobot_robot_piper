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


def test_intervention_steps_do_not_consume_the_policy_budget():
    """개입 스텝은 max_steps에서 빠져야 한다.

    실물 3차 HIL에서 26초 지점에 개입했더니 5.3초 만에 940스텝 상한에 걸려
    끊겼다. 사람이 조작한 시간이 정책 예산을 깎으면 '개입할수록 정책에게
    남는 시간이 줄어드는' 이상한 구조가 된다.
    """
    import pathlib as _p

    src = (_p.Path(__file__).parent / "piper_infer_runner.py").read_text()
    assert "policy_steps = step - self.intervention_steps" in src
    assert "if max_steps and policy_steps >= max_steps:" in src

    # 계산 자체 확인: 전체 1100스텝 중 개입 200이면 정책은 900만 쓴 것
    step, interventions, max_steps = 1100, 200, 940
    assert (step - interventions) < max_steps      # 아직 안 끝난다
    step = 1141
    assert (step - interventions) >= max_steps     # 정책 940스텝을 채우면 끝


# ── 그리퍼 절대값 통과 ────────────────────────────────────
# 실물 HIL에서 "리더 그리퍼를 끝까지 닫아도 팔로워가 안 닫혀 물건을 못 잡는다"가
# 나왔다. 델타 규칙을 그리퍼에 걸면 끝까지 닫는 게 원리적으로 불가능하다.


def test_gripper_passes_leader_absolute_value():
    from hil_clutch import Clutch, GRIPPER_KEY

    c = Clutch()
    # 인계 시점: 팔로워 그리퍼 30, 리더 60
    lead0 = {"joint1.pos": 0.0, GRIPPER_KEY: 60.0}
    foll0 = {"joint1.pos": 10.0, GRIPPER_KEY: 30.0}
    c.engage(lead0, foll0)

    # 리더를 끝까지(100) 닫는다
    out = c.target({"joint1.pos": 0.0, GRIPPER_KEY: 100.0})
    # 델타였다면 30 + (100-60) = 70에서 멈춘다. 절대값이면 100에 도달한다.
    assert out[GRIPPER_KEY] == 100.0, out
    # 관절은 여전히 델타 — 인계 순간 점프 0이 유지돼야 한다
    assert out["joint1.pos"] == 10.0, out


def test_gripper_absolute_does_not_jump_the_joints():
    """그리퍼만 절대값이고 나머지는 그대로 클러치여야 한다."""
    from hil_clutch import Clutch, GRIPPER_KEY

    c = Clutch()
    lead = {"joint1.pos": -50.0, GRIPPER_KEY: 0.0}
    foll = {"joint1.pos": 40.0, GRIPPER_KEY: 80.0}   # 관절이 90 어긋난 상태
    c.engage(lead, foll)
    out = c.target(lead)                              # 리더가 안 움직였으면
    assert out["joint1.pos"] == 40.0                  # 관절 목표 = 팔로워 현재 (점프 0)
    assert out[GRIPPER_KEY] == 0.0                    # 그리퍼는 리더 값 그대로


def test_gripper_absolute_can_be_turned_off():
    from hil_clutch import Clutch, GRIPPER_KEY

    c = Clutch(absolute_keys=())
    c.engage({GRIPPER_KEY: 60.0}, {GRIPPER_KEY: 30.0})
    assert c.target({GRIPPER_KEY: 100.0})[GRIPPER_KEY] == 70.0   # 예전 델타 동작


def test_recording_is_tied_to_the_intervention_toggle():
    """space로 개입하면 녹화가 켜지고 반환하면 에피소드가 저장돼야 한다.

    손이 리더암에 묶여 있어 녹화 버튼을 따로 누르기 어렵다. HIL 데이터 수집의
    목적이 정확히 '개입 구간'이라 토글 경계가 그대로 녹화 경계가 된다.
    """
    import pathlib as _p

    src = (_p.Path(__file__).parent / "piper_infer_runner.py").read_text()
    assert "if settings.record_manual and settings.record_on_intervention:" in src
    assert "if intervened and not self.record_armed:" in src
    assert "self.arm_recording()" in src
    assert "elif not intervened and self.record_armed:" in src
    assert "self.stop_recording()" in src


def test_record_arm_stop_are_idempotent():
    """중복 호출로 에피소드가 두 번 저장되거나 카운터가 어긋나면 안 된다."""
    import threading

    class Runner:
        def __init__(self):
            self.record_armed = False
            self.recorded_episodes = 0
            self._record_lock = threading.Lock()
            self._record_stop_requested = False
            self.logs = []

        def _log(self, m):
            self.logs.append(m)

        arm_recording = None  # 아래에서 실제 구현을 빌려온다

    import piper_infer_runner as R

    r = Runner()
    R.InferenceRunner.arm_recording(r)
    R.InferenceRunner.arm_recording(r)          # 두 번 켜도
    assert r.record_armed is True
    R.InferenceRunner.stop_recording(r)
    assert r.record_armed is False and r._record_stop_requested is True
    r._record_stop_requested = False
    R.InferenceRunner.stop_recording(r)         # 이미 꺼져 있으면 아무 일 없어야 한다
    assert r._record_stop_requested is False


def test_recorder_is_asynchronous_and_drains_by_completion():
    """녹화가 제어 루프를 막으면 안 되고, drain은 '쓰기 완료' 기준이어야 한다.

    실측: 동기 방식은 실제 카메라 프레임(512x512, 카메라 2대)에서 25.3ms/프레임으로
    30Hz 예산(33.3ms)의 76%를 먹었다. raw 1280x720이면 80.5ms(241%)라 실물에서
    제어 주기가 19.53Hz로 떨어졌다.

    drain을 queue.empty()로 구현하면 안 된다 — 마지막 항목을 '꺼낸' 순간 empty가
    True가 되는데 아직 쓰는 중이라, 프레임 하나가 빠진 채 에피소드가 닫히고
    parquet 길이 불일치로 죽는다(실측: 300개 중 299개).
    """
    import pathlib as _p

    src = (_p.Path(__file__).parent / "piper_infer_runner.py").read_text()
    # 쓰기는 별도 스레드
    assert "rollout-writer" in src
    assert "def _writer_loop" in src
    # 제어 루프는 큐에 넣기만 한다
    assert "self._queue.put_nowait(frame)" in src
    # drain은 완료 카운터 기준 (empty() 폴링이 아니라)
    assert "self._idle.wait_for(lambda: self._inflight == 0" in src
    assert "while not self._queue.empty()" not in src
    # save_episode 전에 반드시 drain
    assert "self.drain()          # 쓰기 스레드가 밀린 프레임을 다 반영할 때까지" in src
    # 큐가 차면 버리되 세어서 보고 (블로킹하면 제어 루프가 멈춘다)
    assert "self._dropped += 1" in src
