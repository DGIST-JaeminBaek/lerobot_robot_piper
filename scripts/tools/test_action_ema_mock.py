#!/usr/bin/env python3
"""PiperFollower.send_action 단계 EMA를 하드웨어 없이 검증한다.

이 EMA는 "바닥"이다 — 어떤 경로로 명령이 들어오든(텔레옵, lerobot-record,
lerobot-replay, 정책 runner) send_action()을 지나가므로 항상 적용된다. 추론의
주된 스무딩인 temporal ensemble은 chunk 단위라 여기서는 불가능하고
piper_infer_runner가 처리한다.

지키려는 성질:
  - 기본값(alpha=1.0)에서는 아무것도 바뀌지 않는다 (기존 동작 보존)
  - 첫 목표는 그대로 통과한다 (시작할 때 끌려가지 않음)
  - 안전 클램프(max_relative_target)보다 먼저 걸린다

실행: python -m pytest scripts/tools/test_action_ema_mock.py
"""

from __future__ import annotations

import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lerobot_robot_piper.piper_follower import PiperFollower  # noqa: E402


class FakeConfig:
    def __init__(self, alpha: float) -> None:
        self.action_ema_alpha = alpha


class FakeFollower:
    """PiperFollower.__init__은 실제 버스를 열므로 EMA 메서드만 빌려 쓴다."""

    def __init__(self, alpha: float) -> None:
        self.config = FakeConfig(alpha)
        self._action_ema_state = None

    _apply_action_ema = PiperFollower._apply_action_ema


GOAL = {"joint1": 10.0, "joint2": -4.0}


def test_alpha_one_is_a_passthrough():
    follower = FakeFollower(1.0)
    assert follower._apply_action_ema(GOAL) == GOAL
    assert follower._apply_action_ema({"joint1": 99.0, "joint2": 0.0}) == {
        "joint1": 99.0,
        "joint2": 0.0,
    }


def test_alpha_one_keeps_state_cleared():
    """꺼져 있을 때 상태가 쌓이면, 나중에 켰을 때 옛 값으로 튄다."""
    follower = FakeFollower(1.0)
    follower._apply_action_ema(GOAL)
    assert follower._action_ema_state is None


def test_first_goal_passes_through_unchanged():
    follower = FakeFollower(0.5)
    assert follower._apply_action_ema(GOAL) == GOAL


def test_second_goal_is_blended():
    follower = FakeFollower(0.5)
    follower._apply_action_ema({"joint1": 0.0})
    smoothed = follower._apply_action_ema({"joint1": 10.0})
    assert smoothed["joint1"] == pytest.approx(5.0)


def test_repeated_goal_converges():
    follower = FakeFollower(0.5)
    follower._apply_action_ema({"joint1": 0.0})
    values = [follower._apply_action_ema({"joint1": 10.0})["joint1"] for _ in range(5)]
    assert values == pytest.approx([5.0, 7.5, 8.75, 9.375, 9.6875])
    assert values[-1] < 10.0  # EMA는 목표에 점근할 뿐 넘어서지 않는다


def test_smaller_alpha_is_smoother():
    slow, fast = FakeFollower(0.1), FakeFollower(0.9)
    for follower in (slow, fast):
        follower._apply_action_ema({"joint1": 0.0})
    slow_step = slow._apply_action_ema({"joint1": 10.0})["joint1"]
    fast_step = fast._apply_action_ema({"joint1": 10.0})["joint1"]
    assert slow_step < fast_step


def test_unseen_motor_uses_its_own_goal_as_baseline():
    follower = FakeFollower(0.5)
    follower._apply_action_ema({"joint1": 0.0})
    smoothed = follower._apply_action_ema({"joint1": 10.0, "gripper": 40.0})
    assert smoothed["gripper"] == pytest.approx(40.0)


@pytest.mark.parametrize("alpha", [0.0, -0.5])
def test_non_positive_alpha_rejected(alpha):
    follower = FakeFollower(alpha)
    with pytest.raises(ValueError, match="action_ema_alpha"):
        follower._apply_action_ema(GOAL)


def test_default_config_has_ema_off():
    from lerobot_robot_piper.config_piper import PiperFollowerConfig

    assert PiperFollowerConfig.action_ema_alpha == 1.0


def test_ema_runs_before_the_safety_clamp():
    """순서가 뒤집히면 스무딩이 max_relative_target 제한을 넘길 수 있다."""
    import inspect

    source = inspect.getsource(PiperFollower.send_action)
    # 주석에도 max_relative_target이 나오므로 실제 클램프 문장을 기준으로 본다.
    ema_call = source.index("goal_pos = self._apply_action_ema(goal_pos)")
    clamp = source.index("if self.config.max_relative_target is not None:")
    assert ema_call < clamp


def test_confirm_string_matches_runner():
    """teleop_ui가 runner의 확인 문구를 하드코딩하고 있어 어긋나면 안 된다."""
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "tools"))
    from piper_infer_runner import REAL_ROBOT_CONFIRM

    from lerobot_robot_piper.teleop_ui import INFER_REAL_ROBOT_CONFIRM

    assert INFER_REAL_ROBOT_CONFIRM == REAL_ROBOT_CONFIRM


# ── Dataset Browser 스캔 범위 ─────────────────────────────
def test_scan_root_is_records_regardless_of_dataset_root():
    """DATASET_ROOT가 records/의 하위 폴더를 가리켜도 목록이 거기 갇히면 안 된다.

    예전 구현은 DATASET_ROOT의 부모를 스캔 기준으로 써서,
    DATASET_ROOT=records/local/... 이면 records/outputs/의 학습용 데이터셋이
    Dataset Browser에 전혀 안 떴다.
    """
    from lerobot_robot_piper.teleop_ui import REPO_ROOT, dataset_scan_root

    env = {"DATASET_ROOT": str(REPO_ROOT / "records" / "local" / "piper_pick_pen_sample")}
    assert dataset_scan_root(env) == REPO_ROOT / "records"
    assert dataset_scan_root({}) == REPO_ROOT / "records"


def test_scan_root_can_be_narrowed_explicitly():
    from lerobot_robot_piper.teleop_ui import REPO_ROOT, dataset_scan_root

    assert dataset_scan_root({"DATASET_SCAN_ROOT": "records/0727"}) == REPO_ROOT / "records/0727"
    assert dataset_scan_root({"DATASET_SCAN_ROOT": "/tmp/ds"}) == pathlib.Path("/tmp/ds")


# ── Policy Path 목록 ──────────────────────────────────────
def _make_run(root, name, steps, last=None):
    for step in steps:
        (root / name / "checkpoints" / step / "pretrained_model").mkdir(parents=True)
    if last:
        (root / name / "checkpoints" / "last").symlink_to(last)


def test_discover_policies_finds_pretrained_dirs(tmp_path):
    from lerobot_robot_piper.teleop_ui import discover_policies

    _make_run(tmp_path, "runA", ["005000", "010000"], last="010000")
    found = discover_policies(tmp_path)
    assert [label for label, _ in found] == ["runA / 010000 (last)", "runA / 005000"]
    assert found[0][1].endswith("runA/checkpoints/010000/pretrained_model")


def test_discover_policies_does_not_duplicate_the_last_symlink(tmp_path):
    """`last`를 별도 항목으로 만들면 같은 체크포인트가 두 번 뜬다."""
    from lerobot_robot_piper.teleop_ui import discover_policies

    _make_run(tmp_path, "runA", ["030000"], last="030000")
    found = discover_policies(tmp_path)
    assert len(found) == 1
    assert "(last)" in found[0][0]


def test_discover_policies_skips_dirs_without_pretrained_model(tmp_path):
    from lerobot_robot_piper.teleop_ui import discover_policies

    (tmp_path / "runA" / "checkpoints" / "005000").mkdir(parents=True)
    (tmp_path / "notarun").mkdir()
    assert discover_policies(tmp_path) == []


def test_discover_policies_on_missing_root(tmp_path):
    from lerobot_robot_piper.teleop_ui import discover_policies

    assert discover_policies(tmp_path / "nope") == []


# ── MOVE 모드 ─────────────────────────────────────────────
# MOVE J는 점대점 명령이라 목표마다 컨트롤러가 궤적을 새로 계획한다. 30Hz로
# 목표를 흘려보내면 초당 30번 재계획이 일어나 팔이 떤다. MOVE CPV가 스트리밍용.
def test_default_move_mode_is_unchanged():
    """텔레옵/녹화/파킹은 MOVE J로 잘 동작한다 — 기본값을 바꾸지 않는다."""
    from lerobot_robot_piper.config_piper import PiperFollowerConfig
    from lerobot_robot_piper.motors.piper_motors_bus import MOVE_J

    assert PiperFollowerConfig.move_mode == MOVE_J
    assert PiperFollowerConfig.move_speed_rate == 30


def test_set_action_uses_the_configured_move_mode():
    import inspect

    from lerobot_robot_piper.motors.piper_motors_bus import PiperMotorsBus

    source = inspect.getsource(PiperMotorsBus.set_action)
    assert "ModeCtrl(0x01, self.move_mode, self.move_speed_rate, 0x00)" in source


def test_gripper_cycle_stays_point_to_point():
    """그리퍼 단독 이동은 목표 하나를 주고 도착을 기다리는 방식이라 MOVE J가 맞다."""
    import inspect

    from lerobot_robot_piper.motors.piper_motors_bus import PiperMotorsBus

    source = inspect.getsource(PiperMotorsBus.cycle_gripper)
    assert "ModeCtrl(0x01, MOVE_J, self.move_speed_rate, 0x00)" in source


def test_runner_exposes_move_mode():
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "tools"))
    import piper_infer_runner as runner

    base = ["--dataset-root", "d", "--policy-path", "p"]
    assert runner.settings_from_args(runner.parse_args(base)).move_mode is None
    assert runner.settings_from_args(
        runner.parse_args(base + ["--move-mode", "5"])
    ).move_mode == 5


# ── 세 가지 "제한"은 서로 다른 것 ─────────────────────────
# rate_limit / max_relative_target은 명령값의 상한(천장)이라 올려도 팔이
# 빨라지지 않는다. 팔의 실제 속도는 ModeCtrl의 move_speed_rate가 정한다.
def test_speed_rate_is_independent_of_the_two_command_ceilings():
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "tools"))
    import piper_infer_runner as runner

    settings = runner.settings_from_args(
        runner.parse_args(
            ["--dataset-root", "d", "--policy-path", "p",
             "--move-speed-rate", "60", "--rate-limit", "2", "--max-relative-target", "15"]
        )
    )
    assert settings.move_speed_rate == 60
    assert settings.smoothing.rate_limit == pytest.approx(2.0)
    assert settings.max_relative_target == pytest.approx(15.0)


def test_speed_rate_defaults_to_config_when_unset():
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "tools"))
    import piper_infer_runner as runner
    from lerobot_robot_piper.config_piper import PiperFollowerConfig

    args = runner.parse_args(["--dataset-root", "d", "--policy-path", "p"])
    assert runner.settings_from_args(args).move_speed_rate is None
    assert PiperFollowerConfig.move_speed_rate == 30


# ── MIT 제어 (버스 계층) ──────────────────────────────────
def _mit_bus():
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot_robot_piper.motors.piper_motors_bus import JOINT_NAMES, PiperMotorsBus

    bus = PiperMotorsBus.__new__(PiperMotorsBus)
    bus.apply_drive_mode = False
    bus.move_speed_rate = 30
    bus.calibration = {m: MotorCalibration(i, 0, 0, -150000, 150000)
                       for i, m in enumerate(JOINT_NAMES, 1)}
    bus.calibration["gripper"] = MotorCalibration(7, 0, 0, 0, 68000)
    bus.motors = {m: Motor(i, "AGILEX-M", MotorNormMode.RANGE_M100_100)
                  for i, m in enumerate(JOINT_NAMES, 1)}
    bus.motors["gripper"] = Motor(7, "AGILEX-S", MotorNormMode.RANGE_0_100)

    calls = []

    class Fake:
        def ModeCtrl(self, *a): calls.append(("ModeCtrl",) + a)
        def JointMitCtrl(self, *a): calls.append(("JointMitCtrl",) + a)
        def GripperCtrl(self, *a): calls.append(("GripperCtrl",) + a)
    bus.piper = Fake()
    bus.get_control = lambda: {}
    return bus, calls


def test_mit_sends_impedance_mode_then_one_command_per_joint():
    from lerobot_robot_piper.motors.piper_motors_bus import JOINT_NAMES, MOVE_M

    bus, calls = _mit_bus()
    bus.set_action_mit({m: 0.0 for m in JOINT_NAMES}, {m: 0.0 for m in JOINT_NAMES})
    assert calls[0] == ("ModeCtrl", 0x01, MOVE_M, 30, 0xAD)  # is_mit_mode=0xAD
    assert sum(1 for c in calls if c[0] == "JointMitCtrl") == 6


def test_mit_passes_velocity_through_as_vel_ref():
    """속도를 안 넘기면 chunk가 담고 있는 의도된 속도를 버리게 된다."""
    from lerobot_robot_piper.motors.piper_motors_bus import JOINT_NAMES

    bus, calls = _mit_bus()
    bus.set_action_mit({JOINT_NAMES[0]: 0.0}, {JOINT_NAMES[0]: 100.0}, kp=10.0, kd=0.8)
    cmd = next(c for c in calls if c[0] == "JointMitCtrl")
    _, motor_num, pos_ref, vel_ref, kp, kd, t_ref = cmd
    assert motor_num == 1
    assert vel_ref == pytest.approx(2.618, abs=1e-3)  # 100단위/s -> rad/s
    assert (kp, kd, t_ref) == (10.0, 0.8, 0.0)


def test_mit_rejects_out_of_range_gains():
    from lerobot_robot_piper.motors.piper_motors_bus import JOINT_NAMES

    bus, _ = _mit_bus()
    goal = {JOINT_NAMES[0]: 0.0}
    for kp in (-1.0, 501.0):
        with pytest.raises(ValueError, match="kp"):
            bus.set_action_mit(goal, kp=kp)
    for kd in (-6.0, 6.0):
        with pytest.raises(ValueError, match="kd"):
            bus.set_action_mit(goal, kd=kd)


def test_leave_mit_returns_to_position_control():
    """MIT 상태로 parking()을 부르면 위치 명령이 안 먹는다 — 종료 전 필수."""
    from lerobot_robot_piper.motors.piper_motors_bus import MOVE_J

    bus, calls = _mit_bus()
    bus.leave_mit_mode()
    assert calls == [("ModeCtrl", 0x01, MOVE_J, 30, 0x00)]


def test_mit_is_off_by_default_in_config():
    from lerobot_robot_piper.config_piper import PiperFollowerConfig

    assert PiperFollowerConfig.use_mit_control is False
    assert PiperFollowerConfig.mit_kp == 10.0 and PiperFollowerConfig.mit_kd == 0.8


def test_mit_confirm_matches_between_teleop_ui_and_runner():
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "tools"))
    from piper_infer_runner import MIT_CONFIRM

    from lerobot_robot_piper.teleop_ui import INFER_MIT_CONFIRM

    assert INFER_MIT_CONFIRM == MIT_CONFIRM


# ── 클램프와 속도 피드포워드의 일치 ───────────────────────
# 위치는 자르고 속도는 안 자르면 kd 항이 큰 토크를 만든다. 클램프가 걸린
# 관절은 vel_ref를 0으로 내려야 한다.
def test_clamped_joint_drops_its_velocity_feedforward():
    import inspect

    source = inspect.getsource(PiperFollower.send_action)
    assert "self._pending_goal_velocity[key] = 0.0" in source
    # 클램프 계산 뒤, set_action_mit 호출 전에 있어야 한다
    assert source.index("ensure_safe_goal_position") < source.index(
        "self._pending_goal_velocity[key] = 0.0"
    )
    assert source.index("self._pending_goal_velocity[key] = 0.0") < source.index(
        "set_action_mit"
    )


def test_velocity_is_only_used_by_mit():
    """위치 제어(MOVE J)는 속도 setpoint를 받지 않는다 — 넘겨도 무시돼야 한다."""
    import inspect

    source = inspect.getsource(PiperFollower.send_action)
    mit_branch = source.index('getattr(self.config, "use_mit_control", False)')
    assert source.index("set_action_mit") > mit_branch
    assert "self.bus.set_action(goal_pos, is_conv=True)" in source


# ── 관절별 MIT 게인 ───────────────────────────────────────
# MIT의 정상상태 오차 = 중력토크/kp 이므로, 중력 부담이 다른 관절에 같은 kp를
# 주면 처짐이 제각각이 된다. 실측(kp=10, 뻗은 자세): joint1/4/6 처짐 0.00,
# joint5 0.17, joint2 1.83, joint3 0.72.
def _gains_from(spec, base=10.0):
    from types import SimpleNamespace

    from lerobot_robot_piper.piper_follower import PiperFollower

    follower = PiperFollower.__new__(PiperFollower)
    follower.bus = SimpleNamespace(
        motors={f"joint{i}": None for i in range(1, 7)} | {"gripper": None}
    )
    follower.config = SimpleNamespace(mit_kp=base, mit_kp_overrides=spec)
    return follower._mit_gains("mit_kp", "mit_kp_overrides", base)


def test_no_override_returns_a_single_shared_gain():
    assert _gains_from("") == 10.0


def test_overrides_apply_per_joint_and_exclude_the_gripper():
    gains = _gains_from("joint2=30,joint3=20")
    assert gains["joint2"] == 30.0 and gains["joint3"] == 20.0
    assert gains["joint1"] == 10.0 and gains["joint6"] == 10.0
    assert "gripper" not in gains  # MIT 대상이 아니다


def test_bad_entries_are_ignored_not_fatal():
    """오타 하나로 실행이 죽으면 실물 앞에서 곤란하다 — 무시하고 기본값을 쓴다."""
    gains = _gains_from("nope=5,joint2=bad,joint4=15")
    assert gains["joint4"] == 15.0
    assert gains["joint2"] == 10.0  # 파싱 실패 → 기본값


def test_default_overrides_target_the_heavy_joints():
    from lerobot_robot_piper.config_piper import PiperFollowerConfig

    gains = _gains_from(PiperFollowerConfig.mit_kp_overrides)
    assert gains["joint2"] > gains["joint1"]
    assert gains["joint3"] > gains["joint1"]


def test_bus_accepts_per_joint_gains():
    from lerobot_robot_piper.motors.piper_motors_bus import JOINT_NAMES

    bus, calls = _mit_bus()
    bus.set_action_mit(
        {m: 0.0 for m in JOINT_NAMES}, kp={"joint2": 30.0}, kd=0.8
    )
    commands = [c for c in calls if c[0] == "JointMitCtrl"]
    assert commands[0][4] == 10.0  # joint1 — 기본값
    assert commands[1][4] == 30.0  # joint2 — 덮어쓴 값


def test_per_joint_gain_range_is_still_checked():
    from lerobot_robot_piper.motors.piper_motors_bus import JOINT_NAMES

    bus, _ = _mit_bus()
    with pytest.raises(ValueError, match="joint2 kp"):
        bus.set_action_mit({m: 0.0 for m in JOINT_NAMES}, kp={"joint2": 999.0})
