#!/usr/bin/env python3
"""piper_infer_runner의 모드 프리셋 / 기록 로직을 하드웨어 없이 검증한다.

정책·로봇·LeRobotDataset은 전부 mock으로 대체한다. 여기서 지키려는 성질:
  - 모드는 프리셋일 뿐이고 개별 override가 항상 이긴다 (값을 숨기지 않는다)
  - 기록되는 action은 raw가 아니라 스무딩 후 값이다
  - 실물 전송은 세 조건이 모두 갖춰졌을 때만 열린다
  - discard를 고르면 아무것도 저장되지 않는다

실행: python -m pytest scripts/tools/test_infer_runner_mock.py
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import sys

import numpy as np
import pytest

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import piper_infer_runner as runner  # noqa: E402
from action_smoothing import SmoothingConfig  # noqa: E402


# ── 모드 프리셋 ───────────────────────────────────────────
def test_demo_mode_does_not_record():
    settings = runner.RunSettings.from_mode("demo", dataset_root="d", policy_path="p")
    assert settings.record_dataset is False
    assert settings.prompt_outcome is False


def test_augment_mode_records_raw_frames_and_prompts():
    settings = runner.RunSettings.from_mode("augment", dataset_root="d", policy_path="p")
    assert settings.record_dataset is True
    assert settings.record_raw_frames is True
    assert settings.prompt_outcome is True


def test_both_modes_share_the_measured_best_smoothing():
    # docs/policy/smoothing.md의 실측 최적값. 모드가 이걸 바꾸지는 않는다.
    for name in ("demo", "augment"):
        settings = runner.RunSettings.from_mode(name, dataset_root="d", policy_path="p")
        assert settings.smoothing.ensemble_m == pytest.approx(0.01)
        assert settings.smoothing.temporal_ensemble is True


def test_override_beats_mode_preset():
    """모드는 프리셋일 뿐 — 개별 지정이 항상 이겨야 한다."""
    settings = runner.RunSettings.from_mode(
        "augment", dataset_root="d", policy_path="p", record_dataset=False
    )
    assert settings.record_dataset is False


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="알 수 없는 모드"):
        runner.mode_preset("nope")


def test_cli_smoothing_override_applies_on_top_of_mode():
    args = runner.parse_args(
        ["--dataset-root", "d", "--policy-path", "p", "--mode", "augment", "--ensemble-m", "0.3"]
    )
    settings = runner.settings_from_args(args)
    assert settings.record_dataset is True  # 모드에서 옴
    assert settings.smoothing.ensemble_m == pytest.approx(0.3)  # override


def test_cli_no_record_overrides_augment_mode():
    args = runner.parse_args(
        ["--dataset-root", "d", "--policy-path", "p", "--mode", "augment", "--no-record"]
    )
    assert runner.settings_from_args(args).record_dataset is False


def test_cli_no_ensemble_disables_it():
    args = runner.parse_args(
        ["--dataset-root", "d", "--policy-path", "p", "--no-ensemble"]
    )
    assert runner.settings_from_args(args).smoothing.temporal_ensemble is False


# ── live 카메라 crop ──────────────────────────────────────
# source=robot에서 crop이 비면 preprocess_live_camera_observation이
# KeyError("No live crop configured for dataset camera 'top'")로 죽는다.
# GUI는 입력칸에서 채우지만 CLI에는 그 값이 없어서 teleop_ui의 Infer 프리셋이
# 실물 실행에서 항상 이 에러를 냈다.
def test_robot_source_gets_crops_from_recording_env():
    args = runner.parse_args(["--dataset-root", "d", "--policy-path", "p", "--source", "robot"])
    settings = runner.settings_from_args(args)
    assert set(settings.crops) == {"top", "wrist"}
    assert settings.camera_output_size == 512


def test_crop_flags_override_env():
    args = runner.parse_args(
        ["--dataset-root", "d", "--policy-path", "p", "--source", "robot",
         "--top-crop", "10,20,300"]
    )
    crops = runner.settings_from_args(args).crops
    assert (crops["top"].x, crops["top"].y, crops["top"].size) == (10, 20, 300)
    assert crops["wrist"].size == 720  # env 값 유지


def test_robot_source_without_any_crop_is_rejected(tmp_path):
    """조용히 빈 crop으로 출발해 로봇 연결까지 한 뒤 죽지 않게 미리 막는다."""
    args = runner.parse_args(
        ["--dataset-root", "d", "--policy-path", "p", "--source", "robot",
         "--env-file", str(tmp_path / "missing.env")]
    )
    with pytest.raises(SystemExit, match="crop"):
        runner.settings_from_args(args)


def test_dataset_source_does_not_need_crops(tmp_path):
    args = runner.parse_args(
        ["--dataset-root", "d", "--policy-path", "p",
         "--env-file", str(tmp_path / "missing.env")]
    )
    assert runner.settings_from_args(args).crops == {}


def test_load_env_file_ignores_comments_and_quotes(tmp_path):
    path = tmp_path / "recording.env"
    path.write_text('# comment\nA=1\nB="two"\n\nbroken\n', encoding="utf-8")
    assert runner.load_env_file(path) == {"A": "1", "B": "two"}


# ── 클램프 포화 진단 ──────────────────────────────────────
# 실물 30Hz 실행에서 스텝 200부터 100% 클램프에 걸렸다. 그 상태에서는 명령이
# "실측 위치 + max_relative_target"으로 대체돼 스무딩 결과가 버려지므로,
# smoothing 하이퍼파라미터를 만져도 아무 효과가 없다 — 그걸 알려주는 로직.
def _drain(run):
    messages = []
    while not run.events.empty():
        kind, payload = run.events.get()
        if kind == runner.Event.LOG:
            messages.append(payload)
    return messages


def _run_clamped(fraction: float, steps: int):
    run = runner.InferenceRunner(runner.RunSettings(dataset_root="d", policy_path="p"))
    requested = np.zeros(7, np.float32)
    for i in range(steps):
        offset = 1.0 if i < fraction * steps else 0.0
        sent = {f"{name}.pos": float(offset) for name in runner.MOTOR_NAMES}
        run._track_send(requested, sent)
    return _drain(run)


def test_no_report_when_clamping_is_rare():
    assert _drain(
        runner.InferenceRunner(runner.RunSettings(dataset_root="d", policy_path="p"))
    ) == []
    assert _run_clamped(0.1, runner.InferenceRunner.CLAMP_REPORT_EVERY) == []


def test_saturation_says_smoothing_params_wont_help():
    messages = _run_clamped(1.0, runner.InferenceRunner.CLAMP_REPORT_EVERY)
    assert len(messages) == 1
    assert "포화" in messages[0]
    assert "smoothing 파라미터는 효과가 없습니다" in messages[0]


def test_saturation_advice_is_not_repeated_every_window():
    """매 60스텝마다 같은 설명이 도배되면 다른 로그를 덮는다."""
    messages = _run_clamped(1.0, runner.InferenceRunner.CLAMP_REPORT_EVERY * 3)
    assert len(messages) == 3
    assert sum("smoothing 파라미터는 효과가 없습니다" in m for m in messages) == 1
    assert all("포화" in m for m in messages)


def test_partial_clamping_reports_without_saturation_advice():
    messages = _run_clamped(0.5, runner.InferenceRunner.CLAMP_REPORT_EVERY)
    assert len(messages) == 1
    assert "포화" not in messages[0]
    assert "50%" in messages[0]


def test_track_send_ignores_empty_result():
    run = runner.InferenceRunner(runner.RunSettings(dataset_root="d", policy_path="p"))
    run._track_send(np.zeros(7, np.float32), None)
    run._track_send(np.zeros(7, np.float32), {})
    assert run._clamp_window == []


# ── max_relative_target 분리 ──────────────────────────────
def test_clamp_is_separate_from_smoothing_rate_limit():
    """둘을 같은 값으로 묶으면 스무딩을 조일 때 로봇 클램프까지 조여진다."""
    args = runner.parse_args(
        ["--dataset-root", "d", "--policy-path", "p",
         "--max-relative-target", "20", "--rate-limit", "2"]
    )
    settings = runner.settings_from_args(args)
    assert settings.max_relative_target == pytest.approx(20.0)
    assert settings.smoothing.rate_limit == pytest.approx(2.0)


# ── 실물 전송 게이트 ──────────────────────────────────────
@pytest.mark.parametrize(
    "source,apply,confirm,expected",
    [
        ("dataset", True, runner.REAL_ROBOT_CONFIRM, False),
        ("robot", False, runner.REAL_ROBOT_CONFIRM, False),
        ("robot", True, "", False),
        ("robot", True, "wrong", False),
        ("robot", True, runner.REAL_ROBOT_CONFIRM, True),
    ],
)
def test_real_robot_needs_all_three_conditions(source, apply, confirm, expected):
    settings = runner.RunSettings(
        dataset_root="d",
        policy_path="p",
        source=source,
        apply_to_robot=apply,
        real_robot_confirm=confirm,
    )
    assert settings.real_robot_enabled() is expected


def test_cli_refuses_apply_to_robot_without_confirm(capsys):
    code = runner.main(
        ["--dataset-root", "d", "--policy-path", "p", "--source", "robot", "--apply-to-robot"]
    )
    assert code == 2
    assert runner.REAL_ROBOT_CONFIRM in capsys.readouterr().err


# ── feature 정의 ──────────────────────────────────────────
def test_rollout_features_match_record_shape():
    features = runner.build_rollout_features(
        camera_shapes={"top": (720, 1280, 3), "wrist": (480, 640, 3)},
        state_names=runner.MOTOR_NAMES,
        action_names=runner.MOTOR_NAMES,
    )
    assert features["observation.state"]["shape"] == (7,)
    assert features["action"]["dtype"] == "float32"
    assert features["observation.images.top"]["dtype"] == "video"
    assert features["observation.images.top"]["shape"] == (720, 1280, 3)
    assert features["observation.images.wrist"]["shape"] == (480, 640, 3)


def test_rollout_features_have_no_extra_keys():
    """학습 호환성 — raw chunk 같은 건 feature에 섞지 않고 sidecar로 뺀다."""
    features = runner.build_rollout_features(
        camera_shapes={"top": (512, 512, 3)},
        state_names=runner.MOTOR_NAMES,
        action_names=runner.MOTOR_NAMES,
    )
    assert set(features) == {"observation.state", "action", "observation.images.top"}


# ── 기록 ──────────────────────────────────────────────────
class FakeLeRobotDataset:
    """LeRobotDataset.create가 돌려주는 것 대신 쓰는 mock."""

    created: list[dict] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.frames: list[dict] = []
        self.saved = 0
        self.episode_buffer = None
        FakeLeRobotDataset.created.append(kwargs)

    def add_frame(self, frame):
        self.frames.append(frame)
        self.episode_buffer = {"size": len(self.frames)}

    def save_episode(self):
        self.saved += 1

    def clear_episode_buffer(self):
        self.frames.clear()
        self.episode_buffer = None

    class meta:  # noqa: N801 — LeRobotDataset.meta 흉내
        video_keys = ["observation.images.top"]

    def _get_image_file_dir(self, episode_index, camera_key):
        return self.kwargs["root"] / "images" / camera_key / f"episode-{episode_index:06d}"


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    import lerobot.datasets.lerobot_dataset as lds

    monkeypatch.setattr(lds.LeRobotDataset, "create", staticmethod(FakeLeRobotDataset))
    FakeLeRobotDataset.created.clear()
    return runner.RolloutRecorder(
        root=tmp_path / "rollout",
        repo_id="local/rollout",
        fps=6,
        features=runner.build_rollout_features(
            camera_shapes={"top": (8, 8, 3)},
            state_names=runner.MOTOR_NAMES,
            action_names=runner.MOTOR_NAMES,
        ),
        task="erase the shape",
    )


def test_recorder_writes_task_and_action_on_every_frame(recorder):
    recorder.add_frame(
        state=np.ones(7, np.float32),
        action=np.full(7, 2.0, np.float32),
        images={"top": np.zeros((8, 8, 3), np.uint8)},
    )
    frame = recorder.dataset.frames[0]
    assert frame["task"] == "erase the shape"
    assert frame["observation.images.top"].shape == (8, 8, 3)
    np.testing.assert_allclose(frame["action"], 2.0)
    assert frame["action"].dtype == np.float32


def test_recorder_save_is_noop_without_frames(recorder):
    recorder.save_episode()
    assert recorder.dataset.saved == 0


def test_recorder_discard_drops_frames(recorder):
    recorder.add_frame(
        state=np.ones(7, np.float32),
        action=np.ones(7, np.float32),
        images={"top": np.zeros((8, 8, 3), np.uint8)},
    )
    recorder.discard_episode()
    assert recorder.frames_written == 0
    assert recorder.dataset.saved == 0


def test_discard_removes_leftover_video_frames(recorder):
    """lerobot의 clear_episode_buffer는 image_keys만 지운다 — video dtype인 우리
    카메라의 임시 PNG는 직접 치워야 폐기할 때마다 쌓이지 않는다."""
    frame_dir = recorder.root / "images" / "observation.images.top" / "episode-000000"
    frame_dir.mkdir(parents=True)
    (frame_dir / "frame-000000.png").write_bytes(b"x")

    recorder.add_frame(
        state=np.zeros(7, np.float32),
        action=np.zeros(7, np.float32),
        images={"top": np.zeros((8, 8, 3), np.uint8)},
    )
    recorder.discard_episode()
    assert not frame_dir.exists()


def test_sidecar_accumulates_across_episodes(recorder):
    recorder.write_sidecar({"episode_index": 0, "outcome": "success"})
    path = recorder.write_sidecar({"episode_index": 1, "outcome": "failure"})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert [entry["outcome"] for entry in payload] == ["success", "failure"]


def test_sidecar_records_full_smoothing_condition(recorder):
    """논문에 조건을 명시해야 하므로 스무딩 값이 전부 남아야 한다."""
    config = SmoothingConfig(temporal_ensemble=True, ensemble_m=0.01, ema_alpha=0.5, rate_limit=5.0)
    path = recorder.write_sidecar({"smoothing": dataclasses.asdict(config), "measured_fps": 6.42})
    entry = json.loads(path.read_text(encoding="utf-8"))[0]
    assert entry["smoothing"]["ensemble_m"] == pytest.approx(0.01)
    assert entry["smoothing"]["ema_alpha"] == pytest.approx(0.5)
    assert entry["smoothing"]["rate_limit"] == pytest.approx(5.0)
    assert entry["measured_fps"] == pytest.approx(6.42)


def test_raw_actions_saved_separately(recorder):
    raw = np.arange(21, dtype=np.float32).reshape(3, 7)
    path = recorder.write_raw_actions(raw, episode_index=2)
    assert path.name == "raw_actions_ep0002.npz"
    np.testing.assert_allclose(np.load(path)["raw_first_actions"], raw)


# ── 실측 제어 주기 ────────────────────────────────────────
def test_measured_fps_is_none_until_two_steps():
    run = runner.InferenceRunner(runner.RunSettings(dataset_root="d", policy_path="p"))
    assert run.measured_fps() == 0.0
    run.step_periods.append(0.1)
    assert run.measured_fps() == 0.0


def test_measured_fps_averages_step_periods():
    run = runner.InferenceRunner(runner.RunSettings(dataset_root="d", policy_path="p"))
    run.step_periods.extend([0.1, 0.2])
    assert run.measured_fps() == pytest.approx(1.0 / 0.15)


# ── 기록되는 action이 raw가 아니라 스무딩 후 값인지 ────────
def test_finalize_records_smoothed_not_raw(recorder, monkeypatch):
    settings = runner.RunSettings.from_mode(
        "augment", dataset_root="d", policy_path="p", prompt_outcome=False
    )
    run = runner.InferenceRunner(settings)
    # raw와 smoothed가 다른 상황을 만든다
    run.raw_trajectory = [np.full(7, 9.0, np.float32)]
    run.trajectory = [np.full(7, 1.0, np.float32)]
    recorder.add_frame(
        state=np.zeros(7, np.float32),
        action=run.trajectory[0],
        images={"top": np.zeros((8, 8, 3), np.uint8)},
    )
    run._finalize_recording(recorder, "finished")

    # dataset에 들어간 건 smoothed
    np.testing.assert_allclose(recorder.dataset.frames[0]["action"], 1.0)
    # raw는 sidecar 옆 npz로만
    raw_path = recorder.root / "raw_actions_ep0000.npz"
    np.testing.assert_allclose(np.load(raw_path)["raw_first_actions"], 9.0)
    assert recorder.dataset.saved == 1


def test_finalize_discard_saves_nothing(recorder):
    settings = runner.RunSettings.from_mode(
        "augment", dataset_root="d", policy_path="p"
    )
    run = runner.InferenceRunner(settings, outcome_prompt=lambda: ("discard", ""))
    recorder.add_frame(
        state=np.zeros(7, np.float32),
        action=np.zeros(7, np.float32),
        images={"top": np.zeros((8, 8, 3), np.uint8)},
    )
    run._finalize_recording(recorder, "finished")
    assert recorder.dataset.saved == 0
    assert not (recorder.root / "rollout_meta.json").exists()


def test_finalize_labels_outcome_from_prompt(recorder):
    settings = runner.RunSettings.from_mode("augment", dataset_root="d", policy_path="p")
    run = runner.InferenceRunner(settings, outcome_prompt=lambda: ("failure", "그리퍼 놓침"))
    run.trajectory = [np.zeros(7, np.float32)]
    recorder.add_frame(
        state=np.zeros(7, np.float32),
        action=np.zeros(7, np.float32),
        images={"top": np.zeros((8, 8, 3), np.uint8)},
    )
    run._finalize_recording(recorder, "finished")
    entry = json.loads((recorder.root / "rollout_meta.json").read_text(encoding="utf-8"))[0]
    assert entry["outcome"] == "failure"
    assert entry["note"] == "그리퍼 놓침"
    assert entry["camera_frames"] == "raw"


# ── 정책 ↔ dataset 호환 검사 ──────────────────────────────
# Dataset Browser에 200개가 다 보이게 만든 뒤로 다른 dataset을 고르기 쉬워졌다.
# 안 맞으면 정책 로딩(수십 초)과 로봇 연결이 끝난 뒤에야 텐서 크기 불일치로
# 죽으므로, config.json만 읽어서 미리 막는다.
def _make_policy(root, state_dim=7, dataset_root="/data/train_ds"):
    policy = root / "checkpoints" / "last" / "pretrained_model"
    policy.mkdir(parents=True)
    (policy / "config.json").write_text(
        json.dumps(
            {
                "type": "smolvla",
                "input_features": {
                    "observation.state": {"type": "STATE", "shape": [state_dim]},
                    "observation.images.top": {"type": "VISUAL", "shape": [3, 512, 512]},
                },
                "output_features": {"action": {"type": "ACTION", "shape": [7]}},
            }
        ),
        encoding="utf-8",
    )
    (policy / "train_config.json").write_text(
        json.dumps({"dataset": {"repo_id": "local/train_ds", "root": dataset_root}}),
        encoding="utf-8",
    )
    return policy


def _make_dataset(root, state_dim=7):
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "features": {
                    "observation.state": {"shape": [state_dim]},
                    "action": {"shape": [7]},
                    "observation.images.top": {"shape": [512, 512, 3]},
                },
            }
        ),
        encoding="utf-8",
    )
    return root


def test_matching_dataset_passes(tmp_path):
    policy = _make_policy(tmp_path / "run")
    dataset = _make_dataset(tmp_path / "ds")
    runner.check_policy_dataset_match(policy, dataset)  # 예외 없음


def test_state_dim_mismatch_is_rejected_with_the_training_dataset(tmp_path):
    policy = _make_policy(tmp_path / "run", dataset_root="/data/right_one")
    dataset = _make_dataset(tmp_path / "ds", state_dim=20)
    with pytest.raises(ValueError) as error:
        runner.check_policy_dataset_match(policy, dataset)
    message = str(error.value)
    assert "observation.state" in message
    assert "(7,)" in message and "(20,)" in message
    assert "/data/right_one" in message  # 어느 걸 골라야 하는지 알려준다


def test_missing_feature_is_reported(tmp_path):
    policy = _make_policy(tmp_path / "run")
    dataset = _make_dataset(tmp_path / "ds")
    info = dataset / "meta" / "info.json"
    payload = json.loads(info.read_text(encoding="utf-8"))
    del payload["features"]["action"]
    info.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="dataset에 없음"):
        runner.check_policy_dataset_match(policy, dataset)


def test_image_shape_layout_difference_is_tolerated(tmp_path):
    """정책 config는 CHW, dataset meta는 HWC로 적혀 있어 그대로 비교하면 오탐."""
    policy = _make_policy(tmp_path / "run")
    dataset = _make_dataset(tmp_path / "ds")
    runner.check_policy_dataset_match(policy, dataset)


def test_unverifiable_policy_passes_through(tmp_path):
    """HF repo id처럼 로컬에 config.json이 없으면 막지 않는다."""
    dataset = _make_dataset(tmp_path / "ds")
    runner.check_policy_dataset_match("lerobot/smolvla_base", dataset)


def test_training_dataset_of_reads_train_config(tmp_path):
    policy = _make_policy(tmp_path / "run", dataset_root="/data/xyz")
    assert runner.training_dataset_of(policy) == "/data/xyz"
    assert runner.training_dataset_of(tmp_path / "nope") is None


# ── 추론 워커 ─────────────────────────────────────────────
# 추론(~115ms)을 명령 루프(33ms) 안에서 돌리면 "33ms 4번 → 115ms 1번"이 반복돼
# 6Hz 주기의 규칙적 끊김이 생기고 팔이 그 리듬으로 진동한다. 워커로 빼서
# 명령 루프가 일정한 주기를 지키게 한다.
import threading  # noqa: E402
import time as _time  # noqa: E402


def _new_runner():
    return runner.InferenceRunner(runner.RunSettings(dataset_root="d", policy_path="p"))


def test_worker_returns_chunk_without_blocking_the_caller():
    run = _new_runner()
    run._start_inference_worker(lambda obs: np.full((5, 7), obs["v"], np.float32))

    started = _time.perf_counter()
    run._request_inference({"v": 3.0})
    assert _time.perf_counter() - started < 0.05  # 요청은 즉시 반환

    chunk = run._await_chunk(timeout=5.0)
    assert chunk is not None and chunk[0][0] == pytest.approx(3.0)
    run.stop_event.set()


def test_stale_request_is_skipped_while_worker_is_busy():
    """밀린 상태에서 요청을 쌓으면 오래된 관찰로 추론하게 된다.

    워커가 바쁜 동안 들어온 요청은 버려야 하고, 큐에 쌓여 있다가 나중에
    처리되면 안 된다.
    """
    release = threading.Event()
    seen: list[float] = []
    run = _new_runner()

    def predict(observation):
        seen.append(observation["v"])
        release.wait(5)
        return np.zeros((5, 7), np.float32)

    run._start_inference_worker(predict)

    run._request_inference({"v": 1.0})
    while not seen:  # 워커가 첫 요청을 집을 때까지 기다린다
        _time.sleep(0.005)
    run._request_inference({"v": 2.0})  # 아직 바쁘므로 버려져야 함
    run._request_inference({"v": 3.0})

    release.set()
    assert run._await_chunk(timeout=5.0) is not None
    _time.sleep(0.1)
    assert seen == [1.0]  # 버려진 관찰이 뒤늦게 처리되지 않는다
    run.stop_event.set()


def test_collect_chunks_drains_without_blocking():
    run = _new_runner()
    assert run._collect_chunks() == []
    run._infer_results.put(np.zeros((5, 7), np.float32))
    run._infer_results.put(np.ones((5, 7), np.float32))
    assert len(run._collect_chunks()) == 2
    assert run._collect_chunks() == []


def test_await_chunk_times_out_instead_of_hanging():
    run = _new_runner()
    assert run._await_chunk(timeout=0.05) is None


def test_worker_error_is_surfaced_to_the_loop():
    run = _new_runner()

    def boom(_obs):
        raise ValueError("정책이 NaN/Inf를 출력했습니다")

    run._start_inference_worker(boom)
    run._request_inference({"v": 0.0})
    for _ in range(100):
        if run._infer_error:
            break
        _time.sleep(0.01)
    assert run._infer_error and "NaN/Inf" in run._infer_error
    assert not run._infer_busy.is_set()  # 걸려 있으면 이후 요청이 전부 막힌다
    run.stop_event.set()


# ── 주기 대기 ─────────────────────────────────────────────
# 예전 구현은 while 판정과 sleep 인자 계산 사이에 시각이 지나가면 음수를 넘겨
# ValueError를 던졌다. 그러면 제어 루프가 죽고 팔이 park로 내려간다 — 루프가
# 빨라져 반복 횟수가 늘어난 뒤 실물에서 실제로 터졌다.
def test_sleep_until_never_passes_a_negative_duration():
    run = _new_runner()
    slept: list[float] = []
    # 매 호출마다 시각이 크게 튀어 deadline을 넘겨버리는 최악의 경우
    ticks = iter([0.0, 0.004, 0.0099, 0.02, 0.05, 0.09])

    def clock():
        return next(ticks, 99.0)

    def sleep(duration):
        assert duration >= 0, f"음수 sleep: {duration}"
        slept.append(duration)

    run._sleep_until(0.01, clock=clock, sleep=sleep)
    assert all(d >= 0 for d in slept)


def test_sleep_until_returns_immediately_when_deadline_has_passed():
    run = _new_runner()
    calls: list[float] = []
    run._sleep_until(-1.0, clock=lambda: 0.0, sleep=calls.append)
    assert calls == []


def test_sleep_until_slices_the_wait():
    run = _new_runner()
    now = [0.0]
    slept: list[float] = []

    def sleep(duration):
        slept.append(duration)
        now[0] += duration

    run._sleep_until(0.02, clock=lambda: now[0], sleep=sleep)
    assert slept == pytest.approx([0.005, 0.005, 0.005, 0.005])


def test_sleep_until_stops_on_stop_event():
    run = _new_runner()
    run.stop_event.set()
    calls: list[float] = []
    run._sleep_until(99.0, clock=lambda: 0.0, sleep=calls.append)
    assert calls == []


# ── A. 룩어헤드 / C. MIT ──────────────────────────────────
# MOVE J는 목표마다 궤적을 재계획한다. 33ms 앞의 목표는 거리가 너무 짧아 가속
# 초입만 밟다 교체되므로, 목표를 진행 방향으로 앞당겨 보낸다(pure pursuit).
# MIT는 재계획 자체가 없는 인터페이스라 근본 해결책이지만 토크 제어다.
def test_lookahead_defaults_to_off():
    args = runner.parse_args(["--dataset-root", "d", "--policy-path", "p"])
    assert runner.settings_from_args(args).lookahead_s == 0.0


def test_lookahead_extrapolates_along_the_velocity():
    """룩어헤드 목표 = action + velocity * lookahead_s (루프와 같은 수식)."""
    action = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0], np.float32)
    velocity = np.array([30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], np.float32)
    commanded = np.clip(action + velocity * 0.15, runner.GLOBAL_LOW, runner.GLOBAL_HIGH)
    assert commanded[0] == pytest.approx(14.5)
    assert commanded[6] == pytest.approx(50.0)  # 속도 0이면 그대로


def test_lookahead_stays_inside_the_normalized_range():
    """외삽이 범위를 넘으면 안 된다 — 넘긴 값이 로봇 목표가 된다."""
    action = np.full(7, 95.0, np.float32)
    velocity = np.full(7, 500.0, np.float32)
    commanded = np.clip(action + velocity * 0.2, runner.GLOBAL_LOW, runner.GLOBAL_HIGH)
    assert commanded.max() <= 100.0
    assert commanded[6] <= 100.0


def test_mit_needs_its_own_confirmation(capsys):
    """실물 확인 문구와 별개다 — 토크 제어는 안전장치의 의미가 달라진다."""
    code = runner.main(["--dataset-root", "d", "--policy-path", "p", "--mit"])
    assert code == 2
    assert runner.MIT_CONFIRM in capsys.readouterr().err


def test_mit_confirmed_settings_carry_the_gains():
    args = runner.parse_args(
        ["--dataset-root", "d", "--policy-path", "p", "--mit",
         "--mit-confirm", runner.MIT_CONFIRM, "--mit-kp", "20", "--mit-kd", "1.2"]
    )
    settings = runner.settings_from_args(args)
    assert settings.use_mit is True
    assert settings.mit_kp == pytest.approx(20.0)
    assert settings.mit_kd == pytest.approx(1.2)


def test_mit_is_off_by_default():
    args = runner.parse_args(["--dataset-root", "d", "--policy-path", "p"])
    assert runner.settings_from_args(args).use_mit is False


# ── vel_ref 다듬기 ────────────────────────────────────────
# 30Hz 궤적을 (a - prev) * fps 로 그냥 미분하면 지터가 30배로 증폭된다.
# 실측: 스텝 간 변화가 속도 크기의 72%, 부호 반전 28.7% — 속도가 아니라 노이즈다.
# 그걸 kd*(vel_ref - vel)에 넣으면 초당 10번 방향이 바뀌는 토크가 나간다.
def _ema_velocity(raw, alpha):
    """runner 루프와 같은 수식."""
    out, acc = [], None
    for value in raw:
        acc = value.copy() if acc is None else alpha * value + (1 - alpha) * acc
        out.append(acc.copy())
    return np.stack(out)


def test_smoothing_cuts_the_sign_flips():
    rng = np.random.default_rng(0)
    # 실제 신호(느린 사인) + 미분 노이즈
    t = np.linspace(0, 4, 200)
    raw = (np.sin(t)[:, None] * 10 + rng.normal(0, 8, (200, 7))).astype(np.float32)

    def flips(v):
        return np.mean(np.sign(v[1:]) != np.sign(v[:-1]))

    assert flips(_ema_velocity(raw, 0.2)) < flips(raw) / 2


def test_smoothing_preserves_magnitude():
    """다듬으면서 속도 자체가 사라지면 피드포워드 의미가 없다."""
    t = np.linspace(0, 4, 200)
    raw = np.tile(np.sin(t)[:, None] * 10, (1, 7)).astype(np.float32)
    smoothed = _ema_velocity(raw, 0.2)
    assert np.abs(smoothed).mean() > 0.5 * np.abs(raw).mean()


def test_alpha_one_is_a_passthrough():
    raw = np.arange(21, dtype=np.float32).reshape(3, 7)
    np.testing.assert_allclose(_ema_velocity(raw, 1.0), raw)


def test_vel_scale_zero_disables_feedforward():
    args = runner.parse_args(
        ["--dataset-root", "d", "--policy-path", "p", "--mit-vel-scale", "0"]
    )
    assert runner.settings_from_args(args).mit_vel_scale == 0.0


def test_vel_smoothing_defaults_to_the_measured_choice():
    args = runner.parse_args(["--dataset-root", "d", "--policy-path", "p"])
    settings = runner.settings_from_args(args)
    assert settings.mit_vel_smoothing == pytest.approx(0.2)
    assert settings.mit_vel_scale == pytest.approx(1.0)


def test_modes_enable_the_ema_stage():
    """EMA는 오랫동안 꺼져 있었다(alpha=1.0). MOVE J는 점대점 플래너가 명령
    지터를 뭉개줘서 티가 안 났지만, MIT는 충실한 추종기라 그대로 재현한다 —
    실측에서 위치 명령의 방향 반전이 33.7%였고 alpha=0.2에서 5.8%로 줄었다
    (이동폭은 42.57 → 42.28로 사실상 그대로)."""
    for name in ("demo", "augment"):
        settings = runner.RunSettings.from_mode(name, dataset_root="d", policy_path="p")
        assert settings.smoothing.ema_alpha == pytest.approx(0.2)
        assert settings.smoothing.ema_alpha < 1.0  # 1.0이면 EMA 단계가 꺼진 것


# ── 중단 시 정리 ──────────────────────────────────────────
# Stop 버튼은 프로세스 그룹에 SIGINT를 보낸다. 예전에는 KeyboardInterrupt를 받자마자
# 이벤트 루프를 나가 join(timeout=30)만 했는데, park_lower는 parking(≤10초) + 램프
# 2초 + 그리퍼 여닫기 1.5초×2라 30초를 넘길 수 있다. 그러면 데몬 스레드가 파킹
# 도중에 잘려서 팔이 그 자리에 늘어진다 — 실제로 그렇게 동작했다.
class _FakeRunner:
    def __init__(self, messages, final="finished", alive_after=True):
        self.events = __import__("queue").Queue()
        for message in messages:
            self.events.put((runner.Event.LOG, message))
        if final is not None:
            self.events.put((runner.Event.FINISHED, final))
        self.status = "not_started"
        self._alive = alive_after

    def is_alive(self):
        return self._alive


def test_drain_prints_cleanup_logs_after_interrupt(capsys):
    """정리 로그가 큐에 쌓인 채 버려지면 파킹 여부를 알 수 없다."""
    fake = _FakeRunner(["[ROBOT] MIT 해제", "[DISCONNECT] park=True"])
    assert runner._drain_until_finished(fake) == "finished"
    output = capsys.readouterr().out
    assert "MIT 해제" in output
    assert "[DISCONNECT] park=True" in output


def test_drain_returns_when_thread_dies_without_finished_event():
    fake = _FakeRunner([], final=None, alive_after=False)
    fake.status = "estop"
    assert runner._drain_until_finished(fake) == "estop"


def test_shutdown_timeout_covers_the_park_sequence():
    """parking(≤10초)+램프+그리퍼 사이클 합보다 넉넉해야 한다."""
    assert runner.SHUTDOWN_TIMEOUT_S >= 60.0


def test_sigterm_is_routed_through_the_cleanup_path():
    import inspect

    source = inspect.getsource(runner.main)
    assert "signal.SIGTERM" in source
    assert "raise KeyboardInterrupt" in source
