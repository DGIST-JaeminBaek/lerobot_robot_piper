#!/usr/bin/env python3
"""연속성 제약 재적합(CCR) 경로를 하드웨어·정책 없이 검증한다.

CCR은 새 chunk의 앞을 잘라 버리는 대신, B-스플라인 제어점의 앞쪽 몇 개를 이미
실행된 행동에 맞춰 다시 풀어 궤적을 직전 과거에 고정한다
(ABPolicy arXiv:2602.23901 III-C). 여기서 지키려는 성질:

  - 기본은 꺼져 있다. CCR 이전 동작과 완전히 같아야 한다
  - 정책이 지원하지 않으면 조용히 무시하지 않고 명확히 끈다
  - 실패(이력 부족·예외)는 절단 방식으로 되돌아가며, 실물이 도는 중에 죽지 않는다
  - 성공하면 절단과 같은 길이의 궤적을 돌려준다

실행: python -m pytest scripts/tests/piper/inference/test_ccr_mock.py
"""

from __future__ import annotations

import pathlib
import sys
import types

import numpy as np
import pytest

SCRIPTS_DIR = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "0__launch_gui.sh").is_file())
INFERENCE_DIR = SCRIPTS_DIR / "piper" / "inference"
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

import piper_infer_runner as runner  # noqa: E402

DIM, N_CTRL, HISTORY, HORIZON = 7, 12, 8, 50


class FakeABPPolicy:
    """smolvla_abp 정책의 CCR 인터페이스만 흉내낸다."""

    def __init__(self, n_free=4, raise_on_refit=False):
        self.config = types.SimpleNamespace(
            action_history_horizon=HISTORY, n_ctrl=N_CTRL, ccr_n_free=n_free
        )
        self.last_control_points = None
        self.raise_on_refit = raise_on_refit
        self.refit_calls: list[tuple] = []

    def refit_control_points(self, executed, control_points, n_free=None):
        if self.raise_on_refit:
            raise RuntimeError("특이 행렬")
        self.refit_calls.append((np.asarray(executed).shape, n_free))
        out = np.asarray(control_points, dtype=float).copy()
        out[:4] += 1.0            # 앞쪽만 바뀌었다는 표시
        return out

    def rebuild_actions(self, control_points, drop_history=None):
        drop = HISTORY if drop_history is None else drop_history
        window = HISTORY + HORIZON
        # 실제 구현과 같은 길이 규약: window - drop
        return np.ones((window - drop, DIM), dtype=np.float32) * float(control_points[0, 0])


def _runner(ccr=True, ccr_n_free=None, policy=None, trajectory_len=64, latency_align=True):
    settings = runner.RunSettings.from_mode(
        "demo", dataset_root="d", policy_path="p",
        ccr=ccr, ccr_n_free=ccr_n_free, latency_align=latency_align, horizon=HORIZON,
    )
    r = runner.InferenceRunner(settings)
    r._policy = policy if policy is not None else FakeABPPolicy()
    r._ccr_active = ccr
    r.trajectory = [np.full(DIM, float(i)) for i in range(trajectory_len)]
    return r


# ── 기본값 ────────────────────────────────────────────────
def test_ccr_is_off_by_default():
    settings = runner.RunSettings.from_mode("demo", dataset_root="d", policy_path="p")
    assert settings.ccr is False
    assert settings.ccr_n_free is None


def test_cli_flags_reach_settings():
    base = ["--dataset-root", "d", "--policy-path", "p"]
    off = runner.settings_from_args(runner.parse_args(base))
    assert off.ccr is False and off.ccr_n_free is None

    on = runner.settings_from_args(runner.parse_args(base + ["--ccr", "--ccr-n-free", "6"]))
    assert on.ccr is True and on.ccr_n_free == 6


def test_summary_reports_ccr_state():
    base = ["--dataset-root", "d", "--policy-path", "p"]
    assert "ccr=off" in runner.settings_from_args(runner.parse_args(base)).describe()
    assert "ccr=on" in runner.settings_from_args(runner.parse_args(base + ["--ccr"])).describe()


def test_off_keeps_previous_truncation_behaviour():
    r = _runner(ccr=False)
    chunk = np.arange(HORIZON * DIM, dtype=np.float32).reshape(HORIZON, DIM)
    ctrl = np.zeros((N_CTRL, DIM), dtype=np.float32)
    out = r._align_chunk(requested_step=10, chunk=chunk, step=13, control_points=ctrl)
    np.testing.assert_array_equal(out, chunk[3:])
    assert r._ccr_applied == 0


# ── 동작 ──────────────────────────────────────────────────
def test_refit_is_applied_and_length_matches_truncation():
    r = _runner()
    chunk = np.zeros((HORIZON, DIM), dtype=np.float32)
    ctrl = np.full((N_CTRL, DIM), 3.0, dtype=np.float32)
    lag = 4
    out = r._align_chunk(requested_step=10, chunk=chunk, step=10 + lag, control_points=ctrl)

    assert r._ccr_applied == 1
    assert len(out) == len(chunk) - lag        # 절단과 같은 길이여야 큐 회계가 맞는다
    assert out.dtype == np.float32
    # 실행 이력은 P + lag개를 넘겨야 한다
    (executed_shape, n_free), = r._policy.refit_calls
    assert executed_shape == (HISTORY + lag, DIM)
    assert n_free is None                       # 오버라이드 없으면 정책 기본값


def test_ccr_runs_even_at_zero_lag():
    """지연이 0이어도 새 chunk는 직전 실행 이력과 이어져야 한다."""
    r = _runner()
    out = r._align_chunk(10, np.zeros((HORIZON, DIM), np.float32), 10,
                         np.full((N_CTRL, DIM), 2.0, np.float32))
    assert r._ccr_applied == 1
    assert len(out) == HORIZON
    (executed_shape, _), = r._policy.refit_calls
    assert executed_shape == (HISTORY, DIM)


def test_n_free_override_is_passed_through():
    r = _runner(ccr_n_free=6)
    r._align_chunk(10, np.zeros((HORIZON, DIM), np.float32), 12,
                   np.zeros((N_CTRL, DIM), np.float32))
    (_, n_free), = r._policy.refit_calls
    assert n_free == 6


# ── 되돌아가기 ────────────────────────────────────────────
def test_falls_back_when_history_too_short():
    """에피소드 초반 — 고정할 과거가 없으면 절단으로 돌아간다."""
    r = _runner(trajectory_len=5)          # P=8보다 짧다
    chunk = np.arange(HORIZON * DIM, dtype=np.float32).reshape(HORIZON, DIM)
    out = r._align_chunk(10, chunk, 12, np.zeros((N_CTRL, DIM), np.float32))
    np.testing.assert_array_equal(out, chunk[2:])
    assert r._ccr_applied == 0 and r._ccr_skipped == 1


def test_falls_back_and_logs_once_on_policy_error():
    r = _runner(policy=FakeABPPolicy(raise_on_refit=True))
    chunk = np.arange(HORIZON * DIM, dtype=np.float32).reshape(HORIZON, DIM)
    for _ in range(3):
        out = r._align_chunk(10, chunk, 12, np.zeros((N_CTRL, DIM), np.float32))
        np.testing.assert_array_equal(out, chunk[2:])
    assert r._ccr_skipped == 3
    assert r._ccr_failure_logged is True     # 매 스텝 쏟아지지 않는다


def test_falls_back_on_dim_mismatch():
    r = _runner()
    chunk = np.zeros((HORIZON, DIM), np.float32)
    out = r._align_chunk(10, chunk, 12, np.zeros((N_CTRL, DIM + 1), np.float32))
    np.testing.assert_array_equal(out, chunk[2:])
    assert r._ccr_skipped == 1


def test_stale_chunk_is_dropped_before_refit():
    """이미 다 지나간 chunk는 CCR 이전에 버린다 — 정책을 부를 이유가 없다."""
    r = _runner()
    out = r._align_chunk(10, np.zeros((HORIZON, DIM), np.float32), 10 + HORIZON,
                         np.zeros((N_CTRL, DIM), np.float32))
    assert out is None
    assert r._stale_chunks == 1
    assert r._policy.refit_calls == []


def test_no_control_points_means_truncation():
    """정책이 제어점을 안 주면(stock SmolVLA) 기존 경로 그대로."""
    r = _runner()
    chunk = np.arange(HORIZON * DIM, dtype=np.float32).reshape(HORIZON, DIM)
    out = r._align_chunk(10, chunk, 12, None)
    np.testing.assert_array_equal(out, chunk[2:])
    assert r._ccr_applied == 0 and r._ccr_skipped == 0


# ── 결과 큐 하위 호환 ─────────────────────────────────────
def test_result_queue_accepts_two_and_three_tuples():
    assert runner.InferenceRunner._unpack_result((3, "chunk")) == (3, "chunk", None)
    assert runner.InferenceRunner._unpack_result((3, "chunk", "ctrl")) == (3, "chunk", "ctrl")
