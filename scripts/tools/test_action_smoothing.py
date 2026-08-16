#!/usr/bin/env python3
"""action_smoothing.py 단위 테스트 — 하드웨어/정책 없이 순수 numpy로 검증한다.

    python -m pytest scripts/tools/test_action_smoothing.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from action_smoothing import (  # noqa: E402
    ExponentialMovingAverage,
    RateLimiter,
    SmoothingConfig,
    SmoothingPipeline,
    TemporalEnsemble,
    smoothness_metrics,
)


def constant_chunk(value: float, horizon: int = 5) -> np.ndarray:
    return np.full((horizon, 7), value, dtype=np.float32)


# ── TemporalEnsemble ────────────────────────────────────────────────
def test_single_chunk_passes_through_unchanged():
    ensemble = TemporalEnsemble(m=0.01)
    chunk = np.arange(35, dtype=np.float32).reshape(5, 7)
    ensemble.add_chunk(chunk)
    for step in range(5):
        assert np.allclose(ensemble.pop_action(), chunk[step])


def test_small_m_averages_uniformly():
    """m -> 0이면 겹친 예측들의 산술평균에 수렴한다."""
    ensemble = TemporalEnsemble(m=0.0)
    ensemble.add_chunk(constant_chunk(0.0))
    ensemble.add_chunk(constant_chunk(10.0))
    ensemble.add_chunk(constant_chunk(20.0))
    assert np.allclose(ensemble.pop_action(), 10.0)  # (0 + 10 + 20) / 3


def test_large_m_tracks_newest_prediction():
    """m이 크면 가장 최근 chunk의 예측이 사실상 그대로 나온다."""
    ensemble = TemporalEnsemble(m=20.0)
    ensemble.add_chunk(constant_chunk(0.0))
    ensemble.add_chunk(constant_chunk(100.0))
    assert ensemble.pop_action()[0] == pytest.approx(100.0, abs=1e-4)


def test_m_monotonically_controls_smoothing():
    """같은 입력에서 m이 작을수록 과거 예측 쪽으로 더 끌린다."""
    results = []
    for m in (0.01, 0.3, 1.0, 3.0):
        ensemble = TemporalEnsemble(m=m)
        ensemble.add_chunk(constant_chunk(0.0))
        ensemble.add_chunk(constant_chunk(100.0))
        results.append(ensemble.pop_action()[0])
    assert results == sorted(results), f"m이 커질수록 최신값에 가까워야 함: {results}"


def test_votes_counts_overlapping_predictions():
    ensemble = TemporalEnsemble(m=0.01)
    assert ensemble.votes_for_next() == 0
    ensemble.add_chunk(constant_chunk(1.0))
    assert ensemble.votes_for_next() == 1
    ensemble.add_chunk(constant_chunk(1.0))
    assert ensemble.votes_for_next() == 2


def test_pop_before_add_raises():
    with pytest.raises(RuntimeError):
        TemporalEnsemble().pop_action()


def test_bad_chunk_shape_raises():
    with pytest.raises(ValueError):
        TemporalEnsemble().add_chunk(np.zeros((5, 6), dtype=np.float32))


# ── EMA ─────────────────────────────────────────────────────────────
def test_ema_alpha_one_is_identity():
    ema = ExponentialMovingAverage(alpha=1.0)
    for value in (0.0, 50.0, -30.0):
        assert np.allclose(ema(np.full(7, value, dtype=np.float32)), value)


def test_ema_blends_toward_previous():
    ema = ExponentialMovingAverage(alpha=0.5)
    ema(np.zeros(7, dtype=np.float32))
    assert np.allclose(ema(np.full(7, 10.0, dtype=np.float32)), 5.0)


def test_ema_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        ExponentialMovingAverage(alpha=0.0)
    with pytest.raises(ValueError):
        ExponentialMovingAverage(alpha=1.5)


# ── RateLimiter ─────────────────────────────────────────────────────
def test_rate_limiter_clamps_and_reports_adjustment():
    limiter = RateLimiter(limit=2.0)
    limiter(np.zeros(7, dtype=np.float32))
    limited, adjustment = limiter(np.full(7, 10.0, dtype=np.float32))
    assert np.allclose(limited, 2.0)
    assert adjustment == pytest.approx(8.0)


def test_rate_limiter_disabled_passes_through():
    limiter = RateLimiter(limit=None)
    limiter(np.zeros(7, dtype=np.float32))
    limited, adjustment = limiter(np.full(7, 999.0, dtype=np.float32))
    assert np.allclose(limited, 999.0)
    assert adjustment == 0.0


# ── Pipeline ────────────────────────────────────────────────────────
def test_pipeline_all_off_is_identity_within_range():
    config = SmoothingConfig(
        temporal_ensemble=False, ema_alpha=1.0, rate_limit=None, clip_to_range=False
    )
    pipeline = SmoothingPipeline(config)
    chunk = np.linspace(-50, 50, 7 * 4, dtype=np.float32).reshape(4, 7)
    pipeline.add_chunk(chunk)
    for step in range(4):
        assert np.allclose(pipeline.next_action(), chunk[step])


def test_pipeline_clips_to_piper_normalized_range():
    pipeline = SmoothingPipeline(
        SmoothingConfig(temporal_ensemble=False, rate_limit=None, clip_to_range=True)
    )
    chunk = np.full((1, 7), -500.0, dtype=np.float32)
    pipeline.add_chunk(chunk)
    action = pipeline.next_action()
    assert np.allclose(action[:6], -100.0)  # joints
    assert action[6] == pytest.approx(0.0)  # gripper의 하한은 0


def test_pipeline_smooths_a_jumpy_policy():
    """chunk 경계마다 목표가 튀는 가짜 정책에서 실제로 흔들림이 줄어드는지."""
    rng = np.random.default_rng(0)
    horizon = 20

    def rollout(config: SmoothingConfig) -> np.ndarray:
        rng_local = np.random.default_rng(0)
        pipeline = SmoothingPipeline(config)
        trajectory = []
        for step in range(30):
            # 매 스텝 새 chunk를 예측하되, chunk마다 bias가 다르게 튄다.
            base = np.linspace(step, step + horizon, horizon, dtype=np.float32)[:, None]
            bias = rng_local.normal(0.0, 4.0, size=(1, 7)).astype(np.float32)
            pipeline.add_chunk(np.repeat(base, 7, axis=1) + bias)
            trajectory.append(pipeline.next_action())
        return np.stack(trajectory)

    raw = rollout(
        SmoothingConfig(
            temporal_ensemble=False, ema_alpha=1.0, rate_limit=None, clip_to_range=False
        )
    )
    smoothed = rollout(
        SmoothingConfig(
            temporal_ensemble=True, ensemble_m=0.01, ema_alpha=1.0,
            rate_limit=None, clip_to_range=False,
        )
    )
    raw_tv = smoothness_metrics(raw)["total_variation"]
    smoothed_tv = smoothness_metrics(smoothed)["total_variation"]
    assert smoothed_tv < raw_tv, f"앙상블이 total variation을 줄이지 못함: {raw_tv} -> {smoothed_tv}"
    assert rng is not None  # (lint 방지)


def test_update_config_switches_smoothing_live():
    pipeline = SmoothingPipeline(SmoothingConfig(temporal_ensemble=True, ensemble_m=0.01))
    pipeline.add_chunk(constant_chunk(0.0))
    pipeline.next_action()
    pipeline.update_config(
        SmoothingConfig(temporal_ensemble=False, rate_limit=None, clip_to_range=False),
        state=np.zeros(7, dtype=np.float32),
    )
    assert pipeline.pending_steps == 0
    pipeline.add_chunk(constant_chunk(42.0))
    assert np.allclose(pipeline.next_action(), 42.0)


# ── metrics ─────────────────────────────────────────────────────────
def test_metrics_zero_for_constant_trajectory():
    metrics = smoothness_metrics(np.zeros((50, 7), dtype=np.float32))
    assert metrics["total_variation"] == pytest.approx(0.0)
    assert metrics["rms_jerk"] == pytest.approx(0.0)


def test_metrics_detect_a_single_spike():
    trajectory = np.zeros((20, 7), dtype=np.float32)
    trajectory[10] = 30.0
    metrics = smoothness_metrics(trajectory)
    assert metrics["max_step"] == pytest.approx(30.0)
    assert metrics["total_variation"] > 0.0


def test_metrics_handle_degenerate_input():
    metrics = smoothness_metrics(np.zeros((1, 7), dtype=np.float32))
    assert metrics["total_variation"] == 0.0
