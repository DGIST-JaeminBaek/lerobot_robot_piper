#!/usr/bin/env python3
"""추론 action 궤적의 흔들림(jerk)을 줄이는 smoothing 요소들.

이 모듈은 하드웨어/ROS2/lerobot에 전혀 의존하지 않는 순수 numpy 코드다.
piper_infer_gui.py가 실행 경로에서 이걸 그대로 쓰고, 같은 함수로 오프라인
분석(스무딩 파라미터 스윕)도 할 수 있게 분리해 뒀다.

세 단계가 순서대로 겹쳐서 동작한다:

  1. TemporalEnsemble — chunk 간 "예측 불일치"를 줄인다.
     매 스텝 새 chunk를 예측하면 timestep t는 여러 chunk가 각자 예측하게 되는데,
     그 예측들을 exp(-m*i) 가중평균으로 섞는다(ACT 논문 방식). chunk 경계에서
     목표값이 갑자기 튀는 현상이 여기서 사라진다.
  2. ExponentialMovingAverage — 남은 고주파 노이즈를 깎는다.
  3. RateLimiter — 스텝당 변화량 상한. 한 번의 튀는 예측이 로봇을 크게
     흔드는 걸 막는 마지막 안전장치이며, PiperFollower의 max_relative_target과
     같은 역할을 실행 전 단계에서 미리 수행한다.

세 단계 모두 끄면(ensemble off, alpha=1.0, rate_limit=None) 원본 action이
그대로 나오므로 baseline 비교가 가능하다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ACTION_DIM = 7


# ═══════════════════════════════════════════════════════════════════
# 1. Temporal ensemble
# ═══════════════════════════════════════════════════════════════════
class TemporalEnsemble:
    """timestep별로 겹쳐 들어온 chunk 예측들을 지수가중 평균한다.

    LoRA-SP의 common/utils/utils.py::load_buffer / get_current_action와 같은
    수식이지만, deque/CUDA 의존성을 걷어내고 chunk 길이가 매번 달라져도 되게
    일반화했다.

    사용법:
        ens = TemporalEnsemble(m=0.01)
        for step in ...:
            chunk = policy.predict_action_chunk(obs)   # (H, 7)
            ens.add_chunk(chunk)
            action = ens.pop_action()                  # (7,)

    m의 의미 — buffer[0]의 i번째(0이 가장 최신) 예측 가중치가 exp(-m*i):
        m >= 1.0   최신 1~2개 예측만 반영. 반응은 빠르지만 거의 스무딩 안 됨.
        m ~ 0.1    중간.
        m ~ 0.01   사실상 균등 평균. 가장 부드러움 (ACT 논문 기본값).
    """

    def __init__(self, m: float = 0.01, max_horizon: int = 200) -> None:
        if m < 0:
            raise ValueError("m must be non-negative")
        if max_horizon <= 0:
            raise ValueError("max_horizon must be positive")
        self.m = float(m)
        self.max_horizon = int(max_horizon)
        # buffer[k] = "지금부터 k 스텝 뒤"를 예측한 action들의 리스트.
        # 리스트 안에서 index 0이 가장 오래된 예측이므로, 가중치를 줄 때
        # 뒤집어서 최신 예측이 i=0이 되게 한다.
        self._buffer: list[list[np.ndarray]] = []

    def reset(self) -> None:
        self._buffer.clear()

    @property
    def pending_steps(self) -> int:
        return len(self._buffer)

    def votes_for_next(self) -> int:
        """다음 pop_action()이 평균낼 예측 개수 — 앙상블이 실제로 먹히는지 보는 지표."""
        return len(self._buffer[0]) if self._buffer else 0

    def add_chunk(self, chunk: np.ndarray) -> None:
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[1] != ACTION_DIM:
            raise ValueError(f"chunk must be (H, {ACTION_DIM}), got {chunk.shape}")
        horizon = min(len(chunk), self.max_horizon)
        while len(self._buffer) < horizon:
            self._buffer.append([])
        for step in range(horizon):
            self._buffer[step].append(chunk[step])

    def pop_action(self) -> np.ndarray:
        """현재 timestep의 앙상블 action을 꺼내고 버퍼를 한 칸 전진시킨다."""
        if not self._buffer:
            raise RuntimeError("pop_action() called before any add_chunk()")
        votes = np.stack(self._buffer.pop(0), axis=0)  # (V, 7), index 0이 가장 오래됨
        # 최신 예측이 i=0이 되도록 뒤집는다.
        votes = votes[::-1]
        indices = np.arange(len(votes), dtype=np.float32)
        weights = np.exp(-self.m * indices)
        return (votes * weights[:, None]).sum(axis=0) / weights.sum()


# ═══════════════════════════════════════════════════════════════════
# 2. EMA
# ═══════════════════════════════════════════════════════════════════
class ExponentialMovingAverage:
    """a_t <- alpha * a_t + (1 - alpha) * a_{t-1}. alpha=1.0이면 무동작."""

    def __init__(self, alpha: float = 1.0) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self._previous: np.ndarray | None = None

    def reset(self, state: np.ndarray | None = None) -> None:
        self._previous = None if state is None else np.asarray(state, dtype=np.float32).copy()

    def __call__(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if self.alpha >= 1.0 or self._previous is None:
            self._previous = action.copy()
            return action
        smoothed = self.alpha * action + (1.0 - self.alpha) * self._previous
        self._previous = smoothed
        return smoothed


# ═══════════════════════════════════════════════════════════════════
# 3. Rate limit
# ═══════════════════════════════════════════════════════════════════
class RateLimiter:
    """스텝당 정규화 단위 변화량을 limit 이하로 클램프한다. None이면 무동작.

    reference를 "직전에 내보낸 명령"으로 두므로, 실제 로봇의
    max_relative_target(측정 position 기준)과는 근사 관계다. 실물 실행 시에는
    PiperFollower.send_action()이 다시 한 번 실제 clamp를 적용한다.
    """

    def __init__(self, limit: float | None = None) -> None:
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive or None")
        self.limit = limit
        self._reference: np.ndarray | None = None

    def reset(self, state: np.ndarray | None = None) -> None:
        self._reference = None if state is None else np.asarray(state, dtype=np.float32).copy()

    def __call__(self, action: np.ndarray) -> tuple[np.ndarray, float]:
        """(clamp된 action, 이번 스텝에서 깎아낸 최대량)을 돌려준다."""
        action = np.asarray(action, dtype=np.float32)
        if self.limit is None or self._reference is None:
            self._reference = action.copy()
            return action, 0.0
        limited = np.clip(action, self._reference - self.limit, self._reference + self.limit)
        adjustment = float(np.abs(limited - action).max(initial=0.0))
        self._reference = limited
        return limited, adjustment


# ═══════════════════════════════════════════════════════════════════
# 파이프라인
# ═══════════════════════════════════════════════════════════════════
@dataclass
class SmoothingConfig:
    temporal_ensemble: bool = True
    ensemble_m: float = 0.01
    ema_alpha: float = 1.0
    rate_limit: float | None = 5.0
    # 정규화 범위 clamp (joint -100~100, gripper 0~100)
    clip_to_range: bool = True

    def summary(self) -> str:
        ensemble = f"ensemble(m={self.ensemble_m:g})" if self.temporal_ensemble else "ensemble(off)"
        ema = f"ema(a={self.ema_alpha:g})" if self.ema_alpha < 1.0 else "ema(off)"
        rate = f"rate(<={self.rate_limit:g})" if self.rate_limit else "rate(off)"
        return f"{ensemble} -> {ema} -> {rate}"


GLOBAL_LOW = np.asarray([-100.0] * 6 + [0.0], dtype=np.float32)
GLOBAL_HIGH = np.asarray([100.0] * 7, dtype=np.float32)


class SmoothingPipeline:
    """ensemble -> EMA -> rate limit -> 범위 clamp를 한 번에 적용한다.

    앙상블을 끈 경우에는 chunk를 그대로 큐잉해서 순서대로 흘려보내므로,
    호출부는 앙상블 on/off와 무관하게 같은 코드로 쓸 수 있다.
    """

    def __init__(self, config: SmoothingConfig) -> None:
        self.config = config
        self._ensemble = TemporalEnsemble(config.ensemble_m)
        self._ema = ExponentialMovingAverage(config.ema_alpha)
        self._rate = RateLimiter(config.rate_limit)
        self._passthrough: list[np.ndarray] = []
        self.last_rate_adjustment = 0.0

    def reset(self, state: np.ndarray | None = None) -> None:
        self._ensemble.reset()
        self._ema.reset(state)
        self._rate.reset(state)
        self._passthrough.clear()
        self.last_rate_adjustment = 0.0

    def update_config(self, config: SmoothingConfig, state: np.ndarray | None = None) -> None:
        """GUI에서 실행 중에 파라미터를 바꿀 때 사용. 버퍼는 비운다."""
        self.config = config
        self._ensemble = TemporalEnsemble(config.ensemble_m)
        self._ema = ExponentialMovingAverage(config.ema_alpha)
        self._rate = RateLimiter(config.rate_limit)
        self._passthrough.clear()
        self._ema.reset(state)
        self._rate.reset(state)

    @property
    def votes_for_next(self) -> int:
        if self.config.temporal_ensemble:
            return self._ensemble.votes_for_next()
        return 1 if self._passthrough else 0

    @property
    def pending_steps(self) -> int:
        if self.config.temporal_ensemble:
            return self._ensemble.pending_steps
        return len(self._passthrough)

    def add_chunk(self, chunk: np.ndarray) -> None:
        if self.config.temporal_ensemble:
            self._ensemble.add_chunk(chunk)
        else:
            # 앙상블 off — 새 chunk가 이전 chunk의 남은 부분을 덮어쓴다.
            # (기존 open-loop 실행과 동일한 동작)
            self._passthrough = [np.asarray(a, dtype=np.float32) for a in chunk]

    def next_action(self) -> np.ndarray:
        if self.config.temporal_ensemble:
            action = self._ensemble.pop_action()
        else:
            if not self._passthrough:
                raise RuntimeError("next_action() called with an empty queue")
            action = self._passthrough.pop(0)

        action = self._ema(action)
        action, self.last_rate_adjustment = self._rate(action)
        if self.config.clip_to_range:
            action = np.clip(action, GLOBAL_LOW, GLOBAL_HIGH)
            # clamp 결과를 rate limiter의 기준으로도 되돌려 놓아야 다음 스텝의
            # 상대 제한이 실제로 내보낸 값 기준이 된다.
            self._rate.reset(action)
        return action.astype(np.float32, copy=False)


# ═══════════════════════════════════════════════════════════════════
# smoothness 지표 — 논문에 넣을 정량 비교용
# ═══════════════════════════════════════════════════════════════════
def smoothness_metrics(trajectory: np.ndarray, fps: float = 30.0) -> dict[str, float]:
    """action 궤적의 부드러움을 수치화한다. 값이 작을수록 부드럽다.

    total_variation : 스텝당 |Δa| 평균 (정규화 단위/step)
    max_step        : 한 스텝에서 일어난 최대 변화 — 튀는 지점 탐지용
    mean_jerk       : 3차 차분 크기 평균 (정규화 단위/s^3)
    rms_jerk        : 3차 차분 RMS
    """
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.ndim != 2 or len(trajectory) < 2:
        return {"total_variation": 0.0, "max_step": 0.0, "mean_jerk": 0.0, "rms_jerk": 0.0}

    delta = np.diff(trajectory, axis=0)
    metrics = {
        "total_variation": float(np.abs(delta).sum(axis=1).mean()),
        "max_step": float(np.abs(delta).max(initial=0.0)),
        "mean_jerk": 0.0,
        "rms_jerk": 0.0,
    }
    if len(trajectory) >= 4:
        jerk = np.diff(trajectory, n=3, axis=0) * (fps ** 3)
        metrics["mean_jerk"] = float(np.abs(jerk).mean())
        metrics["rms_jerk"] = float(np.sqrt(np.square(jerk).mean()))
    return metrics
