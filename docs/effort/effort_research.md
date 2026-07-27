# Effort 활용 선행 연구와 적용 아이디어

## 1. 문서 범위

이 문서는 Piper에서 녹화한 effort와 velocity를 학습 및 분석에 활용할 때 참고할
선행 연구와 실험 아이디어를 정리한다. 실제 녹화 구현과 사용 방법은
[`effort.md`](./effort.md)를 참고한다.

Piper의 effort는 별도의 힘/토크 센서가 측정한 값이 아니라 모터 전류를 기반으로
계산한 추정값이다. 따라서 6축 force/torque 센서를 사용한 연구 결과를 그대로 적용할
수 있다고 가정하면 안 된다.

```text
Piper 측정 effort
  = 중력 + 마찰 + 관성 + 접촉에 의한 성분 + 측정 오차
```

## 2. 선행 연구

### TA-VLA

**TA-VLA: Elucidating the Design Space of Torque-aware Vision-Language-Action Models**

- [논문](https://arxiv.org/abs/2509.07962)
- [프로젝트 페이지](https://zzongzheng0918.github.io/Torque-Aware-VLA.github.io/)
- CoRL 2025

TA-VLA는 pretrained VLA에 관절 torque를 어느 위치와 형태로 넣을지 비교했다. 논문이
보고한 핵심 결과는 다음과 같다.

1. Torque를 conditioning encoder보다 action decoder에 넣었을 때 더 좋은 결과를
   보였다.
2. 현재 프레임 하나보다 torque 이력이 유용했다.
3. 여러 이력 토큰을 추가하는 것보다 전체 이력을 하나의 토큰으로 요약해 decoder에
   넣는 방식이 가장 효과적이었다.
4. Action과 함께 미래 torque를 보조 목표로 예측했을 때 성능이 추가로 향상됐다.

이 연구는 본 프로젝트와 센서 조건이 비교적 가깝다. TA-VLA도 별도의 외력 센서가
아니라 모터 전류와 torque constant를 이용해 관절 torque를 추정했다. 다만 로봇,
모터 모델, 캘리브레이션 정확도가 다르므로 값의 품질이 같다고 볼 수는 없다.

현재 프로젝트는 effort를 position 및 velocity와 함께
`observation.state`에 넣는다. 로컬 SmolVLA 구현을 확인하면 state 전체가
`state_proj`를 거쳐 prefix의 state token으로 들어간다. 따라서 현재 방식은
TA-VLA의 decoder 전용 torque adapter나 torque-history token을 구현한 것이 아니라,
가장 단순한 baseline에 해당한다.

### ForceVLA

**ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation**

- [논문](https://arxiv.org/abs/2505.22159)
- NeurIPS 2025

ForceVLA는 외부 6축 force를 vision-language embedding 및 action decoding 과정에
결합한다. Force-aware Mixture-of-Experts를 사용해 상황에 따라 modality별 expert를
선택하도록 구성했으며, vision·proprioception·force가 동기화된 데이터셋으로
contact-rich 작업을 학습했다.

본 프로젝트에 그대로 적용하기 어려운 이유는 입력이 6축 end-effector force인 반면,
Piper의 입력은 7개 모터의 전류 기반 effort이기 때문이다. 그러나 effort를 단순히
state 뒤에 붙이는 대신 별도의 adapter를 거쳐 action 생성 단계에서 결합한다는 설계는
참고할 수 있다.

### FAVLA

**FAVLA: A Force-Adaptive Fast-Slow VLA model for Contact-Rich Robotic Manipulation**

- [논문](https://arxiv.org/abs/2602.23648)
- 2026년 arXiv preprint

FAVLA는 느린 vision-language planning과 빠른 force 기반 반응 제어를 분리한다.
고비용 VLM은 낮은 주기로 실행하고, action expert는 최신 force 이력을 사용해 더 빠른
주기로 접촉 변화에 반응한다. 또한 예상 force 변화에 따라 action expert의 실행 주기를
조절한다.

이 구조는 force 관측과 policy 실행 주기를 함께 고려해야 한다는 연구 사례다. 본
프로젝트의 동기·비동기 action queue와 재계획 주기는
[`policy_execution.md`](../policy_execution.md)에서 별도로 다룬다.

### ForceVLA2

**ForceVLA2: Unleashing Hybrid Force-Position Control with Force Awareness for
Contact-Rich Manipulation**

- [논문](https://arxiv.org/abs/2603.15169)
- CVPR 2026

ForceVLA2는 force를 관측하는 데서 끝나지 않고 hybrid force-position control까지
결합한다. 데이터셋에는 wiping, pressing, assembling을 포함한 contact-rich 작업의
다중 시점 영상, 언어 지시, proprioceptive state와 force가 들어간다.

Wiping 작업을 포함한다는 점에서 보드 지우기 작업과 직접적인 관련성이 있다. 동시에
이 연구는 force를 인식하는 모델과 실제 force를 조절할 수 있는 제어기의 차이를
보여준다. 현재 Piper 정책은 position action을 출력하므로 effort를 관측하더라도
힘을 직접 명령할 수 없고, position을 변경해 간접적으로만 접촉력을 조절할 수 있다.

## 3. 현재 구현의 위치

현재 구현은 다음 단계 중 첫 번째 단계에 해당한다.

| 단계 | 내용 | 현재 상태 |
|---|---|---|
| 데이터 수집 | Position + effort + velocity 동기화 저장 | 구현됨 |
| 단순 입력 baseline | 20차원 `observation.state`로 학습 | 실험 가능 |
| Effort 전처리 | 자세·속도에 따른 정상 effort 제거 | 미구현 |
| 시간 이력 | 최근 effort sequence 사용 | 미구현 |
| 별도 adapter | Effort 전용 token 또는 decoder adapter | 미구현 |
| 보조 예측 | 미래 effort를 action과 함께 예측 | 미구현 |
| 힘 제어 | Hybrid force-position 또는 compliance control | 미구현 |

현재 방식으로도 effort 정보의 유용성을 확인하는 baseline은 만들 수 있다. 다만 성능이
향상되지 않더라도 effort 자체가 무의미하다고 바로 결론 내릴 수는 없다. 현재 프레임의
raw effort를 state token에 함께 넣는 방식과, 선행 연구의 effort 이력·decoder 결합
방식은 서로 다른 조건이기 때문이다.

## 4. 적용 아이디어

### 4.1 Raw effort baseline

가장 먼저 현재 구현만 사용해 두 조건을 비교한다.

```text
Baseline A: position 7개
Baseline B: position 7개 + raw effort 7개 + velocity 6개
```

학습 데이터, seed, action 설정과 평가 조건은 동일하게 유지한다. 이 실험은 현재
구현의 효과를 확인하는 기준선이며, 별도 모델 수정이 필요하지 않다.

### 4.2 Effort residual

접촉하지 않은 상태에서 position과 velocity에 따라 예상되는 effort를 모델링한다.

```text
expected_effort = f(position, velocity)
residual_effort = measured_effort - expected_effort
```

Residual은 중력과 운동 상태로 설명되는 성분을 줄여 접촉에 의한 변화를 더 분명하게
만드는 것이 목적이다. 이를 위해 실제 작업과 같은 도구 및 payload를 장착한 상태에서
접촉 없는 자유운동 데이터를 별도로 수집해야 한다.

비교할 조건:

```text
position only
position + raw effort + velocity
position + residual effort + velocity
```

### 4.3 Effort 이력 사용

TA-VLA 결과를 참고해 현재 프레임 하나가 아니라 최근 여러 프레임의 effort와
velocity를 사용한다.

```text
input = effort[t-H+1 : t], velocity[t-H+1 : t]
```

처음에는 모델 구조를 크게 바꾸기 전에 다음과 같은 작은 temporal encoder로 이력을
하나의 embedding으로 요약할 수 있다.

- 1D convolution
- 작은 MLP
- GRU
- 작은 Transformer

요약 결과를 token 하나로 만들어 action 생성부에 넣는 방식이 첫 구현 후보이다.

### 4.4 Decoder-side effort adapter

현재 SmolVLA의 state token에 effort를 함께 넣는 baseline과 별도로 effort 전용
adapter를 만든다.

```text
effort/velocity history
        ↓
temporal effort encoder
        ↓
single effort token
        ↓
action expert / decoder
```

TA-VLA의 결과가 다른 로봇과 모델에서도 그대로 재현된다고 가정해서는 안 된다.
따라서 기존 state 결합 방식과 decoder adapter 방식을 같은 데이터로 비교해야 한다.

### 4.5 미래 effort 보조 예측

Action을 예측할 때 같은 horizon의 effort도 함께 예측하도록 보조 loss를 추가한다.

```text
total_loss = action_loss + beta * future_effort_loss
```

목표는 배포 시 미래 effort 값을 직접 사용하는 것이 아니라, 모델이 action과 그에
따르는 물리적 결과의 관계를 학습하도록 만드는 것이다. `beta`에 따른 action 성능과
effort 예측 오차를 함께 평가해야 한다.

### 4.6 Effort dropout

모델이 effort를 단순한 동작 방향 표시로만 사용하는 것을 줄이기 위해 학습 중 일부
구간의 effort를 가리거나 noise를 추가하는 방법을 비교할 수 있다.

검증 시에는 다음 조건을 함께 평가한다.

- 정상 effort 입력
- Effort를 0으로 대체
- Effort time shift
- 관절별 effort 일부 masking

정상 입력에서만 성능이 높다는 사실만으로 접촉을 이해했다고 볼 수는 없다. Time shift
또는 masking에 대한 변화까지 확인해야 모델이 어떤 정보를 사용했는지 판단할 수 있다.

## 5. 평가 항목

단순 성공률 외에 접촉 품질과 반응성을 함께 기록한다.

- 작업 성공률
- 지워진 면적 비율
- 작업 완료 시간
- 관절별 peak effort
- 정상 접촉 구간의 평균 및 분산
- Effort 급상승 후 action이 변하기까지의 시간
- 안전 제어 발동 횟수

Raw effort는 자세와 속도의 영향을 받으므로 서로 다른 자세에서 측정한 절대값을 단순
비교하지 않는다. 동일한 궤적과 payload 조건을 유지하거나 residual effort를 함께
보고해야 한다.

## 6. 권장 실험 순서

1. 현재 구현으로 position-only와 raw-effort baseline을 비교한다.
2. 같은 세션에서 접촉 없는 자유운동 데이터를 수집한다.
3. Effort residual을 계산해 raw effort와 비교한다.
4. Effort 이력 encoder와 decoder-side adapter를 추가한다.
5. 미래 effort 보조 예측을 추가한다.
6. 최종 후보에 대해 effort masking 및 time-shift 실험을 수행한다.

각 단계를 따로 비교해야 어떤 변경이 성능에 영향을 줬는지 구분할 수 있다.

## 7. 안전 제어와의 구분

Effort를 정책 입력이나 학습 목표로 사용하는 것과 effort 임계값으로 로봇을
후퇴·parking·종료시키는 안전 제어는 별개의 문제다.

학습된 정책은 예측 실패와 분포 밖 입력이 발생할 수 있으므로 안전 기능을 정책의
effort 활용 능력에 의존시키면 안 된다. 안전 제어는 정책보다 높은 우선순위에서
독립적으로 effort를 감시해야 한다.
