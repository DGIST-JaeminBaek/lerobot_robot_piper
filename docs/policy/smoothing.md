# 추론 시 로봇이 흔들리는 문제와 smoothing

정책 추론을 돌리면 팔이 눈에 띄게 떨리는 문제에 대한 정리와, 그걸 다루는 도구
사용법입니다.

## 왜 흔들리나

SmolVLA/π₀ 같은 chunk 예측 정책은 한 번의 추론에서 앞으로 50 스텝(`chunk_size=50`)의
action을 한꺼번에 뱉습니다. 실행 방식에 따라 두 가지 흔들림이 생깁니다.

1. **chunk 경계 불연속** — chunk를 끝까지 쓰고 나서 다음 chunk를 예측하면, 새 chunk의
   첫 action이 직전 chunk의 마지막 action과 이어질 이유가 없습니다. 50 스텝마다 목표가
   툭 튑니다.
2. **예측 자체의 고주파 노이즈** — 매 스텝 새로 예측하면 경계 문제는 없어지지만,
   이번에는 인접 스텝의 예측이 서로 조금씩 달라서 목표가 잘게 떱니다.

## 세 겹의 대책

`scripts/tools/action_smoothing.py`가 아래 셋을 순서대로 적용합니다. 셋 다 끄면 원본
action이 그대로 나오므로 baseline 비교가 가능합니다.

| 단계 | 무엇을 고치나 | 파라미터 |
| --- | --- | --- |
| Temporal ensemble | chunk 간 예측 불일치 (1번 원인) | `m` |
| EMA | 남은 고주파 노이즈 (2번 원인) | `alpha` |
| Rate limit | 한 번 크게 튀는 예측 | 스텝당 최대 변화량 |

### Temporal ensemble의 `m`

매 스텝 새 chunk를 예측하면 timestep *t*는 여러 chunk가 각자 예측하게 됩니다. 그
예측들을 `exp(-m·i)` 가중평균으로 섞습니다 (`i=0`이 가장 최신 예측). ACT 논문 방식이고,
LoRA-SP 저장소의 `common/utils/utils.py::get_current_action`과 같은 수식입니다.

- `m ≈ 0.01` — 사실상 균등 평균. 가장 부드러움. **ACT 논문 기본값이고 우리 권장값.**
- `m ≈ 0.1~0.3` — 중간.
- `m ≥ 1.0` — 최신 1~2개 예측만 반영. 반응은 빠르지만 스무딩 효과가 거의 없음.

**중요:** temporal ensemble은 매 스텝 새 chunk를 예측해야(=`infer_every=1`) 겹치는
예측이 생겨서 동작합니다. `infer_every`를 키우면 효과가 사라집니다.

## 실측 결과

`smolvla_erase_shape_512`의 30k step 체크포인트, `records/0727/erase_the_shape_512`
episode 0, dataset observation 40 스텝 기준입니다. 값이 작을수록 부드럽습니다.

| 설정 | TV | max_step | RMS jerk | TV 감소 |
| --- | ---: | ---: | ---: | ---: |
| smoothing 없음 | 5.617 | 4.483 | 88,131 | — |
| ensemble m=0.01 | 2.881 | 2.244 | 11,979 | **48.7%** |
| ensemble m=0.1 | 3.022 | 1.910 | 13,255 | 46.2% |
| ensemble m=1.0 | 4.214 | 3.543 | 48,758 | 25.0% |

`m=0.01`에서 RMS jerk가 약 **7.4배** 줄어듭니다. `m=1.0`(LoRA-SP의 기존 하드코딩
기본값)은 개선 폭이 훨씬 작습니다 — 여기가 우리 실험에서 실제로 손해를 보던 지점입니다.

### 같이 확인된 것: 제어 주기가 목표에 못 미침

이 머신에서 SmolVLA 추론 1회는 약 **150ms**(≈6.5Hz)입니다. `fps=30`으로 두면 매
루프가 약 120ms씩 밀리므로, 실제 제어 주기는 설정값이 아니라 추론 속도가 결정합니다.
주기가 들쭉날쭉한 것 자체가 jerk의 원인이므로 아래 중 하나를 선택하세요.

- `fps`를 실제 달성 가능한 값(6~8)으로 낮춰 주기를 일정하게 만든다. 앙상블은 유지됨.
- `infer_every`를 키워 추론 빈도를 낮춘다. 대신 앙상블 효과를 잃는다.

GUI는 루프가 밀리면 `[TIMING]` 경고를 로그에 남깁니다.

## 도구

### 1. GUI — `scripts/11__infer_gui.sh`

```bash
./scripts/11__infer_gui.sh
```

RViz 궤적은 별도 터미널에서 먼저 띄워야 보입니다:

```bash
ros2 launch agx_arm_description display_piper.launch.py
```

- **smoothing 파라미터를 실행 중에 바꿀 수 있습니다** — 슬라이더 조정 후 "현재 설정을
  실행 중인 추론에 적용".
- 큰 빨간 **E-STOP** 버튼(단축키 `Esc`)이 명령 전송을 즉시 끊습니다.
- 실시간 그래프에 최근 300 스텝의 정규화 action 목표가, 하단에 TV / max_step /
  RMS jerk가 계속 갱신됩니다.
- 종료 시 궤적과 지표가 `outputs/infer_gui/run_<timestamp>.{npz,json}`에 저장됩니다.

기본값은 `source=dataset`(로봇 미연결)입니다. 실물 전송은 `source=robot` 선택 +
"실물 Piper에 전송" 체크 + 확인 문구 `I_UNDERSTAND_REAL_ROBOT` 입력을 모두 해야
켜지고, 모든 명령은 `PiperFollower.send_action()`을 지나가므로
`max_relative_target`과 effort 안전 컷오프가 그대로 적용됩니다.

### 2. 파라미터 스윕 — `scripts/tools/piper_smoothing_sweep.py`

하드웨어 없이 `m` 값을 정할 때 씁니다. GUI와 같은 실행 코드를 쓰되 RViz/로봇 없이
dataset observation만 사용합니다.

```bash
python scripts/tools/piper_smoothing_sweep.py \
    --policy-path outputs/train/smolvla_erase_shape_512/checkpoints/030000/pretrained_model \
    --dataset-root records/0727/erase_the_shape_512 \
    --steps 40 --m-values 0.01 0.1 0.3 1.0
```

설정마다 자식 프로세스를 새로 띄웁니다 — 한 프로세스에서 SmolVLA를 두 번 로드하면
CUDA 컨텍스트가 깨지면서 죽기 때문입니다.

### 3. 단위 테스트

```bash
python -m pytest scripts/tools/test_action_smoothing.py
```

## 참고: LoRA-SP 저장소에서 확인한 것

[dhkim-furiosa/LoRA-SP](https://github.com/dhkim-furiosa/LoRA-SP)를 읽고 확인한
내용입니다. 위 구현은 이걸 참고해 우리 스택에 맞게 새로 쓴 것이고, 저쪽 저장소에는
아무것도 반영하지 않았습니다.

- `common/utils/utils.py::get_current_action`이 같은 `exp(-m·i)` 가중평균을 씁니다.
  다만 호출부가 전부 기본값 `m=1.0`을 쓰고 있어서 실질적인 스무딩이 거의 안 걸리고,
  `.cuda()`가 하드코딩되어 CPU에서는 동작하지 않습니다.
- `temporal_ensemble` 기본값이 `False`입니다.
- `scripts/eval_real_time_smolvla.py`에는 앙상블도 chunk 재추론도 없어서
  `select_action()` 결과가 바로 로봇으로 갑니다 — pi0 경로와 SmolVLA 경로의 smoothing
  조건이 서로 달라, 그 저장소 기준으로 백본을 비교하면 조건이 어긋납니다.

우리 실험을 저쪽 코드로 돌릴 일이 생기면 이 세 가지를 먼저 맞춰야 합니다.

### 논문에 쓸 때

smoothing은 모델의 기여가 아니라 **평가 프로토콜**입니다. 모든 baseline
(LoRA / LoRA-MoE / AdaLoRA)에 동일 설정을 적용하고 `(temporal_ensemble, m, alpha,
rate limit, 실제 제어 주기)`를 명시해야 합니다. 위 TV / RMS jerk를 함께 보고하면
"성공률뿐 아니라 예측 일관성도 낫다"는 주장을 정량적으로 뒷받침할 수 있습니다.
