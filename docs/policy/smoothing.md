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

### 정정: `fps=6`은 틀린 결론이었다 (2026-08-04, 실물 확인)

**이전 판에서 "추론이 1회 ~150ms라 `fps=6`이 한계"라고 적었는데, 전제가 틀렸습니다.**
실물에서 돌려보니 팔이 눈에 띄게 느리고 뚝뚝 끊겼습니다. 원인은 두 가지입니다.

**1. 재생 배속.** 정책은 "다음 프레임의 action"을 예측합니다. 학습 데이터가 30fps인데
명령을 5.55Hz로 쏘면 시연 동작이 그대로 **5.4배 슬로모션**이 됩니다.

**2. 명령 주파수 자체가 끊김을 만든다.** 5.5Hz면 팔은 180ms마다 목표를 하나 받고, 그
사이엔 컨트롤러가 움직이다 멈춥니다. 이 "움직였다 멈췄다"가 반복되면서 **끊김이
진동으로 느껴집니다.** `action_smoothing.py`가 줄이는 TV는 *명령 시퀀스*의 매끄러움일
뿐, 명령 주파수가 낮으면 궤적이 아무리 매끄러워도 물리적으로는 티가 안 납니다.

애초에 이 문서의 출발점이었던 "추론 시 팔이 흔들린다"는 증상이, 궤적 노이즈가 아니라
**명령 주파수 문제**였습니다.

**해결: 추론 주기와 명령 주기를 분리합니다.** chunk 하나가 이미 `chunk_size`(=50)스텝
분량을 담고 있으므로 매 스텝 추론할 필요가 없습니다. `infer_every`로 추론 빈도만 낮추면
명령은 계속 30Hz로 쏠 수 있습니다.

| | 이전 (`fps 6`) | 현재 기본값 (`fps 30, infer_every 5`) |
| --- | ---: | ---: |
| 실측 명령 주파수 | 5.55Hz | 18.07Hz |
| 재생 배속 | 5.4배 느림 | 1.7배 느림 |
| ensemble 표수 | 50 | 10 |

추론 간격이 167ms라 115ms 추론이 여유 있게 들어갑니다(이 머신 RTX 5090 기준 추론
1회 **112~118ms** — 이전에 적힌 150ms보다 빠릅니다).

`infer_every`를 키우면 앙상블 효과를 잃는다고 적었던 것도 과장이었습니다. 표수는
`chunk_size / infer_every`이므로 `infer_every=5`에서도 10표가 남습니다. 3표 미만으로
떨어질 때만 runner가 경고합니다.

30Hz를 다 못 채운 건 dataset 소스의 **비디오 디코딩**(스텝당 ~55ms) 때문입니다.
`source=robot`에서는 그 자리를 카메라 캡처가 대신하므로 값이 달라집니다.

명령 주파수가 학습 fps보다 낮으면 runner가 `[WARN]`으로 배속과 권장값을 알려주고,
루프가 밀리면 `[TIMING]` 경고를 남깁니다.

## 실물에서 실제로 흔들림을 만든 것들 (2026-08-04)

랩 Piper로 끝까지 추적한 결과입니다. **처음 가정과 달리 궤적 스무딩은 원인의 일부일
뿐이었고**, 큰 것들은 전부 제어 인터페이스 쪽에 있었습니다. 순서대로 하나씩 제거했고
각 단계에서 실측했습니다.

| # | 원인 | 증상 | 조치 |
| --- | --- | --- | --- |
| 1 | **명령 주파수 5.5Hz** | 30fps 학습 동작이 5.4배 슬로모션 + 뚝뚝 끊김 | `fps=30, infer_every=5` |
| 2 | **클램프 포화** | 스텝 200부터 100% 클램프 — 명령이 `실측+5`로 대체돼 스무딩이 통째로 버려짐 | `max_relative_target`을 smoothing rate limit과 분리, 15로 |
| 3 | **추론이 루프를 막음** | 33ms×4 → 115ms×1 반복, 6Hz 주기 끊김 | 추론을 별도 스레드로 (실측 5.55 → 28.6Hz) |
| 4 | **MOVE J 재계획** | 33ms마다 점대점 궤적을 새로 계획 | MIT(임피던스) 제어로 전환 |
| 5 | **vel_ref가 노이즈** | 유한차분을 그대로 미분해 부호 반전 28.7% | EMA α=0.2로 다듬음 |
| 6 | **위치 명령 지터** | 명령이 스텝의 **33.7%**에서 방향을 뒤집음 | smoothing의 EMA 단계를 켬 (α=0.2 → 5.8%) |

6번이 마지막 결정타였습니다. 파이프라인은 `ensemble → EMA → rate limit` 3단인데
**`ema_alpha`가 1.0(꺼짐)이라 가운데 단이 내내 놀고 있었습니다.** MOVE J일 때는
점대점 플래너가 그 지터를 뭉개줘서 드러나지 않았지만, MIT는 충실한 추종기라 그대로
재현합니다 — MIT로 바꾸자 오히려 더 떨렸던 이유가 이것입니다.

α=0.2에서 **방향 반전 33.7% → 5.8%, 이동폭은 42.57 → 42.28로 거의 그대로**입니다.
실제 움직임은 안 깎이고 지터만 걷힙니다.

### MIT(임피던스) 제어

`JointMitCtrl`로 `pos_ref` + `vel_ref` + `kp/kd`를 스트리밍합니다. 궤적 재계획이
없어 30Hz 스트리밍에 맞습니다. 기본 꺼짐이고 확인 문구가 따로 필요합니다.

관절별 게인이 필요합니다 — 정상상태 오차 = 중력토크 / kp 이므로 중력 부담이 다른
관절에 같은 kp를 주면 처짐이 제각각이 됩니다. 실측(뻗은 자세, 제자리 유지):

| 관절 | kp=10 처짐 | 조정 후 |
| --- | ---: | ---: |
| joint1 / joint4 / joint6 | 0.00 | — |
| joint5 | 0.17 | — |
| joint3 | 0.72 | kp=20에서 0.57 |
| joint2 | 1.83 (≈1.65°) | kp=30에서 0.47 |

joint3는 kp를 2배 올려도 1.26배만 줄었습니다 — kp에 비례하지 않는 마찰/데드밴드
성분이 섞여 있습니다.

MIT를 켜기 전에 `scripts/12__mit_probe.sh`로 관절 하나씩 확인하세요. probe가 끝날
때마다 파킹하므로 `--goto` 없이 실행하면 항상 접힌 자세(중력 모멘트≈0)에서 재게 되어
결과가 무의미합니다.

### 시도했지만 안 된 것: MOVE CPV

`ModeCtrl`의 `move_mode=0x05`(연속 위치-속도)는 펌웨어 S-V1.8-2에서 지원되지만,
실제로 바꾸니 **팔이 전혀 움직이지 않았습니다**(클램프 100% 포화 = 실측 위치 제자리).
`piper_sdk`의 `JointCtrl`은 위치만 보내는데 CPV는 속도 setpoint까지 필요한 것으로
보이고, SDK에 그걸 보내는 API가 없습니다.

## 어디에 들어가 있나

제어 루프는 `scripts/tools/piper_infer_runner.py` 한 곳에 있고, GUI·teleop_ui의
`Infer` 프리셋·CLI가 전부 그걸 씁니다. 어느 경로로 돌리든 같은 smoothing과 같은
안전 게이트가 걸립니다.

| 경로 | 실행 | temporal ensemble |
| --- | --- | --- |
| teleop_ui `Infer` 프리셋 | `piper_infer_runner.py` | O |
| 추론 GUI (`11__infer_gui.sh`) | 같은 runner | O |
| CLI (`piper_infer_runner.py`) | 자기 자신 | O |
| Record / Replay | `lerobot-record` / `lerobot-replay` | X (EMA만) |

`Infer` 프리셋은 예전에 `lerobot-record --policy.path=...`를 그대로 띄웠지만, 그
경로는 action chunk를 노출하지 않아 ensemble을 걸 수 없었습니다. 위 표에서 유일하게
TV를 절반으로 줄인 항목이 ensemble이라 제어 루프를 우리가 들도록 바꿨습니다. runner는
lerobot을 우회하지 않습니다 — `LeRobotDataset` / `make_policy` / `PiperFollower`를
그대로 쓰고, Record와 Replay는 손대지 않았습니다.

### chunk가 없는 경로: `send_action` EMA

`PiperFollowerConfig.action_ema_alpha`(기본 `1.0` = 꺼짐)를 1보다 작게 두면
`send_action()` 단계에서 EMA가 걸립니다. 텔레옵·Record·Replay 등 chunk 개념이 없는
경로에도 적용되는 바닥입니다. 안전 클램프(`max_relative_target`)보다 **먼저** 걸리므로
스무딩이 안전 제한을 넘길 수 없습니다.

rate limit은 여기 따로 두지 않았습니다 — `max_relative_target`이 이미 스텝당 최대
변화량을 제한하고 있어 중복입니다.

## 모드 (시연용 / 증강용)

모드는 **프리셋일 뿐**입니다. 고르면 아래 값이 세팅되지만 전부 화면에 그대로 보이고
개별로 덮어쓸 수 있습니다 — 논문에 실행 조건을 그대로 옮겨 적어야 하므로 값을 모드
뒤에 숨기지 않습니다.

| | 시연용 (`demo`) | 증강용 (`augment`) |
| --- | --- | --- |
| dataset 기록 | X | O |
| 카메라 저장 | — | 크롭 전 원본 |
| 종료 시 | 궤적/지표 npz | 성공/실패 질문 + 기록 |
| smoothing | `m=0.01`, rate≤5 | 동일 |

증강용에서 지키는 것:

- 기록되는 `action`은 정책 raw 출력이 아니라 **스무딩을 거쳐 실제로 `send_action()`에
  넘어간 값**입니다. 그래야 영상과 움직임의 인과가 맞아 BC 학습에 쓸 수 있습니다.
  raw chunk는 학습 호환성을 깨지 않도록 dataset feature가 아니라 옆의
  `raw_actions_ep*.npz`에 따로 남깁니다.
- 카메라는 정책에 먹인 512 크롭이 아니라 **원본 프레임**을 저장합니다. 기존 Record
  경로와 같은 형태라 `prepare_erase_shape_dataset.py`로 똑같이 학습용 변환을 돌릴 수
  있습니다.
- 실행 조건(스무딩 파라미터 전부 + **실측** 제어 주기)이 `rollout_meta.json`에
  남습니다. 논문에 그대로 인용하면 됩니다.

**주의 1 — 실패 롤아웃.** 정책 롤아웃에는 실패가 섞입니다. 걸러내지 않고 학습시키면
자기 실수를 복제합니다. 종료 시 성공/실패를 묻고 `rollout_meta.json`에 남기니, 학습에
쓰기 전에 `qc_studio.py`로 한 번 거르세요.

**주의 2 — 인과 번짐.** temporal ensemble을 켜고 기록하면 `action_t`가 과거 스텝의
chunk 예측에도 영향을 받습니다. 관찰↔행동 인과가 한 스텝 수준에서 살짝 번지므로,
순수한 BC 데이터가 필요하면 ensemble을 끄고 기록하거나 `raw_actions_ep*.npz`를 쓰세요.

**주의 3 — fps.** 제어 주기가 6Hz면 기록된 dataset의 fps도 6입니다. 30fps로 녹화한
기존 데이터와 섞을 때 이 차이를 어떻게 다룰지는 학습 시점의 문제로 남습니다. 설정
fps와 실측이 15% 넘게 벌어지면 runner가 경고를 남깁니다.

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

### 2. CLI — `scripts/tools/piper_infer_runner.py`

GUI 없이 같은 루프를 돕니다. teleop_ui의 `Infer` 프리셋이 조립하는 커맨드도 이겁니다.

```bash
python scripts/tools/piper_infer_runner.py \
    --mode augment \
    --dataset-root records/0727/erase_the_shape_512 \
    --policy-path outputs/train/smolvla_erase_shape_512/checkpoints/030000/pretrained_model \
    --fps 6
```

모드 기본값은 개별 플래그로 덮어쓸 수 있습니다 (`--no-record`, `--ensemble-m 0.3`,
`--no-ensemble` 등). 실물 전송은 `--source robot --apply-to-robot
--real-robot-confirm I_UNDERSTAND_REAL_ROBOT` 셋을 모두 줘야 열립니다.

### 3. 파라미터 스윕 — `scripts/tools/piper_smoothing_sweep.py`

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

### 4. 단위 테스트

```bash
python -m pytest scripts/tools/test_action_smoothing.py scripts/tools/test_infer_runner_mock.py scripts/tools/test_action_ema_mock.py
```

각각 smoothing 수식, runner의 모드/기록 로직, `send_action` EMA를 하드웨어 없이
검증합니다.

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
