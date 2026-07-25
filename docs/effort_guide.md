# effort 활용 논문 조사 및 녹화 기능 구현

> 담당: 조성일 · 작성 2026-07-25 · 기준 코드 `seongil/gui-refactor` · lerobot 0.4.4
>
> **이 문서의 범위** — 내 담당 과제 두 가지 중 첫 번째다.
>
> 1. 로봇 팔 effort를 활용하는 논문 조사 및 effort 녹화 기능 구현 ← **이 문서**
> 2. 대표 VLA 모델의 FPS/이미지 해상도 조사 → `docs/vla_fps_resolution.md`
>
> 코드에서 확인한 값은 파일·줄 위치를 같이 적었다. 검증 안 된 것은 "미검증"으로 표시했다.

---

# Part A. 논문 조사

## A-1. 문제 정의 — 왜 force/torque인가

우리 태스크(지우개로 보드 도형 지우기)는 **contact-rich**다. 성공을 가르는 건
"지우개가 보드에 **얼마나 세게** 닿아 있는가"인데, **RGB 카메라로는 접촉력이 안 보인다.**
영상에서 살짝 스치는 것과 꾹 누르는 것이 거의 같아 보인다.

이 문제가 실제 사고로 나타났다 — 리플레이 중 팔 끝에 힘이 과하게 실려 팔이 뻗었다.
정책이 "이 자세로 가라"는 배웠지만 "이만큼의 힘으로"는 못 배웠기 때문이다.

VLA 분야에서 이건 알려진 공백이다. 최근 연구들이 공통적으로 지적하는 바:
현재 VLA는 미세한 물리적 피드백을 통합하는 능력이 없고, 접촉 상태에 따라 실행
전략을 바꾸지 못한다.

## A-2. 핵심 레퍼런스 — TA-VLA (CoRL 2025)

**TA-VLA: Elucidating the Design Space of Torque-aware Vision-Language-Action Models**
· [arXiv:2509.07962](https://arxiv.org/abs/2509.07962) · [GitHub](https://github.com/ZZongzheng0918/TA-VLA)

우리 연구의 직접적 기반이다. "torque를 VLA에 넣자"가 아니라 **"어떻게 넣어야 하는가"**를
체계적으로 탐색한 논문이라 설계 결정을 그대로 빌려올 수 있다.

### 세 가지 발견

| # | 질문 | 결론 |
|---|---|---|
| **1** | **어디에** 넣나 | torque 어댑터를 **decoder**에 넣는 것이 **encoder**보다 일관되게 우수 |
| **2** | **어떻게** 표현하나 | torque **이력의 단일 토큰 요약**(single-token summary of torque history) |
| **3** | 추가로 뭘 하나 | torque를 **보조 출력으로 같이 예측** — 상호작용 동역학의 물리적 표현을 학습하게 됨 |

10개 태스크에서 일관된 향상, 특히 **insert / plug / turn** 계열에서 두드러졌다.

> 우리 태스크(지우기)는 insert/plug만큼 정밀하진 않지만 **지속적 접촉 + 압력 조절**이
> 핵심이라는 점에서 같은 계열이다.

## A-3. 관련 연구 지형

| 연구 | 접근 | 우리와의 관계 |
|---|---|---|
| **ForceVLA** ([arXiv:2505.22159](https://arxiv.org/abs/2505.22159)) | 외력 감지를 VLA의 **1급 모달리티**로 취급. `FVLMoE` — force-aware Mixture-of-Experts로 시각-언어 임베딩과 실시간 6축 force를 액션 디코딩 단계에서 융합 | **6축 F/T 센서 전제.** 우리는 관절 전류 기반이라 그대로 못 씀. MoE 융합 아이디어는 참고 가능 |
| **FACTR** | force-attending **curriculum training** | 표 7번 "지름길 방지"의 근거. 학습 커리큘럼으로 force 의존을 유도 |
| **FoAR** | 고주파 force/torque + 시각 → 반응형 정책 | **고주파가 핵심.** 우리 30fps + open-loop 50스텝 구조와 충돌 (§C-1) |
| **FAVLA / FD-VLA / ForceFlow** (2026) | fast-slow 구조, force 증류, 접촉 기반 flow matching | 후속 조사 대상. 아직 미확인 |

**공통 흐름**: force/torque를 시각에 얹어 접촉 시 빠른 교정 행동을 만들되, 장기 실행의
안정성은 유지하는 방향. 대부분 **전용 F/T 센서**를 가정한다는 점이 우리와 다르다.

## A-4. ⭐ TA-VLA 발견을 우리 SmolVLA 코드에 대응시키기

이게 이 조사의 핵심 산출물이다. 설치된 `lerobot 0.4.4` 소스를 직접 읽고 대조했다.

### SmolVLA가 state를 처리하는 방식

`policies/smolvla/modeling_smolvla.py`:

```python
# embed_prefix() — 이미지 + 언어 + state
state = pad_vector(state, self.config.max_state_dim)   # 20차원 → 32로 패딩
state_emb = self.state_proj(state)                     # nn.Linear(32 → hidden)
embs.append(state_emb)                                 # 토큰 1개

# embed_suffix(noisy_actions, timestep) — 액션 전문가 쪽
```

`configuration_smolvla.py`:

| 항목 | 값 |
|---|---|
| `max_state_dim` | 32 |
| `chunk_size` / `n_action_steps` | 50 / 50 |
| `resize_imgs_with_padding` | (512, 512) |
| `freeze_vision_encoder` | True |
| `train_state_proj` | True |

### 대응 표

| TA-VLA 발견 | SmolVLA 현재 상태 | 판정 |
|---|---|---|
| **① decoder > encoder** | state는 `embed_prefix`(**encoder 쪽**)에 들어감 | ❌ **TA-VLA가 열등하다고 한 쪽** |
| **② 단일 토큰 요약** | state 전체가 이미 **토큰 1개** | ✅ 형태는 일치 |
| **② 이력(history)** | **현재 프레임 1개뿐**, 이력 없음 | ❌ 미충족 |
| **③ 보조 예측** | 없음 | ❌ 미구현 |

**중요한 함의:** 지금 우리 방식(`observation.state`에 effort를 섞는 것)은
**TA-VLA가 "더 나쁘다"고 결론 낸 encoder 배치**다. 그러니 ablation 3번이
baseline 대비 큰 향상을 안 보여도 놀랄 일이 아니다 — 그게 논문의 예측이다.
**3번은 "effort가 쓸모없다"의 근거가 아니라 4번(decoder 배치)과 비교하기 위한
기준선으로 읽어야 한다.**

또 하나: state 토큰 1개 안에 위치 7 + effort 7 + 속도 6이 **함께 압축**된다.
TA-VLA의 "단일 토큰"은 torque 전용 토큰이지, 다른 신호와 공유하는 토큰이 아니다.

---

# Part B. 녹화 기능 구현

## B-1. 무엇이 어떻게 저장되나

### 값의 정체

```python
# motors/piper_motors_bus.py: get_effort()
GetArmHighSpdInfoMsgs().motor_N.effort * 0.001   # 단위 N·m
```

`piper_sdk`의 **전류 기반 고정계수 변환값**이다. **진짜 토크 센서가 아니다** (§C-2).

`use_effort=true`면 **velocity(rad/s)도 같이** 기록된다. 외력을 추정하려면 자세·속도로
인한 성분을 걸러야 하는데, 그러려면 같은 타임스탬프의 속도가 필요하기 때문이다.
그리퍼는 SDK에 `motor_speed`가 없어 vel이 없다(관절 6개만).

### 저장 위치 — 비디오가 아니라 parquet

실제로 데이터셋을 생성해서 확인한 구조:

```
dataset_root/
├── meta/info.json      ← 스키마 원본 (머지 가능 여부를 결정)
├── data/chunk-000/file-000.parquet    ← effort는 여기
└── videos/…                            ← effort는 여기 없음
```

```
action             fixed_size_list<float>[7]     ← pos만 (effort 안 섞임)
observation.state  fixed_size_list<float>[20]    ← pos 7 + effort 7 + vel 6
timestamp / frame_index / episode_index / index / task_index
```

**`observation.effort` 같은 별도 컬럼이 아니다.** lerobot의 `hw_to_dataset_features()`가
float 타입 observation을 전부 `observation.state` 한 벡터로 합친다. 순서는
`meta/info.json`의 `names`에 남는다:

```
["joint1.pos", …, "gripper.pos",        # 0~6
 "joint1.effort", …, "gripper.effort",  # 7~13
 "joint1.vel", …, "joint6.vel"]         # 14~19
```

### 읽는 법

```python
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("local/erase_circle", root="/path/to/dataset")
names = ds.meta.info["features"]["observation.state"]["names"]
eff = [i for i, n in enumerate(names) if n.endswith(".effort")]

state = ds[0]["observation.state"].numpy()   # (20,)
effort = state[eff]                           # N·m, 7개
```

### 비용 (실측)

| | |
|---|---|
| 프레임당 | 52 바이트 (float32 × 13) |
| **150 에피소드 × 30초 전체** | **7 MB** (데이터셋의 0.1% 미만) |
| 인코딩 시간 영향 | **0** — parquet에만 들어가고 비디오 파이프라인을 안 거침 |
| 녹화 루프 부하 | CAN 읽기 1회 추가 (무시할 수준) |

### 결정적 비대칭

```
찍는 비용    : 7 MB, 인코딩 영향 0
안 찍는 비용 : 150 에피소드 전량 재수집 (수 시간 + 조명·마모 등 조건 재현 불가)
```

그리고 **effort를 찍어두면 baseline(effort 없음)도 같은 데이터에서 만들 수 있다** —
`observation.state`를 앞 7개로 자르면 된다. 반대 방향은 불가능하다.
→ **무조건 켠다.** 고민할 사안이 아니다.

## B-2. 켜는 법

`configs/recording.env`:

```bash
USE_EFFORT=true
```

**기본값이 `true`라 이 줄이 없어도 켜진다.** 끄려면 명시적으로 `false`를 적어야 한다.
GUI 체크박스("Record Effort")로도 켜고 끌 수 있다.

## B-3. 수정 내역 — 조용히 실패하던 경로 3개

`USE_EFFORT`를 켜도 effort가 안 찍히는 경로가 셋 있었다. **전부 에러 없이 실패**해서
데이터를 열어보기 전까지 알 수 없었다. (커밋 `2b7ad8b`)

| # | 증상 | 원인 | 수정 |
|---|---|---|---|
| 1 | `5__record.sh`로 녹화하면 effort 없음 | `run_common.sh`가 `--robot.use_effort`를 안 넘겨 config 기본값(`False`)이 쓰임 | `robot_observation_args()` 추가 → `5__record.sh`·`9__run_client.sh`에 배선 |
| 2 | 에피소드 초반 100프레임 effort 손상 | `smooth_start_frames.py`가 이름을 `.`으로 잘라 매칭 → `joint2.effort` → `parking["joint2"]=-100` 이 써짐 | `.pos` 컬럼만 보간하도록 마스크 적용 |
| 3 | 체크박스를 켜도 effort 없음 | `use_effort_var`가 Command 자동 갱신(`trace_add`) 대상에서 빠져 옛 커맨드로 Launch됨 | `trace_add` 목록에 `use_effort_var` 추가 |

**2번이 특히 위험했다.** Smooth Start는 기본 ON(100프레임)이라 effort를 켜기만 하면
30fps 기준 **모든 에피소드의 초반 3.3초**가 오염됐다. 지금은 `.pos`만 보간하므로
effort/vel은 원본 그대로다 (`scripts/tools/test_smooth_start_mock.py`가 이 계약을 고정).

**1번은 추론 경로(`9__run_client.sh`)에도 넣었다** — 학습/추론 사이 state 차원이
다르면 터지기 때문이다(§C-4).

> **이전 코드를 쓰는 환경이라면** 회피책이 필요하다: 셸 대신 GUI 사용 +
> `SMOOTH_START_FRAMES=0` + 체크박스 변경 후 Preset 재선택.

## B-4. GUI 검증 절차 (실기 담당자용, 약 15분)

> 목표: **"effort가 실제로 데이터에 들어가는가"**를 본 수집 전에 확정한다.
> 로봇을 크게 움직일 필요 없다. 짧은 에피소드 2개면 충분하다.

### STEP 0 — GUI 켜기 전 설정

```bash
USE_EFFORT=true              # 기본값이 true라 없어도 되지만 명시가 안전
REALSENSE_USE_DEPTH=false    # 검증에서는 depth OFF (변수 줄이기)
NUM_EPISODES=2
EPISODE_TIME_S=20
RESET_TIME_S=10
```

### STEP 1 — CAN 연결

1. `scripts/0__launch_gui.sh` 실행
2. **CAN Setup** → `Detect` → leader/follower 인터페이스 확인
3. `Init All`로 활성화

### STEP 2 — CAN 통신 확인 (녹화 없이)

**CAN Monitor** → `Start Monitor` → Joint Positions 값이 갱신되면 정상.
(이 패널은 위치만 보여준다. effort는 STEP 5에서 데이터로 확인한다.)

확인 후 **`Stop Monitor`를 반드시 누른다.** 모니터가 CAN을 잡고 있으면 녹화와 충돌한다.

### STEP 3 — ⭐ Command 창 육안 확인 (제일 중요)

1. **Preset** → **`Record`** 선택
2. **Task** 칸에 영어 입력 (예: `effort check`)
3. **"Record Effort" 체크박스**가 켜져 있는지 확인
4. **Command 입력창을 스크롤해서 확인:**

```
--robot.use_effort=true          ← 없으면 절대 Launch하지 말 것
--dataset.num_episodes=2
```

> 이 한 단계가 검증의 90%다. 여기서 확인 안 하면 20분을 버린다.

### STEP 4 — 짧게 2 에피소드

1. `Launch` → 카메라 warmup으로 20초쯤 응답 없음 (정상)
2. `Recording episode 1/2` 뜨면 시작된 것
3. 텔레옵으로 움직인다. **지우개를 보드에 실제로 눌러보는 동작을 반드시 포함할 것** —
   접촉 없이 움직이면 effort가 중력 성분만 나와서 STEP 6 튜닝을 못 한다
4. 20초 후 자동으로 Reset → 다시 20초 → 종료 시 파킹 + 토크 해제

**⚠️ `Auto-stop at Parking`은 켜지 말 것.** 현재 코드에 2번째 에피소드부터 잘리는
버그가 있다. 로그는 `last_launch.log`에 남는다.

### STEP 5 — 데이터 검증

**Recording History**에서 데이터셋 경로 확인 후:

```bash
python scripts/tools/check_effort.py <데이터셋경로>
```

기대 출력:

```
observation.state 차원 : 20
pos 7개 / effort 7개 / vel 6개
✅ effort 필드 존재
✅ effort 값이 실제로 변하고 있음
✅ 초반 프레임 effort 정상 (덮어쓰기 흔적 없음)
```

| 출력 | 원인 | 조치 |
|---|---|---|
| 차원 7, `effort 없음` | 플래그 미반영 | STEP 3 재확인 |
| `effort가 전부 0` | CAN에서 값을 못 읽음 | 팔 전원·CAN 확인. 스키마만 맞고 데이터는 무용 |
| `초반 100프레임 선형` | Smooth Start 오염 | 구버전 코드. `SMOOTH_START_FRAMES=0` |

### STEP 6 — SAFETY_EFFORT_LIMIT 결정

출력 마지막에 관절별 `|effort|` 최댓값과 권장값이 나온다.
**STEP 4에서 실제 누르는 동작을 했을 때만 유효하다** (§C-3).

---

# Part C. 고려사항 / 한계

## C-1. 🔴 `n_action_steps=50` — effort보다 먼저 해결해야 할 문제

```python
chunk_size: int = 50
n_action_steps: int = 50
```

모델이 액션 50개를 한 번에 뱉고 **그 50개를 다 실행할 때까지 새 관측을 안 본다.**
30fps 기준 **1.67초 완전 open-loop**다.

```
t=0.00s  effort 급상승 — 너무 세게 누름
t=0.00s  모델: 이미 50스텝 계획 확정. 못 바꿈
t=1.67s  드디어 새 관측
```

**"팔이 뻗는" 사고가 정확히 이 구조에서 나온다.** effort를 관측에 넣어도 반응할
기회가 1.67초 뒤다. FoAR 계열이 "고주파 force"를 강조하는 이유가 이것이다.

→ **`n_action_steps`를 줄이는 것이 effort를 쓸모있게 만드는 가장 큰 레버**이고,
**설정값 하나**다(아키텍처 수정 아님). `chunk_size=50`은 두고 `n_action_steps`만
줄이면 "50개 예측 후 앞 N개만 실행하고 재계획"이 된다.

| `n_action_steps` | 반응 주기 (30fps) |
|---|---|
| 50 (기본) | 1.67 s |
| 10 | 0.33 s |
| 5 | 0.17 s |

대가는 추론 횟수 증가다. 5090 ×2면 감당 가능할 것으로 보인다(미검증).

**표 7번("Chunk 축소")에 있는 항목이지만, 3번과 같이 가야 한다고 본다.** 안 그러면
3~5번 결과가 "effort는 도움 안 됨"으로 나오는데 원인은 effort가 아니라 반응 주기다.

## C-2. 진짜 토크 센서가 아니다

전류 기반 추정치라 **자세에 따른 중력·마찰 성분이 섞여 있다.** 접촉이 없어도 팔을
뻗은 자세와 접은 자세의 effort가 다르다.

```
측정 effort = 중력(자세) + 마찰(속도) + 관성(가속) + 접촉력
                                              └ 우리가 원하는 건 이것뿐 ┘
```

→ "effort가 크다 = 접촉이 세다"가 **아니다.**
→ 논문에는 "joint torque"가 아니라 **"current-derived effort estimate"**로 표기해야 한다.

### 🌟 개선 방향 — effort 잔차

접촉 없는 자유운동 데이터로 `τ̂(자세, 속도)`를 학습시킨 뒤 빼면 순수 접촉 신호에
훨씬 가까워진다.

```
잔차 = 측정 effort − τ̂(자세, 속도)
```

- **필요한 입력이 이미 다 로깅된다** (pos, vel, effort)
- **후처리라 나중에 해도 된다** — 재녹화 불필요
- 깔끔한 ablation: raw effort vs 잔차 effort
- 팀 주제("효율적 파인튜닝")와 맞음 — 모델을 키우는 대신 **입력 신호를 정제**

**⚠️ 단, τ̂ 캘리브레이션 데이터는 본 수집과 같은 조건이어야 한다.** 특히 지우개
페이로드가 중력 토크를 바꾼다. **지우개를 단 채로, 보드에 안 닿게 5~10분
자유운동을 본 수집 세션에 같이 찍어둘 것.** (task 이름: `free motion calibration`)

## C-3. 안전 컷오프는 지우기 동작 기준으로

`SAFETY_EFFORT_LIMIT`(기본 8.0 N·m)은 effort가 임계값을 넘으면 명령을 보류하고
마지막 위치를 유지한다. `use_effort` 로깅 여부와 **무관하게 항상 동작**하며
Teleoperate/Record/Infer/Replay 전 경로에 적용된다.

**⚠️ 무접촉 자유운동 기준으로 잡으면 안 된다.** 지우기는 누르는 힘이 본질이라
자유운동 노이즈 기준으로 잡으면 **정상 지우기 중에 팔이 멈춘다.**

1. `SAFETY_EFFORT_LIMIT=15.0`으로 높게 두고 파일럿 5 에피소드
2. 정상 지우기 중 관절별 `|effort|` 최댓값 측정 (`check_effort.py`가 출력)
3. 그 최댓값의 **1.5배**를 임계값으로 확정

## C-4. 학습/추론 대칭성

**학습 데이터에 effort가 있으면 추론 때도 `use_effort=true`여야 한다.** 한쪽만 켜면
state 차원(20 vs 7)이 안 맞아 터진다. 7차원으로 학습한 기존 체크포인트를 돌릴 때만
`false`로 내린다.

## C-5. 데이터셋 머지 제약

여러 세션 데이터셋은 **`fps` / `robot_type` / `features`가 완전히 일치해야만** 합쳐진다
(`lerobot/datasets/aggregate.py`의 `validate_all_metadata`가 `ValueError`).

→ **`USE_EFFORT`를 첫 에피소드 전에 정하고 수집 끝까지 바꾸지 말 것.**

## C-6. 정규화

effort는 스파이크성, 위치는 완만한데 **같은 `observation.state`라 같은 정규화**를 받는다.
lerobot `FeatureType`에 EFFORT가 없어 따로 떼려면 코어 수정이 필요하다.

완화: `NormalizationMode.QUANTILES`/`QUANTILE10`이 있으면 `MEAN_STD`보다 나을 수 있다.
(설치된 버전에 실제로 있는지 먼저 확인할 것)

## C-7. ⚠️ 지름길 학습 — 방법론상 최대 위험

effort를 넣으면 모델이 이런 지름길을 배울 수 있다:

> "지금 effort를 보니 팔이 이 방향으로 움직이는 중 → 하던 대로 계속"

**접촉을 이해한 게 아니라 현재 동작을 읽은 것**이다. 손실은 잘 떨어지고 지표도
좋아 보이는데 실제로는 아무것도 안 배운 상태다. FACTR의 커리큘럼이 겨냥하는 문제.

**검증법**: effort를 0으로 채워 추론했을 때 성능이 거의 안 떨어지면 → 애초에 안
쓰고 있던 것. 많이 떨어지면 → 쓰고는 있으나 지름길인지 접촉 이해인지는 추가 확인 필요.

완화: effort dropout, 더 먼 미래 예측, 보조 예측 과제(TA-VLA 발견 3).

## C-8. 관측 ≠ 제어

**effort를 넣는 건 "인식" 개선이지 "제어"가 아니다.** SmolVLA는 위치 목표를 출력하므로
모델이 "너무 세게 누르고 있다"를 알아도 **힘을 직접 낮출 수단이 없다.** 위치를 바꿔
간접적으로 할 뿐이다.

진짜 힘 제어는 MIT 모드 컴플라이언스(표 8~9번) 영역이다. **논문에서 이 구분을
흐리면 안 된다.**

## C-9. 평가 지표

성공률만으로는 150 에피소드에서 노이즈에 묻힐 수 있다. **접촉 품질을 직접 재는 지표**를
같이 둘 것:

- 지우는 동안 peak effort
- 안전 컷오프 발동 횟수
- 지워진 면적 비율

effort의 효과는 "성공/실패"보다 이런 데서 먼저 보인다.

---

# Part D. 지금 정할 것 vs 나중에 해도 될 것

| | 녹화 시점에 필수 | 나중에 가능 |
|---|:---:|:---:|
| `USE_EFFORT=true` | ✅ | ❌ |
| velocity (effort에 묶여 자동) | ✅ | ❌ |
| fps / 해상도 | ✅ | ❌ |
| τ̂ 자유운동 데이터 | ⚠️ 같은 세션 권장 | △ 조건부 |
| | | |
| 잔차 계산 (§C-2) | | ✅ |
| 정규화 방식 (§C-6) | | ✅ |
| `n_action_steps` 조절 (§C-1) | | ✅ |
| decoder-token 배치 (§A-4) | | ✅ |
| 보조 예측 손실 | | ✅ |
| baseline(7차원) 생성 | | ✅ |

**연구 방향은 전부 오른쪽 열이다.** 녹화 때는 왼쪽만 지키면 된다.

---

# Part E. 다음 단계

- [x] effort/velocity 로깅 구현 (`a975b32`)
- [x] 실행 경로 3개 수정 (`2b7ad8b`)
- [x] 검증 스크립트 `check_effort.py`
- [x] TA-VLA 조사 및 SmolVLA 코드 대응 (§A-4)
- [ ] **랩 PC 실기 검증** — 위 전부 mock 검증만 됨 (§B-4 절차대로)
- [ ] `n_action_steps` 축소 실험 (§C-1) ← 설정만으로 가능, 우선순위 높음
- [ ] ablation 1 vs 3 (baseline vs effort-in-state)
- [ ] effort 잔차 모델 (§C-2) ← 고유 기여 후보
- [ ] decoder-token 배치 (§A-4, TA-VLA 발견 1) — `modeling_smolvla.py` 수정 필요
- [ ] 보조 torque 예측 (TA-VLA 발견 3)
- [ ] FAVLA / FD-VLA / ForceFlow 등 2026 후속 연구 조사

---

## 참고 문헌

- TA-VLA: Elucidating the Design Space of Torque-aware Vision-Language-Action Models (CoRL 2025) — [arXiv:2509.07962](https://arxiv.org/abs/2509.07962) · [GitHub](https://github.com/ZZongzheng0918/TA-VLA) · [OpenReview](https://openreview.net/forum?id=HAmi1X11BO)
- ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation — [arXiv:2505.22159](https://arxiv.org/abs/2505.22159)
- FACTR: Force-attending curriculum training for contact-rich policy learning
- FoAR: Force-Aware Reactive Policy
