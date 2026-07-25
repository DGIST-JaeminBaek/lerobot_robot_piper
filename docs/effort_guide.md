# effort 로깅 — 연구 기여, 사용법, 고려사항

> 작성: 2026-07-25 · 기준 코드: `origin/seongil/gui-refactor` (커밋 `a975b32`) · lerobot 0.4.4
> 이 문서의 수치는 전부 실측/코드 확인값이다. 추정치는 "추정"이라고 표시했다.

---

## 1. 왜 effort인가 — 연구 기여

### 문제

우리 태스크(지우개로 보드 도형 지우기)는 **contact-rich**다. 성공/실패를 가르는 건 "지우개가 보드에 **얼마나 세게** 닿아 있는가"인데, **RGB 카메라는 접촉력을 볼 수 없다.** 영상만 보면 살짝 스치는 것과 꾹 누르는 것이 거의 같아 보인다.

실제로 이 문제가 사고로 나타났다 — 리플레이 중 팔 끝에 힘이 과하게 실려 팔이 뻗는 상황이 발생했다. 정책이 "이 자세로 가라"만 배우고 "이만큼의 힘으로"를 못 배웠기 때문이다.

### 우리 접근

TA-VLA(CoRL 2025)를 따라 **관절별 effort를 observation에 넣어** 모델이 접촉 구간을 인지하게 한다. 팀 연구 주제가 "효율적 파인튜닝으로 VLA 성능 향상"이므로, TA-VLA의 *"effort를 어디에 넣을까"* 질문에 **"얼마나 적은 파라미터로 넣을까"**(Full FT vs LoRA vs frozen+adapter) 축을 추가한 것이 우리 차별점이다.

### ablation 표에서의 위치

| # | 조건 | effort | 이 데이터가 없으면 |
|---|---|---|---|
| 1 | Baseline | ✗ | — |
| 3 | +Effort (STATE) | STATE | **불가능** |
| 4 | +Effort (decoder-token) | 이력→토큰 | **불가능** |
| 5 | +Aux torque 예측 | 토큰+보조예측 | **불가능** |

**3~5번이 전부 effort를 전제한다.** 안 찍으면 실험 자체가 성립하지 않고, 우리 연구의 차별점이 사라진다.

### 결정적인 비대칭

```
찍는 비용    : 7 MB (전체 수집분), 인코딩 시간 영향 0, 성능 영향 없음
안 찍는 비용 : 150 에피소드 전량 재수집 (수 시간 + 조명·마모 등 조건 재현 불가)
```

그리고 **effort를 찍어두면 baseline(1번, effort 없음)도 같은 데이터에서 만들 수 있다.** `observation.state`를 앞 7개로 잘라내는 변환이면 되고, lerobot에 `modify_features()`가 있다(안 되면 parquet 직접 조작 — `smooth_start_frames.py`에서 이미 쓴 패턴). **한 번 수집으로 1번과 3번을 둘 다 커버한다.** 반대 방향은 불가능하다.

→ **결론: 무조건 켠다.** 고민할 사안이 아니다.

---

## 2. 무엇이 어떻게 저장되나

### 값의 정체

```python
GetArmHighSpdInfoMsgs().motor_N.effort * 0.001   # 단위 N·m
```

`piper_sdk`의 **전류 기반 고정계수 변환값**이다. **진짜 토크 센서가 아니다.** (§4 한계 참고)

`use_effort=true`면 **velocity(rad/s)도 같이** 기록된다. 외력 추정 시 자세·속도로 인한 effort 성분을 걸러내려면 같은 타임스탬프의 속도가 필요하기 때문이다. 그리퍼는 SDK에 `motor_speed`가 없어 vel이 없다.

### 저장 위치 — 비디오가 아니라 parquet

실제로 데이터셋을 만들어 확인한 구조:

```
dataset_root/
├── meta/info.json      ← 스키마 원본 (머지 가능 여부를 결정)
├── data/chunk-000/file-000.parquet    ← effort는 여기
└── videos/…                            ← effort는 여기 없음
```

parquet 컬럼:

```
action             fixed_size_list<float>[7]     ← pos만 (effort 안 섞임)
observation.state  fixed_size_list<float>[20]    ← pos 7 + effort 7 + vel 6
timestamp / frame_index / episode_index / index / task_index
```

**effort는 `observation.effort` 같은 별도 컬럼이 아니다.** lerobot의 `hw_to_dataset_features()`가 float 타입 observation을 전부 `observation.state` 한 벡터로 합치기 때문이다. 순서는 `meta/info.json`의 `names`에 남는다:

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

---

## 3. 켜는 법

### 설정

`configs/recording.env`:

```bash
USE_EFFORT=true
```

**기본값이 `true`이므로 이 줄이 없어도 켜진다.** 끄려면 명시적으로 `false`를 적어야 한다.

### 이전에 있던 함정 3가지 (이번 패치로 전부 수정됨)

아래 셋은 모두 **조용히 실패**하는 유형이었다 — 에러가 안 나서 데이터를 열어보기 전엔 몰랐다. 구조를 이해해두면 비슷한 문제를 빨리 찾을 수 있어 기록으로 남긴다.

| # | 증상 | 원인 | 수정 |
|---|---|---|---|
| 1 | `5__record.sh`로 녹화하면 effort가 안 들어감 | `run_common.sh`가 `--robot.use_effort`를 안 넘겨 config 기본값(`False`)이 쓰임 | `robot_observation_args()` 추가 후 `5__record.sh`·`9__run_client.sh`에 배선 |
| 2 | 에피소드 초반 100프레임 effort가 망가짐 | `smooth_start_frames.py`가 이름을 `.`으로 잘라 매칭 → `joint2.effort` → `parking["joint2"]=-100` 이 써짐 | `.pos` 컬럼만 보간하도록 마스크 적용 |
| 3 | 체크박스를 켜도 effort가 안 들어감 | `use_effort_var`가 Command 자동 갱신 대상에서 빠져 있어 옛 커맨드로 Launch됨 | `trace_add` 목록에 `use_effort_var`/`use_depth_var` 추가 |

**함정 2가 특히 위험했다.** Smooth Start는 기본 ON(100프레임)이라, effort를 켜기만 하면 30fps 기준 **모든 에피소드의 초반 3.3초**가 오염됐다. 지금은 `.pos`만 보간하므로 effort/vel은 원본 그대로 남는다 (`scripts/tools/test_smooth_start_mock.py`가 이 계약을 고정한다).

> **이전 코드를 쓰는 환경이라면** 위 회피책이 필요하다: 셸 대신 GUI 사용 + `SMOOTH_START_FRAMES=0` + 체크박스 변경 후 Preset 재선택.

### 검증 스크립트

```bash
python scripts/tools/check_effort.py <데이터셋경로>
```

확인 항목:
- `observation.state` 차원이 20인지 (7이면 effort 미반영)
- effort 값이 **실제로 변하는지** (차원만 맞고 전부 0인 경우를 잡는다 — CAN을 못 읽으면 이렇게 된다)
- 관절별 `|effort|` 최댓값 → `SAFETY_EFFORT_LIMIT` 튜닝 근거로 바로 제시
- Smooth Start가 초반 프레임을 덮어썼는지 (§3 함정 2)

---

## 4. 고려사항 / 한계

### ① 진짜 토크 센서가 아니다 — 제일 중요한 한계

전류 기반 추정치라 **자세에 따른 중력·마찰 성분이 그대로 섞여 있다.** 접촉이 전혀 없어도 팔을 뻗은 자세와 접은 자세의 effort가 다르다.

→ "effort가 크다 = 접촉이 세다"가 **아니다.** 순수 외력을 뽑으려면 자세·속도 기반 보정 모델이 필요하고, 그래서 velocity를 같이 로깅한다. 그 보정은 별도 작업이다.

→ 논문에 쓸 때 "joint torque"가 아니라 **"current-derived effort estimate"**로 정확히 표기해야 한다.

### ② 정규화 문제

effort는 `observation.state` 안에 pos와 **한 벡터로 합쳐져** 있어서 **따로 정규화할 수 없다.** pos는 완만하고 effort는 스파이크가 있는데 같은 정규화를 받는다.

lerobot의 `FeatureType`에는 STATE/VISUAL/ENV/ACTION/REWARD/LANGUAGE만 있고 EFFORT가 없어서, 별도 정규화 키를 두려면 **lerobot 코어 수정이 필요하다.**

완화책: 랩 PC 로컬 lerobot 체크아웃에 `NormalizationMode.QUANTILES`/`QUANTILE10`이 있다. 스파이크 신호에는 `MEAN_STD`보다 이쪽이 적합할 수 있으니 학습 시 시도해볼 것. (설치된 버전에 실제로 있는지 먼저 확인)

### ③ 학습/추론 대칭성

**학습 데이터에 effort가 있으면 추론 때도 반드시 `use_effort=true`여야 한다.** 한쪽만 켜면 state 차원(20 vs 7)이 안 맞아 터진다.

7차원으로 학습한 기존 체크포인트를 돌릴 때만 `USE_EFFORT=false`로 내린다.

### ④ 데이터셋 머지 제약

여러 세션으로 나눠 찍은 데이터셋은 **`fps` / `robot_type` / `features`가 완전히 일치해야만** 합쳐진다 (`lerobot/datasets/aggregate.py`의 `validate_all_metadata`가 `ValueError`를 던진다).

→ **`USE_EFFORT`를 첫 에피소드 전에 정하고 수집 끝날 때까지 절대 바꾸지 말 것.** 중간에 바꾸면 그 전 데이터를 버려야 한다.

### ⑤ 안전 컷오프 임계값은 지우기 동작 기준으로 잡아야 한다

`SAFETY_EFFORT_LIMIT`(기본 8.0 N·m)은 effort가 이 값을 넘으면 명령을 보류하고 마지막 위치를 유지하는 실시간 안전장치다. `use_effort` 로깅 여부와 **무관하게 항상 동작**하며, Teleoperate/Record/Infer/Replay 전 경로에 적용된다.

**⚠️ 무접촉 자유운동 기준으로 잡으면 안 된다.** 지우기는 누르는 힘이 본질이라, 자유운동 노이즈 기준으로 잡으면 **정상 지우기 중에 팔이 멈춘다.**

올바른 순서:

1. `SAFETY_EFFORT_LIMIT=15.0`으로 **높게** 두고 파일럿 5 에피소드 (컷오프가 안 걸리게)
2. 그 데이터에서 정상 지우기 중 관절별 `|effort|` 최댓값 측정:
   ```python
   allv = np.stack([ds[i]["observation.state"].numpy()[eff] for i in range(len(ds))])
   print(np.abs(allv).max(axis=0))
   ```
3. 그 최댓값의 **1.5배**를 임계값으로 확정 → 본 녹화

이러면 "정상 지우기는 통과, 뻗는 건 차단"이 된다.

---

## 5. GUI로 검증하기 (실기 담당자용, 약 15분)

> 목표: **"effort가 실제로 데이터에 들어가는가"**를 본 수집 전에 확정한다.
> 로봇을 크게 움직일 필요 없다. 짧은 에피소드 2개면 충분하다.

### STEP 0 — 녹화 전 설정 (GUI 켜기 전)

`configs/recording.env`를 열어서:

```bash
USE_EFFORT=true              # 기본값이 true라 없어도 켜지지만 명시해두는 편이 안전
USE_DEPTH_OBSERVATION=false  # 이번 검증에서는 변수 줄이기
NUM_EPISODES=2
EPISODE_TIME_S=20
RESET_TIME_S=10
```

**Smooth Start는 이제 켜둬도 된다** — effort를 덮어쓰던 버그가 수정됐다(§3). 다만 이번 검증에서는 변수를 줄이려면 꺼도 무방하다.

### STEP 1 — CAN 연결

1. `scripts/0__launch_gui.sh` 실행
2. **CAN Setup** 패널 → `Detect` → leader/follower 인터페이스가 잡히는지 확인
3. `Init All`(또는 개별 Init)로 인터페이스 활성화

### STEP 2 — ⭐ effort가 실제로 읽히는지 먼저 확인 (녹화 없이)

**CAN Monitor** 패널 → `Start Monitor`

Joint Positions 표에 follower/leader 값이 갱신되면 CAN 통신은 정상이다.
(이 패널은 위치만 보여주고 effort는 안 보여준다. effort는 STEP 5에서 데이터로 확인한다.)

확인 후 **`Stop Monitor`를 반드시 누른다.** 모니터가 CAN을 잡고 있으면 녹화와 충돌할 수 있다.

### STEP 3 — ⭐⭐ Command 창 육안 확인 (제일 중요)

1. **Preset** 드롭다운에서 **`Record`** 선택
2. **Task** 칸에 영어로 입력 (예: `effort check`)
3. 화면 위쪽 **"Record Effort" 체크박스가 켜져 있는지** 확인
4. **Command 입력창을 좌우로 스크롤해서 아래 두 개를 눈으로 확인:**

```
--robot.use_effort=true          ← 이게 없으면 절대 Launch하지 말 것
--dataset.num_episodes=2
```

`use_effort=false`로 보이면 → 체크박스를 껐다 켜고 **Preset을 다시 선택**해서 재조립시킨다. 그래도 안 되면 `recording.env`를 고치고 GUI를 재시작한다.

> 이 한 단계가 전체 검증의 90%다. 여기서 확인 안 하고 녹화하면 20분을 버린다.

### STEP 4 — 짧게 2 에피소드 녹화

1. `Launch`
2. 카메라 warmup으로 20초 정도 응답이 없을 수 있다 — **정상이다.** 기다린다
3. 진행률 라벨에 `Recording episode 1/2`가 뜨면 시작된 것
4. 텔레옵으로 팔을 움직인다. **지우개를 보드에 실제로 눌러보는 동작을 반드시 포함할 것** — 접촉 없이 움직이면 effort가 중력 성분만 나와서 STEP 6 튜닝을 못 한다
5. 20초가 지나면 자동으로 Reset 구간 → 다시 20초 → 2 에피소드 완료 후 자동 종료
6. 종료되면 파킹 자세로 이동 + 토크 해제된다

**⚠️ 이번 검증에서는 `Auto-stop at Parking`을 켜지 말 것.** 현재 코드에는 2번째 에피소드부터 잘리는 버그가 있다. `End Episode (Save)` 버튼도 이번에는 안 눌러도 된다(시간 만료로 넘어가면 충분).

문제가 생기면 전체 로그가 `last_launch.log`에 남는다.

### STEP 5 — 데이터 검증

**Recording History** 패널에서 방금 만들어진 데이터셋 경로를 확인한 뒤:

```bash
python scripts/tools/check_effort.py <데이터셋경로>
```

**기대 출력:**

```
observation.state 차원 : 20
pos 7개 / effort 7개 / vel 6개
✅ effort 필드 존재
✅ effort 값이 실제로 변하고 있음
✅ 초반 프레임 effort 정상 (덮어쓰기 흔적 없음)
```

**실패 패턴별 원인:**

| 출력 | 원인 | 조치 |
|---|---|---|
| 차원 7, `effort 없음` | 플래그 미반영 | STEP 3 재확인. 셸 스크립트로 찍었는지도 확인 |
| `effort가 전부 0` | CAN에서 값을 못 읽음 | 팔 전원·CAN 연결 확인. 스키마만 맞고 데이터는 무용 |
| `초반 100프레임 선형` | Smooth Start 오염 | `SMOOTH_START_FRAMES=0` 확인 후 재녹화 |

### STEP 6 — SAFETY_EFFORT_LIMIT 결정

STEP 5 출력 마지막에 관절별 `|effort|` 최댓값과 권장값이 나온다.

**단, STEP 4에서 실제 지우기 동작(누르는 힘 포함)을 했을 때만 유효하다.** 허공에서만 움직였다면 그 값으로 임계값을 잡으면 안 된다 — 실제 지우기에서 팔이 멈춘다.

권장값을 `recording.env`의 `SAFETY_EFFORT_LIMIT`에 반영하고 본 수집을 시작한다.

### 검증 완료 판정

- [ ] Command 창에 `--robot.use_effort=true` 확인됨
- [ ] `check_effort.py`가 ✅ 3개 전부 출력
- [ ] 관절별 effort 최댓값이 물리적으로 말이 되는 범위 (수 N·m 수준)
- [ ] `SAFETY_EFFORT_LIMIT` 값 결정됨

---

## 6. 본 수집 전 최종 체크리스트

- [ ] `USE_EFFORT=true` (GUI 켜기 **전에** recording.env에)
- [ ] Launch 직전 Command 창에서 `--robot.use_effort=true` 육안 확인 (§5 STEP 3)
- [ ] GUI/셸 어느 쪽으로 찍든 동일 — 다만 **한 데이터셋 안에서 섞지 말 것**
- [ ] 첫 에피소드 후 `check_effort.py` 통과
- [ ] `SAFETY_EFFORT_LIMIT`을 파일럿 데이터로 튜닝 (§4 ⑤)
- [ ] `FPS` / 해상도 / `USE_DEPTH_OBSERVATION`도 같이 고정하고 수집 끝까지 유지 (§4 ④)

---

## 7. 미해결 / 다음 단계

- [x] ~~`run_common.sh`에 `--robot.use_effort` 추가~~ (완료)
- [x] ~~`smooth_start_frames.py` 마스크 패치~~ (완료)
- [x] ~~GUI 체크박스 Command 갱신~~ (완료)
- [ ] **랩 PC 실기 검증** — 위 전부 mock 검증만 됨 (§5 절차대로)
- [ ] effort 중력·마찰 보정 모델 (§4 ①)
- [ ] ablation 4~5번(decoder-token 주입, aux torque 예측)은 `modeling_smolvla.py` 수정 필요
- [ ] baseline(7차원) 변환 스크립트 — 학습 시작 전까지
