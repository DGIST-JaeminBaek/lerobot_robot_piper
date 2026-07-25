# TA-VLA / Depth / Effort 통합 — 진행 상황 및 다음 방향

> ⚠️ **낡은 기록 (2026-07-24 시점).** 이후 depth 구현이 jmbaek의 12-bit 로그 양자화
> 방식으로 교체되면서 여기 적힌 turbo 컬러맵 관련 내용(`depth_utils.py`,
> `USE_DEPTH_OBSERVATION`, `DEPTH_MIN_M/MAX_M`, "Record Depth" 체크박스,
> `test_depth_mock.py`)은 전부 제거됐다. 현재 상태는 `docs/depth_guide.md`와
> `docs/effort_guide.md`를 볼 것. 이 문서는 당시 판단 근거로만 남긴다.
>
> 최종 갱신: 2026-07-24. 관련 선행 자료: `tavla-depth-force-integration.md`(다운로드 원본 가이드).
> 이 문서는 그 가이드의 `TODO(확인)` 항목들을 실제 리포 + 랩 PC 기준으로 검증하고,
> 이후 진행한 코드 작업과 앞으로 남은 일을 정리한다.

## 1. 연구 방향 요약

- **베이스**: SmolVLA + LeRobot, PiPER 로봇팔, 지우개로 보드마카 도형 지우기 태스크.
- **핵심 문제의식**: RGB만으로는 접촉력을 볼 수 없다. TA-VLA(CoRL 2025)를 따라 관절 effort를
  observation에 넣어 contact-rich 구간(접촉 순간/과압)을 모델이 인지하게 한다.
- **연구 주제와의 접점**: "효율적 파인튜닝으로 VLA 성능 향상"이 팀 주제이므로, TA-VLA의
  "effort를 어디에 넣을까" 질문에 "얼마나 적은 파라미터로 넣을까"(Full FT vs LoRA vs
  frozen+adapter-only) 축을 하나 더 추가한 것이 우리만의 차별점.
- **성능 우선 트랙**: 주제와 다소 어긋나더라도, PiPER SDK가 실제로 지원하는 관절별
  MIT 모드(`JointMitCtrl`, kp/kd/t_ref)를 활용한 컴플라이언스 제어는 태스크 성공률에
  직접 기여할 수 있어 별도로 계속 탐색하기로 함(아래 ablation 표의 성능 상한 실험).

## 2. 하드웨어/환경 검증 결과 (랩 PC `ugrp43`, conda env `ugrp`)

| 항목 | 결과 |
|---|---|
| GPU | RTX 5090 (32GB) × 2 — Full FT 배치 32+ 여유 있음, LoRA는 메모리가 아니라 소량 데이터(100~300 에피소드) 과적합 방지 목적으로 검토 |
| `piper_sdk` | 0.6.1 (로컬 Mac과 동일) |
| effort 읽기 | `GetArmHighSpdInfoMsgs().motor_N.effort` 존재 확인 — **current 기반 고정계수 변환값**(0.001 N·m 단위), 진짜 토크 센서 아님. 자세(중력/마찰)에 따라 접촉 없이도 값이 바뀔 수 있음 |
| MIT 모드 | `MotionCtrl_2(is_mit_mode=0xAD)` + `JointMitCtrl(motor_num, pos_ref, vel_ref, kp, kd, t_ref)` 실존 확인. kp 범위 [0,500], kd [-5,5], t_ref [-18,18] N·m |
| 전류/토크 상한 API | 없음. `MotorMaxAccLimitConfig`류(가속도 제한)만 존재 — 소프트웨어 전류 클램프는 불가, 물리적 컴플라이언스가 여전히 1순위 |
| `lerobot` | v0.4.0 (huggingface 공식, 로컬 소스 체크아웃) — `chunk_size=50`, `n_action_steps=50`, `max_state_dim=32` |
| `NormalizationMode` | `MIN_MAX/MEAN_STD/IDENTITY` + **`QUANTILES`/`QUANTILE10`**(로컬 pip판보다 많음) — effort처럼 스파이크가 있는 신호엔 QUANTILES가 MEAN_STD보다 적합할 수 있음 |
| `FeatureType` | STATE/VISUAL/ENV/ACTION/REWARD/LANGUAGE뿐, EFFORT 없음 — effort를 별도 정규화 키로 두려면 lerobot 코어 수정 필요 |
| `hw_to_dataset_features()` | **float 타입 observation feature를 전부 `observation.state`로 합침** — effort를 `observation_features`에 추가만 해도 자동으로 TA-VLA STATE(DePre) 방식이 됨. 별도 스키마 배선 불필요 |
| RealSense depth | `RealSenseCameraConfig(use_depth=True)`가 `config_piper.py`에 이미 배선되어 있었음. 단 `async_read()`는 color만 반환 — depth는 `latest_depth_frame`을 직접 읽어야 함(lerobot 코어 자체에 "Missing implementation for depth for now" 표시됨) |
| 데이터셋 경로 | CLAUDE.md의 `/home/ugrp308/Group43/datasets/`는 낡은 정보. 실제로는 `recording.env`의 `DATASET_ROOT`(기본 `~/UGRP/lerobot_robot_piper/records/...`)를 따름 — 문서 갱신 필요 |
| git remote | 랩 PC의 `lerobot_robot_piper` clone은 `origin`이 업스트림(`DGIST-JaeminBaek`)을 직접 가리킴. 앞으로 push할 때 remote 이름을 반드시 확인할 것(fork 추가 권장, 아직 안 함) |

## 3. 완료된 코드 작업 (이번 세션, `seongil/gui-refactor` 브랜치)

모두 **기본 OFF**, 옵션으로 켜고 끌 수 있게 구현. 개인 PC에는 하드웨어가 없어 전부 mock 기반으로만 검증했고, 랩 PC 실기 검증은 아직 안 했다.

| 파일 | 변경 내용 |
|---|---|
| `lerobot_robot_piper/config_piper.py` | `use_effort`, `use_depth_observation`, `depth_min_m`, `depth_max_m`, `depth_scale` 필드 추가 |
| `lerobot_robot_piper/motors/piper_motors_bus.py` | `get_effort()` — 관절 6개 + 그리퍼 effort를 N·m로 읽음 |
| `lerobot_robot_piper/piper_follower.py` | `use_effort=True`일 때 `.effort` 필드를 observation에 병합. `use_depth_observation=True` + 해당 카메라의 `use_depth=True`일 때 `<cam>_depth`(turbo 컬러맵 RGB)를 observation에 병합 |
| `lerobot_robot_piper/depth_utils.py` (신규) | `depth_to_colormap()` — raw uint16 depth → 고정범위 클리핑 → turbo 컬러맵. `jet` 대신 `turbo`(OpenCV 5.0에서 `COLORMAP_TURBO` 확인됨) |
| `configs/recording.env(.example)` | `USE_EFFORT`, `USE_DEPTH_OBSERVATION`, `DEPTH_MIN_M`, `DEPTH_MAX_M` 키 추가 |
| `lerobot_robot_piper/teleop_ui.py` | Record preset에 "Record Effort" / "Record Depth" 체크박스 추가. `_camera_args()`가 `--robot.use_effort`/`--robot.use_depth_observation`/`--robot.depth_min_m`/`--robot.depth_max_m`을 lerobot-record/lerobot-record(--policy.path) 커맨드에 자동 반영. "Save as Default"로 recording.env에 영속화 가능 |
| `scripts/tools/test_effort_mock.py`, `test_depth_mock.py` (신규) | 하드웨어 없이 mock으로 on/off 분기 검증. 실행: `PYTHONPATH=. python scripts/tools/test_effort_mock.py` |

### 알아둘 것 — depth 저장 방식이 원 가이드와 다름

원 가이드(`tavla-depth-force-integration.md`)는 "raw 16bit 저장, 변환은 dataloader에서"를 원칙으로 제시했다. 하지만 lerobot의 `hw_to_dataset_features()`가 카메라 feature를 전부 `(H,W,3)` uint8 video/image로 가정하는 구조라, raw uint16 단일채널을 그대로 저장하려면 lerobot 코어(dataset writer)까지 손대야 한다. 지금은 **캡처 시점에 turbo 컬러맵으로 변환해서 저장**하는 쪽으로 갔다 — 스코프를 줄이기 위한 의도적 단순화이며, 나중에 컬러맵/정규화 범위를 바꾸고 싶으면 재수집이 필요하다는 트레이드오프가 있다. `depth_min_m`/`depth_max_m`을 recording.env로 노출해둔 것도 이 제약 때문(적어도 범위는 재수집 없이 조정 가능하게).

### 환경 이슈 (발견만 함, 미수정)

- pip editable install(`pip install -e .`) 매핑이 `/Users/seongilcho/piper/lerobot_robot_piper`라는 없어진 경로를 가리킴 → `pip install -e . --force-reinstall --no-deps`로 갱신 권장.

### 이번 세션(effort 안전 컷오프 추가) 중 발견 및 수정

- `teleop_ui.py`의 `_robot_safety_args()` 중복 정의(뒤 정의가 덮어씀) 문제를 고치면서 확인해보니, 뒤 정의만 남기고 넘어갔다면 실제로는 문제가 없었을 단순 중복이 아니라 **커버리지 갭**이었다: effort 안전 컷오프(`safety_enabled`/`safety_effort_limit`) CLI 인자를 처음엔 `_observation_toggle_args()`(Record/Infer의 `_camera_args()` 경유)에만 넣어서, Teleoperate와 **Replay(지우개 사고가 난 바로 그 경로)** 에는 안 실렸다. `PiperFollowerConfig` 기본값(ON, 8.0N·m)에는 걸려서 실제로 안전하지 않았던 건 아니지만, 랩 PC에서 `SAFETY_EFFORT_LIMIT`을 recording.env로 튜닝해도 Replay/Teleoperate엔 반영이 안 되는 상태였다. 지금은 두 정의를 하나로 합치고 그 안에 안전 인자를 넣어서 Teleoperate/Record/Infer/Replay 네 커맨드가 전부 같은 값을 쓴다.

## 4. Ablation 표 (최신, Track 병합본)

| # | 조건 | FT 방식 | effort | depth | compliance 출력 | 목적 |
|---|---|---|---|---|---|---|
| 1 | Baseline | Full FT | ✗ | ✗ | ✗ | 기준선 |
| 2 | Baseline | LoRA | ✗ | ✗ | ✗ | 소량 데이터에서 LoRA로 기준 성능 재현되는지 |
| 3 | +Effort(STATE) | Full FT | STATE | ✗ | ✗ | TA-VLA DePre, Full FT 상한선 |
| 4 | +Effort(decoder-token) | LoRA | 이력→토큰 | ✗ | ✗ | TA-VLA 발견1 (encoder vs decoder 배치) |
| 5 | +Aux torque 예측 | LoRA | 이력→토큰+보조예측 | ✗ | ✗ | TA-VLA 발견3 |
| 6 | +Depth | LoRA | (5) 유지 | ✓ | ✗ | 3D 정보 기여 |
| 7 | +Chunk 축소+Dropout | LoRA | (6) 유지 | ✓ | ✗ | open-loop 한계 + FACTR 지름길 방지 |
| 8 | +이산 phase→kp 프리셋 | LoRA | (7) 유지 | ✓ | 이산 프리셋(3~4종) | 저위험 컴플라이언스 |
| 9 | +연속 kp 예측 | Full FT | (7) 유지 | ✓ | 연속값 | 성능 상한 실험 (주제 이탈 허용) |

1~2번(위 표에서 아직 코드로 준비 안 된 부분: STATE/decoder-token 주입은 지금 effort가 STATE에만 붙어 있어 3번까지는 바로 실험 가능. 4번(decoder-token) 이후는 SmolVLA `modeling_smolvla.py` 수정이 필요한 다음 단계.

## 5. 다음 단계 후보 (우선순위 제안)

1. **랩 PC 실기 검증** — `use_effort`/`use_depth_observation` 켜고 mock 대신 실제 하드웨어로 `get_observation()` 한 번 돌려서 값 범위/타이밍 확인 (CLAUDE.md 안전 규칙: 실물 명령은 사용자 확인 후).
2. **컴플라이언스 마운트 + PlotJuggler 진단** (원 가이드 Part 5 [0]단계) — effort/depth보다 먼저 하는 게 맞음. 오염된 데이터로 학습하면 effort를 넣어도 소용없다는 게 원 가이드의 핵심 경고.
3. `pip install -e .` 정리 (환경 이슈 항목).
4. 학습 스크립트(`7__train.sh`)에 LoRA 옵션 추가 여부 확인 — 지금 `configuration_smolvla.py`엔 LoRA 관련 필드가 없어서, lerobot 쪽에 있는지부터 확인 필요.
5. Ablation 표 3~5번(decoder-token 주입, aux torque 예측)은 `modeling_smolvla.py` 수정이 필요한 더 큰 작업 — 위 1~2번이 끝난 뒤 착수 권장.
