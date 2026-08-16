# 인간 승인형 Policy 실행 파이프라인

## 1. 목적

이 문서는 SmolVLA가 예측한 action chunk를 실제 Piper에 바로 보내지 않고, 먼저
RViz에서 재생한 뒤 사람의 명시적 승인을 받은 경우에만 실행하는 절차를 정의한다.

```text
최신 observation 획득
  → action chunk 추론
  → 정적 검사
  → chunk를 실행 구간으로 분할
  → 현재 구간 RViz 재생
  → 현재 구간 인간 승인
  → 승인 직전 상태 재검사
  → 승인된 현재 구간만 실행
  → 남은 chunk의 다음 구간을 RViz 재생
  → 전체 chunk 완료 후 새 observation으로 재추론
```

목표는 정책의 출력을 사람이 실행 전에 확인할 수 있게 만드는 것이다. RViz 확인은
안전 보조 수단이며 충돌 없음이나 실물 안전을 보증하는 검증기는 아니다.

## 2. 동기 실행을 선택하는 이유

LeRobot의 기본 async client는 action queue에 새 chunk가 도착하면 기존 미래
action과 병합하고 곧바로 실행한다. 실행 중 queue가 일정 크기 이하로 내려가면 다음
observation을 전송해 추론도 겹쳐서 진행한다.

인간 승인에는 소요 시간이 정해져 있지 않다. 따라서 다음 이유로 기존 async queue를
그대로 사용하지 않는다.

- Chunk 수신과 로봇 실행 사이에 승인 대기 상태가 필요하다.
- 승인 전에는 action이 자동으로 소비되면 안 된다.
- 추론 당시 부여된 timestamp는 승인 대기 중 낡게 된다.
- 대기 중 로봇이나 작업 환경이 변하면 기존 chunk를 실행하면 안 된다.
- 거부, 승인 시간 초과 및 안전 검사 실패 시 chunk를 폐기해야 한다.

이 파이프라인은 **chunk 내부 구간 단위의 동기식 승인 실행**으로 설계한다. 모델
추론, ROS publish 및 화면 갱신은 별도 thread나 process에서 수행할 수 있지만, 실제
`robot.send_action()` 호출은 승인 상태를 통과한 실행기만 수행한다.

## 3. 상태 머신

```text
CONNECT
  ↓
HOLD
  ↓
CAPTURE
  ↓
INFERENCE
  ↓
VALIDATE
  ├─ 실패 ───────────────→ REJECTED → HOLD
  ↓
SPLIT_CHUNK
  ↓
RVIZ_PREVIEW_SEGMENT
  ↓
WAIT_SEGMENT_APPROVAL
  ├─ replay ──────────────→ RVIZ_PREVIEW_SEGMENT
  ├─ discard ─────────────→ 남은 chunk 폐기 → CAPTURE
  ├─ quit/시간 초과 ──────→ 종료
  ↓ 승인
STATE_RECHECK
  ├─ 상태 변화 큼 ───────→ STALE → 같은 구간 재확인
  ↓
EXECUTE_SEGMENT
  ├─ effort/통신/한계 초과 → SAFETY_STOP
  ↓
남은 구간 있음 ──────────→ RVIZ_PREVIEW_SEGMENT
  ↓ 없음
CAPTURE
```

### `HOLD`

현재 구현은 승인 대기 중 새 action을 보내지 않는다. 따라서 하위 제어기에는 마지막
목표가 남아 있으며, 별도의 명시적 `hold_current_position()` API는 아직 구현하지
않았다. 승인 구간 사이의 실제 자세는 다음 구간 preview 전에 다시 읽는다.

이때 승인 실행 경로에서는 `robot.use_action_offset=false`를 사용해야 한다. 현재
데이터셋의 `action`은 follower에 실제로 전송된 값을 저장하고 정책도 그 좌표계의
action을 학습하므로, 추론 결과나 현재 자세 hold 목표에 leader/follower offset을
다시 적용하면 안 된다.

### `CAPTURE`

정지한 상태에서 최신 카메라 영상과 `observation.state`를 한 묶음으로 읽는다. 이
observation과 현재 관절 위치는 이후 stale 판정에 사용할 수 있도록 chunk와 함께
보관한다.

### `INFERENCE`

SmolVLA에서 최대 50개의 action을 예측한다. 이 단계에서는 action을 로봇 queue에
넣지 않는다. 승인 대기 시간이 있으므로 async server가 붙인 기존 실행 timestamp를
사용하지 않고, 순서가 있는 순수 action 배열로 보관한다.

### `VALIDATE`

RViz에 보내기 전에 최소한 다음 항목을 검사한다.

- action이 `(N, 7)` 형태인지
- NaN 또는 Inf가 없는지
- Preview/실행 action 수가 양수이고 policy chunk size를 넘지 않는지
- 실행 구간 크기가 preview action 수를 넘지 않는지
- FPS와 stale tolerance 설정이 유효한지

Joint/gripper 전역 범위 초과는 실패시키는 대신 범위로 clamp하고 조정 수와 최대
조정량을 기록한다. Relative target도 preview에서 clamp한다. 연속 궤적의 물리적
안전성이나 충돌을 자동 판정하는 검사는 구현하지 않았다.

### `RVIZ_PREVIEW`

현재 구간의 예측 action을 `/joint_states`에 publish해 Piper URDF로 재생한다. 이
단계는 실제 Piper의 CAN 명령 경로와 분리한다. 사람은 `r`로 현재 구간을 반복
재생할 수 있다.

RViz는 기본적으로 관절 자세를 표시할 뿐 다음 항목을 자동으로 판정하지 않는다.

- 작업대, 물체 및 사람과의 충돌
- 케이블 간섭
- 실제 모터 effort
- 실물의 추종 오차와 동역학

따라서 RViz 화면을 봤다는 사실만으로 안전 검사가 완료된 것으로 처리하지 않는다.

### `WAIT_APPROVAL`

터미널 입력은 다음 네 가지다.

- `a`: 현재 구간만 승인하고 실행한다.
- `r`: 현재 구간을 RViz에서 다시 재생한다.
- `d`: 현재 구간을 포함한 남은 chunk를 폐기하고 새로 추론한다.
- `q`: 실행기를 종료한다.

승인 timeout이 설정되어 시간이 초과되면 action을 보내지 않고 종료한다.

초기 구현은 터미널의 명시적인 승인 입력으로 충분하다. GUI에 연결할 경우에도
기본값은 거부로 두고, 승인 이벤트는 chunk ID와 일치할 때 한 번만 소비해야 한다.

### `STATE_RECHECK`

승인 직전에 현재 관절 위치를 다시 읽어 `CAPTURE` 당시 위치와 비교한다. 허용 오차를
넘으면 preview의 시작 조건이 더 이상 유효하지 않으므로 action을 보내지 않는다.
현재 구현은 남은 chunk를 자동으로 폐기하지 않고, 새 실제 state를 시작점으로 같은
구간을 다시 preview해 사람이 다시 판단하게 한다.

카메라 장면이나 작업 대상이 움직일 수 있는 환경에서는 이미지 변화 검사도 추가한다.
단순 픽셀 차이만으로 안전을 보증할 수는 없지만, 큰 장면 변화가 있으면 stale로
처리하는 보조 조건으로 사용할 수 있다.

### `EXECUTE_SEGMENT`

SmolVLA의 50-action chunk를 설정한 구간 크기로 나눈다. 기본값 10이면 동일 chunk를
다음과 같이 유지하면서 순서대로 승인한다.

```text
chunk [0:50)
  → [0:10) preview/승인/실행
  → [10:20) preview/승인/실행
  → [20:30) preview/승인/실행
  → [30:40) preview/승인/실행
  → [40:50) preview/승인/실행
```

정상 승인된 뒤 남은 40개를 폐기하거나 바로 재추론하지 않는다. 각 구간 실행
전후의 실제 자세를 반영해 다음 구간을 다시 RViz에서 보여준다. 전체 50개가
완료되거나 사용자가 `d`로 남은 chunk를 폐기한 뒤에만 새 observation으로 다시
추론한다.

10 action은 30 FPS에서 약 0.33초다. 실행 중에는 매 action마다 기존 follower의
실제 위치 기준 `max_relative_target`과 effort 안전 컷오프를 통과해야 한다.

## 4. 시간 처리

이 구조에서는 추론 지연이나 승인 시간이 자동 action queue를 고갈시키지 않는다.
승인 대기 중에는 새 action을 보내지 않고 하위 제어기의 마지막 목표가 유지된다.

현재 JSONL 로그에 저장하는 항목:

- Chunk와 segment ID 및 action 범위
- 추론 시간
- Global/relative clamp 수와 최대 조정량
- 승인 결과와 승인 대기 시간
- 실물 적용 여부와 실제 전송된 action 수
- Stale 판정 및 safety trip 여부

Observation 획득 시간, RViz publish 시간과 달성한 실제 FPS의 상세 계측은 아직
추가하지 않았다.

승인 버튼을 누른 시점을 실행 기준시각으로 삼아 action을 `1 / FPS` 간격으로
전송한다. Observation 생성 시점에 붙었던 async용 timestamp를 재사용하지 않는다.

## 5. 거부 및 비정상 상황 처리

다음 상황에서는 action을 실행하지 않거나 실행기를 종료한다.

- 추론 실패 또는 제한 시간 초과
- RViz/ROS publish 실패
- 정적 검사 실패
- 사람이 거부함
- 승인 대기 시간 초과
- 승인 직전 state가 허용 오차보다 많이 변함
- 실행 중 CAN 통신 오류
- 실행 중 effort 안전 컷오프 발생
- 잘못된 실행 설정

Effort 컷오프가 발생하면 현재 `PiperFollower`의 safety latch가 이후 일반 명령을
차단한다. `safety_on_overload=park`이면 기존 parking 경로가 실행된다. 이 경우 현재
chunk와 승인 상태를 모두 폐기하고 실행기를 종료해야 하며, 같은 프로세스에서
자동으로 안전 latch를 해제하면 안 된다.

## 6. 구현 파일

### 인간 승인 실행기

```text
scripts/tools/piper_human_approved_inference.py
```

구현된 기능:

- Dataset 또는 live Piper observation 선택
- SmolVLA와 dataset statistics 기반 pre/postprocessor 로드
- Policy 입력 TOP/WRIST 이미지 OpenCV 표시
- Action shape와 NaN/Inf 검사
- Piper 전역 joint/gripper 범위 clamp
- RViz preview용 순차 `max_relative_target` clamp
- Chunk를 지정한 action 수로 분할
- 각 구간의 `a/r/d/q` 터미널 승인
- 실물 전송 직전 state stale 검사
- 승인된 구간만 `PiperFollower.send_action()`으로 전송
- 실제 전송 target, chunk 및 승인 결과 저장
- Effort safety latch 감지, parking 대기 및 종료

Policy action은 follower 절대 position target이므로 실물 실행 config에서
`use_action_offset=false`를 강제한다. 외부 LeRobot clone과 `teleop_ui.py`는 이
실행기를 위해 수정하지 않았다.

### Mock 검증

```text
scripts/tools/test_human_approved_inference_mock.py
```

검증한 항목:

1. 실행 함수를 호출하기 전 `send_action()` 0회
2. 승인된 구간 크기만 정확히 전송
3. 정상적인 여러 구간 승인에서 남은 chunk를 유지해 전체 action 전송
4. Effort trip을 모사하면 해당 action은 실제 전송으로 기록하지 않고 잔여 중단
5. 이미 safety latch가 켜진 로봇에는 action 0회

2026-07-28 기준 5개 test가 모두 통과했다. 이는 Python 상태 머신 검증이며 실물
effort, CAN, parking 안전성을 증명하지 않는다.

## 7. 실행 모드와 안전 게이트

기본 모드는 다음과 같다.

```text
HUMAN_APPROVED_SOURCE=dataset
HUMAN_APPROVED_APPLY_TO_ROBOT=false
```

이 모드에서는 CAN이나 Piper에 연결하지 않고 dataset observation, policy, RViz와
입력 영상만 사용한다.

Live observation만 읽고 action을 보내지 않으려면 다음과 같이 설정할 수 있다.

```text
HUMAN_APPROVED_SOURCE=robot
HUMAN_APPROVED_APPLY_TO_ROBOT=false
```

이 모드는 camera와 robot state를 읽기 위해 Piper에 연결하므로 완전한
하드웨어-free 모드는 아니다.

실물 action 전송은 다음 조건을 모두 만족해야 시작된다.

```text
HUMAN_APPROVED_SOURCE=robot
HUMAN_APPROVED_APPLY_TO_ROBOT=true
HUMAN_APPROVED_REAL_ROBOT_CONFIRM=I_UNDERSTAND_REAL_ROBOT
SAFETY_ENABLED=true
SAFETY_ON_OVERLOAD=park
```

조건이 하나라도 맞지 않으면 실행 전에 거부한다. 기본 구간 설정은 다음과 같다.

```text
HUMAN_APPROVED_PREVIEW_ACTIONS=50
HUMAN_APPROVED_EXECUTE_ACTIONS=10
HUMAN_APPROVED_STALE_STATE_TOLERANCE=2.0
HUMAN_APPROVED_PARK_ON_EXIT=true
HUMAN_APPROVED_TOP_CROP=280,0,720
HUMAN_APPROVED_WRIST_CROP=280,0,720
HUMAN_APPROVED_CAMERA_OUTPUT_SIZE=512
```

설정 예시는 `configs/recording.env.example`에 있다. 실제 `recording.env`에
실물 확인 문구를 상시 저장하지 않고 실행할 때 명시적으로 export하는 편이 안전하다.

Robot source에서는 policy에 넣기 전에 TOP/WRIST 1280×720 RGB에 학습 데이터와
동일한 전처리를 적용한다.

```text
1280×720 live RGB
  → x=280, y=0, size=720 square crop
  → OpenCV INTER_AREA로 512×512 resize
  → SmolVLA observation
```

Dataset source는 이미 512×512로 가공되어 있으므로 다시 crop하지 않는다. Live
crop이 실제 frame 범위를 벗어나거나 출력 크기가 dataset metadata의
`(512, 512, 3)` HWC와 다르면 추론과 action 전송 전에 실패한다. Policy tensor
단계에서는 `[3, 512, 512]` CHW로 변환된다. Policy 입력 영상 창에도 crop이 적용된
512×512 TOP/WRIST가 표시된다.

Dataset/RViz-only 실행 예:

```bash
source /opt/ros/humble/setup.bash
conda activate ugrp

python scripts/tools/piper_human_approved_inference.py \
  --dataset-root records/0727/erase_the_shape_512 \
  --episode 0 \
  --policy-path outputs/train/smolvla_erase_shape_512/checkpoints/030000/pretrained_model \
  --task "erase the shape" \
  --source dataset \
  --apply-to-robot false \
  --preview-actions 50 \
  --execute-actions 10
```

실물 실행 경로는 코드만 준비된 상태다. 향후 제한적 실물 검증을 수행할 때는
`recording.env`의 CAN·카메라·effort 설정을 먼저 확인한 뒤 다음 세 값을 명시해야
한다.

```bash
export HUMAN_APPROVED_SOURCE=robot
export HUMAN_APPROVED_APPLY_TO_ROBOT=true
export HUMAN_APPROVED_REAL_ROBOT_CONFIRM=I_UNDERSTAND_REAL_ROBOT
```

이 문서 작성 시점에는 위 실물 모드를 실행하지 않았다.

최초 실물 검증은 기본 50/10 실행보다 작은 1-action으로 시작한다.

```bash
HUMAN_APPROVED_SOURCE=robot \
HUMAN_APPROVED_APPLY_TO_ROBOT=true \
HUMAN_APPROVED_REAL_ROBOT_CONFIRM=I_UNDERSTAND_REAL_ROBOT \
HUMAN_APPROVED_PREVIEW_ACTIONS=1 \
HUMAN_APPROVED_EXECUTE_ACTIONS=1 \
HUMAN_APPROVED_MAX_CHUNKS=1 \
MAX_RELATIVE_TARGET=1.0 \
python scripts/tools/piper_human_approved_inference.py
```

1-action의 live crop, RViz 자세, 실제 이동 방향과 종료 동작을 확인한 뒤에만
`1 → 3 → 5 → 10` 순서로 구간 크기를 늘린다. 시험 중에는 물리 비상정지 수단에
즉시 접근할 수 있어야 한다.

## 8. 정적 clamp와 실물 clamp

Policy 출력에는 두 단계 제한이 적용된다.

1. `[-100, 100]`, gripper `[0, 100]`의 전역 범위 clamp
2. RViz preview에서 직전 목표가 달성됐다고 가정한 순차 relative clamp

두 번째 단계는 preview 근사값이다. 실물에서는 각
`PiperFollower.send_action()` 호출이 실제 현재 position을 다시 읽어
`MAX_RELATIVE_TARGET`을 적용한다. 따라서 저장된 preview target과 실제 follower가
받은 target이 다를 수 있으며, 실행기는 실제 반환 target을 구간별 NPY로 저장한다.

예전 Dataset/RViz 시험 중 policy 출력 `joint3=100.3534`가 기존의 범위 초과 예외를
발생시킨 적이 있다. 이후 전역 clamp를 추가해 범위를 약간 벗어난 finite 출력은
기록하고 제한한 뒤 preview하도록 수정했다. NaN/Inf나 잘못된 action shape는 계속
실패로 처리한다.

## 9. Effort safety 동작

실물 실행의 모든 action은 `PiperFollower.send_action()`을 거친다.

```text
실제 position 읽기 및 relative clamp
  → effort 읽기
  → 관절 하나라도 abs(effort) > SAFETY_EFFORT_LIMIT
  → 현재 action을 set_action하지 않음
  → safety latch 설정
  → background parking
  → 현재 구간과 남은 chunk 중단
  → parking 완료 대기
  → 실행기 종료
```

현재 설정의 limit은 `8.0 N·m`이다. 이 effort는 전용 토크 센서가 아니라 motor
current 기반 추정값이다. Dataset의 effort를 policy 입력으로 사용하지 않아도 안전
검사는 실시간으로 별도 수행된다.

기존 데이터 확인에서는 `erase_the_circle` 정상 episode의 최대값이 약
`5.42~6.99 N·m`였고, outlier `circle_triangle` 2개는 joint5가 약
`14.16~14.27 N·m`까지 기록됐다. 따라서 `8.0 N·m`는 일부 실제 동작에서 trip될 수
있다. 적절한 limit과 trip 후 parking은 아직 실물 검증이 필요하다.

## 10. 종료 동작

정상 종료와 `Ctrl+C`는 `finally`에서 follower disconnect를 수행한다.

현재 기본 설정:

```text
HUMAN_APPROVED_PARK_ON_EXIT=true
DISABLE_TORQUE_ON_DISCONNECT=true
```

따라서 정상 종료 또는 `Ctrl+C` 시 parking한 뒤 torque를 해제하고 연결을 종료한다.
Effort trip에서는 safety parking을 먼저 기다리고, 중복 parking 없이 disconnect한
뒤 기본 설정에 따라 torque를 해제한다.

`q`는 승인 입력 대기 중 사용하는 정상 종료다. `Ctrl+C`는 action 구간 실행 중에도
Python loop를 중단할 수 있지만 물리 E-stop은 아니다. `Ctrl+C` 후에도 parking
동작이 발생하므로 충돌 상황에서 parking 경로 자체가 안전하다고 보장할 수 없다.
물리 비상정지 입력 연동과 parking 없는 즉시 정지 경로는 아직 구현하지 않았다.

## 11. 현재 검증 상태

- [x] Dataset observation에서 SmolVLA chunk 생성
- [x] Policy 입력 TOP/WRIST 이미지 표시
- [x] Live TOP/WRIST에 학습 데이터와 동일한 720 crop → 512 resize
- [x] Crop 좌표·출력 shape·범위 오류 synthetic test
- [x] Dataset 기반 RViz preview
- [x] Episode 0, 2, 36, 55의 RViz-only 승인 기록
- [x] Global 범위 clamp와 relative preview clamp
- [x] 50-action chunk를 10-action 구간으로 유지·분할하는 코드
- [x] 실물 실행을 위한 3중 설정 gate
- [x] `use_action_offset=false` 강제
- [x] 승인 직전 state stale 검사
- [x] `PiperFollower.send_action()` 안전 경로 연결
- [x] Action 실행 mock 5개와 live crop mock 4개 통과
- [ ] 새 구간별 UI의 Dataset/RViz 수동 재검증
- [ ] 실제 Piper에서 승인 전 action 0회 확인
- [ ] 실제 Piper에서 10-action 구간 전송 확인
- [ ] 실제 Piper에서 `d`, `q`, timeout 및 `Ctrl+C` 종료 확인
- [ ] 실물 effort trip과 parking 확인
- [ ] 안전한 effort limit 결정
- [ ] 물리 E-stop 또는 parking 없는 긴급정지 경로

## 12. 관련 문서

- [Policy 실행 주기와 Action Chunk](README.md)
- [Offline Action Chunk Rollout](offline_chunk_rollout.md)
- [ROS 2/RViz/URDF 재현 절차](../rviz_setup.md)
- [Effort 검증 절차](../effort/verification_effort.md)
