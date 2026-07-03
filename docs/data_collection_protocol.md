# Data Collection Protocol

이 문서는 특정 task에 묶이지 않는 LeRobotDataset 수집 기준을 정리합니다. 실제 task 문장, episode 길이, 카메라 구성, dataset 이름은 `configs/recording.env`에서 실험마다 정합니다.

## 1. Task 정의

녹화를 시작하기 전에 아래 항목을 먼저 고정합니다.

| 항목 | 설명 |
|---|---|
| `TASK` | episode가 수행해야 하는 동작을 한 문장으로 작성 |
| 성공 조건 | episode가 성공으로 인정되는 최소 조건 |
| 실패 조건 | 충돌, 물체 이탈, 카메라 가림, 중단 등 폐기 기준 |
| 시작 상태 | robot, 물체, 작업 공간의 초기 배치 |
| 종료 상태 | episode가 끝날 때 기대하는 안전한 자세 또는 배치 |

`TASK`는 너무 넓게 쓰지 말고, 같은 dataset 안에서 일관되게 해석될 정도로 구체적으로 작성합니다.

예시:

```text
pick up the object and place it in the target area
open the drawer and move the handle back to the start position
press the button once and return to the neutral pose
```

## 2. Episode 시작 조건

각 episode를 시작하기 전에 아래 조건을 맞춥니다.

1. leader와 follower의 CAN 포트가 올바르게 매핑되어 있다.
2. follower가 안전한 시작 자세에 있다.
3. 작업 공간의 물체, 도구, fixture 위치가 task 기준에 맞게 배치되어 있다.
4. 상방 카메라가 전체 작업 공간과 follower 움직임을 볼 수 있다.
5. 팔목 카메라가 접촉 지점, gripper, 조작 대상 중 중요한 영역을 볼 수 있다.
6. `scripts/4__teleoperate.sh`로 짧게 반응을 확인했다.
7. episode 시작 전 화면, 조명, 케이블, 장애물이 수집을 방해하지 않는다.

## 3. Episode 수행 기준

각 episode는 아래 흐름을 따릅니다.

1. 시작 자세에서 안정적으로 출발한다.
2. task 수행에 필요한 접근, 조작, 이동 동작을 포함한다.
3. 중간에 불필요한 멈춤이나 재시도가 과도하게 들어가지 않도록 한다.
4. task 성공 또는 실패가 영상과 state/action에서 판단 가능해야 한다.
5. episode 마지막에는 안전한 종료 자세로 이동하거나 움직임을 안정적으로 멈춘다.

같은 dataset 안에서는 시작 위치, 물체 배치, 조작 순서를 가능한 한 일관되게 유지합니다. 변화를 주는 실험이라면 어떤 요소를 변화시키는지 별도로 기록합니다.

## 4. 품질 기준

성공 episode 기준:

- `TASK`에 정의한 동작이 끝까지 수행됨
- top/wrist image가 task 판단에 필요한 장면을 충분히 보여줌
- `observation.state`와 `action`이 저장됨
- follower 움직임이 급격히 튀거나 비정상적으로 끊기지 않음
- episode 길이가 task 수행에 충분하고 불필요하게 길지 않음

실패 또는 폐기 기준:

- robot 충돌, 작업 공간 이탈, 물체 낙하 등 안전 문제가 발생함
- 카메라가 가려져 task 성공 여부를 판단하기 어려움
- recording 중단, frame 누락, 센서 오류가 발생함
- leader/follower 매핑 오류나 action offset 문제로 follower가 튐
- task와 무관한 동작이 episode 대부분을 차지함

실패 episode를 보존할지 폐기할지는 실험 목적에 맞춰 결정합니다. 정책 학습용 성공 demonstration dataset이라면 실패 episode는 별도 dataset으로 분리하는 것을 권장합니다.

## 5. 첫 수집 절차

처음에는 짧은 episode로 pipeline을 검증합니다.

권장 순서:

1. `NUM_EPISODES=1`, `EPISODE_TIME_S=10` 정도로 smoke recording
2. dataset feature 확인
3. 영상과 parquet 정합성 확인
4. task 문장, 카메라 위치, episode 길이 조정
5. 문제가 없으면 3-5 episode로 확장
6. 최종 설정이 안정되면 본 수집 진행

녹화:

```bash
bash scripts/5__record.sh
```

Dataset 확인:

```bash
python3 scripts/tools/wego_dataset_check.py \
  --dataset-repo-id <DATASET_REPO_ID> \
  --dataset-root <DATASET_ROOT> \
  --episode 0
```

확인할 항목:

- `observation.state` 저장 여부
- `action` 저장 여부
- top/wrist image 저장 여부
- `task` 값이 의도한 문장인지 여부
- 첫 frame과 마지막 frame의 action/state가 비정상적으로 튀지 않는지 여부
- episode 길이와 frame 수가 설정값과 맞는지 여부

## 6. 수집 메모

실험마다 아래 정보를 남기면 이후 학습/추론 문제를 추적하기 쉽습니다.

| 항목 | 예시 |
|---|---|
| dataset id | `local/example_task_v1` |
| task | `pick up the object and place it in the target area` |
| robot port | leader/follower CAN 이름 |
| camera | top/wrist camera serial 또는 index |
| episode count | 성공/실패/폐기 개수 |
| 변경 사항 | 조명, 물체 위치, camera angle, episode length 등 |
| 특이 사항 | follower 튐, frame drop, 충돌, 재시도 패턴 등 |
