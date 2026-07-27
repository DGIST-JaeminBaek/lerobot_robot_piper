# Policy 실행 주기와 Action Chunk

## 1. 문서 범위

이 문서는 policy가 예측한 action chunk를 로봇에서 실행하는 방식과 다음 설정의 관계를
설명한다.

- `fps`
- `chunk_size`
- `n_action_steps`
- `actions_per_chunk`
- `chunk_size_threshold`

## 2. 서로 다른 세 가지 크기

### `chunk_size`

Policy가 한 번에 예측하도록 학습된 action sequence의 최대 길이다. 현재 SmolVLA
기본값은 50이다.

### `n_action_steps`

동기 실행에서 예측한 chunk 중 로컬 action queue에 넣을 개수다. 현재 SmolVLA의
`select_action()`은 queue가 비었을 때만 새 chunk를 예측하고, 앞의
`n_action_steps`개를 queue에 넣는다.

```text
queue가 비어 있음
  → action chunk 예측
  → 앞 n_action_steps개를 queue에 저장
  → 프레임마다 하나씩 실행
  → queue가 다시 비면 재추론
```

따라서 동기 실행에서는 실제 제어 루프가 목표 fps를 유지한다는 가정 아래 한 번
추론한 action을 실행하는 시간이 대략 다음과 같다.

```text
n_action_steps / fps
```

예를 들어 30 fps에서 `n_action_steps=50`이면 계산상 약 1.67초다. 실제 시간은 카메라
읽기, 통신 및 추론 지연 때문에 달라질 수 있으므로 로그로 측정해야 한다.

### `actions_per_chunk`

Async inference server가 예측 결과 중 client에 보낼 action 개수다. 로컬
`scripts/9__run_client.sh`의 기본값은 50이다.

이 값은 async 경로에서 사용하는 설정이며 SmolVLA의 동기 실행용
`n_action_steps`와 같은 변수가 아니다.

## 3. `fps`의 의미

Async client의 `fps`는 client control loop의 목표 주기다.

```text
environment_dt = 1 / fps
```

Client는 한 loop에서 가능한 경우 action 하나를 실행하고, queue 조건을 만족하면
observation을 전송한 뒤 남은 시간만큼 대기한다. Server도 자신의 `fps`에서 계산한
시간 간격으로 chunk의 각 action에 timestep과 timestamp를 부여한다. 로컬
`8__run_server.sh`와 `9__run_client.sh`는 같은 `FPS` 환경변수를 사용하므로 두 값을
같게 유지해야 한다.

여기서 `fps`는 다음과 구분해야 한다.

- 카메라가 설정된 영상 fps
- 데이터셋 녹화 fps
- GPU가 실제로 처리할 수 있는 inference 횟수

일반적으로 같은 값으로 맞추더라도 의미는 서로 다르다. Control loop 처리 시간이
`1 / fps`보다 길면 설정값만큼의 실제 주기를 달성하지 못한다.

## 4. 동기 Policy 실행

SmolVLA의 동기 `select_action()` 경로는 다음과 같이 동작한다.

1. Action queue가 비었는지 확인한다.
2. 비어 있으면 최신 observation으로 action chunk를 예측한다.
3. 앞의 `n_action_steps`개를 queue에 넣는다.
4. 호출할 때마다 action 하나를 꺼낸다.
5. Queue가 빌 때까지 새 action chunk를 예측하지 않는다.

따라서 observation이 매 프레임 생성되더라도 queue가 남아 있는 동안에는 그
observation이 새 action 계산에 사용되지 않는다.

`n_action_steps`를 줄이면 더 자주 최신 observation으로 재계획할 수 있지만 다음 비용이
발생한다.

- 추론 호출 횟수 증가
- GPU 부하 증가
- 추론이 느릴 경우 제어 주기 지연
- Chunk 사이 action이 불연속적으로 변할 가능성

`chunk_size`는 학습 horizon으로 유지하면서 `n_action_steps`만 줄여 앞부분을 실행하고
더 자주 재계획하는 실험이 가능하다.

## 5. Async inference 실행

Async 경로는 동기 경로처럼 action queue가 완전히 빌 때마다 순차적으로 추론하는
구조가 아니다.

```text
Robot client                         Policy server
------------                         -------------
observation 전송  ─────────────────→ 최신 observation 보관
                                     action chunk 추론
action chunk 수신 ←───────────────── actions_per_chunk개 반환
로컬 queue에서 action 실행
queue가 threshold 이하가 되면
새 observation 전송 ───────────────→ 다음 chunk 추론
새 chunk를 기존 미래 action과 병합
```

Client는 다음 조건이 참일 때 새 observation을 전송할 수 있다.

```text
현재 queue 크기 / 수신한 chunk 크기 <= chunk_size_threshold
```

현재 `scripts/9__run_client.sh` 기본값은 다음과 같다.

```text
FPS=30
ACTIONS_PER_CHUNK=50
CHUNK_SIZE_THRESHOLD=0.8
AGGREGATE_FN_NAME=average
```

50개를 받은 직후에는 비율이 1.0이다. 약 10개를 소비해 40개가 남으면 비율이 0.8이
되어 새 observation 전송 조건을 만족한다. Server가 새 chunk를 반환하면 client는
같은 timestep의 미래 action을 설정된 aggregate 함수로 결합한다.

따라서 async 기본 설정을 단순히 `50 / 30 = 1.67초 open-loop`라고 설명하면
부정확하다. 새 관측과 추론은 기존 queue가 완전히 비기 전에 진행될 수 있다.

다만 실제 재계획 간격은 다음 요소의 영향을 받는다.

- Client가 달성한 실제 control fps
- Observation 캡처 및 직렬화 시간
- Client-server 네트워크 지연
- Server inference 시간
- `actions_per_chunk`
- `chunk_size_threshold`
- Server의 중복·유사 observation 필터링
- Action aggregate 방식

## 6. 주요 설정의 영향

| 설정 변경 | 기대 효과 | 주의점 |
|---|---|---|
| `n_action_steps` 감소 | 동기 실행에서 재추론 증가 | GPU 부하와 지연 증가 |
| `actions_per_chunk` 감소 | Async client가 받는 horizon 감소 | 추론이 늦으면 queue 고갈 가능 |
| `chunk_size_threshold` 증가 | Queue가 많이 남았을 때 새 관측 전송 | 추론·통신 빈도 증가 |
| `chunk_size_threshold` 감소 | 기존 chunk를 더 오래 사용 | 최신 관측 반영이 늦어질 수 있음 |
| `fps` 증가 | 목표 제어 주기 증가 | 카메라·CAN·추론이 따라오지 못할 수 있음 |

## 7. 확인할 측정값

설정값만으로 실제 동작을 판단하지 않고 다음 값을 로그로 측정한다.

- Client control loop의 실제 평균 fps
- Observation 캡처 시간
- Client에서 server까지의 지연
- Server inference 시간
- Server에서 client까지의 지연
- Action queue의 시간별 크기
- 새 observation과 새 chunk가 생성된 timestep
- Observation 전송부터 새 action 적용까지 걸린 시간
- Action queue가 비어 제어가 끊긴 횟수

Async client의 `debug_visualize_queue_size` 옵션을 사용하면 종료 후 queue 크기 변화를
확인할 수 있다.

## 8. 관련 코드

- `/home/ugrp43/UGRP/lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py`
- `/home/ugrp43/UGRP/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py`
- `/home/ugrp43/UGRP/lerobot/src/lerobot/async_inference/robot_client.py`
- `/home/ugrp43/UGRP/lerobot/src/lerobot/async_inference/policy_server.py`
- `scripts/8__run_server.sh`
- `scripts/9__run_client.sh`
