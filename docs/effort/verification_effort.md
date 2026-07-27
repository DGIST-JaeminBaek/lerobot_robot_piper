# Effort 녹화 실기 검증 (GUI 기준)

## 현재 검증 상태

- 실물 Piper 녹화에서 effort 값이 데이터셋에 저장되는 것은 확인했다.
- Effort 안전 임계값은 아직 결정하지 않았다.
- Effort 임계값 초과 시 동작하는 안전 컷오프는 실물에서 검증하지 않았다.
- 현재 `SAFETY_EFFORT_LIMIT` 값은 검증된 안전 기준으로 간주하면 안 된다.

## 0. 이 검증의 범위

- effort/velocity가 데이터셋에 실제로 들어가는지 확인
- GUI와 `5__record.sh`에서 `USE_EFFORT` 설정이 전달되는지 확인

## 1. 사전 준비

`configs/recording.env`:

```bash
USE_EFFORT=true              # 기본값이 true라 없어도 되지만 명시 권장
NUM_EPISODES=2
EPISODE_TIME_S=20
RESET_TIME_S=10
```

## 2. CAN 연결

```bash
ip link show | grep can        # 인터페이스 존재 확인
bash scripts/1__init_can.sh    # 필요 시
bash scripts/0__launch_gui.sh
```

GUI에서: **CAN Setup** 패널 → `Detect` → leader/follower 인터페이스 확인 → `Init All`

## 3. CAN 통신 확인 (로봇 안 움직임)

**CAN Monitor** 패널 → `Start Monitor` → Joint Positions 값이 실시간으로 갱신되면 정상.
리더 팔을 손으로 조금 움직여서 값이 따라 변하는지 확인.

확인 후 **`Stop Monitor`를 반드시 누른다** — 모니터가 CAN을 잡고 있으면 녹화와 충돌한다.
(이 패널은 위치만 보여준다. effort는 §5에서 데이터로 확인한다.)

## 4. Command 확인

1. **Preset** → `Record` 선택
2. **Task** 입력
3. **"Record Effort" 체크박스**가 켜져 있는지 확인
4. **Command**에서 다음 인자를 확인:

```
--robot.use_effort=true
--dataset.num_episodes=2
```

`use_effort=false`로 보이면 체크박스와
`recording.env`의 `USE_EFFORT=true`를 확인하고 GUI를 재시작한다.

## 5. 짧게 2 에피소드 녹화

1. `Launch`
2. 카메라 warmup이 끝날 때까지 기다린다
3. 진행률에 `Recording episode 1/2`가 뜨면 시작된 것
4. 텔레옵으로 관절을 움직여 position, effort, velocity 값이 변하는 데이터를 만든다
5. 20초 후 자동으로 Reset 구간 → 다시 20초 → 2 에피소드 후 자동 종료
6. 종료 시 파킹 자세로 이동 + 토크 해제

전체 로그는 `last_launch.log`에 남는다.

## 6. 데이터 검증

**Recording History** 패널에서 방금 만들어진 데이터셋 경로를 확인한 뒤:

```bash
python scripts/tools/check_effort.py <데이터셋경로>
```

### ✅ 통과 기준

```
observation.state 차원 : 20
pos 7개 / effort 7개 / vel 6개
✅ effort 필드 존재
✅ effort 값이 실제로 변하고 있음
```

### ❌ 실패 패턴

| 출력 | 원인 | 조치 |
|---|---|---|
| `차원 7`, `effort 없음` | 플래그 미반영 | §4를 다시 확인 |
| `effort가 전부 0` | CAN에서 값을 못 읽음 | 팔 전원과 CAN 연결 확인 |

## 7. 셸 경로 확인

```bash
DRY_RUN=true bash scripts/5__record.sh 2>&1 | grep use_effort
```

`--robot.use_effort=true`가 출력되면 통과.

## 8. 안전 컷오프 실물 검증 — 미실시

현재 기본 설정은 다음과 같다.

```text
SAFETY_ENABLED=true
SAFETY_EFFORT_LIMIT=8.0
```

`8.0 N·m`은 실물에서 적절성이 확인된 값이 아니다. 현재 구현도 effort 초과 시 새
action 전송을 생략할 뿐, 후퇴·parking·제어 종료를 수행하지 않는다. 따라서 아래
절차는 현재 컷오프의 트리거와 실제 로봇 반응을 확인하기 위한 것이며, 완성된 안전
기능의 검증 절차가 아니다.

### 준비

1. 로봇 주변의 장애물과 도구를 치운다.
2. 로봇을 안정된 자세에 두고 이동 속도를 낮춘다.
3. 검증 담당자는 비상정지 장치에 손을 둔다.
4. `check_effort.py`로 정상 동작 중 관절별 effort 범위를 먼저 확인한다.
5. 정상 effort 범위를 확인하기 전에는 `8.0 N·m`을 안전 기준으로 확정하지 않는다.

### 트리거 확인

1. 장애물과 접촉하지 않은 안정된 자세에서 시작한다.
2. 현재 자세의 effort보다 조금 낮은 시험용 임계값을 설정한다.
3. 작은 범위의 텔레옵 명령을 보내 컷오프를 발생시킨다.
4. `last_launch.log`에서 다음 경고를 확인한다.

```text
safety cutoff: effort {...} exceeds ... N·m
```

5. 경고 이후 로봇의 실제 움직임과 모터 상태를 관찰한다.
6. 시험이 끝나면 임계값을 원래 설정으로 복구한다.

장애물을 누르거나 충돌을 만들어 컷오프를 시험하지 않는다. 로그가 출력되는 것만으로
물리적 정지가 검증됐다고 판정해서도 안 된다.

### 통과 판정

현재 구현에 대해 확인할 항목:

- 설정한 effort 임계값에서 컷오프 경고가 발생한다.
- 컷오프가 발생한 프레임에서 새 `set_action()`이 호출되지 않는다.
- 실제 로봇이 기존 목표를 계속 추종하는지, 그 자리에서 유지되는지 별도로 기록한다.

저속 후퇴 → parking → teleop/policy 종료 기능을 구현하면 다음 항목을 추가로 검증해야
한다.

- Effort 초과 후 정상 action이 차단된다.
- 저속 후퇴 중 effort가 감소한다.
- Effort가 해소된 뒤 parking position으로 이동한다.
- Parking 완료 후 teleop/policy 제어가 종료된다.
- 후퇴 실패나 effort 증가 시 parking을 강행하지 않는다.

## 9. 검증 완료 체크리스트

- [x] Command 창에 `--robot.use_effort=true` 확인
- [x] `check_effort.py`에서 position 7개 / effort 7개 / velocity 6개 확인
- [x] Effort와 velocity 값이 실제로 변하는지 확인
- [x] 셸 경로 `DRY_RUN`에서 `--robot.use_effort=true` 확인
- [ ] 안전 컷오프 실물 검증 완료

## 참고

- effort 녹화 기능: `docs/effort/effort.md`
