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
SAFETY_ON_OVERLOAD=park
```

`8.0 N·m`은 실물에서 적절성이 확인된 값이 아니다. 세 값 모두 GUI(teleop_ui)의
Script Launcher 행에서 바로 바꿀 수 있고(Safety Cutoff / Effort limit /
On overload), `Save as Default`를 누르면 `configs/recording.env`에 저장된다 —
GUI 값이 recording.env보다 우선한다.

`SAFETY_ON_OVERLOAD`는 임계값 초과 시 동작을 정한다.

- `hold` — 트립 순간의 자세에서 그대로 정지.
- `park` — parking 자세까지 `SAFETY_PARK_RAMP_S`(기본 4초) 동안 천천히 복귀한 뒤 정지.
  0으로 두면 옛 동작(목표를 한 번에 쏴서 최고속 이동) — 파킹 자세가 팔이 수직으로
  뻗은 자세라서 트립 순간 팔이 확 뻗는 것처럼 보이고, 외력으로 트립된 직후엔 사람
  손이 팔에 닿아 있을 수 있어 위험하다.

둘 다 리더/정책 목표는 무시하되 **트립 자세를 매 스텝 계속 재전송한다**
(`SAFETY_HOLD_RESEND=true`). 이게 중요한데, 처음 구현은 "명령을 차단한다"를 문자
그대로 구현해 `set_action`을 아예 안 불렀고, Piper는 목표 스트림이 끊기면 유지
토크를 놓기 때문에 **팔이 힘을 잃고 늘어졌다**(실기 확인 2026-07-30 — 트립 시점
`enable status [True]×6`, `Arm Status NORMAL`, `Error Code 0`이라 펌웨어 보호가 아닌
우리 쪽 문제였다). 얼어붙히는 목표는 트립 순간의 *실측* 자세라서 밀린 자세에서
멈추고 사람과 힘겨루기를 하지 않는다. 임계값 0.5 N·m로 강제 트립시킨 뒤 5초간
자세 변화는 **최대 0.56°**였다.

torque는 켜진 상태로 남는다 — 해제는 GUI의 `Safe Torque Release` 또는
`safe_release_torque.py`로 한다.

`park`이어도 후퇴(외력 방향으로 물러나기)나 컴플라이언스 전환은 아직 없다. 따라서
아래 절차는 컷오프의 트리거와 실제 로봇 반응을 확인하기 위한 것이며, 완성된 안전
기능의 검증 절차가 아니다.

### torque 해제 자세 (`PARK_RELEASE_MODE`) — 실기 측정 완료 (2026-07-28)

파킹 자세에서 torque를 풀고 8초간 관절값을 기록한 결과:

| joint | 해제 시 움직인 양 |
|---|---|
| joint1~4, joint6 | **0.00°** (전혀 안 움직임) |
| joint5 (손목) | **-24.4°** |

즉 팔 전체가 떨어지는 게 아니라 **손목만 24.4° 뚝 떨어진다** — 감속비가 커서 큰
관절은 자기 무게로 역구동되지 않는다. "쿵" 하고 놓이는 느낌의 정체가 이 손목
낙차다. 놓기 전에 손목을 미리 그 각도까지 내려두면 해제 시 움직임이 **0.6°**로
줄어드는 것을 확인했다(24.4 → 0.6, 약 40배).

해제 자세는 GUI의 `Torque release`에서 고른다.

- `lower` (기본) — 팔은 그 자리에 그대로 두고 손목만 `PARK_RELEASE_WRIST_REST_DEG`
  (기본 24.4°)까지 `PARK_RELEASE_RAMP_S` 동안 내린 뒤 해제. 팔을 옮기지 않으므로
  이동 위험이 없다. **이 값은 상대 델타가 아니라 절대 각도(자연 정지각)다** —
  상대로 하면 손목이 이미 정지각에 있을 때 그보다 더 아래로 명령했다가 놓는 순간
  그만큼 튕겨 올라온다(실기 확인 2026-07-31: 손목 30°에서 +24.4°를 더 줬다가 해제하니
  29.9°로 복귀). 이미 정지각보다 아래면 그대로 두고 들어올리지 않는다.
- `in_place` — 이동 없이 그 자리에서 해제(손목은 그대로 떨어짐).
- `park` — 기존 동작(파킹 자세로 이동 후 해제).

손목 정지각은 팔 자세에 따라 달라지므로(파킹에서 24.4°, 다른 자세에서 29~30°),
평소 쓰는 종료 자세에서 GUI의 `Measure Wrist Rest`로 다시 재서 `Save as Default`
하는 것을 권장한다(그 버튼은 일부러 "그냥 놓아서" 정지각과 낙차를 재므로 손목이
한 번 뚝 떨어진다).

검증(2026-07-31): 손목을 0°로 들어올린 상태에서 `release_torque_safely()` 실행 →
22.8°까지 내려간 뒤 해제 → **해제 시 움직임 0.00°**, 3초 후에도 그대로.

팔 관절 이동 중에는 gripper를 건드리지 않는다(잡고 있는 물체/손이 끼지 않도록).

### 그리퍼 (실기 확인 2026-07-30)

그리퍼는 팔 모터와 **별개 노드**(CAN 0x159, `GripperCtrl`)라 `DisablePiper()`로는
풀리지 않는다 — "팔은 풀렸는데 그리퍼가 계속 물고 있다"의 원인이었다. 우리 코드가
항상 `code=0x03`(사용+에러클리어)만 보내고 있었기 때문.

- 해제 경로에서 **항상** `GripperCtrl(현재각도, effort=0, code=0x00)`으로 실능시킨다.
  `Slave Torque OFF`(bus.disable_torque)도 같이 그리퍼를 푼다.
- `PARK_RELEASE_GRIPPER_CYCLE=true`(기본, GUI의 `Gripper open/close on release`)면
  풀기 전에 그리퍼를 한 번 열고(`PARK_RELEASE_GRIPPER_OPEN`, 기본 100=완전 열림)
  다시 파킹 위치(닫힘)로 닫은 뒤 실능시킨다 — 물고 있던 물체를 놓고 보관 자세로
  돌리는 용도. 실측: 0.4 -> 98.7 -> 0.4 (정규화값).
- **그리퍼 각도 명령은 팔이 CAN 제어 모드일 때만 먹는다** — `set_action()`은 매번
  `ModeCtrl`을 먼저 보내지만, 그리퍼만 단독으로 움직이는 `cycle_gripper()`는 그게
  없어서 각도가 전혀 안 변했다(실기 확인). 지금은 `cycle_gripper()`가 `ModeCtrl`을
  직접 한 번 보낸다.
- 실능 여부는 눈이 아니라 `status_code` bit[6]으로 확인한다. 한 번만 쏘면 씹힐 수
  있어서 확인될 때까지 `0x00`/`0x02`를 번갈아 최대 6회 재시도한다.
- **주의**: 여는 순간 잡고 있던 물체가 떨어지고, 닫을 때 손가락이 끼일 수 있다.

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
- 컷오프 이후 리더/정책 목표가 로봇에 나가지 않는다(트립 자세만 재전송된다).
- 트립 후 팔이 늘어지지 않고 그 자세를 유지한다(`enable status`가 계속 True).

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
