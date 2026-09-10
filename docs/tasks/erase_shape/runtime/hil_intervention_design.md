# HIL(사람 개입) + 개입 데이터 학습 설계 — 보류

관련 코드: `scripts/tasks/erase_shape/runtime/erase_run.py --hil`, `hil_clutch.py`,
`erase_hil_panel.py`

**지금은 안 쓴다.** 평가 툴은 게이트(1회 판정)만 쓰고 HIL 개입은 하지 않는다 —
현재 평가 설계는 [erase_run_design.md](erase_run_design.md)를 볼 것. 이 문서는 리더암
클러치 인계, 개입 토글, 개입 데이터로 재학습(DAgger 계열)에 대한 설계를 남겨둔
것이다 — 코드(`--hil`)는 살아있고 동작 확인도 됐지만(§3), 지금 평가 세션에서는
켜지 않는다.

---

## 1. HIL 개입

### 1.1 키보드 토글 채택

처음에는 리더암 움직임으로 개입을 자동 감지하려 했다. 실패했고, 그 과정에서
mock 테스트가 버그 두 개를 잡았다:

1. **이탈 판정에 편차를 쓰면 안 된다.** 개입이 시작되면 팔로워가 리더를 따라가므로
   편차가 0으로 수렴 → 사람이 잡고 있는데 즉시 반환.
2. **진입 판정에 편차만 쓰면 안 된다.** 손을 뗀 뒤 리더가 그 자리에 남으면 정책이
   팔로워를 몰고 가면서 편차가 다시 커짐 → 무한 재개입.

고칠 수는 있지만 임계값 3개(`takeover_th`, `still_speed`, `still_steps`)를 실측으로
맞춰야 하고 중력 처짐에 따라 재조정이 필요하다.
**명시적 키보드 토글은 이 전부를 없앤다.** 오검출이 원리적으로 0이다.

LeRobot HIL-SERL도 같은 선택을 한다 — `space`로 정책에서 인계, 다시 `space`로 반환.
`pynput`은 이미 리포에서 쓰고 있다(`teleop_ui.py`).

현재 매핑: **`space` = 개입 on/off, `q` = 시도 중단.**

> 양손이 리더암에 있으면 키를 못 누른다. 거슬리면 USB 풋페달이 표준 해법이다.

### 1.2 ★ PiPER 고유 제약 — 리더암은 명령을 받을 수 없다

리포 소스에서 확인한 사실 ([piper_leader.py](../../../../lerobot_robot_piper/piper_leader.py) `connect()` 주석):

> Piper 공식 dual-arm 문서: "Master arm: only sends control frame messages."
> 즉 leader(master) 팔은 EnableArm 대상이 아님 — 사람이 손으로 움직이는 값을 CAN으로
> 흘려보내기만 하고, 모터 서보 활성화가 필요 없음(**활성화 시도 시 계속 타임아웃남,
> 실제 하드웨어에서 확인됨**).

이게 왜 중요하냐면 — LeRobot HIL-SERL 문서는 SO101 리더를 권장하면서
**"reduced gears가 있어 정책 실행 중 리더가 팔로워를 추종할 수 있고, 그래서 인계가
부드럽다"** 고 명시한다. PiPER는 이게 **불가능하다.** 리더를 구동할 수 없으니까.

따라서 개입 시점에 리더와 팔로워는 **반드시 어긋나 있다.** 24초간 정책이 팔로워를
몰고 다녔으니 어긋남이 클 수도 있다.

### 1.3 해법: 델타(클러치) 인계 — park 정렬 불필요

세 가지 선택지를 검토했다.

| 방식 | 평가 |
|---|---|
| park 갔다가 리더 각도 맞추고 재개 | 동작은 한다. 그러나 **개입의 목적을 무너뜨린다** — 실패가 벌어지는 순간에 끼어들어야 하는데, park를 거치면 그 순간이 지나간다. 매 개입마다 수 초 + 사람이 정렬해야 함 |
| 리더 절대 자세를 그대로 전송 | 팔로워가 튄다. `MAX_RELATIVE_TARGET=5.0` 클램프가 물리적으로는 막지만, 그만큼 스텝당 5씩 미지의 경로로 기어간다 |
| **리더 변화량만 전송 (클러치/인덱싱)** | **채택.** 점프가 원리적으로 0 |

```
engage 시점:  lead0 = 리더 자세,  foll0 = 팔로워 자세
개입 중:      target = foll0 + gain × (leader_now − lead0)
```

인계 첫 스텝의 target이 팔로워 현재 자세와 정확히 같으므로 **어긋남이 아무리 커도
점프가 없다.** 마우스를 들었다 놓는 것(인덱싱), 자동차 클러치와 같은 원리다.
테스트 `test_clutch_no_jump_on_engage`가 60만큼 어긋난 상태에서 이를 검증한다.

트레이드오프:
- 리더의 절대 자세가 팔로워의 절대 자세와 더 이상 대응하지 않는다.
  짧은 교정 개입에는 문제없지만, 긴 개입에서는 사람의 감각과 어긋날 수 있다.
- `gain < 1`로 두면 정밀 보정이 쉬워진다(`--clutch-gain`).
- 개입을 껐다 켜면 기준점이 다시 잡힌다(`test_clutch_reengage_uses_new_reference`).

### 1.3.1 그래도 리더암을 팔로워와 비슷한 자세로 두는 게 좋다 — 안전이 아니라 조작감 때문

**필수는 아니다.** 클러치 방식이라 어긋남이 아무리 커도 점프는 0이다.
그래도 space를 누르기 전에 리더를 팔로워와 대충 맞춰두면 세 가지가 좋아진다.

1. **직관적 대응.** 관절 공간 델타라서 관절별 각도 변화는 정확히 1:1로 전달되지만,
   **같은 관절 변화가 만드는 손끝 움직임은 자세에 따라 다르다.** 리더가 팔로워와
   비슷한 형상이어야 "앞으로 민다"가 팔로워에서도 앞으로 나간다.
2. **가동 여유(headroom).** 리더 관절이 자기 한계 근처에 있으면 그 방향으로 더 못 민다.
   반대로 팔로워가 한계 근처면 리더를 밀어도 클램프에 걸린다.
   대충 맞춰두면 양쪽 다 여유가 최대가 된다.
3. **누적 오차.** 개입이 길어질수록 델타가 쌓인다. 시작이 비슷하면 어긋남이 덜 커진다.

**실전 절차**: 정책이 도는 걸 보다가 개입이 필요해지면 →
리더암을 눈대중으로 팔로워 자세에 맞춘 뒤 → `space` → 조작 → `space`로 반환.
맞추는 동안 팔로워는 정책이 계속 몰고 있으므로 완벽할 수 없다. 대충이면 된다.

> 로그의 `engage_deviation`이 인계 시점의 실제 어긋남을 기록한다.
> 이 값이 매번 크면 "개입이 필요할 것 같을 때 미리 리더를 근처에 두는" 습관이 필요하다는 뜻이다.

`deviation()`은 판정에서 빠졌지만 **인계 시점 값을 로그에 남긴다**(`engage_deviation`).
나중에 자동 감지를 다시 시도할 때 실측 근거가 된다.

### 1.4 LeRobot HIL-SERL — 재사용 가능한 것과 아닌 것

LeRobot에 HIL-SERL이 이미 포팅되어 있다(`lerobot.rl.gym_manipulator`, PR #644).
**직접 재사용은 어렵지만, 왜 어려운지가 우리 설계 근거가 된다.**

| 항목 | HIL-SERL | 우리 | 재사용? |
|---|---|---|---|
| 개입 토글 | `space` | `space` | 설계 차용 ✓ |
| 개입 장치 | gamepad / keyboard / **leader** (`control_mode: "leader"`) | PiperLeader | 개념 동일 |
| 액션 공간 | **EE delta (x,y,z)**, URDF+IK 필수 | joint-space 정규화 | ✗ |
| 제어 주기 | 10 fps | 30 fps | ✗ |
| 권장 horizon | **5~10초** | 24초 | ✗ |
| 판정기 | resnet10 학습 분류기, 이진, threshold 0.5 | 룰베이스 픽셀 metric | 우리가 더 정확 |
| 종료 | `terminate_on_success: true` | 우리 게이트 | 개념 동일 ✓ |
| 학습 | SAC actor-learner (gRPC 2프로세스) | SmolVLA flow matching | ✗ |
| 이미지 | crop ROI → 128×128 | 640×480 | 조정 필요 |

특히 문서가 명시하는 대목: *"learning in joint space for reinforcement learning in
manipulation is often a harder problem — some tasks are nearly impossible to learn in
joint space but become learnable when the action space is transformed to end-effector
coordinates."* 우리는 joint-space이고 URDF/IK 경로가 정비되어 있지 않다.

**결론: HIL-SERL 파이프라인에 올라타지 않고, 자체 러너(`erase_run.py`)로 간다.**
차용하는 것은 (a) space 토글, (b) `terminate_on_success` 개념,
(c) intervention rate를 학습 진척 지표로 보고하는 관행.

### 1.5 추론 3방식(`--aggregate-fn`)이 게이트·HIL과 어떻게 얽히는가

`--aggregate-fn`은 정책이 뱉은 chunk들을 **겹치는 구간에서 어떻게 합칠지**를 정한다.
구현은 `action_smoothing.py`.

| 값 | 수식 | 출처 |
|---|---|---|
| `weighted_average` (기본) | `0.3·old + 0.7·new`, 새 chunk가 올 때마다 pairwise 적용 | lerobot async_inference |
| `latest_only` | `new` — 최신 chunk가 이전 것을 그냥 덮는다 | lerobot async_inference |
| `temporal_ensemble` | 겹치는 예측 **전부**를 `exp(-m·i)`로 한 번에 평균 (m=0.01 ≈ 균등) | ACT |

#### 판정·잔량 추정에는 영향이 없다 (구조적으로)

판정은 시도 경계의 park 프레임 한 장에서 카메라로 잰다
([erase_run_design.md](erase_run_design.md)). 액션 파이프라인을 전혀 지나지
않으므로 `aggregate_fn`과 경로가 겹치지 않는다. `residual()`도 같은 프레임을 쓴다.

단, **간접 경로는 있다** — 방식이 궤적의 부드러움을 바꾸고, 그게 지우개가 보드에
제대로 닿는지를 바꾸고, 그래서 `erased_frac`의 **값**이 달라진다. 측정 방법이 아니라
측정 대상이 달라지는 것이다.

> ⚠️ 평가할 때의 함의: 조건 간 비교표에서는 `aggregate_fn`을 **반드시 고정**해야
> 한다. 안 그러면 게이트의 효과와 스무딩 방식의 효과가 섞인다.

#### HIL에는 영향이 있다 — 개입이 아니라 **반환** 시점에

인계(engage)는 방식과 무관하게 안전하다. 클러치가 `foll0 + gain·(leader − lead0)`로
첫 스텝 목표를 팔로워 현재 자세와 같게 만들기 때문이고, 60만큼 어긋난 상태에서
`test_clutch_no_jump_on_engage`가 이를 검증한다(§1.3).

문제는 반환이다. 개입 중 러너는 **매 스텝 `pipeline.reset(action)`** 을 호출한다
(`piper_infer_runner.py`, "개입 중 스무딩 파이프라인은 계속 정책 궤적을 밀고 있으므로"
주석). 이건 옳은 처리다 — 안 하면 반환 첫 스텝에 목표가 튄다. 그런데 `reset()`은
**앙상블 버퍼를 통째로 비우므로**, 반환 직후에는:

1. `pending_steps == 0` → 루프가 `_await_chunk()`로 **추론 1회(약 115ms ≈ 3~4스텝)를
   기다린다.** 개입을 껐다 켤 때마다 이 정지가 생긴다.
2. 새 chunk가 하나뿐이라 `votes == 1` → **겹침 평균이 없다.** 즉 `docs/policy/smoothing.md`가
   확립한 스무딩이 하필 **가장 튀기 쉬운 순간에 빠진다.**

여기서 방식별로 갈린다:

| 방식 | 반환 직후 잃는 것 | 회복 |
|---|---|---|
| `temporal_ensemble` | 가장 크다. 평상시 여러 표를 균등 평균(m=0.01)하다가 1표로 떨어진다 | chunk가 몇 번 겹쳐 쌓일 때까지 |
| `weighted_average` | 중간. pairwise라 표 수에 덜 민감하다 | 빠름 |
| `latest_only` | 없다. 원래 항상 최신 1개만 쓴다 | 즉시 |

**역설적인 결론**: HIL을 많이 쓰는 실험에서는 가장 부드러운 방식(`temporal_ensemble`)이
반환 시 가장 거칠어진다. 반대로 개입이 없는 순수 게이트 실험에서는 이 항이 0이라
아무 차이가 없다.

**아직 실물로 확인 안 했다.** 확인에 필요한 데이터는 `erase_run_log.steps.npz`의
`attempt{i}_votes`와 `attempt{i}_intervention`이다(둘 다 저장하도록 해 뒀다).
반환 프레임을 기준으로 votes가 1에서 회복되는 데 걸린 스텝 수를 방식별로 세면 된다.

---

## 2. 개입 데이터로 학습 — 전제 조건

아직 구현하지 않았다. 착수 전에 알아야 할 것들.

### 2.1 DAgger 계열의 핵심 문제

- **covariate shift**: BC로 학습한 정책은 자기 실수가 만든 상태를 본 적이 없다.
  DAgger는 학습자의 정책으로 굴러간 상태에서 전문가 라벨을 모아 이를 메운다.
  정책이 부분적으로 지운 상태를 다시 손보지 못하는 것이 정확히 이 문제다.
- **HG-DAgger (human-gated)**: 사람이 게이트 역할을 하고 개입 구간만 데이터로 쓴다.
  우리 설계와 같다. 한계로 지적되는 것은 **1:1 상시 감독이 필요해 확장성이 없다**는 점.
- **ConRFT가 HG-DAgger를 깐 이유**(p.6): 사람 교정이 일관성 없고 노이즈가 많아
  정밀 접촉 과제(Insert Wheel, Hang Chinese Knot)에서 개선이 제한적이었다.
  Table I에서 HG-DAgger는 에피소드 길이를 1.1×밖에 못 줄였다(HIL-ConRFT는 1.9×).
  → **리더암은 게임패드보다 노이즈가 작다**는 게 우리 반론 근거다.

### 2.2 우리 경우의 추가 제약

- **자동 리셋 불가.** 지운 마카를 되돌릴 수 없다. ConRFT는 스크립트 모션이나 사람 손으로
  리셋했고, RL-100은 이를 미해결 한계로 명시했다("learned or semiautomated reset
  strategies"). 우리는 매 에피소드 사람이 다시 그린다.
- **credit assignment.** 30fps × 720스텝이면 에피소드당 결정이 720개다.
  HIL-SERL이 다루는 10fps × 5~10초(50~100스텝)보다 한 자릿수 크다.
  sparse reward만으로 RL을 돌리기에는 불리하다.
- **데이터 형식.** 개입 궤적을 LeRobotDataset으로 저장하려면 `add_frame`/`save_episode`
  경로를 타야 하고, `intervention` 플래그를 어느 필드에 넣을지 정해야 한다.
  현재 `erase_run.py`는 JSON 로그만 남긴다 — 데이터셋 변환은 별도 작업.

### 2.3 가장 게으른 경로 (권장)

RL로 가기 전에 **BC 재학습부터** 시도한다:
개입 궤적 + `erase_check`의 성공 라벨 → 기존 데이터셋에 추가 → SmolVLA fine-tune.
이게 iRe-VLA / RLDG가 보고한 "RL로 모은 데이터를 SFT에 쓴다" 구조이고,
ConRFT Table III에서 RLDG가 SFT 대비 크게 앞선 결과가 있다(83.3 vs 58.3).
파이프라인이 이미 있으니 추가 인프라가 거의 없다.

---

## 3. 검증 절차 (완료 — 지금은 켜지 않을 뿐 동작은 확인됨)

### 3.1 개인 PC

```bash
python scripts/tests/tasks/erase_shape/evaluation/test_erase_run_mock.py
```
8개 통과. 하드웨어·cv2·lerobot 불필요.

| 테스트 | 검증 대상 |
|---|---|
| `test_no_hil` | 개입 없을 때 플래그 False, park 강제 |
| `test_abort_stops_attempt` | `q` 중단 시에도 park 보장 |
| `test_clutch_no_jump_on_engage` | **60만큼 어긋난 상태에서 인계 점프 0** |
| `test_clutch_tracks_leader_delta` | 리더 변화량만 반영 |
| `test_clutch_gain` | gain 축소 반영 |
| `test_clutch_reengage_uses_new_reference` | 재인계 시 기준점 갱신 |
| `test_intervention_logged_and_no_jump_in_loop` | 루프 통합, 개입 구간 기록, `engage_deviation` |
| `test_release_returns_control_to_policy` | 반환 후 정책이 다시 몬다 |

### 3.2 랩 PC (`piper` env) — HIL 전용 단계

기본 게이트가 먼저 정상 동작하는 걸 확인한 뒤에 진행한다
([eval_session_howto.md](../evaluation/eval_session_howto.md) §2, §4).

**리더암 읽기 점검 (로봇에 명령 안 보냄)**
```bash
bash scripts/1__init_can.sh
python scripts/tasks/erase_shape/runtime/erase_run.py --probe-leader --follower-port can_follower --leader-port can_leader
```
값이 갱신되는지, 손으로 움직이면 따라 바뀌는지 확인.
클러치 방식이라 편차가 커도 인계는 안전하다 — 이 단계는 배선/정렬 확인용이다.

**HIL 실행 ★ 사용자 입회**
```bash
python scripts/tasks/erase_shape/runtime/erase_run.py ... --hil --confirm
```
확인 항목:
- `space` 누른 순간 팔로워가 **튀지 않는가** (클러치 검증의 실기 확인)
- 리더암을 움직인 만큼만 팔로워가 따라오는가
- `space`로 반환 후 정책이 정상 재개하는가
- 껐다 켜기를 반복해도 튐이 없는가
- `q`로 중단 시 park로 복귀하는가
- 로그의 `engage_deviation` 값 — 실제로 얼마나 어긋난 상태에서 인계했는가

---

## 4. 미해결 / 리스크

| 항목 | 상태 |
|---|---|
| 잔여 잉크 위치를 정책에 전달하는 방법 | **통로 없음** — 리커버리 데이터 없이는 해결 안 됨 |
| 개입 궤적 → LeRobotDataset 변환 | 미구현 |
| "덜 지운 부분을 끝내 안 닦는다" 증상 | **게이트만으로는 안 고쳐진다.** 리커버리 데이터 필요 |
| HIL 반환 시 스무딩 앙상블 버퍼 초기화 영향 | 실물 미확인(§1.5) |
