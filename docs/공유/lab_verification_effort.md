# 랩 PC 실기 검증 절차 — effort 녹화 (GUI 기준)

> 대상: 랩 PC에서 실물 로봇으로 검증하는 사람
> 작성: 조성일 · 2026-07-25 · 검증 대상 커밋 `cc0ad67`
> 예상 소요: **약 20분** (카메라 warmup 포함)

---

## 0. 이 검증의 범위

### 검증 대상 — effort 로깅만

```
✅ effort/velocity가 데이터셋에 실제로 들어가는가
✅ 셸(5__record.sh)·GUI 두 경로 모두에서 들어가는가
✅ Smooth Start가 effort를 덮어쓰지 않는가
✅ 안전 컷오프가 살아있는가
```

### 검증 대상 아님 — depth

**depth는 이 브랜치가 건드리지 않았다.** jmbaek의 12-bit 구현을 그대로 채택했고,
우리 쪽 depth 코드(turbo 컬러맵)는 전량 삭제했다. **이번 검증에서는 depth를 끄고 한다**
(변수를 줄이기 위해서다. depth 동작 자체는 `docs/depth/README.md` 담당).

### 바뀐 것 요약

| 파일 | 무엇이 바뀌었나 |
|---|---|
| `scripts/lib/run_common.sh` | `robot_observation_args()` 추가 — 셸 경로에도 `--robot.use_effort` 전달 |
| `scripts/5__record.sh`, `9__run_client.sh` | 위 인자 배선 |
| `lerobot_robot_piper/config_piper.py` | `use_effort` 기본값 `False` → **`True`** |
| `lerobot_robot_piper/teleop_ui.py` | "Record Effort" 체크박스가 Command를 갱신하도록 수정 |
| `scripts/tools/smooth_start_frames.py` | `.pos` 컬럼만 보간 (effort/vel 보존) |
| `scripts/tools/check_effort.py` | **신규** — 검증 스크립트 |

---

## 1. 사전 준비

### 1-1. 브랜치

```bash
cd ~/UGRP/lerobot_robot_piper      # 실제 clone 경로에 맞출 것
git fetch --all
git checkout seongil/gui-refactor
git pull
git log --oneline -1               # cc0ad67 이어야 함
```

> ⚠️ 랩 PC clone의 `origin`이 upstream(DGIST-JaeminBaek)을 가리키고 있다면 그대로
> `git pull`하면 된다. 이 브랜치는 upstream에도 올라가 있다.

### 1-2. lerobot 확인

이 브랜치는 **jmbaek의 depth 백포트가 적용된 lerobot**을 필요로 한다
(`piper_follower.py`가 `lerobot.datasets.depth_utils`를 import).

```bash
conda activate ugrp                # 랩 PC에서 쓰는 env 이름에 맞출 것
python -c "import lerobot, lerobot.datasets.depth_utils as d; print(lerobot.__file__); print('depth 백포트 OK')"
```

**실패하면** stock lerobot이 잡힌 것이다. jmbaek의 clone(`~/UGRP/lerobot`)이
editable install 되어 있는지 확인할 것. 이게 안 되면 아래 전 과정이 진행 불가다.

### 1-3. recording.env

```bash
USE_EFFORT=true              # 기본값이 true라 없어도 되지만 명시 권장
REALSENSE_USE_DEPTH=false    # 이번 검증에서는 depth OFF
NUM_EPISODES=2
EPISODE_TIME_S=20
RESET_TIME_S=10
SAFETY_EFFORT_LIMIT=15.0     # 튜닝 전이므로 넉넉하게 (§5에서 확정)
```

---

## 2. CAN 연결

```bash
ip link show | grep can        # 인터페이스 존재 확인
bash scripts/1__init_can.sh    # 필요 시
bash scripts/0__launch_gui.sh
```

GUI에서:

1. **CAN Setup** 패널 → `Detect`
2. leader / follower 인터페이스가 각각 잡히는지 확인
3. `Init All` (또는 개별 `Init`)

> CAN 버스는 공유 자원이다. 다른 사람이 쓰는 중인지 먼저 확인할 것.

---

## 3. CAN 통신 확인 (로봇 안 움직임)

**CAN Monitor** 패널 → `Start Monitor`

- Joint Positions 표의 값이 실시간으로 갱신되면 통신 정상
- 리더 팔을 손으로 조금 움직여서 값이 따라 변하는지 확인

확인 후 **`Stop Monitor`를 반드시 누른다.**
모니터가 CAN을 잡고 있으면 녹화와 충돌한다.

> 이 패널은 **위치만** 보여준다. effort는 §5에서 데이터로 확인한다.

---

## 4. ⭐ Command 창 육안 확인 — 제일 중요

**이 단계가 전체 검증의 90%다. 여기서 걸러야 20분을 안 버린다.**

1. **Preset** 드롭다운 → **`Record`** 선택
2. **Task** 칸에 영어로 입력 (예: `effort check`)
3. **"Record Effort" 체크박스가 켜져 있는지** 확인
4. **Command 입력창을 좌우로 스크롤**해서 아래를 눈으로 확인:

```
--robot.use_effort=true          ← 없으면 절대 Launch하지 말 것
--dataset.num_episodes=2
--robot.safety_enabled=true
--robot.safety_effort_limit=15.0
```

### `use_effort=false`로 보인다면

체크박스를 껐다 켜보고, 그래도 안 되면 `recording.env`의 `USE_EFFORT=true`를
확인하고 GUI를 재시작한다.

> 참고: 예전에는 체크박스를 켜도 Command가 갱신되지 않는 버그가 있었다.
> 이번 커밋에서 고쳤고, **이 단계가 바로 그 수정을 검증하는 것이다.**

---

## 5. 짧게 2 에피소드 녹화

1. `Launch`
2. **카메라 warmup으로 20초 정도 응답이 없다 — 정상이다.** 기다린다
3. 진행률에 `Recording episode 1/2`가 뜨면 시작된 것
4. 텔레옵으로 팔을 움직인다

### ⚠️ 반드시 포함할 동작

**지우개(또는 팔 끝)를 보드에 실제로 눌러보는 동작을 넣을 것.**
허공에서만 움직이면 effort가 중력 성분만 나와서 §7의 임계값 튜닝을 못 한다.

5. 20초 후 자동으로 Reset 구간 → 다시 20초 → 2 에피소드 후 자동 종료
6. 종료 시 파킹 자세로 이동 + 토크 해제

### 이번 검증에서 쓰지 말 것

- **`Auto-stop at Parking`** — 2번째 에피소드부터 잘리는 버그가 남아 있다(별건)
- **`End Episode (Save)` 버튼** — 시간 만료로 넘어가면 충분하다

전체 로그는 `last_launch.log`에 남는다.

---

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
✅ 초반 프레임 effort 정상 (덮어쓰기 흔적 없음)
```

### ❌ 실패 패턴

| 출력 | 원인 | 조치 |
|---|---|---|
| `차원 7`, `effort 없음` | 플래그 미반영 | §4를 다시 확인. 이 커밋이 맞는지 `git log -1` |
| `effort가 전부 0` | CAN에서 값을 못 읽음 | 팔 전원·CAN 확인. **스키마만 맞고 데이터는 무용** |
| `초반 100프레임 선형` | Smooth Start 오염 | 구버전 코드일 가능성. `git log -1` 확인 |

---

## 7. 안전 컷오프 임계값 결정

`check_effort.py` 출력 마지막에 관절별 `|effort|` 최댓값과 권장값이 나온다.

```
=== 관절별 |effort| 최댓값 (SAFETY_EFFORT_LIMIT 튜닝용) ===
  joint1.effort   ...
  → 이 동작이 '정상 지우기'였다면 SAFETY_EFFORT_LIMIT ≈ N 권장
```

**§5에서 실제로 누르는 동작을 했을 때만 유효하다.**

⚠️ **무접촉 자유운동 기준으로 잡으면 안 된다.** 지우기는 누르는 힘이 본질이라
자유운동 노이즈 기준으로 잡으면 **정상 지우기 도중에 팔이 멈춘다.**

권장값을 `recording.env`의 `SAFETY_EFFORT_LIMIT`에 반영한다.

---

## 8. 셸 경로도 확인 (선택, 5분)

GUI 말고 `5__record.sh`로도 effort가 들어가는지 확인한다.
(이번 커밋의 핵심 수정 중 하나 — 예전엔 셸 경로에서 effort가 통째로 빠졌다.)

```bash
DRY_RUN=true bash scripts/5__record.sh 2>&1 | grep use_effort
```

`--robot.use_effort=true`가 출력되면 통과.

---

## 9. 안전 컷오프 동작 확인 (선택, 주의 필요)

> ⚠️ **로봇이 실제로 움직인다. 사람이 비상정지에 손을 두고 진행할 것.**
> 자신 없으면 건너뛰어도 된다 — §6까지 통과했으면 로깅 검증은 끝난 것이다.

`SAFETY_EFFORT_LIMIT`을 일부러 낮게(예: 2.0) 두고 텔레옵으로 가볍게 눌러본다.
`last_launch.log`에 아래 경고가 찍히고 팔이 그 자리에서 멈추면 정상이다.

```
safety cutoff: effort {...} exceeds 2.0 N·m, holding last commanded position
```

확인 후 **반드시 §7에서 정한 값으로 되돌릴 것.**

---

## 10. 검증 완료 체크리스트

- [ ] `git log -1`이 `cc0ad67` (또는 그 이후)
- [ ] `lerobot.datasets.depth_utils` import 성공
- [ ] Command 창에 `--robot.use_effort=true` 확인
- [ ] `check_effort.py`가 ✅ 3개 전부 출력
- [ ] 관절별 effort 최댓값이 물리적으로 말이 되는 범위 (수 N·m 수준)
- [ ] `SAFETY_EFFORT_LIMIT` 값 결정 및 `recording.env` 반영
- [ ] (선택) 셸 경로 `DRY_RUN` 확인
- [ ] (선택) 안전 컷오프 실동작 확인

---

## 문제가 생기면

1. `last_launch.log` 전문
2. `check_effort.py` 출력 전문
3. `git log --oneline -1`
4. `python -c "import lerobot; print(lerobot.__file__)"`

이 4개를 조성일에게 전달하면 된다.

---

## 참고

- effort 배경·설계·연구 맥락: `docs/effort_guide.md`
- VLA 모델별 FPS/해상도: `docs/vla_fps_resolution.md`
- depth(담당 아님): `docs/depth/README.md` (jmbaek), `docs/depth_guide.md` (조사 기록)
