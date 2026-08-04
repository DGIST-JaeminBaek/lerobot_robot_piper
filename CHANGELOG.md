# 변경 이력

날짜순 기록입니다. "지금 코드가 왜 이런 모양인지"에 대한 주제별 설명은
[`docs/change.md`](docs/change.md)(WEGO 원본 대비)를 참고하세요.

## 2026-08-04

### 추론 경로 통합 + 롤아웃 dataset 기록

추론 실행 경로가 4개로 흩어져 있고 그중 하나(teleop_ui의 `Infer` 프리셋)만
smoothing이 안 걸리던 걸 정리했습니다. 자세한 내용은
[`docs/policy/smoothing.md`](docs/policy/smoothing.md).

- `scripts/tools/piper_infer_runner.py` **신규** — 제어 루프 본체. GUI에 있던
  `InferenceWorker`를 여기로 뺐고, GUI·teleop_ui·CLI가 전부 이 하나를 씁니다.
  lerobot을 우회하지 않습니다 — `LeRobotDataset` / `make_policy` / `PiperFollower`를
  그대로 쓰고, 실물 명령은 전부 `send_action()`을 지나갑니다.
- `teleop_ui.py`의 `Infer` 프리셋이 `lerobot-record --policy.path=...` 대신 이 runner를
  호출합니다. 그 경로는 action chunk를 노출하지 않아 temporal ensemble을 걸 수
  없었는데, 실측에서 TV를 절반으로 줄인 유일한 항목이 ensemble이었습니다.
  **Record / Replay 프리셋은 그대로 `lerobot-record` / `lerobot-replay`를 쓰므로
  영향받지 않습니다.**
- `piper_infer_gui.py`는 runner 위의 화면으로 축소(1018 → 720줄). 실시간 그래프,
  슬라이더, E-STOP은 그대로입니다.
- **모드 프리셋** `demo`(시연용) / `augment`(증강용) 추가. 모드는 프리셋일 뿐이고
  값은 전부 화면에 보이며 개별로 덮어쓸 수 있습니다 — 논문에 실행 조건을 그대로
  옮겨 적어야 하므로 모드 뒤에 값을 숨기지 않습니다.
- **롤아웃 dataset 기록**(증강용). 기록되는 `action`은 raw 출력이 아니라 스무딩을
  거쳐 실제로 `send_action()`에 넘어간 값이고, 카메라는 크롭 전 원본 프레임입니다
  (기존 Record와 같은 형태라 `prepare_erase_shape_dataset.py`로 그대로 변환 가능).
  raw chunk와 실행 조건(스무딩 파라미터 전부 + 실측 제어 주기)은 학습 호환성을
  깨지 않도록 dataset feature가 아니라 `rollout_meta.json` / `raw_actions_ep*.npz`로
  뺐습니다. 종료 시 성공/실패를 물어 같이 남깁니다 — 실패 롤아웃을 걸러내지 않고
  학습시키면 자기 실수를 복제하기 때문입니다.
- `PiperFollowerConfig.action_ema_alpha` 추가(기본 `1.0` = 꺼짐). `send_action()`
  단계 EMA라 텔레옵·Record·Replay 등 chunk가 없는 경로에도 걸리는 바닥입니다.
  안전 클램프(`max_relative_target`)보다 먼저 적용되므로 스무딩이 안전 제한을
  넘길 수 없습니다. rate limit은 `max_relative_target`과 중복이라 따로 두지
  않았습니다.
- `teleop_ui.py`의 `self.python_executable`이 어디에도 정의돼 있지 않아 Sync Player
  경로가 `AttributeError`로 죽던 문제 수정. 이제 GUI를 띄운 인터프리터를 그대로
  씁니다.
- 테스트 39개 추가(`test_infer_runner_mock.py` 27, `test_action_ema_mock.py` 12).
  `scripts/tools/` 전체 98개 통과.

되돌릴 지점: 태그 `inference-before-runner-merge` (커밋 `8c136a3`).

### 실물에서 흔들림 원인을 끝까지 추적

랩 Piper로 검증하면서 찾은 것들입니다. 자세한 실측표는
[`docs/policy/smoothing.md`](docs/policy/smoothing.md).

- **명령 주파수** — `fps=6`이 한계라던 이전 결론이 틀렸습니다. chunk 하나가 이미
  50스텝 분량이라 매 스텝 추론할 필요가 없습니다. `fps=30, infer_every=5`로 바꾸고
  추론을 별도 스레드로 빼서 **실측 5.55Hz → 28.6Hz**(지연 0회, 최대 5ms).
- **클램프 포화** — `max_relative_target`을 smoothing의 rate limit과 같은 값으로
  묶어놨던 탓에, 스텝 200부터 100% 포화되어 명령이 `실측+5`로 대체되고 있었습니다.
  스무딩 결과가 통째로 버려지던 상태입니다. 둘을 분리하고 15로 올려 858회 → 2회.
- **MIT(임피던스) 제어** — `JointMitCtrl` 경로 추가. MOVE J는 목표마다 궤적을
  재계획해서 30Hz 스트리밍에 맞지 않습니다. 기본 꺼짐 + 별도 확인 문구.
  관절별 게인(`mit_kp_overrides`) 지원 — 처짐 = 중력토크/kp라 공통 게인으로는
  못 맞춥니다. `scripts/12__mit_probe.sh`로 관절 하나씩 확인.
- **`ema_alpha`를 켰습니다(1.0 → 0.2).** 스무딩 파이프라인 3단 중 EMA가 내내 꺼져
  있었습니다. 실물 위치 명령이 스텝의 **33.7%에서 방향을 뒤집고** 있었는데, MOVE J는
  플래너가 가려줬지만 MIT는 그대로 재현합니다. α=0.2에서 방향 반전 5.8%,
  이동폭은 그대로(42.57 → 42.28).
- 정지 시 **`park_lower`** 모드 추가 — 보관 자세로 간 뒤 손목까지 내리고 해제.
  예전에는 `PARK_RELEASE_MODE=lower`가 `park=True`를 덮어써서 추론이 끝난 자리에
  팔이 그대로 늘어졌습니다. 중단(SIGINT/SIGTERM) 시 정리를 끝까지 기다리도록
  고쳤습니다(예전엔 30초 타임아웃이라 파킹 도중 데몬 스레드가 잘렸습니다).

버그 수정:
- 주기 대기 루프가 음수 sleep으로 `ValueError`를 던져 제어 루프가 죽고 팔이 park로
  내려가던 문제. 루프가 빨라져 반복이 늘어난 뒤 실물에서 터졌습니다.
- `[TIMING]` 경고가 `step % 30 == 0`과 추론 스텝이 수학적으로 겹치지 않아 죽어 있던 문제.
- Dataset Browser가 `DATASET_ROOT`의 부모만 스캔해 `records/outputs/`의 학습용
  데이터셋이 안 보이던 문제 → `records/` 전체 스캔(200개, 0.03초).
- `source=robot`인데 카메라 crop이 비어 있으면 로봇 연결 후에야 `KeyError`로 죽던 문제
  → `recording.env`에서 읽고, 없으면 시작 전에 거부.
- 정책과 참조 dataset이 안 맞으면 정책 로딩·로봇 연결이 끝난 뒤에야 텐서 크기
  불일치로 죽던 문제 → `config.json`만 읽어 미리 검사하고 학습 dataset을 알려줌.
- `11__infer_gui.sh` 등 4개 스크립트에 conda 활성화가 없어 base에서 실행되며
  `ModuleNotFoundError`로 죽던 문제 → `run_common.sh`에 `activate_conda_env()`로 공통화.

시도했지만 안 된 것:
- **MOVE CPV(`move_mode=0x05`)** — 펌웨어 S-V1.8-2에서 지원되지만 실제로는 팔이
  전혀 움직이지 않았습니다. `JointCtrl`은 위치만 보내는데 CPV는 속도 setpoint가
  필요한 것으로 보이고 SDK에 해당 API가 없습니다.
- **룩어헤드(`--lookahead-s`)** — 목표를 진행 방향으로 앞당겨 보내는 방식.
  0.15초로 실물 확인했으나 체감 개선이 없었습니다. 코드는 남겨뒀습니다(기본 0=꺼짐).

**남은 미검증**: 롤아웃 기록의 `source=robot` 경로(실물 카메라 원본 프레임 저장),
MIT + 속도 피드포워드(`vff=1.0`) 조합, `park_lower` 정지 동작.

## 2026-07-31

랩 PC에 쌓여 있던 작업분을 한 번에 정리해서 올린 회차입니다. 실기 검증이 필요한
항목은 각 절에 그 근거를 적어 두었습니다.

### 추론 시 팔이 떨리는 문제 — smoothing 도구 일습

자세한 내용은 [`docs/policy/smoothing.md`](docs/policy/smoothing.md)에 있습니다.

- `scripts/tools/action_smoothing.py` — temporal ensemble(ACT 방식 `exp(-m·i)`
  가중평균) → EMA → rate limit을 순서대로 적용하는 순수 numpy 모듈. 하드웨어/ROS2/
  lerobot 의존이 없어 오프라인 분석에도 그대로 씁니다. 셋 다 끄면 원본 action이
  그대로 나오므로 baseline 비교가 됩니다. 단위 테스트 19개
  (`scripts/tools/test_action_smoothing.py`).
- `scripts/11__infer_gui.sh` / `scripts/tools/piper_infer_gui.py` — 추론 전용 GUI.
  실행 중에 smoothing 파라미터를 바꿀 수 있고, E-STOP 버튼(`Esc`)과 RViz 궤적
  publish, 최근 300스텝 그래프, TV/jerk 실시간 표시를 포함합니다. 기본은
  `source=dataset` 안전 모드이고, 실물 전송은 3중 게이트를 모두 통과해야 켜집니다.
- `scripts/tools/piper_smoothing_sweep.py` — 하드웨어 없이 `m`을 고르기 위한 스윕.
  설정마다 자식 프로세스를 새로 띄웁니다(한 프로세스에서 SmolVLA를 두 번 로드하면
  CUDA 컨텍스트가 깨져 죽습니다).

`smolvla_erase_shape_512` 30k 체크포인트로 실측한 결과, `m=0.01`에서 total variation이
5.617 → 2.881로 절반이 됩니다. 기존에 흔히 쓰던 `m=1.0`은 25% 감소에 그칩니다.

같이 확인된 것: 이 머신의 SmolVLA 추론이 1회 ~150ms라 `fps=8`은 유지되지 않습니다
(실제 6.65Hz). `fps=6`에서 지터 표준편차 7.5ms로 안정적이라 GUI 기본값을 6으로 뒀고,
`INFER_FPS`로 덮어쓸 수 있습니다. 제어 주기가 들쭉날쭉한 것 자체가 jerk 원인이므로
녹화 FPS(30)를 그대로 쓰지 마세요.

**실기 미검증:** 위 수치는 전부 dataset observation 기반이고 실제 Piper로는 아직
돌리지 않았습니다. `source=robot` 경로의 검증이 남아 있습니다.

### 안전 컷오프 — 트립 후 팔이 늘어지던 문제

`safety_effort_limit`을 넘겨 트립되면 팔에서 힘이 빠지는 현상이 있었습니다. 원인은
트립 처리가 "명령을 아예 끊는" 것이었기 때문입니다. Piper는 `JointCtrl` 목표를 계속
스트리밍해야 자세를 잡는 구조라, 명령이 끊기면 유지 토크도 같이 사라집니다.

- 트립 후 동작을 "명령 차단"에서 "트립 순간의 자세 하나를 계속 재전송"으로 변경.
  리더/정책 명령은 그대로 무시하되 토크는 유지됩니다 (`safety_hold_resend`, 기본 true).
- 얼어붙히는 목표는 마지막 명령 목표가 아니라 트립 순간의 **실측** 자세입니다.
  외력으로 밀린 상태에서 밀리기 전 목표를 계속 쏘면 사람과 힘겨루기가 되어 effort가
  계속 높게 유지되기 때문입니다.
- `safety_on_overload="park"`의 복귀를 최고속에서 램프 이동으로 변경
  (`safety_park_ramp_s`, 기본 4.0초). 파킹 자세가 "팔이 수직으로 뻗은" 자세라
  한 번에 쏘면 트립 직후 팔이 확 뻗는데, 외력 트립 직후에는 사람 손이 팔에 닿아
  있을 수 있어 위험합니다. 0으로 두면 이전 동작이 됩니다.
- 트립 시점의 `GetArmEnableStatus()` / `GetArmStatus()`를 로그에 남깁니다. 늘어짐의
  원인이 우리 쪽 명령 중단인지 컨트롤러 펌웨어 자체 보호인지 구분하기 위한 것입니다.
- parking 스레드가 목표를 소유하는 구간(`_safety_park_active`)을 표시해, 제어 루프와
  parking이 서로 목표를 덮어써 팔이 떠는 것을 막습니다.

### 그리퍼가 종료 후에도 안 풀리던 문제

그리퍼는 팔 모터(0x471)와 별개 노드(0x159)라 `DisablePiper()`로 풀리지 않습니다.

- `disable_gripper()` 추가 — 상태코드 bit[6]으로 실제로 풀렸는지 확인하면서
  `0x00`(실능)과 `0x02`(실능+에러클리어)를 번갈아 재시도합니다. 한 번만 쏘면 프레임이
  씹히거나 드라이버 에러 상태에서 무시되는 경우가 실기에서 확인됐습니다.
- `disable_torque()`가 `disable_gripper()`를 함께 호출합니다.
- `cycle_gripper()` 추가 — torque 해제 전에 그리퍼를 한 번 열고 닫아 물고 있던 것을
  놓고 파킹 위치(닫힘)로 되돌립니다 (`park_release_gripper_cycle`, 기본 true).
  팔 토크를 내리기 **전에** 수행합니다 — 팔이 늘어진 뒤에 여닫으면 반력으로 팔이
  흔들립니다. 주의: 여는 순간 물체가 떨어지고 닫을 때 손가락이 낄 수 있습니다.
- 팔이 CAN 제어 모드가 아니면 그리퍼 각도 명령이 무시되므로 `ModeCtrl`을 먼저 보냅니다.

### 손목 정지각 — 상대 델타에서 절대 각도로

`WRIST_RELEASE_DROP_DEG` → `WRIST_RELEASE_REST_DEG` (설정: `park_release_wrist_drop_deg`
→ `park_release_wrist_rest_deg`, env: `PARK_RELEASE_WRIST_DROP_DEG` →
`PARK_RELEASE_WRIST_REST_DEG`). GUI 버튼도 "Measure Wrist Drop" → "Measure Wrist Rest".

상대 델타로 두면 손목이 이미 정지각 근처일 때 그보다 더 아래로 명령하게 되고, 놓는
순간 그만큼 튕겨 올라옵니다 (실기 확인 2026-07-31: 손목 30도에서 +24.4도를 더 준 뒤
해제하니 29.9도로 복귀). 이미 정지각보다 아래면 그대로 둡니다 — 도로 들어올리면 놓을
때 다시 떨어지기 때문입니다. 정지각은 자세에 따라 달라집니다(파킹 24.4도, 다른 자세
29~30도).

### QC 종료 프레임 기준 선택 옵션

에피소드 종료 프레임을 그리퍼 릴리즈의 어느 쪽에 걸지 고를 수 있게 했습니다
(`--end-event`). 두 시점의 간격은 88개 에피소드에서 평균 16프레임(0.5초), 최대
44프레임(1.5초)으로 실측됐습니다.

- `release_start` (기본값) — 그리퍼가 유지 plateau를 벗어나는 프레임, 마진 6.
  기존 수동 라벨 60개가 여기 있어 기본값으로 유지했습니다 (MAE 2.78프레임, 변경 전과 동일).
- `release_done` — 그리퍼가 다 벌어진 프레임, 마진 0. 지우개를 내려놓는 동작이 컷
  안에 들어옵니다. 느린 에피소드에서는 plateau를 처음 벗어나는 게 팔이 아직 지우개를
  받침대에 정렬하는 중의 부분적 이완이라, 기존 기준은 놓는 동작을 통째로 잘라냈습니다.

`release_done`의 마진을 기존 라벨에 피팅하면 19가 나오지만 일부러 쓰지 않았습니다 —
그 값을 쓰면 컷이 다시 그리퍼가 열리기 전으로 돌아가 옵션의 의미가 사라집니다. 기존
라벨 자체가 `release_start` 기준으로 찍혔기 때문입니다. 새 기준으로 라벨링 세션을
한 번 돌린 뒤에 `--fit`으로 다시 잡아야 합니다.

`qc_studio.py`에는 "종료 기준" 라디오를 추가해 실행 중 전환됩니다 — 확인 완료한
에피소드는 건드리지 않고, 파케이/영상 재읽기 없이 즉시 재계산됩니다. 트레이스에는 두
시점이 항상 함께 표시됩니다. 옵션은 `autofill_frame_ranges.py`, `review_cuts.py`,
`export_cut_plan.py`에도 동일하게 연결했습니다.

검증: 21개 에피소드 전수에서 `release_done`이 모든 에피소드의 컷을 연장(+10~+50프레임),
시작 프레임은 불변. 영상 확인 결과 `release_start` 종료 프레임에서는 그리퍼가 아직
지우개를 물고 있고 `release_done`에서는 벌어진 채 받침대에 놓여 있습니다.

### CAN 재부팅 복구

이 시스템에는 CAN 이름을 고정하는 udev 규칙이 없어 재부팅하면 `can0`/`can1`로 돌아갑니다.

- `scripts/tools/recover_can_after_reboot.sh` — 인터페이스를 bring-up하고 실제 역할을
  판별해 `can_leader`/`can_follower`로 이름을 바꾼 뒤 다시 UP.
- `scripts/tools/detect_can_roles.py` — `ctrl_mode`를 읽어 역할 판별
  (0x06 = Linkage teaching input mode면 leader). USB 포트 순서로 추측하면 틀릴 수
  있어서 실제로 읽습니다. sudo 불필요.
- `scripts/tools/check_startup_push.py` — 시작 직후 팔로워가 아래로 누르는 힘이 생기는
  원인 진단(이동 명령 없이 읽기만). 컨트롤러가 마지막 `JointCtrl` 목표를 기억하기
  때문이라는 가설을 확인하기 위한 것입니다.

### 합성 데이터 파이프라인 (`synthetic/`)

시뮬레이터가 아니라 **컴퓨터가 궤적/action stream을 생성 → 실제 Piper가 실행 → 그동안
카메라·state·action을 재녹화**해 현실 LeRobotDataset을 만드는 방식입니다. 배경은
[`synthetic_data.md`](synthetic_data.md), 설계는 [`synthetic/README.md`](synthetic/README.md).

구성: `kinematics`(FK/IK), `trajectory`(도형 궤적 생성/합성/타이밍), `transforms`
(homography, 이미지 변환), `calibration`(보드 점 선택 — 웹 UI 포함), `preprocessing`,
`preview`(오버레이/플롯/RViz 어댑터). 테스트 194개 전부 통과.

### 기타

- `piper_human_approved_inference.py`에 `--repeat-last-frame` 추가 — dataset source에서
  episode 끝에 도달해도 마지막 frame을 계속 observation으로 재사용합니다(RViz 반복 확인용).
- `piper_session.py`의 conda env를 `piper-gui-refactor` → `ugrp`,
  dataset 경로를 현재 레포 위치로 수정.
- `.gitignore`: `명령어.txt` → `메모장.txt`.
- 학습 로그 추가: `docs/training/logs/` (smolvla erase_shape_512 30000 steps —
  loss curve, metrics csv, tmux 로그).

### 검증 상태

- 자동 테스트 통과: mock 5종(release/safety/smooth_start/dataset_features/effort),
  `synthetic/tests` 194개.
- 실기 확인 (상세는 [`docs/effort/verification_effort.md`](docs/effort/verification_effort.md)):
  - 안전 트립 유지 (2026-07-30) — 임계값 0.5 N·m로 강제 트립 후 5초간 자세 변화
    최대 0.56°. 트립 시점 `enable status [True]×6`, `Arm Status NORMAL`, `Error Code 0`
    이라 펌웨어 보호가 아닌 우리 쪽 명령 중단이 원인임을 확인.
  - 그리퍼 실능/사이클 (2026-07-30).
  - 손목 정지각 (2026-07-31) — 손목 0°에서 해제 시 22.8°까지 내려간 뒤 **움직임 0.00°**.
- QC 종료 기준 옵션 — 실제 에피소드 21개 전수 + 영상 대조.
