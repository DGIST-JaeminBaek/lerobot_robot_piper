# Claude Code 작업 인계서: 현실 실행 기반 합성 데이터 파이프라인 — 2단계

## 1. 이 문서의 용도

새 대화창에서 `synthetic/` 작업을 이어서 맡기기 위한 인계서다. 이전 인계
(6A~6E, offline 코드 골격)는 끝났고, 이 문서는 그 다음 단계를 다룬다.

작업을 시작하기 전에 다음을 **이 순서로** 읽는다.

1. `synthetic/README.md` — 지금까지 구현된 모든 모듈의 현황·사용법·테스트
   결과 (11~16장이 이번 인계 내용). **가장 먼저, 전체를 읽는다.**
2. `synthetic/NEXT_STEPS.md` — 사용자가 실물에서 측정/확인해야 하는 항목
   체크리스트. 지금 이 인계의 실제 다음 작업은 대부분 이 체크리스트의
   진행 상황에 달려 있다.
3. `synthetic_data.md` — 최초 아이디어와 배경(참고용, 필수는 아님).
4. 이 문서.

## 2. 지금까지 완료된 것 (요약)

`synthetic/` 아래에 다음이 전부 구현되고 테스트됐다 (2026-07-29 기준
194 tests, `python -m unittest discover -s synthetic/tests -v`로 재확인
가능). 상세 설명과 사용법은 README.md에 있으므로 여기서는 반복하지 않는다.

| 모듈 | 역할 |
|---|---|
| `calibration/` | TOP 이미지 ↔ board 평면 homography |
| `preprocessing/` | VLA별 이미지 crop/resize profile (원본 pixel 고정) |
| `transforms/` | board 평면 ↔ Piper base rigid transform solver (Kabsch) |
| `trajectory/` | segment 스키마, 원/삼각형/사각형 경로, Cartesian 합성 |
| `kinematics/` | `piper_sdk` FK wrapper, `ikpy` 기반 IK, 7차원 action 변환 |
| `preview/` | validation report, plot, overlay, mock RViz, `generate_preview.py` CLI |

**중요**: 지금까지 코드에 들어간 board 크기, 기준점, 접촉/비접촉 높이,
tool 자세, IK seed는 전부 가상(illustrative) 값이다. 실측이 아니다. 모든
calibration/config는 `status="unverified"`이고, `generate_preview.py`가
만드는 report는 항상 `real_execution_allowed=False`다. 이건 버그가
아니라 의도된 안전장치이니 실측 전에는 건드리지 않는다.

## 3. 작업 시작 전 확인할 것

1. `git status --short` — synthetic과 무관한 변경이 있을 수 있다 (예:
   `.gitignore`, `scripts/tools/piper_human_approved_inference.py` 수정,
   `docs/training/logs/` 등). **손대지 않는다.**
2. `synthetic/NEXT_STEPS.md`의 체크리스트를 사용자에게 어디까지
   진행했는지 확인한다. 대화로 물어보는 게 가장 정확하다.
3. `synthetic/outputs/`에 실측 기반 파일이 새로 생겼는지 본다 (예:
   `board_points_epNNN.json`이 여러 개, `--unit mm`으로 저장돼 있는지).
   `synthetic/outputs/`는 재생성 가능한 산출물이라 git에 없으므로, 이전
   대화의 산출물이 남아있지 않을 수 있다 — 없다고 이상한 게 아니다.

## 4. 다음 작업 — 실측 진행 상황에 따라 분기

### 4A. 아직 `NEXT_STEPS.md` 체크리스트를 다 못 채운 경우

코드 작업이 아니라 안내가 우선이다. 사용자와 남은 항목을 확인하고,
필요하면 이미 있는 도구(`select_board_points_web.py`,
`aggregate_board_points.py` 등) 사용법을 다시 안내한다. 실측값 없이
`board_base.example.json`이나 `board_motion_config`의 예시 값을 "그럴듯해
보인다"는 이유로 실측인 것처럼 코드에 반영하지 않는다.

### 4B. 실측 데이터(board 크기, 기준점 correspondences, 접촉/비접촉 높이,
tool 자세, IK seed)가 준비된 경우

1. `synthetic/transforms/solve_board_base.py`로 실제 board↔base
   transform을 계산한다. correspondences 입력은
   `synthetic/configs/board_base.example.json`과 같은 스키마를 쓰되
   실측값을 넣는다.
2. `synthetic/trajectory/compose.py`의 `BoardMotionConfig`를 실측
   hover/contact height, `tool_rpy_deg`로 채운 JSON을 만든다.
3. `synthetic/preview/generate_preview.py`를 실제 값으로 실행해 사람이
   검토할 overlay/plot/validation report를 생성한다.
4. reprojection error, IK residual, 도달 범위가 기대한 만큼인지 사용자와
   같이 확인한다. 이상하면 실측이나 관례(convention) 오류를 먼저
   의심하고, 통과시키려고 코드의 허용오차나 검증 로직을 임의로 완화하지
   않는다.
5. GRASP/PLACE 실제 template(사람이 녹화한 teleop 시연)을
   `synthetic/trajectory/compose.py`의 `require_recorded_template()`에
   연결해 11-segment 전체 조합을 시도한다. **이 template을 실제 녹화
   파일(LeRobotDataset 등)에서 읽어오는 로더가 아직 없다** — 사용자에게
   그 녹화가 어떤 형식으로 저장되는지 먼저 확인하고, 기존 record
   파이프라인과 같은 포맷이면 그걸 읽는 얇은 adapter만 추가한다
   (기존 파이프라인 코드 자체는 수정하지 않는다, 6절 참고).

### 4C. 4B가 offline로 안정적으로 검증된 뒤: 단계 6 착수 여부

`synthetic/README.md`의 "단계 6. 사람 승인형 실물 실행 adapter" 절에 이미
설계가 정의돼 있다 (구현 내용/완료 조건 포함). 이 단계부터는 실제
`PiperFollower.send_action()`을 호출하게 되므로:

- **사용자의 명시적 승인 없이는 시작하지 않는다.** "4B까지 끝났으니
  당연히 다음은 6단계"라고 스스로 판단해서 넘어가지 않는다.
- 시작하더라도 `offline`/`rviz-only`/`real` 모드 분리와 fake-robot 대상
  mock 테스트부터 만들고, 실물 연결은 맨 마지막이다.
- `scripts/tools/piper_human_approved_inference.py`의 상태 머신을
  참고 대상으로 재사용한다 (import해서 갖다 쓰는 게 아니라 구조를
  참고).

## 5. 환경

```text
Project:        /home/ugrp43/UGRP/lerobot_robot_piper
LeRobot clone:   /home/ugrp43/UGRP/lerobot
Python:          3.10
Conda:           ugrp
LeRobot:         0.4.4 editable install
Piper URDF:      /home/ugrp43/UGRP/agx_arm_urdf/piper/urdf/piper_description.urdf
```

ROS 2 Humble과 `rclpy` 때문에 Python 3.10을 유지한다. `ikpy`(IK)와
`matplotlib`(plot)은 이미 `ugrp` 환경에 설치돼 있다 — 새로 설치할 필요
없다.

## 6. 파일 수정 경계 (계속 유효)

새 코드는 원칙적으로 `synthetic/` 안에 둔다. 다음 파일은 여전히 직접
수정하지 않는다.

- `lerobot_robot_piper/teleop_ui.py`
- `lerobot_robot_piper/piper_follower.py`
- `scripts/tools/piper_human_approved_inference.py`
- 외부 `/home/ugrp43/UGRP/lerobot` clone 전체

수정이 반드시 필요하다고 판단되면 먼저 이유와 정확한 파일을 사용자에게
보고하고 승인을 기다린다.

## 7. 작업 트리 보존 / 백업 / 커밋

- `git status --short`로 synthetic과 무관한 변경을 다시 확인하고,
  수정·staging·삭제·되돌리기를 하지 않는다.
- `git reset --hard`, 사용자 변경을 지우는 `git checkout --`, 범위가
  불명확한 재귀 삭제를 쓰지 않는다.
- 기존 파일(6절 목록 외의 것이라도)을 수정하기 전에는
  `tmp/claude_synthetic_backup_<YYYYMMDD_HHMMSS>/`에 원본을 백업한다.
  새 파일은 백업 없이 바로 추가할 수 있다.
- 커밋과 push는 사용자가 명시적으로 요청할 때만 한다.

## 8. 이번 단계에서도 하지 말아야 할 작업

4C가 사용자 승인으로 명시적으로 시작되기 전까지는 전부 금지.

- 실물 Piper 연결, CAN bring-up/down
- `PiperFollower.send_action()` 호출
- RealSense live stream, RViz GUI 실제 실행
- 자동 parking, torque enable/disable, effort threshold 튜닝
- 실제 SmolVLA checkpoint 실행, dataset 재녹화
- `configs/recording.env`, teleop GUI 변경

실행 명령이 하드웨어에 연결될 가능성이 조금이라도 있으면 먼저 사용자에게
보고한다.

## 9. 코드 품질과 검증 규칙 (계속 유효)

- Python 3.10 문법, 순수 계산 코드는 ROS/CAN/GUI 미의존.
- 좌표계와 단위를 함수/JSON 이름에 명시.
- NaN/Inf와 잘못된 shape를 fail-fast. 결과를 clamp해서 성공처럼
  만들지 않는다.
- 모든 output에 config snapshot과 status 저장. 같은 입력·seed면 같은
  결과.
- 새 기능마다 `synthetic/tests/`에 테스트를 추가하고 기존 테스트를 계속
  통과시킨다 (`python -m unittest discover -s synthetic/tests -v`).
- 재생성 가능한 output(`synthetic/outputs/`)과 smoke artifact를 git에
  추가하지 않는다.

## 10. 단계별 보고 형식

각 단계 완료 후 사용자에게 다음을 보고한다: 추가/수정한 파일, 수정 이유,
실행한 테스트와 통과 개수, 하드웨어 없이 검증한 범위, 아직 unverified인
값, 다음에 현실 측정이 필요한 지점, 예상과 달랐던 기존 아키텍처. 문서만
쓰고 구현이 끝난 것처럼 표시하지 않는다. Mock 통과를 실물 검증으로
표현하지 않는다.
