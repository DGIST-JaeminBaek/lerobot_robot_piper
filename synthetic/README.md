# 현실 실행 기반 합성 데이터 생성 계획

## 1. 목적

이 작업에서 말하는 합성 데이터는 시뮬레이터가 만든 이미지나 로봇 상태가 아니다.
컴퓨터가 작업 궤적과 action stream을 생성하고, 실제 Piper가 이를 실행하는 동안
카메라·state·action을 다시 녹화해 현실 LeRobotDataset을 만드는 방식이다.

초기 목표 작업은 다음과 같다.

```text
고정된 시작 자세
  → 고정된 위치의 지우개 파지
  → TOP 이미지에서 지정한 도형 위치로 이동
  → 도형을 따라 지우기
  → 지우개를 원위치에 놓기
  → gripper를 열고 종료
```

원래 아이디어와 배경은 [`../synthetic_data.md`](../synthetic_data.md)를 참고한다.

이 문서는 구현 계획만 정의한다. 현재 이 폴더에는 실물 로봇으로 action을 전송하는
실행 코드를 두지 않는다.

## 2. 초기 범위

첫 구현의 범위는 의도적으로 제한한다.

| 항목 | 초기 범위 |
|---|---|
| 로봇 시작 자세 | 고정 |
| 지우개 시작 위치와 방향 | 고정 |
| 카메라와 보드 위치 | 고정 |
| 보드 작업 영역 | 보정된 직사각형 내부 |
| 도형 위치 | 작업 영역 안에서 변경 가능 |
| 도형 크기 | 도형별 고정 |
| 도형 회전 | 우선 고정 |
| 도형 종류 | 원 1종부터 시작 |
| 목표 지정 | TOP 이미지에서 사람이 중심을 클릭 |
| Action FPS | 현재 데이터셋과 동일한 30 FPS |
| 실행 방식 | 전체 offline 검증 후 구간별 사람 승인 |

초기 범위에 포함하지 않는 항목:

- 시뮬레이터 이미지 생성
- 자동 도형 검출 또는 segmentation
- 임의 크기와 임의 회전을 동시에 처리하는 궤적
- 학습된 policy가 생성 궤적을 수정하는 기능
- 별도 force/torque sensor 기반 force control
- 사람 승인 없는 완전 자동 실물 실행
- 합성 데이터 생성과 SmolVLA 학습을 한 프로세스로 연결하는 기능

## 3. 고정할 데이터 계약

### 3.1 좌표계

코드에서 다음 좌표계를 명시적으로 구분한다.

| 이름 | 단위 | 의미 |
|---|---|---|
| `image_px` | pixel | TOP 원본 이미지 좌표 |
| `board_xy` | mm | 보드 평면 위의 2차원 좌표 |
| `base_xyzrpy` | mm, degree | Piper base 기준 EEF pose |
| `joint_physical` | radian 또는 SDK raw unit | Piper 6개 관절의 물리 좌표 |
| `action_normalized` | 현재 LeRobot 범위 | 관절 6개와 gripper를 합친 7차원 action |

좌표 변환 흐름:

```text
image_px
  → 카메라 왜곡 보정
  → homography
  → board_xy
  → board-to-base transform와 접촉 높이 적용
  → base_xyzrpy
  → IK
  → joint_physical
  → 현재 Piper 정규화 규칙
  → action_normalized (N, 7)
```

중간 좌표를 생략하거나 단위를 암묵적으로 변환하지 않는다. 각 파일에는 좌표계,
단위와 보정 버전을 함께 기록한다.

### 3.2 Action 형식

최종 생성 action은 현재 학습 데이터와 같은 형식을 사용한다.

```text
shape: (N, 7)
order: joint1, joint2, joint3, joint4, joint5, joint6, gripper
meaning: follower에 전달할 normalized absolute position target
fps: 30
```

상대 이동량이나 Cartesian delta를 최종 action으로 저장하지 않는다. 상대 이동이나
Cartesian 경로는 생성 과정의 중간 표현으로만 사용하고, 최종 결과는 항상 위의
절대 joint target으로 변환한다.

### 3.3 보정값

다음 값은 코드에 하드코딩하지 않고 versioned config로 관리한다.

- TOP 카메라 intrinsic과 distortion
- 이미지와 보드 사이 homography
- 보드 좌표계와 Piper base 사이 transform
- 보드 평면의 위치와 기울기
- 지우개 tool center point offset
- 지우개 파지 pose
- 접촉 높이와 비접촉 이동 높이
- 작업 가능 보드 범위
- joint/action 한계
- 최대 frame 간 action 변화량

실물에서 측정하기 전에는 해당 값에 `unverified` 상태를 표시한다. 검증되지 않은
보정 config로는 실물 실행 모드를 활성화할 수 없게 한다.

## 4. 예정 디렉터리 구조

향후 관련 코드는 모두 이 폴더 안에서 개발한다.

```text
synthetic/
├── README.md
├── configs/
│   ├── board_calibration.example.json
│   ├── trajectory.example.json
│   └── safety.example.json
├── calibration/
│   ├── select_board_points.py
│   ├── solve_homography.py
│   └── validate_calibration.py
├── trajectory/
│   ├── pickup_template.py
│   ├── transport_path.py
│   ├── erase_templates.py
│   ├── return_template.py
│   └── compose_trajectory.py
├── kinematics/
│   ├── ik_solver.py
│   ├── action_conversion.py
│   └── validate_fk_ik.py
├── preview/
│   ├── render_board_path.py
│   └── publish_rviz.py
├── execution/
│   ├── approved_runner.py
│   └── record_generated_episode.py
├── tests/
└── outputs/
```

`outputs/`에는 재생성 가능한 calibration preview, trajectory NPZ, plot과 실행 로그를
둔다. 원본 녹화 데이터와 최종 LeRobotDataset은 기존 `records/` 규칙을 유지한다.

## 5. 기존 코드와의 경계

중복 구현을 피하기 위해 다음 검증된 기능은 기존 프로젝트에서 재사용한다.

| 기존 기능 | 재사용 대상 |
|---|---|
| Piper action 정규화/역정규화 | `PiperMotorsBus`와 motor table |
| FK | `piper_sdk`의 `C_PiperForwardKinematics.CalFK()` |
| RViz joint 표시 | `scripts/tools/piper_infer_preview.py` |
| 구간별 사람 승인 | `scripts/tools/piper_human_approved_inference.py`의 상태 머신 |
| 실제 action 제한 | `PiperFollower.send_action()` |
| Effort trip과 parking | 기존 `PiperFollower` safety latch |
| LeRobotDataset 녹화 | 현재 record 실행 경로 |

`synthetic`의 순수 계산 모듈은 CAN, ROS와 LeRobotDataset을 직접 import하지 않도록
분리한다. 좌표 변환·trajectory·IK는 NumPy 배열과 JSON/NPZ만 입출력하고, RViz와
실물 실행은 adapter 계층에서만 연결한다.

외부 `/home/ugrp43/UGRP/lerobot` clone은 이 기능 때문에 수정하지 않는다.

## 6. 단계별 구현 계획

### 단계 0. 형식과 안전 경계 고정

구현 내용:

- 좌표계와 단위 dataclass 정의
- calibration/trajectory/action 파일 schema 정의
- config 검증기 구현
- 모든 실행기의 기본값을 `offline`으로 고정
- 검증되지 않은 calibration의 실물 실행 차단

완료 조건:

- 잘못된 단위, shape, NaN/Inf와 누락된 보정값을 즉시 거부한다.
- 같은 config와 seed로 항상 같은 trajectory가 생성된다.
- 이 단계의 코드는 ROS와 CAN 없이 테스트된다.

### 단계 1. TOP 이미지와 보드 좌표 보정

구현 내용:

- TOP 이미지 위에서 보드 기준점을 선택하는 도구
- 기준점의 실제 `board_xy` 입력
- homography 계산과 저장
- 픽셀을 보드 좌표로, 보드 좌표를 픽셀로 변환
- 보정점과 검증점의 reprojection error 시각화
- 보정된 작업 영역 밖의 목표 거부

실물 없이 가능한 검증:

- 인위적으로 만든 사각형과 알려진 homography의 round-trip
- 기존 TOP 녹화 이미지에 보드 경계와 목표 위치 overlay

실물에서 필요한 값:

- 실제 보드 기준점 좌표
- 카메라가 고정된 상태의 실제 TOP frame
- 보드 기준점 측정값

완료 조건:

- 별도로 남겨둔 검증점에서 정한 허용오차를 만족한다.
- 목표 픽셀을 `board_xy`로 변환한 뒤 다시 투영했을 때 같은 위치로 돌아온다.

### 단계 2. 보드와 Piper base 좌표 연결

구현 내용:

- `board_xy`를 `base_xyzrpy`로 바꾸는 rigid transform
- 보드 평면 기울기와 tool offset 적용
- 접촉하지 않는 이동 높이와 접촉 높이 분리
- board boundary와 robot workspace 검사

실물 없이 가능한 검증:

- 가상의 보드 transform으로 네 모서리와 중심 좌표 계산
- 변환의 정방향/역방향 round-trip
- RViz에 보드 경계와 목표 EEF marker 표시

실물에서 필요한 값:

- 보드 기준점에 대응하는 실제 EEF pose
- 지우개 TCP offset
- 보드 평면과 안전한 접근 높이

완료 조건:

- 보드의 네 모서리와 중심이 일관된 base 좌표로 변환된다.
- 작업 영역 밖 또는 설정 높이 밖의 EEF 목표가 거부된다.

### 단계 3. Cartesian trajectory template

전체 작업을 다음 segment로 분리한다.

```text
PARKING_TO_PREGRASP
GRASP
LIFT
TRANSFER_ABOVE_BOARD
DESCEND
ERASE
LIFT_FROM_BOARD
RETURN_ABOVE_ERASER
PLACE
OPEN_GRIPPER
FINISH
```

구현 내용:

- 녹화한 pick-up/return 궤적을 template로 읽는 형식
- segment 사이의 연속 보간
- 도형 중심에 따라 erase template 평행 이동
- 고정 크기 원의 erase path 생성
- 각 segment의 속도와 FPS에 따른 sample 수 계산
- 향후 삼각형과 사각형 template를 추가할 수 있는 공통 인터페이스

완료 조건:

- 시작과 끝 pose가 config와 일치한다.
- segment 경계에서 EEF pose가 불연속적으로 뛰지 않는다.
- 고정 seed에서 같은 결과가 생성된다.
- 지우기 구간을 제외한 이동은 항상 보드의 안전 높이 위에 있다.

### 단계 4. IK와 7차원 action 생성

구현 내용:

- URDF와 현재 Piper joint convention을 사용하는 수치 IK
- 직전 joint solution을 다음 frame의 seed로 사용
- 복수 IK 해 중 직전 자세와 가장 가까운 해 선택
- joint limit, velocity와 frame 간 변화량 검사
- gripper template 결합
- 물리 joint 값을 현재 7차원 normalized action으로 변환
- 생성 action을 다시 FK해 목표 EEF trajectory와 비교

현재 `CalFK()`는 실물 EEF 피드백과 검증됐지만 IK는 검증된 상태가 아니다. 따라서
FK가 맞다는 이유로 IK도 맞다고 간주하지 않는다.

완료 조건:

- 전체 frame에서 IK가 수렴한다.
- FK(IK(target))의 위치·자세 오차가 정한 허용범위 안에 있다.
- joint limit와 frame 간 최대 변화량을 위반하지 않는다.
- IK branch가 중간에 바뀌어 관절이 급격히 뛰는 frame이 없다.

### 단계 5. Offline preview와 정적 검증

구현 내용:

- TOP 이미지 위에 목표점과 erase path overlay
- EEF XYZ와 joint action plot
- RViz에서 전체 궤적과 segment별 궤적 재생
- global action clamp가 필요한 값 탐지
- max-relative-target 적용 전후 비교
- 검증 결과를 JSON report로 저장

이 단계에서는 실제 CAN을 열지 않는다. RViz는 관절과 trajectory를 표시하지만
보드·지우개·케이블과의 실제 충돌이나 접촉 effort를 보증하지 않는다.

완료 조건:

- 한 명령으로 이미지 overlay, plot, RViz용 trajectory와 검증 report가 생성된다.
- clamp를 전제로만 실행 가능한 trajectory는 기본적으로 실패 처리한다.
- 모든 segment를 순서대로 또는 개별적으로 재생할 수 있다.

### 단계 6. 사람 승인형 실물 실행 adapter

이 단계는 0~5단계가 완료되고 실물 검증을 시작할 때만 구현한다.

구현 내용:

- `offline`, `rviz-only`, `real` 실행 모드 분리
- `real`에 별도 확인 문자열 요구
- segment를 더 작은 action 묶음으로 나누어 미리보기
- 각 묶음의 명시적 사람 승인
- 승인 직전 실제 state와 trajectory 시작 state 비교
- `PiperFollower.send_action()`을 통한 기존 안전 제한 재사용
- effort trip 시 남은 action 차단, parking과 프로세스 종료
- Ctrl+C와 정상 종료의 parking/torque 동작 명시
- 실제 전송 action과 피드백 state 기록

초기 실물 시험에서 자동 반복 실행은 제공하지 않는다.

완료 조건:

- fake robot에서 승인하지 않은 action이 한 건도 전송되지 않는다.
- stale state, limit 위반, effort trip과 Ctrl+C 경로가 mock test를 통과한다.
- 실물에서는 1 action부터 시작해 별도 승인에 따라 전송 개수를 늘린다.

### 단계 7. 실제 실행과 LeRobotDataset 녹화

구현 내용:

- 생성 trajectory ID와 calibration version을 episode metadata에 기록
- 실행 시작과 동시에 기존 TOP/WRIST/state/action 녹화
- 계획 action과 실제 전송 action을 분리해 보존
- 실패·사람 중단·safety trip 결과 기록
- 성공한 episode만 자동으로 학습 데이터에 넣지 않고 검수 대상으로 분리

완료 조건:

- 이미지, 실제 state와 실제 전송 action의 frame 정렬을 확인한다.
- 생성 계획과 실제 전송 결과의 차이를 재현할 수 있다.
- 실패 episode가 성공 episode로 잘못 분류되지 않는다.

### 단계 8. 다양성 확장

기본 원 궤적이 안정화된 뒤 다음 순서로 확장한다.

- 원의 위치 변화
- 삼각형과 사각형의 고정 크기 template
- 도형 방향 변화
- 제한된 크기 변화
- 지우개와 시작 자세의 작은 변화
- 성공 가능한 범위 안의 task-space noise
- 이미지 기반 자동 도형 중심/윤곽 검출

임의 joint noise를 직접 더하지 않는다. 추가하는 변화는 먼저 offline 검사와 RViz를
통과하고 실제 성공률을 측정한 범위로 제한한다.

## 7. 파일 산출물

한 trajectory 생성 결과는 최소한 다음 파일을 갖는다.

```text
outputs/<trajectory_id>/
├── request.json
├── calibration_snapshot.json
├── cartesian_trajectory.npz
├── joint_actions.npz
├── validation_report.json
├── board_overlay.png
├── joint_plot.png
└── execution_log.jsonl
```

주요 의미:

| 파일 | 내용 |
|---|---|
| `request.json` | 도형 종류, 중심, 크기, seed와 생성 요청 |
| `calibration_snapshot.json` | 생성 당시 사용한 모든 보정값 |
| `cartesian_trajectory.npz` | segment, EEF pose와 시간 |
| `joint_actions.npz` | 물리 joint와 normalized 7차원 action |
| `validation_report.json` | IK 오차, joint range와 연속성 검사 |
| `execution_log.jsonl` | 승인, 전송, feedback와 safety event |

재현에 필요한 config와 수치 데이터는 저장하지만, 재생성 가능한 대형 preview는
Git에 올리지 않는다.

## 8. 검증 원칙

### 하드웨어 없이 완료할 검증

- 좌표 변환 round-trip
- homography reprojection
- 단위와 shape 검사
- trajectory segment 연속성
- IK 수렴과 FK residual
- joint/action 전역 범위
- frame 간 최대 변화량
- deterministic seed
- TOP overlay
- RViz 재생
- fake robot 승인/거부/safety state machine

### 실물에서만 가능한 검증

- 카메라와 보드의 실제 보정 오차
- EEF 목표와 실제 도달 위치의 차이
- 지우개 파지 성공률
- 보드 위치별 접촉 높이
- 지우기 중 effort 분포
- 케이블·보드·주변 물체 간섭
- 지우기 성공 여부
- 지우개 원위치 복귀 성공률

현재 기본 `SAFETY_EFFORT_LIMIT=8.0`은 이 작업에 적절한 값으로 실물 검증된 것이
아니다. 이 값을 접촉 허용 기준으로 간주하지 않는다.

## 9. 예상 일정

한 사람이 실물 장비를 사용할 수 있다는 기준의 예상이다.

| 작업 | 예상 |
|---|---:|
| 형식, config와 순수 계산 골격 | 1~2일 |
| 이미지–보드 보정 도구 | 2~4일 |
| 보드–base 변환과 EEF trajectory | 2~4일 |
| IK와 action 변환/검증 | 3~7일 |
| Offline plot과 RViz 통합 | 2~4일 |
| 사람 승인형 실행 adapter와 mock test | 2~4일 |
| 실물 보정과 첫 고정 크기 원 검증 | 3~7일 |
| 반복 안정화와 녹화 연결 | 1~2주 |

소프트웨어 골격과 offline 검증은 추가 데이터 수집 전 준비할 수 있다. 실물 보정,
접촉 높이, effort 기준과 성공률 검증은 코드만으로 완료할 수 없다.

## 10. 초기 완료 목표

첫 번째 milestone은 다음 한 줄로 정의한다.

```text
녹화된 TOP 이미지에서 사람이 원의 중심을 선택하면,
고정 위치의 지우개를 집고 해당 원을 지운 뒤 되돌리는 7차원 action을 생성하며,
실물 연결 없이 overlay·수치 검사·RViz 재생을 모두 통과한다.
```

이 milestone이 완료되기 전에는 자동 반복 녹화나 도형 종류 확장으로 넘어가지 않는다.

## 11. 현재 구현 상태

2026-07-29 기준 단계 0의 homography 관련 데이터 계약과 단계 1의 offline 도구를
구현했다.

| 파일 | 상태 | 역할 |
|---|---|---|
| `calibration/common.py` | 구현 | schema 검증, 좌표 parsing, homography와 projection |
| `calibration/select_board_points_web.py` | 구현 | SSH port forwarding용 브라우저 네 점 선택 |
| `calibration/select_board_points.py` | 구현 | 로컬 OpenCV 또는 좌표 문자열로 네 점 선택 |
| `calibration/aggregate_board_points.py` | 구현 | 여러 frame 선택의 대표점과 편차 계산 |
| `calibration/solve_homography.py` | 구현 | 픽셀↔보드 homography 계산 및 JSON 저장 |
| `calibration/validate_calibration.py` | 구현 | grid overlay와 픽셀 좌표 조회 |
| `configs/board_calibration.example.json` | 구현 | point-selection JSON 예시 |
| `tests/` | 구현 | homography, 웹 저장과 다중 관측 집계 검사 |

이 도구들은 ROS, CAN, Piper와 LeRobot을 import하거나 연결하지 않는다. 생성되는
calibration 상태는 항상 `unverified`이며 실물 실행 허가로 사용되지 않는다. 웹
서버는 원격 PC의 `127.0.0.1`에만 열리고 random token이 포함된 URL만 허용한다.

### SSH 브라우저에서 네 점 선택

아래 예시는 원 데이터 첫 episode의 유효 시작 frame 42를 사용한다.

```bash
cd /home/ugrp43/UGRP/lerobot_robot_piper
conda activate ugrp

python synthetic/calibration/select_board_points_web.py \
  --video records/0727/erase_the_circle/erase_the_circle_0726-162803/videos/observation.images.top/chunk-000/file-000.mp4 \
  --frame 42 \
  --output synthetic/outputs/board_points.json
```

터미널에 `[URL] http://127.0.0.1:8765/?token=...`가 출력된다. VS Code의
`PORTS` 탭에서 `8765`를 forward하고 Windows 브라우저에서 이 URL을 연다. URL의
token을 제거하면 접근할 수 없다. 다른 port를 사용하려면 `--port`를 지정한다.

클릭 순서는 고정이다.

```text
top_left → top_right → bottom_right → bottom_left
```

조작:

- 마우스 왼쪽 클릭: 현재 선택점의 대략적인 위치 지정
- 숫자 `1~4`: 수정할 `top_left`~`bottom_left` 선택
- `Previous point` / `Next point`: 수정할 점 변경
- 각 점의 X/Y 입력칸: 원본 pixel 좌표 직접 입력
- `Enter`: 다음 점으로 이동
- `Reset`: 전체 초기화
- `Clear active`: 현재 점만 삭제
- 방향키: 현재 선택점을 원본 기준 1 pixel 이동
- `Shift+방향키`: 현재 선택점을 원본 기준 10 pixel 이동
- `Fit` / `100%` / `200%`: 이미지 표시 배율 변경
- `Save board_points.json`: 네 점 저장 후 서버 자동 종료

오른쪽 확대경은 현재 선택점 주변 원본 40×40 pixel을 nearest-neighbor로 확대한다.
선택점이 아직 없을 때만 mouse cursor를 따라간다. 화면 전체가 브라우저 크기에 맞게
축소되더라도 클릭 좌표는 원본 영상의 1280×720 좌표로 환산되고 소수점 좌표까지
보존된다.

OpenCV 창을 직접 사용할 수 있는 환경에서는
`select_board_points_web.py` 대신 `select_board_points.py`를 사용할 수 있다.

실제 보드 가로·세로를 mm로 측정한 뒤에는 다음 옵션을 함께 사용한다.

```bash
--unit mm --board-width <가로_mm> --board-height <세로_mm>
```

크기를 아직 측정하지 않았으면 기본 normalized `1×1` 좌표로 소프트웨어 동작만
검증할 수 있다.

### 여러 episode의 선택 결과 집계

카메라와 보드가 움직이지 않은 같은 수집 세션에서 5~10개 episode를 선택한다. 각
episode에 대해 위 명령을 실행하되 output 이름을 다르게 저장한다.

```text
synthetic/outputs/board_points_ep000.json
synthetic/outputs/board_points_ep001.json
synthetic/outputs/board_points_ep002.json
...
```

개별 선택을 합친다.

```bash
python synthetic/calibration/aggregate_board_points.py \
  --inputs \
    synthetic/outputs/board_points_ep000.json \
    synthetic/outputs/board_points_ep001.json \
    synthetic/outputs/board_points_ep002.json \
  --output synthetic/outputs/board_points_aggregated.json
```

기본 대표값은 점별 pixel 좌표의 중앙값이다. 한 번의 부정확한 클릭에 덜 민감하도록
homography 행렬을 평균하지 않고, 원본 point 좌표를 먼저 집계한 뒤 homography
하나를 계산한다. 출력 JSON에는 다음 값이 함께 남는다.

- 모든 입력 파일과 source frame
- 각 frame에서 선택한 네 점
- 점별 평균, 중앙값과 x/y 표준편차
- 대표점으로부터의 평균·최대 pixel 편차

카메라나 보드가 움직인 데이터는 같은 집계에 넣지 않는다. 허용 편차를 정한 뒤
초과 시 실패시키려면 `--fail-above-px <값>`을 사용한다.

### Homography 계산

```bash
python synthetic/calibration/solve_homography.py \
  --points synthetic/outputs/board_points_aggregated.json \
  --output synthetic/outputs/board_calibration.json
```

### Grid overlay 및 좌표 확인

중앙 pixel 하나를 명령행에서 확인하는 예:

```bash
python synthetic/calibration/validate_calibration.py \
  --calibration synthetic/outputs/board_calibration.json \
  --output synthetic/outputs/board_overlay.png \
  --pixel 640,360
```

이미지에서 여러 위치를 직접 클릭해 좌표를 확인하려면 `--interactive`를 추가한다.
각 클릭의 `image_px → board_xy` 결과가 터미널에 출력된다.

### 테스트

```bash
python -m unittest discover -s synthetic/tests -v
```

현재 검증 결과:

```text
15 tests passed
실제 TOP HEVC MP4 frame 42 decode 통과
localhost 웹 frame 응답·token 거부·JSON 저장·자동 종료 통과
3개 episode point JSON → 중앙값 집계 → calibration JSON 통과
point selection JSON → calibration JSON → overlay PNG 통과
ROS/CAN 연결 및 로봇 action 실행 없음
```

`synthetic/outputs/`의 결과는 재생성 가능한 산출물이므로 Git에 포함하지 않는다.

## 12. 모델별 이미지 전처리 (`preprocessing/`)

2026-07-29 기준 물리 calibration을 원본 TOP `1280x720`에 고정하고, VLA별
crop/resize/letterbox를 별도 profile로 분리하는 작업을 구현했다.

| 파일 | 상태 | 역할 |
|---|---|---|
| `preprocessing/profiles.py` | 구현 | `SourceShape`/`CropRegion`/`ResizeSpec`/`ImageProfile` dataclass, JSON 직렬화, `smolvla_v1_profile()` 예시 |
| `preprocessing/image_transform.py` | 구현 | crop/scale/pad geometry 계산, raw↔model point 변환, 실제 이미지에 동일 transform 적용, bounds 검사 |
| `preprocessing/preview_transform.py` | 구현 | 실제 frame에 profile을 적용해 raw/model overlay와 JSON report 생성 |
| `configs/image_profiles.example.json` | 구현 | 현재 SmolVLA checkpoint profile 예시 |
| `tests/test_image_transform.py` | 구현 | round-trip, crop 경계, scaling, letterbox padding, 다양한 aspect ratio, image/point 일치, 잘못된 입력 거부, JSON 직렬화 |

### Resize mode

- `stretch`: crop을 목표 width/height로 종횡비 무시하고 resize.
- `fit`: 종횡비를 유지한 채 목표 box 안에 들어가도록 resize. Padding이 없으므로
  실제 출력 크기가 한 축에서 목표보다 작을 수 있다.
- `letterbox`: `fit`과 같은 scale을 적용한 뒤 `pad_value`로 목표 크기까지
  padding한다. Padding 폭은 짝수가 아니면 위/왼쪽을 `floor`로 잘라 계산한다.

Crop의 right/bottom 경계는 `synthetic/calibration/common.py`와 동일하게
exclusive(`x < crop.x + crop.width`)다.

### 사용법

```bash
conda activate ugrp
cd /home/ugrp43/UGRP/lerobot_robot_piper

python synthetic/preprocessing/preview_transform.py \
  --video records/0727/erase_the_circle/erase_the_circle_0726-162803/videos/observation.images.top/chunk-000/file-000.mp4 \
  --frame 42 \
  --profile-name smolvla_v1 \
  --raw-point 640,360 \
  --output-dir synthetic/outputs/preprocessing_smoke \
  --overwrite
```

`--profile <경로>`로 `configs/image_profiles.example.json` 같은 커스텀 profile
JSON을 불러올 수도 있다. `--raw-point`는 여러 번 지정할 수 있고, crop 밖으로
잘린 점은 빨간색, model 이미지 안에 남는 점은 초록색으로 표시된다.

이 tool은 ROS, CAN, Piper와 LeRobot을 import하거나 연결하지 않는다.

### 테스트

```bash
python -m unittest discover -s synthetic/tests -v
```

현재 검증 결과:

```text
42 tests passed (기존 15개 + 신규 27개)
실제 TOP HEVC MP4 frame 42에 SmolVLA profile 적용 -> raw/model overlay 생성 확인
crop 안 점 board 중심(640,360) -> model(256,256) 일치 확인
crop 밖 점(100,100) -> visible=False, non-strict 좌표 반환 확인
ROS/CAN 연결 및 로봇 action 실행 없음
```

## 13. 보드 좌표 ↔ Piper base 좌표 변환 (`transforms/`)

2026-07-29 기준 `board_xy_mm ↔ base_xyz_mm` rigid transform solver를 구현했다.
보드 평면은 `[x, y, 0]`으로 3차원에 embedding하고, Kabsch/SVD로 회전과 이동을
계산한다. Scale은 fit하지 않으며, 입력 좌표는 반드시 `mm` 단위여야 한다.

| 파일 | 상태 | 역할 |
|---|---|---|
| `transforms/board_base.py` | 구현 | `RigidTransform`, Kabsch solver, plane normal, residual 통계, correspondence 파싱/검증 |
| `transforms/solve_board_base.py` | 구현 | correspondence JSON -> `board_base_transform` JSON (forward/inverse 4x4, residual, `status=unverified`) |
| `transforms/validate_board_base.py` | 구현 | 저장된 transform의 residual 재계산, `board_xy↔base_xyz` 조회 |
| `configs/board_base.example.json` | 구현 | 예시 correspondence 5점 (수치는 illustrative일 뿐 실측 아님) |
| `tests/test_board_base.py` | 구현 | 알려진 rotation/translation 복원, forward/inverse round-trip, noise residual, 공선점/중복점/최소 3점 미만/단위 불일치 거부, orthonormality/determinant 검증 |

### 평면 소스 데이터의 반사(reflection) 처리

`board_xy`는 항상 `z=0`으로 embedding되므로 Kabsch의 cross-covariance 행렬은
항상 rank ≤2이고, 회전의 법선(out-of-plane) 축 부호는 데이터로 결정되지
않는다. 이 자유도 때문에 raw SVD 결과가 우연히 improper(determinant −1)로
나올 수 있는데, 이는 실제 반사 오류가 아니라 임의의 부호 선택일 뿐이다.
`solve_rigid_transform`은 표준 Kabsch 보정을 적용해 항상 proper rotation
(determinant +1)을 반환한다. 다만 JSON에서 불러오거나 수동으로 구성한
`RigidTransform`이 실제로 improper matrix라면 `RigidTransform.validate()`가
거부한다.

### 사용법

```bash
conda activate ugrp
cd /home/ugrp43/UGRP/lerobot_robot_piper

python synthetic/transforms/solve_board_base.py \
  --correspondences synthetic/configs/board_base.example.json \
  --output synthetic/outputs/board_base_transform.json \
  --overwrite

python synthetic/transforms/validate_board_base.py \
  --transform synthetic/outputs/board_base_transform.json \
  --board-xy 200,150 \
  --base-xyz 204.381,203.182,43.496
```

두 tool 모두 ROS, CAN, Piper와 LeRobot을 import하거나 연결하지 않는다.

### 테스트

```bash
python -m unittest discover -s synthetic/tests -v
```

현재 검증 결과:

```text
65 tests passed (기존 42개 + 신규 23개)
가상 rotation/translation -> 5개 correspondence -> solver -> 원래 값 복원 확인 (오차 <1e-6)
예시 config로 CLI end-to-end 실행 확인 (residual mean=0.000328mm)
board_xy(200,150) -> base_xyz -> board_xy round trip 확인 (board_z≈0.0002mm)
공선점/중복점/2점 이하/단위 불일치("normalized") 거부 확인
ROS/CAN 연결 및 로봇 action 실행 없음
```

## 14. Cartesian trajectory 데이터 계약과 도형 경로 (`trajectory/`)

2026-07-29 기준 11개 segment 순서, `board_xy`/`base_xyzrpy`/gripper를 담는
segment 스키마, 원/삼각형/사각형 고정 크기 도형 경로, FPS/속도 기반 sample 수
계산, 그리고 도형 경로를 `DESCEND`/`ERASE`/`LIFT_FROM_BOARD` Cartesian
segment로 합성하는 코드를 구현했다.

| 파일 | 상태 | 역할 |
|---|---|---|
| `trajectory/schema.py` | 구현 | `SEGMENT_ORDER`, `SegmentTrajectory`, `FullTrajectory` (경계 연속성/순서/gripper 점프 검사), JSON 직렬화 |
| `trajectory/timing.py` | 구현 | FPS/속도 -> sample 수, arc-length 기반 polyline 재샘플, 두 pose 사이 선형 보간 |
| `trajectory/shapes.py` | 구현 | `fixed_circle_path`/`fixed_triangle_path`/`fixed_rectangle_path` (중심 평행이동, `rotation_deg` 실제 동작) |
| `trajectory/compose.py` | 구현 | `BoardMotionConfig`(hover/contact height, 고정 tool 자세, 속도), board_xy+height→base_xyzrpy 변환, `DESCEND/ERASE/LIFT_FROM_BOARD` 합성, hover 높이 위반 검사, pick-up/return용 `require_recorded_template` |
| `tests/test_trajectory_shapes.py` | 구현 | 도형 반지름/폐곡선/평행이동/회전/최소 점 개수/결정론성 |
| `tests/test_trajectory_timing.py` | 구현 | sample 수, arc-length 재샘플, pose 선형보간 |
| `tests/test_trajectory_compose.py` | 구현 | segment 연속성, hover 높이 위반, 11-segment 전체 합성, 누락 segment 보고, JSON round-trip |

### Pick-up/return을 합성하지 않는 이유

`PARKING_TO_PREGRASP`/`GRASP`/`LIFT`/`RETURN_ABOVE_ERASER`/`PLACE`는 실제
녹화된 사람 시연 template가 아직 없으므로 이 단계에서 좌표를 만들어내지
않는다. `compose.require_recorded_template()`은 이 5개 segment에 대해
외부에서 제공한 template의 스키마(segment 일치, 유효성)만 검증하고, template이
없으면 명확한 오류로 거부한다. 반대로 `TRANSFER_ABOVE_BOARD`/`DESCEND`/
`ERASE`/`LIFT_FROM_BOARD`/`OPEN_GRIPPER`/`FINISH`는 도형 경로와 (unverified)
board-base transform, hover/contact 높이, 고정 tool 자세만으로 이 단계에서
완전히 계산할 수 있다.

### 사용법 (Python)

```python
from synthetic.transforms.board_base import RigidTransform
from synthetic.trajectory.compose import BoardMotionConfig, build_descend_erase_lift
from synthetic.trajectory.shapes import fixed_circle_path

transform = RigidTransform.from_dict(board_base_transform_json["transform_base_from_board"])
config = BoardMotionConfig(
    hover_height_mm=30.0,
    contact_height_mm=0.0,
    tool_rpy_deg=(180.0, 0.0, 0.0),
    fps=30.0,
    transfer_speed_mm_per_s=200.0,
    descend_lift_speed_mm_per_s=50.0,
    erase_speed_mm_per_s=80.0,
)
path = fixed_circle_path([200.0, 150.0], radius_mm=40.0, num_points=64)
descend, erase, lift = build_descend_erase_lift(path, board_to_base=transform, config=config)
```

`hover_height_mm`/`contact_height_mm`/`tool_rpy_deg`는 모두 `status="unverified"`
config이며, 실제 측정 전에는 이 수치를 실물 실행 근거로 사용하지 않는다.

### 테스트

```text
124 tests passed (기존 65개 + 신규 59개)
가상 rotation/translation 및 6B에서 solve한 실제(비-identity) transform 모두로
DESCEND/ERASE/LIFT_FROM_BOARD 생성 및 segment 경계 연속성 확인
ERASE는 hover 높이 아래로 내려가고 TRANSFER_ABOVE_BOARD는 내려가지 않음을 확인
11개 segment 전체 합성 시 순서/연속성/gripper 점프 검사, 누락 segment 보고 확인
동일 입력 재호출 시 bit-identical 결과 확인 (deterministic)
ROS/CAN 연결 및 로봇 action 실행 없음
```

## 15. IK와 7차원 action offline 변환 (`kinematics/`)

2026-07-29 기준 Cartesian pose 시퀀스를 physical joint로 풀고, 기존
정규화 상수로 7차원 `(N, 7)` normalized action으로 바꾸는 코드를 구현했다.

| 파일 | 상태 | 역할 |
|---|---|---|
| `kinematics/piper_fk.py` | 구현 | `piper_sdk.kinematics.piper_fk.C_PiperForwardKinematics.CalFK()`의 순수 wrapper (`dh_is_offset=1`), 단일/batch EEF pose 계산 |
| `kinematics/piper_ik.py` | 구현 | `ikpy` + 기존 검증된 URDF로 수치 IK, 이전 프레임 seed 사용, 여러 seed 후보 중 직전 자세와 가장 가까운 해 선택, `CalFK` 기준 residual 검증, joint limit/연속성 검사 |
| `kinematics/action_conversion.py` | 구현 | `piper_follower.py`에 하드코딩된 calibration을 그대로 복사해 physical joint(rad)/gripper(mm) ↔ normalized 7차원 action 변환 |
| `tests/test_piper_fk.py` | 구현 | zero-config 값이 `piper_sdk` 문서값과 일치, 단위/shape 검증 |
| `tests/test_piper_ik.py` | 구현 | FK→IK→FK round-trip, seed 연속성, unreachable/joint-limit/NaN 실패, pose↔matrix round-trip |
| `tests/test_action_conversion.py` | 구현 | 정규화 상수 round-trip, 범위 밖 값 거부, `(N,7)` 컬럼 순서 |
| `tests/test_kinematics_trajectory_integration.py` | 구현 | 6C의 `DESCEND/ERASE/LIFT_FROM_BOARD`를 실제로 IK에 넣어 `(N,7)` action까지 생성 |

### IK 라이브러리 선택과 실수 하나

`piper_sdk`에는 IK가 없다(`CalFK`/`GetFK`만 있음). `ugrp` 환경에 이미 설치된
`ikpy 4.0.0`과, RViz가 쓰는 것과 같은 URDF
(`/home/ugrp43/UGRP/agx_arm_urdf/piper/urdf/piper_description.urdf`, 이미
`CalFK`와 0.1mm 이내로 검증된 상태)를 그대로 사용했다. 새 패키지 설치나 URDF
convention 변경은 하지 않았다.

구현 중 실수를 하나 발견해서 고쳤다: `ikpy`의 `inverse_kinematics_frame`은
**기본값이 position만 최적화**하고 orientation은 최적화하지 않는다
(`orientation_mode` 인자를 명시적으로 `"all"`로 줘야 함). 처음에는 이걸 몰라서
position은 거의 정확히 맞는데 orientation이 수십 도씩 어긋나는 결과가
나왔다 — 통합 테스트를 만들어 실제로 돌려보다가 발견했다. `orientation_mode="all"`을
추가한 뒤로는 zero seed에서도 position/rotation 오차가 거의 0으로 수렴한다.

### 계산 규칙

- `CalFK` 입력은 라디안, 출력은 `[x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg]`
  (EEF = `CalFK(...)[-1]`). 전부 `docs/kinematics/kinematics_check.md`와
  `piper_first_chunk_fk_analysis.py`에서 그대로 가져왔다.
- IK 검증은 `ikpy` 자체 FK가 아니라 프로젝트가 신뢰하는 `piper_sdk.CalFK`
  기준으로 residual을 계산한다.
- 정규화 상수(`joint1: raw -150000~150000` 등)는 `piper_follower.py`의
  하드코딩된 `MotorCalibration`을 그대로 복사했다 — 이 프로젝트는 calibration을
  JSON에서 읽지 않고 이 상수 자체가 calibration이다. `PiperMotorsBus.apply_drive_mode`가
  `False`라 어떤 motor도 부호 반전이 없다는 것도 확인했다.
- 실행 중 `_normalize`/`_unnormalize`(`piper_motors_bus.py`)는 안전을 위해
  범위를 벗어난 값을 clamp하지만, 이 offline 변환은 **clamp하지 않고 범위를
  벗어나면 즉시 실패**시킨다(생성된 궤적이 잘못됐다는 신호를 숨기지 않기 위해).
- gripper는 arm IK와 완전히 분리되어 있다 — `synthetic/trajectory/schema.py`의
  `gripper` closed-fraction(0=open, 1=closed)을 `gripper_fraction_to_mm()`으로
  물리 mm(0=닫힘, 68mm=완전 열림, `tables.py`의 "gripper는 닫힘 0mm" 주석 기준)로
  바꾼 뒤 정규화한다.

### 사용법 (Python)

```python
from synthetic.kinematics.piper_ik import solve_ik_sequence
from synthetic.kinematics.action_conversion import build_normalized_action

joint_rad_sequence, ik_solutions = solve_ik_sequence(
    base_xyzrpy_sequence,          # (N, 6) mm, deg -- 예: FullTrajectory에서 이어붙인 pose
    initial_seed_rad=previous_known_joint_rad,
    position_tol_mm=1.0,
    angle_tol_deg=1.0,
    max_joint_step_rad=0.3,
)
action = build_normalized_action(joint_rad_sequence, gripper_closed_fraction_sequence)  # (N, 7)
```

### 테스트

```text
160 tests passed (기존 124개 + 신규 36개)
FK zero-config 값이 piper_sdk 문서의 init_pos와 일치 확인 (dh_is_offset=1)
알려진 joint -> CalFK pose -> solve_ik -> CalFK round-trip, 오차 <1mm/<1deg 확인
동일 seed로 재요청 시 원래 joint를 그대로 복원(가장 가까운 해 선택 확인)
6C의 board shape 경로(80 frame) 전체를 IK에 넣어 (80, 7) action 생성 및 range 확인
unreachable pose, joint limit 위반, NaN/Inf seed/target 모두 명확히 실패 확인
ROS/CAN 연결 및 로봇 action 실행 없음 (piper_sdk.kinematics.piper_fk만 import, C_PiperInterface_V2 미사용)
```

## 16. Offline preview와 validation report (`preview/`)

2026-07-29 기준 6A~6D를 한데 묶어 실물 연결 없이 한 번에 검토할 수 있는 CLI와
검증 report를 구현했다. `CLAUDE_HANDOFF.md` 6E의 필수 검증 7개를 모두
구현했다.

| 파일 | 상태 | 역할 |
|---|---|---|
| `preview/validation.py` | 구현 | 7개 필수 검증(raw/model/board 일관성, segment 연속성, IK residual, joint limit, frame 간 joint 변화, global action range, config status → 실행 가능 여부) — 각 검증은 예외를 던지지 않고 pass/fail 구조체를 반환 |
| `preview/plots.py` | 구현 | `matplotlib`(Agg, headless) 기반 EEF plot, joint/action plot |
| `preview/overlays.py` | 구현 | board 경로를 raw TOP 이미지와 model-input 이미지에 각각 투영해 그리기 |
| `preview/rviz_adapter.py` | 구현 | `piper_infer_preview.py`와 동일한 `JointState` wire format(이름/물리 단위)을 재현하는 **mock** — `rclpy`/`sensor_msgs` import 없음, RViz 실제 실행 없음 |
| `preview/generate_preview.py` | 구현 | 위 전부를 묶어 `outputs/<trajectory_id>/`에 9개 파일(`request.json`, `preprocessing_profile.json`, `calibration_snapshot.json`, `cartesian_trajectory.npz`, `joint_actions.npz`, `validation_report.json`, `raw_board_overlay.png`, `model_input_overlay.png`, `eef_plot.png`, `joint_plot.png`) 생성 |
| `tests/test_validation_report.py` | 구현 | 7개 검증 각각의 pass/fail 케이스, `build_validation_report`가 항상 `unverified`/`real_execution_allowed=False`인지 확인 |
| `tests/test_preview_plots.py` | 구현 | plot 생성/오류 입력 거부 |
| `tests/test_preview_overlays.py` | 구현 | overlay shape, 원본 프레임 비수정, crop 밖 경로 처리 |
| `tests/test_rviz_adapter.py` | 구현 | `JointState` 이름/물리 단위(rad, gripper meter) 정확성, 잘못된 입력 거부 |

이번 preview는 `TRANSFER_ABOVE_BOARD`/`DESCEND`/`ERASE`/`LIFT_FROM_BOARD` 세
segment만 생성한다 — pick-up/return은 6C와 마찬가지로 아직 실제 template가
없어서 생성하지 않는다.

### 실제 데이터로 스모크 검증

가상의(비-실측) 그러나 자기 일관적인 입력으로 CLI를 처음부터 끝까지 실행했다.

- image↔board calibration: 기존 `board_points_ep000.json`(실제 TOP frame 42의
  4점)에 illustrative 500×380mm 보드 크기를 붙여 mm homography로 재계산
- board↔base transform: `piper_fk`로 실제 도달 가능한 pose 하나를 구한 뒤 그
  지점에 board 중심을 identity rotation으로 고정한 hand-built fixture
- board motion config: hover 15mm, contact 0mm, 그 reachable pose의 orientation을
  고정 tool 자세로 사용

```bash
python synthetic/preview/generate_preview.py \
  --calibration <mm calibration JSON> \
  --board-base-transform <board_base transform JSON> \
  --board-motion-config <BoardMotionConfig JSON> \
  --video <TOP mp4> --frame 42 \
  --preprocessing-profile-name smolvla_v1 \
  --shape circle --center 250,190 --radius-mm 20 --num-points 96 \
  --initial-seed-joint-rad "0.1,0.6,-0.4,0.0,0.2,0.0" \
  --position-tol-mm 1.0 --angle-tol-deg 1.0 --max-joint-step-rad 0.5 \
  --output-dir synthetic/outputs/<trajectory_id> --overwrite
```

결과: `[OK] preview written`, `all_checks_passed=True`,
`real_execution_allowed=False`(항상). `raw_board_overlay.png`/
`model_input_overlay.png`을 직접 열어 실제 화이트보드·지우개 영상 위에
원 경로가 올바른 위치에 그려지는 것을, `eef_plot.png`에서 hover→contact→hover
z 변화와 원형 x/y 궤적을, `joint_plot.png`에서 매끄러운(불연속 없는) joint/액션
곡선을 확인했다.

반지름을 300mm로 키워 도달 불가능한 pose를 만들면 IK가 명확한 오류로
실패하고(clamp 없음), 어떤 출력 파일도 생성되지 않는 것도 확인했다(fail-fast).

### 테스트

```text
194 tests passed (기존 160개 + 신규 34개)
7개 필수 검증 각각의 pass/fail 케이스 확인
validation_report가 모든 검증을 통과해도 status=unverified,
  real_execution_allowed=False로 고정되는 것 확인
CLI 스모크: 실제 TOP frame 42 + reachable transform으로 9개 출력 파일 전부 생성,
  overlay/plot 육안 확인
도달 불가능한 목표에서 CLI가 파일을 하나도 남기지 않고 명확히 실패하는 것 확인
ROS/CAN 연결 및 로봇 action 실행 없음, RViz GUI 미실행 (JointState는 mock만)
```
