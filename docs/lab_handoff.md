# 실험실 PC 핸드오프 — GUI 리팩터링 (seongil/gui-refactor)

이 문서는 개인 컴퓨터(하드웨어 없음)에서 진행한 GUI 리팩터링 작업을 실험실 PC(실물
Piper 로봇 + CAN + ROS2 연결)로 넘길 때 새 Claude Code 세션이 맥락을 빠르게 따라잡기
위한 요약입니다. 실험실 PC에서 이 문서를 먼저 읽고 아래 순서대로 진행하세요.

## 0. 왜 이 문서가 있는가

- 원본 작업은 DGIST UGRP Group 43 VLA 로봇 조작 프로젝트: PiPER 로봇팔 + LeRobot 데이터
  수집 파이프라인.
- 2026-07-03에 UGRP repo → `lerobot_robot_piper`(WeGo-Robotics fork, DGIST-JaeminBaek이
  다시 fork)로 마이그레이션. 바뀐 것 두 가지:
  1. 액션 공간: EEF-space, raw SDK 정수 → **joint-space, 정규화 -100~100(gripper 0~100)**
  2. CAN: 단일 CAN(`can0`) → **leader/follower 물리적으로 분리된 두 CAN 인터페이스**
     (`can_leader1`/`can_follower1`)
- 개인 컴퓨터(맥, CAN/로봇 없음)에서 `scripts/legacy_tools/`의 구 UGRP 4개 CLI 도구를
  새 구조에 맞게 리팩터링하고, `teleop_ui.py`를 녹화/추론/재생 통합 UI로 확장함.
  **하드웨어가 필요한 부분은 전부 "코드는 맞게 짰지만 실제로 못 돌려봄" 상태.**

## 1. Git 상태

```
origin   = https://github.com/BLINCE1/lerobot_robot_piper-gui-refactor.git   (개인, 작업용)
upstream = https://github.com/DGIST-JaeminBaek/lerobot_robot_piper.git      (원본, 절대 직접 push 금지)
branch   = seongil/gui-refactor
```

실험실 PC에서:
```bash
git clone https://github.com/BLINCE1/lerobot_robot_piper-gui-refactor.git
cd lerobot_robot_piper-gui-refactor
git checkout seongil/gui-refactor
git remote rename origin origin   # 이미 BLINCE1이 origin이라 그대로 둬도 됨
git remote add upstream https://github.com/DGIST-JaeminBaek/lerobot_robot_piper.git
```

**제약사항 (계속 유지):**
- `upstream`(원본)에는 검증 끝나기 전까지 **절대 push 금지**. 조성일님이 직접 지시할 때만.
- `git push` 실행 전에는 항상 어느 remote/브랜치로 가는지 먼저 보고하고 진행할 것.
- 실물 로봇에 명령 보내는 코드는 사용자 확인 없이 임의로 실행하지 말 것 (Launch 버튼
  누르는 것 자체는 사용자 행동이니 OK, 하지만 Claude가 스스로 CLI를 실행해서 로봇을
  움직이면 안 됨).
- CAN 관련 systemd/네트워크 설정 자체(파일 위치, 서비스 등)는 변경 금지. 코드 안의
  변수명/로직 수정은 OK.
- 기존 `scripts/1~9__*.sh` 번호형 스크립트는 그대로 유지 (삭제/변경 금지).

## 2. 환경 세팅

```bash
conda create -n piper-gui-refactor python=3.10
conda activate piper-gui-refactor
pip install -e .
pip install textual   # scripts/legacy_tools/piper_tui.py(은퇴됨, 참고용)가 필요로 함 — 필수는 아님
```

`configs/recording.env`는 아직 이 브랜치에 없습니다(`.example`만 있음) — 실제 CAN
포트/카메라 인덱스로 채워서 만들어야 함:
```bash
cp configs/recording.env.example configs/recording.env
# LEADER_PORT, FOLLOWER_PORT, TOP_CAM, WRIST_CAM 등을 실제 값으로 수정
```

RViz 관련 스텝(`piper_session.py --step rviz`, `piper_replay_viz.py`,
`piper_infer_preview.py`)은 ROS2 환경이 source되어 있어야 함
(`source /opt/ros/humble/setup.bash`). `agx_arm_urdf`
(https://github.com/agilexrobotics/agx_arm_urdf) 의 `piper/urdf/piper_with_gripper_description.xacro`가
로봇 모델 URDF — `joint1~joint6`(revolute, radian), `gripper`(prismatic, 0~0.1m,
`gripper_joint1`/`gripper_joint2`는 `<mimic joint="gripper">`로 자동 추종)가 실제 joint
이름/타입임을 소스로 확인해뒀음 — 이 이름이 `robot_state_publisher`가 구독하는
URDF와 실제로 일치하는지 여기서 최종 확인 필요.

## 3. 이번 세션에서 만든 것 (파일별 요약)

| 파일 | 상태 |
|---|---|
| `lerobot_robot_piper/teleop_ui.py` | 대폭 확장 — 아래 4절 참고 |
| `scripts/legacy_tools/piper_session.py` | CAN 이중화, joint-space 전환 (can_up/down, joint_check, teleop_check, data_check, calc_range, rviz_preview) |
| `scripts/legacy_tools/piper_replay_viz.py` | 재작성 — EEF 마커 → joint_states publish (실제 로봇 모델이 RViz에서 움직임) |
| `scripts/legacy_tools/piper_tui.py` | 은퇴 처리 (docstring에 명시, 삭제 안 함) |
| `scripts/legacy_tools/piper_validate.py` | 은퇴 처리 (piper_session.py가 상위호환) |
| `scripts/tools/piper_infer_preview.py` | 신규 — 정책 추론 결과를 실제 로봇 전송 전 RViz로 미리보기 (open-loop) |
| `docs/lab_handoff.md` | 이 문서 |

## 4. `teleop_ui.py` (piper-teleop) 상세

기존 CAN 모니터 UI를 녹화/추론/재생 통합 콘솔로 확장함. 5개 Preset:

- **Teleoperate** — `lerobot-teleoperate` (discovery 인자 누락 버그 발견해서 고침)
- **Record** — `scripts/5__record.sh`와 동등한 41개 인자, `configs/recording.env` 값 우선
- **Infer** — `lerobot-record --policy.path=...`로 SmolVLA 등 정책 추론. **주의: dry-run
  없음, Launch 누르면 즉시 실제 로봇에 명령 전송됨**
- **Replay (RViz)** — Dataset Browser에서 고른 dataset/episode를 `piper_replay_viz.py`로 재생
- **Infer Preview (RViz)** — `piper_infer_preview.py`로 정책 예측을 실제 로봇 없이 RViz 미리보기

추가 UI 요소: Dataset Browser(records/ 스캔), Recording History(task/episode/frame/fps
요약 표), E-STOP 버튼(follower+leader CAN 즉시 차단), recording.env 로드 상태줄,
녹화 진행률 표시("Recording episode N/M", lerobot-record stdout 파싱), 카메라
release 로직(녹화 종료 시 OpenCV 카메라 index open/close 사이클로 강제 release).

## 5. 실험실 PC에서 확인해야 할 체크리스트 (하드웨어 필요해서 여태 못 한 것)

우선순위 순:

1. **CAN 연결 기본 동작** — `python scripts/legacy_tools/piper_session.py --step joint_check
   --check_leader` 실행해서 follower/leader 둘 다 `[OK]` 뜨는지. 관절값이 실제 팔 자세와
   맞는지(관절 부호/방향 정상인지) 눈으로 확인.
2. **`teleop_ui.py` 실제 실행** — `python -m lerobot_robot_piper.teleop_ui`로 창 띄우고,
   Teleoperate 프리셋으로 leader→follower 텔레옵 실제 확인. 이후 Record로 짧게 1 episode
   녹화 테스트.
3. **카메라 타임아웃 재현 여부** — 원래 문제였던 "녹화마다 카메라 타임아웃" 현상이
   `CAMERA_RELEASE_WAIT_S`(현재 1.5초, `teleop_ui.py` 상단 상수) 도입 후에도 재현되는지.
   재현되면 이 값을 늘려보고, OpenCV가 아니라 RealSense를 쓴다면
   `_reset_opencv_cameras()`가 RealSense는 건너뛰게 되어 있어서(hardware_reset() 미구현)
   별도 처리 필요할 수 있음.
4. **RViz 조인트 이름 검증** — `piper_replay_viz.py`/`piper_infer_preview.py`/
   `piper_session.py --step rviz_preview`가 publish하는 `/joint_states`의 `joint1~6`,
   `gripper` 이름이 실제 robot_state_publisher가 기대하는 이름과 일치하는지. 안 맞으면
   RViz에서 로봇이 안 움직임(에러 없이 조용히 무시될 수 있음 — 꼭 시각 확인 필요).
5. **gripper 물리 단위 재확인** — `piper_replay_viz.py`/`piper_infer_preview.py`/
   `piper_session.py`의 gripper 변환 로직이 "raw 값을 mm로 해석해 미터로 변환"하는데
   (README 표는 "0-68 deg"라 적혀있지만 URDF가 prismatic이라 mm 해석이 맞다고 판단함,
   docs 참고: 각 파일 상단 주석), 실제 그리퍼 열림 정도와 맞는지 눈으로 대조.
6. **E-STOP 버튼 실동작** — `teleop_ui.py`의 빨간 E-STOP 버튼이 실제로 follower/leader
   CAN을 즉시 내리는지. (macOS엔 `ip` 명령이 없어서 로직만 mock으로 검증했음, 실제
   `sudo ip link set ... down` 명령이 제대로 도는지 여기서 처음 확인하는 것.)
7. **`max_delta_per_step`(현재 20, 정규화 단위) 재보정** — 실제 정상 녹화 데이터의
   프레임간 delta를 보고 이 임계값이 적절한지 조정. (`piper_session.py` CFG)
8. **`CFG["venv_activate"]` 등 UGRP 실험실 PC 하드코딩 경로 정리** — `piper_session.py`의
   `source_prefix()`가 여전히 `/home/ugrp308/Group43/...` 참조함. 실제 이 PC의 venv/ROS2
   경로로 갱신 필요 (지금은 `|| true`로 실패를 삼켜서 동작엔 지장 없지만 노이즈 로그가 남음).
9. **정책 체크포인트 확보 후 Infer/Infer Preview 검증** — 이 프로젝트는 아직 SmolVLA를
   학습시키지 않아서 실제 체크포인트로 테스트해본 적이 없음. 학습 끝나면 여기서 처음 검증.

## 6. 참고 — 이번 세션에서 실측/확인한 것 (재검증 불필요)

- `PiperFollower.get_observation()`/`PiperLeader.get_action()` 반환 dict key는
  `f"{motor}.pos"` 형태 (joint1.pos~joint6.pos, gripper.pos) — 소스 확인 완료.
- Plugin 등록 이름: `piper_follower`/`piper_leader` — `config_piper.py`/
  `config_piper_leader.py`의 `register_subclass()` 확인 완료.
- `--robot.discover_packages_path=lerobot_robot_piper` 없으면 타입 등록 자체가 안 됨 —
  실제 원인은 확인했지만 "정말 저 인자 없이 실행하면 에러 나는지"는 여기서 실행해봐야 함.
- `record_sample/local/piper_write_light_rs_10s_3eps_v5` (레포에 포함된 유일한 실제
  녹화 데이터)로 `piper_replay_viz.py`의 데이터 로딩~변환~안전검사까지는 실제로 돌려서
  검증 완료 (ROS2 없어서 RViz publish 직전까지만).
- `lerobot-record`가 `--policy.path=...`를 지원하고(별도 smolvla-inference CLI 불필요),
  `smolvla`가 이미 lerobot에 등록된 policy 타입인 것 확인 완료.
