# CLAUDE.md

DGIST UGRP Group 43 — VLA 기반 PiPER 로봇팔 조작 프로젝트 (SmolVLA + LeRobot).
이 리포는 `DGIST-JaeminBaek/lerobot_robot_piper` (WeGo-Robotics fork 계열)의 개인 작업 fork로,
GUI 리팩터링 작업이 진행 중이다.

## 프로젝트 개요

- **목표**: PiPER 로봇팔로 손글씨(5글자 단어) 시연 데이터를 수집하고 SmolVLA를 fine-tuning하여 자율 조작 수행
- **파이프라인**: teleoperation → LeRobotDataset 녹화 → SmolVLA 학습 → 추론(서버/클라이언트)
- **현재 작업**: 기존 keyboard 기반 record 흐름을 Tkinter GUI(`teleop_ui.py` 확장)로 대체하는 리팩터링

## Git / 브랜치 규칙 (중요)

- `origin` = `BLINCE1/lerobot_robot_piper-gui-refactor` (개인 fork, private) — push는 여기만
- `upstream` = `DGIST-JaeminBaek/lerobot_robot_piper` (원본, 공용) — **절대 push 금지.**
  개인 PC 검증 → 랩 PC 검증이 모두 끝난 뒤, 사용자가 명시적으로 지시할 때만 반영
- 작업 브랜치: `seongil/gui-refactor` (브랜치 명명 규칙: `이름/작업내용`)
- 공용 PC를 여러 명이 쓰므로 다른 사람 브랜치(`*/dev` 등)는 건드리지 않는다

## 모터값 컨벤션 (매우 중요)

2026-07-03에 구 리포(`DGIST-JaeminBaek/UGRP`)에서 마이그레이션하면서 컨벤션이 바뀌었다:

| | 구 (UGRP) | 현재 (이 리포) |
|---|---|---|
| 좌표계 | EEF Cartesian | Joint-space |
| 값 범위 | raw SDK 정수 | 정규화 −100 ~ +100 (그리퍼 0~100) |
| 클래스 | Piper / PiperSlaveOnly | **PiperFollower / PiperLeader** |

구 리포(UGRP)의 코드나 스크립트를 참고할 때는 이 차이를 반드시 반영해서 포팅할 것.
raw SDK 정수 값을 그대로 send_action에 넣는 코드는 버그다.

## 리포 구조 (scripts/)

- `1__init_can.sh` — CAN 인터페이스 활성화 (`can_leader1`, `can_follower1` 등)
- `2__find_camera.sh`, `3__set_camera.sh` — RealSense 카메라 탐색 및 `configs/recording.env` 반영
- `4__teleoperate.sh` — 텔레옵 드라이런/점검
- `5__record.sh` — 실제 녹화 (PiperFollower + PiperLeader, LeRobotDataset 저장)
- `6__replay.sh`, `7__train.sh`, `8__run_server.sh` / `9__run_client.sh` — 재생 / 학습 / 추론
- `scripts/gui_tools/` — UGRP 시절 GUI/QC 도구 보관 (`piper_record_gui.py`, `piper_correct_episodes.py`, `wego_dataset_check.py` 등)
- `docs/lab_handoff.md` — 랩 PC 이관 핸드오프 문서. 환경 세팅 순서의 기준 문서
- `configs/recording.env` — 카메라 시리얼, CAN 인터페이스명, 데이터셋 경로 등 환경별 값
  (**커밋 금지 대상인지 .gitignore 확인**; 시리얼/경로가 이미 히스토리에 있다면 새로 노출시키지 말 것)

<!-- TODO: 실제 리포 열어서 폴더 구조와 파일명이 위와 일치하는지 검증 후 수정 -->

## 리팩터링 방향

- `piper_record_gui.py`가 `5__record.sh`를 subprocess로 부르는 구조 대신,
  **PiperFollower / PiperLeader / LeRobotDataset API를 직접 호출하는 얇은 GUI 래퍼**로 재작성
- `teleop_ui.py`에 **Record preset** (task 이름 / episode 수 입력)을 추가하는 방식으로 확장
- 리팩터링 대상 레거시 4종: `piper_session.py`, `piper_tui.py`, `piper_validate.py`, `piper_replay_viz.py`
  — 하드웨어 접근 단계를 dual CAN + PiperFollower/PiperLeader 기준으로 재작성
- lerobot 버전에 주의: 코드 검증은 설치된 lerobot 소스를 직접 읽고 확인할 것 (버전 간 API 변경 잦음)

## 환경 2종

### 개인 PC (하드웨어 없음)
- conda env: `piper-gui-refactor` (python 3.10)
- CAN/카메라 하드웨어 없음 → **실제 소켓 연결 테스트 금지**, mock 기반 로직 테스트만
- GUI 로직 검증: 정상 녹화 / 조기 종료 / 재녹화 / 전체 중단 4개 시나리오

### 랩 PC (추론·제어 PC, ugrp43 / ugrp308)
- conda env: `piper` (신규 검증용은 `piper_test`)
- 카메라 시리얼: top `327122074262`, wrist `243322071626` — 랩 PC 연결 실물과 일치하는지 매번 확인
- CAN: `ip link show | grep can`으로 인터페이스 존재 확인 후 `1__init_can.sh` 실행
- 데이터셋 root: `/home/ugrp308/Group43/datasets/`

## 안전 규칙

1. **실물 로봇을 움직이는 명령은 사용자 확인 없이 실행하지 않는다.** 특히 첫 joint_check, 텔레옵, replay
2. CAN 버스는 공유 자원 — 다른 사람이 사용 중일 수 있으니 하드웨어 단계 전 상태 확인
3. `upstream`에는 어떤 경우에도 push하지 않는다
4. `configs/recording.env`의 실측값(시리얼, 경로)은 커밋 전 노출 여부 확인
5. 파일을 옮길 때는 이동이 아니라 복사(copy)로, 원본 보존

## 자주 쓰는 검증 명령

```bash
git branch --show-current          # seongil/gui-refactor 인지 확인
git remote -v                      # origin=BLINCE1, upstream=DGIST-JaeminBaek
python -c "import lerobot; print(lerobot.__file__)"   # env·설치 경로 확인
ip link show | grep can            # (랩 PC) CAN 인터페이스 확인
```

## 알려진 이슈 / 주의점 (구 리포에서 발견)

- `piper_replay.py`의 `obs_action_mismatch` 검사는 구조적으로 항상 통과(tautology) — 신뢰하지 말 것
- `smolvla_inference.py`의 `ACTION_MIN/MAX`는 하드코딩 — 새 task마다 데이터셋 기준 재계산 필요
- 카메라 warmup 비대칭(top=0s, wrist=5s) → 녹화 초반 프레임 드랍 가능성
- 녹화 fps 30 기준, 카메라 타임아웃 발생 시 GUI가 죽지 않고 복구하도록 처리할 것
