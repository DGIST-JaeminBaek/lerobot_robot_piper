# Roadmap

목표는 준비된 번호형 스크립트와 통합 GUI(`teleop_ui.py`)로 Piper teleoperation, dataset recording, training, async inference까지 반복 가능한 실험 흐름을 만드는 것입니다.

## 완료

### 기반 구조
- [x] `piper_follower` / `piper_leader` 기반 dual-CAN teleoperation 구조 확인
- [x] 운영 파일을 레포 루트의 `scripts/`, `configs/`, `docs/`, `record_sample/`로 정리
- [x] 번호형 스크립트 `1__init_can.sh`부터 `9__run_client.sh` 구성
- [x] 공통 shell helper를 `scripts/lib/run_common.sh`로 분리
- [x] 진단 도구를 `scripts/tools/`로 분리
- [x] OpenCV / Intel RealSense 설정을 `configs/recording.env`로 집중
- [x] `DRY_RUN=true` 명령 확인 경로 유지

### GUI / 통합 콘솔 (`teleop_ui.py`)
- [x] 구 UGRP CLI 도구(`piper_tui.py`, `piper_validate.py` 등)를 `piper_session.py` 하나로 통합, 나머지는 은퇴 후 삭제
- [x] Record/Infer/Replay(Real Robot)/Infer Preview(RViz) 프리셋, Dataset Browser, Recording History, E-STOP 버튼 구현
- [x] 녹화 종료 후 카메라 release 처리, 녹화 초반 프레임 parking 자동 보정(`smooth_start_frames.py`, `SMOOTH_START_FRAMES`로 조절/비활성화)
- [x] GUI 안에 **RViz Start/Stop 토글 버튼** 추가 — `piper_session.py --step rviz`와 동일한 launch를 GUI 자식 프로세스로 관리 (cv2가 심어두는 `QT_QPA_PLATFORM_PLUGIN_PATH` 오염 문제, `/bin/sh`에서 `source` 안 먹는 문제 확인 후 수정)
- [x] Dataset Browser의 **▶ Play 버튼**이 RViz 실행 여부에 따라 `piper_replay_player.py`(RViz 없음)/`piper_replay_player_rviz.py`(RViz 동기화 재생)로 자동 분기하도록 변경, 이제는 중복되던 "Replay (RViz)" 프리셋 제거
- [x] Record/Infer에 **Episode Time(s) / Reset Time(s) / FPS** 입력창 추가(기존엔 Num Episodes만 있었음), `recording.env` 값은 초기값으로만 사용
- [x] 녹화 폴더명에 **Task 텍스트 반영** — `DATASET_REPO_ID`/`DATASET_ROOT`의 네임스페이스(`local/` 등)는 유지하고 마지막 세그먼트만 task 슬러그로 교체(`_task_slug()` / `scripts/lib/run_common.sh`의 `task_slug()`), 셸(`5__record.sh`)과 GUI 양쪽에 동일 적용
- [x] `arm_setup_ui.py`(`piper-setup`)와 그 설계 문서(`FIND-ARM.md`) 제거 — 기능이 `teleop_ui.py`의 CAN Setup 패널(ctrl_mode 자동 판별 포함)로 대체됨
- [x] 미사용 도구/자산 정리: `piper_replay_sim.py`, `piper_urdf_cache/`, `asset/piper-setup.png`, `asset/piper-teleop.png` 삭제. `lerobot_sync_player.py`→`piper_replay_player.py`, `piper_replay_viz_video.py`→`piper_replay_player_rviz.py`로 이름 통일

### RViz / 기구학 검증
- [x] RViz 실행 경로(`piper_session.py --step rviz`, `ros2_ws/src/agx_arm_description`) 실물 PC에서 정상 기동 확인
- [x] **joint 이름/좌표계 컨벤션 검증 완료** — `piper_sdk`의 `CalFK`(DH 파라미터 기반 순수 계산, 하드웨어 불필요)가 존재함을 확인하고, `dh_is_offset=0x01`(기본값)이 Agilex 공식 User Manual DH 표와 정확히 일치함을 수치로 확인. RViz용 URDF(`piper_description.urdf`)로 `ikpy` FK를 돌려 zero-config 결과가 `piper_sdk` `CalFK`와 0.1mm 이하 오차로 일치함을 검증. 우리 팔 펌웨어(`S-V1.8-2`, `configs/config.json`)가 이 컨벤션에 해당하는 신형(J2/J3 좌표계 2° 이동)임도 확인 — **RViz 표시와 SDK FK 계산이 어긋날 걱정 없음**
- [x] `GetFK(mode="feedback"/"control")`이 각각 `GetArmJointMsgs`/`GetArmJointCtrl`(state/action) 값을 받아 `CalFK`를 호출하는 것뿐임을 소스로 확인 — EEF state/action을 추가로 기록하려면 이 함수를 그대로 쓰면 됨(구현은 아직 안 함)

### 확인된 사실 (재검증 불필요)
- [x] `PiperFollower.get_observation()`/`PiperLeader.get_action()` 반환 dict key는 `f"{motor}.pos"` 형태 (joint1.pos~joint6.pos, gripper.pos)
- [x] Plugin 등록 이름: `piper_follower`/`piper_leader` — `config_piper.py`/`config_piper_leader.py`의 `register_subclass()`로 확인
- [x] `--robot.discover_packages_path=lerobot_robot_piper` 없으면 타입 등록 자체가 안 됨
- [x] `record_sample/local/piper_write_light_rs_10s_3eps_v5`로 데이터 로딩~변환~안전검사 e2e 확인 완료
- [x] `lerobot-record`가 `--policy.path=...`를 지원(별도 smolvla-inference CLI 불필요), `smolvla`는 이미 lerobot에 등록된 policy 타입

## 다음 검증

우선순위 순:

### 하드웨어
- [ ] **CAN 연결 기본 동작** — `python scripts/tools/piper_session.py --step joint_check --check_leader`로 follower/leader 둘 다 `[OK]` 뜨는지, 관절값이 실제 팔 자세와 맞는지(부호/방향) 확인
- [ ] `teleop_ui.py`에서 Teleoperate 프리셋으로 leader→follower 텔레옵 실제 확인, 이후 Record로 짧게 1 episode 녹화 테스트 (RViz Start/Play 버튼 자체는 실물 로봇 없이 이미 동작 확인함 — 남은 건 실제 팔 움직임 확인)
- [ ] action offset 자동 보정값이 반복 실행에서 안정적인지 확인
- [ ] `max_relative_target=5.0`이 작업에 적절한지 확인
- [ ] **E-STOP 버튼 실동작** — 실제로 `sudo ip link set ... down`이 즉시 먹는지, sudo 비밀번호 프롬프트에 안 막히는지(NOPASSWD 설정 필요 여부 포함)
- [ ] `piper_session.py`의 `source_prefix()`/`CFG["venv_activate"]` 등 구 UGRP 실험실 PC(`ugrp308`) 하드코딩 경로를 이 PC 기준으로 정리

### 카메라
- [ ] `scripts/2__find_camera.sh`로 카메라 탐색 확인
- [ ] `scripts/tools/realsense_view.py`로 top/wrist serial과 화면 방향 확인
- [ ] RealSense 2대 동시 stream에서 frame 누락 여부 확인
- [ ] `configs/recording.env`의 해상도/FPS/warmup 값 확정
- [ ] "녹화마다 카메라 타임아웃" 현상이 `CAMERA_RELEASE_WAIT_S` 도입 후에도 재현되는지

### 데이터
- [ ] `scripts/5__record.sh` 또는 GUI Record로 1 episode smoke recording (Episode Time/Reset Time/FPS 입력창, Task 기반 폴더명 실측 확인 포함)
- [ ] `scripts/tools/wego_dataset_check.py`로 action/state/camera feature 확인
- [ ] `scripts/6__replay.sh` 또는 GUI Replay (Real Robot)로 기록 action replay 확인
- [ ] gripper 물리 단위 해석(raw→mm→m)이 실제 그리퍼 열림 정도와 맞는지 눈으로 대조
- [ ] `max_delta_per_step`(현재 20, 정규화 단위) 재보정 — 실제 정상 녹화 데이터의 프레임간 delta 기준
- [ ] 실패 episode 처리 규칙 확정

### 학습/추론
- [ ] `scripts/7__train.sh` 학습 명령 dry-run 확인
- [ ] 작은 dataset으로 학습 실행 확인
- [ ] 정책 체크포인트 확보 후 Infer/Infer Preview(RViz) 프리셋 실제 검증 (현재까지 체크포인트 없어서 미검증)
- [ ] `scripts/8__run_server.sh` / `scripts/9__run_client.sh` async 경로 확인
- [ ] 반복 trial 기준 성공률 기록 방식 결정

## 권장 진행 순서

1. `configs/recording.env`를 실제 장비 포트와 카메라 serial로 맞춘다.
2. `DRY_RUN=true bash scripts/4__teleoperate.sh`와 `DRY_RUN=true bash scripts/5__record.sh`로 명령을 확인한다.
3. 실제 teleop을 짧게 실행해 시작 자세와 offset을 확인한다.
4. 1 episode만 녹화하고 dataset feature를 확인한다.
5. 안정화 후 episode 수를 늘린다.
6. training과 async inference를 별도 단계로 검증한다.
