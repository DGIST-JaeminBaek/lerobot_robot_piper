# Roadmap

목표는 준비된 번호형 스크립트와 통합 GUI(`teleop_ui.py`)로 Piper teleoperation, dataset recording, training, async inference까지 반복 가능한 실험 흐름을 만드는 것입니다.

## 완료

### 기반 구조
- [x] 단일 Piper leader/follower 구조를 LeRobot plugin 타입(`piper_leader`, `piper_follower`)으로 정리
- [x] 실행 진입점을 `configs/recording.env`, 번호형 스크립트(`scripts/0__launch_gui.sh`~`9__run_client.sh`), `scripts/lib/run_common.sh`로 통일
- [x] 진단/보조 도구를 `scripts/tools/`로 정리하고, 주요 스크립트에 `DRY_RUN=true` 확인 경로 유지
- [x] 현재 검증 기준을 Python 3.10 + LeRobot v0.4.4로 정리

### GUI / 통합 콘솔 (`teleop_ui.py`)
- [x] CAN Setup, Teleoperate, Record, Replay, Infer, RViz, Dataset Browser, Recording History를 `piper-teleop` 중심으로 통합
- [x] Record/Infer에서 Task, episode/reset time, FPS, Push to Hub, Smooth Start, Dataset Root override를 GUI로 조절하고 `configs/recording.env`에 저장 가능
- [x] Dataset Browser의 Play 동작을 RViz 실행 여부와 RGB/depth view 옵션에 맞게 자동 분기
- [x] E-STOP 버튼 실동작 확인 완료
- [x] 중복/미사용 GUI와 자산(`piper-setup`, 구 replay/sync 도구, 관련 이미지)을 정리

### RViz / 기구학 검증
- [x] RViz 실행 경로와 URDF 표시를 실물 PC에서 확인
- [x] `piper_sdk` `CalFK`, RViz URDF FK, 팔 펌웨어 EEF 피드백(`GetArmEndPoseMsgs()`)의 컨벤션 일치 확인
- [x] 현재 팔 펌웨어(`S-V1.8-2`) 기준으로 계산 FK를 EEF state/action 계산에 사용 가능함을 확인

### 카메라 / Depth / 녹화
- [x] top/wrist RealSense serial, 화면 방향, 해상도/FPS/warmup 설정 확정
- [x] RGB+depth read와 RealSense connect를 병렬화해 2대 동시 사용 경로를 안정화
- [x] LeRobot v0.4.4 로컬 clone에 depth 백포트를 재적용하고, 실제 RealSense depth 녹화/저장/재로드 검증 완료
- [x] depth replay와 depth-only viewer에서 mm depth 컬러맵 표시 경로 확인
- [x] 반복 녹화 시 카메라 release/timeout 경로 확인 완료

### Teleop / 데이터 검증
- [x] CAN 연결, leader→follower teleop, action offset, `max_relative_target=5.0` 안전 제한 동작 확인
- [x] `PiperFollower.get_observation()`/`PiperLeader.get_action()` key와 plugin 등록 이름(`piper_follower`, `piper_leader`) 확인
- [x] `lerobot-record`가 저장하는 `action`을 실제 follower에 보낸 offset 적용 후 목표값으로 수정
- [x] "End Episode (Save)" 버튼으로 현재 에피소드 저장 및 다음 에피소드 진행 확인
- [x] `scripts/tools/wego_dataset_check.py` 기준 action/state/camera feature 확인
- [x] 기록 action replay와 `max_delta_per_step` 안전검사 확인

## 다음 검증

우선순위 순:

### 데이터
- [ ] gripper 물리 단위 해석(raw→mm→m)이 실제 그리퍼 열림 정도와 맞는지 눈으로 대조

### 학습/추론
- [ ] `scripts/7__train.sh` 학습 명령 dry-run 확인
- [ ] 작은 dataset으로 학습 실행 확인
- [ ] 정책 체크포인트 확보 후 Infer/Infer Preview(RViz) 프리셋 실제 검증 (현재까지 체크포인트 없어서 미검증)
- [ ] `scripts/8__run_server.sh` / `scripts/9__run_client.sh` async 경로 확인
- [ ] 반복 trial 기준 성공률 기록 방식 결정

## 권장 진행 순서

1. gripper 물리 단위 해석을 실제 열림 정도와 눈으로 대조한다.
2. `scripts/7__train.sh`를 dry-run으로 확인한다.
3. 작은 dataset으로 학습을 실행한다.
4. 정책 체크포인트를 확보한 뒤 GUI Infer / Infer Preview(RViz)를 검증한다.
5. `scripts/8__run_server.sh` / `scripts/9__run_client.sh` async 경로를 검증한다.
6. 반복 trial의 성공률 기록 방식을 정한다.
