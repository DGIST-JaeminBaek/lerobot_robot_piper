# Roadmap

목표는 준비된 번호형 스크립트로 Piper teleoperation, dataset recording, training, async inference까지 반복 가능한 실험 흐름을 만드는 것입니다.

## 완료

- [x] `piper_follower` / `piper_leader` 기반 dual-CAN teleoperation 구조 확인
- [x] 운영 파일을 레포 루트의 `scripts/`, `configs/`, `docs/`, `record_sample/`로 정리
- [x] 번호형 스크립트 `1__init_can.sh`부터 `9__run_client.sh` 구성
- [x] 공통 shell helper를 `scripts/lib/run_common.sh`로 분리
- [x] 진단 도구를 `scripts/tools/`로 분리
- [x] OpenCV / Intel RealSense 설정을 `configs/recording.env`로 집중
- [x] `DRY_RUN=true` 명령 확인 경로 유지

## 다음 검증

### 하드웨어

- [ ] 실제 장비에서 `scripts/1__init_can.sh` 실행 확인
- [ ] `can_leader1` / `can_follower1` 이름 고정 확인
- [ ] `scripts/4__teleoperate.sh`로 follower 추종 안정성 확인
- [ ] action offset 자동 보정값이 반복 실행에서 안정적인지 확인
- [ ] `max_relative_target=5.0`이 작업에 적절한지 확인

### 카메라

- [ ] `scripts/2__find_camera.sh`로 카메라 탐색 확인
- [ ] `scripts/tools/realsense_view.py`로 top/wrist serial과 화면 방향 확인
- [ ] RealSense 2대 동시 stream에서 frame 누락 여부 확인
- [ ] `configs/recording.env`의 해상도/FPS/warmup 값 확정

### 데이터

- [ ] `scripts/5__record.sh`로 1 episode smoke recording
- [ ] `scripts/tools/wego_dataset_check.py`로 action/state/camera feature 확인
- [ ] `scripts/6__replay.sh`로 기록 action replay 확인
- [ ] 실패 episode 처리 규칙 확정

### 학습/추론

- [ ] `scripts/7__train.sh` 학습 명령 dry-run 확인
- [ ] 작은 dataset으로 학습 실행 확인
- [ ] `scripts/8__run_server.sh` / `scripts/9__run_client.sh` async 경로 확인
- [ ] 반복 trial 기준 성공률 기록 방식 결정

## 권장 진행 순서

1. `configs/recording.env`를 실제 장비 포트와 카메라 serial로 맞춘다.
2. `DRY_RUN=true bash scripts/4__teleoperate.sh`와 `DRY_RUN=true bash scripts/5__record.sh`로 명령을 확인한다.
3. 실제 teleop을 짧게 실행해 시작 자세와 offset을 확인한다.
4. 1 episode만 녹화하고 dataset feature를 확인한다.
5. 안정화 후 episode 수를 늘린다.
6. training과 async inference를 별도 단계로 검증한다.
