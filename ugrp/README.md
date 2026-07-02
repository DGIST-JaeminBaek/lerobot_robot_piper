# UGRP Piper LeRobot 녹화 도구

Piper 마스터-슬레이브 텔레옵으로 LeRobot 데이터셋을 녹화하기 위한 팀 작업용 저장소입니다.

## 목표 작업

- 입력: 상방 카메라, 팔목 카메라, Piper follower 현재 상태, task text `write AIIII`
- 시연: leader Piper를 손으로 조종해서 follower Piper가 펜을 집고 화이트보드에 `AIIII` 쓰기
- 출력: imitation learning 학습에 사용할 LeRobot dataset

## 저장소 역할

이 디렉터리는 `lerobot_robot_piper` 플러그인 레포 안에 포함된 UGRP 운영용 스크립트 모음입니다. Piper 로봇 구현은 레포 루트의 `lerobot_robot_piper/` 패키지가 담당하고, 이 `ugrp/` 디렉터리는 CAN 초기화, 카메라 설정, teleoperation, recording, replay, training, async inference 실행 흐름을 정리합니다.

레포 루트에서 플러그인을 editable install 한 뒤 `ugrp/` 스크립트를 실행합니다.

```bash
cd lerobot_robot_piper
pip install -e .
cd ugrp
```

## 빠른 시작

설정 파일 생성:

```bash
cp configs/recording.env.example configs/recording.env
```

`configs/recording.env`에서 서버 환경에 맞게 수정:

- `LEADER_PORT`: leader arm CAN 포트
- `FOLLOWER_PORT`: follower arm CAN 포트
- `CAMERA_TYPE`: `opencv` 또는 `intelrealsense`
- `TOP_CAM`: 상방 카메라 index 또는 RealSense serial/name
- `WRIST_CAM`: 팔목 카메라 index 또는 RealSense serial/name
- `HF_USER` 또는 `DATASET_REPO_ID`: 저장할 dataset 이름

Linux 서버에서 CAN 초기화:

```bash
bash scripts/1__init_can.sh
```

카메라 번호 확인:

```bash
bash scripts/2__find_camera.sh
```

RealSense만 확인하려면:

```bash
bash scripts/2__find_camera.sh realsense
```

확인한 카메라 번호 저장:

```bash
bash scripts/3__set_camera.sh 6 0
```

Intel RealSense를 LeRobot RealSense backend로 쓰려면 `configs/recording.env`에서:

```bash
CAMERA_TYPE=intelrealsense
TOP_CAM=327122074262
WRIST_CAM=243322071626
REALSENSE_USE_DEPTH=false
```

텔레옵 점검:

```bash
bash scripts/4__teleoperate.sh
```

`write AIIII` 녹화:

```bash
bash scripts/5__record.sh
```

## 기존 ugrp 대비 변경점

이 디렉터리는 기존 `ugrp` 스크립트 묶음을 `lerobot_robot_piper` 플러그인 레포 안에서 바로 실행할 수 있도록 정리한 버전입니다. 기존 ugrp의 실험 흐름은 유지하되, LeRobot 0.4.0과 플러그인 패키지 구조에 맞게 실행 방식과 설정 방식을 바꿨습니다.

주요 차이는 다음과 같습니다.

| 항목 | 기존 ugrp | 이 디렉터리 |
|---|---|---|
| 레포 구조 | ugrp 단독 실행 스크립트 묶음 | `lerobot_robot_piper/ugrp` 하위 운영 스크립트로 포함 |
| Piper 등록 | 실행 환경에 이미 잡힌 Piper 코드를 전제로 함 | 모든 주요 CLI 호출에 `--robot.discover_packages_path=lerobot_robot_piper`, `--teleop.discover_packages_path=lerobot_robot_piper` 추가 |
| LeRobot 실행 방식 | 기존 스크립트/레포 구조 기준 호출 | LeRobot 0.4.0 CLI인 `lerobot-teleoperate`, `lerobot-record`, `lerobot-replay`, `lerobot-train` 기준 |
| 설정 관리 | 스크립트별 인자와 설정이 나뉘어 있음 | `configs/recording.env` 하나에 CAN, camera, dataset, train, async 설정 집중 |
| 번호형 실행 흐름 | 원본 번호형 스크립트 흐름 | 플러그인 패키지에서 실행 가능하도록 `scripts/1__init_can.sh`~`9__run_client.sh` 정리 |
| 카메라 설정 | OpenCV index 중심 | `CAMERA_TYPE=opencv` 또는 `intelrealsense` 선택 가능, top/wrist별 혼합도 가능 |
| RealSense 값 | 사용자가 별도 입력 | top `327122074262`, wrist `243322071626` 확인값을 README/env example에 반영 |
| 실행 전 점검 | 실제 실행 위주 | `DRY_RUN=true`로 하드웨어 실행 전 명령 확인 가능 |
| 데이터 점검 | 별도 확인 필요 | `wego_dataset_check.py`로 LeRobot dataset feature/action/state 확인 |

즉, 이 폴더는 기존 ugrp 흐름을 버린 것이 아니라, `DGIST-JaeminBaek/lerobot_robot_piper` 레포에 포함해 팀원이 같은 플러그인 설치 환경에서 재현할 수 있게 맞춘 운영용 래퍼입니다.

## lerobot_piper 번호 스크립트 포팅

`lerobot_piper` fork 레포의 `1__init_can.sh`부터 `9__run_client.sh` 흐름을 플러그인 패키지용으로 옮긴 스크립트입니다. fork 내부의 `python ./src/lerobot/...` 호출 대신 현재 환경에 설치된 `lerobot-*` CLI와 `python -m lerobot...`를 사용합니다.

| 파일 | 역할 |
|---|---|
| `scripts/1__init_can.sh` | CAN 인터페이스 bitrate 설정 및 선택적 USB bus 기반 rename |
| `scripts/2__find_camera.sh` | LeRobot 카메라 탐색 또는 `camera_check.py` 실행 |
| `scripts/3__set_camera.sh` | `configs/recording.env`의 `TOP_CAM`/`WRIST_CAM` 갱신 |
| `scripts/4__teleoperate.sh` | `piper_follower`/`piper_leader` 텔레옵 점검 |
| `scripts/5__record.sh` | 카메라 포함 LeRobot dataset 녹화 |
| `scripts/6__replay.sh` | 녹화 dataset replay |
| `scripts/7__train.sh` | `lerobot-train` 기반 정책 학습 |
| `scripts/8__run_server.sh` | LeRobot async policy server 실행 |
| `scripts/9__run_client.sh` | LeRobot async robot client 실행 |

명령만 확인하고 하드웨어를 건드리지 않으려면 `DRY_RUN=true`를 붙입니다.

```bash
DRY_RUN=true bash scripts/4__teleoperate.sh
DRY_RUN=true bash scripts/5__record.sh
DRY_RUN=true bash scripts/7__train.sh
```

비동기 추론용 `8__run_server.sh`, `9__run_client.sh`는 `grpc` 의존성이 필요합니다. 현재 환경에서 `python -c "import grpc"`가 실패하면 먼저 async 관련 의존성을 설치해야 합니다.

## Python 스크립트

기존 Python wrapper도 유지합니다.

```bash
python3 scripts/wego_teleop_smoke.py --dry-run
python3 scripts/wego_record_write_light.py --dry-run
```

녹화 후 dataset 구조 확인:

```bash
python3 scripts/wego_dataset_check.py --dataset-repo-id your_hf_name/piper_write_light --episode 0
```

## 파일 구성

| 파일 | 역할 |
|---|---|
| `configs/recording.env.example` | 서버별 설정 예시 |
| `scripts/run_common.sh` | 번호형 Bash 스크립트 공통 env/명령 로더 |
| `scripts/1__init_can.sh` ~ `scripts/9__run_client.sh` | `lerobot_piper` fork의 운영 스크립트를 플러그인용으로 정리한 실행 흐름 |
| `scripts/setup_can.sh` | Linux CAN 포트 초기화 |
| `scripts/teleop_smoke_test.sh` | 텔레옵 점검용 Bash wrapper |
| `scripts/record_write_light.sh` | 녹화용 Bash wrapper |
| `scripts/common_env.py` | Python 스크립트 공통 env 로더 |
| `scripts/wego_teleop_smoke.py` | WeGo `piper_follower`/`piper_leader` 기반 텔레옵 점검 |
| `scripts/wego_record_write_light.py` | WeGo 기반 `write AIIII` dataset 녹화 |
| `scripts/wego_dataset_check.py` | LeRobot dataset feature/action/state 확인 |
| `docs/setup_checklist.md` | 실험 전 체크리스트 |
| `docs/data_collection_protocol.md` | 데이터 수집 프로토콜 |

## 안전 기본값

기본 `MAX_RELATIVE_TARGET`은 `5`입니다. leader와 follower 초기 자세가 잘 맞지 않으면 follower가 갑자기 움직일 수 있으므로 더 낮은 값으로 시작합니다.

```bash
MAX_RELATIVE_TARGET=2 bash scripts/4__teleoperate.sh
```

초기 자세가 계속 어긋나면 `piper-calibrate`로 zero-point를 먼저 맞춥니다.
