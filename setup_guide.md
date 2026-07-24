# Setup Guide

이 문서는 `lerobot_robot_piper` 레포를 실제 Piper leader/follower 실험에 사용하기 위한 환경 준비와 설정 방법을 정리합니다. 실행 순서는 [docs/operations.md](docs/operations.md)를 참고합니다.

## 1. 기본 환경

권장 환경:

- Ubuntu 22.04 또는 호환 Linux 환경
- Python 3.10 (현재 정상 동작을 확인한 Python 버전은 3.10뿐)
- CAN-USB adapter 2개
- Piper leader arm, Piper follower arm
- 선택: OpenCV camera 또는 Intel RealSense camera

Python 가상환경 예시:

```bash
conda create -n piper_lerobot python=3.10
conda activate piper_lerobot
```

LeRobot과 플러그인을 설치합니다.

```bash
pip install -e .
```

현재 프로젝트는 LeRobot v0.4.4에 맞춰 검증된 상태입니다. 0.4.x의 다른 버전은
비슷한 저장 구조를 쓰더라도 별도 diff 검토와 재검증이 필요합니다.

설치 확인:

```bash
python -c "import lerobot; print(lerobot.__file__)"
python -c "import lerobot_robot_piper; print('piper plugin OK')"
python -c "from piper_sdk import C_PiperInterface_V2; print('piper_sdk OK')"
python -c "import wego_piper; print('wego_piper OK')"
```

RealSense를 사용할 경우:

```bash
python -c "import pyrealsense2 as rs; print('pyrealsense2 OK')"
```

## 2. CAN 설정

기본 CAN bitrate는 `1000000`입니다.

수동 초기화 예시:

```bash
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
sudo ip link set can1 type can bitrate 1000000
sudo ip link set can1 up
```

레포의 도구 스크립트로 수동 초기화할 수도 있습니다.

```bash
bash scripts/tools/setup_can.sh
```

실험 흐름에서는 `configs/recording.env` 값을 기준으로 아래 스크립트를 사용하는 것을 권장합니다.

```bash
bash scripts/1__init_can.sh
```

권장 포트 매핑:

| Arm | Interface 예시 |
|---|---|
| Leader arm | `can_leader1` 또는 `can1` |
| Follower arm | `can_follower1` 또는 `can0` |

통합 GUI(`piper-teleop`)의 CAN Setup 패널(Detect/Init All)로 포트를 식별하고 이름을 고정할 수 있습니다.

USB 물리 포트 기준으로 이름을 고정하려면 `configs/recording.env`의 `LEADER_USB_BUS`, `FOLLOWER_USB_BUS`를 사용합니다. bus 정보는 다음 명령으로 확인합니다.

```bash
sudo ethtool -i can0 | grep bus-info
```

## 3. 설정 파일

설정 파일을 생성합니다.

```bash
cp configs/recording.env.example configs/recording.env
```

주요 설정:

| 변수 | 설명 |
|---|---|
| `LEADER_PORT` | leader arm CAN 포트 |
| `FOLLOWER_PORT` | follower arm CAN 포트 |
| `BITRATE` | CAN bitrate |
| `CAMERA_TYPE` | `opencv` 또는 `intelrealsense` |
| `TOP_CAM` | 상방 카메라 index 또는 RealSense serial/name |
| `WRIST_CAM` | 팔목 카메라 index 또는 RealSense serial/name |
| `CAM_WIDTH`, `CAM_HEIGHT`, `FPS` | 카메라 및 dataset 기록 설정 |
| `DATASET_REPO_ID` | LeRobotDataset repo id |
| `DATASET_ROOT` | 로컬 dataset 저장 경로 |
| `TASK` | episode task 문장 |
| `PARK_ON_CONNECT` | 시작 시 follower parking 여부 |
| `USE_ACTION_OFFSET` | leader/follower 시작 자세 차이 자동 보정 여부 |

처음에는 `PARK_ON_CONNECT=false`, `USE_ACTION_OFFSET=true`를 권장합니다.

## 4. 카메라 설정

카메라 탐색:

```bash
bash scripts/2__find_camera.sh
```

OpenCV 카메라를 사용할 경우 `TOP_CAM`, `WRIST_CAM`에는 video index를 넣습니다.

```bash
CAMERA_TYPE=opencv
TOP_CAM=0
WRIST_CAM=1
```

Intel RealSense를 사용할 경우 serial number를 넣습니다.

```bash
CAMERA_TYPE=intelrealsense
TOP_CAM=327122074262
WRIST_CAM=243322071626
```

RealSense serial과 화면 방향 확인:

```bash
python3 scripts/tools/realsense_view.py --list
python3 scripts/tools/realsense_view.py --serial 327122074262
python3 scripts/tools/realsense_view.py --serial 243322071626
```

확인한 카메라 값을 env 파일에 반영하려면:

```bash
bash scripts/3__set_camera.sh TOP_CAM WRIST_CAM
```

## 5. 동작 확인

CAN 초기화:

```bash
bash scripts/1__init_can.sh
```

Teleoperation dry-run:

```bash
DRY_RUN=true bash scripts/4__teleoperate.sh
```

실제 teleoperation 점검:

```bash
bash scripts/4__teleoperate.sh
```

확인할 것:

- leader/follower 포트가 뒤바뀌지 않음
- follower가 갑자기 튀지 않음
- action offset report가 과도하지 않음
- 카메라 preview 또는 display가 의도한 방향을 보여줌

## 6. 녹화 전 최종 확인

- `configs/recording.env`가 실제 장비 값을 가리킴
- `DATASET_REPO_ID`, `DATASET_ROOT`가 새 실험 이름을 사용함
- top/wrist 카메라 위치가 맞음
- follower 시작 자세가 안전함
- 비상 정지 또는 전원 차단 수단이 준비됨

녹화 명령:

```bash
bash scripts/5__record.sh
```

녹화 후 dataset 확인:

```bash
python3 scripts/tools/wego_dataset_check.py \
  --dataset-repo-id local/piper_write_light \
  --dataset-root records/local/piper_write_light \
  --episode 0
```
