# lerobot_robot_piper

**Agilex Piper** 7-DOF 로봇팔을 위한 LeRobot 플러그인입니다. `piper_follower` 로봇 인터페이스와 `piper_leader` 텔레오퍼레이터 인터페이스를 제공하며, 이 레포에서는 실험 실행을 위해 `configs/recording.env`와 `scripts/` 번호형 스크립트를 함께 제공합니다.

## 주요 기능

- **Leader-Follower Teleoperation**: leader 팔의 움직임을 follower 팔에 실시간 반영
- **Dataset Recording**: joint position과 camera frame을 LeRobotDataset으로 기록
- **CAN Bus Communication**: `piper_sdk`, `wego_piper` 기반 하드웨어 제어
- **Safety Limits**: `max_relative_target`으로 timestep별 joint 이동량 제한
- **Camera Integration**: OpenCV/Intel RealSense 카메라를 follower observation으로 기록
- **GUI Tools**: `piper-ui`, `piper-teleop`, `piper-setup`, `piper-calibrate` 제공
- **Experiment Scripts**: CAN 초기화부터 record/train/async 실행까지 번호형 스크립트 제공

## 요구 사항

- Python >= 3.10
- LeRobot >= 0.3.0
- [`piper_sdk`](https://github.com/agilexrobotics/piper_sdk)
- [`wego_piper`](https://github.com/agilexrobotics/wego_piper)
- Piper arm 및 CAN-USB interface
- 선택: OpenCV camera 또는 Intel RealSense camera

설치:

```bash
pip install -e .
```

## 빠른 시작

먼저 설정 파일을 만듭니다.

```bash
cp configs/recording.env.example configs/recording.env
```

`configs/recording.env`에서 장비에 맞게 아래 값을 수정합니다.

| 설정 | 설명 |
|---|---|
| `LEADER_PORT` | leader arm CAN 포트 |
| `FOLLOWER_PORT` | follower arm CAN 포트 |
| `CAMERA_TYPE` | `opencv` 또는 `intelrealsense` |
| `TOP_CAM` | 상방 카메라 index 또는 RealSense serial/name |
| `WRIST_CAM` | 팔목 카메라 index 또는 RealSense serial/name |
| `DATASET_REPO_ID` | 저장할 dataset 이름 |
| `DATASET_ROOT` | 로컬 dataset 저장 경로 |

기본 실행 순서:

```bash
bash scripts/1__init_can.sh
bash scripts/2__find_camera.sh
bash scripts/4__teleoperate.sh
bash scripts/5__record.sh
```

명령만 확인하려면 `DRY_RUN=true`를 붙입니다.

```bash
DRY_RUN=true bash scripts/4__teleoperate.sh
DRY_RUN=true bash scripts/5__record.sh
```

자세한 실행 절차는 [docs/operations.md](docs/operations.md)를 참고합니다.

## 번호형 스크립트

| 파일 | 역할 |
|---|---|
| `scripts/1__init_can.sh` | CAN 인터페이스 bitrate 설정 및 선택적 USB bus 기반 rename |
| `scripts/2__find_camera.sh` | LeRobot 카메라 탐색 또는 `scripts/tools/camera_check.py` 실행 |
| `scripts/3__set_camera.sh` | `configs/recording.env`의 `TOP_CAM`/`WRIST_CAM` 갱신 |
| `scripts/4__teleoperate.sh` | `piper_follower`/`piper_leader` 텔레옵 점검 |
| `scripts/5__record.sh` | 카메라 포함 LeRobot dataset 녹화 |
| `scripts/6__replay.sh` | 녹화 dataset replay |
| `scripts/7__train.sh` | `lerobot-train` 기반 policy 학습 |
| `scripts/8__run_server.sh` | LeRobot async policy server 실행 |
| `scripts/9__run_client.sh` | LeRobot async robot client 실행 |

공통 로직은 `scripts/lib/run_common.sh`에 있습니다. 번호형 스크립트들이 이 파일을 `source`해서 env 로드, dry-run, camera/action 인자 생성을 공유합니다.

## 도구 스크립트

| 파일 | 역할 |
|---|---|
| `scripts/tools/setup_can.sh` | 수동 CAN 초기화 도구 |
| `scripts/tools/camera_check.py` | OpenCV 카메라 index grid viewer |
| `scripts/tools/realsense_view.py` | RealSense serial 확인 및 RGB stream 미리보기 |
| `scripts/tools/wego_dataset_check.py` | 녹화된 LeRobotDataset feature/action/state 점검 |

예시:

```bash
python3 scripts/tools/realsense_view.py --list
python3 scripts/tools/realsense_view.py --serial 327122074262
python3 scripts/tools/wego_dataset_check.py --dataset-repo-id local/piper_write_light --episode 0
```

## 하드웨어 설정

각 CAN 인터페이스는 연결 전에 활성화되어 있어야 합니다. 일반적인 수동 초기화는 다음과 같습니다.

```bash
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
```

실험에서는 보통 `scripts/1__init_can.sh`를 사용합니다.

권장 매핑:

| Arm | Interface 예시 |
|---|---|
| Follower robot | `can0` 또는 `can_follower1` |
| Leader teleoperator | `can1` 또는 `can_leader1` |

`piper-setup`으로 여러 CAN 포트를 감지하고 `can_leader1`, `can_follower1` 같은 고정 이름으로 설정할 수 있습니다.

## GUI 도구

![piper-ui](asset/piper-ui.png)
![piper-teleop](asset/piper-teleop.png)
![piper-setup](asset/piper-setup.png)

### `piper-setup`

```bash
piper-setup
```

여러 arm 설정용 wizard입니다. CAN 포트 스캔, leader/follower 역할 지정, arm 식별, CAN 이름 고정을 지원합니다.

### `piper-ui`

```bash
piper-ui
```

단일 Piper arm 직접 제어 UI입니다. CAN 설정, role 전환, torque on/off, parking, joint slider 제어를 제공합니다.

### `piper-teleop`

```bash
piper-teleop
```

Teleoperation 모니터링 UI입니다. leader/follower 상태 확인과 teleoperation/recording 명령 실행을 지원합니다.

## 직접 사용 예시

번호형 스크립트가 기본 진입점이지만, 플러그인을 직접 사용할 수도 있습니다.

### Python API

```python
from lerobot_robot_piper import PiperFollowerConfig, PiperLeaderConfig, PiperFollower, PiperLeader

follower_cfg = PiperFollowerConfig(port="can0")
leader_cfg = PiperLeaderConfig(port="can1")

follower = PiperFollower(follower_cfg)
leader = PiperLeader(leader_cfg)

follower.connect()
leader.connect()

try:
    while True:
        action = leader.get_action()
        follower.send_action(action)
        obs = follower.get_observation()
finally:
    follower.disconnect()
    leader.disconnect()
```

### LeRobot CLI

```bash
python -m lerobot.teleoperate \
    --robot.type=piper_follower \
    --robot.port=can0 \
    --teleop.type=piper_leader \
    --teleop.port=can1 \
    --robot.discover_packages_path=lerobot_robot_piper \
    --teleop.discover_packages_path=lerobot_robot_piper
```

## 설정 요약

### `PiperFollowerConfig`

| Parameter | Type | 설명 |
|---|---|---|
| `port` | `str` | follower arm CAN 포트 |
| `disable_torque_on_disconnect` | `bool` | disconnect 시 torque off 여부 |
| `park_on_connect` | `bool` | connect 시 parking pose 이동 여부 |
| `cameras` | `dict[str, CameraConfig]` | observation camera 설정 |
| `max_relative_target` | `float \| dict \| None` | timestep별 최대 joint 이동량 |
| `use_action_offset` | `bool` | leader/follower 시작 자세 차이 보정 |

### `PiperLeaderConfig`

| Parameter | Type | 설명 |
|---|---|---|
| `port` | `str` | leader arm CAN 포트 |
| `gripper_open_pos` | `float` | gripper open 기준값 |

## Motor Configuration

Piper arm은 7개 joint를 사용합니다. 정규화된 값 범위는 아래와 같습니다.

| Joint | Model | Normalized Range | Physical Range |
|---|---|---|---|
| Joint 1 | AGILEX-M | -100 to +100 | +/-150 deg |
| Joint 2 | AGILEX-M | -100 to +100 | 0-180 deg |
| Joint 3 | AGILEX-M | -100 to +100 | -170-0 deg |
| Joint 4 | AGILEX-S | -100 to +100 | +/-100 deg |
| Joint 5 | AGILEX-S | -100 to +100 | +/-65 deg |
| Joint 6 | AGILEX-S | -100 to +100 | +/-100-130 deg |
| Gripper | AGILEX-S | 0 to 100 | 0-68 deg |

Parking pose의 normalized 값은 `0, -100, 100, 0, 35, 0, 0`입니다.

## 문서

| 문서 | 내용 |
|---|---|
| [docs/operations.md](docs/operations.md) | 실행 절차 |
| [setup_guide.md](setup_guide.md) | 환경 준비와 설정 방법 |
| [docs/data_collection_protocol.md](docs/data_collection_protocol.md) | 데이터 수집 프로토콜 |
| [docs/roadmap.md](docs/roadmap.md) | 남은 작업 |
| [docs/change_history/](docs/change_history/) | 변경 이력 |

## Project Structure

```text
.
|-- lerobot_robot_piper/      # LeRobot plugin source
|-- scripts/                  # Numbered experiment scripts
|   |-- lib/                  # Shared shell helpers
|   `-- tools/                # Manual diagnostics and checks
|-- configs/                  # Environment templates
|-- docs/                     # Project docs
|-- record_sample/            # Sample LeRobot dataset
|-- asset/                    # README/UI images
|-- pyproject.toml
`-- README.md
```

## License

Apache-2.0
