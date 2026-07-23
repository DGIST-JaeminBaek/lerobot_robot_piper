# WEGO 원본 대비 변경 사항

비교 기준:

- 원본: [WeGo-Robotics/lerobot_robot_piper](https://github.com/WeGo-Robotics/lerobot_robot_piper)
- 기준 HEAD: `32c55dd44b6ca4f8b793a7fd9852afd3d304b443`

원본 WEGO 구현의 leader/follower 구조(및 `teleop_ui.py`의 subprocess-launcher + CAN monitor 골격)는 그대로 유지했습니다. 아래는 그 위에서 실제로 무엇을, 왜 바꿨는지를 주제별로 정리한 것입니다 — 날짜순 이력이 아니라 "지금 코드가 왜 이런 모양인지"에 대한 설명입니다.

## 1. LeRobot 0.4 호환성 보완

WEGO 원본은 `MotorsBusBase` 기반 구현입니다. 이 레포는 LeRobot 0.4 계열의 `MotorsBus` 추상 인터페이스에 맞춰 `PiperMotorsBus`를 수정했습니다.

Before (WEGO 원본 `lerobot_robot_piper/motors/piper_motors_bus.py`):

```python
from lerobot.motors.motors_bus import MotorsBusBase

class PiperMotorsBus(MotorsBusBase):
    apply_drive_mode = False
```

After (`lerobot_robot_piper/motors/piper_motors_bus.py`):

```python
from lerobot.motors.motors_bus import MotorsBus

class PiperMotorsBus(MotorsBus):
    apply_drive_mode = False
    available_baudrates = [1000000]
    default_baudrate = 1000000
    default_timeout = 1000
    model_baudrate_table = {model: [1000000] for model in MODEL_BAUDRATE_TABLE}
    model_ctrl_table = {model: {} for model in MODEL_NUMBER_TABLE}
    model_encoding_table = MODEL_ENCODING_TABLE
    model_number_table = MODEL_NUMBER_TABLE
    model_resolution_table = MODEL_RESOLUTION_TABLE
    normalized_data = ["Present_Position", "Goal_Position"]
```

`MotorsBus`가 요구하는 serial-protocol method stub도 추가했습니다(CAN 기반 Piper에는 맞지 않는 기능이라 실제 구현 없이 compatibility layer로만 둠):

```python
def _assert_protocol_is_compatible(self, instruction_name: str) -> None:
    pass

def _handshake(self) -> None:
    pass

def _find_single_motor(self, motor: str, initial_baudrate: int | None = None) -> tuple[int, int]:
    raise NotImplementedError("Piper CAN bus does not support single motor discovery.")
```

**의미**: 기능 추가가 아니라 LeRobot 버전 차이로 인한 실행 불가 문제를 막기 위한 순수 호환성 패치입니다.

## 2. CLI/env 기반 카메라 설정 추가

WEGO 원본은 `PiperFollowerConfig.cameras`에 이미 구성된 `CameraConfig` dict를 넘기는 구조입니다. 이 레포는 `top_cam`, `wrist_cam`, `camera_type` 같은 단순 필드를 받고 `__post_init__()`에서 실제 camera config를 생성하도록 바꿨습니다.

Before (WEGO 원본 `config_piper.py`):

```python
disable_torque_on_disconnect: bool = True
cameras: dict[str, CameraConfig] = field(default_factory=dict)
max_relative_target: float | dict[str, float] | None = None
```

After (`config_piper.py`):

```python
disable_torque_on_disconnect: bool = True
park_on_connect: bool = False
cameras: dict[str, CameraConfig] = field(default_factory=dict)
camera_type: str = "opencv"
top_cam_type: str = ""
wrist_cam_type: str = ""
top_cam: str = ""
wrist_cam: str = ""
cam_width: int = 640
cam_height: int = 480
camera_fps: int = 30
realsense_use_depth: bool = False
realsense_warmup_s: float = 5.0
camera_connect_warmup: bool = False
camera_post_connect_wait_s: float = 2.0
```

**의미**: `--robot.cameras={...}` 같은 복잡한 dict CLI 인자 없이, `configs/recording.env`에서 카메라 값을 단순 문자열로 관리할 수 있게 됩니다. OpenCV index와 RealSense serial을 같은 설정 경로로 처리합니다.

## 3. Follower 강제 parking 비활성화

WEGO 원본은 follower 연결 시 항상 parking pose로 이동합니다. 이 레포는 `park_on_connect` 설정을 추가하고 기본값을 `False`로 두었습니다.

```python
# After (piper_follower.py)
def connect(self, calibrate: bool = True) -> None:
    self.bus.connect()
    self.bus.enable_torque()
    if calibrate and self.config.park_on_connect:
        self.bus.parking()
```

**의미**: 프로그램을 재실행해도 follower가 매번 초기 자세로 강제 이동하지 않습니다. 필요하면 `park_on_connect=true`로 원본과 같은 동작을 켤 수 있습니다.

## 4. Leader/Follower action offset 추가

WEGO 원본은 leader action을 그대로 follower goal로 보냅니다. 이 레포는 첫 leader action과 follower 현재 위치의 차이를 offset으로 계산해서, 이후 action에 더해 follower 현재 자세 기준의 상대 추종으로 변환합니다(`use_action_offset`, `ACTION_OFFSET_*` 설정).

**의미**: leader/follower 시작 자세 차이를 자동으로 흡수하고, 프로그램 재시작 후에도 follower가 현재 자세 기준으로 이어서 움직입니다. Offset report로 시작 자세 차이가 큰 joint를 확인할 수 있습니다.

## 5. RealSense 동시 사용 안정화

WEGO 원본은 camera별 기본 `connect()`를 그대로 호출합니다. 이 레포는 warmup 여부와 post-connect wait를 설정으로 제어합니다(`camera_connect_warmup`, `camera_post_connect_wait_s`, `realsense_warmup_s`).

**의미**: RealSense 두 대를 동시에 쓸 때 stream 시작 순서/warmup으로 인한 timeout 문제를 줄입니다.

## 6. Bimanual(양팔) Piper 지원 — WEGO 원본에 없는 기능

WEGO 원본에는 `bi_piper_follower.py`/`bi_piper_leader.py`/`config_bi_piper.py`/`config_bi_piper_leader.py`에 해당하는 파일이 없습니다(원본은 단일 팔 `piper_follower`/`piper_leader`만 지원). 이 레포는 follower/leader 두 벌을 하나의 로봇/텔레옵 단위로 묶는 bimanual wrapper 타입을 추가했습니다.

**의미**: `--robot.type=bi_piper_follower` / `--teleop.type=bi_piper_leader`로 양팔 조작·녹화가 가능해집니다. observation/action은 `left_`/`right_` 접두사로 구분됩니다(예: `left_joint1.pos`, `right_joint1.pos`).

## 7. `teleop_ui.py` — 원본의 launcher 골격을 유지한 채 대폭 확장

WEGO 원본의 `teleop_ui.py`(엔트리포인트 `piper-monitor`)도 이미 "CAN 인터페이스 감지/초기화 + 미리 구성된 원격조종/기록 명령 실행 + 관절 위치·팔로워 상태 실시간 표시"라는 기본 골격을 갖고 있었습니다. 이 레포는 그 골격(subprocess launcher + CAN monitor) 자체는 그대로 두고, 그 위에 다음을 추가했습니다:

- Record/Infer/Replay(Real Robot)/Infer Preview(RViz) 프리셋과 Dataset Browser, Recording History
- E-STOP 버튼(follower+leader CAN 즉시 차단), RViz Start/Stop 토글, Play 버튼의 RViz 유무 자동 분기 재생
- 녹화 종료 후 카메라 release 처리, 녹화 초반 프레임 parking 자동 보정(`smooth_start_frames.py`)
- Task 텍스트를 dataset 폴더명에 반영, Episode Time/Reset Time/FPS를 GUI에서 직접 조절

엔트리포인트 이름도 `piper-monitor`에서 `piper-teleop`으로 바꿨습니다.

**의미**: CAN 신호를 보내는 각 도구를 따로 실행하지 않고, GUI 하나에서 텔레옵→녹화→재생→추론 미리보기까지 이어지는 실험 흐름을 처리할 수 있습니다.

## 8. `piper-setup`(`arm_setup_ui.py`) 제거 — WEGO 원본엔 있었지만 이 레포에서 삭제함

WEGO 원본은 CAN 포트 스캔/역할 지정을 위한 별도 마법사(`arm_setup_ui.py`, `piper-setup`)를 제공합니다. 이 레포는 이 파일을 삭제했습니다 — 같은 기능(CAN 감지, 이름 고정)이 `teleop_ui.py`의 CAN Setup 패널에 이미 있었고, 여기에 `ctrl_mode`를 읽어 leader/follower 역할을 자동 판별하는 기능까지 추가되어 있어 완전히 대체 가능했기 때문입니다.

**의미**: 같은 일을 하는 GUI가 두 개로 나뉘어 있던 걸 하나로 합쳤습니다. 다만 `piper-setup`이 갖고 있던 "USB bus-info ↔ CAN 이름 매핑을 파일로 저장했다가 재부팅 후 복원"하는 기능은 아직 `teleop_ui.py` 쪽으로 옮기지 않았습니다 — 필요해지면 별도로 이식해야 합니다.

## 최종 해석

WEGO 구현을 대체하거나 재작성한 게 아니라, WEGO의 Piper follower/leader 구현과 `teleop_ui.py` 골격을 그대로 주 경로로 두고 다음을 보완/확장한 것입니다.

1. LeRobot 0.4 계열과의 interface 불일치 해소
2. CLI/env 기반 camera 설정으로 실험 스크립트에서 다루기 쉽게 함
3. follower 강제 parking으로 인한 시작 자세 틀어짐 방지
4. leader/follower 시작 자세 차이로 인한 follower jump 방지
5. RealSense 두 대 사용 시 stream 안정성 확보
6. bimanual(양팔) 지원 추가
7. `teleop_ui.py`를 녹화/추론/재생/RViz까지 아우르는 통합 콘솔로 확장
8. 중복되던 `piper-setup` 마법사를 통합 콘솔의 CAN Setup 패널로 흡수
