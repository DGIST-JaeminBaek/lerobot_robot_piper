# ROS 2 / RViz / URDF Setup

이 문서는 새 PC에서 이 프로젝트의 RViz 환경을 재현하는 절차를 설명합니다.
현재 RViz는 로봇을 제어하지 않고 다음 용도로만 사용합니다.

- Piper URDF와 TF 표시
- 녹화한 joint state replay
- policy action 미리보기
- `/preview_trajectory` marker 표시

실제 leader/follower 제어는 ROS 드라이버가 아니라 이 프로젝트의
`piper_sdk`와 `PiperMotorsBus`가 담당합니다.

## 1. 구성 선택

현재 프로젝트에서는 `agilexrobotics/agx_arm_ros` 전체를 설치하지 않고 다음
구성을 사용합니다.

```text
ROS 2 Humble 최소 패키지
  └─ 별도 ROS2 workspace
      └─ agx_arm_description (이 프로젝트가 제공하는 wrapper)
          └─ agx_arm_urdf (AgileX 공식 URDF)
```

AgileX 공식 `agx_arm_urdf` 문서도 전체 ROS 드라이버를 쓰지 않는 환경에서는
`agx_arm_description` wrapper를 만들고 URDF를 독립적으로 사용하는 방법을
지원합니다.

`agx_arm_ros` 전체에는 `agx_arm_ctrl`, MoveIt, ROS 2 Control, 메시지 패키지와
CAN 제어 경로가 포함됩니다. 현재 프로젝트는 이미 `piper_sdk`로 CAN을 제어하므로
전체 드라이버를 함께 실행하지 않습니다. 두 제어기가 같은 로봇/CAN 인터페이스에
동시에 명령을 보내면 안 됩니다.

다음 기능으로 전환할 때만 `agx_arm_ros` 전체 도입을 별도로 검토합니다.

- ROS topic을 주 제어 인터페이스로 사용
- MoveIt 또는 ROS 2 Control로 실제 로봇 제어
- AgileX의 `agx_arm_ctrl`을 `PiperMotorsBus` 대신 사용

## 2. 검증 기준

현재 PC에서 확인한 기준은 다음과 같습니다.

| 항목 | 값 |
|---|---|
| OS | Ubuntu 22.04 계열 |
| ROS | ROS 2 Humble |
| 설치 방식 | ROS 공식 APT 저장소의 binary package |
| URDF 저장소 | `https://github.com/agilexrobotics/agx_arm_urdf.git` |
| URDF 브랜치 | `main` |
| 검증 커밋 | `f6642ce0d7872c686f29c99e9e10cd23d1d49313` |
| ROS package | `agx_arm_description` |
| 사용 모델 | `piper_with_gripper_description.xacro` |

URDF의 `main` 최신 상태를 무조건 사용하는 대신 위 커밋을 먼저 사용해야 현재
검증 환경과 같은 모델을 얻을 수 있습니다. 업데이트가 필요하면 새 커밋에서 FK,
mesh, joint name과 replay를 다시 검증한 뒤 고정 커밋을 변경합니다.

## 3. ROS 2 Humble 설치

Ubuntu 22.04에서 ROS 공식 문서에 따라 ROS 2 APT 저장소를 먼저 등록합니다.

- ROS 2 Humble Ubuntu 설치:
  <https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html>
- ROS APT source 배포:
  <https://github.com/ros-infrastructure/ros-apt-source>

저장소 등록 후 이 프로젝트에 필요한 최소 패키지를 설치합니다.

```bash
sudo apt update
sudo apt install -y \
  ros-humble-ros-base \
  ros-humble-rviz2 \
  ros-humble-robot-state-publisher \
  ros-humble-xacro \
  python3-colcon-common-extensions
```

확인:

```bash
test -f /opt/ros/humble/setup.bash
source /opt/ros/humble/setup.bash
ros2 pkg prefix rviz2
ros2 pkg prefix robot_state_publisher
ros2 pkg prefix xacro
```

`ros-humble-desktop` 전체 설치는 필수가 아닙니다.

## 4. 경로 준비

아래 예시는 프로젝트들을 `~/UGRP` 아래에 배치합니다. 다른 위치를 사용해도 되지만
`configs/recording.env`에는 반드시 실제 절대경로를 넣습니다.

```bash
export PIPER_PROJECT_ROOT="$HOME/UGRP/lerobot_robot_piper"
export PIPER_ROS2_WS="$HOME/UGRP/ros2_ws"
export PIPER_URDF_DIR="$HOME/UGRP/agx_arm_urdf"

mkdir -p "$PIPER_ROS2_WS/src"
```

이 변수는 설치 명령을 짧게 쓰기 위한 현재 셸의 임시 변수입니다. 프로젝트 실행
시에는 7절의 `configs/recording.env` 값을 사용합니다.

## 5. URDF clone 및 커밋 고정

```bash
git clone https://github.com/agilexrobotics/agx_arm_urdf.git \
  "$PIPER_URDF_DIR"
git -C "$PIPER_URDF_DIR" checkout \
  f6642ce0d7872c686f29c99e9e10cd23d1d49313
```

확인:

```bash
git -C "$PIPER_URDF_DIR" rev-parse HEAD
```

출력이 다음 값과 같아야 합니다.

```text
f6642ce0d7872c686f29c99e9e10cd23d1d49313
```

## 6. `agx_arm_description` wrapper 구성

이 레포의 `ros2/agx_arm_description`은 현재 PC에서 검증한 wrapper 원본입니다.
새 ROS workspace에 복사한 뒤, 고정한 URDF 파일을 그 안에 넣습니다.

```bash
test ! -e "$PIPER_ROS2_WS/src/agx_arm_description"

cp -a \
  "$PIPER_PROJECT_ROOT/ros2/agx_arm_description" \
  "$PIPER_ROS2_WS/src/agx_arm_description"

mkdir -p \
  "$PIPER_ROS2_WS/src/agx_arm_description/agx_arm_urdf"

git -C "$PIPER_URDF_DIR" archive --format=tar \
  f6642ce0d7872c686f29c99e9e10cd23d1d49313 \
  | tar -x -C \
  "$PIPER_ROS2_WS/src/agx_arm_description/agx_arm_urdf"
```

wrapper는 다음 파일을 제공합니다.

| 파일 | 역할 |
|---|---|
| `package.xml` | ROS 2 package metadata와 실행 의존성 |
| `CMakeLists.txt` | URDF, launch, RViz config 설치 |
| `launch/display_piper.launch.py` | xacro, `robot_state_publisher`, RViz 실행 |
| `rviz/piper.rviz` | RobotModel, TF, `/preview_trajectory` 표시 설정 |

빌드:

```bash
source /opt/ros/humble/setup.bash
cd "$PIPER_ROS2_WS"
colcon build --packages-select agx_arm_description
source "$PIPER_ROS2_WS/install/setup.bash"
```

확인:

```bash
ros2 pkg prefix agx_arm_description
test -f \
  "$PIPER_ROS2_WS/install/agx_arm_description/share/agx_arm_description/agx_arm_urdf/piper/urdf/piper_with_gripper_description.xacro"
```

## 7. 프로젝트 환경변수

`configs/recording.env.example`을 복사한 후 새 PC의 절대경로를 입력합니다.

```bash
cp configs/recording.env.example configs/recording.env
```

예:

```dotenv
ROS_DISTRO_NAME=humble
ROS_SETUP_PATH=/opt/ros/humble/setup.bash
ROS2_WS=/home/USER/UGRP/ros2_ws
URDF_LOCAL_DIR=/home/USER/UGRP/agx_arm_urdf
URDF_REPO=https://github.com/agilexrobotics/agx_arm_urdf.git
```

`ROS2_WS`와 `URDF_LOCAL_DIR`을 비우면 프로젝트 상위 디렉터리의 `ros2_ws`와
`agx_arm_urdf`를 기본값으로 사용하지만, 다른 PC에서는 경로 혼동을 피하기 위해
명시적인 절대경로를 권장합니다.

## 8. 실행 및 검증

직접 실행:

```bash
source /opt/ros/humble/setup.bash
source "$PIPER_ROS2_WS/install/setup.bash"
ros2 launch agx_arm_description display_piper.launch.py
```

프로젝트 도구로 실행:

```bash
python scripts/tools/piper_session.py --step rviz
```

또는 통합 GUI의 `RViz Start` 버튼을 사용합니다.

확인할 항목:

- Piper와 gripper mesh가 누락 없이 표시됨
- RViz Fixed Frame이 `world`이고 TF 오류가 없음
- replay 또는 infer preview가 `/joint_states`를 publish하면 관절이 움직임
- trajectory preview 실행 시 `/preview_trajectory` marker가 표시됨

토픽 확인:

```bash
ros2 topic list
ros2 topic echo /joint_states --once
```

## 9. `agx_arm_ros`와 혼용하지 않는 규칙

`agx_arm_ros`를 별도 시험 목적으로 설치하는 것은 가능하지만 현재 LeRobot 실행과
동시에 다음 노드를 시작하지 않습니다.

- `agx_arm_ctrl`
- 실제 CAN 포트를 여는 AgileX ROS driver
- `/control/*`을 통해 실제 로봇에 명령을 전달하는 launch

공식 RViz의 `follow:=true, control:=false`처럼 표시 전용 구성은 제어 명령을
발행하지 않는다는 것을 확인한 경우에만 별도 시험할 수 있습니다. 현재 운영 경로는
이 문서의 독립 wrapper 하나로 통일합니다.

## 10. 참고 자료

- AgileX ROS 2 driver:
  <https://github.com/agilexrobotics/agx_arm_ros/tree/ros2>
- AgileX URDF:
  <https://github.com/agilexrobotics/agx_arm_urdf>
- `agx_arm_ros`의 URDF submodule 설정:
  <https://github.com/agilexrobotics/agx_arm_ros/blob/ros2/.gitmodules>
- ROS 2 Humble Ubuntu binary 설치:
  <https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html>
