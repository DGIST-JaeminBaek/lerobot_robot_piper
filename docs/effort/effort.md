# Effort 및 Velocity 녹화

## 1. 추가한 이유

이 프로젝트의 작업은 지우개를 보드에 접촉시킨 상태로 수행된다. RGB 영상과 관절
위치만으로는 지우개가 보드에 닿았는지, 어느 정도 힘이 걸렸는지 직접 확인하기 어렵다.
같은 자세라도 보드를 가볍게 스치는 경우와 강하게 누르는 경우가 영상에서는 비슷하게
보일 수 있다.

Piper SDK는 각 모터의 전류를 기반으로 계산한 `effort` 값을 제공한다. 실제 힘/토크
센서의 측정값은 아니지만, 녹화해 두면 접촉 상태 분석과 학습 입력 후보로 사용할 수
있다. 녹화하지 않은 값은 나중에 복원할 수 없고, float 값 몇 개를 parquet에 추가하는
비용은 영상 저장 비용에 비해 작기 때문에 기본 녹화 대상으로 추가했다.

Effort에는 다음 성분이 함께 포함된다.

```text
측정 effort = 중력 + 마찰 + 관성 + 접촉에 의한 성분
```

따라서 effort만 보고 접촉력을 바로 판단할 수는 없다. 특히 마찰과 관성의 영향을
분석하려면 같은 시점의 관절 속도가 필요하므로 `velocity`도 함께 녹화한다.

- Position: 관절 6개와 그리퍼, 총 7개
- Effort: 관절 6개와 그리퍼, 총 7개
- Velocity: 관절 6개, 총 6개

Piper SDK가 그리퍼 속도를 제공하지 않기 때문에 그리퍼 velocity는 저장하지 않는다.

## 2. 저장 형식

Effort와 velocity는 영상이 아니라 LeRobotDataset의 parquet 데이터에 저장된다.

```text
dataset_root/
├── meta/info.json
├── data/chunk-000/file-000.parquet
└── videos/
```

Piper plugin이 position, effort, velocity를 각각 float observation feature로
선언하면 LeRobot의 기존 feature 변환 로직이 이를 하나의 `observation.state`에
합친다. 외부 LeRobot core는 effort 녹화를 위해 수정하지 않았다.

`USE_EFFORT=true`일 때 state 구성은 다음과 같다.

```text
observation.state: 20차원

0  ~ 6  : joint1.pos    ~ joint6.pos, gripper.pos
7  ~ 13 : joint1.effort ~ joint6.effort, gripper.effort
14 ~ 19 : joint1.vel    ~ joint6.vel
```

정확한 순서는 데이터셋의 다음 메타데이터에 기록된다.

```text
meta/info.json
└── features
    └── observation.state
        └── names
```

Action은 기존과 동일하게 7개 position 값으로만 구성되며 effort와 velocity는
포함하지 않는다.

## 3. 구현 방법

### `lerobot_robot_piper/motors/piper_motors_bus.py`

Piper SDK 메시지에서 effort와 velocity를 읽는 메서드를 추가했다.

```python
effort = motor.effort * 0.001
velocity = motor.motor_speed * 0.001
```

- `get_effort()`는 관절 6개와 그리퍼의 값을 N·m 단위로 반환한다.
- `get_velocity()`는 관절 6개의 값을 rad/s 단위로 반환한다.
- Effort는 전류 기반 추정값이며 실제 토크 센서 측정값이 아니다.

### `lerobot_robot_piper/config_piper.py`

Effort 및 velocity observation을 활성화하는 설정을 추가했다.

```python
use_effort: bool = True
```

별도의 velocity 플래그는 만들지 않았다. Effort를 분석할 때 같은 시점의 velocity가
필요하므로 두 값은 `use_effort` 하나로 함께 활성화한다.

### `lerobot_robot_piper/piper_follower.py`

`use_effort=true`일 때 observation feature에 effort와 velocity 이름을 추가하고,
`get_observation()`에서 모터 값을 읽어 observation dictionary에 넣도록 수정했다.

```text
joint1.effort ... gripper.effort
joint1.vel    ... joint6.vel
```

LeRobot은 이 float feature들을 position과 함께 `observation.state`로 변환한다.

### `scripts/lib/run_common.sh`

셸 실행 경로에서도 설정이 빠지지 않도록 다음 인자를 생성하는
`robot_observation_args()`를 추가했다.

```text
--robot.use_effort=true
```

### `scripts/5__record.sh`

`robot_observation_args()`가 만든 인자를 `lerobot-record` 명령에 전달한다. 따라서
GUI가 아닌 셸 스크립트로 녹화할 때도 effort와 velocity가 저장된다.

### `scripts/9__run_client.sh`

Effort를 포함한 20차원 state로 학습한 정책은 추론할 때도 같은 observation 구성이
필요하다. 비동기 inference client에도 `--robot.use_effort`를 전달하도록 수정했다.

### `lerobot_robot_piper/teleop_ui.py`

GUI에 `Record Effort` 체크박스를 추가하고 Record 및 Infer 명령에
`--robot.use_effort`가 반영되도록 했다. 체크박스 변경 시 화면의 실행 명령도 즉시
다시 생성된다.

### `scripts/tools/smooth_start_frames.py`

Effort가 추가되면 `observation.state`에 position 이외의 값도 함께 들어간다. 기존
Smooth Start는 feature 이름에서 관절 이름만 추출해 모든 state 값을 parking position
쪽으로 보간했기 때문에 effort와 velocity까지 position 값으로 덮어쓸 수 있었다.

이를 방지하기 위해 `.pos`로 끝나는 항목만 보간하고 `.effort`, `.vel` 항목은 원본을
그대로 유지하도록 마스크를 추가했다.

## 4. 사용 방법

`configs/recording.env`에서 설정한다.

```bash
USE_EFFORT=true
```

기본값도 `true`다. 기존 7차원 state 데이터나 이를 사용해 학습한 체크포인트와
호환해야 할 때만 명시적으로 끈다.

```bash
USE_EFFORT=false
```

한 데이터셋을 수집하는 도중에는 값을 변경하면 안 된다. 설정을 바꾸면
`observation.state`가 20차원과 7차원으로 달라져 데이터셋 schema가 일치하지 않는다.
학습 데이터에 effort와 velocity가 포함되어 있다면 추론할 때도 반드시 같은 설정을
사용해야 한다.

## 5. 확인 방법

녹화된 데이터셋은 다음 도구로 확인한다.

```bash
python scripts/tools/check_effort.py <데이터셋 경로>
```

정상 데이터의 확인 항목은 다음과 같다.

- `observation.state`가 20차원이다.
- position 7개, effort 7개, velocity 6개의 이름이 존재한다.
- effort와 velocity가 전부 0으로 고정되지 않고 실제로 변한다.
- Smooth Start가 적용된 초반 프레임에서도 effort와 velocity가 position 값으로
  덮어써지지 않는다.

관련 mock 테스트:

```bash
python scripts/tools/test_effort_mock.py
python scripts/tools/test_dataset_features_mock.py
python scripts/tools/test_smooth_start_mock.py
```

## 6. 수정 파일

Effort 및 velocity 녹화를 위해 수정한 파일:

- `lerobot_robot_piper/motors/piper_motors_bus.py`
- `lerobot_robot_piper/config_piper.py`
- `lerobot_robot_piper/piper_follower.py`
- `lerobot_robot_piper/teleop_ui.py`
- `scripts/lib/run_common.sh`
- `scripts/5__record.sh`
- `scripts/9__run_client.sh`
- `scripts/tools/smooth_start_frames.py`

추가한 확인 도구와 테스트:

- `scripts/tools/check_effort.py`
- `scripts/tools/test_effort_mock.py`
- `scripts/tools/test_dataset_features_mock.py`
- `scripts/tools/test_smooth_start_mock.py`

## 7. 안전 기능과의 구분

Effort 및 velocity **녹화**와 effort 임계값을 이용한 **안전 제어**는 별개의 기능이다.
`USE_EFFORT=false`로 녹화를 끄는 것과 안전 제어의 활성화 여부는 서로 독립적이다.

이 문서는 observation에 effort와 velocity를 추가해 데이터셋에 저장하는 기능만
설명한다. 과부하 감지 후 로봇을 어떻게 후퇴·parking·종료시킬지는 별도의 안전 제어
설계로 다뤄야 한다.

Effort 활용 연구는 [`effort_research.md`](./effort_research.md)를 참고한다.
