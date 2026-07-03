# Operations

이 문서는 준비된 번호형 스크립트로 Piper leader/follower 실험을 실행하는 절차만 다룹니다. 환경 준비와 설정 방법은 [../setup_guide.md](../setup_guide.md), 세부 실험 품질 기준은 [data_collection_protocol.md](data_collection_protocol.md)를 봅니다.

## 1. 설치

레포 루트에서 플러그인을 editable install 합니다.

```bash
pip install -e .
```

설정 파일을 생성합니다.

```bash
cp configs/recording.env.example configs/recording.env
```

## 2. 설정

`configs/recording.env`에서 장비 환경에 맞게 값을 수정합니다.

| 설정 | 설명 |
|---|---|
| `LEADER_PORT` | leader arm CAN 포트 |
| `FOLLOWER_PORT` | follower arm CAN 포트 |
| `BITRATE` | CAN bitrate, 기본 `1000000` |
| `CAMERA_TYPE` | `opencv` 또는 `intelrealsense` |
| `TOP_CAM` | 상방 카메라 index 또는 RealSense serial/name |
| `WRIST_CAM` | 팔목 카메라 index 또는 RealSense serial/name |
| `DATASET_REPO_ID` | 저장할 LeRobotDataset repo id |
| `DATASET_ROOT` | 로컬 dataset 저장 경로 |
| `TASK` | episode task 문장 |

## 3. 실행 순서

CAN 초기화:

```bash
bash scripts/1__init_can.sh
```

카메라 확인:

```bash
bash scripts/2__find_camera.sh
```

RealSense serial과 화면을 직접 확인하려면:

```bash
python3 scripts/tools/realsense_view.py --list
python3 scripts/tools/realsense_view.py --serial 327122074262
```

카메라 값을 env 파일에 저장하려면:

```bash
bash scripts/3__set_camera.sh TOP_CAM WRIST_CAM
```

Teleoperation 점검:

```bash
bash scripts/4__teleoperate.sh
```

Dataset 녹화:

```bash
bash scripts/5__record.sh
```

녹화 데이터 replay:

```bash
bash scripts/6__replay.sh
```

Policy 학습:

```bash
bash scripts/7__train.sh
```

Async inference server/client:

```bash
bash scripts/8__run_server.sh
bash scripts/9__run_client.sh
```

## 4. Dry Run

실제 하드웨어 명령을 실행하지 않고 생성되는 명령만 확인할 수 있습니다.

```bash
DRY_RUN=true bash scripts/4__teleoperate.sh
DRY_RUN=true bash scripts/5__record.sh
DRY_RUN=true bash scripts/7__train.sh
```

## 5. Dataset 확인

녹화 후 feature와 episode parquet를 확인합니다.

```bash
python3 scripts/tools/wego_dataset_check.py \
  --dataset-repo-id local/piper_write_light \
  --dataset-root records/local/piper_write_light \
  --episode 0
```

## 6. 스크립트 구성

| 경로 | 역할 |
|---|---|
| `scripts/1__init_can.sh` | CAN 초기화 및 선택적 USB bus 기반 rename |
| `scripts/2__find_camera.sh` | 카메라 탐색 |
| `scripts/3__set_camera.sh` | env 파일의 카메라 값 갱신 |
| `scripts/4__teleoperate.sh` | leader/follower teleop |
| `scripts/5__record.sh` | LeRobotDataset 녹화 |
| `scripts/6__replay.sh` | 녹화 데이터 replay |
| `scripts/7__train.sh` | policy 학습 |
| `scripts/8__run_server.sh` | async policy server |
| `scripts/9__run_client.sh` | async robot client |
| `scripts/lib/run_common.sh` | 번호형 스크립트 공통 함수 |
| `scripts/tools/` | 수동 진단/점검 도구 |
