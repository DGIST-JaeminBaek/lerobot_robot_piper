# RealSense Depth 녹화 백포트

## 1. 목적

이 문서는 Python 3.10과 LeRobot 0.4.x를 유지하면서 RealSense의 `uint16`
depth 스트림을 `LeRobotDataset`에 저장하고 다시 읽는 구현을 설명한다.

이 프로젝트는 ROS2 Humble의 `rclpy` 때문에 Python 3.10을 사용하며,
LeRobot 0.5.0 이상은 Python 3.12를 요구한다. 따라서 LeRobot 0.6.0에서
추가된 공식 depth 양자화·인코딩 방식을 LeRobot 0.4.0의 기존 저장 구조에
맞게 백포트했다.

적용 대상은 다음 두 저장소다.

- LeRobot clone: `/home/ugrp43/UGRP/lerobot`
- Piper plugin: `/home/ugrp43/UGRP/lerobot_robot_piper`

원래 백포트는 LeRobot `v0.4.0` 기준으로 만들었으나, 2026-07-24에 `v0.4.4`로
업그레이드하고 이 패치들을 재적용했다(8번 항목 참고). 아래 본문에서 "0.4.0
저장 구조/아키텍처"라고 설명하는 부분은 이 백포트가 원래 맞춰 설계된 기준
아키텍처를 뜻하며, `v0.4.4`도 같은 PNG → batch video 구조를 그대로 유지하고
있어 그 설명은 여전히 유효하다. 현재 실제로 쓰는 버전은 `v0.4.4`다.

`docs/depth/modified_files`에는 실제 검증에 사용한 수정본을 원래 디렉터리
구조대로 보관한다. 이 파일들은 문서용 snapshot이며 런타임에서 직접 import하지
않는다.

## 2. 저장 파이프라인

```text
RealSense z16 depth
    uint16, mm, (H, W)
          |
          v
Piper observation
    uint16, (H, W, 1)
          |
          v
12-bit 로그 양자화
    uint16 code, 0..4095, (H, W)
          |
          v
임시 16-bit PNG
    PIL I;16
          |
          v
HEVC + gray12le + x265 lossless
    .mp4
          |
          v
PyAV gray12le 디코딩
          |
          v
역양자화
    float32, mm, (1, H, W)
```

LeRobot 0.4.0은 녹화 중 각 프레임을 PNG로 저장하고 에피소드가 끝난 뒤
비디오로 일괄 인코딩한다. 이 구조는 유지하고 depth 프레임에만 별도의
양자화·HEVC 설정을 적용했다.

기본 양자화 설정은 다음과 같다.

| 항목 | 값 |
|---|---:|
| 양자화 bit | 12 |
| code 범위 | 0–4095 |
| 최소 거리 | 0.0 m |
| 최대 거리 | 10.0 m |
| log shift | 3.5 m |
| 로그 양자화 | 사용 |
| pixel format | `gray12le` |
| codec | HEVC |
| codec option | `x265-params=lossless=1` |

여기서 무손실은 **양자화된 12-bit code가 인코딩 전후 동일하다**는 의미다.
16-bit 원본을 12-bit로 줄이는 양자화 자체에는 작은 오차가 있다. 합성
10 mm–10 m 데이터에서 확인한 최대 오차는 2 mm였다.

## 3. Feature 메타데이터

Depth feature는 일반 RGB video와 구분하기 위해 `features[*].info`에 다음
정보를 저장한다.

```json
{
  "is_depth_map": true,
  "video.is_depth_map": true,
  "video.codec": "hevc",
  "video.pix_fmt": "gray12le",
  "video.depth_min": 0.0,
  "video.depth_max": 10.0,
  "video.shift": 3.5,
  "video.use_log": true,
  "video.extra_options": {
    "x265-params": "lossless=1"
  }
}
```

읽을 때 `depth_min`, `depth_max`, `shift`, `use_log`가 녹화 시점과 반드시
같아야 한다. `DepthFeature`가 shape `(H, W, 1)`과 이 메타데이터를 함께
전달한다.

LeRobot 0.6.0 기준 구현의 기본 최소 거리는 0.01 m이지만, 이 프로젝트는
RealSense가 invalid pixel을 0 mm로 반환하는 의미를 보존하기 위해 새
녹화의 기본값을 0.0 m로 변경했다. 로그 양자화 전에 3.5 m shift를 더하므로
0 m도 안전하게 처리되며 `0 mm → code 0 → 0 mm`로 복원된다.

공식 구현에서도 `depth_min`은 센서 고정 사양이 아니라 사용자가 조정할 수
있는 양자화 파라미터다. 공식 RealSense 관련 변경은 카메라 출력에서 0을
invalid-pixel sentinel로 보존한다. 공식 토론에서 기본값 0.01 m와 이 sentinel
사이의 충돌을 직접 논의한 기록은 확인되지 않았으므로, 0.0 m는 D435if의
실측 가능 범위를 넓히려는 값이 아니라 sentinel 보존을 위한 프로젝트별
기본값이다.

기존에 `video.depth_min: 0.01`로 녹화된 데이터셋은 파일에 저장된 값을
사용해 이전 방식대로 읽는다. 따라서 코드 기본값 변경으로 기존
데이터셋의 해석이 바뀌지는 않는다.

## 4. LeRobot에서 수정한 파일

수정본 위치:
`docs/depth/modified_files/lerobot/src/lerobot/datasets`

### `image_writer.py`

- `(H, W)`, `(1, H, W)`, `(H, W, 1)` 단일채널 입력 지원
- `uint16`을 `PIL I;16` 모드로 보존
- PNG와 TIFF별 저장 옵션 분리
- 기존 RGB 변환 경로 유지

Snapshot:
[`modified_files/lerobot/src/lerobot/datasets/image_writer.py`](modified_files/lerobot/src/lerobot/datasets/image_writer.py)

### `depth_utils.py` — 신규

- `DepthFeature` feature spec
- `quantize_depth()`
- `dequantize_depth()`
- m/mm 입력 단위 처리
- 12-bit 로그 및 선형 양자화
- Python 3.10 호환 NumPy/Torch 경로

0.6.0의 PyAV streaming 의존성은 가져오지 않고 0.4.0 batch 구조에 맞는
NumPy 반환 경로만 사용한다.

Snapshot:
[`modified_files/lerobot/src/lerobot/datasets/depth_utils.py`](modified_files/lerobot/src/lerobot/datasets/depth_utils.py)

### `video_utils.py`

- `encode_video_frames()`에 `extra_options` 추가
- `gray*` 입력의 RGB 강제 변환 제거
- `uint16 gray12le` PyAV 프레임 생성
- 8-bit 정규화를 하지 않는 `decode_depth_video_frames()` 추가
- HEVC seek 시 1초 pre-roll을 적용해 frame reordering 처리

Snapshot:
[`modified_files/lerobot/src/lerobot/datasets/video_utils.py`](modified_files/lerobot/src/lerobot/datasets/video_utils.py)

### `lerobot_dataset.py`

- depth feature 판별
- `add_frame()`에서 depth를 12-bit code로 양자화
- depth video에 HEVC/`gray12le`/lossless 옵션 적용
- 읽기 시 raw 12-bit code를 디코딩하고 mm depth로 역양자화
- 실제 codec 정보와 기존 depth 메타데이터 병합

Snapshot:
[`modified_files/lerobot/src/lerobot/datasets/lerobot_dataset.py`](modified_files/lerobot/src/lerobot/datasets/lerobot_dataset.py)

### `compute_stats.py`

- depth PNG를 RGB/`uint8`로 변환하지 않고 단일채널 `uint16`으로 로드
- code를 mm depth로 역양자화한 뒤 통계 계산
- depth 통계 shape `(1, 1, 1)` 지원
- 기존 RGB 통계 shape `(3, 1, 1)` 유지

Snapshot:
[`modified_files/lerobot/src/lerobot/datasets/compute_stats.py`](modified_files/lerobot/src/lerobot/datasets/compute_stats.py)

### `utils.py`

이 파일은 최초 체크리스트에는 없었지만 0.4.0 아키텍처상 반드시 필요했다.

- `DepthFeature`가 가진 `info`를 dataset feature로 전달
- `load_image_as_numpy()`에 grayscale 보존 옵션 추가
- 기존 RGB 호출은 기본적으로 이전 동작 유지

Snapshot:
[`modified_files/lerobot/src/lerobot/datasets/utils.py`](modified_files/lerobot/src/lerobot/datasets/utils.py)

### `test_depth_recording.py` — 신규

- 합성 depth 생성
- quantize → PNG → HEVC → decode → dequantize
- 인코딩 전후 12-bit code의 전체 픽셀 비교

Snapshot:
[`modified_files/lerobot/tests/datasets/test_depth_recording.py`](modified_files/lerobot/tests/datasets/test_depth_recording.py)

## 5. Piper plugin에서 수정한 파일

### `piper_follower.py`

- `use_depth=True`인 카메라에 `<camera>_depth` feature 추가
- `DepthFeature(H, W)`로 메타데이터 전달
- `cam.read_depth()` 결과를 `(H, W, 1)`로 observation에 추가
- RGB 단계에서는 TOP/WRIST를 병렬로 읽고, 이어지는 depth 단계에서도
  depth 사용 카메라들을 병렬로 읽음

Snapshot:
[`modified_files/lerobot_robot_piper/lerobot_robot_piper/piper_follower.py`](modified_files/lerobot_robot_piper/lerobot_robot_piper/piper_follower.py)

현재 LeRobot 0.4.0 RealSense API에는 RGB-D 한 쌍을 한 번에 반환하는
비동기 API가 없다. 따라서 Piper 경로의 RGB와 depth는 같은 파이프라인을
사용하지만 정확히 같은 RealSense frameset이라고 보장하지 않는다.

## 6. 수정본 디렉터리 구조

```text
docs/depth/
├── README.md
├── modified_files/
│   ├── lerobot/
│   │   ├── src/lerobot/datasets/
│   │   │   ├── compute_stats.py
│   │   │   ├── depth_utils.py
│   │   │   ├── image_writer.py
│   │   │   ├── lerobot_dataset.py
│   │   │   ├── utils.py
│   │   │   └── video_utils.py
│   │   ├── src/lerobot/utils/visualization_utils.py
│   │   ├── src/lerobot/scripts/lerobot_record.py
│   │   └── tests/datasets/test_depth_recording.py
│   └── lerobot_robot_piper/
│       └── lerobot_robot_piper/piper_follower.py
└── tools/
    ├── depth_video_viewer.py
    └── realsense_depth_record_test.py
```

## 7. 다른 LeRobot 0.4.x clone에 적용

먼저 대상 파일을 반드시 백업한다. 아래 snapshot은 이 프로젝트에서 실제 사용 중인
LeRobot `v0.4.4` 구조를 기준으로 하므로(2026-07-24, v0.4.0에서 업그레이드함)
다른 0.4.x 버전에는 diff를 검토한 뒤 적용해야 한다.

```bash
DEPTH_DOC=/home/ugrp43/UGRP/lerobot_robot_piper/docs/depth
LEROBOT=/home/ugrp43/UGRP/lerobot

cp "$DEPTH_DOC"/modified_files/lerobot/src/lerobot/datasets/*.py \
   "$LEROBOT"/src/lerobot/datasets/

cp "$DEPTH_DOC"/modified_files/lerobot/src/lerobot/utils/visualization_utils.py \
   "$LEROBOT"/src/lerobot/utils/visualization_utils.py

cp "$DEPTH_DOC"/modified_files/lerobot/src/lerobot/scripts/lerobot_record.py \
   "$LEROBOT"/src/lerobot/scripts/lerobot_record.py

cp "$DEPTH_DOC"/modified_files/lerobot/tests/datasets/test_depth_recording.py \
   "$LEROBOT"/tests/datasets/
```

Piper plugin 연결 코드:

```bash
cp "$DEPTH_DOC"/modified_files/lerobot_robot_piper/lerobot_robot_piper/piper_follower.py \
   /home/ugrp43/UGRP/lerobot_robot_piper/lerobot_robot_piper/
```

이 snapshot은 `pyproject.toml`을 포함하지 않는다. 현재 LeRobot clone에 이미
적용된 `opencv-python-headless` → `opencv-python` GUI preview 패치는 그대로
유지해야 한다.

## 8. v0.4.4 업그레이드와 추가 수정 (2026-07-24)

`git stash`(로컬 패치 전부, untracked 포함) → `git checkout v0.4.4` → `git stash
pop`(3-way merge) 순서로 진행했다. 충돌은 4개 파일(`lerobot_dataset.py`,
`video_utils.py`, `lerobot_record.py`, `visualization_utils.py`)에서 났고
전부 수동으로 병합했다. `compute_stats.py`/`image_writer.py`는 v0.4.0/v0.4.4
사이에 거의 안 바뀌어서 충돌 없이 그대로 적용됐다.

### parallel_encoding 기본값(True)이 depth를 깨뜨리는 문제

v0.4.4는 `save_episode()`에 `parallel_encoding`(기본 `True`) 기능을 새로
추가했는데, 카메라가 2개 이상이면 `ProcessPoolExecutor`로 모든 video_key를
`_encode_video_worker`(범용 워커, `self.vcodec` 하나만 적용)에 병렬 제출한다.
depth는 자기만의 codec/pix_fmt/extra_options(hevc/gray12le/lossless)가
필요해서 이 범용 워커를 그대로 타면 안 되는데, 기본값이 `True`라 아무 설정도
안 건드리면 depth 녹화가 바로 깨질 뻔했다.

수정: `depth_keys`/`rgb_keys`를 분리하고, RGB는 기존 `_encode_video_worker`를,
depth는 새로 만든 `_encode_depth_video_worker`(feature의 `info`에서
codec/pix_fmt/extra_options를 읽어 적용)를 **같은 `ProcessPoolExecutor` 풀에
같이 제출**해서 전부 동시에 인코딩되게 했다. 인코더(x265/SVT-AV1)는 프로세스당
CPU 코어 수만큼 스레드를 잡으려 하므로, `self._encoder_threads`를 명시적으로
안 정했으면 `(총 코어 수) / (동시 인코딩 개수)`로 나눠서 프로세스당 스레드
수를 캡 씌워 오버서브스크립션을 막는다.

실측(1280×720, RTX 5090 워크스테이션, 16코어): 순차 처리(RGB 병렬 17s + depth
순차 2개 21s+16s ≈ 54s) 대비 통합 병렬 처리는 51.3s로 **개선폭이 예상(약 21s)
보다 훨씬 작았다** — 스레드를 4등분(16÷4)하면서 각 인코딩 자체가 그만큼
느려져서, "동시에 더 많이" 얻은 이득을 상당 부분 상쇄했다. 이건 우리 코드가
아니라 x265/SVT-AV1 같은 인코더의 스레드 스케일링(오버헤드, 캐시 경합 등)
알고리즘 특성이라 근본적으로 바꾸기 어렵다.

### lerobot_record.py: 기록되는 action이 leader raw 값이었던 문제

`lerobot_record.py`는 원래 `robot.send_action()`이 리턴하는 "실제로 로봇에
보낸 값"(offset 보정 완료)을 무시하고, 그 이전 단계의 `action_values`(teleop
raw 값)를 데이터셋 `action` 컬럼에 저장하고 있었다(코드에 이를 인정하는
`TODO(steven, pepijn, adil)` 주석도 있었음). `piper_follower.py`의
`action_offset`(leader/follower 좌표계 보정)이 `send_action()` 내부에서만
적용되고 기록은 안 됐던 것 — `so100_follower.send_action()` 등 다른 lerobot
로봇 클래스 docstring도 전부 "returns the action actually sent to the motors"
라고 명시하므로, `action_values` → `sent_action`(`send_action()`의 리턴값)으로
교체했다. 이 수정 이후 녹화되는 데이터셋의 action은 leader raw가 아니라
follower에 실제로 내려간 보정된 절대 목표값이다 — 이 수정 이전에 녹화된
데이터셋과 의미가 다르니 섞어서 학습하지 않도록 주의.

이에 맞춰 `scripts/6__replay.sh`도 `--robot.use_action_offset=false`로
바꿨다 — action이 이미 offset 보정 완료된 값이라 replay 때 또 보정하면 이중
보정이 된다.

### piper_follower.py: 카메라 read/connect 병렬화

`get_observation()`이 RGB 전부 읽은 뒤에야 depth를 읽기 시작하는 구조라서
(순차 두 단계) 실측 25.5ms+33.3ms ≈ 59ms/프레임(목표 30Hz의 절반, 15Hz)이
걸렸다. 카메라당 RGB+depth 요청을 한 번에 다 같이 제출하도록 합치고,
`ThreadPoolExecutor`의 `max_workers`도 카메라 수만큼이 아니라 (카메라 수 +
depth 사용 카메라 수)로 늘려서 실제로 다 같이 병렬로 돌게 했다.

`connect()`도 마찬가지로 RealSense 카메라별 `warmup_s`(기본 10초)가 순차로
곱해져서 카메라 2대면 20초 이상 걸렸다. 예전에 병렬 연결을 시도했다가 실제
하드웨어에서 "read failed"/타임아웃이 나서 순차로 되돌렸었는데, 그 원인이
USB 대역폭 경합이 아니라 당시 CPU 쿨링 문제였을 가능성이 제기돼
`scripts/tools/camera_parallel_connect_test.py`(로봇 없이 카메라만 테스트)로
재검증 — 3회 연속 성공(~10.3~10.4s, 카메라 1대 warmup_s와 거의 동일해서 실제로
겹쳐서 도는 것도 확인됨), depth 프레임도 정상이라 병렬 연결로 되돌렸다.

추가로 `RealSenseCamera.read_depth()`가 v0.4.4에서 `timeout_ms` 파라미터를
deprecated 처리했는데(내부적으로 값을 안 쓰고 경고만 찍음) 우리가 인자 없이
호출해서 기본값(200, truthy)이 매 프레임 경고를 찍었다 — `timeout_ms=0`을
명시해서 제거했다.

### 진단용 환경변수 `PIPER_LOG_TIMING=1`

`piper_follower.py`/`lerobot_dataset.py` 양쪽에 `PIPER_LOG_TIMING=1`일 때만
`logger.debug`/`print`로 프레임별·단계별(카메라 read, CAN sync_read/set_action,
`_wait_image_writer`, `compute_episode_stats`, 영상 인코딩, `meta.save_episode`)
소요시간을 찍는 코드를 넣어뒀다. 평소엔 이 환경변수 없이 실행하면 조용하다.

## 9. 사용 설정

`configs/recording.env`:

```dotenv
CAMERA_TYPE=intelrealsense
TOP_CAM=327122074262
WRIST_CAM=243322071626
REALSENSE_USE_DEPTH=true
TOP_REALSENSE_USE_DEPTH=true
WRIST_REALSENSE_USE_DEPTH=true
CAM_WIDTH=1280
CAM_HEIGHT=720
FPS=30
```

일반 Piper Record:

```bash
conda activate ugrp
cd /home/ugrp43/UGRP/lerobot_robot_piper
bash scripts/5__record.sh
```

## 10. 로봇 없는 카메라 단독 검증

실행 파일:
[`tools/realsense_depth_record_test.py`](tools/realsense_depth_record_test.py)

```bash
conda activate ugrp
cd /home/ugrp43/UGRP/lerobot_robot_piper

python scripts/tools/realsense_depth_record_test.py \
  --seconds 5 \
  --depth-only \
  --preview
```

이 테스트는 다음을 자동 검증한다.

1. TOP/WRIST RealSense 연결
2. z16 depth 수집
3. LeRobotDataset 저장
4. HEVC `gray12le` 전 프레임 디코딩
5. 저장 전후 12-bit code SHA-256 비교
6. `info.json` 메타데이터 검사
7. LeRobotDataset 재로드

x265가 VNC를 점유하지 않도록 테스트 도구는 기본 4 thread를 사용한다.
조정하려면 `--encoder-threads`를 사용한다.

## 11. 녹화 중 실시간 모니터링(rerun)

`lerobot-record --display_data=true`(teleop_ui.py Record/Infer 프리셋 포함)가
쓰는 `lerobot/utils/visualization_utils.py`의 `log_rerun_data()`는 원래 모든
카메라 observation을 `rr.Image(arr)`로 그대로 로깅했다. depth observation은
아직 양자화 이전의 raw `uint16` mm 배열(`(H, W, 1)`)이라, RGB 픽셀처럼 그리면
uint16 전체 범위(0–65535) 기준으로 정규화돼 실제 값(수백~수천 mm대)이 전부
새까맣게 보인다. 특히 그리퍼에 가까운 WRIST 카메라는 depth 값 자체가 작아서
거의 아무것도 안 보이는 것처럼 나타난다(TOP은 상대적으로 값이 커서 조금 덜
두드러짐).

`visualization_utils.py`를 수정해 1채널(depth) 배열만 `rr.DepthImage()`로
분기해서 로깅하도록 했다. `rr.DepthImage`는 프레임마다 자기 자신의 min/max로
자동 정규화 + 컬러맵을 적용하므로 근접 거리 값도 정상적으로 색이 들어간
이미지로 보인다. 이 변경은 depth 저장 파이프라인과 무관한, 순수 미리보기
전용 수정이다(저장되는 데이터는 그대로).

```python
elif arr.ndim >= 2 and arr.shape[-1] == 1:
    rr.log(key, rr.DepthImage(arr.squeeze(-1)), static=True)
else:
    rr.log(key, rr.Image(arr), static=True)
```

수정 위치: `/home/ugrp43/UGRP/lerobot/src/lerobot/utils/visualization_utils.py`
(로컬 lerobot clone). depth 저장 파이프라인(`datasets/*.py`)과는 디렉터리가
다르지만, 다른 clone에 백포트를 재적용할 때 이 미리보기 수정도 같이
따라가도록 `docs/depth/modified_files/lerobot/src/lerobot/utils/visualization_utils.py`에도
동일하게 snapshot을 보관한다(6번, 7번 항목 참고).

## 12. Replay에서 RGB/Depth 보기

`piper_replay_player.py`(RViz 없이)와 `piper_replay_player_rviz.py`(RViz 동기화
재생) 둘 다 `info.json`의 `dtype=video`인 feature를 전부 자동으로 찾아서 화면에
쌓는다. depth 녹화 이후로는 카메라마다 RGB + Depth 두 스트림이 함께 잡혀서
창에 4개가 한꺼번에 쌓이게 됐고, depth 스트림도 다른 일반 플레이어처럼
`bgr24`로 그대로 디코딩되면 12-bit 양자화 code를 색상 픽셀인 것처럼 잘못
표시했다.

두 스크립트 모두 `--view {both,rgb,depth}` 옵션을 추가했다(기본 `both` = 기존
동작 그대로). `rgb`/`depth`를 주면 `info.json`의 `is_depth_map` 메타데이터로
스트림을 필터링하고, depth 스트림은 `gray12le` code를 그대로 디코딩한 뒤
`dequantize_depth()`로 mm 값을 복원해서 `docs/depth/tools/depth_video_viewer.py`와
동일한 방식(고정 범위 100–3000mm, `COLORMAP_TURBO`, invalid(≤100mm)는 검정)으로
컬러맵을 입힌다. 이렇게 만든 BGR 프레임은 RGB 프레임과 배열 형태가 같아서
나머지 화면 구성 코드(패널, 리사이즈, vconcat/hconcat)는 그대로 재사용된다.

`teleop_ui.py`의 Dataset Browser에도 Play 버튼 옆에 "View" 콤보박스(both/rgb/
depth)를 추가해서, `--video-key`를 직접 지정하지 않아도 GUI에서 바로 필터를
고를 수 있게 했다.

```bash
# RGB만
python scripts/tools/piper_replay_player.py --dataset-root <root> --episode 0 --view rgb
# Depth만
python scripts/tools/piper_replay_player.py --dataset-root <root> --episode 0 --view depth
```

## 13. Depth 영상 단독 보기 (동기화 재생 없이)

일반 플레이어는 원래 `gray12le`의 양자화 code를 거리 영상으로 표시하지
못했다(12번 항목에서 두 replay 플레이어에는 이미 해결됨). 이 도구는 joint
데이터 동기화나 패널 없이, depth MP4만 빠르게 훑어보고 싶을 때 쓰는 더 가벼운
별도 뷰어다. 원본 MP4를 변경하거나 별도 영상을 만들지 않고 화면에서만 mm
depth를 컬러맵으로 변환한다.

실행 파일:
[`tools/depth_video_viewer.py`](tools/depth_video_viewer.py)

```bash
python scripts/tools/depth_video_viewer.py \
  records/local/realsense_depth_test_0724-122313 \
  --camera both
```

- `Space`: 일시정지/재생
- `q` 또는 `Esc`: 종료
- `--min-mm`, `--max-mm`: 컬러맵 표시 범위

## 14. 검증 결과

실물 RealSense D435IF 두 대, 1280×720, 30 FPS, 5초 녹화 결과:

| 항목 | 결과 |
|---|---|
| TOP depth | 150/150프레임 code 전체 픽셀 일치 |
| WRIST depth | 150/150프레임 code 전체 픽셀 일치 |
| codec | HEVC |
| pixel format | `gray12le` |
| dataset reload | 성공 |
| TOP 유효 depth 비율 | 87.67% |
| WRIST 유효 depth 비율 | 64.62% |

합성 데이터와 실제 카메라 데이터 모두 codec 전후 12-bit code가 완전히
동일했다.

## 15. 알려진 제한사항

- 에피소드 종료 후 CPU `libx265`로 영상을 일괄 인코딩한다.
- VLC 등 일반 범용 비디오 플레이어에서는 depth MP4를 직접 열어도 12-bit
  code가 raw grayscale로만 보인다(mm 컬러맵 없음). `piper_replay_player.py`/
  `piper_replay_player_rviz.py`(`--view depth`)와 `depth_video_viewer.py`,
  녹화 중 rerun 미리보기는 컬러맵을 적용해서 정상적으로 보여준다(10, 11, 12번
  항목 참고).
- NVIDIA NVENC는 현재 환경에서 `gray12le` 단일채널 12-bit 입력을 지원하지
  않아 CPU 인코더를 사용한다.
- 새 녹화에서는 센서 invalid 값 0 mm가 양자화 code 0을 거쳐 다시 0 mm로
  복원된다. 기존 `depth_min=0.01` 데이터의 code 0에는 0–10 mm 구간의
  정보가 이미 합쳐져 있어 원래 값을 사후에 구분할 수 없다.
- 10 m보다 먼 값은 최대 code로 clamp된다.
- 정확히 동기화된 RGB-D frameset이나 depth-to-color alignment가 필요하면
  RealSense 카메라 계층에 별도의 paired read API를 추가해야 한다.

## 16. 참고 자료

### LeRobot 이슈와 Pull Request

- [LeRobot issue #1144 — Depth Images](https://github.com/huggingface/lerobot/issues/1144)
  - 16-bit PNG, RGB channel 분할, 일반 H.264/HEVC 적용 시도와 정밀도 손실
    사례를 확인하는 데 참고했다.
- [LeRobot PR #2604 — 공식 depth image/video 지원](https://github.com/huggingface/lerobot/pull/2604)
  - `DepthEncoderConfig`, 12-bit 로그 양자화, HEVC lossless 저장 방식의 기준
    구현이다.
- [LeRobot PR #3023 — Python 3.12 전환](https://github.com/huggingface/lerobot/pull/3023)
  - LeRobot 0.5.0 이상을 이 프로젝트에서 직접 사용할 수 없는 Python 버전
    제약을 확인하는 데 참고했다.
- [LeRobot PR #3253](https://github.com/huggingface/lerobot/pull/3253)
  - depth video 인코딩 설계 토론에서 log quantization과 최소·최대 거리는
    사용 사례에 따라 달라질 수 있는 파라미터라고 설명한다.
- [LeRobot PR #3644 — v0.6.0에 병합된 depth 지원](https://github.com/huggingface/lerobot/pull/3644)
  - 최종 depth/RealSense 통합, 조정 가능한 `depth_min`/`depth_max`, 12-bit
    HEVC 구현과 메타데이터 저장 방식을 확인했다.
- [LeRobot PR #3447 — RealSense depth 단위 및 invalid sentinel](https://github.com/huggingface/lerobot/pull/3447)
  - D4xx depth를 mm로 변환하면서 0을 invalid-pixel sentinel로 보존해야
    한다는 RealSense 처리 방침을 확인했다.
- [LeRobot PR #1744](https://github.com/huggingface/lerobot/pull/1744)
  - video 인코딩 파라미터 제어 관련 변경을 추적하기 위한 참고 링크다.

### LeRobot v0.6.0 공식 원본 소스

아래 파일은 그대로 복사하지 않고, LeRobot 0.4.0의 PNG → batch video 구조에
맞게 필요한 로직만 백포트했다.

- [`configs/video.py`](https://raw.githubusercontent.com/huggingface/lerobot/v0.6.0/src/lerobot/configs/video.py)
  - depth 기본값, `DepthEncoderConfig`, `gray12le`, HEVC lossless 옵션
- [`datasets/depth_utils.py`](https://raw.githubusercontent.com/huggingface/lerobot/v0.6.0/src/lerobot/datasets/depth_utils.py)
  - `quantize_depth()`와 `dequantize_depth()` 수식 및 단위 처리
- [`datasets/pyav_utils.py`](https://raw.githubusercontent.com/huggingface/lerobot/v0.6.0/src/lerobot/datasets/pyav_utils.py)
  - 공식 v0.6.0 streaming 구현의 16-bit PyAV plane 처리 방식
- [`datasets/image_writer.py`](https://raw.githubusercontent.com/huggingface/lerobot/v0.6.0/src/lerobot/datasets/image_writer.py)
  - 단일채널 `uint16`/`float32` PIL 변환과 저장 옵션
- [`datasets/lerobot_dataset.py`](https://raw.githubusercontent.com/huggingface/lerobot/v0.6.0/src/lerobot/datasets/lerobot_dataset.py)
  - 공식 depth feature 저장·읽기 흐름 비교

### 백포트 대상 LeRobot v0.4.0

- [LeRobot v0.4.0 tag](https://github.com/huggingface/lerobot/tree/v0.4.0)
- [v0.4.0 datasets 디렉터리](https://github.com/huggingface/lerobot/tree/v0.4.0/src/lerobot/datasets)
- [v0.4.0 `image_writer.py`](https://github.com/huggingface/lerobot/blob/v0.4.0/src/lerobot/datasets/image_writer.py)
- [v0.4.0 `video_utils.py`](https://github.com/huggingface/lerobot/blob/v0.4.0/src/lerobot/datasets/video_utils.py)
- [v0.4.0 `lerobot_dataset.py`](https://github.com/huggingface/lerobot/blob/v0.4.0/src/lerobot/datasets/lerobot_dataset.py)
- [v0.4.0 `compute_stats.py`](https://github.com/huggingface/lerobot/blob/v0.4.0/src/lerobot/datasets/compute_stats.py)
- [v0.4.0 `utils.py`](https://github.com/huggingface/lerobot/blob/v0.4.0/src/lerobot/datasets/utils.py)

### Codec, PyAV, RealSense

- [x265 CLI documentation — lossless option](https://x265.readthedocs.io/en/stable/cli.html#cmdoption-lossless)
  - `x265-params=lossless=1` 설정 참고
- [PyAV Video API](https://pyav.basswood-io.com/docs/stable/api/video.html)
  - `VideoFrame`, pixel format 변환, encode/decode 처리 참고
- [Intel RealSense librealsense](https://github.com/IntelRealSense/librealsense)
  - RealSense z16 depth stream과 Python wrapper 참고
- [NVIDIA NVENC Programming Guide — High Bit Depth Encoding](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvenc-video-encoder-api-prog-guide/#high-bit-depth-encoding)
  - NVENC의 HEVC 고 bit-depth 지원이 Main10 중심임을 확인하는 데 참고했다.
- [NVIDIA NVENC Application Note](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvenc-application-note/index.html)
  - GPU별 HEVC lossless와 10-bit 인코딩 capability 비교에 참고했다.
