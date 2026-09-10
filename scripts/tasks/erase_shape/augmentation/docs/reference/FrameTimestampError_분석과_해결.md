# FrameTimestampError: 원인 분석과 근본 해결

작성 2026-08-02. 대상: `records/outputs/erase_the_shape_aug_512` 학습 실패 건.

---

## 1. 증상

방해 도형을 합성한 새 데이터셋으로 SmolVLA 학습을 시작하니 **step 10에서 크래시**했다.

```text
lerobot.datasets.video_utils.FrameTimestampError:
One or several query timestamps unexpectedly violate the tolerance
(tensor([0.1000]) > tolerance_s=0.0001).

queried timestamps: tensor([948.5667])
loaded  timestamps: tensor([948.6667])
video: records/outputs/erase_the_shape_aug_512/videos/observation.images.wrist/chunk-000/file-000.mp4
backend: pyav
```

요청한 시각과 실제로 얻은 프레임이 **0.1초(30fps 기준 3프레임)** 어긋났다.

주의할 점: 에러가 난 영상은 **WRIST**다. 이번에 증강한 것은 TOP뿐이고 WRIST는 원본을 그대로 재인코딩했다. 즉 **도형 합성 자체와 무관한 문제**다.

---

## 2. 환경

```text
lerobot      0.4.4
torch        2.7.1+cu128
torchvision  0.22.1+cu128
av (PyAV)    15.1.0
torchcodec   설치 실패 상태 (3절 참고)
```

---

## 3. 원인

### 3.1 lerobot의 프레임 접근 방식

`lerobot/datasets/video_utils.py`의 `decode_video_frames_torchvision()`:

```python
torchvision.set_video_backend(backend)
if backend == "pyav":
    keyframes_only = True   # pyav doesn't support accurate seek

reader = torchvision.io.VideoReader(video_path, "video")
reader.seek(first_ts, keyframes_only=keyframes_only)

for frame in reader:        # 키프레임부터 앞으로 재생하며 목표 도달
    ...
```

pyav 백엔드는 정확한 seek을 못 하므로 **키프레임으로 되감은 뒤 앞으로 재생**해서 목표 프레임에 도달한다.

이 방식의 전제는 **"seek이 목표 시각보다 앞(과거)에 착지한다"** 이다. 뒤에 착지하면 목표 프레임은 앞으로 재생해도 영원히 나오지 않는다. 그러면 가장 가까운 프레임이 tolerance(`0.0001s`)를 넘게 되고 위 예외가 발생한다.

### 3.2 실측

문제의 시각에 대해 두 데이터셋에서 seek을 직접 호출한 결과:

```text
OLD (기존 학습 성공): seek(948.5667) -> pts=940.3333   -8.233s (-247 frames)  정상
NEW (이번 실패)     : seek(948.5667) -> pts=948.6667   +0.100s ( +3 frames)  실패
```

OLD는 247프레임 뒤로 착지한 뒤 앞으로 감아 정확히 도달한다. NEW는 3프레임 **뒤에** 착지해 도달이 불가능하다.

### 3.3 발생 빈도

30,545개 프레임 시각 중 1,222개(1/25)를 샘플링해 seek을 실측했다.

| 데이터셋 | wrist | top |
|---|---:|---:|
| OLD (`records/0727/erase_the_shape_512`) | 0 / 1222 | 0 / 1222 |
| NEW (`records/outputs/erase_the_shape_aug_512`) | 4 / 1222 | 8 / 1222 |

실패율은 0.3~0.7%로 낮다. 하지만 30,000 step × batch 8 = **240,000 샘플**을 뽑는 학습에서는 즉시 터진다. 실제로 step 10에서 걸렸다.

### 3.4 인코딩 차이

두 데이터셋의 컨테이너를 비교하면 키프레임 배치가 다르다.

| | 키프레임 수 | 간격 min/max |
|---|---:|---:|
| OLD wrist | 153 | 1 / **250** |
| NEW wrist | 153 | 5 / **254** |
| NEW top | 153 | 4 / **251** |

`prepare_erase_shape_dataset.py`는 GOP를 250으로 지정한다.

```python
# LeRobot 0.4.4 does not expose the streaming GOP through create().
assert writer._streaming_encoder is not None
writer._streaming_encoder.g = gop_size    # 250
```

OLD는 250을 정확히 지키는데 **NEW는 251~254로 초과**한다. 같은 스크립트, 같은 `--gop-size 250`, WRIST는 같은 원본인데도 결과가 다르다.

### 3.5 미해결로 남은 부분

**왜 같은 설정에서 키프레임 배치가 달라졌는지는 규명하지 못했다.** 가설은 x265의 scenecut 자동 삽입, 또는 스트리밍 인코더의 스레딩 비결정성인데 어느 쪽도 확인하지 않았다.

이것이 중요한 이유: 원인을 모르면 **같은 스크립트로 다시 빌드해도 또 터질 수 있다.** 이번에 OLD가 멀쩡했던 것은 설계가 아니라 운일 가능성이 있다.

---

## 4. 조사 과정에서의 오류 (반복 방지용)

키프레임 목록으로 실패를 예측하는 정적 분석을 **두 번 시도했고 두 번 다 틀렸다.**

1차 — "pyav는 가장 가까운 키프레임을 고른다" 가정:

```text
OLD wrist  도달불가 15069 / 30545
NEW wrist  도달불가 15069 / 30545     <- OLD/NEW 구분 못 함. 모델이 틀림
```

2차 — "쿼리 이하의 최대 키프레임" 가정:

```text
OLD wrist  도달불가 0
NEW wrist  도달불가 0                 <- 실패하는 NEW를 정상으로 판정. 모델이 틀림
```

둘 다 **알려진 실패를 재현하지 못했다.** pyav의 실제 seek 동작은 이 단순 모델들과 다르다.

교훈: **키프레임 메타데이터 분석으로 판정하지 말 것.** 실제로 seek을 호출하거나 `LeRobotDataset.__getitem__`을 돌려서 실측할 것.

---

## 5. 적용한 해결

전 프레임을 키프레임으로 인코딩(`--gop-size 1`)해 "seek이 뒤에 착지" 자체를 구조적으로 불가능하게 만들었다.

```bash
python scripts/tools/prepare_erase_shape_dataset.py \
  --per-shape-tasks \
  --top-video-dir records/outputs/0727_shape_aug_top \
  --output records/outputs/erase_the_shape_aug_512_g1 \
  --gop-size 1 \
  --top-crop 280,0,720 --wrist-crop 280,0,720 --image-size 512
```

검증 결과:

```text
seek 실측 (1222 지점)      wrist 0 실패 / top 0 실패
전수 디코딩 (30,545 전부)  실패 0건
   -> LeRobotDataset.__getitem__ 으로 학습과 동일 경로 확인
```

| | OLD | NEW gop250 | NEW gop1 |
|---|---:|---:|---:|
| seek 실패 (1222) | 0 | 12 | 0 |
| 전수 디코딩 | — | — | 실패 0 |
| 용량 | 21M | 20M | **264M** |

용량이 12배 늘었다. 디스크 여유(66G)로는 문제없다.

### 이것은 완전한 근본 해결이 아니다

`--gop-size 1`은 **증상을 확실히 막지만 원인(3.5절)을 고치지는 않는다.** 부작용:

- 용량 12배
- inter-frame 압축을 포기하므로 I/O 증가
- 데이터셋이 커지면 비용이 선형으로 늘어남

---

## 6. 근본 해결 후보

### 6.1 torchcodec 백엔드 (가장 근본적)

torchcodec은 정확한 seek을 지원하므로 키프레임 배치와 무관해진다. lerobot이 이미 지원한다.

```python
if backend == "torchcodec":
    return decode_video_frames_torchcodec(video_path, timestamps, tolerance_s)
```

현재 환경에서는 import가 실패한다:

```text
FFmpeg version 7: libavutil.so.59: cannot open shared object file
FFmpeg version 6: libavutil.so.58: cannot open shared object file
FFmpeg version 5: libavutil.so.57: cannot open shared object file
FFmpeg version 4: libavutil.so.56: cannot open shared object file
```

FFmpeg 공유 라이브러리가 없다. conda로 ffmpeg를 설치하면 해결될 가능성이 높다.

```bash
conda install -c conda-forge ffmpeg
python -c "import torchcodec; print('ok')"
```

성공하면 `--dataset.video_backend=torchcodec`으로 학습하고, GOP는 250으로 되돌려 용량을 회복할 수 있다.

**미검증이다.** 실제로 시도해서 확인해야 한다.

### 6.2 빌드 후 검증 게이트 (즉시 적용 가능)

이번 사고의 진짜 문제는 인코딩보다 **검증 부재**였다. 빌드 스크립트의 `validate_output()`은 프레임 수, task, state/action 정합만 확인하고 **디코딩 가능 여부는 전혀 보지 않는다.** 그래서 "PASS"가 뜬 데이터셋이 학습에서 터졌다.

`prepare_erase_shape_dataset.py`의 `validate_output()` 마지막에 다음을 추가할 것:

```python
# 학습과 동일 경로로 전수 디코딩. 이걸 통과 못 하면 데이터셋을 내보내지 않는다.
dataset = LeRobotDataset(output_repo_id, root=root, video_backend="pyav")
failures = []
for index in range(len(dataset)):
    try:
        dataset[index]
    except Exception as exc:
        failures.append((index, type(exc).__name__))
if failures:
    raise AssertionError(f"디코딩 실패 {len(failures)}건: {failures[:5]}")
```

30,545 프레임 기준 수 분에서 십수 분 걸린다. 4시간짜리 학습을 날리는 것보다 훨씬 싸다.

샘플링으로 줄이려면 실패율이 0.3%대였다는 점을 감안해 최소 수천 개는 봐야 한다. 1222개 샘플에서는 12건이 잡혔지만, 이는 운이 좋은 경우로 봐야 한다.

---

## 7. 권고 순서

1. **지금**: `erase_the_shape_aug_512_g1`으로 학습 진행 (검증 완료)
2. **다음**: 6.2의 검증 게이트를 빌드 스크립트에 넣기 — 가장 확실한 재발 방지
3. **여유 있을 때**: 6.1의 torchcodec 설치 시도. 되면 GOP 250 + torchcodec 조합으로 용량과 안정성 양립
4. **장기**: 3.5절의 키프레임 비결정성 규명. 안 되면 gop-size 1을 기본값으로 고정하는 것도 방법

---

## 8. 관련 경로

```text
records/outputs/0727_shape_aug_top/            증강 TOP mp4 60개 + index.json
records/outputs/erase_the_shape_aug_512/       gop250, 학습 불가 (원인 비교용 보존)
records/outputs/erase_the_shape_aug_512_g1/    gop1, 검증 통과. 학습용
records/0727/erase_the_shape_512/              기존 데이터셋. 미변경
scripts/tools/prepare_erase_shape_dataset.py   --top-video-dir, --per-shape-tasks 추가됨
```

---

## 부록: 별개로 남은 데이터 이슈

이번 크래시와 무관하지만 같이 확인된 사항.

**크롭에 의한 방해 도형 잘림.** 증강기는 보드 ROI(x 210~749) 기준으로 배치했는데 학습 크롭은 `--top-crop 280,0,720`이다. 60개 중 36개의 방해 도형이 좌측에서 잘린다(최악 55%만 보임). 완전 소실은 0개. 재생성 시 `x < 300`을 manual exclusion으로 막을 것.

**위치 편향.** 512 프레임 기준 중심 x를 측정한 결과:

```text
TARGET      min 107   mean 220   max 270      <- 폭 163으로 좁게 몰림
DISTRACTOR  min   5   mean  87   max 273

방해 도형이 정답보다 왼쪽: 56 / 60  (93%)
```

절대 위치 분포는 겹치지만 상대 순서가 93% 한쪽으로 쏠린다. "왼쪽이 가짜" 규칙으로 대부분 맞힐 수 있다는 뜻이다.

원인은 방해 도형 배치가 아니라 **원본 60개 녹화에서 사람이 그린 정답 도형 위치가 좁게 몰려 있는 것**이다. 증강기는 정답 도형과 로봇팔 궤적을 피해 배치하므로 남는 안전 영역이 자연히 그 왼쪽이 된다. 증강만으로는 제거할 수 없고, 도형 위치를 다양하게 재녹화해야 한다.

따라서 학습 후 평가에서는 **같은 영상에 언어 지시만 반대 도형으로 바꿔 행동이 실제로 달라지는지** 확인해야 한다. 안 바뀌면 위치로 푼 것이다.
