# SmolVLA 학습 방법 (기본 모델 — memory module 없음)

이 문서는 `erase the shape` 계열 task로 SmolVLA를 **기본 구조(memory module 없이,
top+wrist 또는 top-only)로 처음부터 다시 학습시키는 방법**을 정리한 것이다. 데이터가
어디서 왔든(원본 녹화든 증강된 데이터든) 아래 절차를 그대로 따르면 된다. 실제 사례와
실측치는 `smolvla_finetuning.md`(최초 60-episode 학습)와
`new_smolvla_finetuning.md`(150-episode 재학습, 2026-08-08)에 있다.

## 0. 환경

```text
conda 환경:     ugrp   (conda activate ugrp)
lerobot:        /home/ugrp43/UGRP/piper_sdk/lerobot  (로컬 editable clone, v0.4.4)
GPU:            RTX 5090 32GB × 2 (index 0, 1 — nvidia-smi로 확인)
프로젝트 루트:  /home/ugrp43/UGRP/lerobot_robot_piper
```

모든 명령은 프로젝트 루트에서, `conda activate ugrp` 후 실행한다.

## 1. 데이터셋 준비

### 1.1 원본 녹화에서 프레임 범위 정하기 (QC)

각 소스 녹화 폴더(`records/<날짜>/erase_the_<도형>_<타임스탬프>/`)는 로봇이 움직이기 전
대기 시간과 정리 동작이 섞여 있어서, 학습에 쓸 유효 프레임 범위(시작~끝)를 먼저 정해야
한다. `qc_studio.py`가 이 작업을 자동 제안 + 사람 확인으로 처리한다.

**GUI로 검수(권장, 사람이 직접 확인):**

```bash
python qc_studio.py --folder records/<날짜>
```

에피소드 하나씩 보여주면서 자동 제안된 시작/끝 프레임을 확인·조정하고 `Enter`로
확정한다. 결과가 `configs/qc_review_<날짜>.json`에 저장된다(`--output`으로 경로 지정 가능).

**GUI 없이 자동 계산만(빠르게 상태 확인할 때):**

```bash
python qc_studio.py --folder records/<날짜> --report
```

전부 GREEN이면 사람 검수 없이 자동 계산값을 그대로 써도 된다(이번 150-episode 재학습의
0804 데이터가 이 경우였다 — 30개 전부 GREEN이라 수작업 QC를 생략했다).

### 1.2 manifest JSON 만들기

`prepare_erase_shape_dataset.py`가 실제로 읽는 manifest 형식은 QC 리뷰 파일과 스키마가
약간 다르다(QC 리뷰엔 `auto_start_frame`, `qc_level` 같은 부가 필드가 있고, manifest는
`target_task`가 필수다). 여러 날짜의 QC 리뷰 결과를 모아 하나의 manifest로 합칠 때는
직접 변환 스크립트를 짜야 한다. `configs/erase_shape_150_manifest.json`이 실제 예시다:

```json
{
  "format_version": 1,
  "target_task": "erase the shape",
  "range_semantics": "start_frame is inclusive; end_frame is exclusive",
  "instructions": ["..."],
  "episodes": [
    {
      "source_dataset": "records/outlier/0802_joint4_incorrect/erase_the_circle_0802-130542",
      "total_frames": 758,
      "enabled": true,
      "start_frame": 98,
      "end_frame": 607
    }
  ]
}
```

주의할 점:

- `source_dataset` 경로는 **실제 존재하는 경로를 정확히 적어야 한다.** `resolve_source()`가
  `records/local/...` 같은 옛날 경로를 `records/0727/...`로 자동 치환해주는 예외 처리가
  있긴 하지만(0727 데이터 한정), 그 외 날짜는 안 통한다 — 직접 올바른 경로로 적을 것.
- `target_task`는 스크립트 상단 상수 `TARGET_TASK = "erase the shape"`와 **정확히
  일치**해야 한다(`load_manifest()`가 검증함).
- task를 도형별로 나누고 싶으면(`erase the circle` 등) 현재 스크립트엔 그 기능이 없다.
  1.5절 참고.

### 1.3 데이터셋 빌드

```bash
python scripts/tasks/erase_shape/dataset/prepare_erase_shape_dataset.py \
  --manifest configs/<manifest>.json \
  --output records/outputs/<output_name> \
  --cameras top,wrist \
  --top-crop 280,0,720 \
  --wrist-crop 280,0,720 \
  --image-size 512 \
  --gop-size 1 \
  --vcodec hevc \
  --encoder-threads 16
```

먼저 `--validate-only`를 붙여서 manifest만 빠르게 검증해보는 걸 권장한다(프레임 범위
오류를 몇 초 안에 잡아준다).

**옵션 설명:**

| 옵션 | 설명 |
|---|---|
| `--cameras` | `top,wrist`(기본) 또는 `top`만. top-only는 이번 세션에 추가한 기능 — wrist 피처 자체가 없는 데이터셋이 만들어진다(단순히 무시하는 게 아니라 아예 존재하지 않음). |
| `--top-crop` / `--wrist-crop` | `x,y,size` 형식, 정사각형 크롭. 지금까지 전 데이터셋이 `280,0,720`(1280×720 원본에서 왼쪽 280px 제외)으로 통일돼 있다. 새 카메라 배치라면 `scripts/piper/validation/preview_video_crop.py`로 미리 확인할 것. |
| `--image-size` | 크롭 후 리사이즈할 정사각형 크기. SmolVLA 기본 입력이 512×512라 지금까지 전부 512. |
| `--gop-size` | 키프레임 간격. **1을 강력히 권장한다** — 아래 경고 참고. |
| `--vcodec` | **반드시 `hevc`(소프트웨어)를 쓸 것.** 아래 경고 참고. |
| `--encoder-threads` | 소프트웨어 인코딩 시 CPU 코어 수만큼 주면(예: 16) 인코딩이 빨라진다. |

### ⚠️ 중요: `--gop-size`는 `--vcodec hevc_nvenc`(GPU 인코더)에서 무시된다

이번 세션에 발견한 lerobot 자체 버그다. `lerobot/datasets/video_utils.py`의
`_get_codec_options()`가 이렇게 돼 있다:

```python
# GOP size (keyframe interval) - supported by VideoToolbox and software encoders
if g is not None and (vcodec in ("h264_videotoolbox", "hevc_videotoolbox") or vcodec not in HW_ENCODERS):
    options["g"] = str(g)
```

`hevc_nvenc`는 `HW_ENCODERS`에 포함돼 있어서 이 조건에 안 걸리고, **GOP 옵션이 인코더에
아예 전달되지 않는다.** `--gop-size 1`을 줘도 조용히 무시되고 NVENC 자체 기본값(실측
~200프레임 간격)으로 인코딩된다 — 에러도 안 나고 로그에도 안 남아서 눈치채기 어렵다.

**결과적으로 GOP을 실제로 통제하려면 `--vcodec hevc`(소프트웨어 libx265)를 써야 한다.**
느리지 않을까 걱정할 수 있는데, `--encoder-threads 16`을 주면 150-episode(75,787프레임,
top+wrist 두 카메라)가 약 15분에 끝난다(RTX 5090 기준). GPU 인코더보다는 느리지만
감수할 만하다.

### 왜 GOP=1이 중요한가 (성능 + 안전성 둘 다)

- **학습 속도**: GOP이 크면(예: ~200) 특정 프레임을 꺼낼 때마다 가까운 키프레임으로
  되감았다가 순서대로 재생해서 복원해야 한다. GOP=1(전 프레임이 키프레임)이면 이 과정이
  필요 없어서 디코딩이 훨씬 단순해진다. 150-episode, top+wrist 기준 실측:
  GOP≈200(`--vcodec hevc_nvenc`, 부지불식간에 이렇게 됨) 2.07 step/s →
  GOP=1(`--vcodec hevc`) 약 10.15 step/s, **약 4.9배**.
- **크래시 방지**: GOP이 크면 `torchvision`/`pyav`의 seek이 드물게(0.3~0.7%) 목표
  프레임보다 한 프레임 앞에 착지하는 `FrameTimestampError`가 난다. 발생 확률은 낮아도
  30,000 step × batch 8 = 240,000 샘플이면 학습 도중 반드시 걸린다. GOP=1이면 이 문제
  자체가 구조적으로 발생할 수 없다.

### 1.4 빌드 후 검증 (선택 — 시간이 꽤 걸린다)

`prepare_erase_shape_dataset.py`가 끝나면서 찍는 `PASS` 로그는 프레임 수·정렬만
확인하고, **실제 디코딩 가능 여부는 확인하지 않는다.** 원하면 학습 시작 전에 전 프레임을
미리 디코딩해볼 수 있다:

```bash
python scripts/piper/validation/decode_check_parallel.py \
  local/<output_name> \
  /home/ugrp43/UGRP/lerobot_robot_piper/records/outputs/<output_name> \
  --workers 14
```

`TOTAL FAIL 0`이면 안심하고 학습 시작. 150-episode(75,787프레임) 기준 GOP=1이면 14
워커로도 약 12~25분 걸린다 — 데이터 양이 많으면 더 오래 걸릴 수 있다. `--workers`는
머신 코어 수(`nproc`)에서 여유분 2 정도 뺀 값을 권장한다.

**생략해도 되는 이유**: 1.3절의 GOP=1 빌드(`--vcodec hevc`)를 이미 따랐다면, 이 검증이
원래 잡으려던 문제(`FrameTimestampError`, GOP이 클 때 seek이 목표 프레임보다 앞에
착지하는 버그)는 GOP=1에서 구조적으로 발생할 수 없다. 즉 GOP=1 데이터셋에서 이 검증은
"혹시 모를 다른 파일 손상"을 잡는 보험 성격이지, 예전(GOP≈200) 데이터셋처럼 생략하면
학습이 높은 확률로 중간에 죽는 상황은 아니다. 시간이 급하면 건너뛰고 바로 학습을
시작해도 괜찮다 — 다만 몇 시간짜리 학습이 도중에 죽는 것보다는 수십 분 먼저 확인하는
쪽이 저렴하다는 점은 감안할 것.

### 1.5 도형별 task 라벨이 필요한 경우 (참고, 현재 미구현)

방해 도형이 있는 데이터(정답 도형 + 다른 도형이 화면에 같이 있어서 언어로 골라야 하는
경우)라면 `erase the shape` 하나로 통합하면 안 되고 `erase the circle` / `erase the
triangle` / `erase the rectangle`처럼 소스 그룹별로 다른 task 문자열을 줘야 한다.
**이 기능(`--top-video-dir`, `--per-shape-tasks` 플래그)은 현재 스크립트에 없다** — 예전에
구현했다가 커밋하지 않아 유실됐다(`new_smolvla_finetuning.md` 이전에 있던
`TODO_증강_재작업.md` 문서 참고, 복구 안 돼 있으면 재구현 필요). 재구현 지점은 다음
4곳이었다:

1. `TASK_BY_GROUP` 상수 + 소스 그룹명 → task 문자열 매핑 헬퍼
2. `add_selected_episode()`에서 TOP 비디오 경로를 별도 디렉터리에서 교체
3. `validate_output()`의 "단일 task 강제" assert 3곳을 에피소드별 검증으로 교체
4. `build_dataset()`/`main()`에 인자 전달

방해 도형 없이 도형 하나만 화면에 있는 경우(지금까지의 150-episode 데이터가 이 경우)는
이 기능이 필요 없다 — `target_task`를 `erase the shape` 하나로 통합하면 된다.

## 2. 학습

### 2.1 기본 명령 (top+wrist)

```bash
conda activate ugrp
tmux new -s <세션이름>
cd /home/ugrp43/UGRP/lerobot_robot_piper

CUDA_VISIBLE_DEVICES=<GPU번호> \
ACCELERATE_MIXED_PRECISION=bf16 \
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.input_features=null \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/<output_name> \
  --dataset.root=/home/ugrp43/UGRP/lerobot_robot_piper/records/outputs/<output_name> \
  --dataset.video_backend=pyav \
  --output_dir=/home/ugrp43/UGRP/lerobot_robot_piper/outputs/train/<run_name> \
  --job_name=<run_name> \
  --batch_size=8 \
  --num_workers=2 \
  --steps=30000 \
  --log_freq=100 \
  --save_freq=5000 \
  --eval_freq=0 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=30000 \
  --wandb.enable=true \
  --wandb.project=<프로젝트명> \
  --wandb.run_id=<고유_run_id>
```

`Ctrl+B, D`로 detach(VS Code 통합 터미널에선 `Ctrl+B`가 사이드바 토글과 겹칠 수 있음 —
안 먹히면 별도 터미널 앱을 쓰거나 tmux prefix를 바꿀 것).

### 2.2 top-only로 학습하려면

1.3절에서 `--cameras top`으로 wrist 없는 데이터셋을 따로 빌드했다면, `--dataset.repo_id`/
`--dataset.root`만 그 데이터셋으로 바꾸면 된다. `--policy.input_features=null`이 데이터셋에
있는 피처만 보고 자동으로 구성을 잡아주기 때문에 다른 옵션은 그대로다.

**주의**: `--policy.input_features`로 wrist를 수동으로 빼는 방식은 안 된다.
`resolve_delta_timestamps()`(`lerobot/datasets/factory.py`)가 policy 설정과 무관하게
데이터셋에 있는 `observation.*` 피처를 전부 로드 대상에 넣기 때문에, top-only로
학습하려면 **데이터셋 자체에 wrist 피처가 없어야 한다.**

### 2.3 설정값 근거

전부 `smolvla_finetuning.md`(최초 60-episode 학습)에서 검증된 값을
그대로 쓴다 — 임의로 바꾸지 말 것(별도 비교 실험 없이는).

| 설정 | 값 | 근거 |
|---|---|---|
| `freeze_vision_encoder` | `true` | SmolVLA preset, VLM은 얼리고 action expert만 학습 |
| `train_expert_only` | `true` | 위와 동일 목적 |
| `batch_size` | `8` | RTX 5090에서 smoke test로 VRAM 여유 확인됨(peak 3.6GB/32GB) |
| `steps` | `30000` | 데이터 규모 기준 약 7.9 epoch(60-episode 기준. episode 수가 늘어도 step/s엔 영향 없음 — 4.3절 참고) |
| `scheduler_warmup_steps` / `decay_steps` | `1000` / `30000` | 전체 학습 길이에 맞춤 |
| `save_freq` | `5000` | 총 6개 체크포인트 저장 |
| `eval_freq` | `0` | 연결된 시뮬레이션 환경 없음 |
| `ACCELERATE_MIXED_PRECISION=bf16` | 켜짐 | **기본값 아님, 명시적으로 켜야 함.** 안 켜면 FP32로 돈다. SmolVLA 공식 파인튜닝 권장값(HF 모델카드의 `--policy.dtype=bfloat16`)과 동일 목표 — 우리 lerobot 0.4.4엔 그 플래그가 없어서 Accelerate 환경변수로 우회한다. |

### 2.4 흔한 실수 (전부 이번 세션에 실제로 겪은 것들)

1. **`output_dir`가 이미 있으면 학습이 아예 안 켜진다.** `FileExistsError`. 재시도할 땐
   `output_dir`/`job_name`을 새 이름으로 바꾸거나 기존 폴더를 지울 것. `--resume=true`로
   이어서 할 수도 있다(`--config_path=<output_dir>/checkpoints/last/pretrained_model/train_config.json`).
2. **`--wandb.run_id`를 재사용하면 기존 run에 새 학습이 그대로 섞여 들어간다.** 이름/태그가
   덮어써지고 `train/loss` 기록도 같은 step 구간이 새 값으로 덮인다. 재시도할 땐 `run_id`도
   반드시 새로 바꿀 것.
3. **W&B는 `--wandb.disable_artifact`를 안 끄면 체크포인트 전체(수백MB)를 매
   `save_freq`마다 아티팩트로 업로드한다.** 무료 플랜 100GB 안에서는 문제없지만, run을
   여러 개 돌릴 계획이면 감안할 것.
4. **GPU 여러 장 동시 학습 시 온도 확인.** `watch -n 2 "sensors | grep Tctl; echo; free -h; echo; nvidia-smi"`로 계속 지켜볼 것. RTX 5090 쓰로틀 시작점은 90°C.

## 3. 참고 문서

- `smolvla_finetuning.md` — 최초 60-episode 학습(GOP≈200) 상세 기록, 설정값 선정 근거표
- `new_smolvla_finetuning.md` — 150-episode 재학습(GOP=1) 진행 기록, 이번에 발견한 NVENC/GOP 버그와 실측 속도 비교
- `scripts/tasks/erase_shape/dataset/prepare_erase_shape_dataset.py` — 데이터셋 빌드 스크립트 (`--help`로 전체 옵션 확인)
- `qc_studio.py` — 프레임 범위 QC 도구
- `scripts/piper/validation/decode_check_parallel.py` — 전수 디코딩 검증 도구(이번 세션에 추가)
