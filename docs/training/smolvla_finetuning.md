# SmolVLA Fine-tuning Setup

## 1. 문서 범위

이 문서는 `erase the shape` 데이터로 SmolVLA를 fine-tuning하기 위해 현재
워크스테이션에서 구성하고 검증한 환경을 정리한다.

현재 기준:

- 프로젝트: `/home/ugrp43/UGRP/lerobot_robot_piper`
- LeRobot clone: `/home/ugrp43/UGRP/lerobot`
- LeRobot: v0.4.4 editable install
- Python: 3.10
- Conda 환경: `ugrp`
- GPU: NVIDIA GeForce RTX 5090 32GB 2장
- 학습에는 우선 GPU 0 한 장만 사용
- Policy: `lerobot/smolvla_base`

공식 참고 자료:

- [LeRobot SmolVLA 문서](https://github.com/huggingface/lerobot/blob/main/docs/source/smolvla.mdx)
- [SmolVLA base model](https://huggingface.co/lerobot/smolvla_base)
- [LeRobot Compute Hardware Guide](https://huggingface.co/docs/lerobot/main/hardware_guide)
- [SmolVLA 논문](https://huggingface.co/papers/2506.01844)

## 2. 학습 데이터셋

학습 데이터셋:

```text
/home/ugrp43/UGRP/lerobot_robot_piper/records/0727/erase_the_shape_512
```

구성:

| 항목 | 값 |
|---|---|
| Task | `erase the shape` |
| Episodes | 60 |
| Frames | 30,545 |
| FPS | 30 |
| `observation.state` | joint position 7차원 |
| `action` | joint position 7차원 |
| TOP | RGB HEVC, 512×512 |
| WRIST | RGB HEVC, 512×512 |

원본 1280×720 영상에 적용한 크롭은 TOP과 WRIST 모두 같다.

```text
x=280, y=0, size=720
```

처리 과정:

```text
1280×720 원본
  → (280, 0)에서 720×720 정사각형 크롭
  → 512×512로 축소
  → HEVC로 저장
```

유효 프레임 범위는 다음 JSON을 사용한다.

```text
configs/erase_shape_frame_ranges.json
```

데이터셋 생성 스크립트:

```text
scripts/tools/prepare_erase_shape_dataset.py
```

재현 명령은 다음과 같다. 기존 출력 경로가 존재하면 스크립트가 덮어쓰지 않고
중단하므로, 현재 완성된 데이터셋에 이 명령을 다시 실행할 필요는 없다.

```bash
conda activate ugrp
cd /home/ugrp43/UGRP/lerobot_robot_piper

python scripts/tools/prepare_erase_shape_dataset.py \
  --manifest configs/erase_shape_frame_ranges.json \
  --output records/0727/erase_the_shape_512 \
  --top-crop 280,0,720 \
  --wrist-crop 280,0,720 \
  --image-size 512 \
  --vcodec hevc_nvenc
```

생성 과정에서 다음 검증을 통과했다.

- 60개 episode와 JSON 프레임 범위 일치
- 총 30,545프레임
- state/action 값이 선택한 원본 프레임과 일치
- episode별 frame index 연속성
- TOP/WRIST와 state/action의 프레임 정렬
- LeRobotDataset loader에서 state/action shape 7
- TOP/WRIST MP4 전체 디코딩 결과 각각 30,545프레임
- TOP/WRIST 영상과 메타데이터 모두 512×512

## 3. 크롭 미리보기

실제 데이터셋을 만들기 전에 SSH 환경에서 크롭 결과를 확인하려면 다음 도구를
사용한다.

```text
scripts/tools/preview_video_crop.py
```

Crop 인자는 원본 영상의 왼쪽 위를 `(0, 0)`으로 하는 `x,y,size` 형식이다.

```text
x: 원본 왼쪽에서 crop 왼쪽 경계까지의 픽셀
y: 원본 위쪽에서 crop 위쪽 경계까지의 픽셀
size: 정사각형 crop의 한 변 길이
```

현재 학습 데이터의 `280,0,720`은 1280×720 원본에서 왼쪽 280픽셀을 제외한
720×720 영역을 선택한다. Preview는 다음 세 화면을 PNG에서 비교할 수 있게 한다.

```text
Crop 경계가 표시된 원본
  → 잘라낸 정사각형 영역
  → 512×512로 축소된 최종 모델 입력
```

유효 범위 중앙 프레임 60개 확인:

```bash
python scripts/tools/preview_video_crop.py \
  records/0727/erase_the_shape \
  --frame-ranges configs/erase_shape_frame_ranges.json \
  --top-crop 280,0,720 \
  --wrist-crop 280,0,720
```

이 명령은 episode마다 PNG를 60개 만드는 것이 아니라, 활성화된 60개 episode의
중앙 frame을 5열 contact sheet 하나로 묶어 camera별로 저장한다.

```text
tmp/crop_preview/top_crop_midpoints.png
tmp/crop_preview/wrist_crop_midpoints.png
```

WRIST의 각 유효 범위 시작 프레임 확인:

```bash
python scripts/tools/preview_video_crop.py \
  records/0727/erase_the_shape \
  --frame-ranges configs/erase_shape_frame_ranges.json \
  --camera wrist \
  --range-point start \
  --wrist-crop 280,0,720
```

이때 출력은 다음 파일이다.

```text
tmp/crop_preview/wrist_crop_start_frames.png
```

끝 frame을 보고 싶으면 `--range-point end`를 사용한다. Manifest와 관계없이 원본
통합 영상의 특정 global frame만 확인할 수도 있다.

```bash
python scripts/tools/preview_video_crop.py \
  records/0727/erase_the_shape \
  --camera both \
  --frames 0,5000,10000 \
  --top-crop 280,0,720 \
  --wrist-crop 280,0,720
```

출력은 기본적으로 Git에 포함되지 않는 다음 경로에 생성된다.

```text
tmp/crop_preview
```

사용한 crop, 선택 frame 및 출력 경로는 함께 생성되는
`tmp/crop_preview/crop_config.json`에서 확인한다. 이 도구는 PNG와 JSON만 만들며
원본 dataset과 MP4는 수정하지 않는다.

실물 인간 승인 추론에서는
`scripts/tools/piper_human_approved_inference.py`가 같은 `280,0,720` crop과
512×512 `INTER_AREA` resize를 live TOP/WRIST에 적용한다. 관련 환경변수는
`HUMAN_APPROVED_TOP_CROP`, `HUMAN_APPROVED_WRIST_CROP`,
`HUMAN_APPROVED_CAMERA_OUTPUT_SIZE`다.

## 4. SmolVLA 의존성

LeRobot clone에서 SmolVLA 선택 의존성을 설치한다.

```bash
conda activate ugrp
cd /home/ugrp43/UGRP/lerobot
python -m pip install -e ".[smolvla]"
```

현재 설치 확인 결과:

```text
lerobot       0.4.4
transformers  4.57.6
torch         2.7.1+cu128
```

`transformers`, `num2words`, `accelerate`, `safetensors`와 SmolVLA import를
확인했다. LeRobot은 외부 clone을 editable install로 사용한다.

## 5. Input feature 설정

`smolvla_base`의 원래 설정에는 다음 feature가 저장되어 있다.

```text
observation.images.camera1
observation.images.camera2
observation.images.camera3
observation.state: 6
action: 6
```

현재 데이터셋은 다음 feature를 사용한다.

```text
observation.images.top
observation.images.wrist
observation.state: 7
action: 7
```

LeRobot v0.4.4에서는 다음 설정으로 pretrained input feature를 비우면 안 된다.

```bash
--policy.input_features='{}'
```

빈 dictionary는 pretrained dictionary의 기존 key를 제거하지 않아
`camera1/2/3`와 `top/wrist`가 충돌한다. 전체 설정을 `None`으로 바꾸고
데이터셋에서 다시 추론하도록 다음 옵션을 사용한다.

```bash
--policy.input_features=null
```

성공한 checkpoint에서 실제로 저장된 feature는 다음과 같다.

```text
observation.state:         [7]
observation.images.top:    [3, 512, 512]
observation.images.wrist:  [3, 512, 512]
action:                    [7]
```

카메라 feature 이름을 `camera1/2`로 바꾸는 것이 아니라, 데이터셋의 `top/wrist`
이름을 그대로 사용한다.

### 학습 정규화

학습 시 `observation.state`와 `action`은 데이터셋 statistics를 사용하는 policy
preprocessor에서 정규화되고, 추론 결과는 postprocessor에서 원래 Piper action
단위로 역변환된다. 현재 학습 데이터의 state와 action은 모두 position 7차원이다.

원본 녹화에 포함됐던 effort와 velocity는 `erase_the_shape_512` 가공 데이터에서
제외했으므로 이 checkpoint의 policy 입력이나 정규화 대상이 아니다. 실물 실행의
effort safety cutoff는 policy 입력과 별개로 Piper SDK에서 읽은 current 기반
추정값(N·m)을 원래 단위 그대로 임계값과 비교한다.

### 설정 출처: SmolVLA preset과 프로젝트 결정

최종 실행 설정의 기준 파일은 다음 두 개다.

```text
LeRobot 기본 preset:
/home/ugrp43/UGRP/lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py

최종 실행에 실제 저장된 설정:
outputs/train/smolvla_erase_shape_512/checkpoints/030000/pretrained_model/train_config.json
```

설정 출처는 다음 네 종류로 구분한다.

| 분류 | 의미 |
|---|---|
| SmolVLA preset 유지 | LeRobot v0.4.4 `SmolVLAConfig`의 기본값을 별도 탐색 없이 사용 |
| 기본값 명시 | 기본값과 같지만 실행 명령에 적어 의도를 고정 |
| 프로젝트 결정 | 데이터 규모, RTX 5090 실측 또는 저장 정책에 맞춰 선택 |
| Pretrained 상속 | `lerobot/smolvla_base`에 저장된 config를 불러오며 결정된 값 |

선정 근거의 우선순위는 다음과 같다.

1. Base model과 기본 fine-tuning 방식은 공식 SmolVLA 문서와
   `SmolVLAConfig` preset을 따른다.
2. Batch size와 mixed precision은 RTX 5090에서 20-step smoke test로 검증한다.
3. Step 수는 데이터셋 frame 수와 목표 epoch를 기준으로 계산한다.
4. Camera crop과 feature 구성은 실제 입력 화면 확인 및 Piper 데이터 형식에 맞춘다.
5. 별도 비교 실험이 없는 optimizer·architecture 값은 임의 변경하지 않고 preset을
   그대로 사용한다.

#### 모델·action 구조

| 설정 | 최종값 | 출처 | 선정 근거 |
|---|---:|---|---|
| Base model | `lerobot/smolvla_base` | 프로젝트 결정 | 공식 SmolVLA fine-tuning 시작점인 pretrained 450M model 사용 |
| `n_obs_steps` | 1 | SmolVLA preset 유지 | 현재 observation 한 시점으로 chunk를 생성하는 기본 구조 |
| `chunk_size` | 50 | SmolVLA preset 유지 | Base model과 공식 SmolVLA action horizon 유지 |
| `n_action_steps` | 50 | SmolVLA preset 유지 | 학습 target은 전체 50-action chunk. 실물에서는 실행기가 별도로 10개씩 승인 |
| `num_steps` | 10 | SmolVLA preset 유지 | Flow-matching action 생성의 기본 integration step 수 |
| State/action 최대 차원 | 32 / 32 | SmolVLA preset 유지 | 실제 7차원 vector를 model 내부 고정 크기에 padding |
| Image resize | 512×512 padding | SmolVLA preset 유지 | Base policy가 기대하는 visual 입력 크기 |
| Empty cameras | 0 | SmolVLA preset 유지 | 실제 TOP/WRIST 두 camera만 사용하며 가짜 세 번째 camera를 추가하지 않음 |
| Delta joint action | `false` | SmolVLA preset 유지 | Piper 데이터의 absolute action을 그대로 학습 |

`n_action_steps=50`은 학습 target 구성이고, 인간 승인 실행기의
`HUMAN_APPROVED_EXECUTE_ACTIONS=10`과는 다른 값이다. 실행기는 policy가 생성한
50개를 10개씩 나누지만 학습 target 자체를 10개로 바꾸지는 않는다.

#### Fine-tuning 범위

| 설정 | 최종값 | 출처 | 선정 근거 |
|---|---:|---|---|
| `freeze_vision_encoder` | `true` | 기본값 명시 | SmolVLA preset 유지, pretrained visual representation 보존 및 메모리·연산 감소 |
| `train_expert_only` | `true` | 기본값 명시 | SmolVLA preset 유지, VLM 본체보다 action expert와 projection 중심으로 학습 |
| `train_state_proj` | `true` | SmolVLA preset 유지 | Piper 7차원 state를 새 입력 공간에 맞추는 projection은 학습 |
| PEFT | `false` | Pretrained/공통 기본값 유지 | LoRA 등 별도 PEFT 실험을 하지 않음 |
| RABC | `false` | 공통 기본값 유지 | Sample reweighting 없이 기본 behavior cloning loss 사용 |
| `torch.compile` | `false` | SmolVLA preset 유지 | 첫 재현 실험에서 compile 관련 변수를 추가하지 않음 |

`freeze_vision_encoder=true`와 `train_expert_only=true`가 최적이라는 비교 실험은
하지 않았다. 이 값은 공식 LeRobot v0.4.4 SmolVLA preset의 메모리 절약형
fine-tuning 경로를 유지한 것이며, 향후 전체 VLM fine-tuning과 성능을 비교하려면
별도 run이 필요하다.

#### 정규화와 feature

| 설정 | 최종값 | 출처 | 선정 근거 |
|---|---|---|---|
| Visual normalization | `IDENTITY` | SmolVLA preset 유지 | Visual 전처리는 pretrained processor 경로 사용 |
| State normalization | `MEAN_STD` | SmolVLA preset 유지 | Dataset statistics로 7차원 position 정규화 |
| Action normalization | `MEAN_STD` | SmolVLA preset 유지 | Dataset statistics로 absolute action 정규화·역정규화 |
| `policy.input_features` | `null`로 초기화 후 dataset에서 추론 | 프로젝트 결정 | Pretrained의 `camera1/2/3`, state 6차원 설정을 제거하고 TOP/WRIST, state 7차원으로 교체 |
| Output feature | action 7차원 | Dataset에서 추론 | Piper joint 6개와 gripper absolute position |
| Effort/velocity | 제외 | 프로젝트 결정 | 이번 첫 policy는 position과 RGB만 사용해 입력 변수를 줄임 |
| Image transforms | 비활성화 | 공통 기본값 유지 | 별도 augmentation 비교 없이 첫 baseline을 원본 crop으로 고정 |

Image augmentation을 끈 것이 최적이라고 검증한 것은 아니다. 현재
`train_config.json`에는 brightness, contrast, affine 등의 후보 설정이 들어 있지만
`image_transforms.enable=false`이므로 실제 학습에는 적용되지 않았다.

#### Optimizer와 scheduler

| 설정 | 최종값 | 출처 | 선정 근거 |
|---|---:|---|---|
| Optimizer | AdamW | SmolVLA preset 유지 | `get_optimizer_preset()` 반환값 사용 |
| Peak learning rate | `1e-4` | SmolVLA preset 유지 | 별도 LR search 없이 공식 코드 기본값 사용 |
| Betas | `(0.9, 0.95)` | SmolVLA preset 유지 | 별도 탐색 없음 |
| Epsilon | `1e-8` | SmolVLA preset 유지 | 별도 탐색 없음 |
| Weight decay | `1e-10` | SmolVLA preset 유지 | 사실상 매우 약한 decay인 공식 코드 기본값 |
| Gradient clip norm | `10.0` | SmolVLA preset 유지 | Gradient 폭주 완화를 위한 기본 제한 |
| Scheduler | Cosine decay with warmup | SmolVLA preset 유지 | `get_scheduler_preset()` 반환값 사용 |
| Warmup | 1,000 steps | 기본값 명시 | 30,000-step run의 초반 약 3.3%에서 LR 상승 |
| Decay | 30,000 steps | 기본값 명시 | 전체 학습 길이에 맞춰 cosine decay 종료점을 final step으로 고정 |
| Final learning rate | `2.5e-6` | SmolVLA preset 유지 | 별도 탐색 없음 |

Optimizer와 scheduler 값은 이 데이터셋에서 grid search로 최적화한 결과가 아니다.
SmolVLA preset을 baseline으로 사용했으며, 성능 비교 없이 임의 변경하지 않았다.

#### 실행·자원·기록 설정

| 설정 | 최종값 | 출처 | 선정 근거 |
|---|---:|---|---|
| Batch size | 8 | 프로젝트 결정 | RTX 5090에서 20-step smoke test로 VRAM/RAM 여유 확인 |
| Steps | 30,000 | 프로젝트 결정 | 30,545 frames에서 240,000 samples, 약 7.9 epoch가 되도록 설정 |
| Mixed precision | BF16 | 프로젝트 결정 | RTX 5090 지원, FP16보다 넓은 exponent range 및 메모리 절감 |
| GPU | GPU 0 한 장 | 프로젝트 결정 | 단일 GPU baseline으로 재현 단순화 |
| DataLoader workers | 2 | 프로젝트 결정 | Smoke test에서 안정적으로 영상 decoding과 학습 완료 |
| Seed | 1000 | LeRobot train 기본값 유지 | 첫 baseline 재현성 고정, seed 비교 실험은 하지 않음 |
| Save frequency | 5,000 steps | 프로젝트 결정 | 복구와 중간 checkpoint 비교를 위해 총 6개 저장 |
| Log frequency | 100 steps | 프로젝트 결정 | 터미널 출력량과 추세 확인 간 절충 |
| Evaluation frequency | 0 | 프로젝트 결정 | 연결된 simulation environment가 없어 online eval 비활성화 |
| W&B | 비활성화 | 프로젝트 결정 | 첫 local run은 외부 tracking 없이 수행 |
| Push to Hub | 비활성화 | 프로젝트 결정 | Local checkpoint만 보존 |

`eval_freq=0`이고 별도 validation split도 만들지 않았으므로 validation loss나 작업
성공률로 checkpoint를 선택하지 않았다. `wandb.enable=false`이고 터미널 출력을
파일로 저장하지 않아 step별 training loss 시계열도 남지 않았다. 따라서 최종
30,000-step checkpoint는 검증 지표가 가장 좋아서 선택한 모델이 아니라 계획한
학습 schedule의 마지막 checkpoint다.

#### Pretrained config에서 상속된 값

최종 `train_config.json`에는 Python `SmolVLAConfig` 클래스의 새 인스턴스 기본값과
다른 값도 있다. 이는 직접 튜닝한 값이 아니라 `lerobot/smolvla_base` checkpoint
config를 로드하며 상속·해석된 값이다.

| 설정 | 클래스 기본값 | 최종 저장값 | 해석 |
|---|---:|---:|---|
| `load_vlm_weights` | `false` | `true` | Pretrained SmolVLA/VLM weight를 불러옴 |
| `prefix_length` | `-1` | `0` | Pretrained config에서 해석된 prefix 설정 |
| `pad_language_to` | `longest` | `max_length` | Base checkpoint의 language padding 방식 |
| `num_expert_layers` | `-1` | `0` | Base checkpoint에 저장된 expert layer 표현 |

이 값들은 이번 Piper 데이터에 맞춰 탐색하거나 변경한 하이퍼파라미터가 아니다.

## 6. BF16 mixed precision

LeRobot v0.4.4의 학습 loop는 Hugging Face Accelerate의 autocast를 사용한다.
현재 시스템에는 Accelerate 기본 설정 파일이 없으므로, 아무 설정도 하지 않으면
FP32로 실행된다.

RTX 5090에서는 FP16보다 표현 범위가 넓은 BF16을 사용한다.

```bash
ACCELERATE_MIXED_PRECISION=bf16
```

이 환경변수가 Accelerate에서 다음과 같이 인식되는 것을 확인했다.

```text
mixed_precision bf16
```

BF16은 주로 GPU activation 메모리를 줄인다. 모델의 일부 연산과 optimizer
상태는 계속 FP32를 사용할 수 있다.

최종 `train_config.json`의 `policy.use_amp=false`와 이 설정은 충돌하지 않는다.
이번 BF16은 policy 내부 AMP flag가 아니라 바깥 학습 loop의 Hugging Face
Accelerate autocast를 `ACCELERATE_MIXED_PRECISION=bf16`으로 활성화한 것이다.

## 7. 검증 완료된 smoke test

다음 조건의 smoke test가 완료되었다.

```text
batch_size=2
num_workers=2
steps=20
GPU=0
mixed precision=BF16
```

실행 명령:

```bash
conda activate ugrp
cd /home/ugrp43/UGRP/lerobot_robot_piper

CUDA_VISIBLE_DEVICES=0 \
ACCELERATE_MIXED_PRECISION=bf16 \
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.input_features=null \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/erase_the_shape_512 \
  --dataset.root=/home/ugrp43/UGRP/lerobot_robot_piper/records/0727/erase_the_shape_512 \
  --dataset.video_backend=pyav \
  --output_dir=/home/ugrp43/UGRP/lerobot_robot_piper/outputs/train/smolvla_erase_shape_512_smoke \
  --job_name=smolvla_erase_shape_512_smoke \
  --batch_size=2 \
  --num_workers=2 \
  --steps=20 \
  --log_freq=1 \
  --save_freq=20 \
  --eval_freq=0 \
  --policy.scheduler_warmup_steps=2 \
  --policy.scheduler_decay_steps=20 \
  --wandb.enable=false
```

검증 결과:

- 실제 training step 20 완료
- forward, loss, backward, optimizer update 완료
- scheduler step 20 완료
- checkpoint 저장 완료
- model weights 약 907MB
- optimizer state 약 413MB
- checkpoint 전체 약 1.23GiB

검증 후 smoke checkpoint는 용량 확보를 위해 삭제했다. 다음 경로는 실행 당시
사용한 위치이며 현재 보존 대상이 아니다.

```text
outputs/train/smolvla_erase_shape_512_smoke/checkpoints/000020
```

이 smoke checkpoint는 설치와 학습 경로 검증용이며, 실제 작업을 학습한 최종
policy로 사용하지 않는다.

## 8. Batch size와 epoch

공식 SmolVLA 예시는 A100 한 장에서 `batch_size=64`, `steps=20,000`을 사용한다.
이 경우 총 1,280,000개의 sample을 처리한다. 공식 문서의 약 4시간이라는 수치는
이 조건에 해당하므로 작은 batch의 step 시간과 직접 비교하면 안 된다.

현재 데이터셋은 30,545프레임이다.

| Batch size | 30,000 step의 처리 sample | 약 epoch |
|---:|---:|---:|
| 2 | 60,000 | 2.0 |
| 4 | 120,000 | 3.9 |
| 8 | 240,000 | 7.9 |

현재 하드웨어 가이드는 imitation learning에서 약 5~10 epoch를 시작점으로
제시한다. 따라서 현재 데이터셋에는 다음 설정을 우선 검토한다.

```text
batch_size=8
steps=30,000
약 7.9 epoch
```

이 설정으로 별도 output 경로에서 20-step smoke test를 수행했으며 정상
완료되었다.

## 9. Batch 8 사전검증 명령

```bash
conda activate ugrp
cd /home/ugrp43/UGRP/lerobot_robot_piper

CUDA_VISIBLE_DEVICES=0 \
ACCELERATE_MIXED_PRECISION=bf16 \
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.input_features=null \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/erase_the_shape_512 \
  --dataset.root=/home/ugrp43/UGRP/lerobot_robot_piper/records/0727/erase_the_shape_512 \
  --dataset.video_backend=pyav \
  --output_dir=/home/ugrp43/UGRP/lerobot_robot_piper/outputs/train/smolvla_erase_shape_512_smoke_b8 \
  --job_name=smolvla_erase_shape_512_smoke_b8 \
  --batch_size=8 \
  --num_workers=2 \
  --steps=20 \
  --log_freq=1 \
  --save_freq=20 \
  --eval_freq=0 \
  --policy.scheduler_warmup_steps=2 \
  --policy.scheduler_decay_steps=20 \
  --wandb.enable=false
```

확인할 항목:

- CUDA OOM이 발생하지 않음
- 20/20 step 완료
- loss가 finite 값임
- GPU peak VRAM
- 안정화된 `step/s` 또는 `update_s`
- `000020` checkpoint 생성

GPU와 RAM 확인:

```bash
watch -n 1 nvidia-smi
watch -n 1 free -h
```

검증 결과:

- batch 8로 20/20 step 완료
- `training_step.json`의 step 20 확인
- scheduler `last_epoch=20` 확인
- TOP/WRIST `[3, 512, 512]` 확인
- state/action 7차원 확인
- model, optimizer, scheduler 및 RNG state checkpoint 생성
- checkpoint 전체 약 1.23GiB
- peak VRAM: 3,567MiB / 32,607MiB
- peak GPU utilization: 48%
- 학습 중 최소 시스템 available RAM: 22,233MiB
- 첫 step 이후 `update_s`: 약 0.09~0.10초
- checkpoint 저장을 포함한 전체 속도: 약 1.8 step/s

VRAM 사용량은 충분히 낮았고 시스템 RAM도 여유가 있었다. 전체 step 속도는 GPU
연산보다 영상 DataLoader의 `data_s` 변동에 더 큰 영향을 받았다.

따라서 현재 시스템에서 batch 8 학습 경로는 정상 동작한다. 향후 다른 프로세스가
GPU 메모리를 사용해 OOM이 발생하면 해당 프로세스를 확인한 후 batch 4로 낮춘다.
batch 4로 5~10 epoch를 맞추려면 30,000 step보다 더 많은 step이 필요할 수 있다.

## 10. 본 학습 명령과 완료 결과

batch 8 smoke test가 통과했으므로 다음 명령을 본 학습 시작점으로 사용한다.

```bash
conda activate ugrp
cd /home/ugrp43/UGRP/lerobot_robot_piper

CUDA_VISIBLE_DEVICES=0 \
ACCELERATE_MIXED_PRECISION=bf16 \
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.input_features=null \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/erase_the_shape_512 \
  --dataset.root=/home/ugrp43/UGRP/lerobot_robot_piper/records/0727/erase_the_shape_512 \
  --dataset.video_backend=pyav \
  --output_dir=/home/ugrp43/UGRP/lerobot_robot_piper/outputs/train/smolvla_erase_shape_512 \
  --job_name=smolvla_erase_shape_512 \
  --batch_size=8 \
  --num_workers=2 \
  --steps=30000 \
  --log_freq=100 \
  --save_freq=5000 \
  --eval_freq=0 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=30000 \
  --wandb.enable=false
```

생성된 checkpoint:

```text
checkpoints/005000
checkpoints/010000
checkpoints/015000
checkpoints/020000
checkpoints/025000
checkpoints/030000
```

본 학습은 2026-07-28에 완료했다.

| 항목 | 결과 |
|---|---:|
| 최종 step | 30,000 |
| 총 학습 시간 | 4시간 1분 35초 |
| 평균 처리 속도 | 2.07 step/s |
| Batch size | 8 |
| 처리 sample | 240,000 |
| 데이터셋 기준 약 epoch | 7.9 |

최종 로그:

```text
Checkpoint policy after step 30000
Training: 100% | 30000/30000 [4:01:35, 2.07step/s]
End of training
```

최종 policy 경로:

```text
outputs/train/smolvla_erase_shape_512/checkpoints/030000/pretrained_model
```

`training_state/training_step.json`에서도 `step: 30000`을 확인했다. 공식 A100의
약 4시간 수치는 batch 64, 20,000 step 조건이므로 현재 결과와 직접 같은 조건은
아니지만, RTX 5090 한 장에서 현재 설정의 실측 시간도 약 4시간이었다.

### 보존된 학습 통계와 제한

현재 checkpoint에 보존된 정보:

- Model weight
- Optimizer state
- Scheduler state
- RNG state
- 5,000-step 간격 training step
- 전체 `train_config.json`
- 총 학습 시간과 전체 평균 step/s

보존되지 않은 정보:

- Step별 또는 시간별 training loss
- Gradient norm과 learning rate 시계열
- Update/DataLoader 시간 시계열
- Validation loss와 작업 성공률

학습 loop는 `log_freq=100`마다 loss, gradient norm, learning rate, update 시간과
data loading 시간을 터미널에 출력했다. 그러나 `wandb.enable=false`였고 stdout을
별도 파일로 저장하지 않아 이 시계열은 checkpoint에서 복구할 수 없다.

다음 학습에서는 W&B를 활성화하거나 터미널 출력을 파일로 함께 저장한다.

```bash
# W&B
--wandb.enable=true \
--wandb.project=lerobot-piper

# 또는 전체 stdout/stderr 보존
lerobot-train ... 2>&1 | tee outputs/train/<run_name>/training.log
```

현재 5,000-step 간격 checkpoint 6개에 동일한 고정 sample을 다시 입력해
checkpoint별 evaluation loss를 비교할 수는 있다. 이 값은 당시 mini-batch에서
기록됐어야 할 원래 training loss와는 다른 사후 평가값이다. 일반화 성능을
비교하려면 학습에 포함하지 않은 validation episode를 별도로 구성해야 한다.

## 11. torchvision 경고

학습 중 다음 경고가 표시될 수 있다.

```text
The video decoding and encoding capabilities of torchvision are deprecated
from version 0.22 and will be removed in version 0.24.
```

현재 `torchvision 0.22.1`과 LeRobot v0.4.4 조합에서는 학습 실패를 의미하지 않는다.
`num_workers=2`이면 각 worker가 경고를 출력해 같은 문장이 두 번 보일 수 있다.

경고를 없애기 위해 torchvision만 임의로 업그레이드하지 않는다. 향후 LeRobot
업그레이드 시 TorchCodec 경로를 별도로 검증한다.

## 12. 현재 완료 상태

- [x] 512×512 TOP/WRIST 크롭 데이터셋 생성
- [x] 60 episodes / 30,545 frames 확인
- [x] 두 MP4 전체 프레임 디코딩
- [x] state/action 7차원 확인
- [x] `input_features=null` 동작 확인
- [x] SmolVLA 의존성 및 RTX 5090 인식 확인
- [x] BF16 Accelerate 설정 확인
- [x] batch 2, 20-step smoke test 완료
- [x] batch 8, 20-step smoke test 완료
- [x] batch 8, 30,000-step 본 학습 완료
- [x] 5,000-step 간격 checkpoint 6개 생성
- [x] 최종 `030000/pretrained_model` 확인
- [x] 최종 checkpoint로 60개 episode 첫 chunk FK 분석
- [x] Dataset observation 기반 인간 승인형 RViz preview
- [x] 인간 승인형 실물 실행 경로 mock 검증
- [ ] 최종 checkpoint의 full offline rollout 재실행
- [ ] 인간 승인형 실행의 실물 Piper 검증
- [ ] 실제 로봇 inference 및 작업 성공률 평가
