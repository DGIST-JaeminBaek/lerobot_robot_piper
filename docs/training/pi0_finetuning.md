# pi0 파인튜닝 (LoRA)

`erase the shape` 태스크를 Physical Intelligence의 pi0로 파인튜닝한 기록. SmolVLA와
비교하기 위한 것이고, 2026-08-12에 처음 도입했다.

SmolVLA 쪽 절차는 `docs/training/method.md`와 `docs/training/smolvla_finetuning.md`를
볼 것. 이 문서는 **pi0에서만 다른 부분**에 집중한다.

## 요약

| 항목 | 값 |
|---|---|
| 베이스 | `lerobot/pi0_base` (3.5B, 14.0GB) |
| 방식 | **LoRA r=32, alpha=64** — 풀 파인튜닝은 VRAM 부족으로 불가 |
| 학습 파라미터 | 2.77M (전체의 0.084%) |
| 데이터 | `erase_the_shape_150` (150 에피소드, 75,787 프레임) |
| 스텝 | 75,000 (7.92 에폭) — SmolVLA 4전략과 동일 |
| 배치 | 8 — SmolVLA와 동일 |
| 소요 | **9.34시간** (2.27 step/s) |
| VRAM | 28.8~32.0 GB / 32.6 GB (**88~98%**) |
| 최종 loss | 0.081 (최소 0.032) |
| 체크포인트 | 32MB (LoRA 어댑터 11MB + optimizer 22MB) |

## 왜 LoRA인가 — 풀 파인튜닝은 불가능

OpenPI 공식 문서 기준 요구 VRAM:

| 방식 | VRAM | 예시 GPU |
|---|---|---|
| Full fine-tuning (원저자 기본) | **> 70 GB** | A100 80GB / H100 |
| LoRA | > 22.5 GB | RTX 4090 |

우리는 RTX 5090 **32GB**다. 직접 계산해도 3.5B를 AdamW로 풀 파인튜닝하면
모델 6.6GB + 그래디언트 6.6GB + optimizer 상태 26.4GB + fp32 마스터 13.2GB ≈ **55~60GB**라
한 장에 안 들어간다.

**GPU 2장을 써도 안 된다.** lerobot v0.4.4는 `DistributedDataParallelKwargs`만 쓰고
(`lerobot_train.py:177-186`) FSDP/DeepSpeed 배선이 없다. DDP는 각 GPU에 모델을 통째로
복제하므로 카드당 요구 메모리가 그대로다. 게다가 두 GPU가 NVLink가 아니라 PCIe(PHB)
연결이라 설령 샤딩을 붙여도 통신이 병목이 된다.

논문에 쓸 때는 이 제약을 명시할 것: **"단일 32GB GPU 제약으로 pi0는 LoRA(r=32)로
파인튜닝했다. 풀 파인튜닝은 원저자 문서 기준 70GB 이상을 요구한다."**

### rank는 왜 32인가

r=8로 낮춰도 **VRAM 절약이 수십 MB뿐**이다. 28.8GB의 대부분은 모델 가중치와 순전파
활성값이고 LoRA 파라미터는 0.084%에 불과하다. 반면 SmolVLA는 전략1·2가 전체의 22%,
HAMLET(전략3·4)이 27%를 학습하므로 이미 260배 차이가 난다. 여기서 더 줄이면 pi0가
불리한 조건에서 비교당하게 되어 "아키텍처가 나쁜지 용량이 부족한지" 구분이 안 된다.

`lora_alpha=64`는 관례인 `2×r`이다.

## 환경 분리 — `ugrp`가 아니라 `pi0`

**pi0는 별도 conda 환경이 필수다.**

lerobot의 `[pi]` extra가 transformers를 특정 git 브랜치로 교체한다:

```toml
pi = ["transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi", "scipy>=1.10.1,<1.15"]
```

`modeling_pi0.py:580-587`이 표준 transformers에 없는 모듈을 요구하기 때문이다:

```python
from transformers.models.siglip import check
if not check.check_whether_transformers_replace_is_installed_correctly():
    raise ValueError("An incorrect transformer version is used, ...")
```

이 브랜치를 `ugrp`에 설치하면 SmolVLA/HAMLET이 깨진다. lerobot 자신도 충돌을 알아서
`all` extra에서 pi를 주석 처리해뒀다.

| 환경 | transformers | 용도 |
|---|---|---|
| `ugrp` | 4.57.6 | SmolVLA, HAMLET, 데이터 분석, 실물 제어 |
| `pi0` | **4.53.3** (git 브랜치) | pi0 전용 |

### 구축 절차

```bash
conda create -n pi0 --clone ugrp        # 약 8.4GB
conda activate pi0
pip install -e "/home/ugrp43/UGRP/piper_sdk/lerobot[pi]"
```

**`[pi,peft]`로 같이 설치하면 실패한다** — `[peft]`가 `transformers>=4.57.1`을 요구해
`[pi]`의 4.53.3과 충돌한다. `peft`는 복제 환경에 이미 0.20.0이 들어 있고 transformers
버전 제약이 없으므로 `[pi]`만 설치하면 된다.

확인:

```bash
python -c "
from transformers.models.siglip import check
print('siglip check:', check.check_whether_transformers_replace_is_installed_correctly())
"
# siglip check: True 여야 한다
```

lerobot은 editable 설치라 **두 환경이 같은 소스 디렉터리를 공유한다**. pi0 실험 중
`/home/ugrp43/UGRP/piper_sdk/lerobot` 소스를 고치면 `ugrp`에도 영향이 간다.

## 준비 단계에서 막히는 것 세 가지

### 1. PaliGemma 토크나이저가 gated repo

pi0의 전처리 파이프라인이 `google/paligemma-3b-pt-224`의 토크나이저를 쓰는데, 이
저장소는 Google이 수동 승인하는 gated repo다(`gated=manual`).

- https://huggingface.co/google/paligemma-3b-pt-224 에서 라이선스 동의 (즉시 승인)
- `huggingface-cli login` 으로 Read 토큰 등록 ("Add token as git credential?"은 `n`으로 충분)

**받는 건 토크나이저 22MB뿐이다.** PaliGemma 가중치 11.7GB는 이미 pi0의 14GB 안에
변환되어 들어 있으므로 따로 받지 않는다.

라이선스는 **Gemma Terms of Use** — 상업적 이용·수정·재배포 허용이지만 Prohibited Use
Policy를 따라야 하고, OSI 승인 오픈소스는 아니다. 파인튜닝 모델을 공개 배포하려면
약관을 함께 배포해야 한다. 동의 철회 기능은 없다(모델 작성자만 접근을 끊을 수 있다).

### 2. HF 저장소 최신 리비전이 v0.4.4와 호환되지 않는다

`lerobot/pi0_base`의 2026-06-03 커밋 **"Add relative action processor steps (#6)"** 이
전처리 파이프라인에 `relative_actions_processor` 단계를 추가했는데, 이 클래스는
lerobot v0.4.4에 **존재하지 않는다**. 설정상 `"enabled": false`로 꺼져 있는데도
파이프라인 구성 시 레지스트리에서 클래스를 먼저 찾으므로 거기서 죽는다.

**해결: 그 직전 리비전을 쓴다.**

```bash
python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('lerobot/pi0_base', revision='26b99b9439'))
"
```

모델 가중치는 그 이후 커밋에서 안 바뀌었으므로 **같은 블롭을 재사용해 재다운로드가
없다**. 이 리비전의 전처리 단계는 전부 v0.4.4가 아는 것들이다:

```
rename_observations_processor / to_batch_processor / pi0_new_line_processor
tokenizer_processor / device_processor / normalizer_processor
```

`--policy.path`에 HF repo id 대신 이 스냅샷 경로를 준다:

```
/home/ugrp43/.cache/huggingface/hub/models--lerobot--pi0_base/snapshots/26b99b9439acb1e352439e34ee9c67af0d76efa3
```

CLI에 `--revision` 옵션은 없다.

**참고**: `relative_actions_processor`는 액션을 절대 위치 대신 "추론 시점 상태로부터의
오프셋"으로 바꾸는 단계다(UMI, Chi et al. 2024). 정규화가 쉬워지고 chunk 간 오차 누적이
없다는 장점이 있지만, 배포된 설정이 `enabled: false`라 **쓰지 않는 것이 기본값**이고
따라서 이전 리비전을 써도 동작이 동일하다. 쓰려면 lerobot 최신 버전과 데이터셋 통계
재계산이 필요하다.

### 3. 카메라 이름이 다르다

pi0는 카메라 이름이 고정이다:

```
observation.images.base_0_rgb
observation.images.left_wrist_0_rgb
observation.images.right_wrist_0_rgb
```

우리는 `top` / `wrist`다. `--rename_map`으로 매핑한다:

```bash
--rename_map='{"observation.images.top": "observation.images.base_0_rgb", "observation.images.wrist": "observation.images.left_wrist_0_rgb"}'
```

**세 번째 카메라는 안 줘도 된다.** `modeling_pi0.py:1210-1214`가 없는 카메라를 자동으로
채운다:

```python
for _num_empty_cameras in range(len(missing_img_keys)):
    img = torch.ones_like(img) * -1   # SigLIP 정규화 범위의 최솟값
    mask = torch.zeros_like(mask)     # 마스크 0 → 어텐션에서 무시
```

마스크가 0이므로 모델이 검은 화면을 실제 관찰로 착각하지 않는다.

해상도도 신경 쓸 필요 없다. pi0는 224×224를 쓰지만 내부에서 `resize_with_pad`로
알아서 줄이므로 512×512 데이터셋을 그대로 쓴다.

## 차원 패딩 — state/action이 32차원

pi0는 여러 로봇을 한 모델로 다루려고 `max_state_dim = max_action_dim = 32`로 잡혀 있다.
우리 Piper는 7 DOF(joint1~6 + gripper)인데 **아무 설정 없이 자동 처리된다**:

```
데이터셋 7차원
  → 정규화 (통계도 7차원)
  → prepare_state/prepare_action: pad_vector(x, 32)  — 뒤에 0을 25개
  → 32차원으로 트랜스포머 통과
  → sample_actions: actions[:, :, :7]                 — 원래 차원만 잘라 반환
  → losses[:, :, :7]                                  — 패딩 부분은 loss에서 제외
```

정규화가 패딩보다 **먼저**이므로 패딩된 차원은 정규화 공간에서 정확히 0(=평균)이다.
loss에도 안 들어가므로 "0을 예측하도록" 학습되지 않는다.

체크포인트 `config.json`이 비대칭으로 보이는 이유도 이것이다:

```
observation.state  [32]   ← 패딩 후 크기 (모델 입력)
action             [7]    ← 원래 크기 (출력에서 자를 기준)
```

## 학습 명령어

```bash
conda activate pi0

CUDA_VISIBLE_DEVICES=0 \
lerobot-train \
  --policy.path=/home/ugrp43/.cache/huggingface/hub/models--lerobot--pi0_base/snapshots/26b99b9439acb1e352439e34ee9c67af0d76efa3 \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.push_to_hub=false \
  --peft.method_type=LORA \
  --peft.r=32 \
  --peft.lora_alpha=64 \
  --dataset.repo_id=local/erase_the_shape_150 \
  --dataset.root=/home/ugrp43/UGRP/lerobot_robot_piper/records/outputs/erase_the_shape_150 \
  --dataset.video_backend=pyav \
  --rename_map='{"observation.images.top": "observation.images.base_0_rgb", "observation.images.wrist": "observation.images.left_wrist_0_rgb"}' \
  --output_dir=/home/ugrp43/UGRP/lerobot_robot_piper/outputs/train/pi0_lora_topwrist_75k \
  --job_name=pi0_lora_topwrist_75k \
  --batch_size=8 \
  --num_workers=2 \
  --steps=75000 \
  --log_freq=100 \
  --save_freq=15000 \
  --eval_freq=0 \
  --wandb.enable=true \
  --wandb.project=smolvla-erase-shape \
  --wandb.run_id=pi0_lora_topwrist_75k \
  --wandb.disable_artifact=true
```

**주의: 이 명령어는 실제로 돌렸던 원본 그대로 남겨둔 기록이고, `--policy.scheduler_decay_steps`가
빠져 있다.** 이 상태로 돌리면 기본값 30000에 걸려 75,000 step 중 60%가 LR floor 상태로
학습된다(아래 "LR 스케줄러" 절 참고). **새로 pi0를 학습할 때는 반드시
`--policy.scheduler_decay_steps=<--steps와 같은 값>`을 추가할 것.**

SmolVLA와 다른 점만 짚으면:

- **`--policy.dtype=bfloat16`** — pi0 기본값이 `float32`라 그냥 두면 VRAM이 2배 든다
  (근거는 아래 "BF16 정밀도" 절 참고)
- **`--peft.*`** — LoRA. `target_modules`는 안 줘도 된다(`modeling_pi0.py:1301`에
  `gemma_expert`의 q/v_proj + action/state projection이 기본값으로 정의돼 있음)
- **`--rename_map`** — 위 참고
- **`--wandb.disable_artifact=true`** — 체크포인트를 W&B에 올리지 않는다. 이걸 빼면
  `~/.cache/wandb/artifacts`에 캐시가 쌓인다(SmolVLA 학습 때 29GB까지 쌓인 적 있음)
- **`--policy.freeze_vision_encoder` / `--policy.train_expert_only` 없음** — LoRA라
  어차피 어댑터만 학습하므로 불필요

## BF16 정밀도

### `--policy.dtype`의 실제 기본값은 `float32`다

`PI0Config.dtype`의 클래스 기본값은 `"float32"`다
(`configuration_pi0.py:34`, `dtype: str = "float32"  # Options: "bfloat16", "float32"`).
그냥 두면 pi0(3.5B)를 fp32로 올리게 되어 VRAM이 두 배로 들고, 32GB 카드 한 장에서는
LoRA를 써도 못 들어간다. 그래서 `--policy.dtype=bfloat16`을 **명시적으로** 줘야 한다.

`PaliGemmaWithExpertModel.__init__`의 `precision` 파라미터 자체는 기본값이
`"bfloat16"`이지만(`modeling_pi0.py:340`), 실제로는 이 값이 쓰이지 않는다 —
`PI0Policy`가 이 클래스를 만들 때 `precision=config.dtype`으로 **config 값을 그대로
전달**하기 때문이다(`modeling_pi0.py:556`). 즉 클래스 시그니처만 보고 "pi0는 기본이
bf16"이라고 판단하면 틀린다. 우리가 실제로 쓰는 `PI0Config` 경로의 기본값은 float32다.

### bf16이어도 일부 파라미터는 fp32로 유지된다

`--policy.dtype=bfloat16`을 줘도 다음은 자동으로 fp32에 남는다
(`to_bfloat16_for_selected_params()`, `modeling_pi0.py:392-412`):

```
vision_tower.vision_model.embeddings.patch_embedding.weight/bias
vision_tower.vision_model.embeddings.position_embedding.weight
input_layernorm / post_attention_layernorm / model.norm  (모든 LayerNorm)
```

이름에 이 문자열이 들어간 파라미터는 `.to(bfloat16)` 이후에 다시 `.to(float32)`로
덮어써진다. Vision patch/position embedding과 모든 정규화 레이어는 정밀도에 민감해서
원저자가 의도적으로 남겨둔 것 — 우리가 조정한 값이 아니라 pi0/OpenPI 자체 구현이다.

### 공식 근거 — bf16이 loss는 더 높다

OpenPI 공식 저장소(https://github.com/Physical-Intelligence/openpi) 문서:

> The system supports either **full bfloat16 (default) or full float32** [...]
> **Mixed precision is not yet supported.** [...] bfloat16 uses less memory but
> **exhibits higher losses compared to float32**.

즉 원저자들 스스로 "bf16이 float32보다 loss가 높다"고 명시한다. JAX 구현은 weight/grad는
fp32, activation만 bf16으로 섞는 진짜 mixed precision을 기본으로 쓰지만, PyTorch
구현(우리가 쓰는 lerobot 포트의 기반)은 그 기능이 없어 **전체를 bf16 아니면 전체를
fp32 중 하나로 선택**해야 한다. 우리가 bf16을 택한 이유는 SmolVLA와 동일하게 32GB VRAM
제약 때문이고(§"왜 LoRA인가" 참고), loss가 float32보다 다소 높게 나올 수 있다는 걸
감안해야 한다 — 실측 최종 loss 0.081(요약 표 참고)을 다른 정밀도 설정과 직접 비교할 때
이 차이를 고려할 것.

SmolVLA의 bf16 근거는 `docs/training/smolvla_finetuning.md`의 "BF16 mixed precision"
절 참고 — 그쪽은 원저자가 애초에 bf16 + Accelerate mixed precision을 학습 레시피로
채택했고(SmolVLA 논문 4.3절), pi0처럼 "float32가 기본, bf16을 명시로 켜야 함" 구조가
아니라는 점이 다르다.

## 실측값

### 속도

| | step/s | 75,000 step |
|---|---|---|
| SmolVLA 전략1 (top+wrist, bs=8) | 10.06 | 2.07시간 |
| **pi0 LoRA (bs=8)** | **2.27** | **9.34시간** |

pi0가 **4.4배 느리다**. `data_s: 0.003` / `updt_s: 0.435`로 데이터 로딩은 병목이
아니다(GOP=1 데이터셋 덕분).

### 메모리

| | 값 |
|---|---|
| VRAM (학습 중) | 28.8~32.0 GB / 32.6 GB |
| 시스템 RAM (학습 정상 상태) | 8.4 GB |
| **시스템 RAM (모델 로딩 피크)** | **21.7 GB** |

로딩 피크가 정상 상태의 2.6배다. 14GB safetensors를 읽고 777개 키를 리매핑하면서
사본이 생기기 때문. 시스템 RAM 30GB에서 18.85GB까지 찼으므로 여유가 11GB뿐이었다.
**pi0급 모델 두 개를 동시에 로드하려면(약 44GB) RAM이 부족하다.**

### 온도·전력 (전력 제한 500W 설정 시)

```
온도 76~77°C, 사용률 98%, 499W/500W, 클럭 2737/3105 MHz
SW Power Cap: Active / HW Thermal Slowdown: Not Active
```

열 쓰로틀링 없이 설정한 전력 상한에만 걸린 정상 상태. 9시간 연속 완주.

## 추론

### 스크립트 수정이 필요했다

`piper_infer_runner.py`와 `piper_human_approved_inference.py`는 SmolVLA 전제로 짜여
있어서 두 군데를 고쳐야 했다 (2026-08-12, 백업 `tmp/inference_backup_20260812/`).

**1. `piper_offline_chunk_rollout.py::load_policy`**

`make_policy`에 `rename_map`을 안 넘겨서 카메라 이름 불일치로 죽었다. 이제 체크포인트의
`policy_preprocessor.json`에서 학습 때 쓴 매핑을 자동으로 꺼내 넘긴다. 그래서 **추론
명령어에는 `--rename_map`을 줄 필요가 없다.**

**2. `piper_infer_runner.py::check_policy_dataset_match`**

정책 로딩 전에 도는 자체 사전 검증이 두 가지를 오판했다:
- 카메라 이름 → `saved_rename_map()`으로 매핑 후 비교하도록 수정
- `observation.state` 32 vs 7 → `pads_vectors()`로 "정책 쪽이 더 크면 패딩"으로 인정

### 추론 명령어

**SmolVLA와 동일하고 `conda activate pi0`만 다르다.**

```bash
source /opt/ros/humble/setup.bash
source ~/UGRP/ros2_ws/install/setup.bash
conda activate pi0

python scripts/tools/piper_infer_runner.py \
  --dataset-root records/outputs/erase_the_shape_150 \
  --policy-path outputs/train/pi0_lora_topwrist_75k/checkpoints/last/pretrained_model \
  --source robot \
  --apply-to-robot \
  --real-robot-confirm I_UNDERSTAND_REAL_ROBOT \
  --mode demo \
  --aggregate-fn latest_only
```

데이터셋 재생(RViz만, 실물 없음)은 `--source dataset --episode 0`으로 바꾸면 된다.

### 추론 지연과 chunk_threshold

| | 추론 지연 | 30Hz 기준 |
|---|---|---|
| SmolVLA | ~115 ms | 3.5 스텝 |
| **pi0** | **~190 ms** | **5.5 스텝** |

첫 추론만 608~744ms로 느리고(콜드스타트) 이후엔 190ms대로 안정된다.

`--chunk-threshold`를 0.3/0.5/0.7/0.9로 스윕한 결과 (dataset 소스, 300 step):

| threshold | 큐 마름 | 최소 pending | votes 평균 | observe |
|---|---|---|---|---|
| **0.3** (기본) | 0회 | 9 | 1.24 | 14 ms |
| 0.5 | 0회 | 19 | 1.70 | 15 ms |
| 0.7 | 0회 | 29 | 2.75 | 17 ms |
| 0.9 | 0회 | 36 | 6.14 | **24 ms** |

**어느 값에서도 큐가 마르지 않는다.** 오히려 높이면 추론 요청이 잦아져 GPU가 바빠지고
`observe`까지 밀린다(0.9에서 14→24ms). `latest_only`를 쓰면 votes가 많아도 의미가 없다.

실물 카메라를 붙이면 `observe`가 33.4ms가 된다(RealSense 2대 병렬 읽기 31.8ms +
크롭/리사이즈 1.6ms). 30fps 카메라의 프레임 간격 33.3ms에 묶인 값이라 더 줄일 수 없다.
다만 루프가 느려지면 같은 시간에 소비하는 큐 스텝도 줄어들어 상대적 여유는 유지된다.

**결론: `chunk_threshold`는 기본값 0.3을 유지한다.** SmolVLA와 조건이 같아 비교에도
유리하다.

## LR 스케줄러 — `scheduler_decay_steps`가 `--steps`를 안 따라간다

### 증상

`PI0Config.scheduler_decay_steps`의 기본값은 `30_000`이다(`configuration_pi0.py:94`).
`--steps`를 그보다 길게 잡아도 이 값은 저절로 늘어나지 않는다 — cosine decay
스케줄러가 30,000 step 지점에서 이미 floor LR(`2.5e-6`)에 도달해버리고, 남은 학습
구간 전체가 사실상 거의 업데이트 없이 흘러간다.

`CosineDecayWithWarmupSchedulerConfig.build()`(`schedulers.py:94-109`)는 **줄이는 것만**
자동으로 한다: `num_training_steps < num_decay_steps`일 때만 비율에 맞춰 축소한다.
반대 방향(`--steps`가 `scheduler_decay_steps`보다 길 때 늘려주는 것)은 없다.
`configuration_pi0.py:91`의 주석 "auto-scale if --steps < scheduler_decay_steps"도
이 방향성만 다룬다는 걸 명시한다.

### 틀리기 쉬운 지점 — `--scheduler.*`는 조용히 버려진다

`lerobot-train`의 `TrainPipelineConfig`엔 top-level `--scheduler.*` 필드도 있어서
draccus가 파싱은 해준다. 하지만 `use_policy_training_preset=True`(기본값)일 때
`__post_init__`이 무조건 정책 preset으로 덮어쓴다(`configs/train.py:134-136`,
`self.scheduler = self.policy.get_scheduler_preset()`). 즉 `--scheduler.num_decay_steps`를
줘도 실행 직후 버려지고 정책 config의 `scheduler_decay_steps`(기본 30000)가 그대로
쓰인다. **반드시 `--policy.scheduler_decay_steps`(정책 config 네임스페이스)로 줘야 한다.**

### 실측 — pi0 체크포인트 5개 중 4개가 이 문제에 걸려 있다

저장된 `checkpoints/last/pretrained_model/train_config.json`을 직접 확인한 결과:

| 체크포인트 | steps | `policy.scheduler_decay_steps` | 상태 |
|---|---:|---:|---|
| `pi0_lora_topwrist_75k` | 75,000 | 미지정(기본값 30000 추정) | 고장 |
| `pi0_lora_topwrist_315` | 168,000 | 30000 (`--scheduler.num_decay_steps`로 줬지만 무시됨) | 고장 |
| `pi0_lora_pickup_prompt_132` | 65,265 | 30000 (기본값 그대로) | 고장 |
| `pi0_lora_pickup_prompt_132_decayfix` | 65,265 | 30000 (여전히 `--scheduler.*`로 줌 — 첫 수정 시도 실패) | 고장 |
| **`pi0_lora_pickup_prompt_132_decayfix3`** | 65,265 | **65265** (`--policy.scheduler_decay_steps=65265`로 정확히 줌) | **정상** |

즉 `pi0_lora_pickup_prompt_132_decayfix3`가 유일하게 전체 학습 구간에서 LR이 정상적으로
decay한 pi0 체크포인트다. **실물 테스트·비교 실험에는 이 체크포인트를 쓸 것.** 나머지
4개는 학습 후반 17.9%~54%가 LR floor 근처에서 거의 헛돈 상태다.

### 앞으로 pi0(및 SmolVLA) 학습 명령을 짤 때 체크리스트

1. `--steps` 값을 정한다
2. `--policy.scheduler_decay_steps`를 **정확히 같은 값**으로 명시한다 (`--scheduler.*` 아님)
3. 저장된 `train_config.json`의 `policy.scheduler_decay_steps` 값이 실제로 그 값인지
   재확인한다 — CLI에 값을 줬다고 믿지 말 것

이 함정은 pi0 전용이 아니라 lerobot 스케줄러 설계 자체의 공통 문제라 SmolVLA에도
동일하게 적용된다. SmolVLA 쪽 근거와 외부 사례(GitHub issue #3287)는
`docs/training/smolvla_finetuning.md`의 "BF16 mixed precision" 절 말미에 정리돼 있다.

## 비교 실험에서 주의할 점

SmolVLA 4전략과 pi0는 **학습 방식이 다르다**:

| 모델 | 방식 | 학습 파라미터 |
|---|---|---|
| SmolVLA 전략1·2 | `freeze_vision_encoder` + `train_expert_only` | 100M / 450M (**22%**) |
| SmolVLA 전략3·4 (HAMLET) | 위 + 메모리 모듈 | 129M / 480M (**27%**) |
| pi0 | **LoRA r=32** | 2.77M / 3.5B (**0.084%**) |

결과를 "pi0 vs SmolVLA 아키텍처 비교"로 해석하면 안 된다. **"각 모델을 32GB VRAM 제약
안에서 최선의 방식으로 학습했을 때의 비교"** 정도가 정직한 프레이밍이다.

맞추려면 SmolVLA도 LoRA로 다시 학습하거나 pi0의 rank를 크게 올려야 하는데, 전자는
4전략을 다 다시 돌려야 하고 후자도 여전히 다른 메커니즘이다.

## 알려진 잔여 이슈

- **`loss_per_dim`이 W&B에 안 올라간다.** pi0가 관절별 loss를 리스트로 내보내는데
  (`modeling_pi0.py:1287`) lerobot의 W&B 래퍼가 스칼라만 처리한다(`wandb_utils.py:180`).
  학습엔 영향 없지만 "어느 관절 예측이 유독 안 맞나"를 볼 수 없다. 필요하면 학습된
  체크포인트로 데이터셋을 한 번 돌려 `forward()`의 `loss_dict`를 직접 받으면 된다.

- **embedding 키 경고는 무시해도 된다 (검증 완료)** — 로딩 시
  `Missing key(s): ...language_model.embed_tokens.weight` 경고가 뜨지만 실제로는
  정상 로드된다. 직접 확인한 내용:

  | 확인 | 결과 |
  |---|---|
  | 체크포인트에 `embed_tokens.weight` | 없음 |
  | 체크포인트에 `lm_head.weight` | 있음 |
  | `tie_word_embeddings` | `True` |
  | 두 텐서가 같은 객체 (`data_ptr()`) | `True` |
  | 값 동일 (`torch.equal`) | `True` |
  | 가중치 분포 | std 0.184, absmax 10.87 — 랜덤 초기화 아님 |

  PaliGemma는 입력 임베딩과 출력 헤드가 **같은 텐서를 공유**하므로 저장 시 하나만
  넣는다. `lm_head.weight`가 채워지면 `embed_tokens.weight`도 자동으로 같은 값이 된다.
  경고는 lerobot의 리매핑 코드가 `strict` 검사에서 키 부재를 보고한 것뿐이다.

- **경고 두 종류는 무시해도 된다**:
  `huggingface/tokenizers: The current process just got forked...` (에폭 경계마다 워커가
  재생성되며 발생, 데드락 예방 동작), `torchvision ... deprecated` (pyav를 쓰므로 무관).
