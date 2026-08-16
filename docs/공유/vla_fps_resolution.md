# 대표 VLA 모델의 FPS / 이미지 해상도 조사

> 담당: 조성일 · 작성 2026-07-25
>
>
> **출처 표기 원칙**
> - 🟢 **코드 검증** — 랩/개인 PC에 설치된 `lerobot 0.4.4` 소스에서 직접 확인. 파일 위치 명시
> - 🟡 **문헌** — 논문·공식 블로그·저장소 기준
> - ⚪ **미확인** — 찾지 못했거나 출처가 불확실. 그대로 인용하지 말 것

---

## 1. 왜 이걸 조사하나

우리가 지금 정해야 하는 값이 둘이다.

```
FPS          = 30  (녹화 프레임레이트)
해상도        = 640×480  vs  1280×720   ← 팀 내 미결정
```

이 값들은 **첫 에피소드 전에 확정해야 하고 수집 끝까지 못 바꾼다** (`features` 불일치 시
데이터셋 머지 불가). 그런데 "모델이 실제로 뭘 먹는지"를 모르면 근거 없이 정하게 된다.

**결론부터**: 대부분의 VLA는 입력 이미지를 **224~512px로 강제 축소**한다.
고해상도로 찍어도 모델은 그걸 안 쓴다 (§4).

---

## 2. 모델별 정리

### 2-1. SmolVLA — 우리가 쓰는 모델 🟢

`lerobot/policies/smolvla/configuration_smolvla.py`

| 항목 | 값 | 출처 |
|---|---|---|
| 이미지 입력 | **512 × 512** (`resize_imgs_with_padding`) | 🟢 L48 |
| 리사이즈 방식 | `resize_with_pad` — 종횡비 유지 + 패딩 | 🟢 `modeling_smolvla.py` L135, L419 |
| VLM 백본 | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | 🟢 L87 |
| `chunk_size` | 50 | 🟢 L32 |
| `n_action_steps` | 50 | 🟢 L33 |
| `max_state_dim` | 32 | 🟢 L44 |
| `freeze_vision_encoder` | True (기본) | 🟢 L72 |
| `train_state_proj` | True (기본) | 🟢 L74 |
| flow matching steps | 10 (`num_steps`) | 🟢 L66 |

**핵심**: 어떤 해상도로 찍든 `resize_with_pad`로 **512×512에 맞춰진다.**

### 2-2. π0 (Pi-Zero) 🟢🟡

`lerobot/policies/pi0/configuration_pi0.py`

| 항목 | 값 | 출처 |
|---|---|---|
| `chunk_size` / `n_action_steps` | 50 / 50 | 🟢 L37–38 |
| `max_state_dim` / `max_action_dim` | 32 / 32 | 🟢 L41–42 |
| `freeze_vision_encoder` | **False** (SmolVLA와 반대) | 🟢 L80 |
| `tokenizer_max_length` | 48 | 🟢 L97 |
| VLM 백본 | PaliGemma (3B) | 🟢 `modeling_pi0.py` L36 |
| 액션 전문가 | 0.315B, flow matching | 🟡 |
| 액션 출력 | **50 Hz로 50스텝 청크** | 🟡 |
| 추론 시간 | RTX GPU에서 **73 ms** (카메라 3대, 50액션 청크) | 🟡 |

### 2-3. π0.5 / π0-FAST 🟢🟡

| 항목 | π0.5 | π0-FAST | 출처 |
|---|---|---|---|
| `chunk_size` / `n_action_steps` | 50 / 50 | 50 / 50 | 🟢 |
| `max_state_dim` | 32 | 32 | 🟢 |
| `tokenizer_max_length` | **200** | **200** | 🟢 |
| 액션 표현 | FAST 토크나이저로 이산 토큰 → 사후학습에 flow matching | 🟡 |

π0.5는 π0 기반에 계층 구조를 얹은 것. `tokenizer_max_length`가 48 → 200으로 커진 게
코드에서 보이는 가장 뚜렷한 차이다(더 긴 언어 지시 처리). 🟢

### 2-4. GR00T 🟢

`lerobot/policies/groot/`

| 항목 | 값 | 출처 |
|---|---|---|
| **`image_size`** | **(224, 224)** | 🟢 L53 |
| `chunk_size` / `n_action_steps` | 50 / 50 | 🟢 L33–34 |
| `max_state_dim` | **64** (다른 모델은 32) | 🟢 L38 |
| `max_action_dim` | 32 | 🟢 L41 |

### 2-5. OpenVLA 🟡

| 항목 | 값 |
|---|---|
| 이미지 입력 | **224 × 224** RGB |
| 비전 인코더 | DINOv2 + SigLIP **융합** (prism-dinosiglip-224px) |
| LLM 백본 | Llama-2 **7B** |
| 액션 공간 | 7-DoF (xyz + rpy + 그리퍼) |
| **제어 주파수 (기본)** | **3~5 Hz** — 자기회귀 디코딩이 memory-bound |
| **제어 주파수 (OFT)** | **~25 Hz** — 병렬 디코딩 + 액션 청킹 |

**OpenVLA vs OpenVLA-OFT 구분이 중요하다.** 기본 OpenVLA는 액션 차원을 하나씩
순차 생성해서 느리다. OFT는 길이 K의 빈 액션 임베딩에 양방향 어텐션을 걸어
**한 번의 forward로 K개 액션을 동시 예측**한다 → 액션 생성 26배, 지연 3배 개선. 🟡

> 부드러운 로봇 제어에는 **20~30 Hz**가 필요한데 기본 VLA들은 엣지에서 3~5 Hz에
> 머문다는 게 이 분야의 공통 문제의식이다. 🟡

### 2-6. RT-2 🟡⚪

| 항목 | 값 | 출처 |
|---|---|---|
| VLM 백본 | PaLI-X (5B / **55B**), PaLM-E (12B) | 🟡 |
| 제어 주파수 | **55B: 1~3 Hz** (멀티 TPU + 네트워크), **5B: 5 Hz** | 🟡 |
| 이미지 해상도 | ⚪ **확인 못 함** | ⚪ |

RT-2는 클라우드 TPU에 올리고 로봇과 네트워크로 연결하는 구조라 우리 환경과
직접 비교가 어렵다. 참고용.

---

## 3. 한눈에 보기

| 모델 | 이미지 입력 | chunk / 실행 | 제어 주파수 | 백본 |
|---|---|---|---|---|
| **SmolVLA** 🟢 | **512×512** | 50 / 50 | ⚪ | SmolVLM2-500M |
| π0 🟢🟡 | ⚪ (코드에 없음) | 50 / 50 | 50 Hz 출력 | PaliGemma 3B |
| π0.5 🟢🟡 | ⚪ | 50 / 50 | ⚪ | PaliGemma 기반 |
| GR00T 🟢 | **224×224** | 50 / 50 | ⚪ | — |
| OpenVLA 🟡 | **224×224** | 1 (기본) | 3~5 Hz | Llama-2 7B |
| OpenVLA-OFT 🟡 | 224×224 | K (청킹) | ~25 Hz | Llama-2 7B |
| RT-2 🟡 | ⚪ | — | 1~5 Hz | PaLI-X 55B / PaLM-E 12B |

**공통점 두 가지가 뚜렷하다:**

1. **이미지는 224~512px.** 어느 모델도 720p를 그대로 안 먹는다.
2. **`chunk_size=50`, `n_action_steps=50`이 lerobot 구현 전반의 기본값**
   (SmolVLA, π0, π0.5, π0-FAST, GR00T 전부). 🟢

---

## 4. ⭐ 우리 프로젝트에 대한 함의

### 4-1. 해상도 — 720p는 SmolVLA에서 대부분 버려진다

**레포 현황 (2026-07-25 확인)**

| 출처 | 값 | 비고 |
|---|---|---|
| `configs/recording.env.example` (추적 파일) | **640×480 (4:3)**, FPS 30 | 레포 기본값 |
| `docs/change.md:79` | `cam_width: int = 640` | config 기본값 기록 |
| jmbaek `docs/depth/README.md` §9 | **1280×720 (16:9)** | **depth 검증 전용 설정** |
| `docs/roadmap.md:73` | "해상도/FPS/warmup 값 확정" | ⚠️ **아직 미완료 TODO** |

즉 **RGB 기본값은 4:3(640×480)이고, 1280×720은 jmbaek이 depth를 검증할 때 쓴
설정이다.** 그리고 해상도 확정은 로드맵상 아직 열린 항목이다.
(과거엔 424×240(16:9)이었다가 640×480으로 바뀐 이력이 있다.)


우리 팀 미결정 사항(640×480 vs 1280×720)에 대한 직접적 답이다.

`modeling_smolvla.py`의 `resize_with_pad`를 실제로 읽어보면:

```python
ratio = max(cur_width / width, cur_height / height)   # 긴 쪽 기준 축소 → 나머지는 패딩
```

즉 **종횡비를 유지한 채 512×512 안에 넣고 남는 부분은 검은 여백**이다. 계산하면:

| 원본 | 종횡비 | 512×512 안 실제 크기 | 내용 비율 | 검은 여백 |
|---|---|---|---|---|
| **640×480** | 4:3 | **512×384** | **75 %** | 25 % |
| **1280×720** | 16:9 | **512×288** | **56 %** | **44 %** |

**16:9는 모델 입력의 44 %가 검은 띠다.** SigLIP은 512×512 전체를 패치로 쪼개 처리하므로
그만큼이 정보 없는 토큰에 쓰인다.

```
640×480  →  512×384 = 196,608 px 가 실제 장면
1280×720 →  512×288 = 147,456 px 가 실제 장면   ← 33 % 적음
```

**즉 진짜 쟁점은 "480이냐 720이냐"가 아니라 "4:3이냐 16:9냐"다.**
960×720(4:3)으로 올려도 512×384로 640×480과 모델 입력이 똑같다 —
해상도를 올려도 SmolVLA에겐 의미가 없고 인코딩 비용만 는다.

한편 **비용은 원본 해상도에 비례한다:**

| | 640×480 | 1280×720 |
|---|---|---|
| 픽셀 수 | 307K | 921K (**3배**) |
| 인코딩 시간 | 기준 | 약 3배 |
| 저장 용량 | 기준 | 약 3배 |
| **SmolVLA가 보는 것** | **512×512** | **512×512 (동일)** |

> **판단**: SmolVLA만 쓸 거라면 **640×480(4:3)이 유리하다.** 1280×720은
> 인코딩·저장을 3배 쓰면서 **모델이 보는 실제 장면은 오히려 33 % 적다.**
>
> **단, 결정 조건이 하나 있다 — 4:3 화각으로 칠판과 팔이 다 들어오는가?**
> 16:9가 더 넓게 본다. RealSense는 4:3 모드에서 수평 화각이 좁아질 수 있으므로
> (센서 native가 wide) 실물로 양쪽 화각을 비교해봐야 한다.
>
> **단, 반론이 있다.** 720p 원본에서 축소한 512×512가 480p에서 축소한 것보다
> 미세하게 선명할 수 있고, 나중에 다른 모델(더 높은 해상도를 쓰는)로 갈아탈 여지가
> 남는다. 재수집이 불가능하다는 걸 감안하면 "일단 크게 찍어두자"도 방어 가능한 입장이다.
>
> **⚠️ jmbaek의 depth 검증이 1280×720 기준으로 되어 있다** (`docs/depth/README.md`).
> depth를 켤 거라면 그쪽 설정과 맞추는 게 안전하다. **팀 합의가 필요한 지점.**

### 4-2. FPS 30은 타당하다 — 다만 "제어 주파수"와 구분해야 한다

혼동하기 쉬운 두 개념:

| | 뜻 | 우리 값 |
|---|---|---|
| **녹화 FPS** | 데이터를 초당 몇 장 저장하나 | 30 |
| **액션 출력 주파수** | 모델이 초당 몇 개 액션을 뱉나 | 30 (녹화와 동일) |
| **재계획 주파수** | 모델이 **새 관측을 보고 계획을 다시 짜는** 주기 | **0.6 Hz** ← 문제 |

`n_action_steps=50`, 30fps → **1.67초에 한 번만 재계획**한다.

π0가 "50 Hz 제어"라고 할 때도 이건 **액션 출력 속도**이지 재계획 주기가 아니다.
50 Hz로 액션을 뱉되 50개 청크를 통으로 실행하면 재계획은 1 Hz다.

**이게 effort와 직결된다.** 접촉 이상을 감지해도 최대 1.67초 뒤에나 반응할 수 있다
(`effort_guide.md` §C-1). 그리고 이건 **녹화 설정이 아니라 학습/추론 설정**이라
나중에 조정 가능하다.

→ **FPS 30은 그대로 가고, `n_action_steps`는 학습 단계에서 실험한다.**

### 4-3. 비전 인코더 동결 여부가 모델마다 다르다 🟢

| 모델 | `freeze_vision_encoder` |
|---|---|
| SmolVLA | **True** |
| π0 / π0.5 | **False** |

팀 주제가 "효율적 파인튜닝"이라 이건 직접적인 실험 축이다. SmolVLA는 기본적으로
비전 인코더를 얼려서 학습 파라미터를 줄이는 설계이고, π0 계열은 푼다.
**SmolVLA에서 이걸 푸는/유지하는 비교**가 ablation 후보가 된다.

### 4-4. state 차원 여유 확인 🟢

| 모델 | `max_state_dim` | 우리 20차원 |
|---|---|---|
| SmolVLA / π0 / π0.5 | 32 | ✅ 여유 |
| GR00T | 64 | ✅ 여유 |

effort를 켜면 `observation.state`가 pos 7 + effort 7 + vel 6 = **20차원**이 되는데,
**모든 모델의 한도 안에 들어간다.** 아키텍처 수정 없이 실험 가능하다.

---

## 5. 남은 조사 항목

- [ ] SmolVLA의 실제 제어 주파수 (랩 PC 5090에서 추론 지연 실측) ⚪
- [ ] π0 계열의 이미지 입력 해상도 — lerobot 코드에 상수가 안 보임. openpi 원본 확인 필요 ⚪
- [ ] RT-2 이미지 해상도 ⚪
- [ ] `resize_with_pad`의 패딩이 SigLIP 성능에 주는 영향 (4:3 vs 16:9 중 어느 쪽이 패딩이 적은지)
- [ ] OpenVLA-OFT의 청킹 방식을 SmolVLA에 적용 가능한지

---

## 참고 자료

**코드 (1차 출처)**
- `lerobot 0.4.4` — `policies/{smolvla,pi0,pi05,pi0_fast,groot}/configuration_*.py`, `modeling_*.py`

**문헌**
- [OpenVLA: An Open-Source Vision-Language-Action Model](https://openvla.github.io/) · [GitHub](https://github.com/openvla/openvla)
- [OpenVLA-OFT](https://openvla-oft.github.io/) · [Fine-Tuning VLA Models: Optimizing Speed and Success](https://arxiv.org/html/2502.19645v1)
- [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://robotics-transformer2.github.io/) · [HuggingFace](https://huggingface.co/papers/2307.15818)
- [π0: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/html/2410.24164v1)
- [PD-VLA: Accelerating VLA Integrated with Action Chunking via Parallel Decoding](https://arxiv.org/html/2503.02310)
