# depth — 현재 구현, 실측 손실, 논문 정리, 도입 판단

> 작성: 2026-07-25 · 기준 코드: `origin/seongil/gui-refactor` (커밋 `a975b32`) · lerobot 0.4.4
> 수치는 전부 실측/코드 확인값. 추정은 "추정"이라고 표시했다.

---

## 0. 요약 (결론부터)

| 질문 | 답 |
|---|---|
| lerobot이 depth를 지원하나? | **아니오.** 컬러맵 파이프라인은 전부 우리 코드다 |
| 저장 형식은? | turbo 컬러맵 RGB → **mp4** (RGB 카메라와 완전히 동일 취급) |
| 정밀도 손실은? | 중앙값 **2.35mm**, 95% 4.7mm 이내. **주범은 압축이 아니라 8비트 양자화** |
| 손실 줄이는 법? | **거리 창을 25cm 이하로 좁히면 센서 해상도급** (논문 구현 불필요) |
| top만 켤 수 있나? | **가능** |
| wrist도 켤 수 있나? | 기술적으론 가능하지만 **구조적으로 부적합** (§4) |
| 저장용량/인코딩이 문제인가? | **아니다** (§5) |
| 1차 수집에 넣을까? | **아니오** — 단, 이유는 비용이 아니라 보정 미완료 + 우선순위 (§7) |

---

## 1. 현재 구현

### lerobot 0.4.4는 depth를 지원하지 않는다

```python
# lerobot/cameras/realsense/camera_realsense.py
# NOTE(Steven): Missing implementation for depth for now
def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:   # color만 반환
```

그래서 우리가 직접 붙였다:

| 파일 | 역할 |
|---|---|
| `depth_utils.depth_to_colormap()` | raw uint16 → 고정범위 클리핑 → turbo 컬러맵 RGB |
| `piper_follower._read_depth_colormap()` | 카메라 백그라운드 스레드의 `latest_depth_frame`을 락 걸고 직접 읽음 |

lerobot 코어가 나중에 depth API를 추가하면 이 부분을 교체해야 한다.

### 파이프라인

```
RealSense raw uint16 (mm)
  → [DEPTH_MIN_M, DEPTH_MAX_M] 로 클리핑 (창 밖은 버림, 0=홀은 dmax로)
  → 0~255 로 정규화                       ← 여기서 8비트 양자화
  → turbo 컬러맵 → RGB uint8 (H,W,3)
  → observation.images.<cam>_depth
  → mp4 (libsvtav1, yuv420p, crf30)        ← 손실 압축
```

### 저장 결과 (실제 데이터셋 생성해 확인)

```
videos/
├── observation.images.top/…mp4
├── observation.images.wrist/…mp4
└── observation.images.top_depth/…mp4   ← depth도 그냥 mp4 하나
```

**모델 입장에서 depth를 켜는 것은 "카메라를 한 대 더 다는 것"과 정확히 같다.** 특별한 배선도 모델 수정도 없다. SmolVLA의 `prepare_images()`가 데이터셋의 image feature를 개수 제한 없이 전부 SigLIP에 넣는다.

→ **나중에 추가해도 코드 변경이 0이다.** 데이터만 다시 찍으면 된다.

---

## 2. 정밀도 — 실측

### "2^12 → 2^8로 줄어드는 것"이 아니다

`depth_to_colormap`은 전체 범위가 아니라 **`DEPTH_MIN_M`~`DEPTH_MAX_M` 창만 잘라서** 256단계로 편다.

```
창 0.6m (현재 0.20~0.80) 기준
  센서 원본 : 1mm 단위 → 창 안에 600단계
  컬러맵    : 256단계  → 2.35 mm/step
  실제 손실 : 600 → 256, 약 2.3배   (4096 → 256이 아님)
```

### 비디오 압축을 통과시킨 복원 오차 (실측)

합성 depth(경사면 + 움직이는 물체 + 1.5mm 센서 노이즈 + 홀)를 **실제 녹화 설정 그대로** 인코딩→디코딩→turbo 역변환:

| | turbo (현재) | grayscale (대안) |
|---|---|---|
| 중앙값 오차 | **2.35 mm** | 2.35 mm |
| 평균 오차 | 1.43 mm | 1.56 mm |
| 95% 오차 | 4.71 mm | 4.71 mm |
| 최대 오차 | 80 mm | 61 mm |
| 10mm 초과 픽셀 | 0.35 % | 0.04 % |

**중앙값 오차 2.35mm = 양자화 1스텝과 정확히 일치.** 즉 **손실의 주범은 비디오 압축이 아니라 8비트 양자화**다. crf30 손실 압축을 통과해도 95%가 2스텝 이내이고, 큰 오차는 물체 경계 0.35%에만 몰린다.

(당초 turbo가 크로마 서브샘플링에 크게 망가질 것으로 예상했으나, 측정 결과 아니었다. grayscale이 경계 아티팩트는 9배 적지만 SigLIP 입력으로는 turbo가 낫고 차이가 교체를 정당화할 만큼은 아니다.)

### ⭐ 가장 싼 개선 — 창 좁히기

```
창 25.6cm 이하  →  256단계가 1mm/step 이하  →  센서 원본 해상도와 동일 (양자화 손실 0)
```

**논문 구현도, 코드 수정도 필요 없다.** 실측한 작업 볼륨에 맞춰 `DEPTH_MIN_M`/`DEPTH_MAX_M`만 좁히면 된다.

---

## 3. 관련 논문

### ⭐ 우리와 제일 잘 맞는 것 — Depth Helps (IROS 2024)

**"Depth Helps: Improving Pre-trained RGB-based Policy with Depth Information Injection"** ([arXiv 2408.05107](https://arxiv.org/html/2408.05107v1), [프로젝트](https://gewu-lab.github.io/DepthHelps-IROS2024/))

**왜 우리와 맞나:** 문제 설정이 동일하다 — *RGB로 사전학습된 정책에 depth를 어떻게 넣을까*. 우리도 SigLIP(자연 RGB 사전학습) 기반 SmolVLA를 쓴다. 실기 실험도 **D435 계열 카메라 2대 + 3인칭 뷰**로 우리 구성과 유사하다.

**방법 (DI² 프레임워크):**
- **DCM** (Depth Completion Module): RGB로부터 depth 피처를 예측 (학습 가능한 spatial token + cross-attention)
- **DAC** (Depth-Aware Codebook): 예측 depth를 512개 벡터로 양자화해 노이즈·누적오차 감소
- **학습 때만 RGB-D를 쓰고, 배포 때는 RGB만** 쓴다

**결과:**

| 구성 | RGB-D 입력 | RGB만 |
|---|---|---|
| RGB 베이스라인 | 57.95% | 57.95% |
| DCM만 | — | 60.20% |
| DCM + Codebook (전체) | **63.95%** | **63.15%** |

**우리에게 주는 시사점 3가지:**

1. **depth는 "학습 시 신호"로도 값어치가 있다.** RGB-D(63.95%)와 RGB만(63.15%) 차이가 0.8%p뿐이다. 즉 depth를 추론 때 안 써도 학습에 썼다면 대부분의 이득이 남는다.
2. **단, 그 이득은 DI² 아키텍처(DCM+DAC)에서 나온 것이다.** 우리가 지금 하는 "turbo depth를 3번째 카메라로 그냥 넣기"는 훨씬 단순·조악한 방식이라 같은 결과를 기대하면 안 된다.
3. ⚠️ **"단순한 물체 조작 태스크에서는 개선이 작다."** 이득은 long-horizon·복잡한 공간 추론 태스크에 몰린다(37.4% vs 24.2%). **우리 태스크는 평면 위 2D 마크를 지우는 것이라 기하학적으로 단순한 쪽에 가깝다.**

### 참고 — 보조 supervision 계열

- **QDepth-VLA** ([arXiv 2510.14836](https://arxiv.org/pdf/2510.14836)): depth를 입력이 아니라 **양자화된 예측 대상(보조 supervision)**으로 씀. 우리 ablation 5번(aux torque 예측)과 발상이 같아서, effort 쪽 설계에 참고할 만하다.
- **Evo-Depth** ([arXiv 2605.14950](https://arxiv.org/pdf/2605.14950)): 경량 depth 강화 VLA. `config_piper.py`의 `depth_raw_dir` 주석에서 이미 언급된 방향(IDEM 보조 supervision). 이쪽으로 가려면 컬러맵(8bit)보다 정밀한 raw가 필요하다.

### 배경 — 인코딩 방식 두 갈래

- **Turbo** ([Google Research, 2019](https://research.google/blog/turbo-an-improved-rainbow-colormap-for-visualization/)): `jet`의 false detail·banding·색맹 모호성을 고친 무지개 컬러맵. Google이 명시한 용도가 **depth/disparity/error map 시각화**다. **지각적으로 균일하지는 않다** — 정확한 값 읽기보다 구조 가시성을 택한 의도적 트레이드오프. 색맹 친화적이지 않고 흑백 출력에 부적합.
- **Pece, Kautz, Weyrich (2011), "Adapting Standard Video Codecs for Depth Streaming"** ([Eurographics DL](https://diglib.eg.org/handle/10.2312/EGVE.JVRC11.059-066)): 16bit depth를 **Y채널(거친 depth) + 크로마(같은 주기·다른 위상의 삼각파)**로 포장해 압축을 견디게 하는 기법. VP8/H.264/JPEG에서 검증.

**둘은 푸는 문제가 다르다:**

| | Turbo | Pece et al. |
|---|---|---|
| 목적 | 사람/**비전 인코더**가 잘 보게 | **압축 후에도 mm 값 복원** |
| 출력 | 자연스러운 색 그라디언트 | 삼각파 줄무늬 (사람 눈엔 괴상) |
| 적합 | **VLA 정책 입력** | metric depth, 포인트클라우드 |

SigLIP은 자연 RGB로 사전학습됐다. Pece의 삼각파는 SigLIP 입장에서 **본 적 없는 고주파 노이즈**라 정책 입력으로는 오히려 해롭다.

→ **정책 입력 목적이면 지금 우리 방식(Turbo)이 맞다. 바꿀 이유 없다.**
→ **Pece는 나중에 기하 계산(포인트클라우드, 3D 접촉점)이 필요해질 때의 카드다.** 다만 우리 조건에서는 §2의 "창 좁히기"가 같은 효과를 훨씬 싸게 낸다.

---

## 4. 카메라별 판단

### top 캠 — 켤 수 있고, 보정만 하면 동작한다

⚠️ **현재 설정 그대로 켜면 백지로 찍힌다.**

```
현재:  DEPTH_MIN_M=0.20  DEPTH_MAX_M=0.80
실제:  보드가 약 1m
```

**1m는 창 밖이라 전부 dmax로 클리핑 → 단색 이미지.** 정보 없는 비디오 스트림만 늘어난다. `0.85~1.10` 같은 값으로 옮겨야 한다 (실측 필요, §6).

### wrist 캠 — 구조적으로 부적합 ❌

**`DEPTH_MIN_M`/`DEPTH_MAX_M`는 상수라 모든 프레임에 같은 창이 적용된다.**

- top: 거리 고정 → 창 하나로 커버 ✅
- wrist: **거리가 계속 변함** → 가까우면 전부 dmin에 포화, 멀면 dmax에 포화

즉 **같은 보드 표면이 팔 자세에 따라 다른 색으로 찍힌다.** "이 색 = 이 거리" 대응이 깨져서 학습 신호가 아니라 노이즈가 된다.

`depth_utils.py` 주석에 이미 있는 경고가 정확히 이것이다:

> 정규화 범위는 태스크마다 재보정 필요 — 고정하지 않으면 프레임마다 같은 높이가 다른 색이 되어 학습이 안 됨

wrist는 **창을 고정해도** 이 문제가 생긴다. 카메라가 움직이기 때문이다. 제대로 하려면 프레임별 적응 정규화(→ 절대 거리 정보 소멸)나 카메라 pose 기반 보정이 필요한데, 둘 다 지금 할 일이 아니다.

### top만 켜는 설정

```bash
CAMERA_TYPE=intelrealsense          # 또는 TOP_CAM_TYPE=intelrealsense
TOP_REALSENSE_USE_DEPTH=true        # top만 depth 스트림 ON
WRIST_REALSENSE_USE_DEPTH=false     # wrist는 OFF
USE_DEPTH_OBSERVATION=true          # observation에 포함
DEPTH_MIN_M=?                       # ← 실측 후 결정 (§6)
DEPTH_MAX_M=?
```

`config_piper.py`가 카메라별 플래그를 지원하므로 **가능하다.** 스트림도 1개만 늘고, wrist 문제도 피한다.

### 그런데 top-only depth가 우리 태스크에 의미가 있나?

솔직한 평가 — **기대값이 낮다.**

1. **기하 구조가 자명하다.** 보드는 평면이고 도형은 그 위의 2D 마크다. depth로 볼 수 있는 3D 구조가 사실상 "평면 하나 + 튀어나온 팔"뿐이다. 도형 자체는 높이 차가 없어서 **depth에 안 보인다** — RGB만 본다.
2. **정말 궁금한 건 "지우개가 보드에 닿았나"인데, 그건 effort가 훨씬 직접적으로 답한다.** 1m 거리 스테레오 depth의 노이즈는 mm 단위이고(거리 제곱에 비례해 커진다), 접촉 여부는 그보다 미세한 사건이다.
3. **논문 근거도 같은 방향이다.** Depth Helps는 이득이 long-horizon·복잡한 공간 추론에 몰리고 **단순 조작에서는 개선이 작다**고 보고한다.

**어디서 도움이 될 수 있나:** 보드까지 거리가 달라지는 상황으로의 일반화, 팔 자세 추정, 충돌 회피. 지금 태스크 설정에서는 셋 다 부차적이다.

→ **ablation 6번으로 나중에 검증할 가치는 있지만, 1차 수집을 붙잡을 이유는 못 된다.**

---

## 5. 비용 — 문제가 아니다

녹화 PC가 학습 PC(RTX 5090 ×2)와 동일한 기계이므로 **CPU/저장용량은 depth 도입의 제약이 아니다.** (이전에 "CPU가 느리다"는 전제로 세웠던 논거는 철회한다.)

### 저장용량 — 1TB 대비

| 시나리오 (150ep × 30초) | 용량 | 1TB 대비 |
|---|---|---|
| 낙관 (매끈한 장면) | 0.5 GB | 0.05% |
| 현실 추정 | 2~10 GB | 0.2~1% |
| 비관 (노이즈 많은 장면, 4스트림) | 68 GB | 6.8% |

**최악을 잡아도 7%.** 수집을 10배로 늘려도 1TB 안에 들어온다. **저장용량을 근거로 depth를 넣거나 빼지 말 것.**

### ⚠️ 숨은 비용 — PNG 중간 파일

`STREAMING_ENCODING=false`(기본)이면 녹화 중 PNG를 디스크에 쌓았다가 나중에 인코딩한다. 실측:

```
10초 스트림당 PNG:  RGB 20 MB  vs  depth 79 MB   ← depth가 4배
```

turbo 컬러맵은 무지개 그라디언트라 PNG 압축이 잘 안 된다. 30초 에피소드 4스트림이면 **에피소드당 약 600MB가 썼다 지웠다** 반복된다. 용량 문제는 아니지만 느린 디스크면 녹화 루프 I/O를 방해할 수 있다.

### GPU 인코딩 (nvenc) — 쉽지만, 아마 불필요

**쉬운가:** lerobot이 `vcodec="auto"`면 HW 인코더를 자동 탐지한다.

```python
# lerobot/datasets/video_utils.py
HW_ENCODERS = ["h264_videotoolbox", "hevc_videotoolbox",
               "h264_nvenc", "hevc_nvenc", "h264_vaapi", "h264_qsv"]
```

**지원 여부 확인 (랩 PC에서):**

```bash
python -c "from lerobot.datasets.video_utils import detect_available_hw_encoders as d; print(d())"
```

`h264_nvenc`가 나오면 쓸 수 있다.

**빠른가:** nvenc는 CPU AV1 인코딩보다 통상 수 배~수십 배 빠르다(추정 — 랩 PC 실측 필요). 다만 **같은 파일 크기 기준 화질은 libsvtav1(AV1)이 h264_nvenc보다 낫다.** 학습 데이터라면 화질 손실이 곧 데이터 품질이므로 무조건 유리한 교환은 아니다.

⚠️ **현재 GitHub 코드로는 쓸 수 없다.** `teleop_ui.py`의 `_dataset_args()`가 `--dataset.vcodec`을 넘기지 않아서 항상 기본값 `libsvtav1`이 쓰인다. 쓰려면 한 줄 추가가 필요하다.

**권고:** 먼저 실제 인코딩 시간을 재고, 에피소드당 10초 이내면 **건드리지 말 것.** 5090 기계의 CPU면 그럴 가능성이 높다.

---

## 6. 아직 못 정한 것 — 카메라·보드 거리 실측

`DEPTH_MIN_M`/`DEPTH_MAX_M`은 **실측 없이는 정할 수 없다.** 잘못 잡으면 데이터가 통째로 무의미해진다.

**측정 절차 (2분):**

1. 카메라를 최종 위치에 고정
2. RealSense로 depth 한 프레임 떠서 관심 영역의 실제 거리 확인:
   ```python
   import pyrealsense2 as rs, numpy as np
   p = rs.pipeline(); cfg = rs.config()
   cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
   prof = p.start(cfg)
   scale = prof.get_device().first_depth_sensor().get_depth_scale()
   print("depth_scale:", scale)          # 0.001 가정이 맞는지도 같이 확인
   for _ in range(30): f = p.wait_for_frames()   # warmup
   d = np.asanyarray(f.get_depth_frame().get_data()) * scale
   roi = d[180:300, 240:400]             # 보드 중앙 부근
   roi = roi[roi > 0]
   print(f"보드까지: {np.median(roi):.3f} m,  범위 {roi.min():.3f}~{roi.max():.3f}")
   p.stop()
   ```
3. **팔을 가장 높이 든 자세**에서도 같은 측정 → 가장 가까운 거리 확인
4. 창 설정:
   ```
   DEPTH_MIN_M = (팔 최고점 거리) - 0.03    # 여유 3cm
   DEPTH_MAX_M = (보드 거리)     + 0.03
   ```
5. **창 폭이 25cm 이하면 양자화 손실 0.** 넘으면 그만큼 정밀도가 준다 (창폭/256 = mm/step)

**또한 `depth_scale`이 0.001이 맞는지 위 스크립트로 확인할 것.** 코드에 하드코딩돼 있다.

---

## 7. 판단 — 1차 수집에서는 빼자

**비용 때문이 아니다** (§5에서 저장용량·CPU 모두 문제 아님을 확인). 이유는 셋:

1. **보정이 안 끝났다.** 카메라-보드 거리 미측정 → `DEPTH_MIN_M`/`MAX_M`을 정할 수 없다. 지금 값(0.20~0.80)으로 켜면 **백지로 찍힌다.** 그리고 잘못 보정된 데이터셋은 재수집 외에 되돌릴 방법이 없다(컬러맵이 캡처 시점에 구워지므로).
2. **기대값이 낮다.** 우리 태스크는 평면 위 2D 마크 지우기라 depth가 볼 3D 구조가 거의 없다. 접촉 여부는 effort가 더 직접적이다. 논문(Depth Helps)도 단순 조작 태스크에서는 개선이 작다고 보고한다.
3. **순서가 안 맞는다.** ablation에서 depth는 6번, effort는 3~5번. **effort 실험 3개는 depth 없이 돌아간다.**

그리고 결정적으로 — **depth는 모델 입장에서 "카메라 한 대 추가"와 동일해서 나중에 붙여도 코드 수정이 0이다.** 급할 이유가 없다.

### depth를 켜기로 한다면 — 전제 조건

- [ ] 카메라-보드 거리 실측 (§6)
- [ ] `depth_scale` 실측 확인
- [ ] `DEPTH_MIN_M`/`MAX_M`을 창폭 25cm 이하로 설정
- [ ] **top만** 켜기 (`WRIST_REALSENSE_USE_DEPTH=false`, §4)
- [ ] 짧게 1 에피소드 찍고 **depth 영상을 눈으로 확인** (단색이면 창이 틀린 것)
- [ ] `USE_DEPTH_OBSERVATION`을 수집 끝까지 고정 (머지 제약)

---

## 8. 열린 질문

- **보드가 수직인가 수평인가?** top 캠이 수평 보드를 내려다보는지, 수직 칠판을 마주보는지에 따라 depth의 의미가 크게 달라진다. 이 문서는 "카메라가 보드면을 마주본다"를 가정했다.
- 1m 거리에서 D435 depth 노이즈가 실제로 얼마인지 (실측 필요 — 이게 크면 창을 좁혀도 소용없다)
- `DEPTH_RAW_DIR` raw sidecar: 카메라당 18MB/s, record loop 안에서 동기 `np.save`, 파일명이 세션 전역 순번이라 `frame_index`와 미대응. **본 녹화에서는 끌 것.** Evo-Depth 계열로 갈 때 별도 세션으로 수집.
