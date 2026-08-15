# CLAUDE.md

이 저장소에서 작업하는 에이전트를 위한 안내. 프로젝트 소개와 설치는 `README.md`,
원본 fork 대비 변경점은 `docs/change.md`를 볼 것. 여기에는 **작업 전에 알아야 문제가
생기지 않는 것들**만 적는다.

## 무엇을 하는 프로젝트인가

Agilex Piper 7-DOF 로봇팔로 **"화이트보드에 그려진 도형을 지우개로 지우고 지우개를
제자리에 갖다 놓는"** 태스크를 VLA(Vision-Language-Action) 모델에 학습시키고 실물에서
평가한다. 논문 작성이 목표이며, 핵심 비교는 **메모리 모듈(HAMLET) 유무**와
**카메라 구성(top+wrist vs top-only)** 의 ablation이다.

## 환경 — conda 두 개를 쓴다

| 환경 | transformers | 용도 |
|---|---|---|
| **`ugrp`** | 4.57.6 | SmolVLA, HAMLET, 데이터 분석, **실물 로봇 제어** (기본) |
| **`pi0`** | 4.53.3 (git 브랜치) | pi0 전용 |

**섞으면 안 된다.** pi0가 요구하는 transformers 브랜치를 `ugrp`에 설치하면 SmolVLA와
HAMLET이 깨진다. 자세한 이유는 `docs/training/pi0_finetuning.md`.

`lerobot`은 `/home/ugrp43/UGRP/piper_sdk/lerobot`에 **editable로 설치돼 두 환경이 소스를
공유한다.** 그 디렉터리를 수정하면 양쪽 모두에 영향이 간다. v0.4.4에 로컬 패치가 몇 개
있으므로(예: `configs/default.py`의 `lora_alpha` 필드) 버전을 올리지 말 것.

RViz나 로봇 제어를 쓰는 스크립트는 실행 전에 ROS2를 source해야 한다:

```bash
source /opt/ros/humble/setup.bash
source ~/UGRP/ros2_ws/install/setup.bash
conda activate ugrp
```

`PYTHONPATH`를 덮어쓰면 `rclpy`를 잃는다. 추가할 때는 반드시 이어붙일 것:
`PYTHONPATH=/home/ugrp43/jmbaek:$PYTHONPATH`

## 실물 로봇을 건드리는 작업

**팔이 실제로 움직이는 명령은 사용자가 직접 실행한다.** 에이전트는 명령어를 준비하고
검증까지만 한다. 사용자가 명시적으로 요청하지 않는 한 `--apply-to-robot`이 붙은 명령을
직접 실행하지 말 것.

주의할 점:
- `--source robot`은 `--apply-to-robot` 없이도 **팔의 토크를 켠다**
  (`piper_follower.py:159`). 카메라만 필요하면 별도 스크립트를 쓸 것.
- 실물 전송은 세 조건이 모두 있어야 열린다: `--source robot`, `--apply-to-robot`,
  `--real-robot-confirm I_UNDERSTAND_REAL_ROBOT`
- 안전장치는 `PiperFollower.send_action()` 안에 있다 — EMA → `max_relative_target`
  클램프 → effort 컷오프(트립 시 parking 후 종료). 이건 어느 경로로 호출하든 걸린다.

## 데이터

| 경로 | 내용 |
|---|---|
| `records/{0727,0802,0804,0805}/` | **원본 녹화** (1280×720, 30fps, 녹화당 에피소드 1개) |
| `records/outputs/erase_the_shape_150/` | 학습용 (150 에피소드, 75,787 프레임, 512×512, top+wrist) |
| `records/outputs/erase_the_shape_150_top_only/` | 같은 데이터의 top 카메라만 |
| `configs/erase_shape_150_manifest.json` | 어떤 원본의 몇 번째 프레임까지 쓸지 정의 |

데이터셋은 매니페스트를 고쳐 `scripts/tools/prepare_erase_shape_dataset.py`로 다시
빌드한다. 원본은 건드리지 않는다.

**비디오 인코딩 함정**: `--vcodec hevc_nvenc`를 쓰면 `--gop-size`가 **조용히 무시된다**
(lerobot `video_utils.py::_get_codec_options()`가 HW 인코더에서 `g` 옵션을 제외).
GOP=1이 필요하면 반드시 소프트웨어 `--vcodec hevc`를 쓸 것. GOP가 크면 학습 중 프레임
디코딩이 5배 가까이 느려진다.

## 학습

150 에피소드 / batch 8 기준 **1 에폭 = 9,473 step**, `--steps=75000`이 7.92 에폭이다.
비교 실험은 전부 이 값으로 맞춰져 있다.

| 전략 | 구성 | 상태 |
|---|---|---|
| 1 | SmolVLA, top+wrist | 완료 (75k) |
| 2 | SmolVLA, top-only | 완료 (75k) |
| 3 | SmolVLA + HAMLET, top+wrist | 완료 (75k) |
| 4 | SmolVLA + HAMLET, top-only | 완료 (75k) |
| — | pi0 LoRA, top+wrist | 완료 (75k) |

**전략 1~4의 체크포인트는 외장 하드로 옮겨졌다.** 로컬 `outputs/train/`에는
`pi0_lora_topwrist_75k`만 있다.

절차 문서:
- `docs/training/method.md` — 데이터 만들기부터 기본 모델 학습까지 (팀원용, 메모리 모듈 제외)
- `docs/training/new_smolvla_finetuning.md` — 150 에피소드 재학습 진행 기록
- `docs/training/pi0_finetuning.md` — pi0 도입 전 과정 (환경 분리, gated repo, 리비전 등)

### W&B 규칙

**`--wandb.run_id`는 매번 새 이름을 써야 한다.** 기존 id를 재사용하면 그 run의 이름·태그·
loss 이력이 병합되어 오염된다. 삭제한 id는 **영구 폐기되어 재사용할 수 없다**(HTTP 410).
사용 전 API로 확인할 것:

```python
import wandb
api = wandb.Api()
api.run(f'{api.default_entity}/smolvla-erase-shape/{run_id}')  # 없어야 안전
```

`--wandb.disable_artifact=true`를 붙이지 않으면 `~/.cache/wandb/artifacts`에 체크포인트
사본이 쌓인다(과거 29GB까지 찬 적 있음).

## 추론

주 도구는 `scripts/tools/piper_infer_runner.py`다. GUI 없이 돌고, RViz는 별도로
띄워야 한다(`ros2 launch agx_arm_description display_piper.launch.py`).

```bash
python scripts/tools/piper_infer_runner.py \
  --dataset-root records/outputs/erase_the_shape_150 \
  --policy-path <checkpoint>/pretrained_model \
  --source dataset --episode 0 --mode demo        # 안전: 실물 전송 없음
```

`--dataset-root`는 `--source robot`일 때도 필요하다 — 정규화 통계와 관찰 형태를 그
데이터셋 메타에서 가져오므로 **정책을 학습시킨 데이터셋을 지정해야 한다.**

관련 도구:
- `piper_human_approved_inference.py` — chunk를 구간별로 RViz 확인 후 승인하며 실행
- `piper_offline_chunk_rollout.py` — 비실물 궤적 검사. `load_policy()`가 여기 있다
- `piper_infer_gui.py` — tkinter GUI (E-STOP 버튼 포함). **HAMLET·pi0는 미지원**
- `block_alignment_tool.py` — 지우개를 학습 데이터와 같은 위치에 놓도록 실시간 안내

### 추론 파이프라인 기본값

SmolVLA 논문 Algorithm 1(비동기 추론)을 이식해뒀다. 근거와 실측은
`docs/policy/smoothing.md`.

```
정책 chunk → aggregate(weighted_average) → EMA(α=0.2) → rate_limit(5.0) → clip
           → 지연 보정(latency_align) → send_action()의 안전 클램프
```

- `--trigger-mode threshold --chunk-threshold 0.3` — 큐 소진 비율 기반 재추론
  (논문의 g=0.7). 스윕 결과 0.3~0.9 어디서도 큐가 마르지 않으므로 기본값 유지가 좋다
- `--aggregate-fn` — `weighted_average`(기본) / `latest_only` / `temporal_ensemble`(ACT 방식)
- `--latency-align` — 추론 지연(SmolVLA 3~5스텝, pi0 5~6스텝)만큼 chunk 앞을 잘라
  '지금'에 맞춘다. `--no-latency-align`으로 이전 동작 재현 가능

진단 로그: `[LAG]`(추론 지연 스텝), `[RATE]`(rate_limit 작동), `[TIMING]`(단계별 시간),
`[CLAMP]`(관절/그리퍼 분리 보고).

**그리퍼가 계속 클램프되는 건 정상이다** — 물체를 쥔 채 "더 조여"를 보내는 상태이고
학습 데이터에서도 동일하다(실측 명령 22.2 / 실측 27.0). 관절 클램프만 진짜 문제다.

## HAMLET (메모리 모듈)

`/home/ugrp43/jmbaek/smolvla_hamlet`에 별도 패키지로 있다. 실행 시:

```bash
PYTHONPATH=/home/ugrp43/jmbaek:$PYTHONPATH ... --policy.discover-packages-path=smolvla_hamlet
```

학습에서는 `--policy.pretrained_path=lerobot/smolvla_base` + `--policy.type=smolvla_hamlet`을
쓴다(`--policy.path`를 쓰면 `type`이 `smolvla`로 고정되어 HAMLET 설정이 무시된다).
`--policy.input_features`도 명시해야 한다.

## 코드 수정 시 관례

- 수정 전 `tmp/inference_backup_YYYYMMDD/` 같은 날짜 폴더에 백업하거나 `*_prev.py`로 사본을 남긴다
- 주석은 한국어로, **"왜 이렇게 했는지"** 를 적는다. 특히 실측으로 알아낸 것(온도, 속도,
  실패 사례)은 수치와 함께 남긴다
- 새 옵션을 추가할 때 기존 동작을 기본값으로 유지하고, 새 동작은 플래그로 연다
  (실물에서 검증된 거동을 조용히 바꾸지 않는다)

## 분석 산출물

`outputs/analysis/` 아래에 있다.

- `shape_positions/` — 도형이 보드 어디에 그려졌는지 (180개 원본 첫 프레임 기준).
  대략 6개 영역에 분포하지만 격자처럼 정밀하지는 않다
- `session_consistency/` — 세션별 종료 상태 차이, 좌표축 오버레이

`scripts/tools/shape_position_analysis.py`로 재생성할 수 있다.

## 현재 미해결

- **실물 성공률이 낮다** — 지우개를 집은 뒤 기준 약 30%. 파지·이동 실패가 은근히 있다.
  지우개 위치 정렬은 원인이 아닌 것으로 확인됨(테스트 내내 2mm 이내 유지)
- **마무리 판단 실패** — 다 지운 뒤에도 그 자리에 머물며 끝내지 못한다. 원인 후보로
  "학습 데이터에서 시연자들이 몇 % 지웠을 때 복귀를 시작했는지가 일관적인가"를
  측정하는 잉크 잔량 분석이 계획돼 있으나 아직 실행 안 함
- **4전략 + pi0 정량 비교** 미실시
