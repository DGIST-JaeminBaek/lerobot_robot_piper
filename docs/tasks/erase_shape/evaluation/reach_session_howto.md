# 도형 선택(도달) 평가 세션 실행 설명서

`reach_eval_ui.py`로 **"프롬프트가 시킨 도형으로 갔나"만** 보는 평가의 실행 절차다.
"끝까지 다 지웠나"를 보는 기존 평가는
[eval_session_howto.md](eval_session_howto.md), 판정 지표의 근거는
[metric_rationale.md](metric_rationale.md)를 본다.

---

## 1. 왜 세션을 나눴나

기존 평가의 성공 판정은 `target erased ≥ 0.90`이다. 여기엔 두 가지 능력이 한
숫자에 섞여 있다.

1. 프롬프트가 가리키는 **도형을 고르는** 능력
2. 그 도형을 **끝까지 지우는** 능력

방해 도형을 넣어 학습한 뒤로는 ①만 따로 보고 싶은데, 기존 지표로는 ①을
제대로 했어도 ②에서 실패하면 그냥 실패로 찍힌다. 실제로 "옳은 도형으로 갔는데
마무리를 못 해 실패"가 반복됐다. 그래서 ①만 재는 창을 나눴다.

**이 세션은 성공/실패를 내지 않는다.** 내는 건 "어느 도형으로 갔나" 하나다.

| | 기존 (`erase_eval_ui.py`) | 이 문서 (`reach_eval_ui.py`) |
|---|---|---|
| 영상·데이터셋 녹화 | 함 (`--mode augment`) | **안 함** (`--mode demo`) |
| 완주 판정 | `erased ≥ 0.90` | 안 봄 |
| 실패 유형 7종 | 있음 | 없음 |
| 잉크 잔여량·residual | 있음 | 없음 |
| 그리퍼 위상·종료 사유 | 있음 | 없음 |
| 최종 판정 | 자동 + 사람 수정 | **사람이 결정** (자동은 추천만) |
| 결과 폴더 | `outputs/evaluation/erase_shape/` | `outputs/evaluation/erase_shape_reach/` |

---

## 2. 환경 준비 (매번)

```bash
conda activate ugrp
cd /home/ugrp43/UGRP/lerobot_robot_piper
```

**`ugrp` 환경을 쓸 것.** 카메라 시리얼은 `configs/recording.env` 기준으로
top `327122074262`, wrist `243322071626`다(카메라를 옮기지 않았다면 고정값).

보드 좌표가 맞는지는 기존 문서
[eval_session_howto.md §2](eval_session_howto.md)와 같은 방법으로 확인한다 —
이 세션도 같은 `--board`/`--exclude`를 쓴다.

---

## 3. 보드 준비 — 타겟과 방해 도형

이 평가의 전제는 **보드에 도형이 둘 이상 있는 것**이다. 하나만 그리면 고를 게
없어서 아무것도 못 잰다.

정렬 도구로 zone 박스를 보면서 그린다:

```bash
python scripts/tasks/erase_shape/analysis/block_alignment_tool.py \
  --shape-zones 315 --size-source global --target-shape circle
```

- 박스 중심: 315개 전체를 위치로 클러스터링한 6개 zone 중심
- 박스 크기: **해당 도형 105개 전체의 median** (`--size-source global`)
- 박스만 그린다. 안쪽 가이드 도형은 없다(2026-08-25에 뺐다 — 선이 많아 보드가
  지저분해지고 따라 그리는 데 도움이 안 됐다)
- 키: `c` 그린 도형 검사 · `s` 스냅샷 · `r` 카메라 재측정 · `q` 종료

### 도형별 박스 크기가 다르다 — 두 번 띄운다

`--target-shape`는 박스 **크기**를 바꾼다. 315 기준 실측 median:

| 도형 | 폭 × 높이 |
|---|---|
| circle | 127 × 124 |
| triangle | 106 × 129 |
| rectangle | 118 × 119 |

circle과 triangle은 폭이 21px(약 20%) 다르다. **circle 박스에 맞춰 삼각형을
그리면 학습 분포보다 눈에 띄게 넓어진다.** 그래서 타겟을 그린 뒤 도구를 끄고
`--target-shape`를 방해 도형으로 바꿔 다시 띄운 다음 방해 도형을 그린다.

GUI의 **[정렬 도구 켜기]** 버튼은 `REACH_TARGET`을 물려받으므로 타겟 크기로만
뜬다. 방해 도형은 위 명령을 터미널에서 직접 띄워 그린다.

### 겹치는 zone을 피한다

6개 zone 박스 중 두 쌍은 실제로 겹친다 — 클러스터 중심 간격이 박스 폭보다 좁다.

```
top-left ↔ top-mid
bottom-left ↔ bottom-mid
```

타겟과 방해 도형을 이 쌍에 나눠 그리면 두 도형이 붙거나 겹쳐서 검출이 섞인다.
**서로 안 겹치는 zone을 고를 것** (예: `top-left` + `top-right`).

### 좌우를 절반씩 바꿀 것

타겟을 계속 같은 쪽에 그리면 "프롬프트를 이해했다"와 "항상 왼쪽 것을 지운다"가
같은 정답률로 나온다. 시행의 절반은 타겟을 왼쪽, 절반은 오른쪽에 둔다. 요약표의
`타겟L`/`타겟R` 열이 그걸 보라고 있는 열이다.

---

## 4. 세션 실행

```bash
REACH_TARGET=circle \
REACH_MODEL=smolvla REACH_CONDITION=distractor_circle_vs_triangle \
REACH_CUTOFF=60 \
REACH_WATCH_DIR=records/hil \
REACH_ROLLOUT_CMD="python scripts/tasks/erase_shape/runtime/erase_run.py \
  --policy_path outputs/train/pick_up_the_eraser_and_erase_the_shape/smolvla_pickup_315_distractor/checkpoints/last/pretrained_model \
  --dataset_root records/outputs/pick_up_the_eraser_and_erase_the_shape/pick_up_the_eraser_315_distractor \
  --task 'pick up the eraser and erase the circle' \
  --target circle \
  --top-cam 327122074262 --wrist-cam 243322071626 \
  --top-crop 280,0,720 --wrist-crop 280,0,720 \
  --mode demo --max-steps 900 --stop-on-release \
  --confirm" \
bash scripts/14__reach_session.sh
```

하드웨어 없이 창만 점검하려면:

```bash
DRY_RUN='records/0822/hil/20260822-15*' \
REACH_TARGET=circle REACH_MODEL=smolvla REACH_CONDITION=dryrun \
bash scripts/14__reach_session.sh
```

### 도형을 바꿀 때는 네 곳을 같이 바꾼다

`REACH_TARGET` · `--task '... erase the OOO'` · `--target OOO` · `REACH_CONDITION`.
하나라도 어긋나면 프롬프트와 판정 대상이 달라져 그 시행은 무의미해진다.

### 환경변수

| 변수 | 설명 |
|---|---|
| `REACH_TARGET` | **필수.** circle / triangle / rectangle |
| `REACH_MODEL` | 집계 축 1 (smolvla / pi0 …) |
| `REACH_CONDITION` | 집계 축 2. 결과 폴더를 가른다 |
| `REACH_TRIALS` | 목표 시행 수 (표시용) |
| `REACH_CUTOFF` | 시도당 컷오프 초 (도구 기본 45) |
| `REACH_WATCH_DIR` | 러너가 시도 폴더를 만드는 곳. `--mode demo` 기본은 `records/hil` |
| `REACH_ROLLOUT_CMD` | 롤아웃 1회 명령. **`--mode demo` 필수** |
| `REACH_OUT_DIR` | 결과 폴더 (기본 `outputs/evaluation/erase_shape_reach/<MMDD>_<model>_<condition>`) |
| `REACH_BOARD` | `"x y w h"` — 카메라를 옮겼으면 지정 |
| `REACH_ALIGN_CMD` | 정렬 도구 명령 직접 지정 (escape hatch) |
| `DRY_RUN` | 로봇 없이 기존 시도 폴더로 창만 점검 |

### `--mode demo`가 왜 필수인가

`--mode demo`는 러너 프리셋에서 `record_dataset=False`다. 영상·데이터셋을 아예
안 만들고 판정용 PNG 두 장(`00_reference.png`, `01_after.png`)만 남긴다. 이 창은
인코딩을 기다리지 않는 전제로 만들어졌다. `--mode demo`가 없으면
`14__reach_session.sh`가 경고를 찍는다.

### `--max-steps`를 왜 900으로 주나

기본값 2100은 70초다. 기존 평가 305시행 실측에서 **완주(release)까지 중앙값
22.7초, p90 37.3초**였다(파지까지는 중앙 8.7초, p90 12.8초). 도달만 볼 거면
900스텝(30초)이면 충분하고, 못 끝내는 시도가 70초를 다 태우는 걸 막는다.

`REACH_CUTOFF`는 그보다 넉넉히(60초) 둔다. 컷오프가 `--max-steps`보다 짧으면
거의 모든 시도가 SIGINT로 강제 중단되고 매번 `[사람이 중단]` 메모가 붙는다.

---

## 5. 창 조작

| 키 | 동작 |
|---|---|
| `space` | 시작 / 중단 |
| `1`~`4` | 판정 선택 |
| `Enter` | 기록하고 다음 |
| `n` | 이번 시도 버리기 (기록 안 함) |
| `x` | 유효/무효 토글 |

⚠️ **[중단]은 안전장치가 아니다.** 롤아웃 프로세스를 종료할 뿐이라 팔이 즉시
서지 않는다. 위험하면 하드웨어 비상정지를 먼저 누른다.

### 판정 4종

| 키 | 판정 | 뜻 |
|---|---|---|
| `1` | `target` | 맞는 도형으로 갔다 |
| `2` | `distractor` | 엉뚱한 도형으로 갔다 |
| `3` | `none` | 어느 도형에도 안 갔다 (지우개를 못 잡음 / 보드에 못 닿음 포함) |
| `4` | `unclear` | 판단 불가 (둘 다 건드림 / 애매함) |

**판정은 눈으로 본 것이 우선이다.** 시도가 도는 동안 팔이 어디로 가는지 보고
있어야 한다 — 사진 두 장으로는 못 가리는 경우가 있다.

### 정렬 도구 토글

시도 사이에 [정렬 도구 켜기]로 다음 도형을 그린다. 롤아웃이 카메라를 쥐고 있는
동안은 버튼이 잠기고, 러너가 `[DISCONNECT]`를 찍은 뒤부터 열린다. 도구를 끄면
2.5초 뒤에 [시작]이 풀린다 — RealSense가 USB 레벨에서 풀릴 시간이다(이걸
안 기다리면 다음 롤아웃이 세그폴트로 죽는다).

---

## 6. 자동 추천과 그 한계

기준 프레임과 최종 프레임의 **도형별 잉크 변화**를 재서 라디오 버튼을 미리
골라준다. 임계값은 `TOUCH = 0.05`다(잉크 노이즈 플로어 ≈0.033의 1.5배, 완주
임계 0.90과는 한참 아래).

| 잉크 변화 | 추천 |
|---|---|
| 타겟만 ≥ 0.05 | `target` |
| 방해만 ≥ 0.05 | `distractor` |
| 둘 다 ≥ 0.05 | `unclear` — 스쳐 지나간 것과 골라서 간 것이 같은 흔적을 남긴다 |
| 둘 다 미만 | `none` |

**한계: 잉크 변화는 "문질렀나"를 재는 것이지 "갔나"를 재는 게 아니다.** 옳은
도형 위까지 갔지만 안 닿았으면 `none`으로 추천된다. 그래서 추천은 참고값이고
기록은 사람이 누른 값으로 남는다.

사람이 추천과 다르게 고르면 `auto_verdict`에 원래 추천이, `agree`에 `False`가
남는다. 나중에 "자동 판정을 믿어도 되나"를 따로 볼 수 있다(요약표의 `자동
일치도` 열).

궤적으로 진짜 도달을 재려면 FK + 카메라 캘리브레이션이 필요해서 지금은 안 한다.

---

## 7. 결과 확인

```
outputs/evaluation/erase_shape_reach/<MMDD>_<model>_<condition>/
├── trials.csv     시행별 기록
├── summary.md     사람이 읽는 요약
└── summary.json   집계값
```

요약은 시행마다 다시 쓴다 — 세션 도중에 죽어도 지금까지 것이 남는다.

### `trials.csv` 주요 컬럼

| 컬럼 | 내용 |
|---|---|
| `verdict` | **사람이 고른 최종 판정** |
| `auto_verdict` / `agree` | 자동 추천과 그 일치 여부 |
| `target_side` / `target_pos` | 타겟이 보드 좌(`L`)/우(`R`) 중 어디, 중심 좌표 |
| `distractors` / `distractor_side` / `distractor_pos` | 방해 도형 라벨·좌우·좌표 (검출 순서로 짝지음) |
| `n_distractors` | 방해 도형 개수. **0이면 선택성을 못 잰 시행이다** |
| `d_target` / `d_distractor` | 도형별 잉크 변화 (추천의 근거) |
| `elapsed_s` | 시도 소요 시간 |

### `summary.md`

```
| 조건 | n | 정답률 (95% CI) | 타겟L | 타겟R | 자동 일치도 |
```

`타겟L`/`타겟R`이 크게 갈리면 위치 편향이다 — 프롬프트가 아니라 자리를 따라간
것이므로, 배치를 절반씩 섞어 다시 본다.

---

## 8. 알려진 함정

- **방해 도형을 안 그리면 아무 의미가 없다.** 첫 시도 채점 후 `보드 배치` 줄에
  `circle@L (타겟)  triangle@R`처럼 둘 다 뜨는지 확인한다. 방해 쪽이 비어 있으면
  `--board` 밖에 그렸거나 너무 옅게 그린 것이다.
- **겹치는 zone 쌍**(`top-left↔top-mid`, `bottom-left↔bottom-mid`)에 타겟과
  방해를 나눠 그리지 않는다(§3).
- **도형별 박스 크기가 다르다.** 방해 도형은 정렬 도구를 그 도형으로 다시 띄워
  그린다(§3).
- **시도당 고정 오버헤드가 약 37초다.** 실측 기준 정책 로딩 ~15초, 카메라
  워밍업+기준 프레임 ~4초, 파킹+판정 ~14초. 로딩을 없애려면 시도 사이에 러너
  프로세스를 살려둬야 하는데, 그러면 카메라를 계속 쥐고 있어 정렬 도구와
  충돌한다 — 지금 구조로는 간단히 못 고친다.
- **`erase_run.py`는 1회용이다.** 시도 반복 인자가 없다(`i = 1`로 고정).

---

## 참고

- 코드: `scripts/tasks/erase_shape/evaluation/reach_eval_ui.py`
  (`--selftest`로 판정 로직 검증 가능)
- 실행 래퍼: `scripts/14__reach_session.sh`
- 완주까지 보는 기존 평가: [eval_session_howto.md](eval_session_howto.md)
- 판정 지표 근거: [metric_rationale.md](metric_rationale.md)
- 런타임 설계: [../runtime/erase_run_design.md](../runtime/erase_run_design.md)
