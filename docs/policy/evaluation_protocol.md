# 지우기 평가 프로토콜

정책(SmolVLA / π₀)이 화이트보드 도형을 지우는 롤아웃을 채점하는 방법.
평가 세션 창(GUI)을 쓰는 것이 기본이고, 도구가 안 될 때 쓰는 손 채점 방법도 §5에 있다.

| | |
|---|---|
| 실행 | `bash scripts/13__eval_session.sh` |
| 창 | `scripts/tools/erase_eval_ui.py` |
| 채점 로직 | `scripts/tools/erase_eval.py` (측정은 기존 `ink_metric.py` 재사용) |
| 결과 저장 | **`evaluation/<MMDD>_<model>_<condition>/`** (리포 최상단) |

작성 2026-08-17. **아직 실물 롤아웃에 써본 적 없다** — §4의 검증 절차를 먼저 돌릴 것.

---

## 1. 세션 창 사용법

```bash
# 하드웨어 없이 창만 만져보기 (기존 에피소드를 롤아웃 결과인 척 먹인다)
DRY_RUN='<데이터셋>/erase_the_rectangle_*' bash scripts/13__eval_session.sh

# 실제 세션
EVAL_MODEL=smolvla EVAL_CONDITION=async EVAL_TARGET=rectangle EVAL_TRIALS=30 \
EVAL_WATCH_DIR=/home/ugrp308/Group43/rollouts \
EVAL_ROLLOUT_CMD="python scripts/tools/erase_run.py --policy_path <ckpt> ... --confirm" \
bash scripts/13__eval_session.sh
```

결과는 `evaluation/0818_smolvla_async/`에 쌓인다.

| 파일 | 내용 |
|---|---|
| `episodes.csv` | 에피소드 1행. 확인 누를 때마다 append |
| `summary.md` | 조건별 요약표. 보고서에 그대로 붙일 수 있다 |
| `summary.json` | 같은 내용, 스크립트/AI 입력용 |

### 창 조작

| 버튼 | 키 | 동작 |
|---|---|---|
| 시작 | space | 롤아웃 1회 실행. 컷오프(`EVAL_CUTOFF`, 기본 60초 — §3 참고) 넘으면 자동 중단 |
| 중단 | space | 롤아웃 프로세스에 SIGINT → park 정리 후 채점으로 |
| 이번 시도 버리기 | n | 채점 없이 폐기. 실험 사고로 시도가 무의미할 때 |
| 실패 유형 | 1~7 | 자동 판정이 틀렸으면 수정 |
| 유효 체크 | x | 끄면 집계에서 제외 |
| 확인하고 다음 | Enter | CSV에 기록하고 요약 갱신 |

시도가 끝나면 점수, 단계 통과(✔/✘), 지움 비율, t@50/t@90, 파지·놓기 시각,
종료 사유, 정체/회복이 한 화면에 뜬다.

사람이 실패 유형을 바꾸면 메모에 `[auto=...]`로 원래 자동 판정이 남는다.
나중에 **자동 판정과 사람 판정의 일치도**를 계산할 수 있다.

> ⚠️ **[중단]은 안전장치가 아니다.** 프로세스를 종료할 뿐이라 팔이 즉시 서지 않을 수
> 있다. 위험하면 하드웨어 비상정지를 먼저 누르고, 그 다음 이 버튼으로 세션을 정리한다.

### 사람이 하는 일

1. **비상 정지 감시** (자동화 대상 아님)
2. **에피소드 유효/무효 판정** — 사람이 건드림, 마커 번짐, 조명 변화 등
3. **실패 유형 확인** — 자동 판정이 틀렸으면 라디오로 수정

실행 중 "다 지웠나?" 판단은 하지 않는다. 컷오프까지 두면 된다.
안전 때문에 손을 댔다면 그 에피소드는 실패로 기록한다.

---

## 2. 무엇을 재는가

### 2.1 연속 지표 (통계 검정은 여기서 한다)

| 지표 | 정의 |
|---|---|
| `erased_target` | 1 − (최종 잉크 픽셀 / 초기 잉크 픽셀), target 도형 |
| `erased_distractor` | 같은 정의, target이 아닌 도형 중 최댓값 |
| `selective_erase` | `erased_target − erased_distractor` |

이진 성공률만 재면 안 되는 이유는 검정력이다. 10~50회 시행에서 성공률 차이는
대부분 신뢰구간이 겹친다(8/10 → 95% CI 0.49–0.94, 40/50 → 0.67–0.89).
연속값은 에피소드마다 정보량이 훨씬 많아 같은 시행 수로 더 작은 차이를 잡는다.
**`selective_erase`를 주지표로 쓰고, 성공률은 병기만 한다.**

### 2.2 단계 점수

| # | 단계 | 판정 | 배점 |
|---|---|---|---:|
| 1 | 접근 | 지우개 파지 성공 | 1 |
| 2 | 접촉 | target이 0.5초 연속 가려짐 = 팔이 도형 위 | 2 |
| 3 | 지우기 개시 | `erased_target ≥ 0.15` | 2 |
| 4 | 실질 완료 | `erased_target ≥ 0.90` | 3 |
| 5 | 선택성 유지 | `erased_distractor ≤ 0.10` | 2 |

- **`task_progress` = 획득 점수 / 10 × 100**
- **`success` = 4·5단계 모두 통과** (= 만점)

**단계 안에서는 차등을 주지 않는다.** "잘 접근했으니 0.7점" 같은 주관 배점은
재현이 안 된다. 차등은 §2.1의 물리 측정값에만 둔다.
단계 간 배점은 "여기서 실패하면 뒤가 전부 불가능한가"로 가중했다.

### 2.3 그 외 기록

`t50_s` / `t90_s`(진행도 50%·90% 도달 시각), `auc_progress`, `stall_events`(3초간
진행 없던 구간 수), `recovery_rate`, `termination`, `failure_mode`, `duration_s`.

> `stall_events`는 **사람 시연에서도 평균 2.2건** 나온다(접근·재정렬 같은 정상 멈춤).
> 절대값이 아니라 이 baseline 대비로 읽을 것.

---

## 3. 임계값과 근거

| 상수 | 값 | 근거 |
|---|---|---|
| 완료 임계 | 0.90 | 이 리포 실측 분리도(target 1.00 vs distractor 0.05). ForceVLA "Wipe Board-2 = 완전히 지웠는가"의 정량 대응 |
| distractor 상한 | 0.10 | 노이즈 플로어 실측(120 에피소드: 평균 0.009 / p95 0.037 / 최대 0.097)의 약 3배 |
| 개시 임계 | 0.15 | ForceVLA "Wipe Board-1 = 지우는 동작을 수행했는가"의 정량 대응 |
| 접촉 | 0.5초 | 스쳐 지나감과 접촉을 가르는 최소 지속 |
| 정체 | 3초 | 실측(0804 시연 30개). 1초면 9.6건 → 3초 + 단조증가 envelope이면 2.2건 |
| 컷오프 | **미정** | 아래 참고 |

**컷오프는 정해두고 시작하지 말 것.** 기본값 60초를 넣어놨지만 근거가 약하다 —
사람 시연 평균이 24초(최대 31.5초)라는 것뿐이고, 정책이 얼마나 걸릴지는 실제로
돌려봐야 안다. 너무 짧으면 될 것도 시간초과로 잘리고, 너무 길면 정체된 에피소드를
계속 기다리게 된다. **처음 3~5개를 돌려보고 정한 뒤, 그 값을 여기에 기록하고
이후 모든 조건에 동일하게 적용한다.** 중간에 바꾸면 그 전 데이터는 버려야 한다.

종료 사유 판정 기준(그리퍼가 얼마나 벌어져야 "놓았다"인지)도 마찬가지로 초기 몇 개의
실제 거동을 보고 조정한다.

완료·distractor 임계는 `ink_metric.py`에서 import한다. 한 곳만 고치면
`erase_check.py`, `make_done_labels.py`와 함께 움직인다.

### 종료 판정은 성공 판정과 분리한다

로봇이 스스로 "끝났다"고 지우개를 놓는 것과 실제로 다 지운 것은 다른 사건이다.
섞으면 착각 종료가 정상 종료로 기록된다.

| 종료 사유 | 조건 |
|---|---|
| `release` | 그리퍼 명령이 파지 수준에서 15 이상 벌어짐 = 스스로 놓음 |
| `stall` | 진행 정체 상태로 끝남 |
| `cutoff` | 시간 초과 |

성공 판정은 종료 사유와 무관하게 종료 시점의 `erased_target`으로만 한다.
덜 지웠는데 놓으면 `premature_release` 실패로 기록된다.

### 실패 유형 (실패 1건당 주된 원인 하나)

| 유형 | 의미 |
|---|---|
| `selectivity_failure` | distractor를 건드림 |
| `grasp_failure` | 지우개를 못 잡음 |
| `approach_failure` | 잡았지만 보드에 못 닿음 |
| `contact_ineffective` | 닿았지만 안 지워짐 |
| `premature_release` | 덜 지웠는데 놓음 |
| `repetition_loop` | 같은 곳만 반복 |
| `incomplete_erase` | 그 외 미완료 |

---

## 4. 처음 쓸 때 검증 절차

**이 도구는 실물 롤아웃에 아직 써본 적이 없다.** 순서대로 통과시킨 뒤 본실험에 들어갈 것.

### 1) 로직 자기검증 — 하드웨어 불필요

```bash
python scripts/tools/erase_eval.py --selftest
python scripts/tools/ink_metric.py --selftest
```

### 2) 창 흐름 검증 — 하드웨어 불필요

```bash
python scripts/tools/test_erase_eval_ui.py '<데이터셋>/erase_the_rectangle_*'
```

시작 → 채점 → 실패 유형 수정 → 확인 → CSV 1행, 버리기는 기록 안 됨,
재시작 시 시행 번호 이어받기까지 자동 확인한다.

### 3) 기존 시연으로 회귀 확인 — 하드웨어 불필요

과거 텔레옵 시연은 전부 성공이므로 채점기가 전부 만점을 줘야 한다.

```bash
python scripts/tools/erase_eval.py '<데이터셋>/0804/erase_the_rectangle_0804-*' \
    --model teleop --condition rect_0804
```

2026-08-17 기준 0804 사각형 30개 실측 — 여기서 크게 벗어나면 카메라 세팅이 바뀐 것:

| 항목 | 기대값 |
|---|---|
| 성공률 | 30/30 |
| `erased_target` | 1.000 ± 0.0003 |
| `erased_distractor` | 0.012 (최대 0.05) |
| 종료 사유 | 전부 `release` |
| `stall_events` | 평균 2.2 (1~4) |
| `t90_s` | 18.3 ± 0.5 |

다른 날짜·다른 도형(0805 원)에서도 동일하게 만점이 나온다.

### 4) 현장 카메라 확인 — **실험 당일 첫 순서**

카메라가 조금이라도 움직이면 보드 좌표가 안 맞는다.

```bash
python scripts/tools/ink_metric.py <에피소드> --dump-roi /tmp/roi.png
```

`/tmp/roi.png`에서 (a) 보드 영역, (b) 도형 bbox, (c) 나무 지우개 블록이 도형으로
잡히지 않았는지를 눈으로 본다. 안 맞으면 `EVAL_BOARD="x y w h"`로 덮어쓰고 값을 기록.

### 5) 첫 3~5개는 영상과 대조

`release_s`가 실제 놓은 시점과, `t90_s`가 육안으로 다 지워진 시점과,
`failure_mode`가 실제 원인과 맞는지 확인한다. 맞으면 나머지는 자동으로 둔다.

### 알려진 함정

| 함정 | 증상 |
|---|---|
| 카메라 warmup 비대칭(top 0s / wrist 5s) | 초반 프레임 드랍이 초기 잉크량 기준을 오염 |
| `MAX_RELATIVE_TARGET`·`rate_limit` 둘 다 5.0 | 모든 조건이 똑같이 느려져 차이가 사라짐 |
| wrist 카메라 | 잉크 신호 없음. 평가에 쓰지 말 것 |
| 도형 미검출 | `[SKIP]`으로 표시되고 채점 안 됨 → `summary.md` 하단 확인 |

---

## 5. 도구가 안 될 때 — 손 채점

카메라·세그멘테이션이 실패하면 **사람이 눈으로 잴 수 있는 지표만** 기록한다.
자동으로만 나오는 것(t@50/t@90, AUC, 정체·회복, 가림 비율, 소수점 지움 비율)은 **버린다.**

### 준비

```bash
cp evaluation/manual_sheet_template.csv evaluation/0818_smolvla_async/episodes.csv
```

예시 2행은 지우고 쓴다. 시행마다 한 줄씩 채운다.

### 채울 칸 (나머지는 빈칸으로 둔다)

| 칸 | 채우는 법 |
|---|---|
| `episode` | 아무 식별자 (`t01`, `t02` …) |
| `model` / `condition` | 조건 이름. 집계 축이므로 철자 통일 |
| `trial` | 1부터 |
| `target` | circle / rectangle / triangle |
| `valid` | 실험 사고면 `False` |
| `scored_by` | `manual` |
| `stage_approach` ~ `stage_selective` | 각 0 또는 1 (§2.2 표) |
| `score` | 통과한 단계 배점 합 (1·2·2·3·2) |
| `task_progress` | `score` × 10 |
| `erased_target` | **눈금 4단계**: 안 지움 `0` / 절반쯤 `0.5` / 거의 다 `0.9` / 완전 `1.0` |
| `erased_distractor` | 안 건드림 `0` / 건드림 `0.5` |
| `success` | 완전히 지웠고 distractor 안 건드렸으면 `True` |
| `duration_s` | 스톱워치 |
| `termination` | `release` / `stall` / `cutoff` |
| `failure_mode` | 실패면 §3 표에서 하나 |
| `note` | 자유 기록 |

`erased_target`을 소수점까지 재려 하지 말 것. **4단계 눈금으로 충분하고,
그게 사람이 재현할 수 있는 유일한 해상도다.**

### 집계

같은 CSV 형식이므로 자동 채점 결과와 동일하게 처리된다.

```bash
python scripts/tools/erase_eval.py --summarize evaluation/0818_smolvla_async/episodes.csv
```

`summary.md` / `summary.json`이 같은 폴더에 생긴다. 자동 채점분과 손 채점분을
한 파일에 섞어도 되고, `scored_by` 칸으로 나중에 구분할 수 있다.

---

## 6. 실험 운영

- **시행 횟수**: 조건당 10회(RTC, ForceVLA, Wall-OSS) ~ 20회(SO-101)가 관행.
  가능하면 30~50회. 단 이진 성공률만으로는 50회도 부족하다(§2.1)
- **페어링**: 같은 초기 조건(도형 위치·모양·팔 초기자세)을 모든 조건에 동일하게 적용
- **순서 효과**: 마커 건조·보드 잔여물·지우개 포화 → 조건 순서 랜덤화, 청소 주기 고정
- **저장 공간**: 평가용은 top 카메라만. 0804 시연 10개 실측으로 1280×720 top 영상이
  **60초당 약 13.5MB**, 90 에피소드면 약 1.2GB. `record_raw_frames`는 끌 것

---

## 참고문헌

아래 6편은 **2026-08-17에 전부 원문(PDF 또는 arXiv HTML)을 직접 대조**했다.
인용한 문장·수치는 원문에 있는 것만 남겼다.

| 가져온 것 | 출처 |
|---|---|
| 단계별 정수 점수, 사후 채점, 시간 컷오프 | RTC, [arXiv:2506.07339](https://arxiv.org/abs/2506.07339) |
| 2단계 성공률(Wipe Board-1 동작 수행 / Wipe Board-2 완전 지움), 화이트보드 10 trials | ForceVLA, [arXiv:2505.22159](https://arxiv.org/abs/2505.22159) |
| 픽셀 기반 잔여 오염 비율 `m₁(r) = A(r)/A(1)×100` | Fernández-Rodicio et al., [arXiv:1903.05635](https://arxiv.org/abs/1903.05635) |
| 실패 taxonomy(4분류 + 주원인 1개 규칙), Recovery Rate, 20 trials | SO-101 benchmark, [arXiv:2606.08881](https://arxiv.org/abs/2606.08881) |
| 통계 유의성 요구, per-instance 원자료 보고 | [arXiv:2606.04233](https://arxiv.org/abs/2606.04233) |
| 만점 10점 단계 누적, `task progress = 점수/만점 × 100`, 10 trajectories | Wall-OSS-0.5, [arXiv:2605.30877](https://arxiv.org/abs/2605.30877) |

원문에서 확인한 참고 수치:

- ForceVLA는 "완전히 지웠다"의 **정량 기준을 제시하지 않는다**
  (*"Success was defined by task-specific criteria such as … effective wiping motion"*).
  우리가 픽셀 비율로 명시하는 것이 그 지점의 개선이다.
- SO-101 회복률: π₀.₅ 30.77% / Wall-X 20.51% / ACT 6.45% / **SmolVLA 3.23%(최하위)**.
  성공률도 SmolVLA 32.5%로 ACT 33.75%보다 낮다. 논문은 시행 수(태스크당 20)가 적어
  차이가 유의하지 않을 수 있다고 단서를 단다. **우리 주력 모델이 실행 실패에서 스스로
  못 빠져나온다는 뜻이므로, 회복률을 반드시 재야 하는 이유가 된다.**
- [arXiv:2606.04233]: LIBERO·SimplerEnv에서 주장된 SOTA 개선의 19.8% / 19.7%만이
  통계적으로 유의했다.

---

---

# 📌 요약

## 측정 방법론

**목표**: SmolVLA vs π₀를 비동기 추론 환경에서 비교한다.

**핵심 원칙 3가지**
1. **연속 지표로 검정한다.** 이진 성공률은 시행이 적어 신뢰구간이 겹친다. 주지표는
   `selective_erase = erased_target − erased_distractor`
2. **차등 점수는 물리 측정값에만 준다.** 단계 통과는 0/1, 지움 비율만 연속값
3. **종료와 성공을 분리한다.** 로봇이 놓았다고 성공이 아니다. 지움 비율로만 판정

**측정 항목**

| 층 | 지표 |
|---|---|
| 연속 (주지표) | `erased_target`, `erased_distractor`, `selective_erase` |
| 단계 점수 | 접근 1 / 접촉 2 / 개시 2 / 완료 3 / 선택성 2 = 10점 → 진행률 % |
| 이진 | `success` = 완료(≥0.90) + 선택성(≤0.10) |
| 부가 | `t50_s`, `t90_s`, `stall_events`, `recovery_rate`, `termination`, `failure_mode` |

**시행**: 조건당 30~50회, 초기 조건은 모든 조건에 동일하게, 순서는 랜덤화.

**컷오프는 처음 3~5개를 돌려본 뒤 정한다.** 정하고 나면 모든 조건에 동일하게 적용하고,
중간에 바꾸지 않는다.

---

## 측정 방법

### 자동 (기본)

```bash
# 0. 당일 첫 순서 — 카메라 확인
python scripts/tools/ink_metric.py <에피소드> --dump-roi /tmp/roi.png

# 1. 세션 시작
EVAL_MODEL=smolvla EVAL_CONDITION=async EVAL_TARGET=rectangle EVAL_TRIALS=30 \
EVAL_WATCH_DIR=<롤아웃 폴더> EVAL_ROLLOUT_CMD="<롤아웃 명령>" \
bash scripts/13__eval_session.sh
```

창에서 **space**(시작) → 롤아웃 → 자동 채점 → 실패 유형 확인 → **Enter**(다음).
반복. 결과는 `evaluation/<MMDD>_<model>_<condition>/`.

사람은 **비상정지 감시 + 유효/무효 판정 + 실패 유형 확인**만 한다.

### 손 채점 (도구 실패 시)

```bash
cp evaluation/manual_sheet_template.csv evaluation/<폴더>/episodes.csv
```

시행마다 한 줄. **눈으로 잴 수 있는 것만**:

| 칸 | 값 |
|---|---|
| 단계 5개 | 각 0 / 1 |
| `score` | 통과 배점 합 (1·2·2·3·2) |
| `erased_target` | `0` / `0.5` / `0.9` / `1.0` 4단계 |
| `erased_distractor` | `0` / `0.5` |
| `success` | True / False |
| `duration_s` | 스톱워치 |
| `termination` | release / stall / cutoff |
| `failure_mode` | 실패면 1개 |

집계는 자동과 동일:

```bash
python scripts/tools/erase_eval.py --summarize evaluation/<폴더>/episodes.csv
```

### 결과 확인

`evaluation/<폴더>/summary.md` — 조건별 성공률(95% CI), 진행률, 지움 비율,
t@90, 회복률, 실패 유형 분포가 표로 정리되어 있다. 보고서에 그대로 붙이면 된다.
